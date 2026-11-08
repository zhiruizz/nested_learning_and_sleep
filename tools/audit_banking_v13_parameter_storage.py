"""CPU-only structural parameter/storage audit for selected V13 checkpoints.

Reads exactly event1_before.pt, event1_after.pt, event2_after.pt, and
postwake15.pt for seed 17 / B32_HIGH. It never loads the Llama backbone or
reads metrics, datasets, test payloads, or other checkpoints. Output contains
only source/checkpoint hashes, tensor shapes/counts/bytes, and active/trainable
structure counts; no tensor values are emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


STAGES = ("event1_before", "event1_after", "event2_after", "postwake15")
MEMORY_PARAM_RE = re.compile(r"^(\d+)\.((?:experts\.slots\.(\d+)\.)?)(a|b)$")
ACTIVE_RE = re.compile(r"^(\d+)\.experts\.slots\.(\d+)\.active$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_info(tree: Any, torch, prefix: str) -> dict[str, Any]:
    descriptors = []
    total_bytes = total_numel = 0
    dtype_bytes: dict[str, int] = {}

    def visit(value, key):
        nonlocal total_bytes, total_numel
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            nbytes = len(raw)
            total_bytes += nbytes
            total_numel += tensor.numel()
            dtype = str(tensor.dtype)
            dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + nbytes
            descriptors.append({"key": key, "shape": list(tensor.shape), "dtype": dtype,
                                "numel": tensor.numel(), "bytes": nbytes,
                                "sha256": hashlib.sha256(raw).hexdigest()})
        elif isinstance(value, dict):
            for child_key in sorted(value, key=str):
                visit(value[child_key], f"{key}.{child_key}" if key else str(child_key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{key}[{index}]")

    visit(tree, prefix)
    digest_payload = json.dumps(descriptors, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True, allow_nan=False).encode("ascii")
    content_descriptors = sorted(
        ({key: item[key] for key in ("shape", "dtype", "numel", "bytes", "sha256")}
         for item in descriptors),
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    content_payload = json.dumps(content_descriptors, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True, allow_nan=False).encode("ascii")
    return {"tensor_count": len(descriptors), "numel": total_numel,
            "tensor_bytes": total_bytes, "dtype_bytes": dict(sorted(dtype_bytes.items())),
            "tensor_descriptor_sha256": hashlib.sha256(digest_payload).hexdigest(),
            "tensor_content_sha256": hashlib.sha256(content_payload).hexdigest(),
            "tensor_descriptors": descriptors}


def _parameter_summary(context: dict, torch, rank: int, capacity: int,
                       levels: int, hidden_size: int, vocab_size: int) -> dict[str, Any]:
    memory = context.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("checkpoint context lacks memory state dictionary")
    trainable = context.get("trainable")
    if not isinstance(trainable, dict) or any(type(v) is not bool for v in trainable.values()):
        raise ValueError("checkpoint context lacks boolean trainable map")

    actual_levels = sorted({int(key.split(".", 1)[0]) for key in memory if "." in key})
    if actual_levels != list(range(levels)):
        raise ValueError("checkpoint memory levels differ from locked model configuration")
    params: dict[str, torch.Tensor] = {}
    slots: dict[int, set[int]] = {level: set() for level in range(levels)}
    active_slots: dict[int, set[int]] = {level: set() for level in range(levels)}
    base_counts = {level: 0 for level in range(levels)}
    slot_counts: dict[tuple[int, int], int] = {}
    slot_dtypes: dict[tuple[int, int], set[str]] = {}

    for name, tensor in memory.items():
        match = MEMORY_PARAM_RE.match(name)
        if match:
            level, _, slot, leaf = match.groups()
            level = int(level)
            if slot is None:
                base_counts[level] += tensor.numel()
                params[f"memory.{name}"] = tensor
            else:
                slot = int(slot)
                slots[level].add(slot)
                slot_counts[(level, slot)] = slot_counts.get((level, slot), 0) + tensor.numel()
                slot_dtypes.setdefault((level, slot), set()).add(str(tensor.dtype))
                params[f"memory.{name}"] = tensor
        active_match = ACTIVE_RE.match(name)
        if active_match and bool(tensor.detach().cpu().item()):
            active_slots[int(active_match.group(1))].add(int(active_match.group(2)))

    for key in ("readout_a", "readout_b"):
        tensor = context.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint context lacks {key}")
        params[key] = tensor
    if (tuple(params["readout_a"].shape) != (2, hidden_size) or
            tuple(params["readout_b"].shape) != (vocab_size, 2)):
        raise ValueError("readout LoRA tensors do not match the rank-2 source definition")

    expected_slot_params = 2 * rank * hidden_size
    per_level = []
    trainable_lookup = trainable
    total_stored_adapter_params = sum(t.numel() for t in params.values())
    expected_total = (levels * expected_slot_params +
                      levels * capacity * expected_slot_params +
                      params["readout_a"].numel() + params["readout_b"].numel())
    if total_stored_adapter_params != expected_total:
        raise ValueError("checkpoint stored parameter count differs from preallocated model formula")
    total_trainable_params = sum(params[name].numel() for name, enabled in trainable_lookup.items()
                                 if enabled and name in params)
    for level in range(levels):
        found_slots = sorted(slots[level])
        if found_slots != list(range(capacity)):
            raise ValueError(f"level {level} does not have the expected preallocated capacity")
        if base_counts[level] != expected_slot_params:
            raise ValueError(f"level {level} base A/B parameter count mismatches rank/dim")
        if any(slot_counts.get((level, slot)) != expected_slot_params for slot in range(capacity)):
            raise ValueError(f"level {level} slot A/B parameter count mismatches rank/dim")
        level_trainable = sum(params[name].numel() for name, enabled in trainable_lookup.items()
                              if enabled and name.startswith(f"memory.{level}.") and name in params)
        per_level.append({
            "level": level,
            "base_parameters_stored": base_counts[level],
            "preallocated_slots": len(found_slots),
            "parameters_per_slot": expected_slot_params,
            "slot_parameters_stored": sum(slot_counts[(level, slot)] for slot in found_slots),
            "active_slot_flags": len(active_slots[level]),
            "active_slot_parameter_equivalents": len(active_slots[level]) * expected_slot_params,
            "trainable_parameters_at_checkpoint": level_trainable,
            "slot_parameter_dtypes": sorted(set().union(*(slot_dtypes[(level, s)] for s in found_slots))),
        })
    head_trainable = sum(params[name].numel() for name in ("readout_a", "readout_b")
                         if trainable_lookup.get(name) is True)
    return {
        "rank": rank, "capacity_per_level": capacity, "levels": levels,
        "hidden_size": hidden_size,
        "parameters_per_slot_formula": "2*rank*hidden_size",
        "parameters_per_slot": expected_slot_params,
        "bytes_per_slot_if_fp32": expected_slot_params * 4,
        "stored_memory_base_parameters": sum(base_counts.values()),
        "stored_expert_slot_parameters_all_levels": sum(slot_counts.values()),
        "stored_head_lora_parameters": params["readout_a"].numel() + params["readout_b"].numel(),
        "stored_adapter_parameter_total": total_stored_adapter_params,
        "trainable_parameter_total_at_checkpoint": total_trainable_params,
        "trainable_readout_parameters_at_checkpoint": head_trainable,
        "stored_parameter_bytes_by_dtype": _dtype_bytes(params, torch),
        "levels_detail": per_level,
    }


def _dtype_bytes(parameters: dict[str, Any], torch) -> dict[str, int]:
    out: dict[str, int] = {}
    for tensor in parameters.values():
        dtype = str(tensor.dtype)
        out[dtype] = out.get(dtype, 0) + tensor.numel() * tensor.element_size()
    return dict(sorted(out.items()))


def run(args) -> dict[str, Any]:
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU-only audit")
    # Import only the checkpoint dataclasses needed for CPU unpickling. No
    # AutoModel/backbone or tokenizer loading is performed.
    sys.path.insert(0, str(Path(args.task_root).resolve() / "tools"))
    sys.path.insert(0, str(Path(args.public_source_root).resolve() / "src"))
    import clinc_lifecycle_v8  # noqa: F401
    import banking_cil_v10_lifecycle  # noqa: F401
    from run_banking_optimizer_boundary_v12 import _same_nested

    model_cfg_path = Path(args.model_config)
    v10_cfg_path = Path(args.v10_config)
    v13_cfg_path = Path(args.v13_config)
    model_cfg = json.loads(model_cfg_path.read_text(encoding="utf-8"))
    v10_cfg = json.loads(v10_cfg_path.read_text(encoding="utf-8"))
    v13_cfg = json.loads(v13_cfg_path.read_text(encoding="utf-8"))
    hidden = int(model_cfg["hidden_size"])
    rank, capacity, levels = (int(v10_cfg[k]) for k in ("rank", "capacity", "levels"))
    if (rank, capacity, levels) != (64, 8, 5):
        raise ValueError("the inspected V13 run is expected to inherit rank64/capacity8/levels5")
    if v13_cfg.get("status") != "frozen":
        raise ValueError("V13 run config is not frozen")
    if any(v13_cfg.get(k) is not None for k in ("rank", "capacity", "levels")):
        raise ValueError("V13 architecture fields should inherit from the pinned V10 config")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized while importing checkpoint dataclasses")

    src_root = Path(args.public_source_root).resolve()
    task_root = Path(args.task_root).resolve()
    source_files = {
        "llama.py": src_root / "src/nested_sleep/lm.py",
        "experts.py": src_root / "src/nested_sleep/experts.py",
        "v10_lifecycle.py": task_root / "tools/banking_cil_v10_lifecycle.py",
        "v10_routing.py": task_root / "tools/banking_cil_v10_routing.py",
        "v13_batching.py": task_root / "tools/banking_cil_v13_batching.py",
        "v13_runtime.py": task_root / "tools/banking_sleep_v13_runtime.py",
        "v13_driver.py": task_root / "tools/run_banking_sleep_lifecycle_v13.py",
        "parameter_audit.py": Path(__file__).resolve(),
    }
    hashes = {name: sha256_file(path) for name, path in source_files.items()}
    config_hashes = {"model_config": sha256_file(model_cfg_path),
                     "v10_config": sha256_file(v10_cfg_path),
                     "v13_config": sha256_file(v13_cfg_path)}
    locked_source_hashes = v13_cfg.get("source_hashes")
    if not isinstance(locked_source_hashes, dict) or not locked_source_hashes:
        raise ValueError("frozen V13 config lacks its task-source hash map")
    actual_locked_source_hashes = {}
    for relative, expected_sha in sorted(locked_source_hashes.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("frozen source lock contains a noncontained path")
        source_path = (task_root / relative_path).resolve()
        if task_root.resolve() not in source_path.parents or not source_path.is_file():
            raise FileNotFoundError(f"locked task source is missing: {relative}")
        actual_sha = sha256_file(source_path)
        if actual_sha != expected_sha:
            raise ValueError(f"locked task source changed: {relative}")
        actual_locked_source_hashes[relative] = actual_sha

    trajectory = Path(args.trajectory_root) / "checkpoints"
    root_out = Path(args.output)
    if root_out.exists():
        raise FileExistsError("audit output path already exists; refusing overwrite")
    cp_records = []
    cuda_checks = []
    for stage in STAGES:
        cp_path = trajectory / f"{stage}.pt"
        if not cp_path.is_file():
            raise FileNotFoundError(f"required selected checkpoint missing: {stage}.pt")
        before_cuda = torch.cuda.is_initialized()
        if before_cuda:
            raise RuntimeError("CUDA initialized before checkpoint read")
        cp_file_sha = sha256_file(cp_path)
        payload = torch.load(cp_path, map_location="cpu", weights_only=False)
        after_cuda = torch.cuda.is_initialized()
        if after_cuda:
            raise RuntimeError("CUDA initialized during CPU checkpoint read")
        cuda_checks.append({"stage": stage, "before_load": before_cuda,
                            "after_load": after_cuda})
        if not isinstance(payload, dict) or (payload.get("seed"), payload.get("arm")) != (17, "B32_HIGH"):
            raise ValueError("selected checkpoint identity mismatch")
        if payload.get("stage") != stage:
            raise ValueError("checkpoint stage does not match selected file")
        if payload.get("config_sha256") != config_hashes["v13_config"]:
            raise ValueError("checkpoint V13 configuration hash differs from the frozen config")
        checkpoint_sources = payload.get("source_hashes")
        if not isinstance(checkpoint_sources, dict) or any(
                checkpoint_sources.get(key) != value
                for key, value in actual_locked_source_hashes.items()):
            raise ValueError("checkpoint source hashes differ from the current frozen source lock")
        context = payload.get("context")
        if not isinstance(context, dict):
            raise ValueError("checkpoint lacks adapter context state")
        context_tensors = tensor_info(context, torch, "context")
        wake = payload.get("wake")
        wake_context = getattr(wake, "context_state", None)
        wake_context_tensors = tensor_info(wake_context, torch, "wake.context_state")
        full_context_equal = _same_nested(context, wake_context, torch)
        if not full_context_equal:
            raise ValueError("context and wake snapshot differ by named state content")
        parameter_summary = _parameter_summary(context, torch, rank, capacity, levels, hidden,
                                               int(model_cfg["vocab_size"]))
        cp_records.append({
            "stage": stage, "checkpoint_file": cp_path.name,
            "checkpoint_size_bytes": cp_path.stat().st_size,
            "checkpoint_sha256": cp_file_sha,
            "config_source_binding_matches": True,
            "locked_task_source_bindings_match": True,
            "context_tensor_storage": context_tensors,
            "wake_snapshot_context_tensor_storage": wake_context_tensors,
            "context_and_wake_snapshot_tensor_content_equal":
                context_tensors["tensor_content_sha256"] == wake_context_tensors["tensor_content_sha256"],
            "context_and_wake_snapshot_full_named_state_equal": full_context_equal,
            "adapter_parameter_structure": parameter_summary,
            "routing_slot_counts_by_level": [
                _routing_slot_count(policy) for policy in context.get("routing", [])],
        })
        del payload, context, wake, wake_context

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized by the end of the audit")
    payload_out = {
        "schema": "banking-v13-selected-checkpoint-parameter-storage-audit-v1",
        "status": "complete_cpu_only_structural_audit",
        "source_files": {name: {"sha256": value} for name, value in hashes.items()},
        "frozen_v13_task_source_lock": actual_locked_source_hashes,
        "configuration_files": config_hashes,
        "model_architecture": {"model_type": model_cfg.get("model_type"),
                               "hidden_size": hidden,
                               "vocab_size": int(model_cfg["vocab_size"]),
                               "backbone_loaded": False,
                               "backbone_load_dtype_from_v10_driver": "torch.bfloat16",
                               "backbone_dtype_source_location": "tools/run_banking_cil_v10.py:459-464"},
        "adapter_architecture": {"rank": rank, "capacity": capacity, "levels": levels,
                                 "rank_capacity_source": "V10 config inherited by frozen V13 config",
                                 "V13_config_architecture_fields": {
                                     key: v13_cfg.get(key) for key in ("rank", "capacity", "levels")}},
        "formula_source_locations": {
            "LlamaCMSLM level/rank/capacity construction": "src/nested_sleep/lm.py:130-146",
            "low-rank base plus expert-bank module shapes": "src/nested_sleep/experts.py:11-29,164-171",
            "readout LoRA rank and shapes": "src/nested_sleep/lm.py:12-18",
            "new receiver activation and allowed trainable A/B scope": "tools/banking_cil_v10_lifecycle.py:140-148",
            "receiver chooses next inactive slot without overwrite": "tools/banking_cil_v10_routing.py:109-132",
            "V10 rank/capacity/levels values inherited by V13": "configs/banking_v10_frozen_k10.json",
            "V13 architecture fields inherit rather than override": "configs/banking_v13_batched_lifecycle_frozen_a.json",
        },
        "sleep_receiver_parameter_formula": "2*rank*hidden_size",
        "sleep_receiver_trainable_parameters_per_event": 2 * rank * hidden,
        "sleep_receiver_parameter_bytes_if_fp32": 2 * rank * hidden * 4,
        "preallocated_adapter_parameter_formula":
            "levels*2*rank*hidden_size + levels*capacity*2*rank*hidden_size + readout_A + readout_B",
        "preallocated_adapter_parameter_total_expected":
            levels * 2 * rank * hidden + levels * capacity * 2 * rank * hidden +
            2 * hidden + int(model_cfg["vocab_size"]) * 2,
        "checkpoint_records": cp_records,
        "cuda_initialized_before_after_each_checkpoint": cuda_checks,
        "cuda_initialized_at_completion": False,
        "scope_notes": [
            "All memory bases and all capacity slots remain allocated at every checkpoint; activation does not add parameter storage.",
            "Active-slot flags and additive routing-slot counts describe structural availability, not per-token route hits or executed FLOPs.",
            "Trainable counts are the saved context flags at each checkpoint; the per-event new-receiver trainable scope is separately derived from the V10 transaction's allowed receiver A/B names.",
            "Checkpoint file size includes lifecycle/RNG/optimizer payloads and is not equal to model adapter storage.",
            "No evaluation metrics, raw rows, prompts, labels, token IDs, backbone weights, or test/P2 artifacts are read or emitted.",
        ],
    }
    root_out.parent.mkdir(parents=True, exist_ok=True)
    _write_new_json(root_out, payload_out)
    return payload_out


def _routing_slot_count(policy: Any) -> int:
    if not isinstance(policy, dict):
        return 0
    rows = policy.get("slots", [])
    return len(rows) if isinstance(rows, list) else 0


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite audit JSON")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os_fsync = __import__("os").fsync
        os_fsync(stream.fileno())
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    task_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--task-root", type=Path, default=task_root)
    parser.add_argument("--public-source-root", type=Path, default=task_root)
    parser.add_argument("--asset-root", type=Path, default=task_root)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--v10-config", type=Path, default=task_root / "configs/banking_v10_frozen_k10.json")
    parser.add_argument("--v13-config", type=Path, default=task_root / "configs/banking_v13_batched_lifecycle_frozen_a.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model_config is None:
        args.model_config = args.asset_root / ".cache/models/llama32-3b/config.json"
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
