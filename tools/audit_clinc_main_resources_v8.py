#!/usr/bin/env python3
"""Read-only resource audit for a completed sealed CLINC V8 main grid.

This reports serialized mutable-state payload/storage, locked model-cache file
sizes, checkpoint bytes, and the existing estimator's partial GEMM+attention
intervals. It does not load the backbone, initialize CUDA, read dataset rows,
or claim measured/complete FLOPs or peak GPU memory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

EXPECTED_MODEL_CONFIG_SHA256 = "35f063e9f381e8678eea34bc56c3ecef3c6d10149f6b2cbdf27b8ee1bad3073d"
SEEDS = (17, 29, 43)
ARMS = ("HOPE", "HOPE_SLEEP_FIXEDMIX")
EXCLUDED_FLOPS = [
    "RMSNorm, activation, RoPE, softmax, and routing arithmetic",
    "cross entropy/KL/reward losses, sampling, and token-selection work",
    "optimizer and gradient-clipping arithmetic, adds/copies, and I/O",
    "any capture-time original/expert GEMMs not represented by the estimator's listed capture terms",
    "frozen-backbone backward and allocator fragmentation/reserved-memory overhead",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def checked_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"artifact path escapes supplied root: {relative}")
    return path


def _tensor_bytes(tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def context_inventory(context: dict, torch, *, levels: int, capacity: int) -> dict:
    """Count checkpoint-owned parameter and buffer tensors without building a model."""
    required = {"memory", "readout_a", "readout_b", "readout_fp32_accumulation",
                "routing", "trainable"}
    if not isinstance(context, dict) or set(context) != required:
        raise ValueError("checkpoint context schema mismatch")
    memory = context["memory"]
    trainable = context["trainable"]
    if not isinstance(memory, dict) or not isinstance(trainable, dict):
        raise ValueError("checkpoint context memory/trainable maps malformed")
    params = {}
    buffers = {}
    for key, tensor in memory.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("non-tensor entry in mutable memory state")
        leaf = key.rsplit(".", 1)[-1]
        full = "memory." + key
        if leaf in ("a", "b"):
            params[full] = tensor
        else:
            buffers[full] = tensor
    for name in ("readout_a", "readout_b"):
        tensor = context[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"context {name} is not a tensor")
        params[name] = tensor
    if set(params) != set(trainable) or any(type(v) is not bool for v in trainable.values()):
        raise ValueError("serialized trainable flags do not match all mutable parameters")
    if not isinstance(context["routing"], list) or len(context["routing"]) != levels:
        raise ValueError("serialized routing policy count differs from configured levels")

    all_tensors = {**params, **buffers}
    unique_storages = {}
    for name, tensor in all_tensors.items():
        storage = tensor.untyped_storage()
        # One CPU checkpoint tensor storage may back multiple views. Count each
        # allocated storage once while retaining logical parameter numel totals.
        key = (int(storage.data_ptr()), int(storage.nbytes()), str(tensor.device))
        unique_storages.setdefault(key, int(storage.nbytes()))

    active = [[False] * capacity for _ in range(levels)]
    active_flags = 0
    for name, tensor in buffers.items():
        match = re.fullmatch(r"memory\.(\d+)\.experts\.slots\.(\d+)\.active", name)
        if match:
            level, slot = map(int, match.groups())
            if level >= levels or slot >= capacity or tensor.numel() != 1:
                raise ValueError("active-slot state lies outside configured expert banks")
            active[level][slot] = bool(tensor.item())
            active_flags += 1
    if active_flags != levels * capacity:
        raise ValueError("checkpoint lacks the complete active-slot buffer grid")

    slot_pairs = {}
    for name, tensor in params.items():
        match = re.fullmatch(r"memory\.(\d+)\.experts\.slots\.(\d+)\.(a|b)", name)
        if match:
            level, slot, side = match.groups()
            slot_pairs.setdefault((int(level), int(slot)), {})[side] = tensor
    if len(slot_pairs) != levels * capacity or any(set(pair) != {"a", "b"} for pair in slot_pairs.values()):
        raise ValueError("checkpoint expert parameter grid differs from configured levels/capacity")
    pair_shapes = {(tuple(pair["a"].shape), tuple(pair["b"].shape),
                    str(pair["a"].dtype), str(pair["b"].dtype)) for pair in slot_pairs.values()}
    if len(pair_shapes) != 1:
        raise ValueError("expert slot parameter shapes/dtypes are not uniform")
    fresh_pair_numel = sum(int(t.numel()) for t in next(iter(slot_pairs.values())).values())
    fresh_pair_bytes = sum(_tensor_bytes(t) for t in next(iter(slot_pairs.values())).values())

    parameter_numel = sum(int(t.numel()) for t in params.values())
    parameter_bytes = sum(_tensor_bytes(t) for t in params.values())
    buffer_numel = sum(int(t.numel()) for t in buffers.values())
    buffer_bytes = sum(_tensor_bytes(t) for t in buffers.values())
    trainable_names = sorted(name for name, flag in trainable.items() if flag)
    trainable_numel = sum(int(params[name].numel()) for name in trainable_names)
    trainable_bytes = sum(_tensor_bytes(params[name]) for name in trainable_names)
    return {
        "serialized_parameter_tensor_count": len(params),
        "serialized_mutable_parameter_numel": parameter_numel,
        "serialized_mutable_parameter_tensor_bytes": parameter_bytes,
        "serialized_buffer_tensor_count": len(buffers),
        "serialized_mutable_buffer_numel": buffer_numel,
        "serialized_mutable_buffer_tensor_bytes": buffer_bytes,
        "serialized_mutable_unique_storage_bytes": sum(unique_storages.values()),
        "serialized_trainable_parameter_numel": trainable_numel,
        "serialized_trainable_parameter_bytes": trainable_bytes,
        "serialized_trainable_parameter_names": trainable_names,
        "active_slots_by_level": [sum(row) for row in active],
        "active_slot_ids_by_level": [[i for i, enabled in enumerate(row) if enabled] for row in active],
        "fresh_receiver_slot_A_B_numel": fresh_pair_numel,
        "fresh_receiver_slot_A_B_bytes": fresh_pair_bytes,
        "context_tensor_payload_bytes": parameter_bytes + buffer_bytes,
        "storage_interpretation": "checkpoint tensor payload/storage estimate; not measured runtime allocator VRAM",
    }


def validate_work_histograms(work: dict, *, label: str) -> None:
    required = ("full_backbone_forward_calls", "tail_calls", "tail_grad_enabled_calls",
                "tail_no_grad_calls", "forward_input_shape_histogram", "tail_shape_histogram",
                "tail_grad_shape_histogram", "tail_no_grad_shape_histogram")
    if any(key not in work for key in required):
        raise ValueError(f"{label}: work ledger lacks required call/histogram fields")
    def count(hist):
        if not isinstance(hist, dict):
            raise ValueError(f"{label}: shape histogram is not an object")
        total = 0
        for shape, n in hist.items():
            if not re.fullmatch(r"[1-9]\d*x[1-9]\d*", shape) or type(n) is not int or n < 0:
                raise ValueError(f"{label}: invalid work histogram entry")
            total += n
        return total
    fwd = count(work["forward_input_shape_histogram"])
    tail = count(work["tail_shape_histogram"])
    grad = count(work["tail_grad_shape_histogram"])
    no_grad = count(work["tail_no_grad_shape_histogram"])
    if fwd != work["full_backbone_forward_calls"] or tail != work["tail_calls"] or \
       grad != work["tail_grad_enabled_calls"] or no_grad != work["tail_no_grad_calls"] or \
       grad + no_grad != tail:
        raise ValueError(f"{label}: work shape histogram sums disagree with recorded calls")


def validate_checkpoint_receipt_work(checkpoint_work: dict, receipt_work: dict, *, label: str) -> None:
    if not isinstance(checkpoint_work, dict) or not isinstance(receipt_work, dict) or \
       set(checkpoint_work) != set(receipt_work):
        raise ValueError(f"{label}: checkpoint and receipt work schemas differ")
    for key in checkpoint_work:
        if key == "active_wall_seconds":
            left, right = checkpoint_work[key], receipt_work[key]
            if (type(left) not in (int, float) or type(right) not in (int, float) or
                not math.isfinite(float(left)) or not math.isfinite(float(right)) or
                left < 0 or right < 0 or float(left) > float(right) + 1e-6):
                raise ValueError(f"{label}: checkpoint/receipt wall time is invalid or reversed")
        elif checkpoint_work[key] != receipt_work[key]:
            raise ValueError(f"{label}: checkpoint and receipt discrete work counters differ at {key}")


def _load_cpu_checkpoint(path: Path, torch):
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before resource audit checkpoint read")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "clinc-cil-v8-full-safe-checkpoint":
        raise ValueError(f"unsupported safe checkpoint schema: {path}")
    return payload


def _model_cache_inventory(model_root: Path, assets: dict) -> dict:
    rows = []
    for relative, expected in sorted(assets.items()):
        path = checked_child(model_root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"locked model-cache asset hash mismatch: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    all_files = [path for path in model_root.rglob("*") if path.is_file()]
    return {"locked_asset_count": len(rows), "locked_asset_bytes": sum(x["bytes"] for x in rows),
        "locked_assets": rows, "model_directory_file_count": len(all_files),
        "model_directory_all_file_bytes": sum(path.stat().st_size for path in all_files),
        "interpretation": "filesystem sizes only; no model weight deserialization or device allocation"}


def audit(*, seal_path: Path, master_path: Path, training_pack: Path,
          foundation_root: Path, run_root: Path, model_root: Path,
          project_root: Path, out_path: Path) -> dict:
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite resource audit: {out_path}")
    started = time.perf_counter()
    seal_path, master_path, training_pack = map(Path, (seal_path, master_path, training_pack))
    foundation_root, run_root, model_root, project_root = map(Path,
        (foundation_root, run_root, model_root, project_root))
    seal_sha, master_sha = sha256_file(seal_path), sha256_file(master_path)
    manifest_sha = sha256_file(training_pack)
    seal, master = load_json(seal_path), load_json(master_path)
    if seal.get("schema") != "clinc-cil-v8-main-six-seal" or seal.get("complete") is not True:
        raise ValueError("only a complete sealed six-run main grid is accepted")
    if seal.get("test_data_included") is not False or master.get("status") != "frozen":
        raise ValueError("seal/master is not test-isolated and frozen")
    if seal.get("master_config_sha256") != master_sha or seal.get("training_manifest_sha256") != manifest_sha:
        raise ValueError("seal/master/training-manifest byte binding mismatch")
    if master.get("model_assets_sha256", {}).get("config.json") != EXPECTED_MODEL_CONFIG_SHA256:
        raise ValueError("frozen model config SHA differs from the published model pin")
    if seal.get("main_grid") != [{"seed": seed, "arm": arm} for seed in SEEDS for arm in ARMS]:
        raise ValueError("sealed grid is not the expected 3-seed by 2-arm design")
    if len(seal.get("verified_foundations", [])) != 3 or len(seal.get("verified_run_files", [])) != 6:
        raise ValueError("seal lacks three foundation or six run-file bindings")

    tools_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
    from tools import run_clinc_cil_v8_full as full
    current_sources = full._source_hashes(project_root, tools_dir)
    if current_sources != seal.get("source_hashes") or full.canonical_sha(current_sources) != seal.get("source_lock_sha256"):
        raise ValueError("current public training source bytes differ from the main seal")
    model_cfg_path = checked_child(model_root, "config.json")
    if sha256_file(model_cfg_path) != EXPECTED_MODEL_CONFIG_SHA256:
        raise ValueError("model config.json does not match the pinned Llama dimensions")
    model_cfg_raw = model_cfg_path.read_bytes()
    model_cfg = json.loads(model_cfg_raw)
    from tools import estimate_clinc_v8_work as estimator
    dims = estimator.parse_config(model_cfg)
    model_cache = _model_cache_inventory(model_root, master["model_assets_sha256"])
    config = master

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("resource audit requires a CPU-only, uninitialized-CUDA process")

    run_file_map = {(int(row["seed"]), row["arm"]): row for row in seal["verified_run_files"]}
    run_rows = {(int(row["seed"]), row["arm"]): row for row in seal["runs"]}
    if set(run_file_map) != {(seed, arm) for seed in SEEDS for arm in ARMS} or set(run_rows) != set(run_file_map):
        raise ValueError("sealed run grid is incomplete or duplicated")
    foundation_map = {int(row["seed"]): row for row in seal["verified_foundations"]}
    if set(foundation_map) != set(SEEDS):
        raise ValueError("sealed common-foundation grid is incomplete")

    checkpoint_inventory = []
    foundation_work = []
    for seed in SEEDS:
        receipt_path = checked_child(foundation_root, f"seed{seed}/foundation-receipt.json")
        receipt = load_json(receipt_path)
        verified = foundation_map[seed]
        checkpoint_path = checked_child(foundation_root / f"seed{seed}", receipt["final_checkpoint"])
        if (sha256_file(receipt_path) != verified["receipt_sha256"] or
            sha256_file(checkpoint_path) != verified["checkpoint_sha256"] or
            receipt.get("final_checkpoint_sha256") != verified["checkpoint_sha256"]):
            raise ValueError(f"seed {seed}: foundation receipt/checkpoint differs from seal")
        if receipt.get("test_data_included") is not False:
            raise ValueError(f"seed {seed}: foundation does not certify test isolation")
        payload = _load_cpu_checkpoint(checkpoint_path, torch)
        if payload.get("mode") != "foundation" or payload.get("seed") != seed or \
           payload["cursor"].get("tasks_completed") != 1 or payload["cursor"].get("task_id") != 0:
            raise ValueError(f"seed {seed}: foundation checkpoint cursor mismatch")
        inventory = context_inventory(payload["context"], torch,
            levels=int(config["levels"]), capacity=int(config["capacity"]))
        work = receipt.get("work")
        validate_work_histograms(work, label=f"seed{seed}/foundation")
        foundation_work.append({"seed": seed, "work": work,
            "flops": estimator.estimate_arm(work, dims, config, arm="HOPE")})
        checkpoint_inventory.append({"seed": seed, "arm": "COMMON_FOUNDATION",
            "mode": "foundation", "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": verified["checkpoint_sha256"],
            "checkpoint_file_bytes": checkpoint_path.stat().st_size,
            "parameter_inventory": inventory, "work": work})
        del payload

    arm_results = []
    for key in sorted(run_rows):
        seed, arm = key
        row, file_row = run_rows[key], run_file_map[key]
        run_dir = run_root / f"seed{seed}" / arm
        receipt_path = checked_child(run_root, file_row["run_receipt_path"])
        checkpoint_path = checked_child(run_root, file_row["checkpoint_path"])
        receipt = load_json(receipt_path)
        sealed_receipt = dict(row)
        sealed_receipt.pop("run_receipt_path", None)
        sealed_receipt.pop("run_receipt_sha256", None)
        if receipt != sealed_receipt:
            raise ValueError(f"seed {seed}/{arm}: on-disk run receipt differs from seal row")
        if sha256_file(receipt_path) != file_row["run_receipt_sha256"] or \
           sha256_file(checkpoint_path) != file_row["checkpoint_sha256"] or \
           receipt.get("final_checkpoint_sha256") != file_row["checkpoint_sha256"]:
            raise ValueError(f"seed {seed}/{arm}: run receipt/checkpoint differs from sealed hashes")
        if receipt.get("test_data_included") is not False:
            raise ValueError(f"seed {seed}/{arm}: run does not certify test isolation")
        payload = _load_cpu_checkpoint(checkpoint_path, torch)
        cursor = payload.get("cursor", {})
        if payload.get("mode") != "continuation" or payload.get("seed") != seed or payload.get("arm") != arm or \
           (cursor.get("tasks_completed"), cursor.get("task_id"), cursor.get("next_step"),
            cursor.get("safe_point")) != (10, 9, 189, "post_all_due_events"):
            raise ValueError(f"seed {seed}/{arm}: final checkpoint is not the sealed endpoint")
        # Wall timers are independently sampled at checkpoint and receipt.
        validate_checkpoint_receipt_work(payload.get("work_cumulative", {}),
            receipt.get("work", {}), label=f"seed{seed}/{arm}")
        validate_work_histograms(receipt["work"], label=f"seed{seed}/{arm}")
        inventory = context_inventory(payload["context"], torch,
            levels=int(config["levels"]), capacity=int(config["capacity"]))
        foundation_receipt = load_json(checked_child(foundation_root,
            f"seed{seed}/foundation-receipt.json"))
        if receipt.get("initial_mutable_sha256") != foundation_receipt.get("foundation_context_sha256"):
            raise ValueError(f"seed {seed}/{arm}: continuation initial state differs from common foundation")
        work = receipt["work"]
        flops = estimator.estimate_arm(work, dims, config, arm=arm)
        events = receipt.get("sleep_event_index")
        adam = receipt.get("physical_sleep_adamw_steps")
        if type(events) is not int or type(adam) is not int or events < 0 or adam < 0:
            raise ValueError(f"seed {seed}/{arm}: typed Sleep counters missing")
        fresh_bytes = inventory["fresh_receiver_slot_A_B_bytes"]
        context_bytes = inventory["context_tensor_payload_bytes"]
        sleep_peak = None
        if arm == "HOPE_SLEEP_FIXEDMIX" and events:
            sleep_peak = {
                "live_teacher_student_context_count": 3,
                "additional_context_copies_beyond_live": 2,
                "live_plus_teacher_plus_student_tensor_payload_bytes": 3 * context_bytes,
                "extra_teacher_and_student_payload_bytes": 2 * context_bytes,
                "fresh_receiver_A_B_trainable_numel": inventory["fresh_receiver_slot_A_B_numel"],
                "fresh_receiver_gradient_bytes": fresh_bytes,
                "two_AdamW_moments_bytes_after_first_step": 2 * fresh_bytes,
                "additional_bytes_beyond_live_context_at_adam_peak": 2 * context_bytes + 3 * fresh_bytes,
                "AdamW_step_scalar_storage": "not included; per-parameter scalar is negligible and framework-version dependent",
                "basis": "source-level estimate from SleepTransaction deep-copies and exact receiver A/B-only optimizer domain; not measured peak memory",
            }
        tail_forward_delta = work["tail_calls"] - work["full_backbone_forward_calls"]
        if tail_forward_delta < 0:
            raise ValueError(f"seed {seed}/{arm}: tail-call count is below backbone forward count")
        validate_endpoint_work = {"full_backbone_forward_calls": work["full_backbone_forward_calls"],
            "forward_histogram_sum": sum(work["forward_input_shape_histogram"].values()),
            "tail_calls": work["tail_calls"], "tail_histogram_sum": sum(work["tail_shape_histogram"].values()),
            "tail_grad_enabled_calls": work["tail_grad_enabled_calls"],
            "tail_grad_histogram_sum": sum(work["tail_grad_shape_histogram"].values()),
            "tail_no_grad_calls": work["tail_no_grad_calls"],
            "tail_no_grad_histogram_sum": sum(work["tail_no_grad_shape_histogram"].values()),
            "tail_calls_minus_backbone_forwards": tail_forward_delta,
            "physical_sleep_adamw_steps": adam,
            "tail_forward_delta_equals_sleep_adamw_steps": tail_forward_delta == adam}
        arm_results.append({"seed": seed, "arm": arm,
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": file_row["checkpoint_sha256"],
            "checkpoint_file_bytes": checkpoint_path.stat().st_size,
            "run_receipt_path": str(receipt_path), "run_receipt_sha256": file_row["run_receipt_sha256"],
            "parameter_inventory": inventory, "sleep_event_index": events,
            "physical_sleep_adamw_steps": adam, "sleep_transaction_working_storage_estimate": sleep_peak,
            "work_histogram_call_crosscheck": validate_endpoint_work,
            "partial_covered_flops": flops, "work_record_scope": "continuation-only; common foundation counted once separately"})
        checkpoint_inventory.append({"seed": seed, "arm": arm, "mode": "continuation",
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": file_row["checkpoint_sha256"],
            "checkpoint_file_bytes": checkpoint_path.stat().st_size,
            "parameter_inventory": inventory, "work": work})
        del payload

    lower = sum(row["flops"]["covered_total_interval"][0] for row in foundation_work)
    upper = sum(row["flops"]["covered_total_interval"][1] for row in foundation_work)
    lower += sum(row["partial_covered_flops"]["covered_total_interval"][0] for row in arm_results)
    upper += sum(row["partial_covered_flops"]["covered_total_interval"][1] for row in arm_results)
    record = {
        "schema": "clinc-v8-main-resource-audit-v1",
        "classification": "read-only artifact/resource accounting; analytical partial FLOPs; not a speed or efficiency result",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "test_data_read": False, "model_weights_deserialized": False,
        "cuda_initialized": False,
        "bindings": {"seal_path": str(seal_path.resolve()), "seal_sha256": seal_sha,
            "master_config_path": str(master_path.resolve()), "master_config_sha256": master_sha,
            "training_manifest_path": str(training_pack.resolve()), "training_manifest_sha256": manifest_sha,
            "source_lock_sha256": seal["source_lock_sha256"], "current_source_hashes": current_sources,
            "audit_tool_sha256": sha256_file(Path(__file__).resolve()),
            "existing_flop_estimator_sha256": sha256_file(Path(estimator.__file__).resolve()),
            "model_config_sha256": sha256_file(model_cfg_path)},
        "model_cache_filesystem_storage": model_cache,
        "checkpoint_filesystem_storage": {"checkpoint_count": len(checkpoint_inventory),
            "total_checkpoint_file_bytes": sum(row["checkpoint_file_bytes"] for row in checkpoint_inventory),
            "files": [{key: row[key] for key in ("seed", "arm", "mode", "checkpoint_path", "checkpoint_sha256", "checkpoint_file_bytes")}
                for row in checkpoint_inventory]},
        "serialized_mutable_contexts": checkpoint_inventory,
        "common_foundation_work_counted_once": foundation_work,
        "partial_analytic_flops": {"formula_version": estimator.FORMULA_VERSION,
            "per_continuation_arm": arm_results,
            "total_with_three_common_foundations_once": [lower, upper],
            "total_teraflops_interval": [lower / 1e12, upper / 1e12],
            "classification": "covered analytical GEMM+attention interval only; not measured or complete model FLOPs",
            "excluded_operators": EXCLUDED_FLOPS,
            "formula_notes": [
                "The existing estimator formula is reused without changing its definitions.",
                "Work shape histograms are checked against recorded forward/tail/backward call counters.",
                "Foundation work is reported separately and added once; sealed continuation work is not duplicated as a second foundation.",
                "Controlled fixed-mix uses the existing two-expert-branch upper bound; natural routing uses one.",
                "Capture-time expert/original MLP GEMMs not explicitly represented by the estimator's capture terms are omitted and listed among exclusions.",
                "No frozen-backbone backward is counted; upper/lower interval refers only to listed mutable/expert GEMM terms."]},
        "resource_interpretation": [
            "Mutable parameter/buffer storage comes from safe checkpoint context tensors and serialized trainable flags; these are payload/storage counts, not runtime allocator peaks.",
            "The Sleep peak is a source-derived live+teacher+student context and receiver A/B gradient/Adam-moment estimate; it excludes framework allocator fragmentation and RNG/temporary activations.",
            "Model-cache numbers are filesystem stat/hash totals for locked assets; no model weights were loaded.",
            "No efficiency or speedup claim follows from these counts."],
        "test_data_included": False,
    }
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA unexpectedly initialized during CPU resource audit")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8", newline="\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("seal", "master-config", "training-pack", "foundation-root", "run-root", "model-root", "project-root", "out"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit(seal_path=args.seal, master_path=args.master_config,
        training_pack=args.training_pack, foundation_root=args.foundation_root,
        run_root=args.run_root, model_root=args.model_root,
        project_root=args.project_root, out_path=args.out)
    print(json.dumps({"status": "complete", "seal_sha256": result["bindings"]["seal_sha256"],
        "checkpoint_count": result["checkpoint_filesystem_storage"]["checkpoint_count"],
        "checkpoint_bytes": result["checkpoint_filesystem_storage"]["total_checkpoint_file_bytes"],
        "covered_teraflops_interval": result["partial_analytic_flops"]["total_teraflops_interval"],
        "output": str(args.out.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
