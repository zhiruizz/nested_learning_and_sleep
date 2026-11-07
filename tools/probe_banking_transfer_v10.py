#!/usr/bin/env python3
"""Development-only Banking77 transfer/readout probe from task-0 foundation.

This candidate compares local additive receiver and legacy fixed-mix controls.
It is not a scheduled main boundary or a paper-faithful HOPE/Sleep reproduction.
The CLI has no test-data argument and refuses any foundation except seed 17's
qualified task-0 training checkpoint.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields, is_dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import sys
import time
from typing import Any, Mapping, Sequence

SCHEMA = "banking-cil-v10-transfer-probe-v1"
SEED = 17
TASK_ID = 0
CALLS = 30
METHODS = ("gkd", "gkd", "lti")
ARMS = ("SLEEP_ADDITIVE", "GOLD_ADDITIVE", "SLEEP_FIXEDMIX", "GOLD_FIXEDMIX")


def _bootstrap_task_paths() -> tuple[Path, Path]:
    """Make both ``tools`` and ``from tools import ...`` work in direct CLI use."""
    tools_dir = Path(__file__).resolve().parent
    task_root = tools_dir.parent
    for entry in (tools_dir, task_root):
        value = str(entry)
        if value not in sys.path:
            sys.path.insert(0, value)
    return task_root, tools_dir


_bootstrap_task_paths()


class ProbeValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeValidationError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.partial")
    try:
        with tmp.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        if path.read_bytes() != raw:
            raise IOError("atomic JSON readback mismatch")
        return hashlib.sha256(raw).hexdigest()
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_torch(path: Path, value: Any, torch) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.partial")
    try:
        with tmp.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        expected = _sha256_file(tmp)
        os.replace(tmp, path)
        if _sha256_file(path) != expected:
            raise IOError("atomic torch checkpoint readback mismatch")
        return expected
    finally:
        if tmp.exists():
            tmp.unlink()


def _tree_hash(value: Any, torch=None) -> str:
    """Stable exact hash for tensor/dataclass/optimizer/RNG snapshots."""
    import numpy as np
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch is not None and isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"TENSOR\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            # A flattened view also supports zero-dimensional optimizer state
            # such as Adam's scalar step counters.
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"NDARRAY\0")
            digest.update(str(item.dtype).encode())
            digest.update(repr(tuple(item.shape)).encode())
            digest.update(item.tobytes(order="C"))
        elif is_dataclass(item) and not isinstance(item, type):
            digest.update(type(item).__qualname__.encode())
            for field in fields(item):
                digest.update(field.name.encode())
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            digest.update(b"MAP\0")
            for key in sorted(item, key=lambda x: repr(x)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"LIST\0" if isinstance(item, list) else b"TUPLE\0")
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode())
            digest.update(json.dumps(item, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        else:
            raise TypeError(f"unsupported snapshot value for exact hash: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def select_old_prompt_schedule(rows: Sequence[Mapping], class_order: Sequence[int], *, calls: int = CALLS) -> list[dict]:
    """Fixed 30-call task-0 class RR over four stable-ID train rows/class."""
    from tools import banking_cil_common_v10 as common
    classes = common.task_class_ids(class_order, 0)
    grouped: dict[int, list[Mapping]] = {class_id: [] for class_id in classes}
    allowed = set(classes)
    for row in rows:
        class_id = int(row["intent_id"])
        if class_id in allowed:
            grouped[class_id].append(row)
    for class_id in classes:
        grouped[class_id] = sorted(grouped[class_id], key=lambda row: str(row["id"]))[:4]
        _require(len(grouped[class_id]) == 4, f"class {class_id} needs four task-0 training prompts")
    _require(calls == CALLS, "the development probe is fixed to 30 call opportunities")
    selected = []
    for serial in range(calls):
        class_position = serial % len(classes)
        row_position = (serial // len(classes)) % 4
        class_id = classes[class_position]
        row = grouped[class_id][row_position]
        selected.append({"call": serial, "event": serial // len(METHODS), "method": METHODS[serial % len(METHODS)],
                         "class_id": class_id, "example_id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest(),
                         "row": dict(row)})
    return selected


def safe_schedule_receipt(schedule: Sequence[Mapping]) -> dict:
    if len(schedule) != CALLS:
        raise ProbeValidationError("probe schedule must have exactly 30 opportunities")
    counts: dict[int, int] = {}
    for index, row in enumerate(schedule):
        if row.get("call") != index or row.get("method") != METHODS[index % len(METHODS)]:
            raise ProbeValidationError("method/call schedule mismatch")
        counts[int(row["class_id"])] = counts.get(int(row["class_id"]), 0) + 1
    _require(len(counts) == 11 and min(counts.values()) >= 2 and max(counts.values()) <= 3,
             "30 opportunities must cover all eleven old classes at least twice")
    return {"schema": "task0-old-30call-class-roundrobin-v1", "calls": CALLS,
            "methods": list(METHODS), "unique_classes": len(counts),
            "calls_by_class": {str(key): counts[key] for key in sorted(counts)},
            "selected_example_id_hashes_sha256": _canonical_sha([row["example_id_sha256"] for row in schedule])}


def validate_foundation_payload(receipt: Mapping, checkpoint: Mapping, *, seed: int,
                                config_sha256: str, source_lock_sha256: str,
                                training_manifest_sha256: str, checkpoint_sha256: str) -> None:
    """Strictly bind the probe to the driver's qualified foundation artifact."""
    if receipt.get("schema") != "banking-cil-v10-foundation-receipt" or receipt.get("status") != "qualified":
        raise ProbeValidationError("foundation receipt is not qualified")
    expected_receipt = {"seed": seed, "arm": "FOUNDATION_CMS_ONLY", "task_id": 0,
                        "config_sha256": config_sha256, "source_lock_sha256": source_lock_sha256,
                        "training_manifest_sha256": training_manifest_sha256,
                        "checkpoint_sha256": checkpoint_sha256, "test_data_included": False,
                        "model_immutable_versions_unchanged": True}
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise ProbeValidationError(f"foundation receipt binding mismatch: {key}")
    _require(receipt.get("gate", {}).get("passed") is True, "foundation gate did not pass")
    _require(float(receipt.get("metric", {}).get("accuracy", -1)) >= 0.5,
             "foundation global 77-class gate metric failed")
    if checkpoint.get("schema") != "banking-cil-v10-checkpoint":
        raise ProbeValidationError("foundation checkpoint schema mismatch")
    expected_cp = {"seed": seed, "arm": "FOUNDATION_CMS_ONLY", "task_id": 0,
                   "config_sha256": config_sha256, "source_lock_sha256": source_lock_sha256,
                   "training_manifest_sha256": training_manifest_sha256}
    for key, expected in expected_cp.items():
        if checkpoint.get(key) != expected:
            raise ProbeValidationError(f"foundation checkpoint binding mismatch: {key}")
    _require(type(checkpoint.get("step")) is int and checkpoint["step"] == receipt.get("steps"),
             "foundation checkpoint terminal step mismatch")
    _require(isinstance(checkpoint.get("context"), Mapping) and checkpoint.get("wake") is not None,
             "foundation checkpoint lacks mutable context/wake state")
    _require(isinstance(checkpoint.get("rng"), Mapping) and isinstance(checkpoint.get("extra"), Mapping),
             "foundation checkpoint lacks RNG or gate metadata")


def _metric_payload(scored: Mapping, context_name: str) -> dict:
    metrics = scored.get("candidate_metrics")
    if not isinstance(metrics, Mapping) or context_name not in metrics:
        raise ValueError("candidate runtime omitted named context metrics")
    metric = metrics[context_name]
    all_rows = metric.get("teacher_forced_sum", {}).get("all77")
    if not isinstance(all_rows, Mapping):
        raise ValueError("candidate runtime omitted global 77-way SUM metric")
    per_class = metric.get("teacher_forced_sum", {}).get("per_class")
    return {"correct": int(all_rows["correct"]), "count": int(all_rows["count"]),
            "accuracy": all_rows["accuracy"], "invalid_predictions": int(all_rows.get("invalid_predictions", 0)),
            "per_class": per_class,
            "length_mean_correct": int(metric.get("teacher_forced_length_mean", {}).get("all77", {}).get("correct", -1)),
            "length_mean_count": int(metric.get("teacher_forced_length_mean", {}).get("all77", {}).get("count", -1))}


def _same_exact(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise RuntimeError(f"resume equivalence mismatch: {label}")


def _budget_snapshot(budget) -> dict:
    value = budget.as_dict()
    value["triplet_chunks"] = int(getattr(budget, "probe_triplet_chunks", 0))
    return value


def _probe_work_snapshot(budget, counted_work: Mapping[str, int]) -> dict:
    """Merge driver work and actual tail invocations without double counting."""
    value = _budget_snapshot(budget)
    value.update({key: int(counted_work.get(key, 0)) for key in
                  ("tail_calls", "tail_calls_with_grad", "tail_calls_without_grad")})
    return value


def _validate_parent_schedule(cfg: Mapping, driver) -> list[str]:
    """Bind the probe to the frozen parent's transaction schedule."""
    if int(cfg["max_new_tokens"]) < 10:
        raise ValueError("Banking77 probe requires at least ten answer-generation tokens")
    if int(cfg["gkd_calls_per_event"]) != 20 or int(cfg["lti_calls_per_event"]) != 10:
        raise ValueError("probe must use the frozen V10 transaction schedule of 20 GKD and 10 LTI calls")
    schedule = driver.sleep_method_schedule(cfg)
    if schedule != list(METHODS) * 10 or CALLS != len(schedule):
        raise ValueError("probe's ten [GKD,GKD,LTI] triplets must match the frozen parent schedule")
    return schedule


def _causal_objectives(tail_runtime):
    kl = getattr(tail_runtime, "causal_token_kl", None)
    ce = getattr(tail_runtime, "causal_token_ce", None)
    if not callable(kl) or not callable(ce):
        raise TypeError("tail runtime must expose causal_token_kl and causal_token_ce")
    return kl, ce


def _apply_gold_update(tx, loss, token_count: int, *, legacy_fixedmix: bool) -> None:
    """Update the gold transaction without mislabeling legacy counters as GKD."""
    tokens = max(1, int(token_count))
    if not legacy_fixedmix:
        tx.optimize(loss, "gold", tokens)
        return
    before = int(tx.counters.get("gkd_calls", 0))
    tx.optimize(loss, "gkd", tokens)  # The legacy lifecycle has no gold kind.
    if int(tx.counters.get("gkd_calls", 0)) != before + 1:
        raise RuntimeError("legacy gold update did not increment its expected compatibility counter")
    tx.counters["gkd_calls"] = before
    tx.counters["gold_ce_calls"] = int(tx.counters.get("gold_ce_calls", 0)) + 1


def _lti_reward_callback(runtime, cfg):
    """Bind probe LTI calls to Banking runtime's full-v8 Eq. 4 local proxy."""
    reward = getattr(runtime, "local_lti_reward", None)
    if not callable(reward):
        raise TypeError("Banking runtime must expose local_lti_reward")

    def callback(teacher_text, student_text, *, teacher_tokens, student_tokens,
                 z0, canonical_labels):
        return reward(teacher_text, student_text, teacher_tokens=teacher_tokens,
            student_tokens=student_tokens, z0=z0, canonical_labels=canonical_labels,
            semantic_weight=float(cfg["lti_semantic_weight"]),
            edit_weight=float(cfg["lti_edit_weight"]))

    return callback


def _run_triplet(model, tx, schedule, *, runtime, common, driver, cfg, tokenizer,
                 classes, torch, device, budget, arm: str) -> dict:
    start = int(schedule[0]["call"])
    _require(len(schedule) == 3 and start % 3 == 0, "event execution requires one complete aligned triplet")
    selected = [(item["method"], item["row"], "old") for item in schedule]
    before_updates = int(tx.counters.get("adamw_steps", 0))
    before_gkd = int(tx.counters.get("gkd_calls", 0))
    before_lti = int(tx.counters.get("lti_calls", 0))
    before_gold = int(tx.counters.get("gold_calls", tx.counters.get("gold_ce_calls", 0)))
    if arm.startswith("SLEEP_"):
        import clinc_tail_runtime_v8 as tail_runtime
        causal_kl, causal_ce = _causal_objectives(tail_runtime)
        import banking_cil_v10_runtime as banking_runtime
        lti_reward_adapter = _lti_reward_callback(banking_runtime, cfg)
        result = runtime.run_sleep_event(model, tx, selected, tokenizer, classes, common, cfg,
            torch, device, budget, tail_runtime, sample_tokens=driver._sample_tokens,
            lti_reward=lti_reward_adapter, causal_token_kl=causal_kl,
            causal_token_ce=causal_ce)
        records = result["per_step"]
    else:
        import clinc_tail_runtime_v8 as tail_runtime
        records = []
        for _kind, row, _role in selected:
            def apply_loss(loss, token_count):
                _apply_gold_update(tx, loss, token_count, legacy_fixedmix=arm.endswith("FIXEDMIX"))
            outcome = runtime.run_gold_replay_step(model, tx.student_context(), row, tokenizer,
                classes, common, cfg, torch, device, budget, tail_runtime, apply_loss=apply_loss)
            records.append(outcome)
    physical = int(tx.counters.get("adamw_steps", 0)) - before_updates
    budget.adamw_updates += physical
    budget.probe_triplet_chunks = int(getattr(budget, "probe_triplet_chunks", 0)) + 1
    budget.check()
    return {"call_start": start, "call_stop_exclusive": start + len(records),
            "opportunities": len(records), "physical_adamw_updates": physical,
            "gkd_calls": int(tx.counters.get("gkd_calls", 0)) - before_gkd,
            "lti_calls": int(tx.counters.get("lti_calls", 0)) - before_lti,
            "gold_ce_calls": (int(tx.counters.get("gold_calls", tx.counters.get("gold_ce_calls", 0))) - before_gold),
            "skipped_zero_reward": sum(bool(item.get("skipped_zero_reward", False)) for item in records),
            "losses": [float(item.get("loss", 0.0)) for item in records],
            "token_counts": [int(item.get("token_count", 0)) for item in records]}


def _score(model, contexts: Mapping[str, object], eval_rows, tokenizer, classes, common,
           runtime, tail_runtime, torch, device, cfg, budget) -> dict:
    with torch.no_grad():
        result = runtime.score_candidates(model, contexts, eval_rows, tokenizer, classes,
            common, tail_runtime, torch=torch, device=device,
            max_sequence_length=int(cfg["max_sequence_length"]),
            batch_size=int(cfg["candidate_batch_size"]), budget=budget,
            class_order=cfg["_probe_class_order_seed17"], task_id=0, include_greedy=False)
    clean = {}
    for name in contexts:
        clean[name] = _metric_payload(result, name)
        if clean[name]["count"] != len(eval_rows) or clean[name]["length_mean_count"] != len(eval_rows):
            raise RuntimeError("candidate readout denominator mismatch")
    return clean


def _safe_output(path: Path, *, source_lock_sha: str, config_sha: str,
                 banking_driver_source_lock_sha: str, manifest_sha: str,
                 foundation_sha: str, receipt_sha: str,
                 schedule_receipt: dict, metrics: dict, arms: dict,
                 work: dict, started_wall: float, status: str = "complete") -> dict:
    return {"schema": SCHEMA, "status": status, "seed": SEED, "task_id": TASK_ID,
            "development_only": True, "test_data_read": False,
            "source_lock_sha256": source_lock_sha, "config_sha256": config_sha,
            "banking_driver_source_lock_sha256": banking_driver_source_lock_sha,
            "training_manifest_sha256": manifest_sha,
            "foundation_checkpoint_sha256": foundation_sha,
            "foundation_receipt_sha256": receipt_sha,
            "schedule": schedule_receipt, "metrics": metrics, "arms": arms,
            "work": work, "elapsed_seconds": time.monotonic() - started_wall,
            "method_limitations": ["local CMS plus additive/fixed-mix receiver routing, not native HOPE/Sleep",
                "old task-0 prompts diagnose transfer from the shared foundation; this is not an independent task boundary",
                "GOLD arms consume natural gold-label CE and are not information- or FLOP-matched to sampled Sleep",
                "candidate likelihood ranking supplies all 77 labels and is a separate inference protocol from free generation"]}


def _preflight_inputs(args) -> dict:
    """Validate frozen V10 inputs and the qualified CP on CPU only."""
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    pack_path = Path(args.training_pack).resolve()
    config_path = Path(args.config).resolve()
    foundation_dir = Path(args.foundation_dir).resolve()
    output_dir = Path(args.out_dir).resolve()
    _require(not output_dir.exists(), "probe output path must be fresh; no overwrite")
    _require(pack_path.is_file() and config_path.is_file() and foundation_dir.is_dir() and model_path.is_dir(),
             "model, training pack, config, and foundation directory are required")
    _require(project_root.is_dir(), "pinned nested_sleep project root is required")

    task_root, tools_dir = _bootstrap_task_paths()
    from tools import run_banking_cil_v10 as driver
    driver._validate_output_path(output_dir, project_root, model_path)
    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = driver.validate_config(raw_cfg, phase="foundation")
    config_sha = driver._config_sha(cfg)
    method_schedule = _validate_parent_schedule(cfg, driver)

    source_hashes, source_lock_sha, manifest, train_rows, dev_rows = driver._source_prepare(
        project_root, task_root, model_path, pack_path, cfg)
    from tools import banking_cil_common_v10 as common
    from tools import banking_cil_v10_runtime as runtime
    import clinc_tail_runtime_v8 as tail_runtime
    from banking_cil_v10_lifecycle import AdditiveSleepTransaction, AdditiveWakeController
    from clinc_controlled_routing_v8 import ControlledSleepTransaction
    from clinc_tail_runtime_v8 import clone_mutable_context

    probe_sources = dict(source_hashes)
    for source in (Path(__file__).resolve(), tools_dir / "probe_banking_transfer_v10.py",
                   tools_dir / "clinc_controlled_routing_v8.py", tools_dir / "banking_cil_v10_runtime.py"):
        if source.is_file():
            probe_sources[str(source.resolve())] = _sha256_file(source)
    probe_source_lock_sha = _canonical_sha(probe_sources)
    foundation_receipt_path = foundation_dir / "foundation-receipt.json"
    if not foundation_receipt_path.is_file():
        raise FileNotFoundError(foundation_receipt_path)
    receipt = json.loads(foundation_receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("seed") == SEED and receipt.get("status") == "qualified",
             "probe is restricted to the qualified seed-17 foundation")
    _require(receipt.get("config_sha256") == config_sha, "foundation config hash mismatch")
    _require(receipt.get("source_lock_sha256") == source_lock_sha, "foundation source lock mismatch")
    _require(receipt.get("training_manifest_sha256") == cfg["training_manifest_sha256"],
             "foundation training manifest mismatch")
    _require(receipt.get("test_data_included") is False, "foundation must affirm test isolation")
    _require(receipt.get("model_immutable_versions_unchanged") is True, "foundation immutable model guard failed")
    cp_relative = Path(str(receipt.get("checkpoint", "")))
    win_relative = PureWindowsPath(str(receipt.get("checkpoint", "")))
    _require(not cp_relative.is_absolute() and not win_relative.is_absolute() and not win_relative.drive and
             ".." not in cp_relative.parts and ".." not in win_relative.parts, "unsafe foundation checkpoint path")
    checkpoint_path = (foundation_dir / cp_relative).resolve()
    _require(foundation_dir in checkpoint_path.parents and checkpoint_path.is_file(), "foundation checkpoint path escaped or missing")
    foundation_sha = _sha256_file(checkpoint_path)
    receipt_sha = _sha256_file(foundation_receipt_path)
    _require(foundation_sha == receipt.get("checkpoint_sha256"), "foundation checkpoint bytes mismatch")
    _require(receipt.get("gate", {}).get("passed") is True, "foundation acquisition gate did not pass")
    _require(float(receipt.get("metric", {}).get("accuracy", -1)) >= float(cfg["task0_global_candidate_gate"]),
             "foundation receipt misses the global candidate gate")
    _require(float(receipt.get("metric", {}).get("task_class_macro_accuracy", -1)) >= float(cfg["task0_class_macro_gate"]),
             "foundation receipt misses the task-0 macro gate")

    # Payload validation stays CPU-only: deserialize only the mutable adapter
    # checkpoint after checking source/data/model-asset hashes.
    import torch
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_foundation_payload(receipt, state, seed=SEED, config_sha256=config_sha,
        source_lock_sha256=source_lock_sha, training_manifest_sha256=cfg["training_manifest_sha256"],
        checkpoint_sha256=foundation_sha)
    _require(not torch.cuda.is_initialized(), "CPU preflight unexpectedly initialized CUDA")
    cpu_checkpoint_receipt = driver._validate_foundation_checkpoint_cpu(checkpoint_path,
        foundation_sha, seed=SEED, cfg=cfg, source_lock=source_lock_sha,
        manifest_sha=cfg["training_manifest_sha256"])
    budget_cfg = driver._budget_config(cfg, "foundation")
    resolved_limits = {name: budget_cfg.get(name) for name in
        ("max_wall_seconds", "max_forward_calls", "max_update_steps")}
    _require(all(type(value) is int and value > 0 for value in resolved_limits.values()),
             "foundation probe budget limits did not resolve from the frozen config")
    budget = driver.Budget(budget_cfg)
    preflight_summary = {"schema": "banking-cil-v10-probe-preflight-v1", "status": "preflight_pass",
        "config_sha256": config_sha, "banking_driver_source_lock_sha256": source_lock_sha,
        "probe_source_lock_sha256": probe_source_lock_sha,
        "training_manifest_sha256": cfg["training_manifest_sha256"],
        "foundation_checkpoint_sha256": foundation_sha,
        "foundation_receipt_sha256": receipt_sha,
        "method_schedule": list(method_schedule), "opportunity_count": CALLS,
        "parent_gkd_calls_per_transaction": int(cfg["gkd_calls_per_event"]),
        "parent_lti_calls_per_transaction": int(cfg["lti_calls_per_event"]),
        "resolved_probe_budget": resolved_limits,
        "train_rows": len(train_rows), "development_rows": len(dev_rows),
        "foundation_checkpoint_cpu_validation": cpu_checkpoint_receipt,
        "model_assets_verified": True, "model_loaded": False, "optimizer_updates": 0,
        "cuda_initialized": False, "test_data_read": False}
    return {"summary": preflight_summary, "project_root": project_root, "model_path": model_path,
        "pack_path": pack_path, "config_path": config_path, "foundation_dir": foundation_dir,
        "output_dir": output_dir, "driver": driver, "cfg": cfg, "config_sha": config_sha,
        "source_hashes": source_hashes, "source_lock_sha": source_lock_sha,
        "manifest": manifest, "train_rows": train_rows, "dev_rows": dev_rows,
        "common": common, "runtime": runtime, "tail_runtime": tail_runtime,
        "AdditiveSleepTransaction": AdditiveSleepTransaction,
        "AdditiveWakeController": AdditiveWakeController,
        "ControlledSleepTransaction": ControlledSleepTransaction,
        "clone_mutable_context": clone_mutable_context, "probe_sources": probe_sources,
        "probe_source_lock_sha": probe_source_lock_sha, "receipt": receipt,
        "foundation_sha": foundation_sha, "receipt_sha": receipt_sha,
        "checkpoint_path": checkpoint_path, "foundation_state": state,
        "budget_cfg": budget_cfg, "budget_preflight": budget, "torch": torch}


def run(args) -> dict:
    checked = _preflight_inputs(args)
    if getattr(args, "phase", "run") == "preflight":
        receipt_path = Path(args.preflight_receipt).resolve()
        _require(not receipt_path.exists(), "preflight receipt path must be fresh; no overwrite")
        checked["driver"]._validate_output_path(receipt_path, checked["project_root"], checked["model_path"])
        _require(receipt_path != checked["output_dir"] and checked["output_dir"] not in receipt_path.parents,
                 "preflight receipt must be outside the probe output directory")
        _atomic_json(receipt_path, checked["summary"])
        return checked["summary"]

    project_root = checked["project_root"]
    model_path = checked["model_path"]
    pack_path = checked["pack_path"]
    output_dir = checked["output_dir"]
    driver = checked["driver"]
    cfg = checked["cfg"]
    config_sha = checked["config_sha"]
    source_hashes = checked["source_hashes"]
    source_lock_sha = checked["source_lock_sha"]
    manifest = checked["manifest"]
    train_rows = checked["train_rows"]
    dev_rows = checked["dev_rows"]
    common = checked["common"]
    runtime = checked["runtime"]
    tail_runtime = checked["tail_runtime"]
    AdditiveSleepTransaction = checked["AdditiveSleepTransaction"]
    AdditiveWakeController = checked["AdditiveWakeController"]
    ControlledSleepTransaction = checked["ControlledSleepTransaction"]
    clone_mutable_context = checked["clone_mutable_context"]
    probe_sources = checked["probe_sources"]
    probe_source_lock_sha = checked["probe_source_lock_sha"]
    receipt = checked["receipt"]
    foundation_sha = checked["foundation_sha"]
    receipt_sha = checked["receipt_sha"]
    checkpoint_path = checked["checkpoint_path"]
    state = checked["foundation_state"]
    budget_cfg = checked["budget_cfg"]
    torch = checked["torch"]
    _require(not output_dir.exists(), "probe output path must be fresh; no overwrite")
    output_dir.mkdir(parents=False, exist_ok=False)
    started_wall = time.monotonic()
    budget = driver.Budget(budget_cfg)
    budget.started = started_wall
    result: dict[str, Any] = {"phase": "load_model", "completed_calls": 0, "completed_arms": []}
    model = None
    original_model_logits = None
    original_tail_logits = None
    counted_work = {"tail_calls": 0, "tail_calls_with_grad": 0, "tail_calls_without_grad": 0}
    try:
        loaded_torch, tokenizer, model, immutable_versions = driver._load_model(project_root, model_path, cfg)
        if loaded_torch is not torch:
            raise RuntimeError("model loader returned a different torch module")
        frozen_versions = immutable_versions(model)
        model.requires_grad_(False)
        original_model_logits = model.logits

        def counted_logits(tokens, mask=None):
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_model_logits(tokens, mask)

        model.logits = counted_logits
        original_tail_logits = tail_runtime.tail_logits

        def counted_tail_logits(tail_model, context, cache):
            active = torch.is_grad_enabled() and any(parameter.requires_grad for parameter in context.parameters())
            budget.check()
            counted_work["tail_calls"] += 1
            counted_work["tail_calls_with_grad"] += int(active)
            counted_work["tail_calls_without_grad"] += int(not active)
            return original_tail_logits(tail_model, context, cache)

        tail_runtime.tail_logits = counted_tail_logits
        model_device = next(model.parameters()).device
        from clinc_tail_runtime_v8 import clone_mutable_context
        # Confirm wake/pending state can be reconstructed; the probe then uses
        # clean unwrapped clones so legacy fixed-mix wrappers never alias hooks.
        validation_context = clone_mutable_context(model)
        validation_wake = AdditiveWakeController(validation_context, periods=cfg["periods"],
            base_lr=cfg["base_lr"], fast_lr=cfg["fast_lr"])
        driver._restore_context_wake(validation_context, validation_wake, state, torch)
        pending = {str(key): int(value) for key, value in validation_wake.clocks.items()}
        _require(int(state["step"]) > 0, "foundation checkpoint has no completed wake updates")
        del validation_context, validation_wake
        gc.collect()

        classes = tuple(manifest["classes"])
        order = tuple(manifest["class_orders"][str(SEED)])
        cfg["_probe_class_order_seed17"] = list(order)
        old_ids = set(common.task_class_ids(order, TASK_ID))
        eval_rows = [row for row in dev_rows if int(row["intent_id"]) in old_ids]
        _require(len(eval_rows) == 110, "expected exactly ten dev prompts for each of eleven task-0 classes")
        schedule = select_old_prompt_schedule(train_rows, order)
        schedule_receipt = safe_schedule_receipt(schedule)
        result = {"phase": "baseline_evaluation", "completed_calls": 0, "completed_arms": []}
        _atomic_json(output_dir / "protocol.json", {"schema": SCHEMA, "development_only": True,
            "test_data_read": False, "seed": SEED, "task_id": TASK_ID, "arms": list(ARMS),
            "foundation_checkpoint_sha256": foundation_sha, "foundation_receipt_sha256": receipt_sha,
            "source_lock_sha256": probe_source_lock_sha, "banking_driver_source_lock_sha256": source_lock_sha,
            "config_sha256": config_sha, "training_manifest_sha256": cfg["training_manifest_sha256"],
            "method_schedule": list(METHODS), "calls": CALLS,
            "prompt_selection": schedule_receipt,
            "prompt_policy": "fixed old-only task-0 train prompts; sorted source-bound IDs; four per class; class round-robin; each of eleven classes appears two or three times",
            "evaluation": "110 task-0 development examples scored by unconstrained global 77-label likelihood; no task mask",
            "foundation_pending_clocks": pending,
            "resume_check": "SLEEP_ADDITIVE snapshot at opportunity 3; continuous and disk-restored opportunities 4-6 must match exact losses, Adam/context, transaction RNG and process RNG",
            "scope_limitations": ["identity prospective base update leaves sender base unchanged; sender reset and receiver policy are explicit interventions",
                "not a scheduled main task boundary", "Gold consumes natural labels; equal call counts are not information/FLOP matched",
                "fixed-mix arms are local routing controls, not native HOPE/Sleep"]})

        # Candidate-ranked teacher reference is checked independently on the
        # fixed task-0 development cohort before any transfer transaction.
        foundation_context = clone_mutable_context(model)
        foundation_context.load_state_dict(state["context"])
        baseline = _score(model, {"foundation": foundation_context}, eval_rows, tokenizer,
            classes, common, runtime, tail_runtime, torch, model_device, cfg, budget)["foundation"]
        _require(baseline["correct"] == int(receipt["metric"]["correct"]),
                 "independent task-0 score differs from foundation receipt correct count")
        result["teacher_competence"] = {"receipt_gate_metric": receipt["metric"], "independent_task0_dev": baseline}
        probe_metrics = {"foundation_before": baseline}
        arm_results: dict[str, Any] = {}
        _atomic_json(output_dir / "progress.json", {"phase": "probe", "status": "running",
            "completed_arms": [], "work": _probe_work_snapshot(budget, counted_work)})

        for arm in ARMS:
            result["phase"] = f"{arm}_initialization"
            driver._restore_rng(state["rng"], torch)
            live = clone_mutable_context(model)
            live.load_state_dict(state["context"])
            names = set(__import__("clinc_lifecycle_v8")._base_names(0))
            params = dict(live.named_parameters())
            prospective = {name: params[name].detach().clone() for name in names}
            tx_seed = int(cfg["initialization_seed"]) + SEED
            if arm.endswith("ADDITIVE"):
                tx = AdditiveSleepTransaction(live, 0, prospective, lr=cfg["sleep_lr"],
                    weight_decay=cfg["sleep_weight_decay"], clip_norm=cfg["sleep_clip_norm"], seed=tx_seed)
            else:
                tx = ControlledSleepTransaction(live, 0, prospective, lr=cfg["sleep_lr"],
                    weight_decay=cfg["sleep_weight_decay"], clip_norm=cfg["sleep_clip_norm"], seed=tx_seed)
            # One actual transaction/boundary per arm; 3-call execution chunks
            # and resume replay work are tracked separately.
            budget.sleep_events += 1
            reset_metrics = _score(model, {"student": tx.student_context()}, eval_rows, tokenizer,
                classes, common, runtime, tail_runtime, torch, model_device, cfg, budget)["student"]
            arm_record: dict[str, Any] = {"route_family": "additive" if arm.endswith("ADDITIVE") else "fixed_mix_0.5",
                "prospective_sender_base": "identity_level0", "sender": 0, "receiver": 1,
                "receiver_slot": int(tx.slot), "reset_policy": "redraw_A_normal_0_0.02_zero_B",
                "reset_state_dev": reset_metrics, "metrics_after_opportunities": {"0": reset_metrics},
                "transaction_counters_at_start": dict(tx.counters)}

            if arm == "SLEEP_ADDITIVE":
                # Opportunity three is a real on-disk recovery proof, not a
                # returned-in-memory copy. Snapshot includes Adam, model,
                # transaction generator and Python/NumPy/Torch process RNG.
                first = _run_triplet(model, tx, schedule[0:3], runtime=runtime, common=common,
                    driver=driver, cfg=cfg, tokenizer=tokenizer, classes=classes, torch=torch,
                    device=model_device, budget=budget, arm=arm)
                arm_record["opportunity_3"] = first
                snap3 = tx.snapshot()
                arm_record["metrics_after_opportunities"]["3"] = _score(model,
                    {"student": tx.student_context()}, eval_rows, tokenizer, classes, common,
                    runtime, tail_runtime, torch, model_device, cfg, budget)["student"]
                envelope = {"schema": "banking-cil-v10-probe-resume-v1", "arm": arm,
                    "seed": SEED, "next_call": 3, "config_sha256": config_sha,
                    "source_lock_sha256": probe_source_lock_sha,
                    "foundation_checkpoint_sha256": foundation_sha,
                    "training_manifest_sha256": cfg["training_manifest_sha256"],
                    "transaction_snapshot": snap3}
                cp_path = output_dir / "private-checkpoints" / "sleep-additive-next-call0003.pt"
                cp_sha = _atomic_torch(cp_path, envelope, torch)
                cp_loaded = torch.load(cp_path, map_location="cpu", weights_only=False)
                _require(cp_loaded.get("schema") == envelope["schema"] and cp_loaded.get("next_call") == 3 and
                         _sha256_file(cp_path) == cp_sha, "probe recovery checkpoint readback mismatch")

                continuous = AdditiveSleepTransaction.from_snapshot(live, snap3)
                continuous_triplet = _run_triplet(model, continuous, schedule[3:6], runtime=runtime,
                    common=common, driver=driver, cfg=cfg, tokenizer=tokenizer, classes=classes,
                    torch=torch, device=model_device, budget=budget, arm=arm)
                continuous_state = continuous.snapshot()
                continuous_hash = _tree_hash(continuous_state, torch)

                restored = AdditiveSleepTransaction.from_snapshot(live, cp_loaded["transaction_snapshot"])
                restored_triplet = _run_triplet(model, restored, schedule[3:6], runtime=runtime,
                    common=common, driver=driver, cfg=cfg, tokenizer=tokenizer, classes=classes,
                    torch=torch, device=model_device, budget=budget, arm=arm)
                restored_state = restored.snapshot()
                restored_hash = _tree_hash(restored_state, torch)
                _same_exact(continuous_triplet, restored_triplet, "calls 4-6 reports")
                _same_exact(continuous_hash, restored_hash, "model/Adam/controller/RNG snapshot hash")
                arm_record["disk_resume_equivalence"] = {"checkpoint_relative_path": str(cp_path.relative_to(output_dir)),
                    "checkpoint_sha256": cp_sha, "continuous_calls_4_6": continuous_triplet,
                    "restored_calls_4_6": restored_triplet,
                    "continuous_snapshot_sha256": continuous_hash,
                    "restored_snapshot_sha256": restored_hash,
                    "exact_model_adam_tx_rng_process_rng": True}
                tx = restored
                remaining_schedule = schedule[6:]
                # Already consumed opportunities 0..5; finish fixed call 30.
                for start in range(0, len(remaining_schedule), 3):
                    trip = _run_triplet(model, tx, remaining_schedule[start:start + 3], runtime=runtime,
                        common=common, driver=driver, cfg=cfg, tokenizer=tokenizer, classes=classes,
                        torch=torch, device=model_device, budget=budget, arm=arm)
                    if trip["call_stop_exclusive"] == 30:
                        arm_record["opportunity_30"] = trip
                arm_record["metrics_after_opportunities"]["30"] = _score(model,
                    {"student": tx.student_context()}, eval_rows, tokenizer, classes, common,
                    runtime, tail_runtime, torch, model_device, cfg, budget)["student"]
            else:
                accepted_calls = 0
                for start in range(0, CALLS, 3):
                    trip = _run_triplet(model, tx, schedule[start:start + 3], runtime=runtime,
                        common=common, driver=driver, cfg=cfg, tokenizer=tokenizer, classes=classes,
                        torch=torch, device=model_device, budget=budget, arm=arm)
                    accepted_calls = trip["call_stop_exclusive"]
                    if accepted_calls in (3, 30):
                        arm_record["opportunity_" + str(accepted_calls)] = trip
                        arm_record["metrics_after_opportunities"][str(accepted_calls)] = _score(model,
                            {"student": tx.student_context()}, eval_rows, tokenizer, classes, common,
                            runtime, tail_runtime, torch, model_device, cfg, budget)["student"]
            arm_record["final_transaction_counters"] = dict(tx.counters)
            arm_record["physical_adamw_updates"] = int(tx.counters.get("adamw_steps", 0))
            arm_record["opportunity_count"] = CALLS
            arm_record["actual_stage_metrics"] = {key: value for key, value in arm_record["metrics_after_opportunities"].items()}
            arm_results[arm] = arm_record
            result["completed_calls"] = CALLS
            result["completed_arms"] = list(arm_results)
            _atomic_json(output_dir / "progress.json", {"phase": "probe", "status": "running",
                "completed_arms": list(arm_results), "active_arm": arm,
                "work": _probe_work_snapshot(budget, counted_work)})
            # No arm may mutate its foundation, and every arm is rebuilt from
            # the same immutable source context before the next comparison.
            del tx, live
            gc.collect()

        _require(immutable_versions(model) == frozen_versions, "immutable model versions changed during transfer probe")
        result_obj = _safe_output(output_dir, source_lock_sha=probe_source_lock_sha,
            banking_driver_source_lock_sha=source_lock_sha,
            config_sha=config_sha, manifest_sha=cfg["training_manifest_sha256"],
            foundation_sha=foundation_sha, receipt_sha=receipt_sha,
            schedule_receipt=schedule_receipt, metrics=probe_metrics,
            arms=arm_results, work=_probe_work_snapshot(budget, counted_work), started_wall=started_wall)
        result_obj["teacher_competence"] = result["teacher_competence"]
        result_obj["model_immutable_versions_unchanged"] = True
        _atomic_json(output_dir / "result.json", result_obj)
        _atomic_json(output_dir / "progress.json", {"phase": "probe", "status": "complete",
            "completed_arms": list(arm_results), "work": _probe_work_snapshot(budget, counted_work)})
        return result_obj
    except Exception as exc:
        _atomic_json(output_dir / "failure.json", {"schema": SCHEMA, "phase": result.get("phase"),
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "completed_calls": result.get("completed_calls", 0),
            "completed_arms": result.get("completed_arms", []),
            "work": _probe_work_snapshot(budget, counted_work),
            "work_scope": "current_attempt_only; archived prior attempts remain separate",
            "test_data_read": False})
        raise
    finally:
        if model is not None and original_model_logits is not None:
            model.logits = original_model_logits
        if original_tail_logits is not None:
            tail_runtime.tail_logits = original_tail_logits
        gc.collect()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-pack", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--foundation-dir", type=Path, required=True,
                        help="seed17 qualified task-0 foundation directory")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="fresh private output directory for diagnostic checkpoints/results")
    parser.add_argument("--phase", choices=("preflight", "run"), default="run",
                        help="CPU-only source/receipt/checkpoint preflight or full model probe")
    parser.add_argument("--preflight-receipt", type=Path,
                        help="fresh CPU-only receipt path required with --phase preflight")
    args = parser.parse_args(argv)
    if args.phase == "preflight" and args.preflight_receipt is None:
        parser.error("--preflight-receipt is required with --phase preflight")
    result = run(args)
    print(json.dumps({"schema": result["schema"], "status": result["status"],
                      "phase": args.phase,
                      "arms": list(result["arms"]) if "arms" in result else None,
                      "resolved_probe_budget": result.get("resolved_probe_budget"),
                      "work": result.get("work")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
