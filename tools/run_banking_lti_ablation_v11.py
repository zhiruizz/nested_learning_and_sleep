#!/usr/bin/env python3
"""Two-task Banking77 LTI reward-term ablation, bound to frozen V10 inputs.

This local ablation changes only the LTI reward callback: either suppress the
reward term for every LTI call or suppress it unless the teacher text normalizes
to one of the canonical Banking labels. The canonical gate does not consult the
row's gold label. GKD calls, LTI prefix sampling, KL, task-1 wake updates, and V10
Sleep scheduling remain shared. This is not a paper-faithful Sleep reproduction
and does not read the official test set.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

SCHEMA = "banking-cil-v11-lti-ablation-config-v1"
CHECKPOINT_SCHEMA = "banking-cil-v11-checkpoint-v1"
RECEIPT_SCHEMA = "banking-cil-v11-run-receipt-v1"
SEEDS = (17, 29, 43)
ARMS = ("LTI_NO_REWARD", "LTI_CANONICAL_TEACHER")
BASE_ARMS = ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")
BASE_SEAL_SCHEMA = "banking-cil-v10-training-seal-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return sha256_bytes(raw)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return path.resolve() != root.resolve()
    except ValueError:
        return False


def _load_driver():
    tools_dir = Path(__file__).resolve().parent
    task_root = tools_dir.parent
    for entry in (tools_dir, task_root):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import run_banking_cil_v10 as driver
    return driver


def candidate_source_hashes(project_root: Path) -> dict[str, str]:
    """Bind the V10 production lock plus this standalone V11 source."""
    driver = _load_driver()
    task_root = Path(__file__).resolve().parents[1]
    base = driver._source_hashes(project_root, task_root)
    own = Path(__file__).resolve()
    base["tools/run_banking_lti_ablation_v11.py"] = sha256_file(own)
    return dict(sorted(base.items()))


def validate_config(cfg: Mapping, base_cfg: Mapping, *, actual_source_hashes: Mapping[str, str],
                    actual_seal_sha256: str, actual_manifest_sha256: str) -> dict:
    required = {
        "schema", "status", "v10_config_sha256", "v10_source_lock_sha256",
        "v10_source_hashes", "source_hashes", "baseline_training_seal_sha256",
        "training_manifest_sha256", "model_revision", "model_assets_sha256",
        "initialization_seed", "seeds", "arms", "task_id", "epochs",
        "batch_size", "sleep_lr", "sleep_weight_decay", "sleep_clip_norm",
        "lti_semantic_weight", "lti_edit_weight",
        "lti_teacher_validity_policy", "objective_version", "test_data_included", "max_wall_seconds",
        "max_forward_calls", "max_update_steps", "candidate_batch_size",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"V11 config missing required keys: {sorted(missing)}")
    if cfg["schema"] != SCHEMA or cfg["status"] != "frozen":
        raise ValueError("V11 execution requires the frozen V11 config schema")
    driver = _load_driver()
    normalized_base = driver.validate_config(dict(base_cfg), phase="main")
    if normalized_base.get("status") != "frozen":
        raise ValueError("V10 base config must be frozen")
    v10_cfg_sha = driver._config_sha(normalized_base)
    v10_source_lock = canonical_sha(normalized_base["source_hashes"])
    if cfg["v10_config_sha256"] != v10_cfg_sha:
        raise ValueError("V11 config V10-config binding mismatch")
    if cfg["v10_source_hashes"] != normalized_base["source_hashes"]:
        raise ValueError("V11 config V10 source-map binding mismatch")
    if cfg["v10_source_lock_sha256"] != v10_source_lock:
        raise ValueError("V11 config V10 source-lock mismatch")
    if dict(cfg["source_hashes"]) != dict(actual_source_hashes):
        raise ValueError("V11 source-lock mismatch")
    if cfg["baseline_training_seal_sha256"] != actual_seal_sha256:
        raise ValueError("V10 baseline seal binding mismatch")
    if cfg["training_manifest_sha256"] != actual_manifest_sha256 or actual_manifest_sha256 != normalized_base["training_manifest_sha256"]:
        raise ValueError("V11 training-manifest binding mismatch")
    for key in ("model_revision", "model_assets_sha256", "initialization_seed"):
        if cfg[key] != normalized_base[key]:
            raise ValueError(f"V11 config differs from V10 for {key}")
    if list(cfg["seeds"]) != list(SEEDS) or list(cfg["arms"]) != list(ARMS):
        raise ValueError("V11 seed/arm grid must be the fixed 3-seed x 2-arm grid")
    if cfg["task_id"] != 1 or cfg["epochs"] != normalized_base["epochs_per_task"] or cfg["batch_size"] != normalized_base["batch_size"]:
        raise ValueError("V11 must run only the two fixed V10 task-1 epochs at batch 8")
    for name, base_name in (("sleep_lr", "sleep_lr"), ("sleep_weight_decay", "sleep_weight_decay"),
                            ("sleep_clip_norm", "sleep_clip_norm")):
        if cfg[name] != normalized_base[base_name]:
            raise ValueError(f"V11 sleep optimizer differs from V10 for {name}")
    for name in ("lti_semantic_weight", "lti_edit_weight"):
        if cfg[name] != normalized_base[name]:
            raise ValueError(f"V11 reward weight differs from V10 for {name}")
    if cfg["lti_teacher_validity_policy"] != "canonical_teacher_required_for_reward":
        raise ValueError("unknown V11 canonical-teacher reward policy")
    if cfg["objective_version"] != "v10_eq4_reward_term_ablation_with_kl_retained_v1":
        raise ValueError("unknown V11 objective version")
    if cfg["test_data_included"] is not False:
        raise ValueError("V11 must not include test data")
    for name in ("max_wall_seconds", "max_forward_calls", "max_update_steps", "candidate_batch_size"):
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["max_wall_seconds"] > normalized_base["main_max_wall_seconds"] or cfg["max_forward_calls"] > normalized_base["main_max_forward_calls"] or cfg["max_update_steps"] > normalized_base["main_max_update_steps"]:
        raise ValueError("V11 budgets may not exceed frozen V10 main caps")
    if cfg["candidate_batch_size"] > normalized_base["candidate_batch_size"]:
        raise ValueError("V11 candidate batch may not exceed frozen V10 cap")
    return dict(cfg)


def validate_foundation(receipt: Mapping, *, seed: int, config_sha256: str,
                        source_lock_sha256: str, manifest_sha256: str,
                        checkpoint_sha256: str, global_gate: float = 0.5,
                        class_macro_gate: float = 0.4) -> None:
    if (receipt.get("schema"), receipt.get("status"), receipt.get("seed"), receipt.get("task_id")) != (
            "banking-cil-v10-foundation-receipt", "qualified", seed, 0):
        raise ValueError("foundation must be a qualified task-0 receipt for this seed")
    if (receipt.get("config_sha256"), receipt.get("source_lock_sha256"),
        receipt.get("training_manifest_sha256"), receipt.get("checkpoint_sha256")) != (
            config_sha256, source_lock_sha256, manifest_sha256, checkpoint_sha256):
        raise ValueError("foundation receipt input binding mismatch")
    metric = receipt.get("metric")
    gate = receipt.get("gate")
    if (not isinstance(metric, dict) or metric.get("count") != 110 or metric.get("classes_evaluated") != 11 or
        not isinstance(metric.get("accuracy"), (int, float)) or metric["accuracy"] < global_gate or
        not isinstance(metric.get("task_class_macro_accuracy"), (int, float)) or metric["task_class_macro_accuracy"] < class_macro_gate or
        not isinstance(gate, dict) or gate.get("passed") is not True or
        gate.get("global_candidate_sum_accuracy") != global_gate or
        gate.get("task0_class_macro_accuracy") != class_macro_gate or
        receipt.get("model_immutable_versions_unchanged") is not True or
        receipt.get("test_data_included") is not False):
        raise ValueError("foundation task-0 gate or test-isolation receipt is invalid")


def validate_baseline_seal(seal_path: Path, runs_root: Path, *, expected_seal_sha256: str,
                           config_sha256: str, source_lock_sha256: str,
                           manifest_sha256: str, source_hashes: Mapping[str, str]) -> dict:
    if not seal_path.is_file() or sha256_file(seal_path) != expected_seal_sha256:
        raise ValueError("V10 baseline seal is missing or changed")
    seal = _json(seal_path)
    if (seal.get("schema"), seal.get("complete"), seal.get("test_data_included"), seal.get("official_test_read")) != (
            BASE_SEAL_SCHEMA, True, False, False):
        raise ValueError("V10 baseline training seal schema/status mismatch")
    if (seal.get("config_sha256"), seal.get("source_lock_sha256"), seal.get("training_manifest_sha256")) != (
            config_sha256, source_lock_sha256, manifest_sha256):
        raise ValueError("V10 baseline seal source/config/data binding mismatch")
    release_sha = seal.get("release_artifact_sha256")
    if not isinstance(release_sha, str) or len(release_sha) != 64:
        raise ValueError("V10 seal lacks the source-bound main release hash")
    rows = seal.get("runs")
    expected = {(seed, arm) for seed in SEEDS for arm in BASE_ARMS}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("V10 baseline seal must contain exactly nine runs")
    seen = set()
    foundation_hashes: dict[int, str] = {}
    foundation_receipt_hashes: dict[int, str] = {}
    verified = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed baseline run row")
        key = (row.get("seed"), row.get("arm"))
        if key in seen or key not in expected:
            raise ValueError("V10 baseline run grid has duplicate or unexpected row")
        seen.add(key)
        run_dir = Path(row.get("run_dir", "")).resolve()
        if not _within(run_dir, runs_root):
            raise ValueError("baseline run path is outside the declared run root")
        receipt_path = run_dir / "run-receipt.json"
        protocol_path = run_dir / "protocol.json"
        if not receipt_path.is_file() or sha256_file(receipt_path) != row.get("run_receipt_sha256"):
            raise ValueError("V10 baseline run receipt bytes mismatch")
        if not protocol_path.is_file():
            raise ValueError("V10 baseline protocol receipt missing")
        protocol = _json(protocol_path)
        parent = protocol.get("parent")
        if (protocol.get("schema"), protocol.get("phase"), protocol.get("seed"), protocol.get("arm"),
            protocol.get("config_sha256"), protocol.get("source_lock_sha256"),
            protocol.get("training_manifest_sha256"), protocol.get("source_hashes")) != (
                "banking-cil-v10-protocol", "main", key[0], key[1], config_sha256,
                source_lock_sha256, manifest_sha256, dict(source_hashes)):
            raise ValueError("V10 baseline protocol identity/source binding mismatch")
        if (not isinstance(parent, dict) or parent.get("release_artifact_sha256") != release_sha or
            not isinstance(parent.get("foundation_checkpoint_sha256"), str) or
            len(parent["foundation_checkpoint_sha256"]) != 64 or
            not isinstance(parent.get("foundation_receipt_sha256"), str) or
            len(parent["foundation_receipt_sha256"]) != 64):
            raise ValueError("V10 baseline protocol parent release/foundation binding missing")
        receipt = _json(receipt_path)
        if (receipt.get("status"), receipt.get("tasks_completed"), receipt.get("seed"), receipt.get("arm"),
            receipt.get("config_sha256"), receipt.get("source_lock_sha256"), receipt.get("training_manifest_sha256")) != (
                "complete", 7, key[0], key[1], config_sha256, source_lock_sha256, manifest_sha256):
            raise ValueError("V10 baseline receipt binding or completion mismatch")
        if receipt.get("model_immutable_versions_unchanged") is not True or receipt.get("test_data_included") is not False:
            raise ValueError("V10 baseline run immutable/test-isolation receipt failed")
        foundation_sha = receipt.get("foundation_checkpoint_sha256")
        prior_foundation = foundation_hashes.setdefault(int(key[0]), foundation_sha)
        if not isinstance(foundation_sha, str) or len(foundation_sha) != 64 or prior_foundation != foundation_sha:
            raise ValueError("paired V10 baseline arms do not share the same foundation checkpoint")
        if parent.get("foundation_checkpoint_sha256") != foundation_sha:
            raise ValueError("V10 protocol parent and run receipt foundation checkpoint differ")
        prior_receipt = foundation_receipt_hashes.setdefault(int(key[0]), parent["foundation_receipt_sha256"])
        if prior_receipt != parent["foundation_receipt_sha256"]:
            raise ValueError("paired V10 baseline arms do not share the same foundation receipt")
        final_cp = (run_dir / str(receipt.get("final_checkpoint", ""))).resolve()
        if not _within(final_cp, run_dir) or not final_cp.is_file() or sha256_file(final_cp) != receipt.get("final_checkpoint_sha256"):
            raise ValueError("V10 baseline final checkpoint bytes mismatch")
        task_rows = receipt.get("task_metrics")
        if not isinstance(task_rows, list) or [item.get("task_id") for item in task_rows if isinstance(item, dict)] != [1, 2, 3, 4, 5, 6]:
            raise ValueError("V10 baseline must contain task endpoints 1..6 in order")
        task1 = task_rows[0]
        if not isinstance(task1.get("checkpoint_sha256"), str):
            raise ValueError("V10 baseline task-1 endpoint checkpoint metadata missing")
        task_cp = (run_dir / "checkpoints" / "task01-final.pt").resolve()
        if not _within(task_cp, run_dir) or not task_cp.is_file() or sha256_file(task_cp) != task1["checkpoint_sha256"]:
            raise ValueError("V10 baseline task-1 checkpoint bytes mismatch")
        if row.get("final_checkpoint_sha256") != receipt.get("final_checkpoint_sha256"):
            raise ValueError("V10 seal row and receipt final checkpoint differ")
        verified[key] = {"run_dir": str(run_dir), "receipt_sha256": row["run_receipt_sha256"],
                         "foundation_checkpoint_sha256": foundation_sha,
                         "task1_checkpoint": "checkpoints/task01-final.pt",
                         "task1_checkpoint_sha256": task1["checkpoint_sha256"],
                         "task1_metrics": task1.get("candidate_metrics")}
    if seen != expected:
        raise ValueError("V10 baseline grid is incomplete")
    return {"runs": verified, "foundation_checkpoint_sha256_by_seed": foundation_hashes,
            "foundation_receipt_sha256_by_seed": foundation_receipt_hashes,
            "release_artifact_sha256": release_sha}


def make_reward_callback(arm: str, runtime, cfg):
    if arm not in ARMS:
        raise ValueError("unknown V11 LTI ablation arm")
    reward = getattr(runtime, "local_lti_reward", None)
    if not callable(reward):
        raise TypeError("Banking runtime must expose local_lti_reward")
    normalize = getattr(runtime, "_normalized_label", None)
    if not callable(normalize):
        raise TypeError("Banking runtime must expose its frozen label normalizer")

    def callback(teacher_text, student_text, *, teacher_tokens, student_tokens,
                 z0, canonical_labels):
        if arm == "LTI_NO_REWARD":
            return 0.0
        # Use the same normalization as the frozen local reward proxy. This
        # tests label-format validity only; no example/gold label is available
        # here, and canonical but incorrect teacher labels remain eligible.
        teacher_label = normalize(teacher_text)
        canonical = {normalize(label) for label in canonical_labels}
        if not teacher_label or teacher_label not in canonical:
            return 0.0
        return float(reward(teacher_text, student_text,
            teacher_tokens=teacher_tokens, student_tokens=student_tokens, z0=z0,
            canonical_labels=canonical_labels,
            semantic_weight=float(cfg["lti_semantic_weight"]),
            edit_weight=float(cfg["lti_edit_weight"])))
    return callback


class RuntimeFacade:
    """Per-run runtime adapter; no mutation of modules or global import state."""
    def __init__(self, runtime, arm: str, cfg):
        self._runtime = runtime
        self._arm = arm
        self._cfg = cfg

    def run_sleep_event(self, *args, **kwargs):
        kwargs["lti_reward"] = make_reward_callback(self._arm, self._runtime, self._cfg)
        return self._runtime.run_sleep_event(*args, **kwargs)


def output_path_guard(path: Path, *, protected_roots: tuple[Path, ...]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing V11 output: {path}")
    resolved = path.resolve()
    for root in protected_roots:
        root = root.resolve()
        if resolved == root or root in resolved.parents or resolved in root.parents:
            raise ValueError("V11 output must be separate from protected inputs")


def _atomic_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.partial")
    try:
        with tmp.open("xb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
        if path.read_bytes() != payload:
            raise IOError("V11 atomic JSON readback mismatch")
        return sha256_bytes(payload)
    finally:
        if tmp.exists():
            tmp.unlink()


def _append_jsonl(path: Path, value: Mapping) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _config_hash(cfg: Mapping) -> str:
    return canonical_sha(dict(cfg))


def _preflight(args) -> dict:
    driver = _load_driver()
    base_cfg = _json(args.v10_config)
    cfg = _json(args.config)
    sources = candidate_source_hashes(args.project_root)
    seal_sha = sha256_file(args.baseline_seal)
    validated = validate_config(cfg, base_cfg, actual_source_hashes=sources,
        actual_seal_sha256=seal_sha,
        actual_manifest_sha256=base_cfg.get("training_manifest_sha256", ""))
    v10_cfg = driver.validate_config(base_cfg, phase="main")
    base_lock = canonical_sha(v10_cfg["source_hashes"])
    baseline = validate_baseline_seal(args.baseline_seal, args.baseline_runs_root,
        expected_seal_sha256=seal_sha, config_sha256=driver._config_sha(v10_cfg),
        source_lock_sha256=base_lock, manifest_sha256=v10_cfg["training_manifest_sha256"],
        source_hashes=v10_cfg["source_hashes"])
    manifest, train_rows, _dev_rows = driver._load_rows(args.training_pack,
        v10_cfg["training_manifest_sha256"])
    foundation_receipt_path = args.foundation_dir / "foundation-receipt.json"
    foundation_receipt = _json(foundation_receipt_path)
    cp = (args.foundation_dir / str(foundation_receipt.get("checkpoint", ""))).resolve()
    if not _within(cp, args.foundation_dir) or not cp.is_file():
        raise ValueError("V10 foundation checkpoint path is invalid")
    cp_sha = sha256_file(cp)
    validate_foundation(foundation_receipt, seed=args.seed,
        config_sha256=driver._config_sha(v10_cfg), source_lock_sha256=base_lock,
        manifest_sha256=v10_cfg["training_manifest_sha256"], checkpoint_sha256=cp_sha,
        global_gate=v10_cfg["task0_global_candidate_gate"],
        class_macro_gate=v10_cfg["task0_class_macro_gate"])
    cpu_cp_receipt = driver._validate_foundation_checkpoint_cpu(cp, cp_sha, seed=args.seed,
        cfg=v10_cfg, source_lock=base_lock, manifest_sha=v10_cfg["training_manifest_sha256"])
    if baseline["foundation_checkpoint_sha256_by_seed"].get(args.seed) != cp_sha:
        raise ValueError("chosen foundation differs from the paired V10 baseline seed")
    if baseline["foundation_receipt_sha256_by_seed"].get(args.seed) != sha256_file(foundation_receipt_path):
        raise ValueError("chosen foundation receipt differs from the paired V10 baseline seed")
    if args.arm not in ARMS or args.seed not in SEEDS:
        raise ValueError("V11 run requires one declared seed and arm")
    output_path_guard(args.output_dir, protected_roots=(args.project_root, args.model_path,
        args.training_pack.parent, args.foundation_dir, args.baseline_runs_root))
    return {"status": "preflight_pass", "config_sha256": _config_hash(validated),
        "source_lock_sha256": canonical_sha(sources), "v10_source_lock_sha256": base_lock,
        "training_manifest_sha256": manifest["files"]["train"]["sha256"],
        "baseline_seal_sha256": seal_sha, "foundation_checkpoint_sha256": cp_sha,
        "comparison_v10_sleep_task1_checkpoint_sha256": baseline["runs"][(args.seed, "SLEEP_ADDITIVE")]["task1_checkpoint_sha256"],
        "seed": args.seed, "arm": args.arm, "train_rows": len(train_rows),
        "development_rows": manifest["files"]["development"]["rows"],
        "task1_classes": 11, "test_data_included": False, "model_loaded": False,
        "optimizer_updates": 0, "foundation_checkpoint_cpu_preflight": cpu_cp_receipt}


def _checkpoint(path: Path, *, torch, context, wake, cfg_sha, v11_lock, v10_lock,
                manifest_sha, seed, arm, foundation_sha, baseline_sha, step):
    payload = {"schema": CHECKPOINT_SCHEMA, "seed": seed, "arm": arm, "task_id": 1,
        "step": int(step), "context": context.state_dict(), "wake": wake.snapshot(),
        "rng": _load_driver()._rng_state(torch), "config_sha256": cfg_sha,
        "v11_source_lock_sha256": v11_lock, "v10_source_lock_sha256": v10_lock,
        "training_manifest_sha256": manifest_sha,
        "foundation_checkpoint_sha256": foundation_sha,
        "baseline_training_seal_sha256": baseline_sha, "test_data_included": False}
    return _load_driver()._atomic_torch(path, payload, torch)


def _run(args, preflight: Mapping) -> dict:
    driver = _load_driver()
    base_cfg = driver.validate_config(_json(args.v10_config), phase="main")
    cfg = _json(args.config)
    sources = candidate_source_hashes(args.project_root)
    v11_cfg = validate_config(cfg, base_cfg, actual_source_hashes=sources,
        actual_seal_sha256=sha256_file(args.baseline_seal),
        actual_manifest_sha256=base_cfg["training_manifest_sha256"])
    v10_lock = canonical_sha(base_cfg["source_hashes"])
    v11_lock = canonical_sha(sources)
    cfg_sha = _config_hash(v11_cfg)
    source_hashes, checked_v10_lock, manifest, train_rows, dev_rows = driver._source_prepare(
        args.project_root, Path(__file__).resolve().parents[1], args.model_path,
        args.training_pack, base_cfg)
    if checked_v10_lock != v10_lock or source_hashes != base_cfg["source_hashes"]:
        raise ValueError("V10 source lock changed after preflight")
    foundation_dir = args.foundation_dir.resolve()
    foundation_receipt_path = foundation_dir / "foundation-receipt.json"
    foundation_receipt_sha = sha256_file(foundation_receipt_path)
    foundation_receipt = _json(foundation_receipt_path)
    foundation_cp = (foundation_dir / str(foundation_receipt["checkpoint"])).resolve()
    foundation_sha = sha256_file(foundation_cp)
    validate_foundation(foundation_receipt, seed=args.seed,
        config_sha256=driver._config_sha(base_cfg), source_lock_sha256=v10_lock,
        manifest_sha256=base_cfg["training_manifest_sha256"], checkpoint_sha256=foundation_sha,
        global_gate=base_cfg["task0_global_candidate_gate"],
        class_macro_gate=base_cfg["task0_class_macro_gate"])
    baseline = validate_baseline_seal(args.baseline_seal, args.baseline_runs_root,
        expected_seal_sha256=sha256_file(args.baseline_seal),
        config_sha256=driver._config_sha(base_cfg), source_lock_sha256=v10_lock,
        manifest_sha256=base_cfg["training_manifest_sha256"],
        source_hashes=base_cfg["source_hashes"])
    if baseline["foundation_checkpoint_sha256_by_seed"].get(args.seed) != foundation_sha:
        raise ValueError("V11 foundation does not match V10 paired baseline grid")
    if baseline["foundation_receipt_sha256_by_seed"].get(args.seed) != foundation_receipt_sha:
        raise ValueError("V11 foundation receipt does not match V10 paired baseline grid")
    output_path_guard(args.output_dir, protected_roots=(args.project_root, args.model_path,
        args.training_pack.parent, foundation_dir, args.baseline_runs_root))
    args.output_dir.mkdir(parents=False, exist_ok=False)
    protocol = {"schema": "banking-cil-v11-protocol-v1", "status": "running",
        "config_sha256": cfg_sha, "source_hashes": sources,
        "source_lock_sha256": v11_lock, "v10_source_lock_sha256": v10_lock,
        "v10_config_sha256": driver._config_sha(base_cfg),
        "training_manifest_sha256": base_cfg["training_manifest_sha256"],
        "baseline_training_seal_sha256": sha256_file(args.baseline_seal),
        "foundation_receipt_sha256": foundation_receipt_sha,
        "foundation_checkpoint_sha256": foundation_sha,
        "comparison_v10_sleep_task1_checkpoint_sha256": baseline["runs"][(args.seed, "SLEEP_ADDITIVE")]["task1_checkpoint_sha256"],
        "seed": args.seed, "arm": args.arm, "task_id": 1, "epochs": 2,
        "test_data_included": False, "test_data_read": False}
    _atomic_json(args.output_dir / "protocol.json", protocol)
    work_cfg = {**base_cfg, "max_wall_seconds": v11_cfg["max_wall_seconds"],
        "max_forward_calls": v11_cfg["max_forward_calls"],
        "max_update_steps": v11_cfg["max_update_steps"]}
    budget = driver.Budget(work_cfg)
    started = time.monotonic(); budget.started = started
    tail_runtime = original_tail = None
    task_metrics = []
    event_records = []
    event_counts = {i: 0 for i, p in enumerate(base_cfg["periods"]) if p is not None}
    selector_event_index = 0
    try:
        driver._init_rng(base_cfg["initialization_seed"], __import__("torch"))
        torch, tokenizer, model, immutable_versions = driver._load_model(
            args.project_root, args.model_path, base_cfg)
        driver._verify_loaded_project_modules(args.project_root)
        if base_cfg["device"].startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        original_logits = model.logits
        def counted(tokens, mask=None):
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_logits(tokens, mask)
        model.logits = counted
        immutable_before = immutable_versions(model)
        for _, parameter in model.named_parameters():
            parameter.requires_grad_(False)
        import clinc_tail_runtime_v8 as tail
        tail_runtime, original_tail = driver._install_tail_counter(budget)
        from banking_cil_v10_lifecycle import AdditiveWakeController
        context = tail.clone_mutable_context(model)
        wake = AdditiveWakeController(context, periods=base_cfg["periods"],
            base_lr=base_cfg["base_lr"], fast_lr=base_cfg["fast_lr"])
        state = driver._restore_checkpoint(foundation_cp, context, wake,
            expected_sha=foundation_sha, seed=args.seed, arm="FOUNDATION_CMS_ONLY",
            cfg=base_cfg, torch=torch)
        if (state.get("config_sha256"), state.get("source_lock_sha256"),
            state.get("training_manifest_sha256"), state.get("task_id")) != (
                driver._config_sha(base_cfg), v10_lock,
                base_cfg["training_manifest_sha256"], 0):
            raise ValueError("foundation checkpoint lineage mismatch")
        if state.get("step") != wake.ticks:
            raise ValueError("foundation wake clock and task-0 step differ")
        order = tuple(manifest["class_orders"][str(args.seed)])
        current_rows = driver.task_rows(train_rows, order, 1)
        by_id = {str(row["id"]): row for row in current_rows}
        pools = driver._group_sleep_pools(train_rows, order, 1)
        import banking_cil_v10_runtime as real_runtime
        facade = RuntimeFacade(real_runtime, args.arm, base_cfg)
        metrics_path = args.output_dir / "metrics.jsonl"
        task_step = 0
        for epoch in range(base_cfg["epochs_per_task"]):
            epoch_ids = driver.deterministic_epoch_ids(current_rows, seed=args.seed,
                task_id=1, epoch=epoch)
            for start in range(0, len(epoch_ids), base_cfg["batch_size"]):
                budget.check()
                chunk = [by_id[row_id] for row_id in epoch_ids[start:start + base_cfg["batch_size"]]]
                tick = driver._train_tick(model, wake.context, wake, chunk, tokenizer,
                    manifest["classes"], base_cfg, torch, next(model.parameters()).device, budget)
                task_step += 1
                _append_jsonl(metrics_path, {"event": "wake_tick", "seed": args.seed,
                    "arm": args.arm, "task_id": 1, "task_step": task_step,
                    "global_wake_tick": wake.ticks, **tick})
                for level in sorted(wake.proposals()):
                    wake.context, records, selector_event_index = driver._due_events(
                        model, wake.context, wake, "SLEEP_ADDITIVE", pools, tokenizer,
                        manifest["classes"], base_cfg, torch, next(model.parameters()).device,
                        budget, facade, event_counts=event_counts,
                        selector_event_index=selector_event_index, seed=args.seed,
                        task_id=1, tx_seed=base_cfg["initialization_seed"] + args.seed * 100003 + 1 * 1009,
                        only_level=level)
                    for record in records:
                        event_records.append(record)
                        _append_jsonl(metrics_path, {"event": "sleep", "seed": args.seed,
                            "arm": args.arm, "task_id": 1, "global_wake_tick": wake.ticks,
                            **record})
                if task_step % 16 == 0:
                    _atomic_json(args.output_dir / "progress.json", {"status": "running",
                        "seed": args.seed, "arm": args.arm, "task_id": 1,
                        "task_step": task_step, "global_wake_tick": wake.ticks,
                        "event_counts": dict(event_counts), "work": budget.as_dict()})
        seen = set(order[:22])
        grouped = defaultdict(list)
        for row in dev_rows:
            if int(row["intent_id"]) in seen:
                grouped[int(row["intent_id"])].append(row)
        eval_rows = [row for cid in sorted(grouped)
                     for row in sorted(grouped[cid], key=lambda item: str(item["id"]))]
        if len(eval_rows) != 220 or len(grouped) != 22 or any(len(grouped[cid]) != 10 for cid in grouped):
            raise ValueError("V11 task-1 endpoint cohort must be all 220 seen-class dev rows")
        scored = driver._candidate_eval(real_runtime, model, wake.context, eval_rows,
            tokenizer, manifest["classes"], base_cfg, torch,
            next(model.parameters()).device, budget)
        summary = driver._extract_candidate_metric(scored)
        task_record = {"task_id": 1, "task_step": task_step,
            "global_wake_tick": wake.ticks, "evaluation_rows": len(eval_rows),
            "candidate_metrics": summary, "event_counts": dict(event_counts),
            "work": budget.as_dict()}
        cp_path = args.output_dir / "checkpoints" / "task01-final.pt"
        cp_sha = _checkpoint(cp_path, torch=torch, context=wake.context, wake=wake,
            cfg_sha=cfg_sha, v11_lock=v11_lock, v10_lock=v10_lock,
            manifest_sha=base_cfg["training_manifest_sha256"], seed=args.seed,
            arm=args.arm, foundation_sha=foundation_sha,
            baseline_sha=sha256_file(args.baseline_seal), step=wake.ticks)
        task_record["checkpoint_sha256"] = cp_sha
        task_metrics.append(task_record)
        from clinc_tail_runtime_v8 import parameter_counts
        slot_counts = [sum(bool(slot.active) for slot in bank.slots) for bank in wake.context.banks()]
        pending_values = list(wake.pending.values())
        pending_bytes = sum(int(value.numel()) * int(value.element_size()) for value in pending_values)
        objective_rows = [step for event in event_records
                          for step in event.get("objectives", {}).get("per_step", [])]
        lti_rows = [step for step in objective_rows if step.get("kind") == "lti"]
        role_exposures = {role: sum(int(event.get("requests_by_role", {}).get(role, 0))
                                   for event in event_records) for role in ("old", "current")}
        sleep_summary = {"events": len(event_records),
            "calls": len(objective_rows),
            "gkd_calls": sum(step.get("kind") == "gkd" for step in objective_rows),
            "lti_calls": len(lti_rows),
            "zero_reward_lti_calls": sum(bool(step.get("zero_reward")) for step in lti_rows),
            "empty_teacher_lti_calls": sum(bool(step.get("teacher_answer_empty")) for step in lti_rows),
            "skipped_zero_objective_calls": sum(bool(step.get("skipped_zero_objective")) for step in objective_rows),
            "physical_adamw_steps": sum(int(event.get("physical_adamw_steps", 0)) for event in event_records),
            "request_exposures_by_role": role_exposures,
            "request_ids_digest_by_event": [event.get("request_ids_sha256") for event in event_records]}
        unchanged = immutable_versions(model) == immutable_before
        if not unchanged:
            raise RuntimeError("frozen backbone changed during V11 task-1 ablation")
        receipt = {"schema": RECEIPT_SCHEMA, "status": "complete", "seed": args.seed,
            "arm": args.arm, "task_id": 1, "epochs": 2, "task1_wake_ticks": task_step,
            "foundation_checkpoint_sha256": foundation_sha,
            "foundation_receipt_sha256": foundation_receipt_sha,
            "baseline_training_seal_sha256": sha256_file(args.baseline_seal),
            "comparison_v10_sleep_task1_checkpoint_sha256": protocol["comparison_v10_sleep_task1_checkpoint_sha256"],
            "checkpoint": cp_path.relative_to(args.output_dir).as_posix(),
            "checkpoint_sha256": cp_sha, "config_sha256": cfg_sha,
            "source_lock_sha256": v11_lock, "v10_source_lock_sha256": v10_lock,
            "training_manifest_sha256": base_cfg["training_manifest_sha256"],
            "task_metrics": task_metrics, "event_counts": dict(event_counts),
            "selector_event_index": selector_event_index, "work": budget.as_dict(),
            "sleep_summary": sleep_summary, "controller_state": {
                "ticks": int(wake.ticks), "clocks": dict(wake.clocks),
                "token_counts": dict(wake.token_counts),
                "pending_tensor_count": len(pending_values),
                "pending_elements": sum(int(value.numel()) for value in pending_values),
                "pending_bytes": pending_bytes, "active_slots_by_level": slot_counts},
            "mutable_parameter_counts": parameter_counts(wake.context),
            "checkpoint_size_bytes": cp_path.stat().st_size,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if base_cfg["device"].startswith("cuda") else None,
            "model_immutable_versions_unchanged": unchanged,
            "test_data_included": False, "test_data_read": False}
        _atomic_json(args.output_dir / "run-receipt.json", receipt)
        receipt["run_receipt_sha256"] = sha256_file(args.output_dir / "run-receipt.json")
        protocol.update({"status": "complete", "run_receipt_sha256": receipt["run_receipt_sha256"],
            "checkpoint": cp_path.relative_to(args.output_dir).as_posix(), "checkpoint_sha256": cp_sha,
            "task1_candidate_metrics": summary})
        _atomic_json(args.output_dir / "protocol.json", protocol)
        _atomic_json(args.output_dir / "progress.json", {"status": "complete",
            "seed": args.seed, "arm": args.arm, "task_id": 1,
            "work": budget.as_dict()})
        driver._remove_tail_counter(tail_runtime, original_tail)
        return receipt
    except Exception as exc:
        driver._remove_tail_counter(tail_runtime, original_tail)
        _atomic_json(args.output_dir / "failure.json", {"schema": "banking-cil-v11-failure-v1",
            "phase": "task1", "seed": args.seed, "arm": args.arm,
            "error_type": type(exc).__name__, "error": str(exc),
            "task_step": task_step if "task_step" in locals() else 0,
            "event_counts": dict(event_counts), "selector_event_index": selector_event_index,
            "work": budget.as_dict(), "test_data_included": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "run"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v10-config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-pack", type=Path, required=True)
    parser.add_argument("--baseline-seal", type=Path, required=True)
    parser.add_argument("--baseline-runs-root", type=Path, required=True)
    parser.add_argument("--foundation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args(argv)
    preflight = _preflight(args)
    if args.phase == "preflight":
        print(json.dumps(preflight, sort_keys=True, ensure_ascii=False))
        return 0
    result = _run(args, preflight)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
