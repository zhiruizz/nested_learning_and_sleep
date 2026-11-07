#!/usr/bin/env python3
"""Strict, aggregate-only CLINC V8 six-run report generator.

Reads the frozen main-grid seal, master config, foundation/run receipts and
checkpoint bytes. It never loads a model or reads a test payload. An optional
completed final-test-evaluation JSON is treated as already-produced aggregate
metadata and validated against the seal and frozen evaluation contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SEEDS = (17, 29, 43)
ARMS = ("HOPE", "HOPE_SLEEP_FIXEDMIX")
TASKS = tuple(range(10))
MAIN_SCHEMA = "clinc-cil-v8-main-six-seal"
RUN_SCHEMA = "clinc-cil-v8-full-run-receipt"
FOUNDATION_SCHEMA = "clinc-cil-v8-common-foundation-receipt"
TEST_SCHEMA = "clinc-v8-final-test-evaluation-v1"
CONTRACT_SCHEMA = "clinc-v8-test-evaluation-contract-v1"
WORK_FIELDS = (
    "active_wall_seconds", "full_backbone_forward_calls", "full_backbone_forward_tokens",
    "tail_calls", "tail_backward_calls", "tail_backward_tokens",
    "optimizer_or_lifecycle_update_steps", "sleep_events",
    "executed_input_tokens_including_padding", "padded_input_tokens",
    "sampled_answer_tokens_including_first_eos",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _accuracy_record(obj: Mapping[str, Any], *, expected_class_count: int,
                     expected_count: int, label: str) -> dict[str, Any]:
    count = obj.get("count")
    class_count = obj.get("class_count")
    accuracy = obj.get("accuracy")
    macro = obj.get("macro_accuracy")
    invalid = obj.get("invalid")
    if type(count) is not int or count != expected_count:
        raise ValueError(f"{label}: invalid count")
    if count == 0:
        # The task-0 old scope is structurally empty: there are no examples to
        # classify, so the source schema uses null accuracy/invalid fields.
        if expected_class_count != 0 or class_count not in (None, 0) or accuracy is not None or macro is not None or invalid is not None:
            raise ValueError(f"{label}: empty scope must use null metrics")
        return {"correct": None, "count": 0, "accuracy": None,
                "macro_accuracy": None, "invalid": None, "class_count": 0,
                "structural_empty": True}
    if type(class_count) is not int or class_count != expected_class_count:
        raise ValueError(f"{label}: class count differs from expected task scope")
    if not _is_number(accuracy) or not 0 <= float(accuracy) <= 1:
        raise ValueError(f"{label}: accuracy missing or out of range")
    if not _is_number(macro) or not 0 <= float(macro) <= 1:
        raise ValueError(f"{label}: macro accuracy missing or out of range")
    if type(invalid) is not int or not 0 <= invalid <= count:
        raise ValueError(f"{label}: invalid-generation count out of range")
    correct = round(count * float(accuracy))
    if abs(correct / count - float(accuracy)) > 1e-12:
        raise ValueError(f"{label}: accuracy does not correspond to integer correct/count")
    if correct > count - invalid:
        raise ValueError(f"{label}: correct count exceeds non-invalid predictions")
    recorded_correct = obj.get("correct")
    if recorded_correct is not None and (type(recorded_correct) is not int or recorded_correct != correct):
        raise ValueError(f"{label}: explicit correct count differs from accuracy/count")
    return {"correct": correct, "count": count, "accuracy": float(accuracy),
            "macro_accuracy": float(macro), "invalid": invalid, "class_count": class_count,
            "structural_empty": False}


def extract_trajectory(receipt: Mapping[str, Any], *, expected_rows: Mapping[int, Mapping[str, Mapping[str, int]]]) -> list[dict[str, Any]]:
    """Validate and return aggregate-only task metrics; never preserves IDs/predictions."""
    items = receipt.get("task_metrics")
    if not isinstance(items, list) or [x.get("task_id") for x in items] != list(TASKS):
        raise ValueError("every run must have exactly one ordered clean-development endpoint for each task 0..9")
    trajectory: list[dict[str, Any]] = []
    for task_id, item in enumerate(items):
        expected_old_classes = task_id * 15
        expected_current_classes = 15
        expected_seen_classes = (task_id + 1) * 15
        result = item.get("result")
        metrics = result.get("metrics") if isinstance(result, dict) else None
        if not isinstance(metrics, dict) or set(metrics) != {"old", "current", "seen"}:
            raise ValueError(f"task {task_id}: metric scope schema mismatch")
        task_expected = expected_rows[task_id]
        old = _accuracy_record(metrics["old"], expected_class_count=task_expected["old"]["class_count"],
                               expected_count=task_expected["old"]["count"],
                               label=f"task {task_id} old")
        current = _accuracy_record(metrics["current"], expected_class_count=task_expected["current"]["class_count"],
                                   expected_count=task_expected["current"]["count"],
                                   label=f"task {task_id} current")
        seen = _accuracy_record(metrics["seen"], expected_class_count=task_expected["seen"]["class_count"],
                                expected_count=task_expected["seen"]["count"],
                                label=f"task {task_id} seen")
        for key in ("count", "correct", "invalid"):
            if key == "invalid" and old[key] is None and old["structural_empty"]:
                left_value = 0
            else:
                left_value = old[key]
            if key == "correct" and old[key] is None and old["structural_empty"]:
                left_value = 0
            if key == "count" and old[key] is None and old["structural_empty"]:
                left_value = 0
            if key == "invalid" and (current[key] is None or seen[key] is None):
                raise ValueError(f"task {task_id}: invalid counts must be typed for nonempty scopes")
            if seen[key] != left_value + current[key]:
                raise ValueError(f"task {task_id}: old/current do not partition seen {key}")
        if any(type(item.get(k)) is not int for k in ("old_count", "current_count", "seen_count")) or item.get("old_count") != old["count"] or item.get("current_count") != current["count"] or item.get("seen_count") != seen["count"]:
            raise ValueError(f"task {task_id}: duplicated outer denominators disagree with metric rows")
        trajectory.append({"task_id": task_id, "old": old, "current": current, "seen": seen})
    return trajectory


def _require_hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} is not a lowercase SHA-256")
    return value


def validate_development_cohorts(cohorts: Mapping[str, Any], *, manifest_sha: str) -> dict[int, dict[int, dict[str, dict[str, int]]]]:
    """Validate frozen train-derived clean-development denominators (never test payload)."""
    if (cohorts.get("schema") != "clinc-v8-expected-clean-development-cohorts-v1"
            or cohorts.get("test_data_read") is not False
            or cohorts.get("model_loaded") is not False
            or cohorts.get("optimizer_steps") != 0
            or cohorts.get("training_manifest_sha256") != manifest_sha
            or cohorts.get("total_clean_rows") != 2997
            or cohorts.get("normalized_overlap_rows") != 3):
        raise ValueError("train-derived development cohort metadata is not frozen or differs from the training manifest")
    raw_seeds = cohorts.get("seeds")
    if not isinstance(raw_seeds, dict) or set(raw_seeds) != {str(s) for s in SEEDS}:
        raise ValueError("development cohort metadata must contain exact seeds 17, 29, 43")
    normalized = {}
    for seed in SEEDS:
        raw_tasks = raw_seeds[str(seed)]
        if not isinstance(raw_tasks, dict) or set(raw_tasks) != {str(t) for t in TASKS}:
            raise ValueError(f"seed {seed}: development cohort tasks must be exactly 0..9")
        normalized[seed] = {}
        previous_seen = None
        for task in TASKS:
            raw_scopes = raw_tasks[str(task)]
            if not isinstance(raw_scopes, dict) or set(raw_scopes) != {"old", "current", "seen"}:
                raise ValueError(f"seed {seed} task {task}: invalid development scope schema")
            scopes = {}
            for scope in ("old", "current", "seen"):
                row = raw_scopes[scope]
                if (not isinstance(row, dict) or type(row.get("count")) is not int
                        or type(row.get("class_count")) is not int or row["count"] < 0
                        or row["class_count"] < 0):
                    raise ValueError(f"seed {seed} task {task} {scope}: count/class_count must be nonnegative integers")
                scopes[scope] = {"count": row["count"], "class_count": row["class_count"]}
            if scopes["seen"]["count"] != scopes["old"]["count"] + scopes["current"]["count"]:
                raise ValueError(f"seed {seed} task {task}: cohort row counts do not partition")
            if scopes["seen"]["class_count"] != scopes["old"]["class_count"] + scopes["current"]["class_count"]:
                raise ValueError(f"seed {seed} task {task}: cohort class counts do not partition")
            if (scopes["current"]["class_count"] != 15
                    or scopes["seen"]["class_count"] != 15 * (task + 1)
                    or scopes["old"]["class_count"] != 15 * task):
                raise ValueError(f"seed {seed} task {task}: class scopes disagree with class-order index")
            if task and scopes["old"] != previous_seen:
                raise ValueError(f"seed {seed} task {task}: old rows/classes differ from prior seen scope")
            if task == 0 and scopes["old"] != {"count": 0, "class_count": 0}:
                raise ValueError(f"seed {seed}: task-0 old scope must be structurally empty")
            previous_seen = scopes["seen"]
            normalized[seed][task] = scopes
        if normalized[seed][9]["seen"]["count"] != 2997:
            raise ValueError(f"seed {seed}: clean task-9 development endpoint must contain 2,997 rows")
    return normalized


def derive_development_cohort_rows(class_orders: Mapping[str, Sequence[int]],
                                   clean_dev_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """Derive counts from the clean development rows returned by the frozen runner input reader."""
    counts = Counter()
    for row in clean_dev_rows:
        intent_id = row.get("intent_id")
        if type(intent_id) is not int or not 0 <= intent_id < 150:
            raise ValueError("clean development row has invalid intent_id")
        counts[intent_id] += 1
    derived = {}
    for seed in SEEDS:
        order = class_orders.get(str(seed))
        if not isinstance(order, (list, tuple)) or len(order) != 150 or set(order) != set(range(150)):
            raise ValueError(f"seed {seed}: training pack class order is invalid")
        tasks = {}
        for task in TASKS:
            old_ids = set(order[:task * 15])
            current_ids = set(order[task * 15:(task + 1) * 15])
            seen_ids = old_ids | current_ids
            row_count = lambda ids: sum(counts[class_id] for class_id in ids)
            tasks[str(task)] = {
                "old": {"class_count": len(old_ids), "count": row_count(old_ids)},
                "current": {"class_count": len(current_ids), "count": row_count(current_ids)},
                "seen": {"class_count": len(seen_ids), "count": row_count(seen_ids)},
            }
        derived[str(seed)] = tasks
    return derived


def validate_development_cohorts_against_pack(cohorts: Mapping[str, Any], *,
        manifest_sha: str, derived_rows: Mapping[str, Any]) -> dict[int, dict[int, dict[str, dict[str, int]]]]:
    normalized = validate_development_cohorts(cohorts, manifest_sha=manifest_sha)
    normalized_derived = {int(seed): {int(task): scopes for task, scopes in tasks.items()}
                          for seed, tasks in derived_rows.items()}
    if normalized != normalized_derived:
        raise ValueError("development cohort sidecar differs from counts re-derived from the verified training pack")
    return normalized


def derive_development_cohorts_from_pack(pack_path: Path, master: Mapping[str, Any]) -> dict[str, Any]:
    """Verify train/dev pack bytes with the frozen full runner and derive clean-dev denominators."""
    if sha256_file(pack_path) != master.get("training_manifest_sha256"):
        raise ValueError("training-pack manifest bytes differ from frozen master SHA")
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    runner_path = tools_dir / "run_clinc_cil_v8_full.py"
    spec = importlib.util.spec_from_file_location("_clinc_full_for_cohort_derivation", runner_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the source-locked full runner for cohort derivation")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    view, train_rows, dev_rows, clean_dev, partition, duplicates, _ = runner._read_inputs(pack_path, dict(master))
    if len(train_rows) != 15000 or len(dev_rows) != 3000 or len(clean_dev) != 2997 or duplicates != 3:
        raise ValueError("verified training pack has unexpected train/dev/clean/overlap counts")
    return {"schema": "clinc-v8-expected-clean-development-cohorts-v1",
        "model_loaded": False, "optimizer_steps": 0, "test_data_read": False,
        "total_clean_rows": len(clean_dev), "normalized_overlap_rows": duplicates,
        "training_manifest_sha256": master["training_manifest_sha256"],
        "seeds": derive_development_cohort_rows(view.class_orders, clean_dev)}


def validate_protocol(seal: Mapping[str, Any], master: Mapping[str, Any], *, master_sha: str) -> tuple[dict[tuple[int, str], dict], dict[int, dict]]:
    if seal.get("schema") != MAIN_SCHEMA or seal.get("complete") is not True:
        raise ValueError("main-six seal missing, incomplete, or wrong schema")
    if seal.get("test_data_included") is not False or seal.get("test_release_ready") is not True:
        raise ValueError("seal does not attest training/test isolation")
    if master.get("schema") != "clinc-cil-v8-full-master-candidate" or master.get("status") != "frozen" or master.get("seed_grid") != list(SEEDS) or master.get("arms") != list(ARMS) or master.get("tasks") != list(TASKS):
        raise ValueError("master config is not the exact frozen 3-seed × 2-arm × 10-task protocol")
    if seal.get("master_config_sha256") != master_sha:
        raise ValueError("seal/master bytes differ")
    manifest_sha = _require_hex64(master.get("training_manifest_sha256"), "master training manifest")
    if seal.get("training_manifest_sha256") != manifest_sha:
        raise ValueError("seal/master data manifest binding differs")
    source_hashes = seal.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("seal lacks source hash inventory")
    for k, v in source_hashes.items():
        _require_hex64(v, f"source hash {k}")
    source_lock = _require_hex64(seal.get("source_lock_sha256"), "source lock")
    if canonical_sha(source_hashes) != source_lock:
        raise ValueError("source-lock digest does not match source hash inventory")
    expected_grid = [{"seed": seed, "arm": arm} for seed in SEEDS for arm in ARMS]
    if seal.get("main_grid") != expected_grid:
        raise ValueError("seal seed/arm grid differs from the frozen protocol")
    if master.get("rank") != 64 or master.get("capacity") != 8 or master.get("levels") != 5 or master.get("periods") != [16, 64, 256, 1024, None]:
        raise ValueError("model/memory architecture differs from frozen protocol")
    if master.get("batch_size") != 8 or master.get("examples_per_task") != 1500 or master.get("max_steps_per_task") != 188:
        raise ValueError("wake schedule differs from frozen protocol")
    if master.get("objective_version") != "canonical_agreement_plus_thresholded_length_normalized_edit_v1":
        raise ValueError("Sleep objective differs from frozen protocol")
    if master.get("fixed_mix_fraction") != 0.5 or master.get("gkd_calls_per_event") != 20 or master.get("lti_calls_per_event") != 10 or master.get("inner_cycles_per_event") != 10:
        raise ValueError("fixed-mix fraction or per-event Sleep schedule differs from frozen protocol")
    if master.get("model_revision") is None or not isinstance(master.get("model_assets_sha256"), dict):
        raise ValueError("frozen model revision/assets missing")
    for k, v in master["model_assets_sha256"].items():
        _require_hex64(v, f"model asset {k}")

    foundations = seal.get("verified_foundations")
    if not isinstance(foundations, list) or len(foundations) != 3:
        raise ValueError("seal must contain exactly three shared foundation receipts")
    foundation_by_seed: dict[int, dict] = {}
    for item in foundations:
        seed = item.get("seed")
        if seed not in SEEDS or seed in foundation_by_seed or item.get("work_counted_once") is not True:
            raise ValueError("foundation set missing, duplicated, or not marked shared/count-once")
        _require_hex64(item.get("receipt_sha256"), f"foundation seed{seed} receipt hash")
        _require_hex64(item.get("checkpoint_sha256"), f"foundation seed{seed} checkpoint hash")
        foundation_by_seed[seed] = item
    if set(foundation_by_seed) != set(SEEDS):
        raise ValueError("foundation seeds must be exactly 17, 29, 43")
    shared_foundations = seal.get("shared_foundation_compute_counted_once")
    if not isinstance(shared_foundations, list) or len(shared_foundations) != 3:
        raise ValueError("seal lacks the three count-once foundation work records")
    shared_by_seed = {x.get("seed"): x for x in shared_foundations}
    if len(shared_by_seed) != 3 or set(shared_by_seed) != set(SEEDS):
        raise ValueError("shared foundation work rows have duplicate or missing seeds")
    for seed in SEEDS:
        meta, shared = foundation_by_seed[seed], shared_by_seed[seed]
        if shared.get("foundation_checkpoint_sha256") != meta.get("checkpoint_sha256") or shared.get("work") != meta.get("work"):
            raise ValueError(f"seed {seed}: count-once foundation work differs from verified foundation receipt")

    runs = seal.get("runs")
    if not isinstance(runs, list) or len(runs) != 6:
        raise ValueError("seal must have exactly six terminal run receipts")
    by_pair: dict[tuple[int, str], dict] = {}
    for run in runs:
        seed, arm = run.get("seed"), run.get("arm")
        key = (seed, arm)
        if seed not in SEEDS or arm not in ARMS or key in by_pair:
            raise ValueError("run receipts contain missing, duplicate, or unexpected seed/arm rows")
        if run.get("schema") != RUN_SCHEMA or run.get("status") != "complete" or run.get("tasks_completed") != 10:
            raise ValueError(f"{key}: failed, partial, or unsupported run receipt")
        if run.get("acquisition_qualified") is not True or run.get("test_data_included") is not False:
            raise ValueError(f"{key}: acquisition/test-isolation receipt is invalid")
        for k, expected in (("master_config_sha256", master_sha), ("training_manifest_sha256", manifest_sha), ("source_lock_sha256", source_lock)):
            if run.get(k) != expected:
                raise ValueError(f"{key}: inconsistent {k}")
        for k in ("final_checkpoint_sha256", "foundation_checkpoint_sha256", "foundation_receipt_sha256", "child_config_sha256"):
            _require_hex64(run.get(k), f"{key} {k}")
        if run.get("foundation_checkpoint_sha256") != foundation_by_seed[seed]["checkpoint_sha256"] or run.get("foundation_receipt_sha256") != foundation_by_seed[seed]["receipt_sha256"]:
            raise ValueError(f"{key}: arms do not share the verified same-seed foundation")
        if run.get("model_revision") not in (None, master["model_revision"]):
            raise ValueError(f"{key}: model revision differs from master")
        if run.get("model_assets_sha256") not in (None, master["model_assets_sha256"]):
            raise ValueError(f"{key}: model asset inventory differs from master")
        if run.get("inner_cycles_per_event") != master.get("inner_cycles_per_event") or run.get("objective_version") != master.get("objective_version"):
            raise ValueError(f"{key}: Sleep repetition/objective differs from master")
        task_items = run.get("task_metrics")
        if not isinstance(task_items, list) or [x.get("task_id") for x in task_items] != list(TASKS):
            raise ValueError(f"{key}: missing or unordered task metrics")
        if (type(run.get("wake_ticks")) is not int or run.get("wake_ticks") != 1880
                or type(run.get("physical_wake_updates")) is not int or run.get("physical_wake_updates") != 1880):
            raise ValueError(f"{key}: global wake cursor should include common 188-step foundation plus nine 188-step tasks")
        expected_sleep_events = 141 if arm == "HOPE_SLEEP_FIXEDMIX" else 0
        expected_sleep_steps = 4230 if arm == "HOPE_SLEEP_FIXEDMIX" else 0
        work = run.get("work", {})
        if (type(run.get("sleep_event_index")) is not int or run.get("sleep_event_index") != expected_sleep_events
                or type(run.get("physical_sleep_adamw_steps")) is not int
                or run.get("physical_sleep_adamw_steps") != expected_sleep_steps):
            raise ValueError(f"{key}: per-run Sleep event/AdamW totals differ from schedule")
        if (type(work.get("sleep_events")) is not int or work.get("sleep_events") != expected_sleep_events
                or type(work.get("optimizer_or_lifecycle_update_steps")) is not int
                or work.get("optimizer_or_lifecycle_update_steps") != 1692 + expected_sleep_steps):
            raise ValueError(f"{key}: work counters must equal continuation wake plus actual Sleep AdamW steps")
        by_pair[key] = run
    if set(by_pair) != {(s, a) for s in SEEDS for a in ARMS}:
        raise ValueError("the complete six-run grid is required")
    # Pairing: exact task-0 common foundation metrics and same initial mutable state.
    for seed in SEEDS:
        hope, sleep = by_pair[(seed, "HOPE")], by_pair[(seed, "HOPE_SLEEP_FIXEDMIX")]
        if hope.get("initial_mutable_sha256") != sleep.get("initial_mutable_sha256"):
            raise ValueError(f"seed {seed}: arms did not start from a byte-identical mutable state")
        if canonical_sha(hope["task_metrics"][0]) != canonical_sha(sleep["task_metrics"][0]):
            raise ValueError(f"seed {seed}: arms disagree on the shared foundation task-0 endpoint")
    verified_run_files = seal.get("verified_run_files")
    if not isinstance(verified_run_files, list) or len(verified_run_files) != 6:
        raise ValueError("seal must include six checkpoint/receipt byte bindings")
    file_pairs = [(x.get("seed"), x.get("arm")) for x in verified_run_files]
    if len(set(file_pairs)) != 6 or set(file_pairs) != set(by_pair):
        raise ValueError("verified run file bindings do not match the six unique run receipts")
    for seed, arm in file_pairs:
        binding, run = next(x for x in verified_run_files if x.get("seed") == seed and x.get("arm") == arm), by_pair[(seed, arm)]
        if binding.get("checkpoint_sha256") != run.get("final_checkpoint_sha256") or binding.get("run_receipt_sha256") != run.get("run_receipt_sha256"):
            raise ValueError(f"seed {seed} {arm}: seal file receipt does not bind to exact run/checkpoint hashes")
    return by_pair, foundation_by_seed


def _stat(values: Sequence[float | None]) -> dict[str, Any]:
    valid = [float(v) for v in values if v is not None]
    return {"n": len(valid), "mean": statistics.mean(valid) if valid else None,
            "sample_sd": statistics.stdev(valid) if len(valid) >= 2 else None,
            "per_seed": list(values)}


def build_dev_summary(seal: Mapping[str, Any], master: Mapping[str, Any], *, master_sha: str,
                      development_cohorts: Mapping[str, Any], development_cohorts_sha: str,
                      foundations: Mapping[int, Mapping] | None = None) -> dict[str, Any]:
    by_pair, foundation_meta = validate_protocol(seal, master, master_sha=master_sha)
    expected_rows = validate_development_cohorts(development_cohorts,
        manifest_sha=master["training_manifest_sha256"])
    _require_hex64(development_cohorts_sha, "development cohorts metadata sha256")
    # Actual foundation receipt data (not just its hash) is needed to verify each shared task-0 metric.
    foundation_metrics: dict[int, list[dict]] = {}
    foundation_cost_rows: list[Mapping] = []
    for seed in SEEDS:
        rec = (foundations or {}).get(seed)
        if rec is None:
            # Seal rows include the exact foundation work but not its task-0 metric; in-memory/test callers
            # must pass actual foundation receipts, while production CLI always does.
            raise ValueError(f"actual foundation receipt for seed {seed} is required")
        if rec.get("schema") != FOUNDATION_SCHEMA or rec.get("status") != "complete" or rec.get("seed") != seed:
            raise ValueError(f"seed {seed}: invalid foundation receipt")
        if rec.get("test_data_included") is not False or rec.get("acquisition_qualified") is not True:
            raise ValueError(f"seed {seed}: foundation receipt lacks qualification/isolation")
        if rec.get("master_config_sha256") != master_sha or rec.get("training_manifest_sha256") != master["training_manifest_sha256"]:
            raise ValueError(f"seed {seed}: foundation master/data binding differs")
        if rec.get("source_lock_sha256") != seal.get("source_lock_sha256"):
            raise ValueError(f"seed {seed}: foundation source binding differs")
        if rec.get("model_revision") != master.get("model_revision") or rec.get("model_assets_sha256") != master.get("model_assets_sha256"):
            raise ValueError(f"seed {seed}: foundation model revision/asset inventory differs from master")
        if rec.get("tasks_completed") != 1 or rec.get("wake_ticks") != 188:
            raise ValueError(f"seed {seed}: foundation is not the task-0-only 188-step common start")
        if sorted(int(v) for v in rec.get("pending_clocks", {}).values()) != [12, 60, 188, 188]:
            raise ValueError(f"seed {seed}: foundation terminal clock is invalid")
        items = rec.get("task_metrics")
        if not isinstance(items, list) or len(items) != 1 or items[0].get("task_id") != 0:
            raise ValueError(f"seed {seed}: foundation contains non-task0 metrics")
        fresult = items[0].get("result")
        fmetrics = fresult.get("metrics") if isinstance(fresult, dict) else None
        if not isinstance(fmetrics, dict) or not isinstance(fmetrics.get("current"), dict):
            raise ValueError(f"seed {seed}: foundation current-intent metrics are missing")
        foundation_current = _accuracy_record(fmetrics["current"],
            expected_class_count=expected_rows[seed][0]["current"]["class_count"],
            expected_count=expected_rows[seed][0]["current"]["count"],
            label=f"seed {seed} foundation task-0 current")
        gate = master.get("task0_clean_accuracy_gate")
        if not _is_number(gate) or foundation_current["accuracy"] is None or foundation_current["accuracy"] < float(gate):
            raise ValueError(f"seed {seed}: actual foundation task-0 accuracy is below the configured acquisition gate")
        foundation_metrics[seed] = items
        foundation_cost_rows.append(rec)
        seal_foundation = foundation_meta[seed]
        if rec.get("final_checkpoint_sha256") != seal_foundation.get("checkpoint_sha256"):
            raise ValueError(f"seed {seed}: foundation checkpoint differs from seal")
        if rec.get("work") != seal_foundation.get("work"):
            raise ValueError(f"seed {seed}: foundation work differs from count-once seal receipt")
        for arm in ARMS:
            if canonical_sha(by_pair[(seed, arm)]["task_metrics"][0]) != canonical_sha(items[0]):
                raise ValueError(f"seed {seed}: run {arm} task0 metrics differ from shared foundation receipt")

    trajectories: dict[tuple[int, str], list[dict]] = {}
    for pair, run in by_pair.items():
        trajectories[pair] = extract_trajectory(run, expected_rows=expected_rows[pair[0]])
    per_seed_rows = []
    for seed in SEEDS:
        for arm in ARMS:
            trajectory = trajectories[(seed, arm)]
            per_seed_rows.append({"seed": seed, "arm": arm, "tasks": trajectory})

    task_stats = []
    paired_differences = []
    for task_id in TASKS:
        by_arm: dict[str, Any] = {}
        for arm in ARMS:
            by_arm[arm] = {}
            for scope in ("old", "current", "seen"):
                for metric_name in ("accuracy", "macro_accuracy"):
                    vals = [trajectories[(seed, arm)][task_id][scope][metric_name] for seed in SEEDS]
                    by_arm[arm].setdefault(scope, {})[metric_name] = _stat(vals)
        task_stats.append({"task_id": task_id, "by_arm": by_arm})
        diff = {"task_id": task_id}
        for scope in ("old", "current", "seen"):
            vals = []
            for seed in SEEDS:
                left = trajectories[(seed, "HOPE_SLEEP_FIXEDMIX")][task_id][scope]["accuracy"]
                right = trajectories[(seed, "HOPE")][task_id][scope]["accuracy"]
                vals.append(None if left is None or right is None else left - right)
            diff[scope] = _stat(vals)
        paired_differences.append(diff)

    continuation_by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = [by_pair[(seed, arm)] for seed in SEEDS]
        continuation_by_arm[arm] = {
            "run_count": len(rows),
            "physical_wake_updates_continuation": sum(int(r["physical_wake_updates"]) - 188 for r in rows),
            "physical_wake_updates_including_foundation_per_run_sum_not_unique": sum(int(r["physical_wake_updates"]) for r in rows),
            "sleep_events": sum(int(r["sleep_event_index"]) for r in rows),
            "physical_sleep_adamw_steps": sum(int(r["physical_sleep_adamw_steps"]) for r in rows),
            "work": _sum_work([r.get("work", {}) for r in rows]),
            "per_seed": [{"seed": r["seed"], "physical_wake_updates_including_foundation": r["physical_wake_updates"],
                          "continuation_wake_updates": r["physical_wake_updates"] - 188,
                          "sleep_events": r["sleep_event_index"],
                          "physical_sleep_adamw_steps": r["physical_sleep_adamw_steps"],
                          "update_counter": r.get("work", {}).get("optimizer_or_lifecycle_update_steps"),
                          "active_seconds": r.get("active_wall_seconds")} for r in rows],
        }
    if continuation_by_arm["HOPE"]["sleep_events"] != 0 or continuation_by_arm["HOPE"]["physical_sleep_adamw_steps"] != 0:
        raise ValueError("HOPE control must have zero actual Sleep events and Sleep AdamW steps")
    if continuation_by_arm["HOPE_SLEEP_FIXEDMIX"]["sleep_events"] != 423 or continuation_by_arm["HOPE_SLEEP_FIXEDMIX"]["physical_sleep_adamw_steps"] != 12690:
        raise ValueError("fixed-mix three-run Sleep totals differ from the frozen 423-event/12,690-AdamW schedule")

    foundation_work = _sum_work([r.get("work", {}) for r in foundation_cost_rows])
    unique_grid_work = _sum_work([*([r.get("work", {}) for r in foundation_cost_rows]),
                                  *[r.get("work", {}) for r in by_pair.values()]])
    actual_exposure = []
    for seed in SEEDS:
        for arm in ARMS:
            r = by_pair[(seed, arm)]
            event_count, sleep_adam = r.get("sleep_event_index"), r.get("physical_sleep_adamw_steps")
            declared = r.get("data_exposure", {}).get("sleep_text_replay")
            actual = type(event_count) is int and event_count > 0 and type(sleep_adam) is int and sleep_adam > 0
            actual_exposure.append({"seed": seed, "arm": arm, "sleep_events": event_count,
                "physical_sleep_adamw_steps": sleep_adam, "receipt_sleep_text_replay_flag": declared,
                "actual_text_replay_exposure_from_counters": actual,
                "receipt_flag_matches_actual": declared == actual})

    final_by_arm = {}
    for arm in ARMS:
        final_by_arm[arm] = {}
        for scope in ("old", "current", "seen"):
            per_seed = [trajectories[(seed, arm)][9][scope] for seed in SEEDS]
            final_by_arm[arm][scope] = {
                "accuracy": _stat([r["accuracy"] for r in per_seed]),
                "macro_accuracy": _stat([r["macro_accuracy"] for r in per_seed]),
                "per_seed": [{"seed": seed, **r} for seed, r in zip(SEEDS, per_seed)],
            }
    final_paired = {scope: _stat([
        trajectories[(seed, "HOPE_SLEEP_FIXEDMIX")][9][scope]["accuracy"] - trajectories[(seed, "HOPE")][9][scope]["accuracy"]
        for seed in SEEDS]) for scope in ("old", "current", "seen")}
    return {
        "schema": "clinc-v8-main-six-aggregate-v1",
        "status": "complete_training_test_pending",
        "claim_scope": "three class-order seeds paired across two arms, ten-task clean development endpoints; not paper-faithful HOPE, and not three independent random initializations",
        "test_status": "test_pending",
        "test_payload_read": False,
        "protocol": {"master_config_sha256": master_sha,
            "training_manifest_sha256": master["training_manifest_sha256"],
            "development_cohorts_sha256": development_cohorts_sha,
            "development_cohorts_schema": development_cohorts.get("schema"),
            "clean_development_rows_per_seed": 2997,
            "normalized_train_development_overlap_rows": 3,
            "model_revision": master["model_revision"], "model_assets_sha256": master["model_assets_sha256"],
            "source_lock_sha256": seal["source_lock_sha256"], "source_hashes": seal["source_hashes"],
            "seed_order": list(SEEDS), "arms": list(ARMS), "tasks": list(TASKS),
            "rank": master["rank"], "capacity": master["capacity"], "levels": master["levels"],
            "periods": master["periods"], "initialization_seed": master["initialization_seed"],
            "examples_per_task": master["examples_per_task"], "batch_size": master["batch_size"],
            "objective_version": master["objective_version"], "inner_cycles_per_event": master["inner_cycles_per_event"]},
        "shared_foundation": {"count": 3, "work_counted_once": True,
            "task0_updates_per_seed": 188, "work": foundation_work,
            "per_seed": [{"seed": s, "checkpoint_sha256": foundation_meta[s]["checkpoint_sha256"],
                          "receipt_sha256": foundation_meta[s]["receipt_sha256"],
                          "task0_clean_accuracy": trajectories[(s, "HOPE")][0]["current"]["accuracy"],
                          "work": foundation_cost_rows[i].get("work")} for i, s in enumerate(SEEDS)]},
        "per_seed_task_endpoints": per_seed_rows,
        "development_trajectory_stats": task_stats,
        "paired_descriptive_differences_fixedmix_minus_hope": paired_differences,
        "final_development_endpoints": final_by_arm,
        "final_development_paired_differences_fixedmix_minus_hope": final_paired,
        "costs": {"unique_grid_work_foundation_once_plus_all_six_continuations": unique_grid_work,
            "shared_foundation_work_counted_once": foundation_work,
            "continuation_work_by_arm": continuation_by_arm,
            "actual_sleep_exposure_audit": actual_exposure,
            "expected_observed_sleep_schedule": {"fixedmix_runs": 3,
                "fixedmix_events_per_run": 141, "fixedmix_events_total": 423,
                "event_formula_per_run": "(floor(1880/16)-floor(188/16))+(floor(1880/64)-floor(188/64))+(floor(1880/256)-floor(188/256))+(floor(1880/1024)-floor(188/1024))=141",
                "fixedmix_physical_adamw_per_run": 4230, "fixedmix_physical_adamw_total": 12690, "hope_events_total": 0,
                "hope_physical_adamw_total": 0,
                "continuation_wake_updates_per_run": 1692,
                "global_wake_ticks_per_run_including_foundation": 1880}},
        "method_and_scope_notes": [
            "HOPE and HOPE_SLEEP_FIXEDMIX use the same per-seed task-0 foundation; its cost is shared and counted once.",
            "Each continuation has 1,692 wake updates after the shared 188-update foundation; receipt global wake cursor is 1,880.",
            "HOPE_SLEEP_FIXEDMIX is a controlled fixed-mix local adaptation, not native/paper-faithful HOPE Sleep. Its additional class-balanced metadata replay pool draws four examples per class from the 15,000-row train pack; the continuation primary schedule contains 13,500 rows after the shared task-0 foundation. Replay exposure is inferred from actual events and optimizer counters, not the receipt availability flag.",
            "The two arms share their qualified foundation and primary new-task wake budget. The fixed-mix arm additionally incurs Sleep replay and optimizer work and applies a 0.5 branch mix; this is a local mechanism ablation, not a paper-faithful HOPE reproduction or a compute-matched causal estimate.",
            "The fixed-mix receipt may set sleep_text_replay=true for both arms as pool availability metadata; HOPE has zero Sleep events and zero Sleep AdamW steps, so actual Sleep text exposure is zero for HOPE.",
            "All three seeds share the fixed model initialization seed; they vary class order and are paired runs, not independent initializations. Report means/sample SD descriptively only; no inferential test is warranted.",
            "Development endpoint groups are old/current/seen at each task boundary. At task 9 these represent 135/15/150 intents; duplicated train-development rows were excluded per the frozen protocol, so example denominators can be slightly below nominal counts.",
            "Only the shared task-0 foundation has the configured >=0.5 acquisition gate; tasks 1-9 have no learning gate. Final earlier-class accuracy therefore combines forgetting with any earlier class that may not have been learned at its task boundary, and is not a pure retention rate. Current-task development accuracy provides context but does not certify earlier acquisition.",
            "The official test evaluation, when run, scores only the six final checkpoints globally. It does not provide initial test scores or an official-test per-class forgetting curve.",
            "The test split is held out from this training workflow but shares the benchmark's intent inventory; it is not a wholly unseen domain/task. Final test evaluation, when available, uses all 4,500 rows globally without task masks or checkpoint selection.",
        ],
    }


def _sum_work(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for key in WORK_FIELDS:
        values = [row.get(key) for row in rows]
        if not rows or any(not _is_number(v) for v in values):
            result[key] = None
        else:
            result[key] = sum(values)
    return result


def verify_filesystem(seal: Mapping[str, Any], *, seal_path: Path, master_path: Path,
                      foundation_root: Path, run_root: Path,
                      source_files: Mapping[str, Path] | None = None) -> tuple[dict[int, dict], dict[str, Any]]:
    master = read_json(master_path)
    master_sha = sha256_file(master_path)
    by_pair, foundation_by_seed = validate_protocol(seal, master, master_sha=master_sha)
    verified_source_hashes = seal.get("source_hashes", {})
    if source_files is not None:
        if set(source_files) != set(verified_source_hashes):
            raise ValueError("source file list does not exactly match sealed source inventory")
        actual_hashes = {name: sha256_file(path) for name, path in sorted(source_files.items())}
        if actual_hashes != verified_source_hashes:
            raise ValueError("current training-source bytes differ from the frozen source inventory")
    actual_foundations: dict[int, dict] = {}
    for seed in SEEDS:
        rpath = foundation_root / f"seed{seed}" / "foundation-receipt.json"
        receipt = read_json(rpath)
        actual_sha = sha256_file(rpath)
        expected = foundation_by_seed[seed]
        if actual_sha != expected["receipt_sha256"]:
            raise ValueError(f"seed {seed}: foundation receipt bytes differ from seal")
        checkpoint_rel = receipt.get("final_checkpoint")
        cp = (rpath.parent / checkpoint_rel).resolve()
        if rpath.parent.resolve() not in cp.parents or not cp.is_file():
            raise ValueError(f"seed {seed}: unsafe/missing foundation checkpoint")
        if sha256_file(cp) != expected["checkpoint_sha256"] or receipt.get("final_checkpoint_sha256") != expected["checkpoint_sha256"]:
            raise ValueError(f"seed {seed}: foundation checkpoint bytes differ from seal")
        actual_foundations[seed] = receipt

    verified_run_files = {(item.get("seed"), item.get("arm")): item for item in seal["verified_run_files"]}
    for (seed, arm), sealed in by_pair.items():
        run_dir = run_root / f"seed{seed}" / arm
        rpath = run_dir / "run-receipt.json"
        receipt = read_json(rpath)
        binding = verified_run_files[(seed, arm)]
        if sha256_file(rpath) != binding.get("run_receipt_sha256") or sealed.get("run_receipt_sha256") != binding.get("run_receipt_sha256"):
            raise ValueError(f"seed {seed} {arm}: run receipt bytes differ from seal")
        if {k: v for k, v in sealed.items() if k not in ("run_receipt_path", "run_receipt_sha256")} != receipt:
            raise ValueError(f"seed {seed} {arm}: sealed receipt body differs from actual receipt")
        cp = (run_dir / receipt.get("final_checkpoint", "")).resolve()
        if run_dir.resolve() not in cp.parents or not cp.is_file():
            raise ValueError(f"seed {seed} {arm}: unsafe/missing final checkpoint")
        if sha256_file(cp) != binding.get("checkpoint_sha256") or receipt.get("final_checkpoint_sha256") != binding.get("checkpoint_sha256"):
            raise ValueError(f"seed {seed} {arm}: final checkpoint bytes differ from seal")
    return actual_foundations, {"seal_sha256": sha256_file(seal_path), "master_config_sha256": master_sha}


def validate_test_result(result: Mapping[str, Any], *, result_sha: str, seal_sha: str,
                         master_sha: str, source_hashes: Mapping[str, str],
                         evaluator_sha: str, contract: Mapping[str, Any], contract_sha: str,
                         expected_checkpoints: Mapping[tuple[int, str], Mapping[str, str]]) -> dict[str, Any]:
    if result.get("schema") != TEST_SCHEMA or result.get("status") != "complete" or result.get("test_data_read") is not True:
        raise ValueError("final test result is missing, incomplete, or unsupported")
    if result.get("development_only") is not False or result.get("primary_rows_used") != 4500:
        raise ValueError("final test result must score all 4,500 rows")
    if result.get("candidate_constrained") is not False or result.get("task_mask_used") is not False or result.get("checkpoint_selection_after_test") is not False:
        raise ValueError("test result used candidate restriction, task mask, or checkpoint selection")
    if result.get("optimizer_steps") != 0 or result.get("backward_calls") != 0:
        raise ValueError("test evaluation must be read-only")
    if result.get("evaluator_source_sha256") != evaluator_sha or result.get("evaluation_contract_sha256") != contract_sha:
        raise ValueError("test result evaluator/contract source binding mismatch")
    if canonical_sha(result.get("evaluation_contract")) != canonical_sha(contract):
        raise ValueError("embedded test contract differs from provided frozen contract")
    if result.get("training_source_hashes") != dict(source_hashes):
        raise ValueError("test result source hashes differ from main seal")
    gate = result.get("release_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("seal_sha256") != seal_sha or gate.get("master_config_sha256") != master_sha:
        raise ValueError("test result release gate does not bind to the completed main seal")
    if gate.get("source_hashes") != dict(source_hashes) or gate.get("evaluation_contract_sha256") != contract_sha:
        raise ValueError("test release gate source/contract hash mismatch")
    rows = result.get("runs")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("test result must contain exactly six final checkpoint rows")
    by_pair = {}
    for row in rows:
        key = (row.get("seed"), row.get("arm"))
        if key not in {(s, a) for s in SEEDS for a in ARMS} or key in by_pair:
            raise ValueError("test result has duplicate or unexpected seed/arm")
        expected = expected_checkpoints.get(key)
        if expected is None or row.get("checkpoint_sha256") != expected.get("checkpoint_sha256") or row.get("run_receipt_sha256") != expected.get("run_receipt_sha256"):
            raise ValueError(f"test row {key} does not match the sealed final checkpoint/receipt")
        p = row.get("primary")
        if not isinstance(p, dict):
            raise ValueError("each final test endpoint must be a metric object")
        count = p.get("count", p.get("total"))
        per_class = p.get("per_class")
        expected_per_class_keys = {str(i) for i in range(150)}
        per_class_keys = set(per_class) if isinstance(per_class, dict) else set()
        class_count = p.get("class_count", len(per_class_keys))
        if (type(count) is not int or count != 4500 or type(class_count) is not int or class_count != 150
                or per_class_keys != expected_per_class_keys or p.get("task_mask_used") not in (None, False)):
            raise ValueError("each final test endpoint must cover all 4,500 rows and 150 classes without a task mask")
        if type(p.get("invalid_predictions")) is not int or not 0 <= p["invalid_predictions"] <= count:
            raise ValueError("test invalid-prediction count must be a typed in-range integer")
        if type(p.get("correct")) is not int or not 0 <= p["correct"] <= count - p["invalid_predictions"]:
            raise ValueError("test correct count must be typed and no larger than non-invalid predictions")
        per_class_correct = 0
        per_class_total = 0
        for class_id in range(150):
            item = per_class[str(class_id)]
            if (not isinstance(item, dict) or type(item.get("total")) is not int
                    or type(item.get("correct")) is not int or item["total"] != 30
                    or not 0 <= item["correct"] <= item["total"]):
                raise ValueError("each of the 150 test classes must have exactly 30 typed rows and in-range correct count")
            per_class_total += item["total"]
            per_class_correct += item["correct"]
        if per_class_total != 4500 or per_class_correct != p["correct"]:
            raise ValueError("test per-class counts do not sum to the global endpoint")
        if not _is_number(p.get("accuracy")) or not 0 <= p["accuracy"] <= 1:
            raise ValueError("invalid final test accuracy")
        correct = p.get("correct")
        if abs(correct / 4500 - float(p["accuracy"])) > 1e-12:
            raise ValueError("test accuracy numerator/count mismatch")
        by_pair[key] = {**p, "count": count, "class_count": class_count}
    if set(by_pair) != {(s, a) for s in SEEDS for a in ARMS}:
        raise ValueError("incomplete test checkpoint grid")
    arms = {}
    for arm in ARMS:
        vals = [float(by_pair[(seed, arm)]["accuracy"]) for seed in SEEDS]
        invalid = [by_pair[(seed, arm)].get("invalid_predictions") for seed in SEEDS]
        arms[arm] = {"accuracy": _stat(vals), "invalid_predictions": _stat([float(v) if type(v) is int else None for v in invalid])}
    paired = _stat([float(by_pair[(s, ARMS[1])]["accuracy"]) - float(by_pair[(s, ARMS[0])]["accuracy"]) for s in SEEDS])
    return {"status": "complete", "rows_per_checkpoint": 4500, "task_mask_used": False,
        "checkpoint_selection_after_test": False, "optimizer_steps": 0, "backward_calls": 0,
        "per_seed": [{"seed": s, "hope_accuracy": by_pair[(s, "HOPE")]["accuracy"],
                      "fixedmix_accuracy": by_pair[(s, "HOPE_SLEEP_FIXEDMIX")]["accuracy"],
                      "fixedmix_minus_hope": by_pair[(s, "HOPE_SLEEP_FIXEDMIX")]["accuracy"] - by_pair[(s, "HOPE")]["accuracy"]} for s in SEEDS],
        "by_arm": arms, "paired_fixedmix_minus_hope": paired,
        "input_artifact_sha256": result_sha, "evaluator_source_sha256": evaluator_sha,
        "evaluation_contract_sha256": contract_sha}


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# CLINC V8 main-six aggregate", "",
        f"Status: **{summary['status']}**. Test status: **{summary['test_status']}**.", "",
        "This is a paired, three-class-order-seed comparison using one fixed Llama initialization. The three seeds are not independent random initializations; statistics are descriptive means and sample SDs only.", "",
        "## Development task-end accuracy", "",
        "Rows show mean accuracy ± sample SD across the three paired class-order seeds. `old`, `current`, and `seen` are intent-scope metrics; denominators can vary slightly after the frozen duplicate-exclusion rule.", "",
        "| Task | Arm | Earlier classes | Current classes | All seen |", "|---:|---|---:|---:|---:|"]
    for row in summary["development_trajectory_stats"]:
        task = row["task_id"]
        for arm in ARMS:
            scopes = row["by_arm"][arm]
            fmt = lambda x: "n/a" if x["mean"] is None else f"{100*x['mean']:.2f}% ± {100*(x['sample_sd'] or 0):.2f}%"
            lines.append(f"| {task} | {arm} | {fmt(scopes['old']['accuracy'])} | {fmt(scopes['current']['accuracy'])} | {fmt(scopes['seen']['accuracy'])} |")
    lines += ["", "## Final development endpoint (task 9)", "", "Earlier-class accuracy is descriptive and combines forgetting with any earlier classes not learned at their own task boundary; only task 0 had an acquisition gate.", "", "| Arm | Scope | Mean accuracy ± sample SD | Per-seed accuracy |", "|---|---|---:|---|"]
    for arm in ARMS:
        for scope in ("old", "current", "seen"):
            rec = summary["final_development_endpoints"][arm][scope]["accuracy"]
            vals = ", ".join("n/a" if v is None else f"{100*v:.2f}%" for v in rec["per_seed"])
            value = "n/a" if rec["mean"] is None else f"{100*rec['mean']:.2f}% ± {100*(rec['sample_sd'] or 0):.2f}%"
            lines.append(f"| {arm} | {scope} | {value} | {vals} |")
    lines += ["", "### Paired fixed-mix minus HOPE differences", "", "| Scope | Mean percentage-point difference ± sample SD | Per-seed differences (pp) |", "|---|---:|---|"]
    for scope, rec in summary["final_development_paired_differences_fixedmix_minus_hope"].items():
        vals = ", ".join("n/a" if v is None else f"{100*v:+.2f}" for v in rec["per_seed"])
        avg = "n/a" if rec["mean"] is None else f"{100*rec['mean']:+.2f} ± {100*(rec['sample_sd'] or 0):.2f}"
        lines.append(f"| {scope} | {avg} | {vals} |")
    lines += ["", "## Work accounting", "", "The common task-0 foundations are shared across both arms and counted once. Per-run global wake ticks include this shared 188-update foundation; continuation-only wake updates are shown separately.", "",
        f"- Shared foundation: {summary['shared_foundation']['work']['full_backbone_forward_calls']} forwards and {summary['shared_foundation']['work']['optimizer_or_lifecycle_update_steps']} update-counter steps, counted once across seeds."]
    for arm in ARMS:
        c = summary["costs"]["continuation_work_by_arm"][arm]
        lines.append(f"- {arm}: {c['physical_wake_updates_continuation']} continuation wake updates; {c['sleep_events']} Sleep events; {c['physical_sleep_adamw_steps']} physical Sleep AdamW steps; {c['work']['full_backbone_forward_calls']} full-backbone forwards; {c['work']['active_wall_seconds']:.1f} seconds summed across its three continuations.")
    lines += ["", "Sleep exposure is derived from actual event and physical AdamW counters. The receipt's `sleep_text_replay` flag describes pool availability; it is not treated as actual exposure for HOPE, which has zero Sleep events and zero Sleep AdamW steps.", "",
        "## Interpretation and limitations", ""]
    lines.extend(f"- {note}" for note in summary["method_and_scope_notes"])
    if summary.get("final_test"):
        t = summary["final_test"]
        lines += ["", "## Final held-out test (all rows, no task mask)", "",
            "| Arm | Mean accuracy ± sample SD over three paired class-order seeds |", "|---|---:|"]
        for arm in ARMS:
            x = t["by_arm"][arm]["accuracy"]
            lines.append(f"| {arm} | {100*x['mean']:.2f}% ± {100*(x['sample_sd'] or 0):.2f}% |")
        p = t["paired_fixedmix_minus_hope"]
        lines.append(f"\nPaired fixed-mix minus HOPE: {100*p['mean']:+.2f} ± {100*(p['sample_sd'] or 0):.2f} percentage points (descriptive, n=3).")
    return "\n".join(lines) + "\n"


def render_plots(summary: Mapping[str, Any], outdir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    colors = {"HOPE": "#285f8f", "HOPE_SLEEP_FIXEDMIX": "#c45a2a"}
    for ax, scope, title in zip(axes, ("old", "current", "seen"), ("Earlier intents", "Current intents", "All seen intents")):
        for arm in ARMS:
            rows = [x["by_arm"][arm][scope]["accuracy"] for x in summary["development_trajectory_stats"]]
            means = [float(x["mean"]) if x["mean"] is not None else math.nan for x in rows]
            sds = [float(x["sample_sd"] or 0) if x["mean"] is not None else 0 for x in rows]
            ax.errorbar(TASKS, means, yerr=sds, marker="o", linewidth=1.7, capsize=3,
                        label=arm, color=colors[arm])
        ax.set_title(title)
        ax.set_xlabel("Completed task")
        ax.set_ylim(0, 1)
        ax.set_xticks(TASKS)
        ax.grid(True, alpha=.25)
    axes[0].set_ylabel("Accuracy (mean ± sample SD, n=3)")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("CLINC V8 clean development task-end accuracy")
    fig.tight_layout()
    paths = []
    for suffix in ("png", "svg"):
        path = outdir / f"clinc-v8-main-six-development.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def _source_file_map(project_root: Path, tools_dir: Path) -> dict[str, Path]:
    result = {
        "candidate_runner": tools_dir / "run_clinc_cil_v8_full.py",
        "r2_runner": tools_dir / "run_clinc_cil_v8_r2.py",
        "common": tools_dir / "clinc_cil_common_v8.py",
        "lifecycle": tools_dir / "clinc_lifecycle_v8.py",
        "controlled_lifecycle": tools_dir / "clinc_controlled_routing_v8.py",
        "controlled_probe": tools_dir / "run_clinc_controlled_probe_v8.py",
        "tail_runtime": tools_dir / "clinc_tail_runtime_v8.py",
    }
    result.update({f"public_{name}": project_root / "src" / "nested_sleep" / f"{name}.py"
                   for name in ("lm", "experts", "accounting", "native")})
    return result


def verify_main_inputs(*, seal_path: Path, master_path: Path, foundation_root: Path,
                       run_root: Path, project_root: Path) -> tuple[dict, dict, dict[int, dict], dict[str, Any]]:
    seal, master = read_json(seal_path), read_json(master_path)
    source_paths = _source_file_map(project_root.resolve(), Path(__file__).resolve().parent)
    if set(source_paths) != set(seal.get("source_hashes", {})):
        raise ValueError("current source inventory does not match the seal's named source inventory")
    actual_sources = {name: sha256_file(path) for name, path in sorted(source_paths.items())}
    if actual_sources != seal.get("source_hashes"):
        raise ValueError("current source bytes differ from the exact frozen full-run source lock")
    actual_foundations, artifact_bindings = verify_filesystem(seal,
        seal_path=seal_path, master_path=master_path,
        foundation_root=foundation_root, run_root=run_root,
        source_files=source_paths)
    # Check current repo full runner has same expected source hash as seal; analyzer must not impersonate it.
    artifact_bindings["aggregate_source_sha256"] = sha256_file(Path(__file__).resolve())
    artifact_bindings["master_config_file"] = master_path.name
    artifact_bindings["seal_file"] = seal_path.name
    return seal, master, actual_foundations, artifact_bindings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("seal", "master-config", "foundation-root", "run-root", "project-root", "training-pack", "development-cohorts", "output-dir"):
        p.add_argument("--" + arg, required=True, type=Path)
    p.add_argument("--test-evaluation", type=Path, default=None,
                   help="optional completed evaluation JSON only; never a test-data file")
    p.add_argument("--test-contract", type=Path, default=None)
    p.add_argument("--test-evaluator", type=Path, default=None)
    args = p.parse_args()
    outdir = args.output_dir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty aggregate directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    seal, master, foundations, bindings = verify_main_inputs(
        seal_path=args.seal.resolve(), master_path=args.master_config.resolve(),
        foundation_root=args.foundation_root.resolve(), run_root=args.run_root.resolve(),
        project_root=args.project_root.resolve())
    cohort_path = args.development_cohorts.resolve()
    cohorts = read_json(cohort_path)
    cohort_sha = sha256_file(cohort_path)
    pack_path = args.training_pack.resolve()
    derived = derive_development_cohorts_from_pack(pack_path, master)
    validate_development_cohorts_against_pack(cohorts,
        manifest_sha=master["training_manifest_sha256"], derived_rows=derived["seeds"])
    summary = build_dev_summary(seal, master,
        master_sha=sha256_file(args.master_config.resolve()),
        development_cohorts=cohorts, development_cohorts_sha=cohort_sha, foundations=foundations)
    summary["provenance"] = bindings
    summary["source_seal_sha256"] = bindings["seal_sha256"]
    summary["provenance"]["training_pack_sha256"] = sha256_file(pack_path)
    summary["provenance"]["development_cohorts_sidecar_sha256"] = cohort_sha
    summary["provenance"]["cohort_counts_rederived_by"] = "run_clinc_cil_v8_full.py::_read_inputs"

    if args.test_evaluation:
        if not args.test_contract or not args.test_evaluator:
            raise ValueError("test result requires --test-contract and --test-evaluator source file")
        result_path = args.test_evaluation.resolve()
        contract_path = args.test_contract.resolve()
        evaluator_path = args.test_evaluator.resolve()
        test_result = read_json(result_path)
        contract = read_json(contract_path)
        evaluator_sha = sha256_file(evaluator_path)
        master_sha = sha256_file(args.master_config.resolve())
        if contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != "frozen":
            raise ValueError("test evaluation contract is not frozen")
        if contract.get("training_master_sha256") != master_sha or contract.get("source_lock_sha256") != seal.get("source_lock_sha256") or contract.get("evaluator_sha256") != evaluator_sha:
            raise ValueError("evaluation contract does not bind to the final source/master/seal")
        summary["final_test"] = validate_test_result(test_result,
            result_sha=sha256_file(result_path), seal_sha=bindings["seal_sha256"],
            master_sha=master_sha, source_hashes=seal["source_hashes"],
            evaluator_sha=evaluator_sha, contract=contract, contract_sha=sha256_file(contract_path),
            expected_checkpoints={(run["seed"], run["arm"]): {
                "checkpoint_sha256": run["final_checkpoint_sha256"],
                "run_receipt_sha256": run["run_receipt_sha256"]} for run in seal["runs"]})
        summary["test_status"] = "complete_all4500_global_unmasked"
        summary["status"] = "complete_training_and_test"
    json_path = outdir / "clinc-v8-main-six-summary.json"
    md_path = outdir / "clinc-v8-main-six-summary.md"
    plot_files = render_plots(summary, outdir)
    summary["plots"] = plot_files
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "test_status": summary["test_status"],
                      "out_files": [json_path.name, md_path.name, *plot_files]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
