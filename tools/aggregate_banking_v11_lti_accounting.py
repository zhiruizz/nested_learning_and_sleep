#!/usr/bin/env python3
"""Offline accounting for the frozen 3-seed x 2-arm Banking V11 ablation.

Reads only frozen configs, baseline/V11 JSON receipts and protocols, and
SHA-bound checkpoint bytes. It never imports torch, loads model state, opens
dataset rows, or reads official-test payloads. Missing work counters remain
unknown (null), not zero.

Input manifest schema::

    {"schema":"banking-v11-offline-accounting-input-v1",
     "project_root":"...public repository...",
     "task_root":"...task workspace...",
     "v11_config_path":"...",
     "v10_config_path":"...",
     "baseline_seal_path":"...",
     "baseline_runs_root":"...",
     "v11_runs_root":"...",
     "runs":[{"seed":17,"arm":"LTI_NO_REWARD","run_dir":"..."}, ...]}

Exactly six V11 run directories are required. V10 task-1 shared-development
metrics are reported as a separate comparator population, never pooled with
the V11 endpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping


INPUT_SCHEMA = "banking-v11-offline-accounting-input-v1"
OUTPUT_SCHEMA = "banking-v11-offline-accounting-v1"
V11_CONFIG_SCHEMA = "banking-cil-v11-lti-ablation-config-v1"
V11_PROTOCOL_SCHEMA = "banking-cil-v11-protocol-v1"
V11_RECEIPT_SCHEMA = "banking-cil-v11-run-receipt-v1"
V10_SEAL_SCHEMA = "banking-cil-v10-training-seal-v1"
V10_PROTOCOL_SCHEMA = "banking-cil-v10-protocol"
V10_RECEIPT_SCHEMA = "banking-cil-v10-run-receipt"
EXPECTED_V11_CONFIG_FILE_SHA256 = "82794fc4dbe310e004e131405fd9b8c17b4a4dc310acbb0f22726feae95375c2"
EXPECTED_V11_SOURCE_LOCK_SHA256 = "ef4bc9777f2011c6ded823331252747e6754f79a5e713f8baffc4825ed05c5d7"
SEEDS = (17, 29, 43)
ARMS = ("LTI_NO_REWARD", "LTI_CANONICAL_TEACHER")
BASE_ARMS = ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")
TERMINAL_CHECKPOINT = ("checkpoints", "task01-final.pt")
WORK_FIELDS = (
    "full_backbone_forward_calls",
    "full_backbone_forward_tokens",
    "executed_input_tokens_including_padding",
    "padded_input_tokens",
    "tail_calls",
    "tail_grad_enabled_calls",
    "tail_no_grad_calls",
    "tail_backward_calls",
    "tail_backward_tokens",
    "wake_updates",
    "physical_adamw_updates",
    "update_attempts",
    "sleep_events",
    "sampled_output_tokens",
    "active_wall_seconds",
)
INTEGER_WORK_FIELDS = set(WORK_FIELDS) - {"active_wall_seconds"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safe_resolve(root: Path, value: str, *, require_file: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a nonempty string")
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not require_file:
        return resolved
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return path.resolve() != root.resolve()
    except ValueError:
        return False


def _strict_checkpoint_relative(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    win = PureWindowsPath(value)
    parts = tuple(value.replace("\\", "/").split("/"))
    return (not win.is_absolute() and not win.drive and not win.root and
            ".." not in win.parts and tuple(win.parts) == TERMINAL_CHECKPOINT and
            parts == TERMINAL_CHECKPOINT)


def _strict_foundation_checkpoint_relative(value: object) -> bool:
    expected = ("checkpoints", "foundation-final.pt")
    if not isinstance(value, str) or not value:
        return False
    win = PureWindowsPath(value)
    parts = tuple(value.replace("\\", "/").split("/"))
    return (not win.is_absolute() and not win.drive and not win.root and
            ".." not in win.parts and tuple(win.parts) == expected and parts == expected)


def validate_event_counts(event_counts: Any) -> int:
    if not isinstance(event_counts, Mapping):
        raise ValueError("event_counts must be an object")
    for level, count in event_counts.items():
        if not isinstance(level, str) or not level.isascii() or not level.isdigit():
            raise ValueError("event-count level keys must be decimal strings")
        if type(count) is not int or count < 0:
            raise ValueError("each event count must be a nonnegative exact integer")
    return sum(event_counts.values())


def _finite(value: Any, label: str, *, integer: bool) -> int | float:
    if integer:
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def normalize_work(work: Any) -> dict:
    """Validate known counters while preserving missing counters as null."""
    if not isinstance(work, Mapping):
        raise ValueError("work must be an object")
    out, missing = {}, []
    for field in WORK_FIELDS:
        value = work.get(field)
        if value is None:
            out[field] = None
            missing.append(field)
        else:
            out[field] = _finite(value, f"work.{field}", integer=field in INTEGER_WORK_FIELDS)
    for name in ("forward_shape_histogram", "tail_shape_histogram"):
        value = work.get(name)
        if value is None:
            out[name] = None
        elif isinstance(value, Mapping):
            parsed = {}
            for shape, count in value.items():
                if not isinstance(shape, str) or "x" not in shape:
                    raise ValueError(f"invalid {name} shape key")
                dims = shape.split("x")
                if len(dims) != 2 or any(not x.isdigit() or int(x) < 1 for x in dims):
                    raise ValueError(f"invalid {name} shape dimensions")
                parsed[shape] = _finite(count, f"{name}.{shape}", integer=True)
            out[name] = parsed
        else:
            raise ValueError(f"{name} must be an object or null")
    out["missing_work_fields"] = missing
    # Check only invariants explicitly tracked by the frozen budget receipt.
    if (out["full_backbone_forward_calls"] is not None and out["forward_shape_histogram"] is not None and
            sum(out["forward_shape_histogram"].values()) != out["full_backbone_forward_calls"]):
        raise ValueError("forward shape histogram does not sum to forward calls")
    if out["forward_shape_histogram"] is not None and out["executed_input_tokens_including_padding"] is not None:
        positions = sum(int(shape.split("x")[0]) * int(shape.split("x")[1]) * count
                        for shape, count in out["forward_shape_histogram"].items())
        if positions != out["executed_input_tokens_including_padding"]:
            raise ValueError("forward shape histogram does not sum to executed padded positions")
    if (out["tail_calls"] is not None and out["tail_shape_histogram"] is not None and
            sum(out["tail_shape_histogram"].values()) != out["tail_calls"]):
        raise ValueError("tail shape histogram does not sum to tail calls")
    if (out["tail_calls"] is not None and out["tail_grad_enabled_calls"] is not None and
            out["tail_no_grad_calls"] is not None and
            out["tail_grad_enabled_calls"] + out["tail_no_grad_calls"] != out["tail_calls"]):
        raise ValueError("grad-enabled/no-grad tail calls do not sum to total tail calls")
    if (out["executed_input_tokens_including_padding"] is not None and
            out["padded_input_tokens"] is not None and
            out["full_backbone_forward_tokens"] is not None and
            out["full_backbone_forward_tokens"] + out["padded_input_tokens"] !=
            out["executed_input_tokens_including_padding"]):
        raise ValueError("valid plus padded forward tokens do not sum to executed positions")
    if (out["wake_updates"] is not None and out["update_attempts"] is not None and
            out["wake_updates"] + out["update_attempts"] > 0 and
            out["physical_adamw_updates"] is not None and
            out["physical_adamw_updates"] > out["update_attempts"]):
        raise ValueError("physical AdamW updates exceed reserved update attempts")
    return out


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "sample_sd": None}
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def summarize_work(records: list[dict]) -> dict:
    out = {}
    for field in WORK_FIELDS:
        vals = [r["work"].get(field) for r in records]
        known = [v for v in vals if v is not None]
        missing = [r.get("label", f"seed{r['seed']}-{r['arm']}")
                   for r in records if r["work"].get(field) is None]
        out[field] = {"known_sum": sum(known), "complete": not missing,
                      "unknown_run_count": len(missing), "unknown_runs": missing,
                      "total": sum(known) if not missing else None}
    return out


def _actual_source_map(cfg: Mapping, project_root: Path, task_root: Path) -> dict[str, str]:
    actual = {}
    for key in sorted(cfg["source_hashes"]):
        if key == "tools/run_banking_lti_ablation_v11.py":
            path = task_root / key
        else:
            path = project_root / key
        if not path.is_file():
            raise FileNotFoundError(f"frozen source is missing: {key}")
        actual[key] = sha256_file(path)
    if actual != dict(cfg["source_hashes"]):
        differing = sorted(k for k in set(actual) | set(cfg["source_hashes"])
                           if actual.get(k) != cfg["source_hashes"].get(k))
        raise ValueError(f"V11 frozen source bytes differ: {differing}")
    return actual


def _v10_config_sha(cfg: Mapping) -> str:
    return canonical_sha({k: v for k, v in cfg.items() if k != "phase"})


def _metric(record: Mapping, *, expected_count: int, expected_classes: int) -> dict:
    if record.get("task_id") != 1 or record.get("evaluation_rows") != expected_count:
        raise ValueError("V11 endpoint is not the frozen task-1 development cohort")
    metric = record.get("candidate_metrics")
    if not isinstance(metric, Mapping):
        raise ValueError("candidate metrics missing")
    count, correct, classes = metric.get("count"), metric.get("correct"), metric.get("classes_evaluated")
    if count != expected_count or type(correct) is not int or not 0 <= correct <= count:
        raise ValueError("invalid candidate accuracy counts")
    if classes != expected_classes:
        raise ValueError("candidate class count differs from frozen task-1 scope")
    accuracy = metric.get("accuracy")
    expected_accuracy = correct / count
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not math.isclose(
            float(accuracy), expected_accuracy, rel_tol=0, abs_tol=1e-12):
        raise ValueError("candidate accuracy does not match correct/count")
    macro = metric.get("task_class_macro_accuracy")
    if (isinstance(macro, bool) or not isinstance(macro, (int, float)) or
            not math.isfinite(float(macro)) or not 0 <= macro <= 1):
        raise ValueError("task-class macro accuracy missing or invalid")
    nll = metric.get("gold_label_token_mean_nll")
    if not isinstance(nll, Mapping) or nll.get("count") != expected_count:
        raise ValueError("per-example gold-label token-mean NLL cohort count mismatch")
    nll_mean = nll.get("mean_token_nll")
    if isinstance(nll_mean, bool) or not isinstance(nll_mean, (int, float)) or not math.isfinite(float(nll_mean)) or nll_mean < 0:
        raise ValueError("gold-label token-mean NLL is invalid")
    return {"count": count, "correct": correct, "accuracy": expected_accuracy,
            "classes_evaluated": classes, "task_class_macro_accuracy": float(macro),
            "gold_label_token_mean_nll": float(nll_mean),
            "gold_label_nll_example_count": nll["count"]}


def _validate_v10_baseline(seal_path: Path, runs_root: Path, release_path: Path,
                           foundation_root: Path, *, seal_sha: str,
                           base_cfg: Mapping, v11_cfg: Mapping) -> dict:
    if sha256_file(seal_path) != seal_sha or seal_sha != v11_cfg["baseline_training_seal_sha256"]:
        raise ValueError("V10 baseline seal bytes do not match frozen V11 binding")
    seal = read_json(seal_path)
    base_lock = canonical_sha(base_cfg["source_hashes"])
    expected_config = _v10_config_sha(base_cfg)
    if (seal.get("schema"), seal.get("complete"), seal.get("test_data_included"), seal.get("official_test_read")) != (
            V10_SEAL_SCHEMA, True, False, False):
        raise ValueError("V10 baseline seal status/schema/test fields are invalid")
    if (seal.get("config_sha256"), seal.get("source_lock_sha256"), seal.get("training_manifest_sha256")) != (
            expected_config, base_lock, v11_cfg["training_manifest_sha256"]):
        raise ValueError("V10 baseline seal config/source/data binding mismatch")
    if v11_cfg["v10_config_sha256"] != expected_config or v11_cfg["v10_source_lock_sha256"] != base_lock:
        raise ValueError("V11 config does not bind the exact frozen V10 config/source")
    if v11_cfg["v10_source_hashes"] != base_cfg["source_hashes"]:
        raise ValueError("V11 config V10 source map mismatch")
    rows = seal.get("runs")
    expected = {(s, a) for s in SEEDS for a in BASE_ARMS}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("V10 baseline seal is not the exact 3x3 grid")
    seen, lineage = set(), {}
    release_sha = seal.get("release_artifact_sha256")
    if not isinstance(release_sha, str) or len(release_sha) != 64:
        raise ValueError("baseline release lineage is missing")
    if sha256_file(release_path) != release_sha:
        raise ValueError("V10 main-release artifact bytes differ from the baseline seal")
    release = read_json(release_path)
    if (release.get("schema"), release.get("status"), release.get("config_sha256"),
            release.get("source_lock_sha256"), release.get("training_manifest_sha256")) != (
            "banking-cil-v10-main-release-v1", "ready", expected_config, base_lock,
            v11_cfg["training_manifest_sha256"]):
        raise ValueError("V10 main-release bindings/status mismatch")
    release_foundations = release.get("foundations")
    if not isinstance(release_foundations, Mapping) or set(release_foundations) != {str(s) for s in SEEDS}:
        raise ValueError("V10 main release does not contain all three foundations")
    actual_foundation = {}
    for seed in SEEDS:
        seed_key = str(seed)
        record = release_foundations[seed_key]
        seed_root = (foundation_root / f"seed{seed}").resolve()
        if not _within(seed_root, foundation_root):
            raise ValueError("foundation seed path escapes foundation root")
        if not isinstance(record, Mapping) or record.get("status") != "qualified":
            raise ValueError("released foundation is not qualified")
        if Path(str(record.get("run_dir", ""))).resolve() != seed_root:
            raise ValueError("foundation release run_dir differs from explicit foundation root")
        receipt_path = seed_root / "foundation-receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file() or receipt_path.resolve().parent != seed_root:
            raise FileNotFoundError(receipt_path)
        receipt_sha = sha256_file(receipt_path)
        if receipt_sha != record.get("receipt_sha256"):
            raise ValueError("foundation receipt bytes differ from main release")
        receipt = read_json(receipt_path)
        if (receipt.get("schema"), receipt.get("status"), receipt.get("seed"),
                receipt.get("arm"), receipt.get("task_id"), receipt.get("config_sha256"),
                receipt.get("source_lock_sha256"), receipt.get("training_manifest_sha256"),
                receipt.get("checkpoint_sha256"), receipt.get("model_immutable_versions_unchanged"),
                receipt.get("test_data_included")) != (
                "banking-cil-v10-foundation-receipt", "qualified", seed, "FOUNDATION_CMS_ONLY", 0,
                expected_config, base_lock, v11_cfg["training_manifest_sha256"],
                record.get("checkpoint_sha256"), True, False):
            raise ValueError("foundation receipt fields do not match frozen V10 lineage")
        if (record.get("config_sha256"), record.get("source_lock_sha256"),
                record.get("training_manifest_sha256")) != (
                expected_config, base_lock, v11_cfg["training_manifest_sha256"]):
            raise ValueError("main release foundation config/source/data binding mismatch")
        metric, gate = receipt.get("metric"), receipt.get("gate")
        global_gate = base_cfg.get("task0_global_candidate_gate")
        macro_gate = base_cfg.get("task0_class_macro_gate")
        if (not isinstance(metric, Mapping) or metric.get("count") != 110 or
                metric.get("classes_evaluated") != 11 or not isinstance(gate, Mapping) or
                gate.get("passed") is not True or gate.get("global_candidate_sum_accuracy") != global_gate or
                gate.get("task0_class_macro_accuracy") != macro_gate):
            raise ValueError("foundation qualification metrics/gate are incomplete")
        accuracy = _finite(metric.get("accuracy"), "foundation.metric.accuracy", integer=False)
        macro = _finite(metric.get("task_class_macro_accuracy"),
                        "foundation.metric.task_class_macro_accuracy", integer=False)
        if (not isinstance(global_gate, (int, float)) or not isinstance(macro_gate, (int, float)) or
                accuracy < global_gate or macro < macro_gate):
            raise ValueError("foundation metric does not pass the frozen task-0 gates")
        cp_relative = receipt.get("checkpoint")
        if not _strict_foundation_checkpoint_relative(cp_relative):
            raise ValueError("unsafe foundation checkpoint relative path")
        cp_path = (seed_root / Path(*("checkpoints", "foundation-final.pt"))).resolve()
        if not _within(cp_path, seed_root) or not cp_path.is_file():
            raise ValueError("foundation checkpoint missing or outside its seed directory")
        cp_sha = sha256_file(cp_path)
        if cp_sha != receipt.get("checkpoint_sha256") or cp_sha != record.get("checkpoint_sha256"):
            raise ValueError("foundation checkpoint bytes differ from receipt/main release")
        actual_foundation[seed] = {"checkpoint_sha256": cp_sha, "receipt_sha256": receipt_sha,
            "run_dir": str(seed_root), "checkpoint_size_bytes": cp_path.stat().st_size,
            "receipt_size_bytes": receipt_path.stat().st_size}
    for row in rows:
        key = (row.get("seed"), row.get("arm")) if isinstance(row, Mapping) else None
        if key not in expected or key in seen:
            raise ValueError("V10 baseline seal has duplicate/unexpected run")
        seen.add(key)
        run_dir = safe_resolve(runs_root, row.get("run_dir", ""), require_file=False)
        if not _within(run_dir, runs_root):
            raise ValueError("V10 baseline run path escapes baseline root")
        rp, pp = run_dir / "run-receipt.json", run_dir / "protocol.json"
        if not rp.is_file() or sha256_file(rp) != row.get("run_receipt_sha256"):
            raise ValueError("V10 baseline receipt bytes differ from seal row")
        receipt, protocol = read_json(rp), read_json(pp)
        if (receipt.get("schema"), receipt.get("status"), receipt.get("tasks_completed"),
            receipt.get("seed"), receipt.get("arm"), receipt.get("config_sha256"),
            receipt.get("source_lock_sha256"), receipt.get("training_manifest_sha256"),
            receipt.get("test_data_included"), receipt.get("model_immutable_versions_unchanged")) != (
                V10_RECEIPT_SCHEMA, "complete", 7, key[0], key[1], expected_config, base_lock,
                v11_cfg["training_manifest_sha256"], False, True):
            raise ValueError("V10 baseline receipt binding/status invalid")
        parent = protocol.get("parent")
        if (protocol.get("schema"), protocol.get("phase"), protocol.get("seed"), protocol.get("arm"),
            protocol.get("config_sha256"), protocol.get("source_lock_sha256"),
            protocol.get("training_manifest_sha256"), protocol.get("source_hashes")) != (
                V10_PROTOCOL_SCHEMA, "main", key[0], key[1], expected_config, base_lock,
                v11_cfg["training_manifest_sha256"], base_cfg["source_hashes"]):
            raise ValueError("V10 baseline protocol identity/source mismatch")
        if not isinstance(parent, Mapping) or parent.get("release_artifact_sha256") != release_sha:
            raise ValueError("V10 baseline parent release mismatch")
        cp_sha, fr_sha = receipt.get("foundation_checkpoint_sha256"), parent.get("foundation_receipt_sha256")
        if not (isinstance(cp_sha, str) and len(cp_sha) == 64 and isinstance(fr_sha, str) and len(fr_sha) == 64):
            raise ValueError("V10 baseline foundation lineage fields invalid")
        if parent.get("foundation_checkpoint_sha256") != cp_sha:
            raise ValueError("V10 receipt and protocol foundation checkpoint differ")
        current = lineage.setdefault(key[0], actual_foundation[key[0]])
        if (current["checkpoint_sha256"], current["receipt_sha256"]) != (cp_sha, fr_sha):
            raise ValueError("V10 baseline arms do not share one seed foundation")
        tasks = receipt.get("task_metrics")
        if (not isinstance(tasks, list) or len(tasks) != 6 or
                [t.get("task_id") for t in tasks if isinstance(t, Mapping)] != [1, 2, 3, 4, 5, 6]):
            raise ValueError("V10 baseline must contain task endpoints 1..6 in order")
        task1 = tasks[0]
        legacy = task1.get("candidate_metrics")
        if (task1.get("evaluation_rows") != 44 or not isinstance(legacy, Mapping) or
                legacy.get("count") != 44 or legacy.get("classes_evaluated") != 22 or
                type(legacy.get("correct")) is not int or not 0 <= legacy["correct"] <= 44):
            raise ValueError("V10 legacy task-1 monitor must be 44 rows across 22 classes")
    if seen != expected or set(lineage) != set(SEEDS):
        raise ValueError("V10 baseline lineage is incomplete")
    return {"release_artifact_sha256": release_sha, "foundation_by_seed": lineage,
            "legacy_task1_monitor_scope": {"runs": 9, "rows_per_run": 44,
                "classes_per_run": 22,
                "note": "V10 task-1 monitor uses 2 development examples/class; its metrics are intentionally omitted and are not comparable to the V11 220-row endpoint."}}


def aggregate(manifest_path: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("input manifest schema mismatch")
    root = manifest_path.parent
    project_root = safe_resolve(root, manifest["project_root"], require_file=False)
    task_root = safe_resolve(root, manifest["task_root"], require_file=False)
    v11_config_path = safe_resolve(root, manifest["v11_config_path"])
    v10_config_path = safe_resolve(root, manifest["v10_config_path"])
    seal_path = safe_resolve(root, manifest["baseline_seal_path"])
    release_path = safe_resolve(root, manifest["main_release_path"])
    foundation_root = safe_resolve(root, manifest["foundation_root"], require_file=False)
    baseline_runs_root = safe_resolve(root, manifest["baseline_runs_root"], require_file=False)
    v11_runs_root = safe_resolve(root, manifest["v11_runs_root"], require_file=False)
    cfg, base_cfg = read_json(v11_config_path), read_json(v10_config_path)
    if cfg.get("schema") != V11_CONFIG_SCHEMA or cfg.get("status") != "frozen":
        raise ValueError("V11 accounting requires frozen V11 config")
    if cfg.get("seeds") != list(SEEDS) or cfg.get("arms") != list(ARMS) or cfg.get("task_id") != 1:
        raise ValueError("V11 frozen config is not the exact 3-seed x 2-arm task-1 grid")
    if cfg.get("test_data_included") is not False:
        raise ValueError("V11 config does not affirm test exclusion")
    cfg_file_sha, v11_cfg_sha = sha256_file(v11_config_path), canonical_sha(cfg)
    if cfg_file_sha != EXPECTED_V11_CONFIG_FILE_SHA256:
        raise ValueError("V11 frozen config bytes differ from the reviewed pin")
    actual_sources = _actual_source_map(cfg, project_root, task_root)
    v11_lock = canonical_sha(actual_sources)
    if v11_lock != EXPECTED_V11_SOURCE_LOCK_SHA256:
        raise ValueError("V11 source-map canonical lock differs from the reviewed pin")
    v10_lock = canonical_sha(base_cfg.get("source_hashes", {}))
    seal_sha = sha256_file(seal_path)
    baseline = _validate_v10_baseline(seal_path, baseline_runs_root, release_path, foundation_root,
                                      seal_sha=seal_sha,
                                      base_cfg=base_cfg, v11_cfg=cfg)
    if (cfg.get("model_revision"), cfg.get("model_assets_sha256"), cfg.get("initialization_seed")) != (
            base_cfg.get("model_revision"), base_cfg.get("model_assets_sha256"), base_cfg.get("initialization_seed")):
        raise ValueError("V11 model/source asset or initialization lineage differs from V10")
    if cfg.get("training_manifest_sha256") != base_cfg.get("training_manifest_sha256"):
        raise ValueError("V11 training manifest differs from V10")

    raw_runs = manifest.get("runs")
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if not isinstance(raw_runs, list) or len(raw_runs) != 6:
        raise ValueError("manifest must contain exactly six V11 runs")
    seen, run_records = set(), []
    for item in raw_runs:
        if not isinstance(item, Mapping):
            raise ValueError("run entry must be an object")
        seed, arm = item.get("seed"), item.get("arm")
        key = (seed, arm)
        if key not in expected or key in seen:
            raise ValueError("duplicate or unexpected V11 seed/arm")
        seen.add(key)
        run_dir = safe_resolve(root, item.get("run_dir", ""), require_file=False)
        if not _within(run_dir, v11_runs_root):
            raise ValueError("V11 run directory escapes its declared run root")
        protocol_path, receipt_path, progress_path = (run_dir / "protocol.json", run_dir / "run-receipt.json",
                                                        run_dir / "progress.json")
        if not protocol_path.is_file() or not receipt_path.is_file() or not progress_path.is_file():
            raise ValueError("V11 run is not terminally sealed")
        protocol, receipt, progress = read_json(protocol_path), read_json(receipt_path), read_json(progress_path)
        receipt_sha = sha256_file(receipt_path)
        if (protocol.get("schema"), protocol.get("status"), protocol.get("seed"), protocol.get("arm"),
            protocol.get("task_id"), protocol.get("config_sha256"), protocol.get("source_lock_sha256"),
            protocol.get("source_hashes"), protocol.get("v10_source_lock_sha256"),
            protocol.get("v10_config_sha256"), protocol.get("training_manifest_sha256"),
            protocol.get("baseline_training_seal_sha256"), protocol.get("test_data_included"),
            protocol.get("test_data_read"), protocol.get("run_receipt_sha256")) != (
                V11_PROTOCOL_SCHEMA, "complete", seed, arm, 1, v11_cfg_sha, v11_lock, actual_sources,
                v10_lock, _v10_config_sha(base_cfg), cfg["training_manifest_sha256"], seal_sha, False, False, receipt_sha):
            raise ValueError(f"V11 protocol binding/status mismatch: {key}")
        if (receipt.get("schema"), receipt.get("status"), receipt.get("seed"), receipt.get("arm"),
            receipt.get("task_id"), receipt.get("epochs"), receipt.get("config_sha256"),
            receipt.get("source_lock_sha256"), receipt.get("v10_source_lock_sha256"),
            receipt.get("training_manifest_sha256"), receipt.get("baseline_training_seal_sha256"),
            receipt.get("foundation_checkpoint_sha256"), receipt.get("foundation_receipt_sha256"),
            receipt.get("model_immutable_versions_unchanged"), receipt.get("test_data_included"),
            receipt.get("test_data_read")) != (
                V11_RECEIPT_SCHEMA, "complete", seed, arm, 1, cfg["epochs"], v11_cfg_sha, v11_lock,
                v10_lock, cfg["training_manifest_sha256"], seal_sha,
                baseline["foundation_by_seed"][seed]["checkpoint_sha256"],
                baseline["foundation_by_seed"][seed]["receipt_sha256"], True, False, False):
            raise ValueError(f"V11 run receipt binding/status mismatch: {key}")
        for name, expected_value in (("foundation_checkpoint_sha256", baseline["foundation_by_seed"][seed]["checkpoint_sha256"]),
                                     ("foundation_receipt_sha256", baseline["foundation_by_seed"][seed]["receipt_sha256"])):
            if protocol.get(name) != expected_value:
                raise ValueError(f"V11 protocol {name} differs from V10 paired foundation")
        if progress.get("status") != "complete" or (progress.get("seed"), progress.get("arm"), progress.get("task_id")) != (seed, arm, 1):
            raise ValueError("V11 final progress receipt is not complete")
        if protocol.get("checkpoint") != "checkpoints/task01-final.pt" or receipt.get("checkpoint") != "checkpoints/task01-final.pt":
            raise ValueError("V11 terminal checkpoint path is not the fixed task-1 endpoint")
        if not _strict_checkpoint_relative(protocol.get("checkpoint")):
            raise ValueError("unsafe V11 checkpoint path")
        cp_path = (run_dir / Path(*TERMINAL_CHECKPOINT)).resolve()
        if not _within(cp_path, run_dir) or not cp_path.is_file():
            raise ValueError("V11 terminal checkpoint missing/outside run directory")
        cp_sha = sha256_file(cp_path)
        cp_bytes = cp_path.stat().st_size
        if cp_sha != receipt.get("checkpoint_sha256") or cp_sha != protocol.get("checkpoint_sha256"):
            raise ValueError("V11 terminal checkpoint hash mismatch")
        if receipt.get("checkpoint_size_bytes") != cp_bytes:
            raise ValueError("V11 terminal checkpoint size differs from receipt")
        task1_ticks = receipt.get("task1_wake_ticks")
        controller_ticks = receipt.get("controller_state", {}).get("ticks")
        if (type(task1_ticks) is not int or task1_ticks < 1 or type(controller_ticks) is not int or
                controller_ticks < task1_ticks):
            raise ValueError("V11 wake/controller tick metadata is invalid")
        task_rows = receipt.get("task_metrics")
        if not isinstance(task_rows, list) or len(task_rows) != 1:
            raise ValueError("V11 receipt must contain exactly one task-1 endpoint")
        metric = _metric(task_rows[0], expected_count=220, expected_classes=22)
        if task_rows[0].get("task_step") != receipt.get("task1_wake_ticks"):
            raise ValueError("V11 task endpoint step differs from receipt")
        work = normalize_work(receipt.get("work"))
        sleep_summary = receipt.get("sleep_summary")
        if not isinstance(sleep_summary, Mapping):
            raise ValueError("V11 sleep summary missing")
        events = sleep_summary.get("events")
        event_counts = receipt.get("event_counts")
        if type(events) is not int or not isinstance(event_counts, Mapping):
            raise ValueError("V11 event accounting missing")
        if work["sleep_events"] is not None and events != work["sleep_events"]:
            raise ValueError("V11 sleep events disagree with authoritative work counter")
        if events != validate_event_counts(event_counts):
            raise ValueError("V11 sleep event summary differs from event-count map")
        calls = sleep_summary.get("calls")
        if type(calls) is not int or calls != sleep_summary.get("gkd_calls", -1) + sleep_summary.get("lti_calls", -1):
            raise ValueError("V11 objective-call summary is internally inconsistent")
        if (work["physical_adamw_updates"] is not None and
                sleep_summary.get("physical_adamw_steps") != work["physical_adamw_updates"]):
            raise ValueError("V11 physical AdamW counters disagree")
        if work["wake_updates"] is not None and work["wake_updates"] != receipt.get("task1_wake_ticks"):
            raise ValueError("V11 task-1 wake update counter mismatch")
        run_records.append({"seed": seed, "arm": arm, "run_dir": str(run_dir),
            "protocol_sha256": sha256_file(protocol_path), "receipt_sha256": receipt_sha,
            "checkpoint_sha256": cp_sha, "checkpoint_size_bytes": cp_bytes,
            "metric": metric, "work": work,
            "sleep_summary": {k: sleep_summary.get(k) for k in (
                "events", "calls", "gkd_calls", "lti_calls", "zero_reward_lti_calls",
                "empty_teacher_lti_calls", "skipped_zero_objective_calls", "physical_adamw_steps",
                "request_exposures_by_role")}})
    if seen != expected:
        raise ValueError("incomplete V11 seed x arm grid")
    by_arm = {}
    for arm in ARMS:
        rows = [r for r in run_records if r["arm"] == arm]
        by_arm[arm] = {
            "n": len(rows), "per_seed": [{"seed": r["seed"], **r["metric"]} for r in sorted(rows, key=lambda x: x["seed"])],
            "metrics": {name: summarize([r["metric"][name] for r in rows])
                        for name in ("accuracy", "correct", "task_class_macro_accuracy", "gold_label_token_mean_nll")},
            "work": summarize_work(rows),
            "checkpoint_bytes": summarize([r["checkpoint_size_bytes"] for r in rows]),
        }
    return {"schema": OUTPUT_SCHEMA,
        "provenance": {"input_manifest_sha256": sha256_file(manifest_path),
            "v11_config_file_sha256": cfg_file_sha, "v11_config_canonical_sha256": v11_cfg_sha,
            "v11_source_lock_sha256": v11_lock, "v10_config_file_sha256": sha256_file(v10_config_path),
            "v10_config_canonical_sha256": _v10_config_sha(base_cfg), "v10_source_lock_sha256": v10_lock,
            "baseline_seal_sha256": seal_sha, "training_manifest_sha256": cfg["training_manifest_sha256"],
            "main_release_sha256": baseline["release_artifact_sha256"],
            "foundation_by_seed": baseline["foundation_by_seed"],
            "model_revision": cfg["model_revision"], "official_test_included": False},
        "scope": {"main_runs": 6, "seeds": list(SEEDS), "arms": list(ARMS),
            "endpoint_task_id": 1, "development_rows_per_run": 220,
            "candidate_classes": 77, "seen_task_classes_for_macro": 22,
            "extra_gold_label_target_access_for_v11_objective": False,
            "extra_label_access_basis": "Frozen V11 intervention gates the Eq4 reward using canonical-label validity; it adds no gold-target loss. Development labels are used only in the fixed scorer.",
            "official_test_rows_read": False},
        "runs": sorted(run_records, key=lambda x: (x["seed"], x["arm"])),
        "by_arm": by_arm,
        "v10_legacy_task1_monitor_not_comparable": baseline["legacy_task1_monitor_scope"],
        "accounting_notes": [
            "The six V11 220-row endpoints are separate from the V10 44-row task-1 monitor; no cross-population accuracy comparison is emitted.",
            "Missing work counters remain null; group totals are null whenever any run is unknown.",
            "Checkpoint files are streamed for SHA-256/size verification only; no torch deserialization or state loading occurs.",
            "No dataset rows or official-test payload are opened.",
            "The V11 extra gold-target access flag describes the frozen objective intervention, not a general data-provenance audit."
        ]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    print(json.dumps({"schema": result["schema"], "runs": 6,
                      "official_test_included": False, "output_sha256": hashlib.sha256(payload).hexdigest()},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
