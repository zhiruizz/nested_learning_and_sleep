#!/usr/bin/env python3
"""Reanalyze completed V12 optimizer-boundary runs from aggregate receipts.

This stdlib-only tool reads final-receipt.json and numeric optimizer-batches.jsonl
for seeds 17/29/43. It never opens teacher rollout tensors, model weights,
checkpoints, training data, development rows, or official-test data. It writes
only aggregate JSON/CSV into a fresh output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "banking-v12-boundary-result-v1"
PROTOCOL_SCHEMA = "banking-v12-boundary-protocol-v1"
CONFIG_SCHEMA = "banking-optimizer-boundary-v12-config-v1"
SEEDS = (17, 29, 43)
ARMS = ("B1_HIGH", "B1_LOW", "B32_HIGH", "B32_LOW")
CONTEXTS = (
    "TEACHER", "CMS_NO_RESET", "RESET_ZERO", "B1_HIGH_30", "B1_HIGH_960",
    "B1_LOW_30", "B1_LOW_960", "B32_HIGH_960", "B32_LOW_960", "B32_LOW_3200",
)
POPS = ("old", "current", "all77")
METRIC_FAMILIES = ("teacher_forced_sum", "teacher_forced_length_mean", "greedy", "gold_label_token_mean_nll")
CONTRASTS = (
    ("B1_HIGH_30", "B1_LOW_30", "matched_request_lr_contrast"),
    ("B1_HIGH_960", "B1_LOW_960", "matched_request_lr_contrast"),
    ("B32_HIGH_960", "B32_LOW_960", "matched_request_lr_contrast"),
    ("B1_HIGH_960", "B32_HIGH_960", "matched_request_batch_size_contrast_updates_differ"),
    ("B1_LOW_960", "B32_LOW_960", "matched_request_batch_size_contrast_updates_differ"),
    ("B32_LOW_3200", "B32_LOW_960", "dose_extension_30_to_100_adam_steps_unmatched_requests"),
    ("CMS_NO_RESET", "RESET_ZERO", "reset_initialization_context_contrast"),
    ("TEACHER", "CMS_NO_RESET", "boundary_proposal_context_contrast"),
)
SAFE_BATCH_KEYS = {
    "seed", "arm", "request_start", "request_count", "requests_completed", "adamw_step",
    "loss_mean", "token_count", "preclip_grad_norm", "parameter_update_l2", "physical_adamw_steps",
}
WORK_KEYS = (
    "full_backbone_forward_calls", "full_backbone_forward_tokens", "executed_input_tokens_including_padding",
    "padded_input_tokens", "tail_backward_calls", "tail_backward_tokens", "tail_calls",
    "tail_grad_enabled_calls", "tail_no_grad_calls", "wake_updates", "physical_adamw_updates",
    "update_attempts", "sleep_events", "sampled_output_tokens", "active_wall_seconds",
)


class AnalysisError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"could not read JSON artifact {path.name}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path.name} must contain a JSON object")
    return value


def _finite_number(x: Any, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise AnalysisError(f"non-finite or non-numeric value at {where}")
    return float(x)


def _integer(x: Any, where: str, minimum: int = 0) -> int:
    if type(x) is not int or x < minimum:
        raise AnalysisError(f"invalid integer at {where}")
    return x


def _same_float(a: Any, b: Any, where: str, *, rel: float = 1e-9, abs_: float = 1e-9) -> None:
    if not math.isclose(_finite_number(a, where), _finite_number(b, where), rel_tol=rel, abs_tol=abs_):
        raise AnalysisError(f"inconsistent value at {where}")


def _subtract_nested(total: Any, part: Any, where: str) -> Any:
    """Subtract a documented subphase from an enclosing receipt delta."""
    if isinstance(total, dict) and isinstance(part, dict):
        result = {}
        for key, value in total.items():
            result[key] = _subtract_nested(value, part.get(key, 0), f"{where}.{key}")
        return result
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        p = part if isinstance(part, (int, float)) and not isinstance(part, bool) else 0
        remainder = total - p
        if remainder < -1e-9:
            raise AnalysisError(f"qualification subphase exceeds enclosing work at {where}")
        return int(remainder) if isinstance(total, int) else float(remainder)
    return total


def sample_summary(values: list[float]) -> dict[str, float | int]:
    if len(values) != 3 or not all(math.isfinite(float(x)) for x in values):
        raise AnalysisError("V12 seed summary requires exactly three finite values")
    mean = sum(values) / 3
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / 2)
    return {"n": 3, "mean": mean, "sample_sd": sd}


def _expected_arm_config(config: dict[str, Any]) -> dict[str, dict[str, int]]:
    arms = config.get("arms")
    if not isinstance(arms, list) or {x.get("name") for x in arms if isinstance(x, dict)} != set(ARMS):
        raise AnalysisError("config arm set is not the frozen four-arm design")
    result = {}
    for arm in arms:
        name = arm["name"]
        result[name] = {
            "batch": _integer(arm.get("effective_batch_size"), f"{name}.batch", 1),
            "requests": _integer(arm.get("max_requests"), f"{name}.requests", 1),
        }
    expected = {
        "B1_HIGH": {"batch": 1, "requests": 960},
        "B1_LOW": {"batch": 1, "requests": 960},
        "B32_HIGH": {"batch": 32, "requests": 960},
        "B32_LOW": {"batch": 32, "requests": 3200},
    }
    if result != expected:
        raise AnalysisError("frozen request/batch budget changed")
    return result


def _read_batch_log(path: Path, seed: int, arm_cfg: dict[str, dict[str, int]], qualification: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise AnalysisError(f"missing numeric batch log {path.name}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != SAFE_BATCH_KEYS:
                    raise AnalysisError(f"unexpected/non-numeric batch-log schema at line {line_no}")
                if row["seed"] != seed or row["arm"] not in ARMS:
                    raise AnalysisError(f"batch-log seed/arm mismatch at line {line_no}")
                for key in ("request_start", "request_count", "requests_completed", "adamw_step", "token_count", "physical_adamw_steps"):
                    _integer(row[key], f"batch line {line_no}.{key}")
                for key in ("loss_mean", "preclip_grad_norm", "parameter_update_l2"):
                    _finite_number(row[key], f"batch line {line_no}.{key}")
                if row["request_count"] < 1 or row["physical_adamw_steps"] != 1:
                    raise AnalysisError(f"invalid committed batch at line {line_no}")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"could not parse numeric batch log {path.name}") from exc

    by_arm = {name: [] for name in ARMS}
    for row in rows:
        by_arm[row["arm"]].append(row)
    batch_summaries = {}
    missing_qualification_core_requests = 0
    for name, spec in arm_cfg.items():
        entries = sorted(by_arm[name], key=lambda x: x["request_start"])
        cursor = 0
        losses, norms, update_norms, token_count = [], [], [], 0
        for row in entries:
            start, count = row["request_start"], row["request_count"]
            if start != cursor:
                gap = start - cursor
                permitted = (seed == 17 and name == "B32_LOW" and cursor == 32 and start == 64 and gap == 32)
                if not permitted:
                    raise AnalysisError(f"unexplained request-log gap in {name}")
                q = qualification or {}
                original = q.get("original_metrics")
                if q.get("original_request_range") != [32, 64] or not isinstance(original, dict):
                    raise AnalysisError("seed-17 B32_LOW log gap lacks original branch metrics")
                for k in ("loss_mean", "preclip_grad_norm", "parameter_update_l2"):
                    _finite_number(original.get(k), f"qualification.original_metrics.{k}")
                token_count += _integer(original.get("token_count"), "qualification.original_metrics.token_count")
                losses.append(float(original["loss_mean"]))
                norms.append(float(original["preclip_grad_norm"]))
                update_norms.append(float(original["parameter_update_l2"]))
                missing_qualification_core_requests += gap
                cursor += gap
            if start != cursor:
                raise AnalysisError(f"batch log is out of order for {name}")
            if count > spec["batch"] or count != spec["batch"]:
                # Every frozen request budget is divisible by its batch size.
                raise AnalysisError(f"unexpected partial batch in {name}")
            end = start + count
            if row["requests_completed"] != end or row["adamw_step"] != end // spec["batch"]:
                raise AnalysisError(f"request/update cursor mismatch in {name}")
            cursor = end
            losses.append(float(row["loss_mean"]))
            norms.append(float(row["preclip_grad_norm"]))
            update_norms.append(float(row["parameter_update_l2"]))
            token_count += row["token_count"]
        if cursor != spec["requests"]:
            raise AnalysisError(f"batch log does not reconcile to configured requests for {name}")
        expected_lines = spec["requests"] // spec["batch"]
        if len(entries) != expected_lines - (1 if name == "B32_LOW" and seed == 17 else 0):
            raise AnalysisError(f"unexpected batch-log row count for {name}")
        batch_summaries[name] = {
            "logged_batch_count": len(entries),
            "core_request_exposures_from_config": spec["requests"],
            "core_adamw_steps_from_config": spec["requests"] // spec["batch"],
            "numeric_batch_metadata_mean": {
                "loss_mean": sum(losses) / len(losses),
                "preclip_grad_norm": sum(norms) / len(norms),
                "parameter_update_l2": sum(update_norms) / len(update_norms),
                "tokens_per_batch_sum": token_count,
                "n_batches_including_seed17_original_32_to_64_metrics": len(losses),
            },
            "batch_loss_note": "stream diagnostic only; differently sized batches and data segments are not a fixed-cohort teacher-KL metric",
        }
    return {"per_arm": batch_summaries, "seed17_original_gap_requests_recovered_from_qualification": missing_qualification_core_requests}, rows


def _validate_metric_context(runtime: dict[str, Any], context: str) -> dict[str, dict[str, dict[str, Any]]]:
    candidates = runtime.get("candidate_metrics")
    counts = runtime.get("candidate_correct_counts")
    greedy = runtime.get("greedy_generation_metrics")
    if not all(isinstance(x, dict) for x in (candidates, counts, greedy)):
        raise AnalysisError("runtime metric families missing")
    if context not in candidates or context not in counts or context not in greedy:
        raise AnalysisError(f"missing metrics for context {context}")
    if counts[context].get("candidate_count_per_example") != 77:
        raise AnalysisError("candidate set is not the full global 77-label set")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for family in ("teacher_forced_sum", "teacher_forced_length_mean"):
        groups = candidates[context].get(family)
        if not isinstance(groups, dict):
            raise AnalysisError(f"missing {family} metrics")
        result[family] = {}
        for pop, count in zip(POPS, (110, 110, 220)):
            group = groups.get(pop)
            if not isinstance(group, dict) or group.get("count") != count:
                raise AnalysisError(f"{context}.{family}.{pop} denominator mismatch")
            correct = _integer(group.get("correct"), f"{context}.{family}.{pop}.correct")
            if correct > count or group.get("invalid_predictions") != 0:
                raise AnalysisError(f"invalid {context}.{family}.{pop} prediction count")
            _same_float(group.get("accuracy"), correct / count, f"{context}.{family}.{pop}.accuracy")
            result[family][pop] = {"count": count, "correct": correct, "accuracy": correct / count}
        per_class = groups.get("per_class")
        if not isinstance(per_class, dict) or len(per_class) != 77:
            raise AnalysisError("expected the full 77-class diagnostic map")
        class_counts = [_integer(v.get("count"), "per_class.count") for v in per_class.values()]
        if sum(class_counts) != 220:
            raise AnalysisError("per-class counts do not sum to 220")
        if sum(count > 0 for count in class_counts) != 22 or any(count not in (0, 10) for count in class_counts):
            raise AnalysisError("development cohort should contain ten examples per represented class")
        if sum(_integer(v.get("correct"), "per_class.correct") for v in per_class.values()) != result[family]["all77"]["correct"]:
            raise AnalysisError(f"{family} per-class correct counts do not reconcile to global total")
    gold = candidates[context].get("gold_label_token_mean_nll")
    if not isinstance(gold, dict):
        raise AnalysisError("missing gold-label-token NLL diagnostics")
    result["gold_label_token_mean_nll"] = {}
    for pop, count in zip(POPS, (110, 110, 220)):
        group = gold.get(pop)
        if not isinstance(group, dict) or group.get("count") != count:
            raise AnalysisError(f"{context}.gold_nll.{pop} denominator mismatch")
        result["gold_label_token_mean_nll"][pop] = {
            "count": count, "mean_token_nll": _finite_number(group.get("mean_token_nll"), f"{context}.gold_nll.{pop}")}
    g = greedy[context]
    if not isinstance(g, dict) or g.get("count") != 220 or g.get("candidate_constrained") is not False:
        raise AnalysisError(f"{context} greedy metric is not full-set unconstrained decoding")
    corr = _integer(g.get("correct"), f"{context}.greedy.correct")
    invalid = _integer(g.get("invalid_predictions"), f"{context}.greedy.invalid_predictions")
    if corr + invalid > 220:
        raise AnalysisError(f"{context} greedy counts exceed denominator")
    _same_float(g.get("accuracy"), corr / 220, f"{context}.greedy.accuracy")
    result["greedy"] = {"all77": {"count": 220, "correct": corr, "invalid_predictions": invalid, "accuracy": corr / 220}}
    # Cross-check duplicate counts emitted by runtime.
    dup = counts[context]
    if dup.get("gold_correct_sum") != result["teacher_forced_sum"]["all77"]["correct"]:
        raise AnalysisError(f"{context} duplicate sum-accuracy count mismatch")
    if dup.get("gold_correct_length_mean") != result["teacher_forced_length_mean"]["all77"]["correct"]:
        raise AnalysisError(f"{context} duplicate length-mean count mismatch")
    return result


def _validate_seed(seed: int, run_dir: Path, config: dict[str, Any], config_sha: str,
                   expected_sources: dict[str, str], arm_cfg: dict[str, dict[str, int]]) -> dict[str, Any]:
    result_path = run_dir / "final-receipt.json"
    protocol_path = run_dir / "protocol.json"
    result = _read_json(result_path)
    protocol = _read_json(protocol_path)
    result_sha, protocol_sha = sha256_file(result_path), sha256_file(protocol_path)
    if result.get("schema") != SCHEMA or result.get("status") != "complete" or result.get("seed") != seed:
        raise AnalysisError(f"seed {seed} result is missing/incomplete")
    if result.get("official_test_read") is not False or result.get("test_data_read") is not False:
        raise AnalysisError(f"seed {seed} receipt does not affirm test exclusion")
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "complete" or protocol.get("seed") != seed:
        raise AnalysisError(f"seed {seed} protocol receipt is incomplete")
    if protocol.get("final_receipt_sha256") != result_sha:
        raise AnalysisError(f"seed {seed} protocol/final-receipt hash mismatch")
    if protocol.get("config_sha256") != config_sha:
        raise AnalysisError(f"seed {seed} frozen config hash mismatch")
    if protocol.get("v12_source_hashes") != expected_sources:
        raise AnalysisError(f"seed {seed} frozen source hash map mismatch")
    if protocol.get("training_manifest_sha256") != config.get("training_manifest_sha256"):
        raise AnalysisError(f"seed {seed} train/dev manifest mismatch")
    if result.get("claim_scope") != config.get("claim_scope"):
        raise AnalysisError(f"seed {seed} claim scope mismatch")

    dev = result.get("development")
    if not isinstance(dev, dict) or dev.get("population") != "first_task0_plus_task1_development_only":
        raise AnalysisError(f"seed {seed} development population mismatch")
    if dev.get("counts") != {"old": 110, "current": 110} or dev.get("global_candidate_labels") != 77:
        raise AnalysisError(f"seed {seed} cohort counts mismatch")
    runtime = dev.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("schema") != "banking-cil-v10-runtime-v1" or runtime.get("classes") != 77:
        raise AnalysisError(f"seed {seed} runtime score schema mismatch")
    if set(runtime.get("candidate_metrics", {})) != set(CONTEXTS):
        raise AnalysisError(f"seed {seed} context set mismatch")
    metrics = {ctx: _validate_metric_context(runtime, ctx) for ctx in CONTEXTS}
    if runtime.get("candidate_sequences") != 220 * 77:
        raise AnalysisError("expected global full-vocabulary candidate sequences")
    if not isinstance(runtime.get("greedy_generation_metrics"), dict):
        raise AnalysisError("greedy metrics missing")

    rollouts = result.get("fixed_teacher_rollouts")
    if not isinstance(rollouts, dict) or rollouts.get("requests") != 3200 or rollouts.get("gold_labels_used_as_loss_targets") is not False:
        raise AnalysisError("shared teacher rollout contract mismatch")
    if rollouts.get("class_role_counts") != {"old": 1600, "current": 1600}:
        raise AnalysisError("fixed teacher rollout old/current balance mismatch")

    boundary = result.get("boundary")
    if not isinstance(boundary, dict) or boundary.get("due_levels") != [0] or boundary.get("prefix_wake_ticks") not in (4, 10, 14):
        raise AnalysisError("first-due level-0 boundary metadata mismatch")

    arm_receipts = result.get("optimizer_arms")
    if not isinstance(arm_receipts, dict) or set(arm_receipts) != set(ARMS):
        raise AnalysisError("optimizer-arm receipt set mismatch")
    for arm, spec in arm_cfg.items():
        rr = arm_receipts[arm]
        if (rr.get("requests") != spec["requests"] or rr.get("adamw_steps") != spec["requests"] // spec["batch"]
                or rr.get("effective_batch_size") != spec["batch"]):
            raise AnalysisError(f"{seed}/{arm} arm counters do not match frozen budget")
    expected_core_steps = sum(x["requests"] // x["batch"] for x in arm_cfg.values())
    expected_core_requests = sum(x["requests"] for x in arm_cfg.values())
    if (expected_core_steps, expected_core_requests) != (2050, 6080):
        raise AnalysisError("internal frozen V12 core budget mismatch")

    qual = result.get("restore_qualification")
    if seed == 17:
        if not isinstance(qual, dict) or qual.get("status") != "pass" or qual.get("post_range_snapshot_state_equal") is not True:
            raise AnalysisError("seed-17 disk-restore qualification did not pass")
        if (qual.get("original_request_range"), qual.get("restored_request_range"),
                qual.get("extra_request_evaluations"), qual.get("extra_physical_adamw_steps")) != ([32,64],[32,64],32,1):
            raise AnalysisError("seed-17 qualification exposure/update accounting mismatch")
        extra = qual.get("extra_work_delta")
        if not isinstance(extra, dict):
            raise AnalysisError("seed-17 qualification extra work missing")
    elif qual is not None:
        raise AnalysisError(f"unexpected restore qualification in seed {seed}")

    log_summary, logs = _read_batch_log(run_dir / "optimizer-batches.jsonl", seed, arm_cfg, qual)
    budget = result.get("budget")
    if not isinstance(budget, dict):
        raise AnalysisError("global V12 work budget missing")
    for key in WORK_KEYS:
        if key not in budget:
            raise AnalysisError(f"global work counter missing: {key}")
        if isinstance(budget[key], (int, float)):
            _finite_number(budget[key], f"budget.{key}")
    # Core updates are fixed by config. In seed 17 the extra restored branch is additional work.
    if budget.get("physical_adamw_updates") != expected_core_steps + (1 if seed == 17 else 0):
        raise AnalysisError("physical AdamW total does not separate the restore replay")
    if budget.get("update_attempts") != expected_core_steps + (1 if seed == 17 else 0):
        raise AnalysisError("optimizer-attempt total does not reconcile to committed V12 steps")
    if budget.get("wake_updates") < boundary["prefix_wake_ticks"]:
        raise AnalysisError("wake update counter is below observed prefix ticks")

    arm_work_including_qualification = {arm: arm_receipts[arm]["work_delta_including_restore_validation"] for arm in ARMS}
    arm_work_core_only = dict(arm_work_including_qualification)
    qualification_extra = qual.get("extra_work_delta") if seed == 17 else None
    if seed == 17:
        arm_work_core_only["B32_LOW"] = _subtract_nested(
            arm_work_including_qualification["B32_LOW"], qualification_extra, "B32_LOW core work")
    phases = {
        "shared_boundary": result.get("shared_boundary_work"),
        "fixed_teacher_rollouts": result.get("fixed_rollout_work"),
        "optimizer_arms_core_only": arm_work_core_only,
        "restore_qualification_extra": qualification_extra,
        "development_evaluation": dev.get("work_delta"),
    }
    # Counter-wise phase accounting is additive only after extracting the disk-restore branch.
    for key in WORK_KEYS:
        if key == "active_wall_seconds":
            continue
        phase_total = 0
        for value in (phases["shared_boundary"], phases["fixed_teacher_rollouts"], phases["development_evaluation"]):
            if isinstance(value, dict) and isinstance(value.get(key), (int, float)):
                phase_total += value[key]
        for arm in ARMS:
            arm_delta = phases["optimizer_arms_core_only"].get(arm, {})
            if isinstance(arm_delta, dict) and isinstance(arm_delta.get(key), (int, float)):
                phase_total += arm_delta[key]
        extra_delta = phases["restore_qualification_extra"]
        if isinstance(extra_delta, dict) and isinstance(extra_delta.get(key), (int, float)):
            phase_total += extra_delta[key]
        if phase_total != budget[key]:
            raise AnalysisError(f"phase counters do not reconcile with total budget at {key}")
    return {
        "seed": seed,
        "result_sha256": result_sha,
        "protocol_sha256": protocol_sha,
        "optimizer_batch_log_sha256": sha256_file(run_dir / "optimizer-batches.jsonl"),
        "boundary_proposal_sha256": boundary.get("proposal_sha256"),
        "foundation_receipt_sha256": protocol.get("foundation_receipt_sha256"),
        "foundation_checkpoint_sha256": protocol.get("foundation_checkpoint_sha256"),
        "v10_source_lock_sha256": protocol.get("v10_source_lock_sha256"),
        "training_manifest_sha256": protocol.get("training_manifest_sha256"),
        "context_metrics": metrics,
        "boundary_prefix_wake_ticks": boundary["prefix_wake_ticks"],
        "fixed_teacher_rollout_requests": rollouts["requests"],
        "fixed_teacher_rollout_tokens_generated": rollouts.get("generated_answer_tokens"),
        "optimizer_arms_including_qualification": arm_work_including_qualification,
        "phase_work": phases,
        "budget_actual": {k: budget[k] for k in WORK_KEYS},
        "core_request_exposures": expected_core_requests,
        "core_adamw_steps": expected_core_steps,
        "extra_restore_request_evaluations": 32 if seed == 17 else 0,
        "extra_restore_adamw_steps": 1 if seed == 17 else 0,
        "batch_log_summary": log_summary,
        "optimizer_batch_log_rows": len(logs),
        "official_test_read": False,
    }


def _metric_rows(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for seed_result in per_seed:
        seed = seed_result["seed"]
        for context, family_map in seed_result["context_metrics"].items():
            for family, populations in family_map.items():
                for population, values in populations.items():
                    row = {"seed": seed, "context": context, "metric": family, "population": population,
                           "count": values["count"]}
                    for key in ("correct", "invalid_predictions", "accuracy", "mean_token_nll"):
                        if key in values:
                            row[key] = values[key]
                    rows.append(row)
    return rows


def _summarize(per_seed: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_rows = _metric_rows(per_seed)
    summary = {}
    index = {(r["seed"], r["context"], r["metric"], r["population"]): r for r in metric_rows}
    keys = sorted({(r["context"], r["metric"], r["population"]) for r in metric_rows})
    for context, metric, pop in keys:
        values = [index[(seed, context, metric, pop)] for seed in SEEDS]
        summary[f"{context}|{metric}|{pop}"] = {
            k: sample_summary([float(v[k]) for v in values]) for k in ("accuracy", "mean_token_nll")
            if all(k in v for v in values)
        }
    contrast_summary = []
    for left, right, label in CONTRASTS:
        for metric in METRIC_FAMILIES:
            pops = ("old", "current", "all77") if metric != "greedy" else ("all77",)
            for pop in pops:
                diffs = []
                for seed in SEEDS:
                    lm = index[(seed, left, metric, pop)]
                    rm = index[(seed, right, metric, pop)]
                    key = "mean_token_nll" if metric == "gold_label_token_mean_nll" else "accuracy"
                    diffs.append(float(lm[key]) - float(rm[key]))
                contrast_summary.append({"left": left, "right": right, "label": label,
                    "metric": metric, "population": pop, "left_minus_right": sample_summary(diffs)})
    return {"metrics_by_context_family_population": summary, "paired_contrasts": contrast_summary}, metric_rows


def _engineering_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = _read_json(path)
    if data.get("schema") != "banking-v12-safe-engineering-history-v1" or data.get("raw_rows_or_model_data_included") is not False:
        raise AnalysisError("engineering-history input is not the reviewed aggregate-only schema")
    events = data.get("events")
    if not isinstance(events, list):
        raise AnalysisError("engineering-history events missing")
    output = []
    for event in events:
        if not isinstance(event, dict):
            raise AnalysisError("malformed engineering-history entry")
        # Strict projection: exclude raw exception text, local paths, and arbitrary nested fields.
        item = {k: event[k] for k in ("event", "status", "error_type", "active_runner_seconds",
              "launcher_seconds_including_preflight_model_load_and_active_runner", "elapsed_wall_seconds",
              "passed", "failed", "pytest_exit_code", "real_model_loaded", "cuda_initialization_prohibited",
              "scientific_receiver_arms_completed") if k in event}
        if isinstance(event.get("work"), dict):
            item["work"] = {k: event["work"][k] for k in (
                "full_backbone_forward_calls", "full_backbone_forward_tokens", "executed_input_tokens_including_padding",
                "padded_input_tokens", "tail_backward_calls", "tail_backward_tokens", "tail_calls",
                "wake_updates", "physical_adamw_updates", "update_attempts", "sleep_events", "active_wall_seconds")
                if k in event["work"]}
        if isinstance(event.get("ledger"), dict):
            item["ledger"] = {k: event["ledger"][k] for k in ("optimizer_attempts", "passed", "limit") if k in event["ledger"]}
        item["evidence_sha256"] = [e["sha256"] for e in event.get("evidence", []) if isinstance(e, dict) and isinstance(e.get("sha256"), str)]
        output.append(item)
    return {"schema": data["schema"], "source_sha256": sha256_file(path), "scope": data.get("scope"),
            "unknown_or_unrecorded_costs": data.get("unknown_or_unrecorded_costs"), "events": output}


def analyze(run_root: Path, config_path: Path, out_dir: Path, history_path: Path | None = None) -> dict[str, Any]:
    if out_dir.exists():
        raise AnalysisError("analysis output directory must be fresh")
    config = _read_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "frozen" or tuple(config.get("seeds", ())) != SEEDS:
        raise AnalysisError("V12 analysis requires the frozen three-seed config")
    arm_cfg = _expected_arm_config(config)
    config_sha = sha256_file(config_path)
    sources = config.get("source_hashes")
    if not isinstance(sources, dict) or any(not isinstance(k, str) or len(v) != 64 for k, v in sources.items()):
        raise AnalysisError("frozen source-hash map missing/malformed")
    per_seed = []
    # A --run-root contains `seed17`, `seed29`, and `seed43` output folders.
    for seed in SEEDS:
        per_seed.append(_validate_seed(seed, run_root / f"seed{seed}", config, config_sha, sources, arm_cfg))
    if len({x["training_manifest_sha256"] for x in per_seed}) != 1 or len({x["v10_source_lock_sha256"] for x in per_seed}) != 1:
        raise AnalysisError("run seeds do not share one locked data/source identity")
    derived, rows = _summarize(per_seed)
    total_core_requests = sum(x["core_request_exposures"] for x in per_seed)
    total_core_adam = sum(x["core_adamw_steps"] for x in per_seed)
    total_extra_requests = sum(x["extra_restore_request_evaluations"] for x in per_seed)
    total_extra_adam = sum(x["extra_restore_adamw_steps"] for x in per_seed)
    if (total_core_requests, total_core_adam, total_extra_requests, total_extra_adam) != (18240, 6150, 32, 1):
        raise AnalysisError("three-seed V12 request/Adam accounting mismatch")
    summary = {
        "schema": "banking-v12-optimizer-boundary-public-aggregate-v1",
        "status": "complete_descriptive_development_analysis",
        "design": {
            "seeds": list(SEEDS), "contexts": list(CONTEXTS), "candidate_labels": 77,
            "population": "task1 development, 220 rows per seed (110 old, 110 current)",
            "primary": "global 77-label teacher-forced sum log-likelihood exact match",
            "secondary": ["global length-mean teacher-forced exact match", "unconstrained greedy exact match", "gold-label token mean NLL"],
            "no_selection": True,
            "interpretation_limits": [
                "three seeds are seeded class orders from a shared pretraining/foundation lineage, not three independent pretrained initializations",
                "request-matched arms can differ in physical optimizer steps and effective batch size",
                "B32_LOW 960-to-3200 is an explicit dose extension, not a request-matched contrast",
                "batch loss summaries are stream diagnostics, not fixed-cohort teacher-KL evaluations",
                "no pooled token NLL/PPL or FLOP estimate is computed",
            ],
        },
        "provenance": {"analysis_tool_sha256": sha256_file(Path(__file__).resolve()),
            "config_sha256": config_sha, "config_status": config["status"],
            "training_manifest_sha256": per_seed[0]["training_manifest_sha256"],
            "v10_source_lock_sha256": per_seed[0]["v10_source_lock_sha256"],
            "source_hashes": sources,
            "runs": [{k:x[k] for k in ("seed","result_sha256","protocol_sha256","optimizer_batch_log_sha256","boundary_proposal_sha256","foundation_receipt_sha256","foundation_checkpoint_sha256")} for x in per_seed]},
        "cost_accounting": {
            "core_request_exposures": total_core_requests, "core_adamw_steps": total_core_adam,
            "additional_seed17_restore_qualification_requests": total_extra_requests,
            "additional_seed17_restore_qualification_adamw_steps": total_extra_adam,
            "seeds": [{"seed":x["seed"],"boundary_prefix_wake_ticks":x["boundary_prefix_wake_ticks"],
                       "core_requests":x["core_request_exposures"],"core_adamw_steps":x["core_adamw_steps"],
                       "qualification_extra_requests":x["extra_restore_request_evaluations"],
                       "qualification_extra_adamw_steps":x["extra_restore_adamw_steps"],
                       "actual_budget":x["budget_actual"],"phase_work":x["phase_work"],
                       "optimizer_batch_log_summary":x["batch_log_summary"]} for x in per_seed],
            "note":"Receipt totals include the separate seed-17 restore qualification; it is shown separately and is not counted as a fourth scientific arm or as extra core exposure.",
        },
        "statistics": derived,
        "engineering_history": _engineering_summary(history_path),
        "raw_rows_or_ids_emitted": False,
        "official_test_read": False,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    json_path = out_dir / "v12-boundary-aggregate.json"
    csv_path = out_dir / "v12-boundary-metrics.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        columns = ["seed", "context", "metric", "population", "count", "correct", "invalid_predictions", "accuracy", "mean_token_nll"]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return {"aggregate_path": json_path, "aggregate_sha256": sha256_file(json_path),
            "metrics_csv_path": csv_path, "metrics_csv_sha256": sha256_file(csv_path), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engineering-history", type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.run_root, args.config, args.out_dir, args.engineering_history)
    print(json.dumps({"status": result["summary"]["status"], "aggregate_sha256": result["aggregate_sha256"],
                      "metrics_csv_sha256": result["metrics_csv_sha256"], "seeds": list(SEEDS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
