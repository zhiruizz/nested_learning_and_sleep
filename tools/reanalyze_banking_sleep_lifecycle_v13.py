#!/usr/bin/env python3
"""Strict, offline aggregation of complete Banking77 V13 lifecycle receipts.

Only aggregate final receipts and frozen source/config files are read. This
tool never opens a checkpoint, dataset, row text, generated token, or test
payload. It is intended for the V13 receipt contract, not for other runs.
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


CONFIG_SCHEMA = "banking-v13-batched-lifecycle-config-v1"
RESULT_SCHEMA = "banking-v13-batched-lifecycle-result-v1"
OUTPUT_SCHEMA = "banking-v13-batched-lifecycle-aggregate-v1"
SEEDS = (17, 29, 43)
ARMS = ("CMS_ONLY", "B1_HIGH", "B1_LOW", "B32_HIGH", "B32_LOW", "B32_LOW_DOSE")
STAGES = ("event1_before", "event1_after", "event2_before", "event2_after", "postwake15")
PARENT_RELEASE = "d9c516cc16da3d8ad7d2cfb2f8173fdae90c3a87"
EXPECTED_ARMS = {
    "CMS_ONLY": (0, None, None),
    "B1_HIGH": (960, 1, 0.001),
    "B1_LOW": (960, 1, 0.000005),
    "B32_HIGH": (960, 32, 0.001),
    "B32_LOW": (960, 32, 0.000005),
    "B32_LOW_DOSE": (3200, 32, 0.000005),
}
PREFIX_TICKS = {"17": 4, "29": 10, "43": 14}
TOTAL_TICKS = {"17": 35, "29": 41, "43": 45}
POPS = ("old", "current", "all77")
METRICS = ("teacher_forced_sum", "teacher_forced_length_mean", "gold_label_token_mean_nll", "greedy")
CONTRASTS = (
    ("B1_HIGH", "B1_LOW", "LR effect, batch 1"),
    ("B32_HIGH", "B32_LOW", "LR effect, batch 32"),
    ("B1_HIGH", "B32_HIGH", "batch effect, high LR"),
    ("B1_LOW", "B32_LOW", "batch effect, low LR"),
    ("B32_LOW_DOSE", "B32_LOW", "3200 vs 960 requests/event dose contrast"),
)
ADDITIVE_WORK_COUNTERS = (
    "full_backbone_forward_calls", "full_backbone_forward_tokens",
    "executed_input_tokens_including_padding", "padded_input_tokens",
    "tail_backward_calls", "tail_backward_tokens", "tail_calls",
    "tail_grad_enabled_calls", "tail_no_grad_calls", "wake_updates",
    "physical_adamw_updates", "update_attempts", "sleep_events",
    "sampled_output_tokens",
)


class AnalysisError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON artifact {path.name}") from exc
    if not isinstance(result, dict):
        raise AnalysisError(f"{path.name} must contain an object")
    return result


def _hex64(x: Any, where: str) -> str:
    if not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x.lower()):
        raise AnalysisError(f"invalid SHA-256 at {where}")
    return x.lower()


def _int(x: Any, where: str, lo: int = 0) -> int:
    if type(x) is not int or x < lo:
        raise AnalysisError(f"invalid integer at {where}")
    return x


def _num(x: Any, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise AnalysisError(f"invalid finite number at {where}")
    return float(x)


def _summary(values: list[float]) -> dict[str, float | int]:
    if len(values) != 3:
        raise AnalysisError("statistics require all three seeds")
    return {"n": 3, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def _same_aggregate(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same_aggregate(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_aggregate(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-8)
    return a == b


def _validate_stage_eval(evaluation: Any, stage: str) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(evaluation, dict) or evaluation.get("schema") != "banking-cil-v10-runtime-v1":
        raise AnalysisError(f"{stage} evaluation runtime schema mismatch")
    if evaluation.get("classes") != 77 or evaluation.get("candidate_sequences") != 220 * 77:
        raise AnalysisError(f"{stage} candidate set must cover 77 labels over 220 rows")
    cand = evaluation.get("candidate_metrics")
    counts = evaluation.get("candidate_correct_counts")
    greedy = evaluation.get("greedy_generation_metrics")
    if not all(isinstance(x, dict) and len(x) == 1 for x in (cand, counts, greedy)):
        raise AnalysisError(f"{stage} must score exactly one candidate context")
    context = next(iter(cand))
    if context not in counts or context not in greedy:
        raise AnalysisError(f"{stage} context differs across metric families")
    cm, cc = cand[context], counts[context]
    required = ("teacher_forced_sum", "teacher_forced_length_mean", "gold_label_token_mean_nll")
    if not isinstance(cm, dict) or any(not isinstance(cm.get(k), dict) for k in required):
        raise AnalysisError(f"{stage} metric family missing")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for family in ("teacher_forced_sum", "teacher_forced_length_mean"):
        groups = cm[family]
        result[family] = {}
        for pop, denom in zip(POPS, (110, 110, 220)):
            group = groups.get(pop)
            if not isinstance(group, dict) or group.get("count") != denom:
                raise AnalysisError(f"{stage}.{family}.{pop} denominator mismatch")
            correct = _int(group.get("correct"), f"{stage}.{family}.{pop}.correct")
            invalid = _int(group.get("invalid_predictions"), f"{stage}.{family}.{pop}.invalid")
            if correct > denom or invalid != 0:
                raise AnalysisError(f"{stage}.{family}.{pop} has invalid/overcounted predictions")
            acc = _num(group.get("accuracy"), f"{stage}.{family}.{pop}.accuracy")
            if not math.isclose(acc, correct / denom, abs_tol=1e-12):
                raise AnalysisError(f"{stage}.{family}.{pop} accuracy mismatch")
            result[family][pop] = {"count": denom, "correct": correct, "invalid_predictions": invalid,
                                   "accuracy": acc}
        per_class = groups.get("per_class")
        if not isinstance(per_class, dict) or len(per_class) != 77:
            raise AnalysisError(f"{stage}.{family} lacks the full 77-class aggregate")
        class_counts = [_int(v.get("count"), f"{stage}.{family}.per_class.count") for v in per_class.values()]
        if sum(class_counts) != 220:
            raise AnalysisError(f"{stage}.{family} per-class denominators do not sum to 220")
        if sum(v > 0 for v in class_counts) != 22 or any(v not in (0, 10) for v in class_counts):
            raise AnalysisError(f"{stage}.{family} expected 22 represented classes with 10 rows each")
        if sum(_int(v.get("correct"), f"{stage}.{family}.per_class.correct") for v in per_class.values()) != result[family]["all77"]["correct"]:
            raise AnalysisError(f"{stage}.{family} per-class correct counts mismatch")
    gold = cm["gold_label_token_mean_nll"]
    result["gold_label_token_mean_nll"] = {}
    for pop, denom in zip(POPS, (110, 110, 220)):
        group = gold.get(pop)
        if not isinstance(group, dict) or group.get("count") != denom:
            raise AnalysisError(f"{stage}.gold-NLL.{pop} denominator mismatch")
        result["gold_label_token_mean_nll"][pop] = {
            "count": denom, "mean_token_nll": _num(group.get("mean_token_nll"), f"{stage}.NLL.{pop}")}
    gr = greedy[context]
    if not isinstance(gr, dict) or gr.get("candidate_constrained") is not False or gr.get("count") != 220:
        raise AnalysisError(f"{stage} greedy metric must be unconstrained over 220 rows")
    gc = _int(gr.get("correct"), f"{stage}.greedy.correct")
    gi = _int(gr.get("invalid_predictions"), f"{stage}.greedy.invalid")
    ga = _num(gr.get("accuracy"), f"{stage}.greedy.accuracy")
    if gc + gi > 220 or not math.isclose(ga, gc / 220, abs_tol=1e-12):
        raise AnalysisError(f"{stage} greedy counts/accuracy mismatch")
    result["greedy"] = {"all77": {"count": 220, "correct": gc, "invalid_predictions": gi, "accuracy": ga}}
    if cc.get("candidate_count_per_example") != 77:
        raise AnalysisError(f"{stage} duplicate full-vocabulary counter mismatch")
    if cc.get("gold_correct_sum") != result["teacher_forced_sum"]["all77"]["correct"]:
        raise AnalysisError(f"{stage} duplicate primary count mismatch")
    return result


def _validate_work_delta(work: Any, where: str) -> dict[str, int | float]:
    if not isinstance(work, dict) or not work:
        raise AnalysisError(f"missing work delta at {where}")
    out: dict[str, int | float] = {}
    for key, value in work.items():
        if type(value) is int:
            if value < 0:
                raise AnalysisError(f"negative work counter at {where}.{key}")
            out[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0:
            out[key] = float(value)
        else:
            raise AnalysisError(f"non-numeric/negative work counter at {where}.{key}")
    return out


def _count_map(value: Any, where: str, expected_total: int) -> dict[str, int]:
    if not isinstance(value, dict):
        raise AnalysisError(f"missing count map at {where}")
    counts = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise AnalysisError(f"invalid count-map key at {where}")
        counts[key] = _int(count, f"{where}.{key}")
    if sum(counts.values()) != expected_total:
        raise AnalysisError(f"count map does not reconcile at {where}")
    return counts


def _sum_deltas(parts: list[dict[str, int | float]]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for part in parts:
        for key, value in part.items():
            result[key] = result.get(key, 0) + value
    return result


def _check_work_close(actual: Any, expected: Any, keys: tuple[str, ...], where: str) -> None:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise AnalysisError(f"work ledger missing at {where}")
    for key in keys:
        a = _num(actual.get(key), f"{where}.{key}")
        e = _num(expected.get(key), f"{where}.{key}")
        if not math.isclose(a, e, rel_tol=1e-10, abs_tol=1e-7):
            raise AnalysisError(f"work counter does not reconcile at {where}.{key}")


def _validate_run(seed: int, arm: str, receipt_path: Path, config: dict[str, Any], config_sha: str,
                  sources: dict[str, str], source_root: Path) -> dict[str, Any]:
    receipt = _json(receipt_path)
    receipt_sha = sha256_file(receipt_path)
    if (receipt.get("schema") != RESULT_SCHEMA or receipt.get("status") != "complete"
            or receipt.get("seed") != seed or receipt.get("arm") != arm):
        raise AnalysisError(f"incomplete or mismatched receipt for seed {seed}/{arm}")
    if receipt.get("test_data_read") is not False:
        raise AnalysisError(f"test data flag is not false for seed {seed}/{arm}")
    if (receipt.get("official_test_read") is not False or receipt.get("P2_accessed") is not False
            or receipt.get("raw_rows_or_trajectories_written_to_json") is not False
            or receipt.get("model_immutable_versions_unchanged") is not True):
        raise AnalysisError(f"data-access or immutability guard failed for seed {seed}/{arm}")
    if receipt.get("config_sha256") != config_sha or receipt.get("source_hashes") != sources:
        raise AnalysisError(f"configuration/source lock mismatch for seed {seed}/{arm}")
    if receipt.get("training_manifest_sha256") != config.get("training_manifest_sha256"):
        raise AnalysisError(f"training-manifest mismatch for seed {seed}/{arm}")
    foundation_receipt = _hex64(receipt.get("foundation_receipt_sha256"), f"{seed}/{arm}.foundation_receipt_sha256")
    foundation_checkpoint = _hex64(receipt.get("foundation_checkpoint_sha256"), f"{seed}/{arm}.foundation_checkpoint_sha256")
    wake_digest = _hex64(receipt.get("wake_request_digest"), f"{seed}/{arm}.wake_request_digest")
    initial_wake_tick = _int(receipt.get("initial_wake_tick"), f"{seed}/{arm}.initial_wake_tick")
    final_wake_tick = _int(receipt.get("final_wake_tick"), f"{seed}/{arm}.final_wake_tick")
    expected_wake_ticks = config.get("expected_total_wake_ticks_by_seed", {}).get(str(seed))
    if type(expected_wake_ticks) is not int or final_wake_tick - initial_wake_tick != expected_wake_ticks:
        raise AnalysisError(f"terminal wake-tick horizon mismatch for seed {seed}/{arm}")
    stages = receipt.get("stages")
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        raise AnalysisError(f"stage set mismatch for seed {seed}/{arm}")
    stages_by_name = {}
    for row in stages:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or row["name"] in stages_by_name:
            raise AnalysisError(f"duplicate or malformed stage record for seed {seed}/{arm}")
        stages_by_name[row["name"]] = row
    if set(stages_by_name) != set(STAGES):
        raise AnalysisError(f"stage names differ from frozen grid for seed {seed}/{arm}")
    prefix = int(config["prefix_ticks_by_seed"][str(seed)])
    expected_ticks = {"event1_before": initial_wake_tick + prefix, "event1_after": initial_wake_tick + prefix,
                      "event2_before": initial_wake_tick + prefix + 16, "event2_after": initial_wake_tick + prefix + 16,
                      "postwake15": initial_wake_tick + prefix + 31}
    stage_rows = {}
    work = {}
    for stage in STAGES:
        row = stages_by_name[stage]
        if not isinstance(row, dict) or row.get("name") != stage:
            raise AnalysisError(f"stage metadata mismatch for seed {seed}/{arm}/{stage}")
        if row.get("evaluation_after_training") is not True:
            raise AnalysisError(f"stage evaluation timing flag is missing for seed {seed}/{arm}/{stage}")
        if row.get("global_wake_tick") != expected_ticks[stage]:
            raise AnalysisError(f"unexpected global wake tick for seed {seed}/{arm}/{stage}")
        _hex64(row.get("context_state_sha256"), f"{seed}/{arm}/{stage}.context_state_sha256")
        work[stage] = _validate_work_delta(row.get("work_delta"), f"{seed}/{arm}/{stage}")
        stage_rows[stage] = _validate_stage_eval(row.get("runtime"), stage)
    events = receipt.get("events")
    if not isinstance(events, list) or len(events) != 2:
        raise AnalysisError(f"exactly two event receipts required for seed {seed}/{arm}")
    qconfig = config.get("restore_qualification")
    q_event = None
    if isinstance(qconfig, dict) and qconfig.get("seed") == seed and qconfig.get("arm") == arm:
        q_event = qconfig.get("event_index")
    arm_cfg = next(a for a in config["arms"] if a["name"] == arm)
    requests_each = arm_cfg["requests_per_event"]
    batch_size = arm_cfg["effective_batch_size"]
    event_rows = []
    for event_pos, event in enumerate(events):
        ei = event_pos + 1
        if not isinstance(event, dict) or event.get("level") != 0 or event.get("event_index") != event_pos:
            raise AnalysisError(f"event ordering mismatch for seed {seed}/{arm}")
        expects_qualification = (event_pos == q_event)
        if event.get("qualification_extra_charged_separately") is not expects_qualification:
            raise AnalysisError(f"restore-qualification charge flag mismatch for seed {seed}/{arm}/event{ei}")
        selected = _int(event.get("selected_requests"), f"{seed}/{arm}/event{ei}.selected_requests")
        if selected != requests_each:
            raise AnalysisError(f"selected request count differs from frozen config for {seed}/{arm}/event{ei}")
        if event.get("effective_batch_size") != batch_size:
            raise AnalysisError(f"effective batch mismatch for {seed}/{arm}/event{ei}")
        digest = _hex64(event.get("selected_request_digest"), f"{seed}/{arm}/event{ei}.selected_request_digest")
        updates = _int(event.get("physical_adamw_steps"), f"{seed}/{arm}/event{ei}.physical_adamw_steps")
        max_updates = selected // batch_size if batch_size else 0
        if updates > max_updates:
            raise AnalysisError(f"physical update count exceeds full microbatches for {seed}/{arm}/event{ei}")
        objective = event.get("objective_aggregates")
        if not isinstance(objective, dict):
            raise AnalysisError(f"missing objective aggregates for {seed}/{arm}/event{ei}")
        if selected == 0:
            if objective or updates:
                raise AnalysisError(f"CMS reference must have empty objectives and zero updates for event {ei}")
            normalized_objective = {}
            event_work = _validate_work_delta(event.get("work_delta"), f"{seed}/{arm}/event{ei}")
            pre_cp = _hex64(event.get("pre_checkpoint_sha256"), f"{seed}/{arm}/event{ei}.pre_checkpoint_sha256")
            post_cp = _hex64(event.get("post_checkpoint_sha256"), f"{seed}/{arm}/event{ei}.post_checkpoint_sha256")
            event_rows.append({"event_index": event_pos, "selected_requests": selected,
                               "effective_batch_size": batch_size, "request_digest": digest,
                               "physical_adamw_steps": updates, "objective_aggregates": normalized_objective,
                               "work_delta": event_work, "pre_checkpoint_sha256": pre_cp,
                               "post_checkpoint_sha256": post_cp})
            continue
        methods = _count_map(objective.get("method_counts"), f"{seed}/{arm}/event{ei}.method_counts", selected)
        if set(methods) - {"gkd", "lti"}:
            raise AnalysisError(f"unknown objective method in {seed}/{arm}/event{ei}")
        q, rem = divmod(selected, 3)
        expected_methods = {"gkd": 2 * q + min(rem, 2), "lti": q}
        if methods != expected_methods:
            raise AnalysisError(f"scheduled GKD/LTI method counts differ for {seed}/{arm}/event{ei}")
        roles = _count_map(objective.get("role_counts"), f"{seed}/{arm}/event{ei}.role_counts", selected)
        owners = _count_map(objective.get("sampling_owner_counts"), f"{seed}/{arm}/event{ei}.sampling_owner_counts", selected)
        if set(roles) != {"old", "current"} or set(owners) != {"teacher", "student"}:
            raise AnalysisError(f"unexpected role/owner classes for {seed}/{arm}/event{ei}")
        if roles != {"old": selected // 2, "current": selected // 2}:
            raise AnalysisError(f"old/current replay roles are not balanced for {seed}/{arm}/event{ei}")
        gkd_teacher = (methods["gkd"] + 1) // 2
        gkd_student = methods["gkd"] // 2
        if owners != {"teacher": gkd_teacher + methods["lti"], "student": gkd_student}:
            raise AnalysisError(f"teacher/student sampling-owner counts do not reconcile for {seed}/{arm}/event{ei}")
        reward_count = _int(objective.get("lti_reward_count"), f"{seed}/{arm}/event{ei}.lti_reward_count")
        zero_reward_count = _int(objective.get("lti_zero_reward_count"), f"{seed}/{arm}/event{ei}.lti_zero_reward_count")
        if reward_count != methods["lti"] or zero_reward_count > reward_count:
            raise AnalysisError(f"LTI reward counters do not reconcile for {seed}/{arm}/event{ei}")
        reward_sum = _num(objective.get("lti_reward_sum"), f"{seed}/{arm}/event{ei}.lti_reward_sum")
        if reward_count == 0 and reward_sum != 0:
            raise AnalysisError(f"nonzero reward sum with no rewards for {seed}/{arm}/event{ei}")
        loss_mean = objective.get("loss_mean")
        if selected == 0:
            if loss_mean is not None:
                loss_mean = _num(loss_mean, f"{seed}/{arm}/event{ei}.loss_mean")
        else:
            loss_mean = _num(loss_mean, f"{seed}/{arm}/event{ei}.loss_mean")
        backward_calls = _int(objective.get("backward_calls"), f"{seed}/{arm}/event{ei}.backward_calls")
        if not updates <= backward_calls <= selected:
            raise AnalysisError(f"backward-call count is outside work bounds for {seed}/{arm}/event{ei}")
        skipped = _int(objective.get("skipped_exact_zero_batches"), f"{seed}/{arm}/event{ei}.skipped_exact_zero_batches")
        expected_batches = selected // batch_size if batch_size else 0
        if updates + skipped != expected_batches:
            raise AnalysisError(f"physical updates and exact-zero skips do not reconcile for {seed}/{arm}/event{ei}")
        if objective.get("physical_adamw_steps") != updates:
            raise AnalysisError(f"objective/event physical update count differs for {seed}/{arm}/event{ei}")
        # Keep a normalized aggregate-only copy; receipts are forbidden from
        # carrying request-level material in this field.
        objective = {"request_count": _int(objective.get("request_count"), f"{seed}/{arm}/event{ei}.objective.request_count"),
                     "method_counts": methods,
                     "role_counts": roles, "sampling_owner_counts": owners,
                     "lti_reward_count": reward_count, "lti_reward_sum": reward_sum,
                     "lti_zero_reward_count": zero_reward_count, "loss_mean": loss_mean,
                     "backward_calls": backward_calls, "physical_adamw_steps": updates,
                     "skipped_exact_zero_batches": skipped}
        if objective["request_count"] != selected:
            raise AnalysisError(f"objective request count mismatch for {seed}/{arm}/event{ei}")
        event_work = _validate_work_delta(event.get("work_delta"), f"{seed}/{arm}/event{ei}")
        pre_cp = _hex64(event.get("pre_checkpoint_sha256"), f"{seed}/{arm}/event{ei}.pre_checkpoint_sha256")
        post_cp = _hex64(event.get("post_checkpoint_sha256"), f"{seed}/{arm}/event{ei}.post_checkpoint_sha256")
        if event_work.get("physical_adamw_updates") != updates:
            raise AnalysisError(f"event work and objective updates differ for {seed}/{arm}/event{ei}")
        expected_sleep_events = 1 if selected else 0
        if event_work.get("sleep_events") != expected_sleep_events:
            raise AnalysisError(f"event sleep counter mismatch for {seed}/{arm}/event{ei}")
        event_rows.append({"event_index": event_pos, "selected_requests": selected,
                           "effective_batch_size": batch_size, "request_digest": digest,
                           "physical_adamw_steps": updates, "objective_aggregates": objective,
                           "work_delta": event_work, "pre_checkpoint_sha256": pre_cp,
                           "post_checkpoint_sha256": post_cp})
    work_phases = receipt.get("work_phases")
    if not isinstance(work_phases, dict):
        raise AnalysisError(f"work phases missing for seed {seed}/{arm}")
    wake_work = _validate_work_delta(work_phases.get("wake"), f"{seed}/{arm}.work_phases.wake")
    eval_work = _validate_work_delta(work_phases.get("evaluation"), f"{seed}/{arm}.work_phases.evaluation")
    stage_work_sum = _sum_deltas(list(work.values()))
    eval_keys = tuple(sorted(set(eval_work) | set(stage_work_sum)))
    _check_work_close(stage_work_sum, eval_work, eval_keys, f"{seed}/{arm}.evaluation")
    qualification = receipt.get("restore_qualification")
    if qualification is None:
        qualification_work = {}
    elif isinstance(qualification, dict):
        if q_event is None or qualification.get("status") != "pass":
            raise AnalysisError(f"unexpected or failed restore qualification for seed {seed}/{arm}")
        qualification_work = _validate_work_delta(qualification.get("extra_work_delta"),
                                                   f"{seed}/{arm}.restore_qualification.extra_work_delta")
    else:
        raise AnalysisError(f"restore qualification record is malformed for seed {seed}/{arm}")
    if q_event is not None and qualification is None:
        raise AnalysisError(f"required restore qualification is absent for seed {seed}/{arm}")
    budget = receipt.get("budget")
    if not isinstance(budget, dict):
        raise AnalysisError(f"total budget counters missing for seed {seed}/{arm}")
    additive_parts = [wake_work, eval_work, *[e["work_delta"] for e in event_rows]]
    if qualification_work:
        additive_parts.append(qualification_work)
    total_parts = _sum_deltas(additive_parts)
    _check_work_close(total_parts, budget, ADDITIVE_WORK_COUNTERS, f"{seed}/{arm}.total_budget")
    expected_wake = expected_wake_ticks
    if wake_work.get("wake_updates") != expected_wake:
        raise AnalysisError(f"wake work count differs from frozen horizon for seed {seed}/{arm}")
    return {"seed": seed, "arm": arm, "receipt_sha256": receipt_sha,
            "foundation_receipt_sha256": foundation_receipt,
            "foundation_checkpoint_sha256": foundation_checkpoint,
            "wake_request_digest": wake_digest, "initial_wake_tick": initial_wake_tick,
            "final_wake_tick": final_wake_tick,
            "stages": stage_rows, "stage_work": work, "events": event_rows,
            "stage_context_hashes": {name: stages_by_name[name]["context_state_sha256"] for name in STAGES},
            "work_phases": {"wake": wake_work, "evaluation": eval_work},
            "restore_qualification_work": qualification_work,
            "budget": {k: _int(budget.get(k), f"{seed}/{arm}.budget.{k}")
                       for k in ADDITIVE_WORK_COUNTERS}}


def analyze(run_root: Path, config_path: Path, source_root: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise AnalysisError("analysis output directory must be fresh")
    config = _json(config_path)
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") != "frozen"
            or config.get("seeds") != list(SEEDS)):
        raise AnalysisError("analysis requires the frozen V13 three-seed config")
    if (config.get("parent_release") != PARENT_RELEASE or config.get("task_id") != 1
            or config.get("sender") != 0 or config.get("sleep_events_per_run") != 2
            or config.get("wake_ticks_between_sleep_events") != 16
            or config.get("wake_ticks_after_last_sleep") != 15
            or config.get("periods") != [16, 64, 256, 1024, None]
            or config.get("prefix_ticks_by_seed") != PREFIX_TICKS
            or config.get("expected_total_wake_ticks_by_seed") != TOTAL_TICKS
            or config.get("test_data_read") is not False
            or config.get("no_test_based_selection") is not True
            or config.get("execution_order_per_seed") != ["B32_HIGH", "CMS_ONLY", "B1_HIGH", "B1_LOW", "B32_LOW", "B32_LOW_DOSE"]
            or config.get("sleep_method_cycle") != ["gkd", "gkd", "lti"]
            or config.get("sleep_selector_event_stride") != 3200
            or config.get("gkd_sampling_owner_policy") != "alternate_teacher_student_start_teacher_each_event"):
        raise AnalysisError("frozen V13 protocol fields differ from the reviewed design")
    expected_evaluation = {"split": "development_only", "old_rows": 110, "current_rows": 110,
                           "candidate_labels": 77, "candidate_batch_size": 32,
                           "primary": "teacher_forced_sum", "include_greedy": True,
                           "stages": list(STAGES)}
    if config.get("evaluation") != expected_evaluation:
        raise AnalysisError("frozen V13 evaluation contract differs from reviewed denominators")
    expected_qualification = {"seed": 17, "arm": "B32_HIGH", "event_index": 0,
                              "snapshot_after_requests": 32, "compare_after_requests": 64,
                              "extra_request_evaluations": 32, "extra_adamw_updates": 1,
                              "require_before_other_trajectories": True,
                              "whole_job_resume_claim": False}
    if config.get("restore_qualification") != expected_qualification:
        raise AnalysisError("frozen V13 restore qualification scope differs from review")
    _hex64(config.get("v10_source_lock_sha256"), "config.v10_source_lock_sha256")
    _hex64(config.get("training_manifest_sha256"), "config.training_manifest_sha256")
    config_sha = sha256_file(config_path)
    sources = config.get("source_hashes")
    if not isinstance(sources, dict) or not sources:
        raise AnalysisError("frozen V13 source hash map missing")
    for rel, expected in sources.items():
        _hex64(expected, f"source_hashes.{rel}")
        source_path = (source_root / rel).resolve(strict=True)
        if not source_path.is_relative_to(source_root.resolve(strict=True)):
            raise AnalysisError(f"source path escapes project root: {rel}")
        if sha256_file(source_path) != expected:
            raise AnalysisError(f"frozen source hash mismatch: {rel}")
    arms = config.get("arms")
    if not isinstance(arms, list) or {a.get("name") for a in arms if isinstance(a, dict)} != set(ARMS):
        raise AnalysisError("V13 six-arm grid mismatch")
    arm_config = {a["name"]: a for a in arms}
    for name, (requests, batch, lr) in EXPECTED_ARMS.items():
        arm = arm_config[name]
        if (arm.get("requests_per_event") != requests or arm.get("effective_batch_size") != batch
                or arm.get("learning_rate") != lr or arm.get("sleep") is not (requests > 0)):
            raise AnalysisError(f"frozen arm definition mismatch for {name}")
    per_run = []
    for seed in SEEDS:
        for arm in ARMS:
            receipt = run_root / f"seed-{seed}" / arm / "final-receipt.json"
            per_run.append(_validate_run(seed, arm, receipt, config, config_sha, sources, source_root))
    by = {(r["seed"], r["arm"]): r for r in per_run}
    # All six runs for a seed must use the same qualified foundation state.
    for seed in SEEDS:
        foundations = {(by[(seed, arm)]["foundation_receipt_sha256"],
                        by[(seed, arm)]["foundation_checkpoint_sha256"]) for arm in ARMS}
        if len(foundations) != 1:
            raise AnalysisError(f"paired runs do not share a foundation for seed {seed}")
        if len({by[(seed, arm)]["wake_request_digest"] for arm in ARMS}) != 1:
            raise AnalysisError(f"paired runs do not share a wake request schedule for seed {seed}")
        if len({(by[(seed, arm)]["initial_wake_tick"], by[(seed, arm)]["final_wake_tick"])
                for arm in ARMS}) != 1:
            raise AnalysisError(f"paired runs have different wake horizons for seed {seed}")
        baseline_run = by[(seed, ARMS[0])]
        for arm in ARMS[1:]:
            candidate_run = by[(seed, arm)]
            if candidate_run["stage_context_hashes"]["event1_before"] != baseline_run["stage_context_hashes"]["event1_before"]:
                raise AnalysisError(f"pre-Sleep contexts differ across arms for seed {seed}")
            if not _same_aggregate(candidate_run["stages"]["event1_before"], baseline_run["stages"]["event1_before"]):
                raise AnalysisError(f"pre-Sleep development metrics differ across arms for seed {seed}")
        # The 2x2 batch/LR comparison is paired on selected prompt identities.
        # This verifies request exposure, not identical model-generated targets.
        matched_arms = ("B1_HIGH", "B1_LOW", "B32_HIGH", "B32_LOW")
        for event_index in (1, 2):
            digests = {by[(seed, arm)]["events"][event_index - 1]["request_digest"]
                       for arm in matched_arms}
            if len(digests) != 1:
                raise AnalysisError(f"matched 2x2 request cohort differs for seed {seed}, event {event_index}")
    rows = []
    for r in per_run:
        for stage in STAGES:
            for metric in METRICS:
                pops = POPS if metric != "greedy" else ("all77",)
                for pop in pops:
                    value = r["stages"][stage][metric][pop]
                    rows.append({"seed": r["seed"], "arm": r["arm"], "stage": stage,
                                 "metric": metric, "population": pop, **value})
    index = {(r["seed"], r["arm"], r["stage"], r["metric"], r["population"]): r for r in rows}
    summaries = {}
    for arm in ARMS:
        for stage in STAGES:
            for metric in METRICS:
                pops = POPS if metric != "greedy" else ("all77",)
                for pop in pops:
                    rs = [index[(seed, arm, stage, metric, pop)] for seed in SEEDS]
                    field = "mean_token_nll" if metric == "gold_label_token_mean_nll" else "accuracy"
                    summaries[f"{arm}|{stage}|{metric}|{pop}"] = {
                        field: _summary([float(x[field]) for x in rs])}
    contrast_rows = []
    for left, right, description in CONTRASTS:
        for stage in STAGES:
            for metric in METRICS:
                for pop in (POPS if metric != "greedy" else ("all77",)):
                    field = "mean_token_nll" if metric == "gold_label_token_mean_nll" else "accuracy"
                    diffs = [float(index[(seed, left, stage, metric, pop)][field]) -
                             float(index[(seed, right, stage, metric, pop)][field]) for seed in SEEDS]
                    contrast_rows.append({"left": left, "right": right, "description": description,
                        "stage": stage, "metric": metric, "population": pop,
                        "left_minus_right": _summary(diffs)})
    cost = []
    for r in per_run:
        arm_cfg = arm_config[r["arm"]]
        expected_requests = int(arm_cfg["requests_per_event"]) * 2
        actual_requests = sum(e["selected_requests"] for e in r["events"])
        expected_adam = (expected_requests // int(arm_cfg["effective_batch_size"])) if expected_requests else 0
        actual_adam = sum(e["physical_adamw_steps"] for e in r["events"])
        qualification_adam = _int(r["restore_qualification_work"].get("physical_adamw_updates", 0),
                                  f"{r['seed']}/{r['arm']}.qualification.physical_adamw_updates")
        if actual_requests != expected_requests or actual_adam > expected_adam:
            raise AnalysisError(f"event exposure/update budget mismatch for {r['seed']}/{r['arm']}")
        if actual_adam + qualification_adam != r["budget"]["physical_adamw_updates"]:
            raise AnalysisError(f"event plus qualification update ledger mismatch for {r['seed']}/{r['arm']}")
        cost.append({"seed": r["seed"], "arm": r["arm"], "requests_expected": expected_requests,
                     "requests_recorded": actual_requests, "adam_updates_expected_if_no_skips": expected_adam,
                     "physical_adamw_steps_recorded": actual_adam,
                     "restore_qualification_physical_adamw_updates": qualification_adam,
                     "all_physical_adamw_updates_including_qualification": actual_adam + qualification_adam,
                     "zero_gradient_or_skipped_updates": expected_adam - actual_adam,
                     "wake_work": r["work_phases"]["wake"],
                     "evaluation_work": r["work_phases"]["evaluation"],
                     "event_work": [e["work_delta"] for e in r["events"]],
                     "restore_qualification_work": r["restore_qualification_work"],
                     "reconciled_budget_work": r["budget"]})
    summary = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete_descriptive_development_analysis",
        "design": {"seeds": list(SEEDS), "arms": list(ARMS), "stages": list(STAGES),
                   "populations": list(POPS), "candidate_labels": 77,
                   "primary": "teacher_forced_sum", "statistics": "mean and sample SD over three seeded class orders"},
        "provenance": {"aggregate_tool_sha256": sha256_file(Path(__file__).resolve()),
            "config_sha256": config_sha, "source_hashes": sources,
            "training_manifest_sha256": config["training_manifest_sha256"],
            "runs": [{k: r[k] for k in ("seed", "arm", "receipt_sha256", "foundation_receipt_sha256", "foundation_checkpoint_sha256", "wake_request_digest")} for r in per_run]},
        "metrics_by_arm_stage_family_population": summaries,
        "paired_contrasts": contrast_rows,
        "costs": cost,
        "interpretation_limits": [
            "three seeds are class-order variants sharing one initialization/foundation lineage, not independent pretraining runs",
            "the 2x2 comparisons match selected prompt requests but not necessarily physical Adam updates or FLOPs",
            "CMS_ONLY is a reference and is not matched to Sleep in information, request, or optimizer compute",
            "B32_LOW_DOSE changes request dose and optimizer updates; it is not a pure batch-size contrast",
            "zero-gradient batches can create opportunities without a committed AdamW update",
            "local canonical-label semantic reward is a proxy, not the paper's frozen reward model; this is not a paper reproduction",
        ],
        "test_data_read": False,
        "raw_rows_or_ids_emitted": False,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    json_path = out_dir / "v13-lifecycle-aggregate.json"
    csv_path = out_dir / "v13-lifecycle-metrics.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    columns = ["seed", "arm", "stage", "metric", "population", "count", "correct", "invalid_predictions", "accuracy", "mean_token_nll"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"aggregate_path": json_path, "aggregate_sha256": sha256_file(json_path),
            "metrics_csv_path": csv_path, "metrics_csv_sha256": sha256_file(csv_path), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.run_root, args.config, args.source_root, args.out_dir)
    print(json.dumps({"aggregate_sha256": result["aggregate_sha256"],
                      "metrics_csv_sha256": result["metrics_csv_sha256"],
                      "status": result["summary"]["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
