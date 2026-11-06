"""Offline, read-only aggregation for the frozen AG News v6 routing study.

Consumes only plans, seals, aggregate score JSON, work ledgers, and the
storage audit. It never imports Torch or reads model/data/test files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

SEEDS = (17, 29, 43)
CONDITIONS = (
    "warm_alone", "new_CMS_reference_alone", "correct_phase_ridge",
    "permuted_phase_ridge", "hash_random", "confidence_marker",
    "confidence_selfjoint", "task_ID_DIAGNOSTIC", "uniform_context_mix",
)
PHYSICAL = (
    "constant_old", "constant_new", "correct_phase_ridge",
    "permuted_phase_ridge", "hash_random", "uniform_context_mix",
)
POSTHOC_TO_PHYSICAL = {
    "warm_alone": "constant_old", "new_CMS_reference_alone": "constant_new",
    "correct_phase_ridge": "correct_phase_ridge",
    "permuted_phase_ridge": "permuted_phase_ridge", "hash_random": "hash_random",
}
METRICS = (
    "full_vocab_two_token_greedy_exact_match", "marker_accuracy",
    "class_given_marker_accuracy", "candidate_accuracy", "marker_nll",
    "class_nll", "answer_nll_nats_per_token",
)
GROUPS = ("all", "old", "new", "class0", "class1", "class2", "class3")
WORK_FIELDS = ("forward_calls", "forward_input_tokens", "gradient_forward_input_tokens")
EXPECTED_A_SOURCE = "7cbfe600399f992533fcd5c81586a3c1a4ecef7744c24fa7706108e94fcb4752"
EXPECTED_B_SOURCE = "1bdea90f357e595fbaa1461d71106b75ee11c0df56fbcfc5e869a89f8c5fa1be"
EXPECTED_B_HELPER = "442e4175b58b5103f2cde9d7047ff5ef807026de9059b0cadd6e5f4cd55488e2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifact(arg: Path, filename: str) -> Path:
    return arg / filename if arg.is_dir() else arg


def mean_sd(values: list[float]) -> dict[str, Any]:
    vals = [float(x) for x in values]
    if not vals:
        raise ValueError("cannot summarize an empty sample")
    return {"n": len(vals), "mean": statistics.fmean(vals),
            "sample_sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "values": vals}


def sum_work(work: dict[str, Any]) -> dict[str, int]:
    totals = {key: 0 for key in WORK_FIELDS}
    for stage, counts in work.items():
        if not isinstance(counts, dict):
            raise ValueError(f"work stage {stage!r} is not an object")
        for key in WORK_FIELDS:
            value = counts.get(key)
            if value is None:
                raise ValueError(f"work stage {stage!r} lacks required counter {key}")
            if not isinstance(value, (int, float)) or int(value) != value or value < 0:
                raise ValueError(f"invalid work counter {stage!r}/{key}: {value!r}")
            totals[key] += int(value)
    return totals


def merge_work_ledgers(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sum per-source and same-named stages; never union-overwrite stages."""
    by_source = {}
    by_stage: dict[str, dict[str, int]] = {}
    grand = {key: 0 for key in WORK_FIELDS}
    for source, ledger in sources.items():
        source_total = sum_work(ledger)
        by_source[source] = {"stage_count": len(ledger), **source_total}
        for key, value in source_total.items():
            grand[key] += value
        for stage, counters in ledger.items():
            row = by_stage.setdefault(stage, {key: 0 for key in WORK_FIELDS})
            for key in WORK_FIELDS:
                row[key] += int(counters[key])
    return {"fields": list(WORK_FIELDS), "by_source": by_source,
            "combined_by_stage": by_stage, "combined_total": grand}


def metric_at(metrics: dict[str, Any], group: str, metric: str) -> float:
    try:
        value = metrics[group][metric]
        examples = metrics[group]["examples"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing score cell {group}/{metric}") from exc
    if not isinstance(examples, int) or examples <= 0:
        raise ValueError(f"invalid denominator for {group}: {examples!r}")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"invalid score for {group}/{metric}: {value!r}")
    return float(value)


def validate_metrics(metrics: dict[str, Any], *, total: int, old: int, new: int,
                     classes: int) -> None:
    expected = {"all": total, "old": old, "new": new,
                **{f"class{i}": classes for i in range(4)}}
    if set(metrics) != set(expected):
        raise ValueError(f"score groups differ: got {sorted(metrics)}, expected {sorted(expected)}")
    for group, count in expected.items():
        if metrics[group].get("examples") != count:
            raise ValueError(f"{group} denominator {metrics[group].get('examples')} != {count}")
        for key in METRICS:
            metric_at(metrics, group, key)


def split_sizes(split: str) -> tuple[int, int, int, int]:
    if split == "train":
        return 1024, 512, 512, 256
    if split == "validation":
        return 512, 256, 256, 128
    raise ValueError(f"unexpected score split: {split}")


def equal_metrics(left: dict[str, Any], right: dict[str, Any], label: str,
                  *, split: str) -> None:
    sizes = (1024, 512, 512, 256) if split == "train" else (512, 256, 256, 128)
    validate_metrics(left, total=sizes[0], old=sizes[1], new=sizes[2], classes=sizes[3])
    validate_metrics(right, total=sizes[0], old=sizes[1], new=sizes[2], classes=sizes[3])
    for group in GROUPS:
        for metric in METRICS:
            a, b = metric_at(left, group, metric), metric_at(right, group, metric)
            tol = 1e-6 if metric.endswith("nll") or "nats" in metric else 1e-12
            if abs(a - b) > tol:
                raise ValueError(f"{label}: {group}/{metric} differs {a} vs {b}")


def extract_score(row: dict[str, Any], condition: str, split: str) -> dict[str, Any]:
    if condition in row.get("posthoc", {}):
        return row["posthoc"][condition][split]
    if condition == "uniform_context_mix":
        return row["physical_shared"][f"{split}:uniform_context_mix"]["metrics"]
    raise ValueError(f"missing condition {condition!r} in row {row.get('seed')}")


def validate_a_result(protocol: dict[str, Any], result: dict[str, Any], phase: str) -> None:
    if protocol.get("schema") != "agnews-routing-fairness-protocol-v6" or protocol.get("phase") != phase:
        raise ValueError(f"unexpected A protocol schema/phase for {phase}")
    if result.get("schema") != "agnews-routing-fairness-result-v6" or result.get("phase") != phase:
        raise ValueError(f"unexpected A result schema/phase for {phase}")
    if result.get("complete") is not True or result.get("confirmation_scored") is not False:
        raise ValueError(f"A {phase} result is incomplete or scored confirmation data")
    if result.get("actual_optimizer_updates") != 0 or result.get("no_holdout_input") is not True:
        raise ValueError(f"A {phase} result violates read-only/no-holdout contract")
    if protocol.get("confirmation_input") is not False or protocol.get("confirmation_scored") is not False:
        raise ValueError(f"A {phase} plan permits confirmation data")
    source = protocol.get("source_sha256", {})
    if source.get("driver_sha256") != EXPECTED_A_SOURCE:
        raise ValueError(f"A {phase} driver source hash differs from reviewed source")
    if result.get("source_sha256") != source:
        raise ValueError(f"A {phase} result source lock differs from its plan")
    if result.get("protocol_sha256") != protocol.get("_file_sha256"):
        raise ValueError(f"A {phase} result does not bind its protocol file")
    if protocol.get("elapsed_limit_seconds") != 1800:
        raise ValueError(f"A {phase} elapsed cap differs from reviewed protocol")
    if protocol.get("training_or_optimizer_updates") != 0:
        raise ValueError(f"A {phase} plan permits updates")
    if (protocol.get("unique_phase_labels") != 1024 or
            protocol.get("phase_label_uses_in_two_gate_fits") != 2048 or
            protocol.get("ridge_fit_count") != (2 if phase == "initial" else 0) or
            "additional task-identity supervision" not in protocol.get("selector_supervision_note", "") or
            "phase labels only" not in protocol.get("ridge_fit_definition", "")):
        raise ValueError(f"A {phase} task-identity supervision/gate-fit contract differs")


def validate_rows(rows: list[dict[str, Any]], *, expected_count: int, allowed_contexts: set[str]) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} A rows, got {len(rows)}")
    keys = [(row.get("seed"), row.get("new_context")) for row in rows]
    expected = {(seed, context) for seed in SEEDS for context in allowed_contexts}
    if set(keys) != expected or len(set(keys)) != len(keys):
        raise ValueError(f"A rows are missing, duplicated, or unexpected: {keys}")
    for row in rows:
        if row.get("seed") not in SEEDS:
            raise ValueError(f"unexpected seed {row.get('seed')}")
        if set(row.get("posthoc", {})) != set(CONDITIONS[:-1]):
            raise ValueError("A posthoc condition set differs from frozen source")
        if set(row.get("physical_shared", {})) != {f"{split}:{condition}"
                for split in ("train", "validation") for condition in PHYSICAL}:
            raise ValueError("A physical condition set differs from frozen source")
        if set(row.get("posthoc", {}) ) != set(CONDITIONS[:-1]):
            raise ValueError("unexpected posthoc selector list")
        for condition in CONDITIONS[:-1]:
            for split in ("train", "validation"):
                metrics = row["posthoc"][condition][split]
                sizes = split_sizes(split)
                validate_metrics(metrics, total=sizes[0], old=sizes[1], new=sizes[2], classes=sizes[3])
        for condition in PHYSICAL:
            for split in ("train", "validation"):
                physical = row["physical_shared"][f"{split}:{condition}"]
                sizes = split_sizes(split)
                validate_metrics(physical["metrics"], total=sizes[0], old=sizes[1], new=sizes[2], classes=sizes[3])
                cost = physical.get("cost", {})
                for key in ("wall_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                            "full_forward_calls", "batches", "unique_context_payload_bytes_excluding_shared_backbone"):
                    if key not in cost or not isinstance(cost[key], (int, float)) or cost[key] < 0:
                        raise ValueError(f"missing/invalid physical cost {key}")
                ops = cost.get("operation_counts", {})
                if not ops:
                    raise ValueError("physical row lacks branch operation counts")
                expected_batches = math.ceil(split_sizes(split)[0] / 8)
                if cost["batches"] != expected_batches or cost["full_forward_calls"] != expected_batches:
                    raise ValueError(f"unexpected physical scoring forward count for {split}/{condition}")
                for op in ("original_final_mlp_calls", "original_lm_head_calls",
                           "old_memory_branch_calls", "new_memory_branch_calls",
                           "old_readout_adapter_calls", "new_readout_adapter_calls"):
                    if ops.get(op) != expected_batches:
                        raise ValueError(f"physical operation {op} count differs for {split}/{condition}")
        for condition, physical_name in POSTHOC_TO_PHYSICAL.items():
            for split in ("train", "validation"):
                equal_metrics(row["posthoc"][condition][split],
                              row["physical_shared"][f"{split}:{physical_name}"]["metrics"],
                              f"physical/posthoc parity {condition}/{split}", split=split)
        for split, key in (("train", "new_branch_train"), ("validation", "new_branch_validation")):
            sizes = split_sizes(split)
            validate_metrics(row[key], total=sizes[0], old=sizes[1], new=sizes[2], classes=sizes[3])


def gate_status(train: dict[str, Any], val: dict[str, Any]) -> dict[str, Any]:
    per_task = {}
    for phase, classes in (("old", (0, 1)), ("new", (2, 3))):
        train_em = metric_at(train, phase, METRICS[0])
        val_em = metric_at(val, phase, METRICS[0])
        class_em = {str(k): metric_at(val, f"class{k}", METRICS[0]) for k in classes}
        passing = train_em >= .80 and val_em >= .75 and all(x >= .70 for x in class_em.values())
        per_task[phase] = {"train_em": train_em, "train_threshold": .80,
            "train_pass": train_em >= .80, "validation_em": val_em,
            "validation_threshold": .75, "validation_pass": val_em >= .75,
            "class_validation_em": class_em, "class_threshold": .70,
            "class_pass": {k: v >= .70 for k, v in class_em.items()}, "pass": passing}
    return {"tasks": per_task, "all_tasks_pass": all(x["pass"] for x in per_task.values())}


def metrics_summary(rows: list[dict[str, Any]], condition: str, split: str) -> dict[str, Any]:
    out = {}
    for group in GROUPS:
        out[group] = {"examples_per_seed": None}
        for metric in METRICS:
            out[group][metric] = mean_sd([metric_at(extract_score(r, condition, split), group, metric)
                                           for r in rows])
        out[group]["examples_per_seed"] = extract_score(rows[0], condition, split)[group]["examples"]
    return out


def build_condition_tables(initial_rows: list[dict[str, Any]], endpoint_rows: list[dict[str, Any]]) -> dict[str, Any]:
    initial = {}
    for condition in CONDITIONS:
        for split in ("train", "validation"):
            initial[f"{condition}:{split}"] = metrics_summary(initial_rows, condition, split)
    endpoint = {}
    for arm in ("aux_new_CE", "aux_old_gold_CE"):
        arm_rows = [r for r in endpoint_rows if r["new_context"] == arm]
        for condition in CONDITIONS:
            for split in ("train", "validation"):
                endpoint[f"{arm}:{condition}:{split}"] = metrics_summary(arm_rows, condition, split)
    return {"initial": initial, "endpoints": endpoint}


def paired_differences(rows: list[dict[str, Any]], left_condition: str, right_condition: str,
                       split: str, *, arm: str | None = None) -> dict[str, Any]:
    selected = [r for r in rows if arm is None or r["new_context"] == arm]
    by_seed = {r["seed"]: r for r in selected}
    if set(by_seed) != set(SEEDS):
        raise ValueError("paired comparison must contain exactly one row per seed")
    differences = {}
    for group in GROUPS:
        differences[group] = {}
        for metric in METRICS:
            vals = []
            for seed in SEEDS:
                a = metric_at(extract_score(by_seed[seed], left_condition, split), group, metric)
                b = metric_at(extract_score(by_seed[seed], right_condition, split), group, metric)
                vals.append(a - b)
            differences[group][metric] = mean_sd(vals)
    return {"contrast": f"{left_condition} minus {right_condition}", "split": split,
            "arm": arm, "paired_by_seed": differences, "no_p_value": True}


def flatten_work_prefix(work: dict[str, Any], predicate) -> dict[str, int]:
    return sum_work({k: v for k, v in work.items() if predicate(k)})


def check_work_expectations(qualification: dict[str, Any], bseal: dict[str, Any],
                           initial: dict[str, Any], endpoints: dict[str, Any]) -> dict[str, Any]:
    qwork = qualification.get("work", {})
    bwork = bseal.get("work", {})
    iwork = initial.get("work", {})
    ework = endpoints.get("work", {})
    qtot = sum_work(qwork)
    btot = sum_work(bwork)
    itot = sum_work(iwork)
    etot = sum_work(ework)
    b_train = flatten_work_prefix(bwork, lambda s: s.endswith("_primary"))
    b_val = flatten_work_prefix(bwork, lambda s: s.startswith("v6_validation_"))
    b_train_end = flatten_work_prefix(bwork, lambda s: s.startswith("v6_train_"))
    if b_train["forward_calls"] != 6912 or b_val["forward_calls"] != 1152 or b_train_end["forward_calls"] != 768:
        raise ValueError(f"B work stage calls differ from source budget: train={b_train}, val={b_val}, end={b_train_end}")
    if qtot["forward_calls"] != 24:
        raise ValueError(f"qualification should record 24 actual forward calls, got {qtot['forward_calls']}")
    if itot["forward_calls"] != 4612 or etot["forward_calls"] != 8064:
        raise ValueError(f"A forward counts differ from source budget: initial={itot}, endpoints={etot}")
    combined = qtot["forward_calls"] + btot["forward_calls"] + itot["forward_calls"] + etot["forward_calls"]
    if btot["forward_calls"] != 8832 or combined != 21532:
        raise ValueError(f"grand forward calls differ from source budget: B={btot}, grand={combined}")
    if qtot["gradient_forward_input_tokens"] != qtot["forward_input_tokens"]:
        raise ValueError("qualification gradient token proxy differs from all-forward tokens")
    if b_train["gradient_forward_input_tokens"] != b_train["forward_input_tokens"]:
        raise ValueError("B training gradient token proxy differs from primary training forward tokens")
    if b_val["gradient_forward_input_tokens"] or b_train_end["gradient_forward_input_tokens"]:
        raise ValueError("B scoring work unexpectedly has gradient-forward tokens")
    if itot["gradient_forward_input_tokens"] or etot["gradient_forward_input_tokens"]:
        raise ValueError("A inference work unexpectedly has gradient-forward tokens")
    q_updates = qualification.get("actual_primary_updates")
    b_updates = bseal.get("actual_primary_updates")
    if q_updates != 12 or b_updates != 3456:
        raise ValueError(f"update receipts differ from contract: qualification={q_updates}, Bmain={b_updates}")
    if bseal.get("actual_teacher_forward_passes") != 0 or qualification.get("actual_teacher_forward_passes") != 0:
        raise ValueError("a teacher forward was reported in gold-only continuation")
    if bseal.get("actual_student_forward_passes") != 6912 or bseal.get("actual_student_backward_passes") != 6912:
        raise ValueError("B main student pass counts differ from frozen contract")
    if bseal.get("actual_primary_updates") != 2 * 3 * 576:
        raise ValueError("B main primary-update count differs from frozen contract")
    grad_calls = qualification.get("actual_student_forward_passes", 0) + bseal.get("actual_student_forward_passes", 0)
    return {"qualification": qtot, "B_main": {"total": btot, "training": b_train,
            "validation_512_at_512_768_1024": b_val, "train_endpoint": b_train_end},
            "A_initial": itot, "A_continuation": etot,
            "combined_totals": {key: qtot[key] + btot[key] + itot[key] + etot[key] for key in WORK_FIELDS},
            "source_derived_full_forward_calls_expected": 21532,
            "actual_optimizer_update_calls": {"qualification_replay_check": q_updates,
                                                "B_main_scientific": b_updates,
                                                "combined": q_updates + b_updates},
            "contract_gradient_forward_calls_derived_from_pass_receipts": grad_calls,
            "contract_gradient_forward_calls_expected": 6936,
            "limitations": ["forward-input-token counts are a proxy, not FLOPs or GPU busy time",
                            "forward_calls do not establish arbitrary autoregressive generation latency"]}


def validate_b_main(protocol: dict[str, Any], seal: dict[str, Any], qualification: dict[str, Any],
                    protocol_sha: str, qualification_sha: str) -> None:
    if protocol.get("schema") != "agnews-fair-continuation-protocol-v6" or seal.get("schema") != "agnews-fair-continuation-development-seal-v6":
        raise ValueError("unexpected B protocol/seal schema")
    if seal.get("complete") is not True or not seal.get("development_only"):
        raise ValueError("B main is not completely sealed development-only work")
    if seal.get("confirmation_scored") is not False or seal.get("no_holdout_input") is not True:
        raise ValueError("B seal indicates confirmation/test access")
    if seal.get("protocol_sha256") != protocol_sha:
        raise ValueError("B seal does not bind B protocol")
    if seal.get("source_sha256") != protocol.get("source_sha256"):
        raise ValueError("B seal source lock differs from B protocol")
    if protocol.get("qualification_sha256") != qualification_sha:
        raise ValueError("B protocol does not bind supplied qualification receipt")
    if qualification.get("passed") is not True or qualification.get("scoring") is not False:
        raise ValueError("B qualification is absent/failed or includes scoring")
    if protocol.get("actual_optimizer_updates") not in (None, 0):
        raise ValueError("B protocol reports nonzero optimizer updates in plan")
    if protocol.get("planned_primary_updates") != 3456:
        raise ValueError("B planned updates differ from frozen budget")
    if protocol.get("source_sha256", {}).get("driver_source_sha256") != EXPECTED_B_SOURCE:
        raise ValueError("B driver source differs from reviewed source")
    if protocol.get("source_sha256", {}).get("continuation_helper_sha256") != EXPECTED_B_HELPER:
        raise ValueError("B helper source differs from reviewed source")
    rows = seal.get("runs", [])
    expected = {(s, arm) for s in SEEDS for arm in ("aux_new_CE", "aux_old_gold_CE")}
    keyed = {(r.get("seed"), r.get("arm")): r for r in rows}
    if len(rows) != 6 or set(keyed) != expected:
        raise ValueError("B seal must contain exactly all six seed/arm runs")
    for seed, arm in expected:
        row = keyed[(seed, arm)]
        if row.get("status") != "complete" or row.get("primary_updates") != 576:
            raise ValueError(f"B run {seed}/{arm} incomplete or wrong update count")
        if row.get("start_step") != 448 or row.get("train_final", {}).get("all", {}).get("examples") != 1024:
            raise ValueError(f"B run {seed}/{arm} has unexpected start/end data")
        val = {int(x["primary_step"]): x["metrics"] for x in row.get("validation", [])}
        if set(val) != {448, 512, 768, 1024}:
            raise ValueError(f"B run {seed}/{arm} validation trajectory incomplete")
        for step, metrics in val.items():
            # Every validation score is over the same 512 held-development rows;
            # step 448 is inherited from v5 and incurs no new forward work.
            expected_n, expected_old_new, expected_class = 512, 256, 128
            validate_metrics(metrics, total=expected_n, old=expected_old_new,
                             new=expected_old_new, classes=expected_class)


def validate_a_pair(initial_protocol: dict[str, Any], initial_result: dict[str, Any],
                    end_protocol: dict[str, Any], end_result: dict[str, Any], bseal_path: Path,
                    initial_protocol_sha: str, initial_result_sha: str,
                    bseal: dict[str, Any]) -> None:
    validate_a_result(initial_protocol, initial_result, "initial")
    validate_a_result(end_protocol, end_result, "continuation")
    if initial_protocol.get("source_sha256") != end_protocol.get("source_sha256"):
        raise ValueError("initial and continuation A source locks differ")
    if end_protocol.get("continuation_seal_sha256") != sha256(bseal_path):
        raise ValueError("A continuation plan does not bind the supplied B seal")
    if end_protocol.get("initial_result_sha256") != initial_result_sha:
        raise ValueError("A continuation plan does not bind initial A result")
    if end_result.get("protocol_sha256") != end_protocol.get("_file_sha256"):
        raise ValueError("A continuation result does not bind its protocol file")
    if initial_result.get("protocol_sha256") != initial_protocol.get("_file_sha256"):
        raise ValueError("initial A result does not bind its protocol file")
    if initial_result.get("source_sha256") != end_result.get("source_sha256"):
        raise ValueError("A result source locks differ")
    if initial_result.get("gate_models_public_hashes", {}).get("correct_gate_sha256") != initial_protocol.get("v5_router_gate_reference_sha256"):
        raise ValueError("initial correct gate does not reproduce locked v5 gate")
    gate_meta = initial_result.get("gate_models_public_hashes", {})
    if (gate_meta.get("fit_count") != 2 or
            gate_meta.get("v5_gate_sha256") != gate_meta.get("correct_gate_sha256") or
            gate_meta.get("permuted_gate_sha256") == gate_meta.get("correct_gate_sha256")):
        raise ValueError("initial A gate fit count or correct/permuted gate identity differs from protocol")
    if end_result.get("gate_models_public_hashes") != initial_result.get("gate_models_public_hashes"):
        raise ValueError("continuation gate hashes differ from the frozen initial fits")
    if initial_protocol.get("ridge_fit_count") != 2 or end_protocol.get("ridge_fit_count") != 0:
        raise ValueError("A gate fit counts do not match two initial fits and zero endpoint fits")
    validate_rows(initial_result.get("rows", []), expected_count=3,
                  allowed_contexts={"v5_CMS_reference_step448"})
    validate_rows(end_result.get("rows", []), expected_count=6,
                  allowed_contexts={"aux_new_CE", "aux_old_gold_CE"})
    cp_by_seed = initial_protocol.get("source_sha256", {}).get("checkpoint_sha256_by_seed", {})
    for row in initial_result["rows"]:
        expected = cp_by_seed.get(str(row["seed"]), {}).get("v5_cms_448")
        if not expected or row.get("checkpoint_sha256") != expected:
            raise ValueError(f"initial A seed{row['seed']} did not score its locked v5 step-448 checkpoint")
    expected_endpoints = {}
    for run in bseal.get("runs", []):
        cps = [cp for cp in run.get("checkpoints", []) if cp.get("primary_step") == 1024]
        if len(cps) != 1:
            raise ValueError(f"B run {run.get('seed')}/{run.get('arm')} lacks one sealed step-1024 checkpoint")
        expected_endpoints[(run["seed"], run["arm"])] = cps[0]["sha256"]
    for row in end_result["rows"]:
        key = (row["seed"], row["new_context"])
        if row.get("checkpoint_sha256") != expected_endpoints.get(key):
            raise ValueError(f"continuation A seed/arm {key} did not score the locked B endpoint checkpoint")
    feature_hashes = [row.get("old_feature_sha256") for row in initial_result["rows"]]
    if len({json.dumps(value, sort_keys=True) for value in feature_hashes}) != 1:
        raise ValueError("initial seed rows do not share identical frozen old-context features")


def summarize_cost(rows: list[dict[str, Any]], *, endpoint: bool) -> dict[str, Any]:
    entries = []
    for row in rows:
        for key, cost in row["physical_shared"].items():
            split, condition = key.split(":", 1)
            entries.append({"seed": row["seed"], "arm": row["new_context"], "split": split,
                            "condition": condition, **cost["cost"]})
    group_keys = sorted({(x["arm"], x["condition"], x["split"]) for x in entries})
    out = []
    numeric = ("wall_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
               "full_forward_calls", "batches", "unique_context_payload_bytes_excluding_shared_backbone")
    op_fields = sorted({k for entry in entries for k in entry.get("operation_counts", {})})
    for arm, condition, split in group_keys:
        subset = [x for x in entries if (x["arm"], x["condition"], x["split"]) == (arm, condition, split)]
        costs = {field: mean_sd([float(x[field]) for x in subset]) for field in numeric}
        costs["operation_counts_mean"] = {field: statistics.fmean(float(x.get("operation_counts", {}).get(field, 0)) for x in subset)
                                           for field in op_fields}
        out.append({"arm": arm, "condition": condition, "split": split, "n": len(subset), "cost": costs})
    return {"scope": "measured physical shared-router scoring only; posthoc selectors reuse branch score cache",
            "runs": out, "endpoint": endpoint}


def build_gates(initial_rows: list[dict[str, Any]], endpoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    routed = ("correct_phase_ridge", "permuted_phase_ridge", "task_ID_DIAGNOSTIC")
    # Main requested endpoint gate ledger: six raw B endpoint branches plus 18 routed cases.
    endpoint_rows_out = []
    for row in endpoint_rows:
        raw_train = row["new_branch_train"]
        raw_val = row["new_branch_validation"]
        endpoint_rows_out.append({"stage": "B_raw_endpoint", "seed": row["seed"], "arm": row["new_context"],
            "condition": "new_context_alone", **gate_status(raw_train, raw_val)})
        for condition in routed:
            endpoint_rows_out.append({"stage": "A_routed_endpoint", "seed": row["seed"], "arm": row["new_context"],
                "condition": condition,
                **gate_status(row["posthoc"][condition]["train"], row["posthoc"][condition]["validation"])})
    if sum(x["stage"] == "B_raw_endpoint" for x in endpoint_rows_out) != 6 or sum(
        x["stage"] == "A_routed_endpoint" for x in endpoint_rows_out) != 18:
        raise AssertionError("endpoint gate ledger must contain 6 raw and 18 routed rows")
    return endpoint_rows_out


def storage_summary(storage: dict[str, Any], initial_protocol: dict[str, Any]) -> dict[str, Any]:
    if storage.get("schema") != "agnews-routing-storage-audit-v6" or storage.get("scoring_or_training") is not False:
        raise ValueError("unexpected or non-read-only storage audit")
    if storage.get("main_seal_sha256") != initial_protocol.get("source_sha256", {}).get("v5_seal_sha256"):
        raise ValueError("storage audit does not bind the same v5 checkpoint seal")
    if storage.get("all_three_gate_fits_identical") is not True:
        raise ValueError("v5 gate fits are not identical across seeds")
    rows = storage.get("rows", [])
    if {r.get("seed") for r in rows} != set(SEEDS) or len(rows) != 3:
        raise ValueError("storage audit must cover all three seeds")
    cp_by_seed = initial_protocol.get("source_sha256", {}).get("checkpoint_sha256_by_seed", {})
    for row in rows:
        if row.get("tensor_bytes_per_context") != 27396800 or row.get("deduplicated_whole_tensor_storage_bytes") != 28576459:
            raise ValueError("retained context payload differs from audited fixed byte counts")
        if row.get("distinct_tensor_count") != 9:
            raise ValueError("retained context changed-tensor count differs from audit")
        expected_cp = cp_by_seed.get(str(row["seed"]), {})
        if row.get("checkpoint_sha256", {}).get("new448") != expected_cp.get("v5_cms_448"):
            raise ValueError(f"storage audit seed{row['seed']} new448 checkpoint differs from A's locked v5 input")
    gate_dims = storage.get("v5_gate_feature_dims")
    one_gate_bytes = storage.get("minimal_gate_payload_bytes_float64_mean_scale_weight")
    if gate_dims != [3072, 3072, 3072] or one_gate_bytes != 3072 * 3 * 8:
        raise ValueError("v5 gate feature dimensions/payload differ from the audited 3,072-dim gates")
    return {"schema": storage["schema"], "seed_count": len(rows),
            "logical_bytes_per_context": 27396800,
            "two_context_logical_bytes": 2 * 27396800,
            "deduplicated_two_context_unique_bytes": 28576459,
            "incremental_unique_bytes_over_one_context": 28576459 - 27396800,
            "changed_tensor_count": 9,
            "one_float64_gate_payload_bytes": one_gate_bytes,
            "two_float64_gate_payload_bytes": 2 * one_gate_bytes,
            "measurement_note": "tensor payload audit, not peak VRAM; A physical route cost reports measured peak separately",
            "warm_checkpoint_provenance_note": "storage audit warm checkpoint is the v5 step-0 saved file and its file SHA need not equal the qualified warm file loaded by A; no exact warm-file identity is inferred here",
            "warm_checkpoint_file_sha_matches_a_input_by_seed": {
                str(r["seed"]): r.get("checkpoint_sha256", {}).get("warm") ==
                cp_by_seed.get(str(r["seed"]), {}).get("warm") for r in rows},
            "audited_warm_checkpoint_sha256_by_seed": {str(r["seed"]): r.get("checkpoint_sha256", {}).get("warm") for r in rows},
            "a_loaded_warm_checkpoint_sha256_by_seed": {str(s): v.get("warm") for s, v in cp_by_seed.items()},
            "v5_gate_hashes": storage.get("v5_gate_model_sha256")}


def make_work_ledger(qualification: dict[str, Any], bseal: dict[str, Any],
                     initial_result: dict[str, Any], endpoint_result: dict[str, Any],
                     preflight_failure: dict[str, Any] | None,
                     a_initial_failure: dict[str, Any] | None = None) -> dict[str, Any]:
    sources = {"qualification": qualification.get("work", {}), "B_main": bseal.get("work", {}),
               "A_initial": initial_result.get("work", {}), "A_continuation": endpoint_result.get("work", {})}
    ledger = merge_work_ledgers(sources)
    contract = check_work_expectations(qualification, bseal, initial_result, endpoint_result)
    ledger["phase_breakdown"] = contract
    ledger["qualification_preflight_failure"] = None
    if preflight_failure is not None:
        err = preflight_failure.get("error", preflight_failure.get("message"))
        ledger["qualification_preflight_failure"] = {
            "error_type": preflight_failure.get("error_type"), "error": err,
            "elapsed_seconds": preflight_failure.get("elapsed_seconds", preflight_failure.get("wall_seconds")),
            "work": sum_work(preflight_failure.get("work", {})) if preflight_failure.get("work") else None,
            "model_loaded": preflight_failure.get("model_loaded", False),
            "counted_in_GPU_work": False,
            "note": "preserved prior config preflight failure; no retry cost is folded into model work"}
    ledger["A_initial_source_R0_failure"] = None
    if a_initial_failure is not None:
        failure_work = a_initial_failure["work"]
        expected_failure = {"forward_calls": 196, "forward_input_tokens": 151672,
                            "gradient_forward_input_tokens": 0}
        if failure_work != expected_failure:
            raise ValueError(f"A initial R0 failure work differs from preserved receipt: {failure_work}")
        ledger["A_initial_source_R0_failure"] = {
            "source_driver_sha256": a_initial_failure["source_driver_sha256"],
            "completed_selector_pairs": 0, "actual_optimizer_updates": 0,
            "actual_ridge_fits": 0, "parity_forward_calls": 4,
            "train_rows_scored": 1024, "validation_rows_scored": 512,
            "active_seconds": a_initial_failure["active_seconds"],
            "wall_seconds": a_initial_failure["wall_seconds"],
            "error": a_initial_failure["error"],
            "evaluation_precision_issue": a_initial_failure["evaluation_precision_issue"],
            "work": failure_work,
            "counted_in_active_final_totals": False,
            "note": "preliminary A source 3fd exact-metric parity attempt; retained separately from successful R3 outputs"}
    failed_work = (a_initial_failure["work"] if a_initial_failure else
                   {key: 0 for key in WORK_FIELDS})
    ledger["inclusive_totals_with_A_initial_R0_failure"] = {
        key: ledger["combined_total"][key] + failed_work[key] for key in WORK_FIELDS}
    ledger["cost_interpretation"] = {
        "token_proxy": "forward_input_tokens includes prompts; gradient_forward_input_tokens is a token-work proxy",
        "not_claimed": ["FLOPs", "GPU busy time", "autoregressive generation latency"],
        "A_posthoc": "confidence and offline selectors reuse old/new full-branch scores; do not add a student forward per selector",
        "A_physical": "one shared backbone forward and two memory plus two readout adapter branches per scored batch",
        "confidence_live_estimate": "requires two full branch forwards per input batch to compare both contexts, not one student forward",
        "qualification": "12 actual optimizer update calls include deterministic fork/resume replay; unique schedule positions are four per arm (449-452)"
    }
    return ledger


def load_a_initial_failure(account_path: Path) -> dict[str, Any]:
    account = read_json(account_path)
    if account.get("schema") != "agnews-v6-metric-replay-failure-account-v1":
        raise ValueError("unexpected A initial failure-account schema")
    source_hash = account.get("source_sha256")
    if isinstance(source_hash, dict):
        source_hash = source_hash.get("driver_sha256")
    if source_hash != "3fdcbc79c744aef8e5ac9f4fd0de83f763e8fe5da7811a756edc5f53e5ff96dd":
        raise ValueError("A initial failure account is not bound to the preserved R0 source")
    expected_fields = (("completed_selector_pairs", 0), ("actual_optimizer_updates", 0),
                       ("actual_ridge_fits", 0), ("parity_forwards", 4),
                       ("parity_training_example_rows", 8), ("train_rows_scored", 1024),
                       ("validation_rows_scored", 512))
    for key, expected in expected_fields:
        if account.get(key) != expected:
            raise ValueError(f"A initial failure account {key} differs from expected {expected}")
    active, wall = account.get("active_seconds"), account.get("process_wall_seconds")
    if not isinstance(active, (int, float)) or not isinstance(wall, (int, float)):
        raise ValueError("A initial failure account lacks measured active/wall seconds")
    raw_failure_path = account_path.parent / "failure.json"
    if not raw_failure_path.is_file():
        raise FileNotFoundError(f"required preserved A initial failure.json missing: {raw_failure_path}")
    raw_failure = read_json(raw_failure_path)
    if (raw_failure.get("schema") != "agnews-routing-fairness-failure-v6" or
            raw_failure.get("completed_pairs") != 0 or
            raw_failure.get("confirmation_scored") is not False):
        raise ValueError("A initial failure.json does not confirm an incomplete, non-confirmation run")
    work = sum_work(raw_failure.get("work", {}))
    expected_work = {"forward_calls": 196, "forward_input_tokens": 151672,
                     "gradient_forward_input_tokens": 0}
    if (work != expected_work or account.get("forward_calls") != 196 or
            account.get("forward_input_tokens") != 151672):
        raise ValueError(f"A initial failure-account and raw work-meter differ: {work}")
    if account.get("gradient_forward_calls") != 0 or account.get("automatic_retry") is not False:
        raise ValueError("A initial failure account reports gradient work or an automatic retry")
    if (account.get("parity_guard_passed_before_failure") is not True or
            account.get("parity_numeric_difference_not_persisted_in_failed_attempt") is not True or
            account.get("partial_numeric_scores_not_persisted_in_failed_attempt") is not True or
            account.get("reference_guard_relaxed") is not False or
            account.get("confirmation_scored") is not False):
        raise ValueError("A initial failure account does not preserve the failed strict-parity history")
    if not isinstance(account.get("error"), str) or not isinstance(account.get("evaluation_precision_issue"), str):
        raise ValueError("A initial failure account lacks the recorded parity failure and precision explanation")
    return {"source_driver_sha256": source_hash, "active_seconds": float(active),
            "wall_seconds": float(wall), "work": work,
            "error": account.get("error"),
            "evaluation_precision_issue": account.get("evaluation_precision_issue"),
            "account_sha256": sha256(account_path), "failure_sha256": sha256(raw_failure_path)}


def aggregate_analysis(initial_protocol_path: Path, initial_result_path: Path,
                       continuation_protocol_path: Path, continuation_seal_path: Path,
                       endpoint_protocol_path: Path, endpoint_result_path: Path,
                       storage_path: Path, qualification_path: Path,
                       preflight_failure_path: Path | None,
                       a_initial_failure_path: Path | None = None) -> dict[str, Any]:
    ip = read_json(initial_protocol_path); ir = read_json(initial_result_path)
    bp = read_json(continuation_protocol_path); bs = read_json(continuation_seal_path)
    ep = read_json(endpoint_protocol_path); er = read_json(endpoint_result_path)
    q = read_json(qualification_path); storage = read_json(storage_path)
    fail = read_json(preflight_failure_path) if preflight_failure_path else None
    a_fail = load_a_initial_failure(a_initial_failure_path) if a_initial_failure_path else None
    for p, filename in ((ip, initial_protocol_path), (ep, endpoint_protocol_path)):
        p["_file_sha256"] = sha256(filename)
    validate_a_result(ip, ir, "initial")
    validate_a_result(ep, er, "continuation")
    validate_b_main(bp, bs, q, sha256(continuation_protocol_path), sha256(qualification_path))
    validate_a_pair(ip, ir, ep, er, continuation_seal_path, sha256(initial_protocol_path),
                    sha256(initial_result_path), bs)
    if ip["source_sha256"].get("records_sha256") != bp.get("source_sha256", {}).get("records_sha256"):
        raise ValueError("A and B phases are not bound to the same training/validation pack")
    initial_rows, endpoint_rows = ir["rows"], er["rows"]
    tables = build_condition_tables(initial_rows, endpoint_rows)
    gates = build_gates(initial_rows, endpoint_rows)
    paired = {
        "initial_correct_minus_permuted": [paired_differences(initial_rows, "correct_phase_ridge", "permuted_phase_ridge", s)
            for s in ("train", "validation")],
    }
    # Pair the two B arms by seed at each selector and split; rows are not averaged before differencing.
    gold_vs_new = []
    by = {(r["seed"], r["new_context"]): r for r in endpoint_rows}
    for condition in CONDITIONS:
        for split in ("train", "validation"):
            diffs = {group: {} for group in GROUPS}
            for group in GROUPS:
                for metric in METRICS:
                    values = [metric_at(extract_score(by[(seed, "aux_old_gold_CE")], condition, split), group, metric)
                              - metric_at(extract_score(by[(seed, "aux_new_CE")], condition, split), group, metric)
                              for seed in SEEDS]
                    diffs[group][metric] = mean_sd(values)
            gold_vs_new.append({"condition": condition, "split": split,
                                "contrast": "aux_old_gold_CE minus aux_new_CE", "paired_by_seed": diffs,
                                "no_p_value": True})
    paired["endpoint_correct_minus_permuted_by_arm"] = []
    for arm in ("aux_new_CE", "aux_old_gold_CE"):
        arm_rows = [r for r in endpoint_rows if r["new_context"] == arm]
        for split in ("train", "validation"):
            paired["endpoint_correct_minus_permuted_by_arm"].append({"arm": arm,
                **paired_differences(arm_rows, "correct_phase_ridge", "permuted_phase_ridge", split)})
    paired["endpoint_gold_minus_new_by_selector"] = gold_vs_new
    cost = {"initial": summarize_cost(initial_rows, endpoint=False),
            "endpoints": summarize_cost(endpoint_rows, endpoint=True)}
    work = make_work_ledger(q, bs, ir, er, fail, a_fail)
    storage_info = storage_summary(storage, ip)
    return {
        "schema": "agnews-routing-fairness-analysis-v6", "complete_inputs_only": True,
        "source_sha256": {"A_driver": ip["source_sha256"]["driver_sha256"],
                          "B_driver": bp["source_sha256"]["driver_source_sha256"],
                          "A_initial_protocol": sha256(initial_protocol_path),
                          "A_endpoint_protocol": sha256(endpoint_protocol_path),
                          "B_protocol": sha256(continuation_protocol_path),
                          "B_seal": sha256(continuation_seal_path),
                          "qualification": sha256(qualification_path),
                          "storage_audit": sha256(storage_path),
                          "qualification_preflight_failure": sha256(preflight_failure_path) if preflight_failure_path else None,
                          "A_initial_R0_failure_account": sha256(a_initial_failure_path) if a_initial_failure_path else None},
        "seeds": list(SEEDS), "conditions": list(CONDITIONS), "physical_conditions": list(PHYSICAL),
        "metric_families": list(METRICS), "denominators": {
            "A_by_split": {"train": {"all": 1024, "old": 512, "new": 512, "each_class": 256},
                           "validation": {"all": 512, "old": 256, "new": 256, "each_class": 128}},
            "B_train_endpoint": {"all": 1024, "old": 512, "new": 512, "each_class": 256},
            "B_validation_all_steps": {"all": 512, "old": 256, "new": 256, "each_class": 128}},
        "summaries_mean_sample_sd_n3": tables, "gates": gates, "paired_contrasts": paired,
        "physical_cost": cost, "storage": storage_info, "work_ledger": work,
        "interpretation_limits": [
            "The correct router uses train old/new phase supervision; the permuted control uses a fixed balanced permutation. This is not label-free routing.",
            "TASK_ID_DIAGNOSTIC uses true old/new group membership and is not a deployable selector.",
            "Two gates are fitted on the same phase labels and shared features across three seeds; the source enforces a single fit per gate and identical gate hashes.",
            "The retained old adapter is an additional stored context; selector results are a routing/capacity control, not a one-context causal method win.",
            "The old-gold auxiliary arm receives old labels; the auxiliary-new arm sees new labels. Equal batch/update counts do not equalize information exposure.",
            "At the v5 CMS step-448 start, the raw old branch had already collapsed. The old-gold arm therefore measures relearning/recovery, not retention from a competent single-context old-task start; archived warm-context access is a separate routing condition.",
            "No p-values, best-case selector, seed selection, independent confirmation, or held-out/test claim is produced."
        ],
        "official_test_accessed": False, "confirmation_scored": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_tables(out: Path, analysis: dict[str, Any]) -> None:
    initial_rows = []
    for key, groups in analysis["summaries_mean_sample_sd_n3"]["initial"].items():
        condition, split = key.rsplit(":", 1)
        for group, metrics in groups.items():
            for metric, summary in metrics.items():
                if metric == "examples_per_seed":
                    continue
                initial_rows.append({"condition": condition, "split": split, "group": group,
                    "metric": metric, "n": summary["n"], "mean": summary["mean"],
                    "sample_sd": summary["sample_sd"], "examples_per_seed": metrics["examples_per_seed"]})
    write_csv(out / "initial-summary.csv", initial_rows,
              ["condition", "split", "group", "metric", "n", "mean", "sample_sd", "examples_per_seed"])
    endpoint_rows = []
    for key, groups in analysis["summaries_mean_sample_sd_n3"]["endpoints"].items():
        arm, condition, split = key.split(":")
        for group, metrics in groups.items():
            for metric, summary in metrics.items():
                if metric == "examples_per_seed":
                    continue
                endpoint_rows.append({"arm": arm, "condition": condition, "split": split,
                    "group": group, "metric": metric, "n": summary["n"], "mean": summary["mean"],
                    "sample_sd": summary["sample_sd"], "examples_per_seed": metrics["examples_per_seed"]})
    write_csv(out / "endpoint-summary.csv", endpoint_rows,
              ["arm", "condition", "split", "group", "metric", "n", "mean", "sample_sd", "examples_per_seed"])
    gate_rows = []
    for row in analysis["gates"]:
        gate_rows.append({"stage": row["stage"], "seed": row["seed"], "arm": row["arm"],
            "condition": row.get("condition", "raw"), "all_tasks_pass": row["all_tasks_pass"],
            "old_train_em": row["tasks"]["old"]["train_em"], "old_validation_em": row["tasks"]["old"]["validation_em"],
            "old_class0_validation_em": row["tasks"]["old"]["class_validation_em"]["0"],
            "old_class1_validation_em": row["tasks"]["old"]["class_validation_em"]["1"],
            "new_train_em": row["tasks"]["new"]["train_em"], "new_validation_em": row["tasks"]["new"]["validation_em"],
            "new_class2_validation_em": row["tasks"]["new"]["class_validation_em"]["2"],
            "new_class3_validation_em": row["tasks"]["new"]["class_validation_em"]["3"]})
    write_csv(out / "endpoint-gates.csv", gate_rows,
              ["stage", "seed", "arm", "condition", "all_tasks_pass", "old_train_em", "old_validation_em",
               "old_class0_validation_em", "old_class1_validation_em", "new_train_em", "new_validation_em",
               "new_class2_validation_em", "new_class3_validation_em"])
    write_csv(out / "work-ledger.csv", [
        {"source": k, **v} for k, v in analysis["work_ledger"]["by_source"].items()],
        ["source", "stage_count", *WORK_FIELDS])


def write_plots(out: Path, analysis: dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plots requires Matplotlib") from exc
    selector_order = list(CONDITIONS)
    fig, ax = plt.subplots(figsize=(13, 6))
    x = list(range(len(selector_order)))
    for j, group in enumerate(("old", "new")):
        means, errs = [], []
        for condition in selector_order:
            cell = analysis["summaries_mean_sample_sd_n3"]["initial"][f"{condition}:validation"][group][METRICS[0]]
            means.append(cell["mean"]); errs.append(cell["sample_sd"] or 0.0)
        ax.errorbar([i + (j - .5) * .18 for i in x], means, yerr=errs, fmt="o", capsize=3,
                    label=f"{group} validation EM (mean ± sample SD, n=3)")
    ax.set_xticks(x, selector_order, rotation=30, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Full-vocabulary two-token EM")
    ax.set_title("Initial warm vs v5 CMS routing controls — validation only")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "initial-routing-validation-em.png", dpi=160)
    fig.savefig(out / "initial-routing-validation-em.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    conditions = ("new_CMS_reference_alone", "correct_phase_ridge", "permuted_phase_ridge", "task_ID_DIAGNOSTIC")
    arms = ("aux_new_CE", "aux_old_gold_CE")
    labels = []
    xs = []
    series = {"old": [], "new": []}
    errs = {"old": [], "new": []}
    xpos = 0
    for arm in arms:
        for condition in conditions:
            labels.append(f"{arm}\n{condition}"); xs.append(xpos)
            for group in ("old", "new"):
                cell = analysis["summaries_mean_sample_sd_n3"]["endpoints"][f"{arm}:{condition}:validation"][group][METRICS[0]]
                series[group].append(cell["mean"]); errs[group].append(cell["sample_sd"] or 0.0)
            xpos += 1
    ax.errorbar([x - .08 for x in xs], series["old"], yerr=errs["old"], fmt="o", capsize=3, label="old validation EM")
    ax.errorbar([x + .08 for x in xs], series["new"], yerr=errs["new"], fmt="o", capsize=3, label="new validation EM")
    ax.set_xticks(xs, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Full-vocabulary two-token EM")
    ax.set_title("All six B endpoints; paired by arm and fixed selector (mean ± sample SD, n=3)")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "endpoint-routing-validation-em.png", dpi=160)
    fig.savefig(out / "endpoint-routing-validation-em.svg")
    plt.close(fig)


def _load_cli_artifact(arg: Path, filename: str) -> Path:
    path = resolve_artifact(arg, filename)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True, help="initial A output directory or protocol.json")
    parser.add_argument("--continuation", type=Path, required=True, help="B main output directory or protocol.json")
    parser.add_argument("--endpoints", type=Path, required=True, help="A continuation output directory or protocol.json")
    parser.add_argument("--storage-audit", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--preflight-failure", type=Path)
    parser.add_argument("--a-initial-failure", type=Path,
                        help="optional account.json for preserved pre-R3 A initial parity failure")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"output directory must be new: {args.out_dir}")
    ip = _load_cli_artifact(args.initial, "protocol.json")
    ir = ip.parent / "result.json"
    bp = _load_cli_artifact(args.continuation, "protocol.json")
    bs = bp.parent / "development-sealed.json"
    ep = _load_cli_artifact(args.endpoints, "protocol.json")
    er = ep.parent / "result.json"
    for path in (ir, bs, er, args.storage_audit, args.qualification):
        if not path.is_file():
            raise FileNotFoundError(path)
    analysis = aggregate_analysis(ip, ir, bp, bs, ep, er, args.storage_audit,
                                  args.qualification, args.preflight_failure, args.a_initial_failure)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                                encoding="utf-8", newline="\n")
    (args.out_dir / "work-ledger.json").write_text(json.dumps(analysis["work_ledger"], indent=2, sort_keys=True, allow_nan=False) + "\n",
                                                   encoding="utf-8", newline="\n")
    write_tables(args.out_dir, analysis)
    if args.plots:
        write_plots(args.out_dir, analysis)


if __name__ == "__main__":
    main()
