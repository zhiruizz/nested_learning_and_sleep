"""Offline validator and aggregate tables for a completed v7 run.

This module uses only JSON, hashes and arithmetic. It never loads model/data
records or imports Torch. The run is accepted only when all 30 arms are sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

SEEDS = (17, 29, 43)
FAMILIES = ("current", "split", "split_narrow", "gold", "new_only")
SCOPES = ("receiver", "shared")
ARMS = tuple(f"{f}_{s}" for f in FAMILIES for s in SCOPES)
EVAL_STEPS = (0, 64, 256, 1024)
GROUPS = ("old", "new", "all", "class0", "class1", "class2", "class3")
EXPECTED_COUNTS = {
    "train": {"old": 512, "new": 512, "all": 1024, "class0": 256, "class1": 256, "class2": 256, "class3": 256},
    "validation": {"old": 256, "new": 256, "all": 512, "class0": 128, "class1": 128, "class2": 128, "class3": 128},
}
PARAM_COUNTS = {"receiver": 98304, "shared": 294912}
METRICS = ("em", "marker", "class", "candidate", "marker_nll", "class_nll", "nll")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _metric_record(row, group, expected, label):
    _require(group in row, f"missing {label}.{group}")
    value = row[group]
    _require(int(value.get("examples", -1)) == expected,
             f"wrong denominator for {label}.{group}: {value.get('examples')} != {expected}")
    for metric in METRICS:
        x = value.get(metric)
        _require(isinstance(x, (int, float)) and math.isfinite(float(x)), f"invalid {label}.{group}.{metric}")
        if metric in ("em", "marker", "class", "candidate"):
            _require(0.0 <= float(x) <= 1.0, f"out-of-range accuracy {label}.{group}.{metric}")
        else:
            _require(float(x) >= 0.0, f"negative NLL {label}.{group}.{metric}")
    return value


def _sum_work(work):
    """Sum only authoritative stage/phase records, never run-local duplicates."""
    result = {"flops": 0, "calls": 0, "stages": {}}
    _require(isinstance(work, dict), "execution-work.work must be an object")
    for stage, phases in work.items():
        _require(isinstance(phases, dict), f"bad work stage {stage}")
        st = {"flops": 0, "calls": 0, "phases": {}}
        for phase, rec in phases.items():
            flops, calls = int(rec.get("flops", -1)), int(rec.get("calls", -1))
            _require(flops >= 0 and calls >= 0, f"bad work record {stage}/{phase}")
            st["flops"] += flops; st["calls"] += calls
            st["phases"][phase] = {"flops": flops, "calls": calls}
        result["flops"] += st["flops"]; result["calls"] += st["calls"]
        result["stages"][stage] = st
    return result


def _sum_local_work(work):
    """Run rows use phase->record shape and overlap the global mode totals."""
    total = {"flops": 0, "calls": 0, "phases": {}}
    _require(isinstance(work, dict), "run-local work must be an object")
    for phase, rec in work.items():
        _require(isinstance(rec, dict), f"bad run-local phase {phase}")
        flops, calls = int(rec.get("flops", -1)), int(rec.get("calls", -1))
        _require(flops >= 0 and calls >= 0, f"bad run-local work {phase}")
        total["flops"] += flops; total["calls"] += calls
        total["phases"][phase] = {"flops": flops, "calls": calls}
    return total


def _add_work_totals(*items):
    out = {"flops": 0, "calls": 0}
    for item in items:
        if item:
            out["flops"] += int(item.get("flops", 0))
            out["calls"] += int(item.get("calls", 0))
    return out


def _summary(values):
    values = [float(x) for x in values]
    return {"n": len(values), "mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else 0.0,
            "values": values}


def _gate(metrics, split):
    old = (metrics["train"]["old"]["em"] >= .80
           and metrics["validation"]["old"]["em"] >= .75
           and all(metrics["validation"][f"class{k}"]["em"] >= .70 for k in (0, 1)))
    new = (metrics["train"]["new"]["em"] >= .80
           and metrics["validation"]["new"]["em"] >= .75
           and all(metrics["validation"][f"class{k}"]["em"] >= .70 for k in (2, 3)))
    return {"old": old, "new": new, "joint": old and new}


def _recovery_update_accounting(completed_original: int, last_observed_updates: int,
                               recovery_updates: int, unknown_count_bound):
    """Report observed attempts and a separate bound; never infer missing FLOPs."""
    lo, hi = map(int, unknown_count_bound)
    _require(completed_original >= 0 and last_observed_updates >= 0 and recovery_updates >= 0,
             "negative recovery update count")
    _require(0 <= lo <= hi, "invalid unknown update-count bound")
    return {"known_observed_update_attempts": completed_original + last_observed_updates + recovery_updates,
            "unknown_original_update_attempts_min": lo,
            "unknown_original_update_attempts_max": hi,
            "logical_updates": 30720,
            "interpretation": "Unknown range is a count bound after the last persisted progress; it is not an inclusive step-index interval."}


def _validate_eval(metrics, split):
    _require(isinstance(metrics, dict), f"missing {split} metrics")
    for group, count in EXPECTED_COUNTS[split].items():
        _metric_record(metrics, group, count, split)


def _validate_main(main_dir: Path, config_path: Path, qualification_path: Path,
                   cache_qual_path: Path | None, loss_qual_path: Path | None,
                   qualification_work_path: Path | None = None,
                   prior_failure_path: Path | None = None,
                   prior_failure_work_path: Path | None = None,
                   source_root: Path | None = None,
                   recovery_qualification_path: Path | None = None,
                   recovery_qualification_work_path: Path | None = None):
    main_dir = Path(main_dir)
    protocol_path = main_dir / "protocol.json"
    seal_path = main_dir / "development-sealed.json"
    for p in (protocol_path, seal_path, Path(config_path), Path(qualification_path)):
        _require(p.is_file(), f"required complete input missing: {p}")
    _require(not (main_dir / "fail.json").exists(), "main output contains fail.json")
    protocol, seal = map(read_json, (protocol_path, seal_path))
    recovery = seal.get("recovery")
    if recovery:
        _require(recovery.get("original_global_work_missing") is True,
                 "recovery seal must disclose missing original global work")
        work_path = main_dir / recovery["recovery_execution_work_path"]
        _require(work_path.is_file(), f"recovery execution-work missing: {work_path}")
    else:
        work_path = main_dir / "execution-work.json"
        _require(work_path.is_file(), f"required complete input missing: {work_path}")
    work = read_json(work_path)
    config, qual = read_json(config_path), read_json(qualification_path)

    _require(protocol.get("schema") == "agnews-single-receiver-protocol-v7", "wrong protocol schema")
    _require(seal.get("schema") == "agnews-single-receiver-development-v7" and seal.get("complete") is True,
             "main run is not fully sealed")
    _require(seal.get("protocol_sha256") == sha256(protocol_path), "seal does not bind protocol.json")
    _require(protocol.get("config_sha256") == sha256(config_path), "protocol/config SHA mismatch")
    _require(config.get("status") == "frozen_development" and config.get("development_only") is True,
             "config is not a frozen development contract")
    _require(tuple(config.get("seeds", ())) == SEEDS and tuple(config.get("arms", ())) == ARMS,
             "config seed/arm grid mismatch")
    _require(int(config.get("steps", -1)) == 1024 and tuple(config.get("evaluation_steps", ())) == EVAL_STEPS,
             "config update/evaluation schedule mismatch")
    _require(int(config.get("main_Adam_updates_expected", -1)) == 30720,
             "config expected update total mismatch")
    _require(config.get("scope_trainable_parameters") == PARAM_COUNTS, "config parameter-scope counts mismatch")
    _require(float(config.get("learning_rate", -1)) == .003 and float(config.get("weight_decay", -1)) == 0.0
             and float(config.get("gradient_clip", -1)) == 1.0, "config optimizer mismatch")
    _require(int(config.get("elapsed_limit_seconds", -1)) == 28800
             and int(config.get("per_run_elapsed_limit_seconds", -1)) == 900, "config elapsed cap mismatch")
    _require(protocol.get("driver_sha256") == config.get("driver_sha256"), "protocol driver/config mismatch")
    _require(protocol.get("cache_helper_sha256") == config.get("cache_helper_sha256"), "protocol cache helper mismatch")
    _require(protocol.get("records_sha256") == config.get("records_sha256"), "protocol records/config mismatch")
    _require(protocol.get("warm_result_sha256") == config.get("warm_result_sha256"), "warm source/config mismatch")
    _require(protocol.get("v6_seal_sha256") == config.get("v6_seal_sha256"), "v6 seal/config mismatch")
    _require(tuple(protocol.get("seeds", ())) == SEEDS and tuple(protocol.get("arms", ())) == ARMS,
             "protocol seed/arm mismatch")
    _require(int(protocol.get("steps", -1)) == 1024 and tuple(protocol.get("evaluation_steps", ())) == EVAL_STEPS,
             "protocol schedule mismatch")
    _require(int(protocol.get("padded_sequence_length", -1)) == 118, "protocol fixed token length mismatch")
    _require(protocol.get("qualification_sha256") == sha256(qualification_path), "protocol qualification SHA mismatch")
    checked_sources = {}
    if source_root is not None:
        source_root = Path(source_root)
        expected_sources = {"driver": (source_root / "tools/run_agnews_single_receiver_v7.py", config["driver_sha256"]),
                            "cache_helper": (source_root / "tools/agnews_tail_cache_v7.py", config["cache_helper_sha256"]),
                            "matmul_counter": (source_root / "tools/matmul_work_v7.py", config["matmul_counter_sha256"])}
        for name, (path, expected) in expected_sources.items():
            _require(path.is_file(), f"source file missing for {name}: {path}")
            actual = sha256(path)
            _require(actual == expected, f"source SHA mismatch for {name}")
            checked_sources[name] = {"path": str(path), "sha256": actual}
    recovery_inputs = None
    if recovery:
        _require(work.get("schema") == "agnews-v7-recovery-execution-work", "wrong recovery work schema")
        _require(work.get("source_sha256") == recovery.get("source_sha256"), "recovery work source mismatch")
        _require(work.get("complete") is True, "recovery execution-work is incomplete")
        _require(recovery_qualification_path is not None and Path(recovery_qualification_path).is_file(),
                 "recovery qualification receipt required")
        _require(recovery_qualification_work_path is not None and Path(recovery_qualification_work_path).is_file(),
                 "recovery qualification execution-work required")
        rq = read_json(recovery_qualification_path)
        _require(sha256(recovery_qualification_path) == recovery.get("qualification_sha256"),
                 "recovery qualification SHA mismatch")
        _require(rq.get("passed") is True and rq.get("independent_restores_exact") is True,
                 "recovery restore qualification failed")
        _require(int(rq.get("optimizer_attempts", -1)) == 4, "recovery qualification must have four proof updates")
        _require(rq.get("source_sha256") == recovery.get("source_sha256"), "recovery qualification source mismatch")
        rqwork = read_json(recovery_qualification_work_path)
        _require(rqwork.get("schema") == "agnews-v7-recovery-execution-work" and rqwork.get("complete") is True,
                 "recovery qualification work incomplete")
        _require(rqwork.get("source_sha256") == recovery.get("source_sha256"), "recovery qualification work source mismatch")
        recovery_dir = work_path.parent
        plan_path = recovery_dir / "plan.json"
        _require(plan_path.is_file(), "recovery plan missing")
        plan = read_json(plan_path)
        _require(sha256(plan_path) == recovery.get("plan_sha256"), "recovery plan SHA mismatch")
        process_loss_path = main_dir / "process-loss.json"
        progress_path = main_dir / "progress.json"
        _require(process_loss_path.is_file() and progress_path.is_file(), "original interruption receipts missing")
        _require(sha256(process_loss_path) == recovery.get("original_attempt_process_loss_sha256"),
                 "original process-loss SHA mismatch")
        process_loss = read_json(process_loss_path)
        _require(sha256(progress_path) == process_loss.get("original_progress_sha256"), "original progress SHA mismatch")
        _require(sha256(protocol_path) == process_loss.get("original_protocol_sha256"), "original protocol SHA mismatch")
        _require(plan.get("original_progress_sha256") == sha256(progress_path)
                 and plan.get("original_protocol_sha256") == sha256(protocol_path), "recovery plan/original input mismatch")
        original_rows = read_json(progress_path).get("runs", [])
        complete_original = [x for x in original_rows if x.get("status") == "complete"]
        interrupted = [x for x in original_rows if x.get("status") != "complete"]
        _require(len(complete_original) == 28 and len(interrupted) == 1
                 and (int(interrupted[0].get("seed", -1)), interrupted[0].get("arm"), int(interrupted[0].get("updates", -1)))
                 == (43, "new_only_receiver", 704), "original partial-progress state mismatch")
        _require(int(plan.get("source_checkpoint_step", -1)) == 256
                 and int(plan.get("last_observed_step", -1)) == 704
                 and int(plan.get("known_replayed_updates", -1)) == 448,
                 "recovery checkpoint/replay disclosure mismatch")
        _require(plan.get("original_unrecorded_updates_inclusive") == [0, 64], "recovery unknown-update bound mismatch")
        _require(rq.get("independent_restores_exact") is True and len(rq.get("branches", [])) == 2
                 and rq["branches"][0] == rq["branches"][1], "recovery branches are not exact")
        _require(int(recovery.get("actual_recovery_optimizer_updates", -1)) == int(work.get("optimizer_attempts", -2)),
                 "recovery work/update count mismatch")
        _require(int(recovery.get("known_replayed_updates", -1)) == 448, "known replay count mismatch")
        _require(recovery.get("unrecorded_original_updates_inclusive") == [0, 64], "unknown original-update bound mismatch")
        recovery_inputs = {"plan": {"path": str(plan_path), "sha256": sha256(plan_path), "data": plan},
                           "process_loss": {"path": str(process_loss_path), "sha256": sha256(process_loss_path), "data": process_loss},
                           "original_progress": {"path": str(progress_path), "sha256": sha256(progress_path)},
                           "qualification": {"path": str(recovery_qualification_path), "sha256": sha256(recovery_qualification_path), "data": rq},
                           "qualification_work": {"path": str(recovery_qualification_work_path), "sha256": sha256(recovery_qualification_work_path),
                                                  "data": rqwork, "authoritative_work": _sum_work(rqwork.get("work"))}}
    else:
        _require(work.get("schema") == "agnews-single-receiver-authoritative-matmul-work-v7", "wrong work schema")
        _require(work.get("driver_sha256") == config.get("driver_sha256"), "execution work source mismatch")
        _require(work.get("scope", "").startswith("Executed mm/addmm/bmm/mv/dot"), "work scope not described as lower bound")
        _require("never add them" in work.get("scope", ""), "local-work non-additivity warning missing")
    _require(protocol.get("confirmation_scored") is False and seal.get("confirmation_scored") is False,
             "confirmation data must remain unscored")
    _require(protocol.get("confirmation_input") is False and seal.get("confirmation_input") in (None, False),
             "confirmation data must not be an input")

    _require(qual.get("passed") is True and qual.get("exact_model_optimizer_policy_rng") is True,
             "resume qualification did not pass")
    _require(qual.get("config_sha256") == sha256(config_path), "qualification/config SHA mismatch")
    _require(qual.get("driver_sha256") == config.get("driver_sha256"), "qualification driver mismatch")
    _require(int(qual.get("actual_Adam_updates", -1)) == 12, "qualification Adam update count mismatch")
    _require(qual.get("scoring") is False and qual.get("confirmation_scored") is False,
             "qualification must be non-scoring")
    qscopes = {x.get("scope"): x for x in qual.get("results", [])}
    _require(set(qscopes) == set(SCOPES), "qualification must cover both scopes")
    _require(all(qscopes[s].get("exact_model_optimizer_policy_rng") is True for s in SCOPES),
             "qualification scope restore mismatch")
    qualification_work = None
    if qualification_work_path is not None:
        qw = read_json(qualification_work_path)
        _require(qw.get("schema") == "agnews-single-receiver-authoritative-matmul-work-v7",
                 "wrong qualification execution-work schema")
        _require(qw.get("driver_sha256") == config.get("driver_sha256"), "qualification work source mismatch")
        _require(int(qw.get("optimizer_attempts", -1)) == 12, "qualification work update count mismatch")
        qualification_work = {"path": str(qualification_work_path), "sha256": sha256(qualification_work_path),
                              "seconds": qw.get("seconds"), "optimizer_attempts": qw.get("optimizer_attempts"),
                              "authoritative_work": _sum_work(qw.get("work")),
                              "qualification_local_work_overlap_not_additive": qual.get("matmul_work_lower_bound")}

    if (prior_failure_path is None) != (prior_failure_work_path is None):
        raise ValueError("prior failure receipt and execution-work must be supplied together")
    prior_failure = None
    if prior_failure_path is not None:
        fd, fw = read_json(prior_failure_path), read_json(prior_failure_work_path)
        _require(fd.get("confirmation_scored") is False, "prior qualification failure touched confirmation data")
        _require(fw.get("schema") == "agnews-single-receiver-authoritative-matmul-work-v7",
                 "wrong prior failure execution-work schema")
        prior_failure = {"receipt_path": str(prior_failure_path), "receipt_sha256": sha256(prior_failure_path),
                         "work_path": str(prior_failure_work_path), "work_sha256": sha256(prior_failure_work_path),
                         "error_type": fd.get("error_type"), "error": fd.get("error"),
                         "actual_Adam_updates": fd.get("actual_Adam_updates"),
                         "optimizer_attempts": fw.get("optimizer_attempts"), "seconds": fw.get("seconds"),
                         "authoritative_work": _sum_work(fw.get("work")),
                         "failure_receipt_local_work_overlap_not_additive": fd.get("matmul_work_lower_bound")}

    numeric_qualifications = {}
    passed_inputs = {"cache": cache_qual_path, "loss_paths": loss_qual_path}
    for name, descriptor in config.get("numerical_qualifications", {}).items():
        path = passed_inputs.get(name)
        if path is None:
            configured = Path(descriptor["path"])
            path = configured if configured.is_file() else None
        _require(path is not None and Path(path).is_file(), f"numerical qualification file unavailable: {name}")
        _require(sha256(path) == descriptor["sha256"], f"numerical qualification SHA mismatch: {name}")
        qdoc = read_json(path)
        _require(qdoc.get("passed") is True and qdoc.get("confirmation_scored") is False,
                 f"numerical qualification failed or scored confirmation: {name}")
        numeric_qualifications[name] = {"path": str(path), "sha256": sha256(path),
                                         "optimizer_updates": int(qdoc.get("actual_Adam_updates", qdoc.get("optimizer_updates", 0))),
                                         "work": qdoc.get("matmul_work_lower_bound", qdoc.get("work"))}

    rows = seal.get("runs")
    _require(isinstance(rows, list) and len(rows) == 30, "seal must contain exactly 30 runs")
    by_key = {}
    common_initial = seal.get("common_initial_by_seed", {})
    _require(set(common_initial) == {str(s) for s in SEEDS}, "common initialization missing seeds")
    for run in rows:
        seed, arm = int(run.get("seed", -1)), run.get("arm")
        key = (seed, arm)
        _require(seed in SEEDS and arm in ARMS and key not in by_key, f"invalid or duplicate run {key}")
        by_key[key] = run
        _require(run.get("status") == "complete", f"incomplete arm {key}")
        _require(int(run.get("updates", -1)) == 1024, f"wrong updates in {key}")
        scope = arm.rsplit("_", 1)[1]
        _require(int(run.get("trainable_parameter_count", -1)) == PARAM_COUNTS[scope], f"scope parameter count mismatch {key}")
        _require(run.get("frozen_scope_guard_passed") is True, f"frozen-scope guard failed {key}")
        _require(run.get("initial") == common_initial[str(seed)], f"arm initialization mismatch {key}")
        _require(run.get("schedule_sha256") == config["schedule_sha256_by_seed_arm"].get(f"{seed}:{arm}"),
                 f"schedule lock mismatch {key}")
        evals = run.get("evaluations", [])
        _require([int(x.get("step", -1)) for x in evals] == list(EVAL_STEPS), f"eval/checkpoint timeline mismatch {key}")
        for ev in evals:
            _validate_eval(ev.get("train"), "train"); _validate_eval(ev.get("validation"), "validation")
        _require(run.get("train_final") == evals[-1]["train"], f"final train metric differs from step 1024 {key}")
        _require([int(x.get("step", -1)) for x in run.get("checkpoints", [])] == [64, 256, 1024],
                 f"checkpoint timeline mismatch {key}")
        for cp in run["checkpoints"]:
            _require(len(cp.get("sha256", "")) == 64 and cp.get("path"), f"checkpoint receipt incomplete {key}")
        expected_draws = {str(k): (0 if arm.startswith("new_only_") and k < 2 else
                                  4096 if arm.startswith("new_only_") else 2048)
                          for k in range(4)}
        _require(run.get("class_draws") == expected_draws, f"training class exposure mismatch {key}")
        class_unique = run.get("class_unique_examples", {})
        _require(set(class_unique) == {str(k) for k in range(4)}, f"class-unique accounting missing {key}")
        for cls in range(4):
            max_unique = 0 if arm.startswith("new_only_") and cls < 2 else (
                16 if arm.startswith("split_narrow_") and cls < 2 else 256)
            count = int(class_unique[str(cls)])
            _require(0 <= count <= max_unique, f"unique-example count exceeds pool {key}/class{cls}")
        _require(sum(int(x) for x in class_unique.values()) == int(run.get("unique_training_examples", -1)),
                 f"unique-example total mismatch {key}")
        if recovery and int(run.get("recovered_from_step") or 0) > 0:
            _require((seed, arm) == (43, "new_only_receiver") and int(run["recovered_from_step"]) == 256,
                     f"unexpected recovered row {key}")
            _require(int(run.get("recovery_actual_updates", -1)) == 768,
                     "recovered receiver must record 768 actual continuation updates")
            _require(int(run.get("logical_student_forward_passes", -1)) == 1024
                     and int(run.get("logical_student_backward_passes", -1)) == 1024,
                     "recovered receiver logical pass count mismatch")
            _require(int(run.get("student_forward_passes", -1)) == 1472
                     and int(run.get("student_backward_passes", -1)) == 1472,
                     "recovered receiver observed pass lower bound mismatch")
            _require(run.get("exposure_counters_describe_logical_schedule") is True,
                     "recovered exposure counters need logical-schedule label")
            _require(run.get("original_unrecorded_optimizer_attempts_inclusive") == [0, 64],
                     "recovered receiver unknown-update bound mismatch")
            _require(isinstance(run.get("original_attempt_training_work_observed"), dict),
                     "original receiver local work must remain separately retained")
        else:
            _require(int(run.get("student_forward_passes", -1)) == 1024
                     and int(run.get("student_backward_passes", -1)) == 1024,
                     f"student pass count mismatch {key}")
        _require(int(run.get("answer_token_targets", -1)) == 16384, f"answer target count mismatch {key}")
        expected_direct = 8192 if arm.startswith(("gold_", "new_only_")) else 0
        expected_teacher_origin = 8192 if arm.startswith(("split_", "split_narrow_")) else 0
        _require(int(run.get("direct_gold_answer_draws", -1)) == expected_direct, f"gold-label exposure mismatch {key}")
        _require(int(run.get("teacher_origin_selection_draws", -1)) == expected_teacher_origin,
                 f"teacher-origin metadata mismatch {key}")
        parity = run.get("parity")
        _require(isinstance(parity, dict) and set(("full", "tail")) <= set(parity), f"missing parity guard {key}")
        _validate_eval(parity["full"], "validation"); _validate_eval(parity["tail"], "validation")
        for group in GROUPS:
            _require(parity["full"][group]["em"] == parity["tail"][group]["em"], f"endpoint parity EM mismatch {key}/{group}")
            _require(abs(parity["full"][group]["nll"] - parity["tail"][group]["nll"]) <= .001,
                     f"endpoint parity NLL mismatch {key}/{group}")
    _require(set(by_key) == {(s, a) for s in SEEDS for a in ARMS}, "missing seed/arm cells")
    if recovery:
        _require(sum(1 for r in rows if int(r.get("recovered_from_step") or 0) > 0) == 1,
                 "expected exactly one resumed arm with a source checkpoint")
        _require(sum(1 for r in rows if r.get("seed") == 43 and r.get("arm") == "new_only_shared") == 1,
                 "expected the formerly unstarted new_only_shared arm")
        shared_row = next(r for r in rows if r.get("seed") == 43 and r.get("arm") == "new_only_shared")
        _require(int(shared_row.get("recovered_from_step", -1)) == 0,
                 "new shared arm must be explicitly recorded as started from step zero")
        _require(int(work.get("optimizer_attempts", -1)) == 1792, "recovery attempt update count mismatch")
        _require(int(recovery.get("actual_recovery_optimizer_updates", -1)) == 1792,
                 "seal recovery update count mismatch")
        _require("optimizer_attempts" not in seal, "recovery seal must not invent an original global count")
        logical_updates = sum(int(r["updates"]) for r in rows)
        _require(logical_updates == 30720, "recovery logical update total mismatch")
        unknown = recovery["unrecorded_original_updates_inclusive"]
        recovery["_derived_update_accounting"] = _recovery_update_accounting(
            28 * 1024, 704, int(recovery["actual_recovery_optimizer_updates"]), unknown)
    else:
        _require(int(work.get("optimizer_attempts", -1)) == 30720, "authoritative optimizer-attempt count mismatch")
        _require(int(seal.get("optimizer_attempts", -1)) == 30720, "seal optimizer-attempt count mismatch")
    _require(sum(int(r["updates"]) for r in rows) == 30720, "per-run update sum mismatch")
    work_seconds = float(work.get("seconds", -1))
    _require(math.isfinite(work_seconds) and work_seconds >= 0, "invalid execution-work time")
    work_totals = _sum_work(work.get("work"))

    tq = seal.get("teacher_quality_by_seed", [])
    _require({int(x.get("seed", -1)) for x in tq} == set(SEEDS), "teacher quality must cover all seeds")
    teacher_rows = {}
    for item in tq:
        seed = int(item["seed"]); _require(item.get("warm_old_gate_passed") is True, f"warm teacher gate failed seed {seed}")
        wt, wv = item["warm_oldteacher"]["train_old_only"], item["warm_oldteacher"]["validation_old_only"]
        _require(wt["old"]["examples"] == 512 and wv["old"]["examples"] == 256, f"teacher denominators mismatch seed {seed}")
        _require(wt["old"]["em"] >= .80 and wv["old"]["em"] >= .75
                 and all(wv[f"class{k}"]["em"] >= .70 for k in (0, 1)), f"warm teacher gate values fail seed {seed}")
        _validate_eval(item["current_newteacher"]["train"], "train")
        _validate_eval(item["current_newteacher"]["validation"], "validation")
        for split, n in (("train_old_only", 512), ("validation_old_only", 256)):
            source = item["warm_oldteacher"][split]
            _metric_record(source, "old", n, f"warm_teacher.{split}")
            for cls, count in ((0, n // 2), (1, n // 2)):
                _metric_record(source, f"class{cls}", count, f"warm_teacher.{split}")
        teacher_rows[str(seed)] = item

    return {"protocol": protocol, "seal": seal, "execution_work": work, "config": config,
            "qualification": qual, "qualification_work": qualification_work,
            "prior_failure": prior_failure, "numeric_qualifications": numeric_qualifications,
            "recovery": recovery, "recovery_inputs": recovery_inputs,
            "runs": by_key, "teacher_quality": teacher_rows, "work_totals": work_totals,
            "paths": {"main_dir": str(main_dir), "protocol": str(protocol_path), "seal": str(seal_path),
                      "execution_work": str(work_path), "config": str(config_path),
                      "qualification": str(qualification_path)}, "checked_sources": checked_sources}


def build_analysis(inputs):
    runs = inputs["runs"]
    recovery = inputs.get("recovery")
    endpoints = {}
    gates = {}
    for arm in ARMS:
        perseed = {}
        for seed in SEEDS:
            run = runs[(seed, arm)]
            ev = run["evaluations"][-1]
            metric_pair = {"train": ev["train"], "validation": ev["validation"]}
            perseed[str(seed)] = {split: {group: {k: rec[k] for k in METRICS}
                                          for group, rec in metrics.items()}
                                  for split, metrics in metric_pair.items()}
            gates[(seed, arm)] = _gate(metric_pair, arm.rsplit("_", 1)[0])
        endpoints[arm] = {split: {group: {metric: _summary([perseed[str(s)][split][group][metric] for s in SEEDS])
                                           for metric in METRICS}
                                  for group in GROUPS}
                          for split in ("train", "validation")}
        endpoints[arm]["per_seed"] = perseed
        endpoints[arm]["per_seed_gates"] = {str(s): gates[(s, arm)] for s in SEEDS}
        endpoints[arm]["gate_counts"] = {k: sum(int(gates[(s, arm)][k]) for s in SEEDS) for k in ("old", "new", "joint")}

    comparisons = []
    pairs = (("current", "split"), ("split_narrow", "split"), ("gold", "split"), ("new_only", "gold"))
    for a, b in pairs:
        for scope in SCOPES:
            left, right = f"{a}_{scope}", f"{b}_{scope}"
            per_seed = {}
            for seed in SEEDS:
                le = runs[(seed, left)]["evaluations"][-1]
                ri = runs[(seed, right)]["evaluations"][-1]
                per_seed[str(seed)] = {
                    split: {
                        group: {m: ri[split][group][m] - le[split][group][m]
                                for m in ("em", "nll")}
                        for group in ("old", "new", "all")
                    }
                    for split in ("train", "validation")
                }
            comparisons.append({"contrast": f"{right} minus {left}", "scope": scope, "per_seed_deltas": per_seed,
                                "no_inference": True})

    run_costs = []
    local_flops = local_calls = 0
    original_partial_train = {"flops": 0, "calls": 0}
    for seed in SEEDS:
        for arm in ARMS:
            r = runs[(seed, arm)]
            local = _sum_local_work(r.get("work") or {})
            evaluation_local = _sum_work(r.get("evaluation_matmul_work_lower_bound", {}))
            endpoint_local = _sum_work(r.get("endpoint_matmul_work_lower_bound", {}))
            original_work = _sum_local_work(r.get("original_attempt_training_work_observed") or {})
            if recovery and original_work["flops"]:
                original_partial_train["flops"] += original_work["flops"]
                original_partial_train["calls"] += original_work["calls"]
            local_flops += local["flops"] + evaluation_local["flops"] + endpoint_local["flops"] + original_work["flops"]
            local_calls += local["calls"] + evaluation_local["calls"] + endpoint_local["calls"] + original_work["calls"]
            run_costs.append({"seed": seed, "arm": arm, "updates": r["updates"],
                              "seconds": r.get("seconds"), "seconds_recovery": r.get("seconds_recovery"),
                              "recovered_from_step": r.get("recovered_from_step"),
                              "recovery_actual_updates": r.get("recovery_actual_updates"),
                              "peak_cuda_allocated_bytes": r.get("peak_cuda_allocated_bytes"),
                              "peak_cuda_reserved_bytes": r.get("peak_cuda_reserved_bytes"),
                              "cache_file_bytes": r.get("cache_file_bytes"), "model_tensor_bytes": r.get("model_tensor_bytes"),
                              "unique_training_examples": r.get("unique_training_examples"), "class_draws": r.get("class_draws"),
                              "class_unique_examples": r.get("class_unique_examples"), "direct_gold_answer_draws": r.get("direct_gold_answer_draws"),
                              "teacher_origin_selection_draws": r.get("teacher_origin_selection_draws"),
                              "student_forward_passes": r.get("student_forward_passes"),
                              "student_backward_passes": r.get("student_backward_passes"),
                              "student_input_tokens": r.get("student_input_tokens"),
                              "local_training_matmul_work": local,
                              "local_evaluation_matmul_work": evaluation_local,
                              "local_endpoint_matmul_work": endpoint_local,
                              "prior_attempt_local_training_work": original_work,
                              "all_local_counters_non_authoritative_not_additive": True})
    if recovery:
        derived = recovery.get("_derived_update_accounting", {})
        # Local row counters overlap the recovery-global ledger for work done
        # during recovery, and only a subset is retained for the old attempt.
        # Keep these ledgers disjoint in the presentation; never sum them.
        recovery_run_work = inputs["work_totals"]
        recovery_q_work = inputs["recovery_inputs"]["qualification_work"]["authoritative_work"]
        recovery_total = _add_work_totals(recovery_run_work, recovery_q_work)
        ledger = {"original_attempt_global_work": "MISSING; no original global execution-work receipt",
                  "retained_row_local_work_counters_not_additive": {"flops": local_flops, "calls": local_calls,
                      "original_partial_receiver_training_only": original_partial_train,
                      "interpretation": "This sum includes overlapping local counters across rows and phases; it is diagnostic only, not an attempt-global total."},
                  "recovery_attempt_run_work": recovery_run_work,
                  "recovery_qualification_work_separate": recovery_q_work,
                  "recovery_attempt_work_including_qualification": recovery_total,
                  "recovery_execution_seconds": inputs["execution_work"].get("seconds"),
                  "sum_recorded_per_run_seconds_not_wall": sum(float(x.get("seconds") or 0) + float(x.get("seconds_recovery") or 0) for x in run_costs),
                  "per_run_seconds_scope": "Sum of per-arm timers only; not total wall time and excludes preparation/qualification not attached to arms.",
                  "recovery_optimizer_attempts": inputs["execution_work"].get("optimizer_attempts"),
                  "known_replayed_updates": recovery.get("known_replayed_updates"),
                  "recovery_qualification_seconds": inputs["recovery_inputs"]["qualification_work"]["data"].get("seconds"),
                  "recovery_qualification_optimizer_attempts": inputs["recovery_inputs"]["qualification_work"]["data"].get("optimizer_attempts"),
                  "logical_vs_observed_updates": derived,
                  "recovery_qualification_receipt": inputs["recovery_inputs"]["qualification"]["data"],
                  "original_qualification_work": inputs["qualification_work"],
                  "prior_failure_separate_not_in_any_total": inputs["prior_failure"],
                  "numeric_qualification_inputs": inputs["numeric_qualifications"],
                  "run_costs": run_costs,
                  "preparation_forward_calls_reconstructed": {"calls": 864,
                      "basis": "fixed warm/train/validation schedule; this is a call count, not a FLOP estimate",
                      "flops": None},
                  "original_process_loss": inputs["recovery_inputs"]["process_loss"]["data"],
                  "interpretation": "The original attempt global FLOPs are missing and are never imputed. Retained row-local counters are incomplete observed subsets. Recovery run work and recovery qualification work are separately itemized; their sum is the complete known recovery cost. Never add row-local counters to these global ledgers. Known observed update attempts plus the bounded unknown original attempts are not a measured FLOP total."}
    else:
        ledger = {"authoritative_main_work": inputs["work_totals"],
              "main_execution_seconds": inputs["execution_work"].get("seconds"),
              "main_optimizer_attempts": inputs["execution_work"].get("optimizer_attempts"),
              "per_run_local_counter_sum_overlap_not_additive": {"flops": local_flops, "calls": local_calls},
              "qualification_updates": inputs["qualification"].get("actual_Adam_updates"),
              "qualification_authoritative_work": inputs["qualification_work"],
              "prior_failure_separate_not_in_main_total": inputs["prior_failure"],
              "numeric_qualification_inputs": inputs["numeric_qualifications"],
              "run_costs": run_costs,
              "interpretation": "The global execution-work meter is the sole main-run arithmetic ledger. Per-run local counters overlap it and are shown only for auditing; do not add them. Matrix-multiply totals are a measured lower bound, not total FLOPs."}
    return {"schema": "agnews-single-receiver-v7-analysis-v1", "accepted_complete_run": True,
            "source": inputs["paths"], "config_sha256": inputs["protocol"]["config_sha256"],
            "protocol_sha256": sha256(Path(inputs["paths"]["protocol"])),
            "driver_sha256": inputs["protocol"]["driver_sha256"],
            "checked_source_files": inputs["checked_sources"],
            "confirmation_scored": False, "seeds": list(SEEDS), "arms": list(ARMS),
            "denominators": EXPECTED_COUNTS, "endpoints": endpoints,
            "paired_contrasts": comparisons,
            "teacher_quality_by_seed": inputs["teacher_quality"],
            "joint_gate_counts": {arm: endpoints[arm]["gate_counts"] for arm in ARMS},
            "work_ledger": ledger}


def _fmt(x):
    return f"{x['mean']:.4f} ± {x['sample_sd']:.4f} (n={x['n']})"


def _write_reports(out_dir: Path, analysis):
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# V7 endpoint summary", "", "Means ± sample SD across all three fixed seeds. No seed or checkpoint selection.", "",
             "| Arm | Scope | Old EM train/val | New EM train/val | All EM train/val | Old NLL train/val | New NLL train/val | Joint gates |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        ep = analysis["endpoints"][arm]; fam, scope = arm.rsplit("_", 1)
        cells = []
        for group, metric in (("old", "em"), ("new", "em"), ("all", "em"), ("old", "nll"), ("new", "nll")):
            cells.append(f"{_fmt(ep['train'][group][metric])} / {_fmt(ep['validation'][group][metric])}")
        gc = ep["gate_counts"]
        lines.append(f"| {fam} | {scope} | " + " | ".join(cells) + f" | old {gc['old']}/3; new {gc['new']}/3; joint {gc['joint']}/3 |")
    (out_dir / "endpoint-table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["# Per-seed endpoint gates", "", "Fixed gate definitions are applied to train/validation only; all three seeds are retained.",
             "", "| Arm | Seed | Old gate | New gate | Joint gate |", "|---|---:|---|---|---|"]
    for arm in ARMS:
        for seed in SEEDS:
            g = analysis["endpoints"][arm]["per_seed_gates"][str(seed)]
            lines.append(f"| {arm} | {seed} | {'PASS' if g['old'] else 'FAIL'} | {'PASS' if g['new'] else 'FAIL'} | {'PASS' if g['joint'] else 'FAIL'} |")
    (out_dir / "per-seed-gates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["# Per-seed paired contrasts", "", "Signed endpoint deltas (right minus left); raw seed values only, no p-values or best-seed selection.",
             "", "| Contrast | Scope | Seed | Old EM Δ | New EM Δ | All EM Δ | Old NLL Δ | New NLL Δ |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in analysis["paired_contrasts"]:
        for seed, splits in c["per_seed_deltas"].items():
            v = splits["validation"]
            lines.append(f"| {c['contrast']} | {c['scope']} | {seed} | {v['old']['em']:+.4f} | {v['new']['em']:+.4f} | {v['all']['em']:+.4f} | {v['old']['nll']:+.4f} | {v['new']['nll']:+.4f} |")
    (out_dir / "paired-contrasts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# Teacher qualification", "", "Old-teacher competence is checked on the fixed development train/validation rows; current-teacher scores are descriptive.",
             "", "| Seed | Warm old train EM | Warm old validation EM | Current teacher old validation EM | Current teacher new validation EM | Warm gate |", "|---:|---:|---:|---:|---:|---|"]
    for seed in SEEDS:
        item = analysis["teacher_quality_by_seed"][str(seed)]
        wt = item["warm_oldteacher"]["train_old_only"]["old"]["em"]
        wv = item["warm_oldteacher"]["validation_old_only"]["old"]["em"]
        cur = item["current_newteacher"]["validation"]
        old_em = cur.get("old", {}).get("em", float("nan"))
        new_em = cur.get("new", {}).get("em", float("nan"))
        lines.append(f"| {seed} | {wt:.4f} | {wv:.4f} | {old_em:.4f} | {new_em:.4f} | PASS |")
    (out_dir / "teacher-quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "work-ledger.json").write_text(json.dumps(analysis["work_ledger"], indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if "original_attempt_global_work" in analysis["work_ledger"]:
        w = analysis["work_ledger"]
        u = w["logical_vs_observed_updates"]
        note = ["# Recovery and cost accounting", "",
                "The original process ended without terminal receipts; the cause is unknown. Its global execution-work ledger is missing and is not reconstructed.",
                "", f"- Logical scheduled updates across the 30 arms: {u['logical_updates']:,}.",
                f"- Known observed update attempts across original progress plus recovery: {u['known_observed_update_attempts']:,}.",
                f"- Additional unrecorded original attempts are bounded at {u['unknown_original_update_attempts_min']} to {u['unknown_original_update_attempts_max']}; this is a count bound after the last persisted progress, not a step-index interval.",
                f"- The recovered receiver resumed from step 256 for 768 updates; 448 repeat updates reproduce work already observed through step 704. It has 1,472 observed student forward/backward passes but a 1,024-step logical schedule.",
                f"- Recovery run work: {w['recovery_attempt_run_work']['flops']:,} recorded arithmetic units across {w['recovery_attempt_run_work']['calls']:,} instrumented calls.",
                f"- Recovery qualification work: {w['recovery_qualification_work_separate']['flops']:,} recorded arithmetic units across {w['recovery_qualification_work_separate']['calls']:,} instrumented calls.",
                "- These are the recovery meter's recorded operation counts, not total hardware FLOPs. Original arm-local counters are incomplete overlapping observations and must not be added to the recovery totals.",
                "- Fixed preparation work is represented by 864 reconstructed forward calls; no FLOP estimate is assigned. Main local timing sums are not wall time.",
                "- Qualification failures and successful qualification runs are separate cost entries in `work-ledger.json`; no confirmation/test data were scored.", ""]
        (out_dir / "recovery-note.md").write_text("\n".join(note), encoding="utf-8")


def analyze(main_dir: Path, config_path: Path, qualification_path: Path, out_dir: Path,
            cache_qual_path: Path | None = None, loss_qual_path: Path | None = None,
            qualification_work_path: Path | None = None,
            prior_failure_path: Path | None = None,
            prior_failure_work_path: Path | None = None,
            source_root: Path | None = None,
            recovery_qualification_path: Path | None = None,
            recovery_qualification_work_path: Path | None = None):
    inputs = _validate_main(main_dir, config_path, qualification_path, cache_qual_path, loss_qual_path,
                            qualification_work_path, prior_failure_path, prior_failure_work_path, source_root,
                            recovery_qualification_path, recovery_qualification_work_path)
    analysis = build_analysis(inputs)
    _write_reports(Path(out_dir), analysis)
    return analysis


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--cache-qualification", type=Path)
    p.add_argument("--loss-path-qualification", type=Path)
    p.add_argument("--qualification-work", type=Path)
    p.add_argument("--prior-qualification-failure", type=Path)
    p.add_argument("--prior-failure-work", type=Path)
    p.add_argument("--recovery-qualification", type=Path)
    p.add_argument("--recovery-qualification-work", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)
    analyze(args.main_dir, args.config, args.qualification, args.out_dir,
            args.cache_qualification, args.loss_path_qualification, args.qualification_work,
            args.prior_qualification_failure, args.prior_failure_work, args.source_root,
            args.recovery_qualification, args.recovery_qualification_work)


if __name__ == "__main__":
    main()
