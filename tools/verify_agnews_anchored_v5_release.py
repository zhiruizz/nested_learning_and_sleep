#!/usr/bin/env python3
"""Verify the staged, JSON/Markdown-only AGNews anchored-v5 release.

All release inputs are read from Git's stage-0 index blobs (`git show :path`),
never from the working tree.  The verifier loads no model, checkpoint, dataset,
token pack, or generated sequence.  It recomputes raw-seal metric summaries,
paired contrasts, gates, event/capacity accounting, and WorkMeter totals, then
checks the frozen analyzer's JSON and Markdown outputs against those values.

Run from a repository checkout, for example:
  python tools/verify_agnews_anchored_v5_release.py --repo-root .

Use --receipt PATH to write a source/hash validation receipt after all checks
pass.  The receipt is not read from the index and should be staged separately
only if desired.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys


CONFIG_PATH = "configs/agnews-anchored-comparison-v5.json"
RELEASE_DIR = "results/agnews-anchored-v5"
EXPECTED_CONFIG_SHA256 = "44e12f283cc4d933ce133b1ab5e51f6a04376d1bda0361a7b2d674058cf55e7c"
EXPECTED_DRIVER_SHA256 = "b0ae6daaeb6d18d17ef2fe96972826e6cc685d7f8a81845d1226d9f2725bed7d"
EXPECTED_PRIVATE_SEAL_SHA256 = "45e44b6bd0479234c0aa2b1f491315918476529b01d33fb27678293481fa2253"

SEEDS = (17, 29, 43)
ARMS = (
    "warm_CMS_reference",
    "warm_anchorKD",
    "warm_anchorGold",
    "warm_anchorKD_sleepP16_latest",
    "warm_anchorKD_sleepP16_warmmix",
    "warm_self_sleepP16_latest",
    "warm_self_sleepP16_warmmix",
)
STEPS = (0, 16, 64, 112, 256, 448)
SLEEP_ARMS = frozenset((
    "warm_anchorKD_sleepP16_latest", "warm_anchorKD_sleepP16_warmmix",
    "warm_self_sleepP16_latest", "warm_self_sleepP16_warmmix",
))
KD_ARMS = frozenset((
    "warm_anchorKD", "warm_anchorKD_sleepP16_latest", "warm_anchorKD_sleepP16_warmmix",
))
GOLD_ARMS = frozenset(("warm_anchorGold",))
WARM_MIX_ARMS = frozenset(("warm_anchorKD_sleepP16_warmmix", "warm_self_sleepP16_warmmix"))
GROUPS = ("old", "new", "all", "class0", "class1", "class2", "class3")
METRICS = (
    "full_vocab_two_token_greedy_exact_match", "answer_nll_nats_per_token", "marker_nll",
    "class_nll", "marker_accuracy", "class_given_marker_accuracy", "candidate_accuracy",
)
EM = METRICS[0]
NLL = METRICS[1]
LABELS = {
    "warm_CMS_reference": "CMS reference",
    "warm_anchorKD": "Anchor KD",
    "warm_anchorGold": "Anchor Gold",
    "warm_anchorKD_sleepP16_latest": "KD Sleep latest",
    "warm_anchorKD_sleepP16_warmmix": "KD Sleep warmmix",
    "warm_self_sleepP16_latest": "Self Sleep latest",
    "warm_self_sleepP16_warmmix": "Self Sleep warmmix",
}
CONTRASTS = (
    ("KD - reference", "warm_anchorKD", "warm_CMS_reference"),
    ("KD latest - self latest", "warm_anchorKD_sleepP16_latest", "warm_self_sleepP16_latest"),
    ("KD warm - self warm", "warm_anchorKD_sleepP16_warmmix", "warm_self_sleepP16_warmmix"),
    ("KD latest - KD", "warm_anchorKD_sleepP16_latest", "warm_anchorKD"),
    ("KD warm - KD", "warm_anchorKD_sleepP16_warmmix", "warm_anchorKD"),
    ("self latest - reference", "warm_self_sleepP16_latest", "warm_CMS_reference"),
    ("self warm - reference", "warm_self_sleepP16_warmmix", "warm_CMS_reference"),
    ("KD warm - KD latest", "warm_anchorKD_sleepP16_warmmix", "warm_anchorKD_sleepP16_latest"),
    ("self warm - self latest", "warm_self_sleepP16_warmmix", "warm_self_sleepP16_latest"),
    ("Gold - KD", "warm_anchorGold", "warm_anchorKD"),
    ("Gold - reference", "warm_anchorGold", "warm_CMS_reference"),
)
INTERACTIONS = (
    ("(KD warm - self warm) - (KD latest - self latest)",
     (("warm_anchorKD_sleepP16_warmmix", "warm_self_sleepP16_warmmix"),
      ("warm_anchorKD_sleepP16_latest", "warm_self_sleepP16_latest"))),
    ("(KD latest - self latest) - (KD - reference)",
     (("warm_anchorKD_sleepP16_latest", "warm_self_sleepP16_latest"),
      ("warm_anchorKD", "warm_CMS_reference"))),
)
TABLES = (
    "endpoint-table.md", "trajectory-table.md", "per-seed-endpoints.md",
    "paired-endpoints.md", "work-ledger.md",
)
FIGURES = ("endpoint_em.png", "endpoint_em.svg", "trajectory_em.png", "trajectory_em.svg")
VALIDATION_NAME = "validation.json"
CHECKPOINT_READBACK_NAME = "private-checkpoint-readback.json"
FORBIDDEN_NEW_SUFFIXES = frozenset((
    ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".model", ".npy", ".npz",
    ".arrow", ".parquet", ".jsonl", ".csv", ".tsv",
))
FORBIDDEN_RAW_KEYS = frozenset((
    "tokens", "token_ids", "input_ids", "prompt_ids", "generated_sequence",
    "generated_sequences", "teacher_data", "prompt_text", "raw_prompt", "record_text",
))


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def close(a, b, *, path: str, tol: float = 1e-12) -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        require(a is b, f"{path}: boolean mismatch {a!r} != {b!r}")
        return
    require(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol),
            f"{path}: {a!r} != recomputed {b!r}")


def assert_tree_close(actual, expected, path="root", tol=1e-12) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{path}: expected object")
        require(set(actual) == set(expected), f"{path}: object keys differ")
        for key in expected:
            assert_tree_close(actual[key], expected[key], f"{path}.{key}", tol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{path}: list differs")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_tree_close(a, e, f"{path}[{i}]", tol)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")
    elif isinstance(expected, (int, float)):
        close(actual, expected, path=path, tol=tol)
    else:
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")


def stats(values_by_seed: dict[int, float]) -> dict:
    values = [float(values_by_seed[s]) for s in SEEDS]
    return {
        "per_seed": {str(s): values_by_seed[s] for s in SEEDS},
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def run_git(root: Path, *args: str, check=True) -> bytes:
    proc = subprocess.run(["git", *args], cwd=root, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if check and proc.returncode:
        raise VerificationError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            + proc.stderr.decode("utf-8", "replace").strip()
        )
    return proc.stdout


def repo_root(path: Path) -> Path:
    out = run_git(path, "rev-parse", "--show-toplevel").decode().strip()
    return Path(out).resolve()


def staged_blob(root: Path, rel: str) -> bytes:
    # The colon syntax explicitly reads the index blob (stage 0), not the file.
    return run_git(root, "show", f":{rel}")


def staged_json(root: Path, rel: str):
    try:
        return json.loads(staged_blob(root, rel).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid staged JSON at {rel}: {exc}") from exc


def index_entries(root: Path) -> dict[str, tuple[str, int, str]]:
    raw = run_git(root, "ls-files", "--stage", "-z")
    entries = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        meta, rel = item.split(b"\t", 1)
        mode, oid, stage = meta.decode("ascii").split()
        entries[os.fsdecode(rel)] = (mode, int(stage), oid)
    return entries


def head_entries(root: Path, prefix: str) -> dict[str, str]:
    raw = run_git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", prefix)
    entries = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        meta, rel = item.split(b"\t", 1)
        fields = meta.decode("ascii").split()
        if len(fields) == 3 and fields[1] == "blob":
            entries[os.fsdecode(rel)] = fields[2]
    return entries


def staged_paths(root: Path) -> list[str]:
    raw = run_git(root, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRTUXB")
    return [os.fsdecode(x) for x in raw.split(b"\0") if x]


def verify_index_stage_zero(root: Path, rel: str, entries: dict) -> None:
    require(rel in entries, f"required staged path missing: {rel}")
    _, stage, _ = entries[rel]
    require(stage == 0, f"required path has unresolved merge stage: {rel}")


def preserve_index_paths(root: Path, entries: dict, *, prefixes=(), paths=()) -> dict:
    """Require historical tracked paths to be byte-identical to HEAD/index."""
    baseline = {}
    for prefix in prefixes:
        baseline.update(head_entries(root, prefix))
    for rel in paths:
        # Explicit path may not exist in a particular older branch; if it does,
        # it is preserved. Config-locked paths are separately required below.
        raw = run_git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", rel)
        for item in raw.split(b"\0"):
            if not item:
                continue
            meta, p = item.split(b"\t", 1)
            fields = meta.decode("ascii").split()
            if len(fields) == 3 and fields[1] == "blob":
                baseline[os.fsdecode(p)] = fields[2]
    for rel, head_oid in baseline.items():
        require(rel in entries, f"historical tracked evidence deleted from index: {rel}")
        mode, stage, index_oid = entries[rel]
        require(stage == 0, f"historical file has unresolved merge stage: {rel}")
        require(index_oid == head_oid, f"historical source/evidence modified in index: {rel}")
    for rel in entries:
        if any(rel.startswith(prefix) for prefix in prefixes):
            require(rel in baseline, f"unexpected new file inside preserved history: {rel}")
    return {"preserved_paths": len(baseline), "prefixes": list(prefixes)}


def metric(metrics: dict, group: str, name: str) -> float:
    return float(metrics[group][name])


def validate_metric_counts(metrics: dict, total: int, old: int, new: int, each: int, where: str) -> None:
    require(metrics["all"]["examples"] == total, f"{where}: all example count")
    require(metrics["old"]["examples"] == old and metrics["new"]["examples"] == new,
            f"{where}: old/new example counts")
    require(all(metrics[f"class{k}"]["examples"] == each for k in range(4)),
            f"{where}: per-class example counts")


def score(bykey: dict, seed: int, arm: str, step: int, group: str, name: str) -> float:
    run = bykey[(seed, arm)]
    point = next(x for x in run["validation"] if int(x["primary_step"]) == step)
    return metric(point["metrics"], group, name)


def compute_endpoint_and_trajectory(bykey: dict) -> tuple[dict, dict]:
    trajectory = {}
    endpoint = {}
    for arm in ARMS:
        trajectory[arm] = {}
        for step in STEPS:
            groups = {}
            for group in GROUPS:
                groups[group] = {}
                for name in METRICS:
                    vals = {s: score(bykey, s, arm, step, group, name) for s in SEEDS}
                    groups[group][name] = stats(vals)
            trajectory[arm][str(step)] = groups
        endpoint[arm] = trajectory[arm][str(STEPS[-1])]
    return endpoint, trajectory


def compare_stats_obj(actual: dict, expected: dict, path: str) -> None:
    require(isinstance(actual, dict), f"{path}: summary not object")
    require(actual.get("n") == expected["n"] == 3, f"{path}: expected n=3")
    close(actual.get("mean"), expected["mean"], path=path + ".mean")
    close(actual.get("sample_sd"), expected["sample_sd"], path=path + ".sample_sd")
    assert_tree_close(actual.get("per_seed"), expected["per_seed"], path + ".per_seed")


def compute_paired(bykey: dict) -> tuple[list, list, list]:
    pairs = []
    for label, a, b in CONTRASTS:
        trajectory = []
        for step in STEPS:
            groups = {}
            for group in GROUPS:
                groups[group] = {}
                for name in METRICS:
                    vals = {s: score(bykey, s, a, step, group, name) -
                            score(bykey, s, b, step, group, name) for s in SEEDS}
                    groups[group][name] = stats(vals)
            trajectory.append({"step": step, "metrics": groups})
        pairs.append({"contrast": label, "minuend": a, "subtrahend": b,
                      "paired_n": 3, "trajectory": trajectory})
    interactions = []
    for label, ((a, b), (c, d)) in INTERACTIONS:
        trajectory = []
        for step in STEPS:
            groups = {}
            for group in GROUPS:
                groups[group] = {}
                for name in METRICS:
                    vals = {s: (score(bykey, s, a, step, group, name) - score(bykey, s, b, step, group, name)) -
                            (score(bykey, s, c, step, group, name) - score(bykey, s, d, step, group, name))
                            for s in SEEDS}
                    groups[group][name] = stats(vals)
            trajectory.append({"step": step, "metrics": groups})
        interactions.append({"interaction": label, "paired_n": 3, "trajectory": trajectory})
    train_pairs = []
    for label, a, b in CONTRASTS:
        groups = {}
        for group in GROUPS:
            groups[group] = {}
            for name in METRICS:
                vals = {s: metric(bykey[(s, a)]["train_final"], group, name) -
                        metric(bykey[(s, b)]["train_final"], group, name) for s in SEEDS}
                groups[group][name] = stats(vals)
        train_pairs.append({"contrast": label, "minuend": a, "subtrahend": b,
                            "paired_n": 3, "train_final_metrics": groups})
    return pairs, interactions, train_pairs


def verify_analysis_metrics(analysis: dict, bykey: dict) -> None:
    endpoint, trajectory = compute_endpoint_and_trajectory(bykey)
    require(analysis.get("validation_states") == 126, "analysis validation state total")
    assert_tree_close(analysis.get("validation_examples_per_state"),
                      {"all": 512, "old": 256, "new": 256, "each_class": 128},
                      "analysis.validation_examples_per_state")
    for arm in ARMS:
        for group in GROUPS:
            for name in METRICS:
                compare_stats_obj(analysis["endpoint_step448_mean_sample_sd"][arm][group][name],
                                  endpoint[arm][group][name],
                                  f"analysis.endpoint.{arm}.{group}.{name}")
        for step in STEPS:
            for group in GROUPS:
                for name in METRICS:
                    compare_stats_obj(analysis["trajectory_mean_sample_sd"][arm][str(step)][group][name],
                                      trajectory[arm][str(step)][group][name],
                                      f"analysis.trajectory.{arm}.{step}.{group}.{name}")
    expected_pairs, expected_interactions, expected_train_pairs = compute_paired(bykey)
    for key, expected in (("paired_contrasts", expected_pairs),
                          ("paired_interactions", expected_interactions),
                          ("paired_endpoint_train_contrasts", expected_train_pairs)):
        assert_tree_close(analysis.get(key), expected, "analysis." + key)
    per_seed = analysis.get("per_seed_runs")
    require(isinstance(per_seed, list) and len(per_seed) == 21, "analysis per-seed run count")
    per_seed_by_key = {(int(x["seed"]), x["arm"]): x for x in per_seed}
    require(set(per_seed_by_key) == set(bykey), "analysis per-seed grid")
    for (seed, arm), run in bykey.items():
        row = per_seed_by_key[(seed, arm)]
        require(int(row["primary_updates"]) == 448
                and int(row["sleep_Adam_updates"]) == int(run["sleep_Adam_updates"])
                and int(row["sleep_events"]) == len(run["events"]),
                f"analysis per-seed cost row {seed}/{arm}")
        assert_tree_close(row.get("qualification"), run.get("qualification"),
                          f"analysis.per_seed.{seed}.{arm}.qualification")
        assert_tree_close(row.get("train_final"), {
            g: {m: metric(run["train_final"], g, m) for m in METRICS} for g in GROUPS
        }, f"analysis.per_seed.{seed}.{arm}.train_final")
        for step in STEPS:
            point = next(x for x in run["validation"] if int(x["primary_step"]) == step)
            expected_metrics = {g: {m: metric(point["metrics"], g, m) for m in METRICS} for g in GROUPS}
            assert_tree_close(row["trajectory"][str(step)], expected_metrics,
                              f"analysis.per_seed.{seed}.{arm}.trajectory.{step}")
        initial_old = score(bykey, seed, arm, 0, "old", EM)
        final_old = score(bykey, seed, arm, 448, "old", EM)
        initial_new = score(bykey, seed, arm, 0, "new", EM)
        final_new = score(bykey, seed, arm, 448, "new", EM)
        for field, expected in (
            ("initial_old_em", initial_old), ("final_old_em", final_old),
            ("old_em_forgetting_step0_minus_step448", initial_old-final_old),
            ("initial_new_em", initial_new), ("final_new_em", final_new),
            ("new_em_change_step448_minus_step0", final_new-initial_new),
        ):
            close(row[field], expected, path=f"analysis.per_seed.{seed}.{arm}.{field}")
    gates_analysis = analysis.get("task_gates", {})
    for arm in ARMS:
        gate_summary = gates_analysis.get(arm, {})
        counts = {"old_pass_seeds": 0, "new_pass_seeds": 0, "both_pass_seeds": 0}
        for seed in SEEDS:
            raw_gate = bykey[(seed, arm)]["qualification"]["gates"]
            both = bool(raw_gate["old"]["pass"] and raw_gate["new"]["pass"])
            counts["old_pass_seeds"] += int(bool(raw_gate["old"]["pass"]))
            counts["new_pass_seeds"] += int(bool(raw_gate["new"]["pass"]))
            counts["both_pass_seeds"] += int(both)
            expected_row = {"old": raw_gate["old"], "new": raw_gate["new"], "both_pass": both}
            assert_tree_close(gate_summary.get("per_seed", {}).get(str(seed)), expected_row,
                              f"analysis.task_gates.{arm}.{seed}")
        for name, value in counts.items():
            require(gate_summary.get(name) == value, f"analysis.task_gates.{arm}.{name}")
    retention = analysis.get("retention_and_acquisition_diagnostics", {})
    for arm in ARMS:
        values = {s: {
            "initial_old_em": score(bykey, s, arm, 0, "old", EM),
            "final_old_em": score(bykey, s, arm, 448, "old", EM),
            "old_em_forgetting_step0_minus_step448": score(bykey, s, arm, 0, "old", EM)-score(bykey, s, arm, 448, "old", EM),
            "initial_new_em": score(bykey, s, arm, 0, "new", EM),
            "final_new_em": score(bykey, s, arm, 448, "new", EM),
            "new_em_change_step448_minus_step0": score(bykey, s, arm, 448, "new", EM)-score(bykey, s, arm, 0, "new", EM),
        } for s in SEEDS}
        for field in values[SEEDS[0]]:
            compare_stats_obj(retention.get(arm, {}).get(field),
                              stats({s: values[s][field] for s in SEEDS}),
                              f"analysis.retention.{arm}.{field}")
    require(analysis.get("inference_limits", {}).get("n_seeds") == 3, "analysis seed count")
    require(analysis["inference_limits"].get("sample_sd_ddof") == 1, "analysis sample SD definition")
    for key in ("p_values", "independence_claim", "endpoint_selection"):
        require(analysis["inference_limits"].get(key) is False,
                f"analysis must not claim {key}")


def validate_gates(run: dict, cfg: dict, where: str) -> dict:
    thresholds = cfg["task_gates"]
    gates = run.get("qualification", {}).get("gates")
    require(isinstance(gates, dict) and set(gates) == {"old", "new"}, f"{where}: missing gates")
    final_val = run["validation"][-1]["metrics"]
    computed = {}
    for phase in ("old", "new"):
        classes = (0, 1) if phase == "old" else (2, 3)
        train_em = metric(run["train_final"], phase, EM)
        val_em = metric(final_val, phase, EM)
        class_em = {str(k): metric(final_val, f"class{k}", EM) for k in classes}
        exp = {
            "train_phase_em": train_em,
            "train_phase_pass": train_em >= float(thresholds["train_EM"]),
            "validation_phase_em": val_em,
            "validation_phase_pass": val_em >= float(thresholds["validation_EM"]),
            "validation_class_em": class_em,
            "validation_class_pass": {str(k): class_em[str(k)] >= float(thresholds["each_class_validation_EM"])
                                      for k in classes},
        }
        exp["pass"] = (exp["train_phase_pass"] and exp["validation_phase_pass"]
                       and all(exp["validation_class_pass"].values()))
        assert_tree_close(gates[phase], exp, f"{where}.qualification.gates.{phase}")
        computed[phase] = exp
    qualification = run["qualification"]
    require(qualification.get("all_tasks_pass") is
            (computed["old"]["pass"] and computed["new"]["pass"]),
            f"{where}: all_tasks_pass inconsistent with raw metrics")
    return computed


def validate_run_grid(seal: dict, protocol: dict, cfg: dict, v4: dict, warm: dict) -> tuple[dict, dict]:
    require(seal.get("schema") == "agnews-anchored-development-seal-v5" and seal.get("complete") is True,
            "incomplete or unknown main seal")
    require(seal.get("development_only") is True, "seal not labeled development-only")
    require(seal.get("confirmation_scored") is False and seal.get("no_holdout_input") is True,
            "seal reports confirmation or holdout access")
    require(protocol.get("confirmation_input") is False and protocol.get("confirmation_scored") is False,
            "protocol allows confirmation use")
    runs = seal.get("runs")
    require(isinstance(runs, list) and len(runs) == 21, "expected 21 complete runs")
    warm_by_seed = {int(x["seed"]): x for x in warm.get("qualified_models", [])}
    require(set(warm_by_seed) == set(SEEDS), "qualified warm checkpoint seed set")
    bykey = {}
    for run in runs:
        seed, arm = int(run["seed"]), run["arm"]
        key = (seed, arm)
        require(seed in SEEDS and arm in ARMS and key not in bykey, f"unknown/duplicate run {key}")
        require(run.get("status") == "complete" and run.get("primary_updates") == 448,
                f"incomplete/incorrect primary budget: {key}")
        require(run.get("attempted_primary_updates") == 448, f"attempt ledger mismatch: {key}")
        require([int(x["primary_step"]) for x in run.get("validation", [])] == list(STEPS),
                f"validation step grid: {key}")
        require([int(x["primary_step"]) for x in run.get("checkpoints", [])] == list(STEPS),
                f"checkpoint step grid: {key}")
        for point in run["validation"]:
            validate_metric_counts(point["metrics"], 512, 256, 256, 128, f"{key}/val{point['primary_step']}")
            require(point.get("controller_counts") == [0, 0], f"checkpoint clock not clear: {key}/{point['primary_step']}")
        validate_metric_counts(run.get("train_final", {}), 1024, 512, 512, 256, f"{key}/train_final")
        require(run["qualification"].get("gate_thresholds") == {
            "train_em": cfg["task_gates"]["train_EM"],
            "validation_em": cfg["task_gates"]["validation_EM"],
            "each_class_validation_em": cfg["task_gates"]["each_class_validation_EM"],
        }, f"{key}: gate thresholds differ from config")
        validate_gates(run, cfg, str(key))
        require(isinstance(run.get("immutable_teacher_fingerprint"), str)
                and len(run["immutable_teacher_fingerprint"]) == 64, f"{key}: missing teacher fingerprint")
        require(run.get("initial_adapter_fingerprint") == run["validation"][0].get("adapter_fingerprint"),
                f"{key}: step-zero fingerprint mismatch")
        warm_fingerprint = warm_by_seed[seed].get("checkpoint", {}).get("adapter_fingerprint")
        require(run.get("initial_adapter_fingerprint") == warm_fingerprint
                and run.get("immutable_teacher_fingerprint") == warm_fingerprint,
                f"{key}: warm model/teacher fingerprint mismatch")
        require(run["checkpoints"][0].get("adapter_fingerprint") == run["initial_adapter_fingerprint"],
                f"{key}: step-zero checkpoint fingerprint mismatch")
        require(run["checkpoints"][0].get("controller_counts") == [0, 0], f"{key}: initial clock state")
        for ckpt, point in zip(run["checkpoints"], run["validation"]):
            require(ckpt.get("controller_counts") == point.get("controller_counts"),
                    f"{key}/step{point['primary_step']}: checkpoint/validation clocks differ")
        require(len(run.get("old_aux_schedule_sha256", "")) == 64, f"{key}: missing old-aux schedule hash")
        provenance = run.get("old_receiver_provenance", {})
        require(provenance.get("receiver_zero_equal_warm") is True
                and provenance.get("warm_active_slot_prefix_equal") is True,
                f"{key}: old receiver prefix provenance did not pass")
        require(provenance.get("warm_active_slot_prefix_count") == 1,
                f"{key}: expected exactly one warm receiver prefix slot")
        bykey[key] = run

    require(set(bykey) == {(s, a) for s in SEEDS for a in ARMS}, "seed/arm grid incomplete")
    # A seed's seven arms must begin from the same warm model, policy, RNG and
    # fixed immutable teacher.  Step-zero metrics are compared as complete JSON.
    starts = {}
    for seed in SEEDS:
        seed_runs = [bykey[(seed, arm)] for arm in ARMS]
        ref = seed_runs[0]
        ref_v0 = ref["validation"][0]
        for run in seed_runs:
            v0 = run["validation"][0]
            require(run["initial_adapter_fingerprint"] == ref["initial_adapter_fingerprint"],
                    f"seed {seed}: arms do not share initial model")
            require(run["immutable_teacher_fingerprint"] == ref["immutable_teacher_fingerprint"],
                    f"seed {seed}: arms do not share immutable teacher")
            assert_tree_close(v0.get("execution_policy"), ref_v0.get("execution_policy"),
                              f"seed{seed}.initial.execution_policy")
            require(v0.get("rng_sha256") == ref_v0.get("rng_sha256"), f"seed {seed}: initial RNG differs")
            assert_tree_close(v0.get("metrics"), ref_v0.get("metrics"), f"seed{seed}.initial.metrics")
            require(v0.get("controller_counts") == [0, 0], f"seed {seed}: initial clock state")
        starts[str(seed)] = {
            "adapter_fingerprint": ref["initial_adapter_fingerprint"],
            "teacher_fingerprint": ref["immutable_teacher_fingerprint"],
            "execution_policy": ref_v0["execution_policy"],
            "rng_sha256": ref_v0["rng_sha256"],
        }

    # The no-Sleep CMS arm is an exact v4 replay at the two locked checkpoints.
    for seed in SEEDS:
        run = bykey[(seed, "warm_CMS_reference")]
        checks = run.get("v4_reference_checks", [])
        require([int(x.get("step", -1)) for x in checks] == [16, 112],
                f"seed {seed}: expected v4 reference checks at 16 and 112")
        old_run = next((x for x in v4.get("runs", [])
                        if int(x.get("seed", -1)) == seed and x.get("arm") == "warm_CMS_no_sleep"), None)
        require(old_run is not None, f"seed {seed}: v4 no-Sleep reference missing")
        for check in checks:
            step = int(check["step"])
            for flag in ("checkpoint_model_equal", "execution_policy_equal", "controller_equal",
                         "rng_equal", "validation_metrics_equal"):
                require(check.get(flag) is True, f"seed {seed}/step{step}: v4 {flag} failed")
            old_cp = next(x for x in old_run["checkpoints"] if int(x["primary_step"]) == step)
            require(check.get("v4_checkpoint_sha256") == old_cp.get("sha256"),
                    f"seed {seed}/step{step}: referenced v4 checkpoint hash mismatch")
            new_point = next(x for x in run["validation"] if int(x["primary_step"]) == step)
            old_point = next(x for x in old_run["validation"] if int(x["primary_step"]) == step)
            require(old_cp.get("controller_counts") is not None
                    and new_point.get("controller_counts") == old_cp.get("controller_counts"),
                    f"seed {seed}/step{step}: controller counts differ from v4 checkpoint metadata")
            assert_tree_close(new_point["metrics"], old_point["metrics"], f"seed{seed}.v4ref{step}.metrics")
            require(new_point.get("adapter_fingerprint") == old_point.get("adapter_fingerprint"),
                    f"seed {seed}/step{step}: adapter fingerprint differs from v4")
            # The public v4 seal predates per-validation RNG digests.  Its
            # matched-checkpoint evidence records the exact in-memory
            # comparison as rng_equal; require that explicit positive result
            # when the v4 point has no digest, and compare digests whenever it
            # does expose one.
            if old_point.get("rng_sha256") is not None:
                require(new_point.get("rng_sha256") == old_point.get("rng_sha256"),
                        f"seed {seed}/step{step}: RNG differs from v4")
            else:
                require(check.get("rng_equal") is True,
                        f"seed {seed}/step{step}: v4 RNG equality was not established")
            assert_tree_close(new_point.get("execution_policy"), old_point.get("execution_policy"),
                              f"seed{seed}.v4ref{step}.policy")

    expected_event_keys = [str(x) for x in range(16, 449, 16)]
    for seed in SEEDS:
        sleep_hashes = protocol.get("sleep_pool_hashes_by_seed", {}).get(str(seed), {})
        require(set(sleep_hashes) == set(expected_event_keys) and all(len(str(v)) == 64 for v in sleep_hashes.values()),
                f"seed {seed}: protocol event-pool hash schedule")
        aux_hash = protocol.get("old_aux_schedule_sha256_by_seed", {}).get(str(seed))
        require(len(str(aux_hash)) == 64, f"seed {seed}: protocol old-aux schedule hash")
        for arm in ARMS:
            require(bykey[(seed, arm)]["old_aux_schedule_sha256"] == aux_hash,
                    f"seed {seed}/{arm}: old-aux schedule hash mismatch")

    # Check fixed teacher use, due-boundary timing, receiver chronology and
    # train-time vs post-commit route policy.  anchor_old_aux_count is a
    # constructed row count even for CMS-only arms; cost is checked from modes
    # and per-stage meters below, not from that descriptive field.
    for (seed, arm), run in bykey.items():
        expected_sleep = arm in SLEEP_ARMS
        events = run.get("events", [])
        expected_boundaries = list(range(16, 449, 16)) if expected_sleep else []
        require([int(e["sleep"].get("primary_boundary", -1)) for e in events] == expected_boundaries,
                f"{seed}/{arm}: Sleep boundary grid")
        require(run.get("sleep_Adam_updates") == 24 * len(expected_boundaries),
                f"{seed}/{arm}: row Sleep Adam budget")
        active_final = run.get("final_active_parameters", {}).get("active_slots_by_bank")
        if expected_sleep:
            require(active_final == [0, 29], f"{seed}/{arm}: expected final active banks [0,29]")
        else:
            require(len(events) == 0 and run.get("sleep_Adam_updates") == 0,
                    f"{seed}/{arm}: unexpected Sleep event")
        mode = "soft_kd" if arm in KD_ARMS else "gold" if arm in GOLD_ARMS else "cms"
        require(len(run.get("validation", [])) == 6, f"{seed}/{arm}: expected 6 validation states")
        for event_index, event in enumerate(events, start=1):
            sm = event.get("sleep", {})
            wake = event.get("wake", {})
            boundary = event_index * 16
            require(sm.get("arm") == arm and int(sm.get("primary_boundary", -1)) == boundary,
                    f"{seed}/{arm}: event identity/boundary")
            require(sm.get("wake_mode") == mode and wake.get("mode") == mode,
                    f"{seed}/{arm}/step{boundary}: wake mode")
            require(sm.get("anchor_old_aux_count") == 4,
                    f"{seed}/{arm}/step{boundary}: constructed old-aux sample count")
            require(sm.get("scope") == "pre_fast_base_update_sleep_v4"
                    and sm.get("teacher_order") == "pre_fast", f"{seed}/{arm}/step{boundary}: Sleep timing scope")
            require(sm.get("input_fast_count") == 4 and sm.get("input_slow_count") == 16,
                    f"{seed}/{arm}/step{boundary}: due CMS clocks")
            require(sm.get("slow_level1_applied_before_teacher") is True
                    and sm.get("fast_level0_applied_before_teacher") is False
                    and sm.get("fast_level0_applied_prospectively") is True,
                    f"{seed}/{arm}/step{boundary}: teacher/base-update order")
            require(sm.get("sender_level") == 0 and sm.get("receiver_level") == 1
                    and sm.get("sender_reset_slot_ids") == [0],
                    f"{seed}/{arm}/step{boundary}: sender reset/receiver levels")
            require(sm.get("output_controller_counts") == [0, 0]
                    and sm.get("candidate_committed") is True,
                    f"{seed}/{arm}/step{boundary}: Sleep commit/clock state")
            require(sm.get("total_adam_updates") == 24 and sm.get("gkd_updates") == 16
                    and sm.get("lti_updates") == 8 and sm.get("dream_updates") == 0,
                    f"{seed}/{arm}/step{boundary}: Sleep optimizer schedule")
            require(sm.get("gkd_teacher_updates") == 8 and sm.get("gkd_student_on_policy_updates") == 8,
                    f"{seed}/{arm}/step{boundary}: GKD teacher/student schedule")
            slot = event_index
            require(sm.get("receiver_slot") == slot, f"{seed}/{arm}/step{boundary}: receiver slot chronology")
            require(sm.get("receiver_active_slots") == list(range(slot + 1)),
                    f"{seed}/{arm}/step{boundary}: active receiver prefix")
            require(sm.get("active_slots_by_bank") == [0, slot + 1],
                    f"{seed}/{arm}/step{boundary}: active slot counts")
            trained_policy = {"mix": slot, "fraction": 0.5}
            post_policy = {"mix": 0, "fraction": 0.5} if arm in WARM_MIX_ARMS else trained_policy
            require(sm.get("receiver_mix_fraction") == 0.5
                    and sm.get("receiver_route_policy") == trained_policy,
                    f"{seed}/{arm}/step{boundary}: Sleep-time receiver policy")
            require(sm.get("receiver_route_after_commit") == post_policy,
                    f"{seed}/{arm}/step{boundary}: post-commit receiver policy")
            require(sm.get("planned_pool_class_counts") == {str(k): 6 for k in range(4)},
                    f"{seed}/{arm}/step{boundary}: sleep pool class counts")
            require(wake.get("due_levels") == [0, 1] and wake.get("controller_counts") == [4, 16],
                    f"{seed}/{arm}/step{boundary}: due proposals not recorded")
            if "logical_primary_updates" in wake:
                require(wake.get("logical_primary_updates") == 1,
                        f"{seed}/{arm}/step{boundary}: wake logical clock count")
            if mode == "soft_kd":
                require(wake.get("logical_primary_updates") == 1
                        and wake.get("proposals_applied") is False
                        and wake.get("global_rng_unchanged") is True,
                        f"{seed}/{arm}/step{boundary}: auxiliary wake mutated proposals or global RNG")
            if mode in ("soft_kd", "gold"):
                require(wake.get("teacher_unchanged") is True,
                        f"{seed}/{arm}/step{boundary}: wake teacher changed")
            if mode == "soft_kd":
                require(len(str(wake.get("teacher_adapter_fingerprint", ""))) == 64,
                        f"{seed}/{arm}/step{boundary}: missing fixed auxiliary-teacher fingerprint")
            require(len(sm.get("initial_student_fingerprint", "")) == 64
                    and len(sm.get("teacher_fingerprint", "")) == 64
                    and len(sm.get("committed_student_fingerprint", "")) == 64,
                    f"{seed}/{arm}/step{boundary}: Sleep fingerprints missing")
    # Both digests describe the same frozen qualified warm-teacher clone.
    # Wake uses tensor_tree_hash(adapter_state_dict), whereas the run-level
    # lifecycle fingerprint uses a different byte encoding of those tensors.
    for seed in SEEDS:
        kd_teacher_fps = {
            event["wake"].get("teacher_adapter_fingerprint")
            for arm in KD_ARMS
            for event in bykey[(seed, arm)]["events"]
        }
        require(len(kd_teacher_fps) == 1 and len(str(next(iter(kd_teacher_fps)))) == 64,
                f"{seed}: soft-KD arms do not share one fixed auxiliary teacher")
    return bykey, starts


def merge_work(runs: list[dict]) -> dict:
    merged = {}
    for run in runs:
        work = run.get("work")
        require(isinstance(work, dict), f"{run.get('seed')}/{run.get('arm')}: no per-run work")
        for stage, counters in work.items():
            target = merged.setdefault(stage, {})
            for name, value in counters.items():
                require(isinstance(value, (int, float)) and not isinstance(value, bool),
                        f"nonnumeric meter counter {stage}/{name}")
                target[name] = target.get(name, 0) + value
    return merged


def verify_work(seal: dict, protocol: dict, analysis: dict, ledger: dict, bykey: dict) -> dict:
    require(seal.get("actual_primary_updates") == 9408 and
            sum(r["primary_updates"] for r in bykey.values()) == 9408, "primary update total")
    require(seal.get("actual_sleep_Adam_updates") == 8064 and
            sum(r["sleep_Adam_updates"] for r in bykey.values()) == 8064, "Sleep Adam total")
    require(sum(len(r["events"]) for r in bykey.values()) == 336, "Sleep event total")
    require(seal.get("actual_kd_old_aux_steps") == 4032 and
            seal.get("actual_gold_old_aux_steps") == 1344, "old-auxiliary step totals")
    require(protocol.get("planned_primary_updates") == 9408
            and protocol.get("planned_sleep_events") == 336
            and protocol.get("planned_sleep_adam_updates") == 8064,
            "protocol primary/Sleep budget")
    require(protocol.get("planned_kd_old_aux_steps") == 4032
            and protocol.get("planned_gold_old_aux_steps") == 1344,
            "protocol old-auxiliary budget")
    # Verify the primary-stage forward-call proxy against each objective mode.
    expected_primary_calls = {"cms": 448, "soft_kd": 4 * 448, "gold": 2 * 448}
    for (seed, arm), run in bykey.items():
        mode = "soft_kd" if arm in KD_ARMS else "gold" if arm in GOLD_ARMS else "cms"
        stage = f"v5_seed{seed}_{arm}_primary"
        counters = run["work"].get(stage)
        require(counters is not None, f"missing primary meter stage {stage}")
        require(counters.get("forward_calls") == expected_primary_calls[mode],
                f"{seed}/{arm}: primary forward-call count inconsistent with {mode}")
        require(counters.get("gradient_forward_input_tokens", 0) > 0,
                f"{seed}/{arm}: primary gradient-forward meter is empty")
        allowed_prefixes = (f"v5_seed{seed}_{arm}_", f"v5_validation_seed{seed}_{arm}_",
                            f"v5_train_seed{seed}_{arm}_")
        require(all(any(stage_name.startswith(prefix) for prefix in allowed_prefixes)
                    for stage_name in run["work"]),
                f"{seed}/{arm}: per-run work contains a foreign stage")
    for arm in ARMS:
        subset = [bykey[(s, arm)] for s in SEEDS]
        expected_events = 84 if arm in SLEEP_ARMS else 0
        expected_adam = 2016 if arm in SLEEP_ARMS else 0
        require(sum(r["primary_updates"] for r in subset) == 1344, f"{arm}: primary arm total")
        require(sum(len(r["events"]) for r in subset) == expected_events, f"{arm}: event arm total")
        require(sum(r["sleep_Adam_updates"] for r in subset) == expected_adam, f"{arm}: Adam arm total")
        arm_cost = analysis.get("cost", {}).get("by_arm", {}).get(arm)
        require(isinstance(arm_cost, dict) and arm_cost.get("runs") == 3,
                f"{arm}: analysis per-arm work row missing")
        require(arm_cost.get("primary_updates") == 1344
                and arm_cost.get("sleep_events") == expected_events
                and arm_cost.get("sleep_Adam_updates") == expected_adam,
                f"{arm}: analysis per-arm cost total")
        assert_tree_close(arm_cost.get("meter_work"), merge_work(subset), f"analysis.cost.by_arm.{arm}.meter_work")
        expected_seconds = stats({s: float(bykey[(s, arm)]["seconds"]) for s in SEEDS})
        seconds_summary = arm_cost.get("mean_seconds", {})
        for field in ("n", "mean", "sample_sd"):
            close(seconds_summary.get(field), expected_seconds[field],
                  path=f"analysis.cost.by_arm.{arm}.mean_seconds.{field}")
    union = merge_work(list(bykey.values()))
    assert_tree_close(seal.get("work"), union, "seal.work")
    cost = analysis.get("cost", {})
    assert_tree_close(cost.get("work_union"), union, "analysis.cost.work_union")
    assert_tree_close(cost.get("global_work"), seal["work"], "analysis.cost.global_work")
    require(cost.get("work_union_equals_seal") is True and cost.get("work_residual") == {},
            "analysis reports work residual")
    require(ledger.get("schema") == "agnews-anchored-work-ledger-v5", "unknown work ledger schema")
    assert_tree_close(ledger.get("cost"), cost, "work-ledger.cost")
    require(ledger.get("external_costs_not_in_seal") == cost.get("external_costs_not_in_seal"),
            "work ledger external-cost note mismatch")
    require(cost.get("primary_updates") == 9408 and cost.get("sleep_events") == 336
            and cost.get("sleep_Adam_updates") == 8064 and cost.get("kd_old_aux_steps") == 4032
            and cost.get("gold_old_aux_steps") == 1344,
            "analysis-derived headline cost totals")
    require(cost.get("kd_old_prefix_rows") == 16128 and cost.get("gold_old_prefix_rows") == 5376
            and cost.get("gold_old_answer_targets") == 5376,
            "analysis-derived old-prefix/answer-target totals")
    return {"primary_updates": 9408, "sleep_events": 336, "sleep_adam_updates": 8064,
            "kd_old_aux_steps": 4032, "gold_old_aux_steps": 1344,
            "run_work_equals_global": True,
            "objective_mode_primary_forward_calls": expected_primary_calls}


def fmt(stat: dict, percent=False, digits=3) -> str:
    scale = 100 if percent else 1
    return f"{stat['mean'] * scale:.{digits}f} ± {stat['sample_sd'] * scale:.{digits}f}"


def expected_tables(analysis: dict) -> dict[str, str]:
    endpoint = [
        "# Endpoint metrics at step 448", "",
        "EM and accuracy are percentages; NLL is nats per answer token. Entries are mean ± sample SD over the three paired seeds. Gates retain all seed outcomes.", "",
        "| Arm | Old EM | New EM | All EM | Old NLL | New NLL | Old pass | New pass | Both pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        e = analysis["endpoint_step448_mean_sample_sd"][arm]
        g = analysis["task_gates"][arm]
        endpoint.append(
            f"| {LABELS[arm]} | {fmt(e['old'][EM], True, 1)} | {fmt(e['new'][EM], True, 1)} | "
            f"{fmt(e['all'][EM], True, 1)} | {fmt(e['old'][NLL])} | {fmt(e['new'][NLL])} | "
            f"{g['old_pass_seeds']}/3 | {g['new_pass_seeds']}/3 | {g['both_pass_seeds']}/3 |"
        )
    trajectory = [
        "# Validation trajectory", "",
        "Exactly 42 arm-by-step mean rows; metric uncertainty is available in analysis.json as sample SD over seeds. The validation state count is 126 (21 runs × 6 checkpoints), with 512 examples per state.", "",
        "| Arm | Step | Old EM | New EM | All EM | Class 0 | Class 1 | Class 2 | Class 3 | Old NLL | New NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for step in STEPS:
            x = analysis["trajectory_mean_sample_sd"][arm][str(step)]
            vals = [fmt(x[g][EM], True, 1) for g in ("old", "new", "all", "class0", "class1", "class2", "class3")]
            trajectory.append("| " + LABELS[arm] + f" | {step} | " + " | ".join(vals +
                [fmt(x["old"][NLL]), fmt(x["new"][NLL])]) + " |")
    per_seed = [
        "# Per-seed endpoint values", "",
        "Validation endpoint at step 448 and the task gates for each seed; no seed is selected or dropped.", "",
        "| Seed | Arm | Old EM | New EM | All EM | Old NLL | New NLL | Initial old EM | Old EM forgetting | Old gate | New gate | Both |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in analysis["per_seed_runs"]:
        m = row["trajectory"]["448"]
        g = row["qualification"]["gates"]
        per_seed.append(
            f"| {row['seed']} | {LABELS[row['arm']]} | {m['old'][EM] * 100:.1f}% | "
            f"{m['new'][EM] * 100:.1f}% | {m['all'][EM] * 100:.1f}% | "
            f"{m['old'][NLL]:.3f} | {m['new'][NLL]:.3f} | {row['initial_old_em'] * 100:.1f}% | "
            f"{row['old_em_forgetting_step0_minus_step448'] * 100:.1f} pp | "
            f"{bool(g['old']['pass'])} | {bool(g['new']['pass'])} | {bool(row['qualification']['all_tasks_pass'])} |"
        )
    paired = [
        "# Paired endpoint contrasts", "",
        "Validation step 448. Each cell is the paired first-minus-second difference mean ± sample SD across seeds; interactions use the frozen signed formulas. No p-values are reported.", "",
        "| Contrast | Old EM Δ | New EM Δ | All EM Δ | Old NLL Δ | New NLL Δ | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    contrast_rows = [(x["contrast"], x["trajectory"][-1]["metrics"]) for x in analysis["paired_contrasts"]]
    contrast_rows += [(x["interaction"], x["trajectory"][-1]["metrics"]) for x in analysis["paired_interactions"]]
    for label, m in contrast_rows:
        paired.append(
            f"| {label} | {fmt(m['old'][EM], True, 1)} | {fmt(m['new'][EM], True, 1)} | "
            f"{fmt(m['all'][EM], True, 1)} | {fmt(m['old'][NLL])} | {fmt(m['new'][NLL])} | 3 |"
        )
    work = [
        "# Work ledger", "",
        "Source counters are retained by explicit row.work keys and checked against the global seal, including zero-valued counters. These token/call counts are compute proxies, not FLOPs.", "",
        f"- Primary wake updates: {analysis['cost']['primary_updates']}",
        f"- Sleep events / Adam updates: {analysis['cost']['sleep_events']} / {analysis['cost']['sleep_Adam_updates']}",
        f"- KD old auxiliary steps / context rows: {analysis['cost']['kd_old_aux_steps']} / {analysis['cost']['kd_old_prefix_rows']}",
        f"- Gold old auxiliary steps / context rows / answer targets: {analysis['cost']['gold_old_aux_steps']} / {analysis['cost']['gold_old_prefix_rows']} / {analysis['cost']['gold_old_answer_targets']}",
        "- The old subset is balanced using class metadata. KD uses frozen-teacher soft targets rather than old gold answers, but the balanced sampling still uses class labels; it is not wholly label-free.",
        "- KD and Gold auxiliary paths have different computation; comparisons do not isolate target type at equal FLOPs.",
        "- External run ledger must separately include the 48-update interference diagnostic, 18-update qualification and earlier failed probes.", "",
        "## Per-arm measured work", "",
        "| Arm | Primary updates | Sleep events | Sleep Adam updates | Seconds mean +/- SD | Metered stages/counters |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        x = analysis["cost"]["by_arm"][arm]
        encoded = json.dumps(x["meter_work"], sort_keys=True, separators=(",", ":"))
        work.append(
            f"| {LABELS[arm]} | {x['primary_updates']} | {x['sleep_events']} | {x['sleep_Adam_updates']} | "
            f"{fmt(x['mean_seconds'])} | `{encoded}` |"
        )
    return {
        "endpoint-table.md": "\n".join(endpoint) + "\n",
        "trajectory-table.md": "\n".join(trajectory) + "\n",
        "per-seed-endpoints.md": "\n".join(per_seed) + "\n",
        "paired-endpoints.md": "\n".join(paired) + "\n",
        "work-ledger.md": "\n".join(work) + "\n",
    }


def validate_config_and_protocol(root: Path, cfg: dict, protocol: dict,
                                 seal: dict, index: dict, release_files: dict) -> dict:
    cfg_bytes = staged_blob(root, CONFIG_PATH)
    cfg_hash = sha256(cfg_bytes)
    require(cfg_hash == EXPECTED_CONFIG_SHA256, "frozen config SHA changed")
    require(cfg.get("schema") == "agnews-anchored-comparison-config-v5"
            and cfg.get("review_status") == "FROZEN_DEVELOPMENT_AFTER_REVIEW",
            "config schema/review status")
    require(cfg.get("arms") == list(ARMS) and cfg.get("seeds") == list(SEEDS)
            and cfg.get("evaluation_steps") == list(STEPS), "config arm/seed/step grid")
    require(cfg.get("primary_steps") == 448 and cfg.get("sleep_period") == 16
            and cfg.get("capacity") == 32 and cfg.get("old_aux_weight") == 1,
            "config budget/period/capacity/anchor weight")
    require(cfg.get("confirmation_input") is False and cfg.get("confirmation_scored") is False,
            "config permits confirmation input/scoring")
    require(cfg.get("elapsed_limit_seconds") == 7200, "config elapsed cap")
    require(protocol.get("schema") == "agnews-anchored-comparison-protocol-v5"
            and protocol.get("status") == "frozen_development", "protocol schema/status")
    protocol_path = f"{RELEASE_DIR}/protocol.json"
    protocol_hash = sha256(release_files[protocol_path])
    require(seal.get("protocol_sha256") == protocol_hash, "seal/protocol staged hash mismatch")
    require(protocol.get("config_sha256") == cfg_hash and seal.get("config_sha256") == cfg_hash,
            "config/protocol/seal hash mismatch")
    require(protocol.get("config") == cfg, "protocol embedded config differs from staged config")
    require(protocol.get("arms") == list(ARMS) and protocol.get("seeds") == list(SEEDS)
            and protocol.get("evaluation_steps") == list(STEPS)
            and protocol.get("primary_steps") == 448, "protocol grid mismatch")
    require(protocol.get("confirmation_input") is False and protocol.get("confirmation_scored") is False,
            "protocol confirmation access flag")
    require(protocol.get("exploratory_only") is True and protocol.get("automatic_retry") is False,
            "protocol must be exploratory with retries disabled")
    try:
        frozen_cfg = _datetime.datetime.fromisoformat(cfg["frozen_utc"])
        frozen_protocol = _datetime.datetime.fromisoformat(protocol["frozen_before_training_utc"])
        seal_completed = _datetime.datetime.fromisoformat(seal["completed_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError("config/protocol/seal timestamps are malformed") from exc
    require(frozen_protocol >= frozen_cfg and seal_completed >= frozen_protocol,
            "protocol/seal chronology inconsistent with frozen config")
    source_map = {
        "driver": ("tools/run_agnews_anchored_comparison_v5.py", cfg["driver_source_sha256"]),
        "wake_helper": ("tools/agnews_anchored_wake_v5.py", cfg["wake_helper_sha256"]),
        "v4_boundary": ("tools/agnews_preupdate_sleep_v4.py", cfg["v4_boundary_sha256"]),
        "common": ("tools/agnews_v3_common.py", cfg["common_sha256"]),
        "v4_engine": ("tools/run_agnews_retention_v2.py", cfg["engine_sha256"]),
        "qualification_driver": ("tools/qualify_agnews_anchored_resume_v5.py", cfg["qualification_driver_sha256"]),
    }
    source_receipt = {}
    for key, (path, expected) in source_map.items():
        verify_index_stage_zero(root, path, index)
        actual = sha256(staged_blob(root, path))
        require(actual == expected, f"staged {key} source hash differs from frozen config")
        if key != "qualification_driver":
            require(protocol.get("source_sha256", {}).get(key) == actual,
                    f"protocol {key} source hash differs from staged source")
        source_receipt[key] = {"path": path, "sha256": actual}
    require(seal.get("source_sha256") == protocol.get("source_sha256"),
            "seal/protocol source hash maps differ")
    nested = cfg.get("nested_source_sha256", {})
    require(len(nested) == 12, "expected all 12 frozen nested_sleep module hashes")
    for name, expected in nested.items():
        module = name.split(".")[-1]
        path = "src/nested_sleep/__init__.py" if name == "nested_sleep" else f"src/nested_sleep/{module}.py"
        verify_index_stage_zero(root, path, index)
        actual = sha256(staged_blob(root, path))
        require(actual == expected, f"staged nested source hash mismatch: {path}")
        source_receipt[name] = {"path": path, "sha256": actual}
    analyzer_path = "tools/summarize_agnews_anchored_v5.py"
    verify_index_stage_zero(root, analyzer_path, index)
    analyzer_sha = sha256(staged_blob(root, analyzer_path))
    source_receipt["release_analyzer"] = {"path": analyzer_path, "sha256": analyzer_sha}
    require(protocol.get("record_pack_sha256") == cfg.get("record_pack_sha256"),
            "locked private record-pack hash differs (payload not read)")
    require(protocol.get("warm_result_sha256") == cfg.get("warm_result_sha256")
            and seal.get("warm_result_sha256") == cfg.get("warm_result_sha256"),
            "warm result provenance mismatch")
    require(protocol.get("v4_development_seal_sha256") == cfg.get("v4_development_seal_sha256")
            and seal.get("v4_development_seal_sha256") == cfg.get("v4_development_seal_sha256"),
            "v4 seal provenance mismatch")
    require(protocol.get("qualification_result_sha256") == cfg.get("qualification_result_sha256")
            and protocol.get("qualification_plan_sha256") == cfg.get("qualification_plan_sha256")
            and protocol.get("qualification_driver_sha256") == cfg.get("qualification_driver_sha256"),
            "qualification source/provenance mismatch")
    old_v4_protocol = "results/agnews-preupdate-v4/protocol.json"
    require(protocol.get("v4_protocol_sha256") == sha256(staged_blob(root, old_v4_protocol)),
            "v4 protocol staged hash mismatch")
    return {"config_sha256": cfg_hash, "protocol_sha256": protocol_hash,
            "source_graph": source_receipt}


def validate_old_public_evidence(root: Path, cfg: dict, index: dict) -> dict:
    evidence = {
        "components": "results/agnews-mechanism-v5/components/result.json",
        "interference": "results/agnews-mechanism-v5/interference/diagnostic-result.json",
        "qualification": "results/agnews-mechanism-v5/qualification/qualification.json",
        "qualification_plan": "results/agnews-mechanism-v5/qualification/plan.json",
        "warm": "results/agnews-preupdate-v4/transfer-result.json",
        "v4_seal": "results/agnews-preupdate-v4/development-sealed.json",
        "v4_protocol": "results/agnews-preupdate-v4/protocol.json",
        "mechanism_analysis": "results/agnews-mechanism-v5/analysis/analysis.json",
        "component_failure": "results/agnews-mechanism-v5/failures/components/failure.json",
        "interference_failure": "results/agnews-mechanism-v5/failures/interference/failure.json",
    }
    hashes = {
        "components": cfg["component_diagnostic_result_sha256"],
        "interference": cfg["interference_diagnostic_result_sha256"],
        "qualification": cfg["qualification_result_sha256"],
        "qualification_plan": cfg["qualification_plan_sha256"],
        "warm": cfg["warm_result_sha256"],
        "v4_seal": cfg["v4_development_seal_sha256"],
    }
    blobs = {}
    for name, path in evidence.items():
        verify_index_stage_zero(root, path, index)
        blobs[name] = staged_blob(root, path)
        if name in hashes:
            require(sha256(blobs[name]) == hashes[name], f"historical {name} evidence hash mismatch")
    components = json.loads(blobs["components"].decode("utf-8"))
    require(components.get("complete") is True and components.get("development_only") is True
            and components.get("confirmation_scored") is False
            and components.get("optimizer_updates") == 0,
            "component diagnostic status/provenance")
    require(len(components.get("results", [])) == 135 and len(components.get("warm_references", [])) == 3
            and components.get("total_scored_states") == 138,
            "component diagnostic state counts")
    require(all(x.get("old_receiver_tensors_unchanged") is True for x in components["results"]),
            "component diagnostic receiver-frozen guard failed")
    warm_by_seed = {int(x["seed"]): x for x in components["warm_references"]}
    require(set(warm_by_seed) == set(SEEDS), "component warm reference seed set")
    controls = [x for x in components["results"] if x.get("probe") == "warm_function_control"]
    require(len(controls) == 15, "warm-function control count")
    for row in controls:
        reference = warm_by_seed[int(row["seed"])]
        for group in GROUPS:
            close(metric(row["metrics"], group, EM), metric(reference["metrics"], group, EM),
                  path=f"component warm function {row['seed']}/{row['arm']}/{row['step']}/{group}")
            close(metric(row["metrics"], group, NLL), metric(reference["metrics"], group, NLL),
                  path=f"component warm NLL {row['seed']}/{row['arm']}/{row['step']}/{group}", tol=1e-5)
    interference = json.loads(blobs["interference"].decode("utf-8"))
    require(interference.get("complete") is True and interference.get("confirmation_scored") is False
            and interference.get("no_holdout_input") is True,
            "interference diagnostic status/provenance")
    require(interference.get("updates") == 48 and interference.get("gradient_probe_forward_calls") == 66
            and len(interference.get("results", [])) == 3,
            "interference diagnostic work/run totals")
    for run in interference["results"]:
        require(int(run["seed"]) in SEEDS, "unexpected interference seed")
        require(all(run.get("endpoint_match_v4_step16", {}).values()),
                f"interference v4 endpoint equivalence failed seed {run['seed']}")
        for snap in run.get("snapshots", []):
            met = snap["metrics"]["metrics"]
            validate_metric_counts(met, 128, 64, 64, 32,
                                   f"interference/{run['seed']}/{snap.get('step')}/{snap.get('tag')}")
    qualification = json.loads(blobs["qualification"].decode("utf-8"))
    plan = json.loads(blobs["qualification_plan"].decode("utf-8"))
    warm = json.loads(blobs["warm"].decode("utf-8"))
    diag_analysis = json.loads(blobs["mechanism_analysis"].decode("utf-8"))
    cfailure = json.loads(blobs["component_failure"].decode("utf-8"))
    ifailure = json.loads(blobs["interference_failure"].decode("utf-8"))
    inputs = diag_analysis.get("inputs", {})
    expected_diag_inputs = {
        "components_corrected": "components",
        "interference_corrected": "interference",
        "qualification": "qualification",
        "components_failed": "component_failure",
        "interference_failed": "interference_failure",
    }
    for summary_key, blob_key in expected_diag_inputs.items():
        require(inputs.get(summary_key, {}).get("sha256") == sha256(blobs[blob_key]),
                f"historical mechanism analysis input hash mismatch: {summary_key}")
    history = diag_analysis.get("attempt_history", {})
    require(history.get("components_corrected", {}).get("states_including_warm_refs") == 138
            and history.get("components_corrected", {}).get("optimizer_updates") == 0,
            "historical component diagnostic accounting")
    require(history.get("interference_corrected", {}).get("updates") == 48
            and history.get("interference_corrected", {}).get("gradient_probe_forward_calls") == 66,
            "historical corrected replay accounting")
    failed_i = history.get("interference_initial_failed", {})
    require(failed_i.get("attempted_wake_updates") == 0
            and failed_i.get("derived_gradient_probe_forward_calls_from_work_stages") == 4
            and failed_i.get("reported_gradient_probe_forward_calls") == 0
            and failed_i.get("counter_mismatch_documented") is True,
            "historical failed replay discrepancy/cost is not transparently retained")
    require(cfailure.get("optimizer_updates", 0) == 0 and ifailure.get("attempted_wake_updates") == 0,
            "historical failure ledger shows updates were attempted")
    require(all(int(v.get("gradient_forward_input_tokens", 0)) == 0
                for v in cfailure.get("work", {}).values()),
            "failed component attempt unexpectedly performed gradient work")
    require(qualification.get("schema") == "agnews-anchored-resume-qualified-v5"
            and qualification.get("passed") is True
            and qualification.get("scoring") is False
            and qualification.get("confirmation_scored") is False
            and qualification.get("exact_model_controller_policy_RNG") is True,
            "qualification pass/no-score flags")
    require(warm.get("all3_qualified") is True and warm.get("phaseB_permitted") is True
            and warm.get("confirmation_scored") is False,
            "warm teacher qualification/development flags")
    require(qualification.get("SGD_updates") == 18 and qualification.get("Sleep_Adam_updates") == 48
            and len(qualification.get("branches", [])) == 2,
            "qualification update/branch totals")
    require(qualification.get("plan_sha256") == cfg["qualification_plan_sha256"]
            and qualification.get("source_sha256") == cfg["qualification_driver_sha256"]
            and qualification.get("anchor_helper_sha256") == cfg["wake_helper_sha256"],
            "qualification result source hashes")
    require(sha256(blobs["qualification_plan"]) == qualification.get("plan_sha256"),
            "qualification plan hash")
    require(plan.get("schema") == "agnews-anchored-resume-plan-v5"
            and plan.get("prefix_steps") == 14 and plan.get("resume_steps") == [15, 16]
            and plan.get("expected_SGD_updates") == 18 and plan.get("expected_Sleep_Adam_updates") == 48
            and plan.get("scoring") is False and plan.get("confirmation_input") is False,
            "qualification plan scope")
    cp = qualification.get("private_checkpoint", {})
    require(cp.get("step") == 14 and cp.get("controller_counts") == [2, 14]
            and len(cp.get("sha256", "")) == 64,
            "qualification checkpoint metadata; payload intentionally not required in Git")
    require(plan.get("source_sha256") == cfg["qualification_driver_sha256"]
            and plan.get("anchor_helper_sha256") == cfg["wake_helper_sha256"]
            and plan.get("boundary_helper_sha256") == cfg["v4_boundary_sha256"],
            "qualification plan/source graph")
    for key in ("model", "controller", "execution_policy", "RNG"):
        require(qualification["branches"][0].get(key) == qualification["branches"][1].get(key),
                f"qualification restore branches differ: {key}")
    v4 = json.loads(blobs["v4_seal"].decode("utf-8"))
    v4_protocol = json.loads(blobs["v4_protocol"].decode("utf-8"))
    require(v4.get("complete") is True and v4.get("confirmation_scored") is False
            and v4.get("no_holdout_input") is True,
            "v4 public seal status/provenance")
    require(v4.get("protocol_sha256") == sha256(blobs["v4_protocol"]), "v4 protocol hash mismatch")
    require(v4_protocol.get("confirmation_input") is False and v4_protocol.get("confirmation_scored") is False,
            "v4 protocol confirmation flags")
    return {"component_hash": hashes["components"], "component_states": 135,
            "component_warm_references": 3, "component_receiver_guards": 135,
            "interference_hash": hashes["interference"], "interference_wake_updates": 48,
            "interference_gradient_passes": 66, "qualification_pass": True,
            "qualification_wake_updates": 18, "qualification_sleep_adam_updates": 48,
            "qualification_checkpoint_step14_counts": [2, 14],
            "warm_hash": hashes["warm"], "v4_seal_hash": hashes["v4_seal"],
            "v4_protocol_hash": sha256(blobs["v4_protocol"])}


def verify_analysis_and_ledger(root: Path, release_files: dict, seal: dict,
                               protocol: dict, bykey: dict, cfg: dict,
                               seal_path: str) -> tuple[dict, dict]:
    rel = f"{RELEASE_DIR}/analysis.json"
    analysis = json.loads(release_files[rel].decode("utf-8"))
    seal_hash = sha256(release_files[seal_path])
    require(analysis.get("schema") == "agnews-anchored-analysis-v5", "analysis schema")
    require(analysis.get("seal_sha256") == seal_hash, "analysis/raw-seal SHA mismatch")
    require(analysis.get("protocol", {}).get("protocol_sha256") == seal.get("protocol_sha256"),
            "analysis protocol provenance")
    require(analysis.get("protocol", {}).get("runner_sha256") == EXPECTED_DRIVER_SHA256,
            "analysis runner source hash")
    require(analysis.get("seeds") == list(SEEDS) and analysis.get("arms") == list(ARMS)
            and analysis.get("evaluation_steps") == list(STEPS), "analysis grid")
    verify_analysis_metrics(analysis, bykey)
    work_ledger = json.loads(release_files[f"{RELEASE_DIR}/work-ledger.json"].decode("utf-8"))
    cost_receipt = verify_work(seal, protocol, analysis, work_ledger, bykey)
    require(analysis.get("protocol", {}).get("config_sha256") == seal.get("config_sha256"),
            "analysis provenance config hash")
    return analysis, {"seal_sha256": seal_hash, **cost_receipt}


def verify_tables(release_files: dict, analysis: dict) -> list[str]:
    expected = expected_tables(analysis)
    for name, text in expected.items():
        rel = f"{RELEASE_DIR}/{name}"
        actual = release_files[rel].decode("utf-8")
        if "\r\n" in actual:
            actual = actual.replace("\r\n", "\n")
        require(actual == text, f"staged table does not reproduce from analysis/raw seal: {name}")
    return sorted(expected)


def verify_figures(release_files: dict) -> dict:
    import xml.etree.ElementTree as ET
    out = {}
    for name in FIGURES:
        rel = f"{RELEASE_DIR}/{name}"
        data = release_files[rel]
        if name.endswith(".png"):
            require(data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 1024,
                    f"invalid staged PNG figure: {name}")
        else:
            try:
                root = ET.fromstring(data.decode("utf-8"))
            except (UnicodeDecodeError, ET.ParseError) as exc:
                raise VerificationError(f"invalid staged SVG figure {name}: {exc}") from exc
            require(root.tag.endswith("svg"), f"staged SVG root tag invalid: {name}")
            for node in root.iter():
                tag = node.tag.rsplit("}", 1)[-1].lower()
                require(tag not in ("script", "image", "foreignobject"),
                        f"unsafe/embedded payload element in SVG {name}: {tag}")
                for key, value in node.attrib.items():
                    if key.rsplit("}", 1)[-1].lower() in ("href", "src"):
                        require(not str(value).lower().startswith(("http:", "https:", "data:")),
                                f"external/embedded SVG resource in {name}")
        out[name] = sha256(data)
    return out


def scan_raw_payloads(value, path="raw-seal") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lower = str(key).lower()
            require(lower not in FORBIDDEN_RAW_KEYS, f"raw token/prompt payload key found: {path}.{key}")
            scan_raw_payloads(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            scan_raw_payloads(child, f"{path}[{i}]")


def verify_release_paths(root: Path, entries: dict, index_changes: list[str], seal_path: str) -> dict:
    required = {
        f"{RELEASE_DIR}/protocol.json", seal_path,
        f"{RELEASE_DIR}/analysis.json", f"{RELEASE_DIR}/work-ledger.json",
        f"{RELEASE_DIR}/{CHECKPOINT_READBACK_NAME}",
        *(f"{RELEASE_DIR}/{name}" for name in TABLES),
        *(f"{RELEASE_DIR}/{name}" for name in FIGURES),
    }
    for rel in required:
        verify_index_stage_zero(root, rel, entries)
    for rel in entries:
        if rel.startswith(RELEASE_DIR + "/"):
            require(rel in required or rel == f"{RELEASE_DIR}/{VALIDATION_NAME}",
                    f"unexpected staged file under release results: {rel}")
    changed = [p for p in index_changes if PurePosixPath(p.lower()).suffix in FORBIDDEN_NEW_SUFFIXES]
    require(not changed, f"binary/data/checkpoint file staged: {changed[:8]}")
    for path in index_changes:
        if path.startswith(RELEASE_DIR + "/"):
            suffix = PurePosixPath(path).suffix.lower()
            require(suffix in (".json", ".md", ".png", ".svg"), f"unexpected release output type: {path}")
    return {"required_release_files": sorted(required), "staged_release_files": sorted(
        p for p in entries if p.startswith(RELEASE_DIR + "/"))}


def verify_preserved_history(root: Path, entries: dict) -> dict:
    preserved = preserve_index_paths(
        root, entries,
        prefixes=(
            "src/nested_sleep/",
            "results/agnews-preupdate-v4/",
            "results/agnews-boundary-transfer-v4/",
            "results/agnews-mechanism-v5/",
        ),
        paths=(
            "tools/agnews_v3_common.py", "tools/run_agnews_retention_v2.py",
            "tools/agnews_preupdate_sleep_v4.py", "tools/qualify_agnews_boundary_transfer_v4.py",
            "tools/qualify_agnews_preupdate_v4.py", "tools/summarize_agnews_preupdate_v4.py",
            "tools/summarize_agnews_mechanism_v5.py", "tools/summarize_agnews_anchored_v5.py",
            "reports/agnews-preupdate-v4.md",
        ),
    )
    hist_path = "results/agnews-mechanism-v5/publication-status.json"
    status = json.loads(staged_blob(root, hist_path).decode("utf-8"))
    require(status.get("main_comparison_status") == "RUNNING_FROZEN_PROTOCOL_NO_FINAL_RESULT_CLAIM",
            "historical mechanism publication status was rewritten")
    return preserved


def self_test() -> None:
    s = stats({17: 0.25, 29: 0.5, 43: 0.75})
    close(s["mean"], 0.5, path="selftest.mean")
    close(s["sample_sd"], 0.25, path="selftest.sd")
    require(s["n"] == 3 and s["per_seed"] == {"17": 0.25, "29": 0.5, "43": 0.75},
            "metric helper self-test")
    out = compute_pair_math_selftest()
    close(out["mean"], 0.125, path="selftest.paired.mean")
    close(out["sample_sd"], 0.125, path="selftest.paired.sd")


def compute_pair_math_selftest() -> dict:
    # Paired seed differences, not the difference of arm-level means/SDs.
    return stats({17: 0.0, 29: 0.125, 43: 0.25})


def run_verification(root: Path, seal_name: str) -> dict:
    root = repo_root(root)
    entries = index_entries(root)
    changes = staged_paths(root)
    config = json.loads(staged_blob(root, CONFIG_PATH).decode("utf-8"))
    require(config.get("base_public_commit"), "frozen config lacks base_public_commit")
    head = run_git(root, "rev-parse", "HEAD").decode().strip()
    # The public base may have later documentation-only commits, but it must
    # remain an ancestor of the staged release's current base.
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", config["base_public_commit"], "HEAD"],
                              cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(ancestor.returncode == 0, "configured base_public_commit is not an ancestor of HEAD")
    seal_path = f"{RELEASE_DIR}/{seal_name}"
    path_receipt = verify_release_paths(root, entries, changes, seal_path)
    release_paths = [
        f"{RELEASE_DIR}/protocol.json", seal_path, f"{RELEASE_DIR}/analysis.json",
        f"{RELEASE_DIR}/work-ledger.json", f"{RELEASE_DIR}/{CHECKPOINT_READBACK_NAME}",
        *(f"{RELEASE_DIR}/{x}" for x in TABLES),
        *(f"{RELEASE_DIR}/{x}" for x in FIGURES),
    ]
    release_files = {rel: staged_blob(root, rel) for rel in release_paths}
    seal = json.loads(release_files[seal_path].decode("utf-8"))
    protocol = json.loads(release_files[f"{RELEASE_DIR}/protocol.json"].decode("utf-8"))
    readback = json.loads(release_files[f"{RELEASE_DIR}/{CHECKPOINT_READBACK_NAME}"].decode("utf-8"))
    require(sha256(staged_blob(root, CONFIG_PATH)) == EXPECTED_CONFIG_SHA256,
            "config hash does not match frozen review")
    seal_sha = sha256(release_files[seal_path])
    require(seal_sha == EXPECTED_PRIVATE_SEAL_SHA256,
            "staged raw seal hash differs from checkpoint-readback review")
    require(readback.get("schema") == "agnews-v5-private-checkpoint-readback-v1"
            and readback.get("passed") is True and readback.get("checkpoints") == 126,
            "private checkpoint readback receipt schema/status/count")
    require(readback.get("seal_sha256") == seal_sha,
            "private checkpoint readback receipt does not bind the staged seal")
    require(isinstance(readback.get("bytes_read"), int) and readback["bytes_read"] > 0,
            "private checkpoint readback byte count")
    require(readback.get("deserialization") is False
            and readback.get("data_or_model_scoring") is False,
            "private checkpoint readback reports deserialization or scoring")
    protocol_receipt = validate_config_and_protocol(root, config, protocol, seal, entries, release_files)
    preserved = verify_preserved_history(root, entries)
    evidence_receipt = validate_old_public_evidence(root, config, entries)
    scan_raw_payloads(seal)
    for rel, data in release_files.items():
        if rel.endswith(".json"):
            scan_raw_payloads(json.loads(data.decode("utf-8")), rel)
    v4_blob = staged_blob(root, "results/agnews-preupdate-v4/development-sealed.json")
    v4 = json.loads(v4_blob.decode("utf-8"))
    warm = staged_json(root, "results/agnews-preupdate-v4/transfer-result.json")
    bykey, starts = validate_run_grid(seal, protocol, config, v4, warm)
    analysis, cost_receipt = verify_analysis_and_ledger(root, release_files, seal, protocol, bykey,
                                                        config, seal_path)
    verified_tables = verify_tables(release_files, analysis)
    figure_hashes = verify_figures(release_files)
    require(analysis.get("inference_limits", {}).get("claim_scope") ==
            "exploratory train/validation development comparison", "analysis claim scope")
    require(analysis.get("cost", {}).get("capacity") == config["capacity"]
            and analysis.get("cost", {}).get("sleep_final_active_slots_by_bank") == [0, 29],
            "analysis Sleep capacity metadata")
    return {
        "schema": "agnews-anchored-v5-release-verification-v1",
        "passed": True,
        "source": "staged Git index blobs only",
        "head_commit": head,
        "configured_base_public_commit": config["base_public_commit"],
        "config_path": CONFIG_PATH,
        "config_sha256": protocol_receipt["config_sha256"],
        "release_prefix": RELEASE_DIR,
        "raw_seal_path": seal_path,
        "raw_seal_sha256": cost_receipt["seal_sha256"],
        "private_checkpoint_readback": {
            "path": f"{RELEASE_DIR}/{CHECKPOINT_READBACK_NAME}",
            "schema": readback["schema"],
            "passed": True,
            "checkpoints": readback["checkpoints"],
            "bytes_read": readback["bytes_read"],
            "seal_sha256_matches": True,
            "deserialization": False,
            "data_or_model_scoring": False,
        },
        "protocol_sha256": protocol_receipt["protocol_sha256"],
        "source_graph": protocol_receipt["source_graph"],
        "historical_evidence": evidence_receipt,
        "preserved_history": preserved,
        "run_grid": {"seeds": list(SEEDS), "arms": list(ARMS), "runs": 21,
                     "validation_states": 126, "validation_examples_each": {"all": 512,
                         "old": 256, "new": 256, "per_class": 128},
                     "train_examples_each": 1024,
                     "same_start_state_per_seed": True,
                     "initial_state_hashes_by_seed": starts,
                     "old_receiver_provenance_all_true": True,
                     "v4_reference_checks": 6,
                     "v4_reference_metadata_limits": {
                         "validation_points_have_rng_sha256": False,
                         "validation_points_have_controller_counts": False,
                         "controller_counts_checked_against_checkpoint_metadata": True,
                         "rng_checked_from_recorded_exact_comparison_flag": True,
                     },
                     "sleep_events_have_expected_slots_and_route_policies": True},
        "recomputed_costs": cost_receipt,
        "verified_tables": verified_tables,
        "figure_sha256": figure_hashes,
        "no_raw_tokens_or_model_binary_paths": True,
        "confirmation_scored": False,
        "verified_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                        help="Git repository containing the staged final artifacts")
    parser.add_argument("--seal-name", default="development-sealed.json",
                        choices=("development-sealed.json", "raw-development-sealed.json",
                                 "rawdevelopment-sealed.json"))
    parser.add_argument("--receipt", type=Path,
                        help="optional path outside the staged evidence for the validation receipt")
    parser.add_argument("--self-test", action="store_true",
                        help="run pure metric-helper checks without requiring a Git repo")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            print(json.dumps({"self_test": "passed", "training_or_data_access": False}))
            return 0
        root = repo_root(args.repo_root)
        receipt = run_verification(root, args.seal_name)
        validation_rel = f"{RELEASE_DIR}/{VALIDATION_NAME}"
        entries = index_entries(root)
        if validation_rel in entries:
            saved = staged_json(root, validation_rel)
            # A receipt records the validation base, before the release commit.
            # Keep every scientific/hash field strict while allowing that base
            # to remain an ancestor of a later unchanged release checkout.
            saved_head = saved.get("head_commit")
            require(isinstance(saved_head, str) and len(saved_head) == 40,
                    "staged validation receipt lacks its validation-base commit")
            base_check = subprocess.run(["git", "merge-base", "--is-ancestor", saved_head, "HEAD"],
                                        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            require(base_check.returncode == 0, "validation-base commit is not an ancestor of HEAD")
            informational = {"verified_utc", "head_commit"}
            expected_stable = {k: v for k, v in receipt.items() if k not in informational}
            saved_stable = {k: v for k, v in saved.items() if k not in informational}
            assert_tree_close(saved_stable, expected_stable, "staged validation receipt")
            receipt["staged_validation_receipt_verified"] = True
        if args.receipt and validation_rel not in entries:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            encoded_to_file = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
            args.receipt.write_text(encoded_to_file, encoding="utf-8", newline="\n")
        encoded = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
        print(encoded, end="")
        return 0
    except (VerificationError, KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
