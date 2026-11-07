#!/usr/bin/env python3
"""Recompute published CLINC V8 main-grid aggregates from public per-seed counts.

Uses only the curated aggregate summary and (when complete) the staged final-test
evaluation's six global count rows. No checkpoint, model, token, or dataset input
is accepted or read. Statistics are descriptive across three paired class-order
seeds, not independent random initializations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

SEEDS = (17, 29, 43)
ARMS = ("HOPE", "HOPE_SLEEP_FIXEDMIX")
SCOPES = ("old", "current", "seen")
TOL = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _close(a, b, label: str) -> None:
    if not _number(a) or not _number(b) or abs(float(a) - float(b)) > TOL:
        raise ValueError(f"stored aggregate mismatch at {label}: {a!r} vs recomputed {b!r}")


def _stat(values: list[float]) -> dict:
    if len(values) != 3:
        raise ValueError("all descriptive statistics require exactly three seeds")
    return {"n": 3, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values), "per_seed": list(values)}


def _check_stat(stored: dict, computed: dict, label: str) -> None:
    if not isinstance(stored, dict) or stored.get("n") != 3:
        raise ValueError(f"{label}: stored aggregate must report n=3")
    _close(stored.get("mean"), computed["mean"], label + ".mean")
    _close(stored.get("sample_sd"), computed["sample_sd"], label + ".sample_sd")
    per_seed = stored.get("per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != 3:
        raise ValueError(f"{label}: stored per_seed vector must have length three")
    for index, (actual, expected) in enumerate(zip(per_seed, computed["per_seed"])):
        _close(actual, expected, f"{label}.per_seed[{index}]")


def _dev_rows(summary: dict, arm: str, scope: str) -> tuple[list[dict], list[float]]:
    try:
        rows = summary["final_development_endpoints"][arm][scope]["per_seed"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing raw final-development count rows for {arm}/{scope}") from exc
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError(f"{arm}/{scope}: expected three final development count rows")
    values = []
    for expected_seed, row in zip(SEEDS, rows):
        if not isinstance(row, dict) or row.get("seed") != expected_seed:
            raise ValueError(f"{arm}/{scope}: seed grid/order must be exactly {SEEDS}")
        correct, count = row.get("correct"), row.get("count")
        if type(correct) is not int or type(count) is not int or count <= 0 or not 0 <= correct <= count:
            raise ValueError(f"{arm}/{scope}: invalid typed correct/count fields")
        value = correct / count
        _close(row.get("accuracy"), value, f"{arm}/{scope}/seed{expected_seed}.accuracy")
        values.append(value)
    return rows, values


def _validate_capture(capture: dict, *, label: str) -> None:
    if not isinstance(capture, dict):
        raise ValueError(f"{label}: missing public-capture metadata")
    if capture.get("schema") not in (
        "clinc-v8-public-development-capture-reference-v1",
        "clinc-v8-public-main-capture-v1",
        "clinc-v8-public-development-capture-v1",
    ):
        raise ValueError(f"{label}: unsupported _public_capture schema")
    raw_sha = capture.get("source_artifact_sha256_raw")
    if not isinstance(raw_sha, str) or len(raw_sha) != 64 or any(c not in "0123456789abcdef" for c in raw_sha):
        raise ValueError(f"{label}: malformed raw source artifact SHA")


def _extract_test_counts(test_result: dict, *, test_result_sha: str,
                         summary: dict) -> dict:
    if test_result.get("schema") != "clinc-v8-final-test-evaluation-v1" or \
       test_result.get("status") != "complete" or test_result.get("test_data_read") is not True:
        raise ValueError("test evaluation artifact is not a completed supported result")
    if test_result.get("primary_rows_used") != 4500 or test_result.get("development_only") is not False:
        raise ValueError("test result must be the full 4,500-row final test evaluation")
    if any(test_result.get(key) is not False for key in
           ("candidate_constrained", "task_mask_used", "checkpoint_selection_after_test")):
        raise ValueError("test result used candidate restriction, a task mask, or checkpoint selection")
    if test_result.get("optimizer_steps") != 0 or test_result.get("backward_calls") != 0:
        raise ValueError("test result must be read-only")
    capture = test_result.get("_public_capture")
    _validate_capture(capture, label="test evaluation")
    raw_sha = capture["source_artifact_sha256_raw"]
    raw_alias = capture.get("raw_input_sha256")
    if raw_alias is not None and raw_alias != raw_sha:
        raise ValueError("test evaluation _public_capture raw SHA aliases disagree")
    public_sha = capture.get("public_artifact_sha256")
    if public_sha is not None and (not isinstance(public_sha, str) or len(public_sha) != 64 or
        any(c not in "0123456789abcdef" for c in public_sha)):
        raise ValueError("test evaluation _public_capture public artifact SHA malformed")
    final_test = summary.get("final_test")
    if not isinstance(final_test, dict) or final_test.get("status") != "complete":
        raise ValueError("curated aggregate lacks a complete final_test summary")
    if final_test.get("input_artifact_sha256") != raw_sha:
        raise ValueError("aggregate final-test raw result SHA differs from _public_capture")

    rows = test_result.get("runs")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("test result must contain exactly six raw global count rows")
    by_pair = {}
    expected_pairs = {(seed, arm) for seed in SEEDS for arm in ARMS}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed test result run row")
        key = (row.get("seed"), row.get("arm"))
        if key not in expected_pairs or key in by_pair:
            raise ValueError("test result has a duplicate or unexpected seed/arm pair")
        primary = row.get("primary")
        if not isinstance(primary, dict):
            raise ValueError("test run lacks primary global counts")
        correct, total = primary.get("correct"), primary.get("total")
        if type(correct) is not int or type(total) is not int or total != 4500 or not 0 <= correct <= total:
            raise ValueError(f"test {key}: expected integer correct and total=4500")
        class_rows = primary.get("per_class")
        if not isinstance(class_rows, dict) or set(class_rows) != {str(i) for i in range(150)}:
            raise ValueError(f"test {key}: per-class aggregate must enumerate 150 classes")
        class_total = class_correct = 0
        for class_id in range(150):
            item = class_rows[str(class_id)]
            if not isinstance(item, dict) or type(item.get("total")) is not int or \
               type(item.get("correct")) is not int or item["total"] != 30 or \
               not 0 <= item["correct"] <= item["total"]:
                raise ValueError(f"test {key}: invalid per-class count at class {class_id}")
            class_total += item["total"]
            class_correct += item["correct"]
        if class_total != total or class_correct != correct:
            raise ValueError(f"test {key}: per-class counts do not sum to global counts")
        accuracy = correct / total
        by_pair[key] = {"correct": correct, "count": total, "accuracy": accuracy}
    if set(by_pair) != expected_pairs:
        raise ValueError("test result grid is incomplete")

    per_arm = {}
    for arm in ARMS:
        vals = [by_pair[(seed, arm)]["accuracy"] for seed in SEEDS]
        computed = _stat(vals)
        stored = final_test.get("by_arm", {}).get(arm, {}).get("accuracy")
        _check_stat(stored, computed, f"final_test.by_arm.{arm}.accuracy")
        per_arm[arm] = {"accuracy": computed,
            "per_seed_counts": [{"seed": seed, **by_pair[(seed, arm)]} for seed in SEEDS]}
    paired_values = [by_pair[(seed, ARMS[1])]["accuracy"] - by_pair[(seed, ARMS[0])]["accuracy"]
                     for seed in SEEDS]
    paired = _stat(paired_values)
    _check_stat(final_test.get("paired_fixedmix_minus_hope"), paired,
        "final_test.paired_fixedmix_minus_hope")
    stored_per_seed = final_test.get("per_seed")
    if not isinstance(stored_per_seed, list) or len(stored_per_seed) != 3:
        raise ValueError("final_test per-seed summary must have three rows")
    for index, seed in enumerate(SEEDS):
        row = stored_per_seed[index]
        if row.get("seed") != seed:
            raise ValueError("final_test per-seed grid/order mismatch")
        _close(row.get("hope_accuracy"), by_pair[(seed, "HOPE")]["accuracy"], f"final_test.seed{seed}.hope")
        _close(row.get("fixedmix_accuracy"), by_pair[(seed, "HOPE_SLEEP_FIXEDMIX")]["accuracy"], f"final_test.seed{seed}.fixedmix")
        _close(row.get("fixedmix_minus_hope"), paired_values[index], f"final_test.seed{seed}.paired")
    return {"by_arm": per_arm, "paired_fixedmix_minus_hope": paired,
        "input_artifact_sha256": raw_sha, "public_test_evaluation_sha256": test_result_sha,
        "rows_per_checkpoint": 4500, "n": 3}


def reanalyze(summary: dict, *, test_result: dict | None = None,
              test_result_sha: str | None = None,
              summary_sha: str | None = None) -> dict:
    if summary.get("schema") != "clinc-v8-main-six-aggregate-v1":
        raise ValueError("unsupported curated main-grid summary schema")
    _validate_capture(summary.get("_public_capture"), label="aggregate summary")
    if summary.get("status") not in ("complete_training_test_pending", "complete_training_and_test"):
        raise ValueError("aggregate summary is not a completed main-grid result")
    if summary.get("test_payload_read") is not False:
        raise ValueError("training aggregate must document that it did not read test data")
    scopes = {}
    paired_scopes = {}
    for scope in SCOPES:
        scopes[scope] = {}
        by_arm_values = {}
        for arm in ARMS:
            rows, values = _dev_rows(summary, arm, scope)
            computed = _stat(values)
            stored = summary["final_development_endpoints"][arm][scope]["accuracy"]
            _check_stat(stored, computed, f"final_development_endpoints.{arm}.{scope}.accuracy")
            by_arm_values[arm] = values
            scopes[scope][arm] = {"accuracy": computed,
                "per_seed_counts": [{"seed": row["seed"], "correct": row["correct"],
                    "count": row["count"], "accuracy": value}
                    for row, value in zip(rows, values)]}
            # Task-9 trajectory is a duplicate public aggregate; verify it too.
            trajectory = summary.get("development_trajectory_stats", [])
            last = next((item for item in trajectory if item.get("task_id") == 9), None)
            if not isinstance(last, dict):
                raise ValueError("development trajectory lacks task-9 aggregate")
            _check_stat(last["by_arm"][arm][scope]["accuracy"], computed,
                f"development_trajectory_stats.task9.{arm}.{scope}.accuracy")
        deltas = [b - a for a, b in zip(by_arm_values["HOPE"], by_arm_values["HOPE_SLEEP_FIXEDMIX"])]
        paired = _stat(deltas)
        _check_stat(summary["final_development_paired_differences_fixedmix_minus_hope"][scope],
            paired, f"final_development_paired_differences.{scope}")
        trajectory_pair = summary.get("paired_descriptive_differences_fixedmix_minus_hope", [])
        last_pair = next((item for item in trajectory_pair if item.get("task_id") == 9), None)
        if not isinstance(last_pair, dict):
            raise ValueError("paired development trajectory lacks task-9 differences")
        _check_stat(last_pair[scope], paired, f"paired_descriptive_differences.task9.{scope}")
        paired_scopes[scope] = paired

    if test_result is None:
        if summary.get("test_status") != "test_pending" or summary.get("final_test") is not None:
            raise ValueError("complete aggregate requires its original test-evaluation artifact")
        final_test = None
    else:
        if not isinstance(test_result_sha, str) or len(test_result_sha) != 64:
            raise ValueError("test evaluation file SHA is required")
        if summary.get("test_status") != "complete_all4500_global_unmasked":
            raise ValueError("summary/test artifact completeness markers disagree")
        final_test = _extract_test_counts(test_result, test_result_sha=test_result_sha, summary=summary)
    if summary_sha is None:
        summary_sha = ""
    return {"schema": "clinc-v8-main-public-aggregate-reanalysis-v1",
        "status": "complete" if final_test is not None else "development_only_test_pending",
        "claim_scope": summary["claim_scope"],
        "source_summary_sha256": summary_sha,
        "development_final_task9": {"by_scope": scopes,
            "paired_fixedmix_minus_hope": paired_scopes},
        "final_test_global": final_test,
        "test_rows_read_by_reanalysis": False,
        "data_or_checkpoint_access": False,
        "statistics_note": "means and sample SD across three paired class-order seeds; not independent random initializations"}


def render_markdown(result: dict) -> str:
    lines = ["# CLINC V8 public aggregate reanalysis", "",
        f"Status: **{result['status']}**. Scope: {result['claim_scope']}", "",
        "Means and sample SDs were recomputed from the six run-level count records for each endpoint. The three seeds are paired class-order seeds, not independent random initializations.", "",
        "## Final development endpoint (task 9)", "",
        "| Scope | HOPE mean ± sample SD | Fixed-mix mean ± sample SD | Paired difference mean ± sample SD |",
        "|---|---:|---:|---:|"]
    dev = result["development_final_task9"]
    for scope in SCOPES:
        a, b = dev["by_scope"][scope]["HOPE"]["accuracy"], dev["by_scope"][scope]["HOPE_SLEEP_FIXEDMIX"]["accuracy"]
        p = dev["paired_fixedmix_minus_hope"][scope]
        lines.append(f"| {scope} | {a['mean']:.6f} ± {a['sample_sd']:.6f} | {b['mean']:.6f} ± {b['sample_sd']:.6f} | {p['mean']:+.6f} ± {p['sample_sd']:.6f} |")
    if result["final_test_global"] is not None:
        lines += ["", "## Final test (global, unmasked)", "",
            "| Arm | Mean accuracy ± sample SD | Per-seed correct / 4,500 |",
            "|---|---:|---|"]
        t = result["final_test_global"]
        for arm in ARMS:
            s = t["by_arm"][arm]["accuracy"]
            counts = ", ".join(str(row["correct"]) for row in t["by_arm"][arm]["per_seed_counts"])
            lines.append(f"| {arm} | {s['mean']:.6f} ± {s['sample_sd']:.6f} | {counts} |")
        p = t["paired_fixedmix_minus_hope"]
        lines += ["", f"Paired fixed-mix minus HOPE: {p['mean']:+.6f} ± {p['sample_sd']:.6f} (n=3)."]
    lines += ["", "The tool checks stored published aggregates against these recomputed values. It does not open test rows, checkpoints, or models.", ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path,
        help="curated results/clinc-v8-main/summary.json")
    parser.add_argument("--test-evaluation", type=Path,
        help="curated original evaluation result JSON; required when summary includes final_test")
    parser.add_argument("--out-dir", required=True, type=Path,
        help="new empty output directory; writes reanalysis.json and reanalysis.md")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.out_dir}")
    summary_raw = args.summary.read_bytes()
    summary = json.loads(summary_raw)
    test_result = test_sha = None
    if args.test_evaluation:
        test_raw = args.test_evaluation.read_bytes()
        test_sha = hashlib.sha256(test_raw).hexdigest()
        test_result = json.loads(test_raw)
    result = reanalyze(summary, test_result=test_result, test_result_sha=test_sha,
                       summary_sha=hashlib.sha256(summary_raw).hexdigest())
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "reanalysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    (args.out_dir / "reanalysis.md").write_text(render_markdown(result),
        encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "output": str(args.out_dir.resolve()),
        "summary_sha256": result["source_summary_sha256"],
        "test_result_sha256": None if result["final_test_global"] is None else
            result["final_test_global"]["public_test_evaluation_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
