#!/usr/bin/env python3
"""Recompute safe aggregate checks for the accepted CLINC V9 readout diagnostic.

This is a standard-library-only analyzer. It consumes the accepted aggregate JSON,
not model outputs, examples, tokens, checkpoints, or datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_CONTEXTS = ("FROZEN_BASE", "HOPE_TASK1", "SLEEP_TASK1", "TASK0_FOUNDATION")
READOUTS = ("v8_greedy_generation", "teacher_forced_sum", "teacher_forced_length_mean")
TASK_GROUPS = ("task0", "task1", "all60")


class ReadoutValidationError(ValueError):
    """Raised when the accepted result violates its declared contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadoutValidationError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _integer(value: Any, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} must be a nonnegative integer")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"{label} must be finite and nonnegative")
    return float(value)


def _accuracy(correct: int, count: int) -> float:
    _require(count > 0, "metric denominator must be positive")
    return correct / count


def analyze_result(payload: dict[str, Any], *, input_sha256: str) -> dict[str, Any]:
    """Validate and recompute all requested readout aggregates."""
    _require(_is_sha256(input_sha256), "input_sha256 must be a lowercase or uppercase SHA-256")
    _require(payload.get("schema") == "clinc-label-readout-diagnostic-v9", "unexpected input schema")
    _require(payload.get("mode") == "evaluate", "input must be an evaluate result")
    _require(payload.get("development_only") is True, "result must be marked development_only")

    cohort = payload.get("fixed_cohort")
    _require(isinstance(cohort, dict), "missing fixed_cohort")
    expected_cohort = {"classes": 30, "rows": 60, "rows_per_class": 2, "task0_rows": 30, "task1_rows": 30}
    for key, expected in expected_cohort.items():
        _require(cohort.get(key) == expected, f"fixed_cohort.{key} must equal {expected}")
    _require(_is_sha256(cohort.get("stable_id_set_sha256")), "fixed cohort ID-set hash missing or invalid")

    vocab = payload.get("label_vocabulary")
    _require(isinstance(vocab, dict), "missing label_vocabulary")
    _require(vocab.get("count") == 150 and vocab.get("all_labels_as_candidates") is True, "expected all 150 labels scored as candidates")
    _require(vocab.get("candidate_sequences") == 9000, "candidate sequence count must be 9000")
    _require(_is_sha256(vocab.get("canonical_order_sha256")), "canonical label order hash missing or invalid")

    immutable = payload.get("immutability")
    _require(isinstance(immutable, dict), "missing immutability guard")
    before = immutable.get("context_hashes_before")
    after = immutable.get("context_hashes_after")
    _require(isinstance(before, dict) and isinstance(after, dict), "missing context hashes")
    _require(set(before) == set(EXPECTED_CONTEXTS) and set(after) == set(EXPECTED_CONTEXTS), "immutable context set mismatch")
    _require(all(_is_sha256(before[k]) and _is_sha256(after[k]) for k in EXPECTED_CONTEXTS), "invalid immutable context hash")
    _require(before == after, "one or more context hashes changed")
    _require(immutable.get("all_adapters_frozen") is True, "adapters were not declared frozen")
    _require(immutable.get("backbone_versions_unchanged") is True, "backbone versions changed")

    for key in ("context_map_sha256", "master_config_sha256", "source_lock_sha256", "training_manifest_sha256"):
        _require(_is_sha256(payload.get(key)), f"{key} missing or invalid")
    capture = payload.get("_public_capture")
    source_provenance: dict[str, Any] = {}
    if capture is not None:
        _require(isinstance(capture, dict), "_public_capture must be an object when present")
        for key in ("source_artifact_sha256_raw", "raw_input_sha256", "public_artifact_sha256"):
            value = capture.get(key)
            if value is not None:
                _require(_is_sha256(value), f"_public_capture.{key} must be a SHA-256 when present")
                source_provenance[key] = value
        if capture.get("schema") is not None:
            _require(isinstance(capture["schema"], str), "_public_capture.schema must be a string")
            source_provenance["capture_schema"] = capture["schema"]

    methods = payload.get("methods")
    _require(isinstance(methods, dict), "missing methods")
    greedy = methods.get("v8_greedy_generation")
    _require(isinstance(greedy, dict) and greedy.get("primary_reference") is True and greedy.get("candidate_constrained") is False,
             "greedy reference contract mismatch")
    _require(methods.get("teacher_forced_sum", {}).get("candidate_constrained") is True, "sum candidate scoring must be constrained")
    _require(methods.get("teacher_forced_length_mean", {}).get("candidate_constrained") is True, "length-mean candidate scoring must be constrained")
    _require(methods.get("teacher_forced_sum", {}).get("includes_eos") is True, "sum readout must include EOS")
    _require(methods.get("teacher_forced_length_mean", {}).get("includes_eos") is True, "length-mean readout must include EOS")

    candidate_work = methods.get("candidate_scoring_work")
    greedy_work = greedy.get("work")
    work = payload.get("work")
    _require(isinstance(candidate_work, dict) and isinstance(greedy_work, dict) and isinstance(work, dict), "missing work counters")
    c_fw = _integer(candidate_work.get("backbone_forwards"), "candidate backbone_forwards")
    c_tail = _integer(candidate_work.get("tail_calls"), "candidate tail_calls")
    g_fw = _integer(greedy_work.get("backbone_forwards"), "greedy backbone_forwards")
    g_tail = _integer(greedy_work.get("tail_calls"), "greedy tail_calls")
    _require((c_fw, c_tail, g_fw, g_tail) == (1136, 4544, 303, 303), "readout work components do not match the frozen 4-context contract")
    _require(c_tail == 4 * c_fw, "candidate tail count must be four times candidate backbone forwards")
    _require(_integer(work.get("backbone_forwards"), "total backbone_forwards") == c_fw + g_fw == 1439, "total backbone-forward count mismatch")
    _require(_integer(work.get("tail_calls"), "total tail_calls") == c_tail + g_tail == 4847, "total tail-call count mismatch")
    _require(_integer(work.get("optimizer_updates"), "optimizer_updates") == 0, "diagnostic must have zero optimizer updates")
    elapsed = work.get("elapsed_seconds")
    _require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, "elapsed_seconds invalid")

    metrics = payload.get("metrics")
    _require(isinstance(metrics, dict) and set(metrics) == set(EXPECTED_CONTEXTS), "context metrics grid mismatch")
    result_contexts: dict[str, Any] = {}
    for context in EXPECTED_CONTEXTS:
        context_metrics = metrics[context]
        _require(isinstance(context_metrics, dict), f"metrics.{context} must be an object")
        rows: dict[str, Any] = {}
        for readout in READOUTS:
            _require(readout in context_metrics, f"metrics.{context}.{readout} missing")
            group_metrics = context_metrics[readout]
            _require(isinstance(group_metrics, dict) and set(group_metrics) == set(TASK_GROUPS), f"metrics.{context}.{readout} groups mismatch")
            validated: dict[str, Any] = {}
            for group in TASK_GROUPS:
                item = group_metrics[group]
                _require(isinstance(item, dict), f"{context}.{readout}.{group} must be an object")
                expected_count = 60 if group == "all60" else 30
                count = _integer(item.get("count"), f"{context}.{readout}.{group}.count")
                correct = _integer(item.get("correct"), f"{context}.{readout}.{group}.correct")
                invalid = _integer(item.get("invalid_predictions"), f"{context}.{readout}.{group}.invalid_predictions")
                _require(count == expected_count and correct <= count and invalid <= count, f"{context}.{readout}.{group} denominator/counters invalid")
                _require(math.isclose(float(item.get("accuracy")), _accuracy(correct, count), rel_tol=0, abs_tol=1e-12),
                         f"{context}.{readout}.{group} accuracy does not match correct/count")
                if readout != "v8_greedy_generation":
                    _require(invalid == 0, f"candidate readout {context}.{readout}.{group} has invalid predictions")
                if group == "all60":
                    for field in ("count", "correct", "invalid_predictions"):
                        _require(item[field] == group_metrics["task0"][field] + group_metrics["task1"][field],
                                 f"{context}.{readout}.{field} all60 does not equal task0+task1")
                validated[group] = {"count": count, "correct": correct, "accuracy": _accuracy(correct, count),
                                    "invalid_predictions": invalid}

            if readout != "v8_greedy_generation":
                for group in TASK_GROUPS:
                    source = group_metrics[group]
                    delta_value = source.get("delta_correct_vs_v8_generation")
                    _require(type(delta_value) is int, f"{context}.{readout}.{group}.delta_correct_vs_v8_generation must be an integer")
                    delta = delta_value
                    greedy_correct = rows["v8_greedy_generation"][group]["correct"]
                    _require(delta == source["correct"] - greedy_correct, f"{context}.{readout}.{group} delta vs greedy mismatch")
                    validated[group]["delta_correct_vs_v8_generation"] = delta
            rows[readout] = validated

        for group in TASK_GROUPS:
            rows["v8_greedy_generation"][group]["delta_correct_vs_v8_generation"] = 0

        nll_groups = context_metrics.get("gold_length_mean_nll")
        _require(isinstance(nll_groups, dict) and set(nll_groups) == set(TASK_GROUPS), f"{context} gold NLL groups mismatch")
        nll: dict[str, Any] = {}
        for group in TASK_GROUPS:
            item = nll_groups[group]
            _require(isinstance(item, dict), f"{context}.gold_length_mean_nll.{group} must be an object")
            count = _integer(item.get("count"), f"{context}.gold_length_mean_nll.{group}.count")
            mean = _finite_nonnegative(item.get("mean"), f"{context}.gold_length_mean_nll.{group}.mean")
            expected_count = 60 if group == "all60" else 30
            _require(count == expected_count, f"{context}.gold_length_mean_nll.{group} count mismatch")
            nll[group] = {"count": count, "mean": mean}
        weighted_mean = (nll["task0"]["mean"] * nll["task0"]["count"] +
                         nll["task1"]["mean"] * nll["task1"]["count"]) / nll["all60"]["count"]
        _require(math.isclose(nll["all60"]["mean"], weighted_mean, rel_tol=1e-12, abs_tol=1e-12),
                 f"{context} all60 gold NLL is not the weighted task mean")
        result_contexts[context] = {"readouts": rows, "gold_length_mean_nll": nll}

    return {
        "schema": "clinc-label-readout-reanalysis-v9",
        "input": {"schema": payload["schema"], "sha256": input_sha256},
        "source_provenance": source_provenance,
        "development_only": True,
        "not_primary_metric": True,
        "interpretation": "Candidate-constrained teacher-forced label scoring is a readout diagnostic, not free-generation accuracy and not a recomputation of model-level absolute NLL.",
        "cohort": {"rows": 60, "task0_rows": 30, "task1_rows": 30, "classes": 30, "rows_per_class": 2,
                   "stable_id_set_sha256": cohort["stable_id_set_sha256"]},
        "label_candidates": {"count": 150, "candidate_sequences": 9000, "canonical_order_sha256": vocab["canonical_order_sha256"]},
        "context_hashes_unchanged": True,
        "context_hashes": before,
        "work": {"optimizer_updates": 0, "backbone_forwards": 1439, "greedy_backbone_forwards": 303,
                 "candidate_backbone_forwards": 1136, "tail_calls": 4847, "greedy_tail_calls": 303,
                 "candidate_tail_calls": 4544, "candidate_tail_calls_per_forward": 4,
                 "input_tokens": _integer(work.get("input_tokens"), "input_tokens"),
                 "elapsed_seconds": float(elapsed)},
        "contexts": result_contexts,
        "source_hashes": {key: payload[key] for key in ("context_map_sha256", "master_config_sha256", "source_lock_sha256", "training_manifest_sha256")},
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# CLINC V9 label readout reanalysis",
        "",
        f"Accepted input SHA-256: `{analysis['input']['sha256']}`.",
        "",
        "This standard-library reanalysis recomputes denominators, accuracies, greedy-relative correct-count deltas, and weighted task NLL checks from the accepted aggregate. It does not read examples, predictions, checkpoints, or model outputs. Candidate-constrained teacher-forced scores are diagnostic readouts, not free-generation accuracy or a recomputation of absolute model NLL.",
        "",
        "All four contexts kept identical immutable hashes before and after evaluation. The diagnostic used 60 fixed development examples (30 from each task), scored 150 candidate labels (9,000 candidate sequences), and performed zero optimizer updates.",
        "",
        "## Accuracy and invalid predictions",
        "",
        "Correct counts are shown as task 0 / task 1 / all 60; invalid predictions are shown in parentheses. Candidate-constrained readouts have zero invalid predictions by contract.",
        "",
        "| Context | Readout | Correct (30 / 30 / 60) | Invalid (30 / 30 / 60) | Δ correct vs greedy (30 / 30 / 60) |",
        "|---|---|---:|---:|---:|",
    ]
    for context in EXPECTED_CONTEXTS:
        group = analysis["contexts"][context]["readouts"]
        greedy = group["v8_greedy_generation"]
        for readout in READOUTS:
            row = group[readout]
            correct = "/".join(str(row[g]["correct"]) for g in TASK_GROUPS)
            invalid = "/".join(str(row[g]["invalid_predictions"]) for g in TASK_GROUPS)
            delta = "/".join(f"{row[g]['delta_correct_vs_v8_generation']:+d}" for g in TASK_GROUPS)
            lines.append(f"| {context} | {readout} | {correct} | {invalid} | {delta} |")
    lines += ["", "## Gold-label length-mean NLL", "", "These values are aggregate checks from the accepted teacher-forced evaluation, not independently recomputed model-level absolute NLL.", "",
              "| Context | Task 0 (n=30) | Task 1 (n=30) | Weighted all 60 |", "|---|---:|---:|---:|"]
    for context in EXPECTED_CONTEXTS:
        nll = analysis["contexts"][context]["gold_length_mean_nll"]
        lines.append(f"| {context} | {nll['task0']['mean']:.6f} | {nll['task1']['mean']:.6f} | {nll['all60']['mean']:.6f} |")
    w = analysis["work"]
    lines += ["", "## Work counters", "", f"Backbone forwards: {w['backbone_forwards']} total = {w['greedy_backbone_forwards']} greedy + {w['candidate_backbone_forwards']} candidate. Tail calls: {w['tail_calls']} total = {w['greedy_tail_calls']} greedy + {w['candidate_tail_calls']} candidate; candidate tail calls are 4 × candidate forwards. Optimizer updates: {w['optimizer_updates']}. Elapsed seconds: {w['elapsed_seconds']:.3f}.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/clinc-v9-mechanism-preparation/accepted-readout-result.json"))
    parser.add_argument("--expected-source-sha", help="Optional SHA-256 to require for the exact input artifact bytes")
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh output directory for reanalysis.json and reanalysis.md")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if args.expected_source_sha is not None and digest.lower() != args.expected_source_sha.lower():
        raise SystemExit(f"input SHA-256 mismatch: expected {args.expected_source_sha}, got {digest}")
    payload = json.loads(raw.decode("utf-8"))
    analysis = analyze_result(payload, input_sha256=digest)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "reanalysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.out_dir / "reanalysis.md").write_text(render_markdown(analysis), encoding="utf-8", newline="\n")
    print(f"reanalysis complete: {args.out_dir}; input_sha256={digest}; contexts={len(analysis['contexts'])}; updates=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
