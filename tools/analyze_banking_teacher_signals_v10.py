"""Aggregate-only analysis of sealed Banking v10 teacher/student rollouts.

Consumes completed receipt/protocol/closed-metrics JSONL files only. It never
reads source examples, checkpoints, prompts, decoded strings, token arrays, or
official-test data. The output contains grouped counts/rates and artifact hashes
only. This is a train/development mechanism diagnostic, not confirmatory evidence.

Input manifest (paths absolute or relative to this file)::

    {"schema":"banking-v10-teacher-signal-input-v1",
     "config_path":".../banking_v10_frozen_k10.json",
     "config_sha256":"<frozen config SHA>",
     "source_lock_sha256":"<frozen run source lock>",
     "training_manifest_sha256":"<frozen train/dev manifest SHA>",
     "sleep":{"receipt_path":".../run-receipt.json",
              "receipt_sha256":"<receipt SHA>",
              "protocol_path":".../protocol.json",
              "metrics_path":".../metrics.jsonl"},
     "gold":null}

Optional `gold` has the same shape as `sleep` and must be the matching same-seed
GOLD_ADDITIVE run. It is compared by task/global selector event/level request
hash, but hashes themselves are never emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


INPUT_SCHEMA = "banking-v10-teacher-signal-input-v1"
OUTPUT_SCHEMA = "banking-v10-teacher-signal-summary-v1"
EXPECTED_METHODS = {"gkd": 20, "lti": 10}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_json_stable(path: Path) -> tuple[dict, str]:
    before = sha256_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    after = sha256_file(path)
    if before != after:
        raise ValueError(f"artifact changed while being read: {path.name}")
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value, after


def _read_metrics_stable(path: Path) -> tuple[list[dict], str]:
    before = sha256_file(path)
    selected = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"metrics JSONL line {line_number} is incomplete or invalid") from exc
            if not isinstance(record, dict):
                raise ValueError(f"metrics JSONL line {line_number} is not an object")
            if record.get("event") == "sleep_or_boundary":
                selected.append(record)
    after = sha256_file(path)
    if before != after:
        raise ValueError(f"metrics JSONL changed while being read: {path.name}")
    return selected, after


def _number(value, label: str, *, integer=False):
    if integer:
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_run(spec: Mapping, *, base: Path, expected_arm: str,
                  config_sha256: str, source_lock_sha256: str,
                  training_manifest_sha256: str, expected_source_hashes: Mapping,
                  expected_seed: int | None = None) -> dict:
    if not isinstance(spec, Mapping):
        raise TypeError("run spec must be an object")
    receipt_path = _resolve(base, spec["receipt_path"])
    protocol_path = _resolve(base, spec["protocol_path"])
    metrics_path = _resolve(base, spec["metrics_path"])
    receipt, receipt_sha = _read_json_stable(receipt_path)
    protocol, protocol_sha = _read_json_stable(protocol_path)
    metrics, metrics_sha = _read_metrics_stable(metrics_path)
    if receipt_sha != spec.get("receipt_sha256"):
        raise ValueError("run receipt SHA differs from the manifest pin")
    if (receipt.get("schema"), receipt.get("status"), receipt.get("arm"), receipt.get("tasks_completed")) != (
            "banking-cil-v10-run-receipt", "complete", expected_arm, 7):
        raise ValueError(f"expected complete {expected_arm} run receipt")
    seed = receipt.get("seed")
    if type(seed) is not int or (expected_seed is not None and seed != expected_seed):
        raise ValueError("run seed differs from expected seed")
    for key, expected in (("config_sha256", config_sha256),
                          ("source_lock_sha256", source_lock_sha256),
                          ("training_manifest_sha256", training_manifest_sha256)):
        if receipt.get(key) != expected:
            raise ValueError(f"receipt {key} binding mismatch")
    if receipt.get("test_data_included") is not False or receipt.get("model_immutable_versions_unchanged") is not True:
        raise ValueError("run receipt does not affirm test exclusion and immutable backbone")
    if (protocol.get("schema"), protocol.get("phase"), protocol.get("seed"), protocol.get("arm")) != (
            "banking-cil-v10-protocol", "main", seed, expected_arm):
        raise ValueError("protocol identity mismatch")
    for key, expected in (("config_sha256", config_sha256),
                          ("source_lock_sha256", source_lock_sha256),
                          ("training_manifest_sha256", training_manifest_sha256)):
        if protocol.get(key) != expected:
            raise ValueError(f"protocol {key} binding mismatch")
    if protocol.get("source_hashes") != expected_source_hashes:
        # The exact helper graph is pinned by the byte-hashed frozen config.
        raise ValueError("protocol source hash graph differs from manifest binding")
    if protocol.get("test_data_included") is not False or protocol.get("test_reader_available") is not False:
        raise ValueError("protocol does not affirm test isolation")

    work = receipt.get("work")
    counts = receipt.get("event_counts")
    if not isinstance(work, Mapping) or not isinstance(counts, Mapping):
        raise ValueError("receipt lacks work/event counters")
    event_rows = [row for row in metrics if row.get("seed") == seed and row.get("arm") == expected_arm]
    if len(event_rows) != len(metrics):
        raise ValueError("closed metrics contain non-matching run identities")
    expected_event_count = _number(work.get("sleep_events"), "work.sleep_events", integer=True)
    expected_adamw = _number(work.get("physical_adamw_updates"), "work.physical_adamw_updates", integer=True)
    if len(event_rows) != expected_event_count:
        raise ValueError("sleep_or_boundary event count differs from receipt work")
    level_counts = Counter()
    physical_total = 0
    selector_ids = []
    seen_by_level = Counter()
    for row in event_rows:
        if row.get("mode") == "apply_without_sleep":
            raise ValueError("Sleep/Gold arm unexpectedly contains apply_without_sleep record")
        task = row.get("task_id")
        level = row.get("level")
        selector = row.get("global_selector_event_index")
        if type(task) is not int or task not in range(1, 7) or type(level) is not int or level not in range(4):
            raise ValueError("sleep event task/level is outside the frozen schedule")
        if type(selector) is not int or selector < 0:
            raise ValueError("global selector event index is missing")
        if type(row.get("event_index")) is not int or row["event_index"] != seen_by_level[level]:
            raise ValueError("per-level event index is not contiguous")
        seen_by_level[level] += 1
        selector_ids.append(selector)
        level_counts[str(level)] += 1
        adam = _number(row.get("physical_adamw_steps"), "event physical_adamw_steps", integer=True)
        physical_total += adam
        role_counts = row.get("requests_by_role")
        if not isinstance(role_counts, Mapping) or role_counts.get("old") != 15 or role_counts.get("current") != 15:
            raise ValueError("each event must preserve 15 old and 15 current prompt opportunities")
        objective = row.get("objectives")
        per_step = objective.get("per_step") if isinstance(objective, Mapping) else None
        if not isinstance(per_step, list) or len(per_step) != 30:
            raise ValueError("each event must record exactly 30 objective rows")
        observed_methods = Counter()
        observed_roles = Counter()
        for step in per_step:
            if not isinstance(step, Mapping):
                raise ValueError("malformed objective per-step record")
            kind = step.get("kind")
            if expected_arm == "SLEEP_ADDITIVE":
                if kind not in EXPECTED_METHODS:
                    raise ValueError("Sleep per-step kind is not GKD/LTI")
                observed_methods[kind] += 1
                role = step.get("role")
                owner = step.get("sample_owner")
                if role not in ("old", "current") or owner not in ("teacher", "student"):
                    raise ValueError("Sleep per-step role/owner is missing")
                observed_roles[role] += 1
            elif kind not in (None, "gold"):
                raise ValueError("Gold event contains a non-gold per-step objective")
        if expected_arm == "SLEEP_ADDITIVE" and dict(observed_methods) != EXPECTED_METHODS:
            raise ValueError("each Sleep event must contain exactly 20 GKD and 10 LTI steps")
        if expected_arm == "SLEEP_ADDITIVE" and observed_roles != Counter({"old": 15, "current": 15}):
            raise ValueError("per-step Sleep roles disagree with request-role counts")
        if expected_arm == "GOLD_ADDITIVE" and any("role" in step for step in per_step):
            if any("role" not in step for step in per_step) or Counter(step.get("role") for step in per_step) != Counter({"old": 15, "current": 15}):
                raise ValueError("Gold per-step roles do not match selected prompt opportunities")
    if sorted(selector_ids) != list(range(len(selector_ids))):
        raise ValueError("global selector event indices are not a unique contiguous sequence")
    normalized_expected_counts = {str(k): _number(v, f"event_counts.{k}", integer=True) for k, v in counts.items()}
    normalized_observed_counts = {str(level): level_counts.get(str(level), 0) for level in range(4)}
    if normalized_expected_counts != normalized_observed_counts:
        raise ValueError("receipt event_counts differ from observed metrics")
    if physical_total != expected_adamw:
        raise ValueError("event physical AdamW sum differs from receipt work")
    return {"seed": seed, "arm": expected_arm, "receipt": receipt, "metrics": event_rows,
            "artifacts": {"receipt_sha256": receipt_sha, "protocol_sha256": protocol_sha,
                          "metrics_sha256": metrics_sha,
                          "metrics_jsonl_bytes": metrics_path.stat().st_size}}


def _empty_aggregate():
    return {"count": 0, "canonical_valid_count": 0, "gold_agreement_count": 0,
            "decoded_empty_count": 0, "sampled_first_token_eos_count": 0,
            "lti_teacher_decoded_empty_count": 0, "lti_teacher_token_empty_count": 0,
            "lti_student_full_decoded_empty_count": 0,
            "positive_reward_with_strict_teacher_empty_count": 0,
            "positive_reward_count": 0, "positive_reward_with_canonical_teacher_count": 0,
            "positive_reward_with_noncanonical_teacher_count": 0,
            "positive_reward_with_teacher_eos_count": 0,
            "zero_reward_count": 0, "reward_sum": 0.0, "reward_count": 0}


def aggregate_sleep_signals(run: Mapping) -> dict:
    groups = defaultdict(_empty_aggregate)
    for event in run["metrics"]:
        task, level = int(event["task_id"]), int(event["level"])
        for step in event["objectives"]["per_step"]:
            kind = step["kind"]
            owner, role = step["sample_owner"], step["role"]
            group = groups[(int(run["seed"]), task, level, kind, owner, role)]
            group["count"] += 1
            group["canonical_valid_count"] += int(step.get("sampled_label_canonical") is True)
            group["gold_agreement_count"] += int(step.get("sampled_label_matches_gold") is True)
            group["decoded_empty_count"] += int(step.get("decoded_sample_empty") is True)
            strict_empty = step.get("sampled_first_token_eos") is True
            decoded_empty = step.get("decoded_sample_empty") is True
            group["sampled_first_token_eos_count"] += int(strict_empty)
            if kind == "lti":
                if strict_empty and not decoded_empty:
                    raise ValueError("LTI first-token EOS must imply sampled decoded-empty")
                group["lti_teacher_decoded_empty_count"] += int(step.get("teacher_answer_empty") is True)
                if owner != "teacher":
                    raise ValueError("frozen v10 LTI signal expected teacher-owned rollout")
                group["lti_teacher_token_empty_count"] += int(strict_empty)
                group["lti_student_full_decoded_empty_count"] += int(step.get("student_full_answer_empty") is True)
                reward = _number(step.get("lti_reward"), "LTI reward")
                group["reward_sum"] += reward
                group["reward_count"] += 1
                group["zero_reward_count"] += int(step.get("zero_reward") is True)
                group["positive_reward_with_strict_teacher_empty_count"] += int(strict_empty and reward > 0)
                positive = reward > 0
                canonical = step.get("sampled_label_canonical") is True
                group["positive_reward_count"] += int(positive)
                group["positive_reward_with_canonical_teacher_count"] += int(positive and canonical)
                group["positive_reward_with_noncanonical_teacher_count"] += int(positive and not canonical)
                group["positive_reward_with_teacher_eos_count"] += int(positive and strict_empty)
    output = []
    for key, value in sorted(groups.items()):
        seed, task, level, method, owner, role = key
        n, rewards = value["count"], value["reward_count"]
        output.append({"seed": seed, "task_id": task, "level": level,
            "method": method, "sample_owner": owner, "role": role,
            **{k: v for k, v in value.items() if k not in {"reward_sum", "reward_count"}},
            "mean_lti_reward": value["reward_sum"] / rewards if rewards else None,
            "reward_observations": rewards,
            "canonical_valid_rate": value["canonical_valid_count"] / n if n else None,
            "gold_agreement_rate": value["gold_agreement_count"] / n if n else None,
            "decoded_empty_rate": value["decoded_empty_count"] / n if n else None,
            "strict_first_token_eos_rate": value["sampled_first_token_eos_count"] / n if n else None})
    return output


def compare_request_cohorts(sleep: Mapping, gold: Mapping | None) -> dict:
    if gold is None:
        return {"status": "gold_reference_not_provided", "sleep_events": len(sleep["metrics"]),
                "gold_events": None, "matched_events": None, "mismatched_events": None,
                "missing_gold_events": None}
    if sleep["seed"] != gold["seed"]:
        raise ValueError("Sleep and Gold runs must share the same seed")
    def index(run):
        rows = {}
        for row in run["metrics"]:
            key = (row.get("task_id"), row.get("global_selector_event_index"), row.get("level"))
            digest = row.get("request_ids_sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("event request_ids_sha256 is malformed")
            if key in rows:
                raise ValueError("duplicate Sleep/Gold event identity")
            rows[key] = digest
        return rows
    left, right = index(sleep), index(gold)
    common = set(left) & set(right)
    return {"status": "compared", "sleep_events": len(left), "gold_events": len(right),
            "matched_events": sum(left[key] == right[key] for key in common),
            "mismatched_events": sum(left[key] != right[key] for key in common),
            "missing_gold_events": len(set(left) - set(right)),
            "sleep_only_events": len(set(left) - set(right)),
            "gold_only_events": len(set(right) - set(left))}


def analyze_manifest(manifest_path: str | Path, output_path: str | Path) -> dict:
    manifest_path, output_path = Path(manifest_path).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    base = manifest_path.parent
    manifest, manifest_sha = _read_json_stable(manifest_path)
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("teacher-signal manifest schema mismatch")
    config_path = _resolve(base, manifest["config_path"])
    config_file_sha = sha256_file(config_path)
    if config_file_sha != manifest.get("config_file_sha256"):
        raise ValueError("frozen config file byte SHA mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    canonical_cfg = {key: value for key, value in config.items() if key != "phase"}
    config_sha = hashlib.sha256(json.dumps(canonical_cfg, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    if config_sha != manifest.get("config_sha256"):
        raise ValueError("frozen logical config SHA mismatch")
    if (config.get("schema"), config.get("status"), config.get("gkd_calls_per_event"),
            config.get("lti_calls_per_event")) != ("banking-cil-v10-driver-v1", "frozen", 20, 10):
        raise ValueError("config does not pin the frozen 20 GKD/10 LTI schedule")
    source_lock = manifest.get("source_lock_sha256")
    data_sha = manifest.get("training_manifest_sha256")
    if not isinstance(source_lock, str) or len(source_lock) != 64 or not isinstance(data_sha, str) or len(data_sha) != 64:
        raise ValueError("manifest must bind source lock and train/development manifest SHA")
    sleep = _validate_run(manifest["sleep"], base=base, expected_arm="SLEEP_ADDITIVE",
        config_sha256=config_sha, source_lock_sha256=source_lock, training_manifest_sha256=data_sha,
        expected_source_hashes=config.get("source_hashes"),
        expected_seed=manifest.get("seed"))
    gold_spec = manifest.get("gold")
    gold = _validate_run(gold_spec, base=base, expected_arm="GOLD_ADDITIVE",
        config_sha256=config_sha, source_lock_sha256=source_lock, training_manifest_sha256=data_sha,
        expected_source_hashes=config.get("source_hashes"),
        expected_seed=sleep["seed"]) if gold_spec is not None else None
    # Recheck input files after all processing, not just at individual read.
    run_specs = [(sleep, manifest["sleep"])]
    if gold is not None:
        run_specs.append((gold, gold_spec))
    for run, spec in run_specs:
        for key, path_key in (("receipt_sha256", "receipt_path"), ("protocol_sha256", "protocol_path"),
                              ("metrics_sha256", "metrics_path")):
            path = _resolve(base, spec[path_key])
            if sha256_file(path) != run["artifacts"][key]:
                raise ValueError(f"{path_key} changed after aggregation")
    output = {"schema": OUTPUT_SCHEMA, "claim_scope": "exploratory train/development teacher-signal diagnostic; not confirmatory",
        "config_sha256": config_sha, "source_lock_sha256": source_lock,
        "config_file_sha256": config_file_sha,
        "training_manifest_sha256": data_sha, "input_manifest_sha256": manifest_sha,
        "sleep": {"seed": sleep["seed"], "arm": sleep["arm"], "artifacts": sleep["artifacts"],
                  "receipt_work": {k: sleep["receipt"]["work"].get(k) for k in
                     ("full_backbone_forward_calls", "full_backbone_forward_tokens", "tail_calls",
                      "tail_backward_calls", "tail_backward_tokens", "physical_adamw_updates",
                      "sleep_events", "active_wall_seconds")},
                  "event_counts_by_level": sleep["receipt"].get("event_counts"),
                  "signals_by_task_level_method_owner_role": aggregate_sleep_signals(sleep)},
        "gold": None if gold is None else {"seed": gold["seed"], "arm": gold["arm"],
            "artifacts": gold["artifacts"],
            "receipt_work": {k: gold["receipt"]["work"].get(k) for k in
                ("full_backbone_forward_calls", "tail_calls", "physical_adamw_updates", "sleep_events", "active_wall_seconds")},
            "request_cohort_comparison": compare_request_cohorts(sleep, gold)},
        "lti_signal_interpretation": {
            "sampled_first_token_eos_is_strict_teacher_token_empty": True,
            "teacher_answer_empty_is_decoded_whitespace_empty_and_not_equivalent_to_token_empty": True,
            "gold_agreement_is_logged_against_training_row_label_and_is_not_the_gkd_or_lti_loss_target": True,
            "raw_examples_prompts_decoded_text_token_ids_and_request_hashes_emitted": False}}
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = analyze_manifest(args.manifest, args.output)
    print(json.dumps({"schema": output["schema"], "sleep_seed": output["sleep"]["seed"],
                      "sleep_events": output["sleep"]["receipt_work"]["sleep_events"],
                      "gold_present": output["gold"] is not None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
