"""Synthetic aggregate-only tests for Banking v10 teacher signals."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from tools.analyze_banking_teacher_signals_v10 import (
    INPUT_SCHEMA,
    aggregate_sleep_signals,
    analyze_manifest,
    compare_request_cohorts,
)


H = "a" * 64
SOURCE_HASHES = {"tools/run_banking_cil_v10.py": "b" * 64}


def _step(kind, role, i):
    lti_index = i - 20 if kind == "lti" else -1
    strict_empty = (kind == "gkd" and i == 0) or (kind == "lti" and lti_index == 0)
    decoded_empty = (kind == "gkd" and i == 0) or (kind == "lti" and lti_index in (0, 2))
    return {"kind": kind, "role": role,
        "sample_owner": "teacher" if kind == "lti" or i % 2 == 0 else "student",
        "sampled_label_canonical": True,
        "sampled_label_matches_gold": i % 2 == 0,
        "decoded_sample_empty": decoded_empty,
        "sampled_first_token_eos": strict_empty,
        "lti_reward": (0.0 if lti_index == 2 else .5) if kind == "lti" else None,
        "zero_reward": kind == "lti" and lti_index == 2,
        "teacher_answer_empty": kind == "lti" and lti_index in (0, 2),
        "student_full_answer_empty": kind == "lti" and lti_index == 4,
        # Must never be included in aggregate output.
        "example_id_sha256": f"private-example-{i}"}


def _make_run(seed, arm, *, offset=0, physical=30):
    kinds = ["gkd"] * 20 + ["lti"] * 10 if arm == "SLEEP_ADDITIVE" else ["gold"] * 30
    per_step = [_step(kind, "old" if i % 2 == 0 else "current", i) for i, kind in enumerate(kinds)]
    event = {"event": "sleep_or_boundary", "seed": seed, "arm": arm,
        "task_id": 1, "global_wake_tick": 16, "level": 0, "event_index": 0,
        "global_selector_event_index": offset, "request_ids_sha256": H,
        "physical_adamw_steps": physical, "requests_by_role": {"old": 15, "current": 15},
        "objectives": {"steps": 30, "per_step": per_step}}
    work = {"sleep_events": 1, "physical_adamw_updates": physical,
        "full_backbone_forward_calls": 40, "full_backbone_forward_tokens": 1234,
        "tail_calls": 50, "tail_backward_calls": 30, "tail_backward_tokens": 300,
        "active_wall_seconds": 3.5}
    receipt = {"schema": "banking-cil-v10-run-receipt", "status": "complete",
        "seed": seed, "arm": arm, "tasks_completed": 7, "config_sha256": H,
        "source_lock_sha256": "c" * 64, "training_manifest_sha256": "d" * 64,
        "test_data_included": False, "model_immutable_versions_unchanged": True,
        "event_counts": {"0": 1, "1": 0, "2": 0, "3": 0}, "work": work}
    protocol = {"schema": "banking-cil-v10-protocol", "phase": "main", "seed": seed,
        "arm": arm, "config_sha256": H, "source_lock_sha256": "c" * 64,
        "training_manifest_sha256": "d" * 64, "source_hashes": SOURCE_HASHES,
        "test_data_included": False, "test_reader_available": False}
    return event, receipt, protocol


def test_sleep_signal_groups_distinguish_empty_definition_and_reward():
    event, receipt, protocol = _make_run(17, "SLEEP_ADDITIVE")
    run = {"seed": 17, "metrics": [event]}
    groups = aggregate_sleep_signals(run)
    lti_teacher = next(x for x in groups if x["method"] == "lti" and x["role"] == "old")
    # 5 old LTI steps. Two are decoded-empty by the synthetic fixture, but only
    # one strict first-token EOS; positive reward under that strict condition is counted.
    assert lti_teacher["count"] == 5
    assert lti_teacher["lti_teacher_decoded_empty_count"] == 2
    assert lti_teacher["lti_teacher_token_empty_count"] == 1
    assert lti_teacher["positive_reward_with_strict_teacher_empty_count"] == 1
    assert lti_teacher["positive_reward_count"] == 4
    assert lti_teacher["positive_reward_with_canonical_teacher_count"] == 4
    assert lti_teacher["positive_reward_with_noncanonical_teacher_count"] == 0
    assert lti_teacher["positive_reward_with_teacher_eos_count"] == 1
    assert lti_teacher["zero_reward_count"] == 1
    assert lti_teacher["mean_lti_reward"] is not None
    assert all("example_id_sha256" not in row for row in groups)


def test_lti_first_token_eos_must_be_included_in_decoded_empty_count():
    event, *_ = _make_run(17, "SLEEP_ADDITIVE")
    first_lti = next(x for x in event["objectives"]["per_step"] if x["kind"] == "lti")
    first_lti["sampled_first_token_eos"] = True
    first_lti["decoded_sample_empty"] = False
    with pytest.raises(ValueError, match="first-token EOS must imply sampled decoded-empty"):
        aggregate_sleep_signals({"seed": 17, "metrics": [event]})


def test_request_cohort_comparison_is_by_task_selector_level_without_exposing_hash():
    event, _receipt, _protocol = _make_run(17, "SLEEP_ADDITIVE", offset=0)
    gold_event, *_ = _make_run(17, "GOLD_ADDITIVE", offset=0)
    sleep = {"seed": 17, "metrics": [event]}
    gold = {"seed": 17, "metrics": [gold_event]}
    result = compare_request_cohorts(sleep, gold)
    assert result["matched_events"] == 1
    assert result["mismatched_events"] == 0
    gold_event["request_ids_sha256"] = "e" * 64
    result = compare_request_cohorts(sleep, gold)
    assert result["mismatched_events"] == 1
    assert "request_ids_sha256" not in json.dumps(result)


def test_full_manifest_aggregation_validates_seals_and_emits_only_grouped_output():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        root = Path(td)
        cfg = {"schema": "banking-cil-v10-driver-v1", "status": "frozen",
            "gkd_calls_per_event": 20, "lti_calls_per_event": 10,
            "source_hashes": SOURCE_HASHES}
        cfg_raw = json.dumps(cfg, sort_keys=True).encode()
        (root / "config.json").write_bytes(cfg_raw)
        config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        event, receipt, protocol = _make_run(17, "SLEEP_ADDITIVE")
        gold_event, gold_receipt, gold_protocol = _make_run(17, "GOLD_ADDITIVE")
        artifacts = []
        for prefix, rec, prot, row in (("sleep", receipt, protocol, event),
                                       ("gold", gold_receipt, gold_protocol, gold_event)):
            rec["config_sha256"] = config_sha
            prot["config_sha256"] = config_sha
            (root / f"{prefix}-receipt.json").write_text(json.dumps(rec), encoding="utf-8")
            (root / f"{prefix}-protocol.json").write_text(json.dumps(prot), encoding="utf-8")
            (root / f"{prefix}-metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            artifacts.append({"receipt_path": f"{prefix}-receipt.json",
                "receipt_sha256": hashlib.sha256((root / f"{prefix}-receipt.json").read_bytes()).hexdigest(),
                "protocol_path": f"{prefix}-protocol.json", "metrics_path": f"{prefix}-metrics.jsonl"})
        manifest = {"schema": INPUT_SCHEMA, "seed": 17, "config_path": "config.json",
            "config_sha256": config_sha,
            "config_file_sha256": hashlib.sha256(cfg_raw).hexdigest(),
            "source_lock_sha256": "c" * 64,
            "training_manifest_sha256": "d" * 64,
            "sleep": artifacts[0], "gold": artifacts[1]}
        (root / "input.json").write_text(json.dumps(manifest), encoding="utf-8")
        output = analyze_manifest(root / "input.json", root / "output.json")
        encoded = json.dumps(output)
        assert output["sleep"]["receipt_work"]["sleep_events"] == 1
        assert output["gold"]["request_cohort_comparison"]["matched_events"] == 1
        assert "private-example" not in encoded
        assert H not in encoded


def test_schedule_mismatch_rejected():
    event, _receipt, _protocol = _make_run(17, "SLEEP_ADDITIVE")
    event["objectives"]["per_step"] = event["objectives"]["per_step"][:-1]
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        root = Path(td)
        # Exercise the shape-level fail-closed guard by constructing a complete
        # synthetic run object; no filesystem/model loading is involved.
        from tools.analyze_banking_teacher_signals_v10 import _validate_run
        cfg = {"schema": "banking-cil-v10-driver-v1", "status": "frozen",
            "gkd_calls_per_event": 20, "lti_calls_per_event": 10,
            "source_hashes": SOURCE_HASHES}
        (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        _, receipt, protocol = _make_run(17, "SLEEP_ADDITIVE")
        (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        (root / "metrics.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
        spec = {"receipt_path": "receipt.json",
            "receipt_sha256": hashlib.sha256((root / "receipt.json").read_bytes()).hexdigest(),
            "protocol_path": "protocol.json", "metrics_path": "metrics.jsonl"}
        with pytest.raises(ValueError, match="exactly 30 objective rows"):
            _validate_run(spec, base=root, expected_arm="SLEEP_ADDITIVE", config_sha256=H,
                source_lock_sha256="c" * 64, training_manifest_sha256="d" * 64,
                expected_source_hashes=SOURCE_HASHES, expected_seed=17)

