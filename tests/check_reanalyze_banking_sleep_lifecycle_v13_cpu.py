import hashlib
import json
from pathlib import Path

import pytest

from tools.reanalyze_banking_sleep_lifecycle_v13 import (
    ARMS, SEEDS, STAGES, AnalysisError, analyze,
)
from tools.plot_banking_sleep_lifecycle_v13 import create_plots


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _runtime_eval():
    populations = {"old": 110, "current": 110, "all77": 220}
    per_class = {str(i): {"count": (10 if i < 22 else 0), "correct": (5 if i < 22 else 0),
                          "accuracy": (0.5 if i < 22 else None)} for i in range(77)}
    groups = {name: {"count": n, "correct": n // 2, "accuracy": 0.5,
                     "invalid_predictions": 0} for name, n in populations.items()}
    groups["per_class"] = per_class
    nll = {name: {"count": n, "mean_token_nll": 1.25} for name, n in populations.items()}
    nll["per_class"] = {key: {"count": value["count"], "mean_token_nll": (1.25 if value["count"] else None)}
                         for key, value in per_class.items()}
    return {
        "schema": "banking-cil-v10-runtime-v1", "classes": 77, "candidate_sequences": 220 * 77,
        "candidate_metrics": {"task1": {"teacher_forced_sum": groups,
                                            "teacher_forced_length_mean": groups,
                                            "gold_label_token_mean_nll": nll}},
        "candidate_correct_counts": {"task1": {"candidate_count_per_example": 77,
                                                   "gold_correct_sum": 110,
                                                   "gold_correct_length_mean": 110}},
        "greedy_generation_metrics": {"task1": {"count": 220, "correct": 100,
                                                   "invalid_predictions": 2, "accuracy": 100 / 220,
                                                   "candidate_constrained": False}},
    }


def _arms():
    return [
        {"name": "CMS_ONLY", "requests_per_event": 0, "effective_batch_size": None, "learning_rate": None, "sleep": False},
        {"name": "B1_HIGH", "requests_per_event": 960, "effective_batch_size": 1, "learning_rate": 0.001, "sleep": True},
        {"name": "B1_LOW", "requests_per_event": 960, "effective_batch_size": 1, "learning_rate": 0.000005, "sleep": True},
        {"name": "B32_HIGH", "requests_per_event": 960, "effective_batch_size": 32, "learning_rate": 0.001, "sleep": True},
        {"name": "B32_LOW", "requests_per_event": 960, "effective_batch_size": 32, "learning_rate": 0.000005, "sleep": True},
        {"name": "B32_LOW_DOSE", "requests_per_event": 3200, "effective_batch_size": 32, "learning_rate": 0.000005, "sleep": True},
    ]


def _make_receipt(seed, arm, config_sha, source_hashes):
    prefix = {17: 4, 29: 10, 43: 14}[seed]
    initial_tick = {17: 348, 29: 326, 43: 322}[seed]
    tick_by_stage = {"event1_before": initial_tick + prefix, "event1_after": initial_tick + prefix,
                     "event2_before": initial_tick + prefix + 16, "event2_after": initial_tick + prefix + 16,
                     "postwake15": initial_tick + prefix + 31}
    stage_work = {"full_backbone_forward_calls": 4, "full_backbone_forward_tokens": 120,
                  "executed_input_tokens_including_padding": 120, "padded_input_tokens": 0,
                  "tail_backward_calls": 0, "tail_backward_tokens": 0, "tail_calls": 0,
                  "tail_grad_enabled_calls": 0, "tail_no_grad_calls": 0, "wake_updates": 0,
                  "physical_adamw_updates": 0, "update_attempts": 0, "sleep_events": 0,
                  "sampled_output_tokens": 0}
    stages = [{"name": stage, "global_wake_tick": tick_by_stage[stage],
                      "runtime": _runtime_eval(), "work_delta": stage_work.copy(),
                      "evaluation_after_training": True,
                      "context_state_sha256": _sha(f"{seed}/{stage}" if stage == "event1_before"
                                                    else f"{seed}/{arm}/{stage}")} for stage in STAGES]
    arm_cfg = next(a for a in _arms() if a["name"] == arm)
    events = []
    qualification_case = seed == 17 and arm == "B32_HIGH"
    additive_keys = ("full_backbone_forward_calls", "full_backbone_forward_tokens",
                     "executed_input_tokens_including_padding", "padded_input_tokens",
                     "tail_backward_calls", "tail_backward_tokens", "tail_calls",
                     "tail_grad_enabled_calls", "tail_no_grad_calls", "wake_updates",
                     "physical_adamw_updates", "update_attempts", "sleep_events",
                     "sampled_output_tokens")
    event_work_rows = []
    for i in (0, 1):
        req = arm_cfg["requests_per_event"]
        batch = arm_cfg["effective_batch_size"]
        updates = req // batch if batch else 0
        method_counts = {"gkd": req - req // 3, "lti": req // 3}
        q = req // 3
        owners = {"teacher": method_counts["gkd"] // 2 + q + method_counts["gkd"] % 2,
                  "student": method_counts["gkd"] // 2}
        objective = ({"request_count": req, "method_counts": method_counts,
                      "role_counts": {"old": req // 2, "current": req // 2},
                      "sampling_owner_counts": owners, "lti_reward_count": q,
                      "lti_reward_sum": float(q), "lti_zero_reward_count": 0,
                      "loss_mean": 0.5, "backward_calls": req,
                      "physical_adamw_steps": updates,
                      "skipped_exact_zero_batches": 0} if req else {})
        delta = {key: 0 for key in additive_keys}
        delta["physical_adamw_updates"] = updates
        delta["sleep_events"] = 1 if req else 0
        event_work_rows.append(delta)
        events.append({"level": 0, "event_index": i, "selected_requests": req,
                       "effective_batch_size": batch, "physical_adamw_steps": updates,
                       "objective_aggregates": objective, "work_delta": delta,
                       "qualification_extra_charged_separately": qualification_case and i == 0,
                       "selected_request_digest": _sha(f"{seed}/{i}/{req}"),
                       "pre_checkpoint_sha256": _sha(f"pre/{seed}/{arm}/{i}"),
                       "post_checkpoint_sha256": _sha(f"post/{seed}/{arm}/{i}")})
    wake_work = {key: 0 for key in additive_keys}
    wake_work["wake_updates"] = {17: 35, 29: 41, 43: 45}[seed]
    evaluation_work = {key: 0 for key in additive_keys}
    evaluation_work.update({key: value * len(STAGES) for key, value in stage_work.items()})
    qualification_work = {key: 0 for key in additive_keys}
    if qualification_case:
        qualification_work["physical_adamw_updates"] = 1
        qualification_work["update_attempts"] = 1
        qualification_work["tail_backward_calls"] = 1
        qualification_work["tail_backward_tokens"] = 30
        qualification_work["tail_calls"] = 1
        qualification_work["tail_grad_enabled_calls"] = 1
    budget = {}
    for key in additive_keys:
        budget[key] = (wake_work[key] + evaluation_work[key] + sum(x[key] for x in event_work_rows)
                       + qualification_work[key])
    return {"schema": "banking-v13-batched-lifecycle-result-v1", "status": "complete",
            "seed": seed, "arm": arm, "config_sha256": config_sha, "source_hashes": source_hashes,
            "initial_wake_tick": initial_tick, "final_wake_tick": initial_tick + {17: 35, 29: 41, 43: 45}[seed],
            "wake_request_digest": _sha(f"wake-schedule-{seed}"),
            "source_lock_sha256": _sha("v10-source"), "training_manifest_sha256": _sha("manifest"),
            "foundation_receipt_sha256": _sha(f"foundation-receipt-{seed}"),
            "foundation_checkpoint_sha256": _sha(f"foundation-checkpoint-{seed}"),
            "test_data_read": False, "official_test_read": False, "P2_accessed": False,
            "raw_rows_or_trajectories_written_to_json": False,
            "model_immutable_versions_unchanged": True, "stages": stages, "events": events,
            "work_phases": {"wake": wake_work, "evaluation": evaluation_work},
            "restore_qualification": ({"status": "pass", "extra_work_delta": qualification_work}
                                       if qualification_case else None),
            "budget": budget}


def _fixture(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True)
    source_file = source_root / "runner.py"
    source_file.write_text("# locked v13 source\n", encoding="utf-8")
    source_hashes = {"runner.py": hashlib.sha256(source_file.read_bytes()).hexdigest()}
    config = {"schema": "banking-v13-batched-lifecycle-config-v1", "status": "frozen",
              "parent_release": "d9c516cc16da3d8ad7d2cfb2f8173fdae90c3a87",
              "task_id": 1, "sender": 0, "sleep_events_per_run": 2,
              "wake_ticks_between_sleep_events": 16, "wake_ticks_after_last_sleep": 15,
              "periods": [16, 64, 256, 1024, None], "test_data_read": False,
              "no_test_based_selection": True,
              "execution_order_per_seed": ["B32_HIGH", "CMS_ONLY", "B1_HIGH", "B1_LOW", "B32_LOW", "B32_LOW_DOSE"],
              "sleep_method_cycle": ["gkd", "gkd", "lti"], "sleep_selector_event_stride": 3200,
              "gkd_sampling_owner_policy": "alternate_teacher_student_start_teacher_each_event",
              "evaluation": {"split": "development_only", "old_rows": 110, "current_rows": 110,
                             "candidate_labels": 77, "candidate_batch_size": 32,
                             "primary": "teacher_forced_sum", "include_greedy": True,
                             "stages": list(STAGES)},
              "seeds": list(SEEDS), "source_hashes": source_hashes,
              "v10_source_lock_sha256": _sha("v10-source"), "training_manifest_sha256": _sha("manifest"),
              "prefix_ticks_by_seed": {"17": 4, "29": 10, "43": 14},
              "expected_total_wake_ticks_by_seed": {"17": 35, "29": 41, "43": 45},
              "arms": _arms(),
              "restore_qualification": {"seed": 17, "arm": "B32_HIGH", "event_index": 0,
                                        "snapshot_after_requests": 32, "compare_after_requests": 64,
                                        "extra_request_evaluations": 32, "extra_adamw_updates": 1,
                                        "require_before_other_trajectories": True,
                                        "whole_job_resume_claim": False}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    run_root = tmp_path / "runs"
    for seed in SEEDS:
        for arm in ARMS:
            dest = run_root / f"seed-{seed}" / arm / "final-receipt.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(_make_receipt(seed, arm, config_sha, source_hashes)), encoding="utf-8")
    return run_root, config_path, source_root


def test_complete_grid_aggregates_every_seed_and_exposure(tmp_path):
    run_root, config, source_root = _fixture(tmp_path)
    result = analyze(run_root, config, source_root, tmp_path / "aggregate")
    assert result["summary"]["status"] == "complete_descriptive_development_analysis"
    assert len(result["summary"]["provenance"]["runs"]) == 18
    costs = result["summary"]["costs"]
    assert len(costs) == 18
    b32 = next(x for x in costs if x["seed"] == 17 and x["arm"] == "B32_HIGH")
    assert b32["requests_recorded"] == 1920
    assert b32["physical_adamw_steps_recorded"] == 60
    assert result["summary"]["test_data_read"] is False
    plotted = create_plots(result["aggregate_path"], result["metrics_csv_path"], tmp_path / "plots")
    assert len(plotted["plots"]) == 4
    assert (tmp_path / "plots" / "plot-manifest.json").is_file()


def test_rejects_missing_run_and_tampered_stage_contract(tmp_path):
    run_root, config, source_root = _fixture(tmp_path)
    (run_root / "seed-43" / "B32_LOW_DOSE" / "final-receipt.json").unlink()
    with pytest.raises(AnalysisError):
        analyze(run_root, config, source_root, tmp_path / "missing-out")

    run_root, config, source_root = _fixture(tmp_path / "second")
    receipt_path = run_root / "seed-29" / "B1_HIGH" / "final-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["stages"][0]["name"] = "event1_after"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AnalysisError):
        analyze(run_root, config, source_root, tmp_path / "tampered-out")


def test_rejects_wrong_foundation_or_request_budget(tmp_path):
    run_root, config, source_root = _fixture(tmp_path)
    receipt_path = run_root / "seed-17" / "B32_HIGH" / "final-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["events"][0]["selected_requests"] = 959
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AnalysisError):
        analyze(run_root, config, source_root, tmp_path / "budget-out")


def test_rejects_unmatched_2x2_request_digest_and_work_residue(tmp_path):
    run_root, config, source_root = _fixture(tmp_path)
    receipt_path = run_root / "seed-17" / "B32_LOW" / "final-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["events"][1]["selected_request_digest"] = _sha("different-cohort")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AnalysisError):
        analyze(run_root, config, source_root, tmp_path / "cohort-out")

    run_root, config, source_root = _fixture(tmp_path / "work")
    receipt_path = run_root / "seed-29" / "CMS_ONLY" / "final-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["budget"]["full_backbone_forward_calls"] += 1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AnalysisError):
        analyze(run_root, config, source_root, tmp_path / "work-out")
