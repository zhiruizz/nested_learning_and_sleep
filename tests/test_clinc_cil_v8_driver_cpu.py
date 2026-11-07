"""Pure-Python guards for the bounded CLINC v8 pilot driver."""
from __future__ import annotations

import copy

import pytest

from tools.run_clinc_cil_v8 import (
    fixed_eval_rows,
    lti_reward,
    normalize_text,
    simulate_slot_peaks,
    task_groups,
    token_edit_similarity,
    validate_config,
)


def config():
    return {
        "schema": "clinc-cil-v8-pilot",
        "source_revision": "unit-test-lock",
        "model_revision": "local-model@rev",
        "model_assets_sha256": {"weights.safetensors": "a" * 64},
        "training_manifest_sha256": "b" * 64,
        "expected_normalized_train_dev_duplicates": 3,
        "seed": 17,
        "initialization_seed": 1907008,
        "tasks": [0, 1],
        "arms": ["HOPE", "HOPE_SLEEP"],
        "rank": 64,
        "capacity": 8,
        "levels": 5,
        "periods": [16, 64, 256, 1024, None],
        "epochs_per_task": 1,
        "examples_per_task": 1500,
        "batch_size": 8,
        "max_steps_per_task": 188,
        "base_lr": 0.001,
        "fast_lr": 0.1,
        "sleep_lr": 0.001,
        "sleep_weight_decay": 0.0,
        "sleep_clip_norm": 1.0,
        "sleep_prompt_policy": "current",
        "gkd_calls_per_event": 2,
        "lti_calls_per_event": 1,
        "max_new_tokens": 8,
        "lti_edit_distance_z0": 8,
        "lti_semantic_weight": 0.5,
        "lti_edit_weight": 0.5,
        "temperature": 0.8,
        "eval_examples_per_class": 2,
        "task0_acquisition_gate": 0.5,
        "endpoint_full_clean": True,
        "eval_full_development": True,
        "checkpoint_every_steps": 64,
        "max_wall_seconds": 3600,
        "max_forward_calls": 20000,
        "max_update_steps": 4000,
        "device": "cuda:0",
        "max_sequence_length": 128,
    }


def test_config_binds_fixed_pilot_and_computes_slot_cap():
    got = validate_config(config())
    assert got["derived_steps_per_epoch"] == 188
    assert len(got["derived_slot_peaks"]) == 5
    assert max(got["derived_slot_peaks"]) <= got["capacity"]


@pytest.mark.parametrize("field,value", [
    ("seed", 29), ("tasks", [0, 2]), ("periods", [16, 64, 128, 1024, None]),
    ("batch_size", 4), ("eval_full_development", "yes"),
    ("endpoint_full_clean", False), ("max_steps_per_task", 189),
])
def test_config_rejects_scope_or_budget_drift(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_fixed_eval_subset_is_id_sorted_and_gold_independent():
    rows = [
        {"id": "c" * 64, "intent_id": 0, "text": "third"},
        {"id": "a" * 64, "intent_id": 0, "text": "first"},
        {"id": "b" * 64, "intent_id": 0, "text": "second"},
        {"id": "d" * 64, "intent_id": 1, "text": "other"},
    ]
    chosen = fixed_eval_rows(rows, per_class=2)
    assert [x["id"] for x in chosen] == ["a" * 64, "b" * 64, "d" * 64]
    assert normalize_text("  BANK_transfer\t") == "bank transfer"


def test_task_grouping_uses_fixed_class_order_not_row_position():
    order = list(range(150))
    rows = [{"intent_id": 15}, {"intent_id": 0}, {"intent_id": 29}, {"intent_id": 50}]
    groups = task_groups(rows, order, 1)
    assert [r["intent_id"] for r in groups["old"]] == [0]
    assert [r["intent_id"] for r in groups["new"]] == [15, 29]


def test_token_edit_reward_requires_valid_canonical_agreement():
    labels = ["book_flight", "transfer_money"]
    assert token_edit_similarity([1, 2], [1, 2], z0=8) == 1.0
    assert lti_reward("book flight", "book_flight", teacher_tokens=[1, 2],
                      student_tokens=[1, 2], z0=8, canonical_labels=labels) == 1.0
    # Matching invalid empty strings do not receive semantic agreement.
    assert lti_reward("", "", teacher_tokens=[], student_tokens=[], z0=8,
                      canonical_labels=labels) == 0.5
    assert lti_reward("arbitrary", "arbitrary", teacher_tokens=[1], student_tokens=[1],
                      z0=8, canonical_labels=labels) == 0.5


def test_capacity_simulation_carries_clocks_across_task_boundary():
    periods = (16, 64, 256, 1024, None)
    one = simulate_slot_peaks(188, periods)
    two = simulate_slot_peaks(376, periods)
    assert two == [1, 4, 4, 1, 0]
    assert max(two) <= 8
    assert two[1] >= one[1]
    assert two[2] >= one[2]
    with pytest.raises(ValueError):
        simulate_slot_peaks(1, (16, 64))


def test_tail_cost_shapes_keep_gradient_and_no_gradient_work_separate():
    from types import SimpleNamespace
    from tools.run_clinc_cil_v8 import Budget
    budget = Budget(config())
    cache = SimpleNamespace(y=SimpleNamespace(shape=(2, 3, 7)))
    budget.tail(cache, True)
    budget.tail(cache, False)
    work = budget.as_dict()
    assert work["tail_calls"] == 2
    assert work["tail_shape_histogram"] == {"2x3": 2}
    assert work["tail_grad_shape_histogram"] == {"2x3": 1}
    assert work["tail_no_grad_shape_histogram"] == {"2x3": 1}
