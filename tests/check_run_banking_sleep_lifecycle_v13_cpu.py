import copy
import json
from pathlib import Path

import pytest

from tools.run_banking_sleep_lifecycle_v13 import (
    ARMS, EXECUTION_ORDER, predicted_budget_per_seed, validate_clock_horizon, validate_config,
)


@pytest.fixture
def draft():
    path = Path(__file__).resolve().parents[1] / "configs" / "banking_v13_batched_lifecycle_draft.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_draft_plan_and_request_update_accounting(draft):
    assert validate_config(draft, require_frozen=False) is draft
    totals = predicted_budget_per_seed(draft)
    assert totals["total_requests"] == 2 * (4 * 960 + 3200)
    assert totals["total_updates"] == 2 * 2 * 960 + 2 * 2 * 30 + 2 * 100
    assert totals["per_arm"]["B32_LOW_DOSE"] == {"requests": 6400, "committed_updates": 200}
    assert [a["name"] for a in ARMS] == [x["name"] for x in draft["arms"]]
    assert EXECUTION_ORDER == draft["execution_order_per_seed"]


@pytest.mark.parametrize("field,value", [
    ("wake_batch_size", 8),
    ("sleep_selector_event_stride", 960),
    ("sleep_events_per_run", 3),
    ("test_data_read", True),
    ("wake_ticks_after_last_sleep", 16),
])
def test_locked_design_drift_rejected(draft, field, value):
    changed = copy.deepcopy(draft)
    changed[field] = value
    with pytest.raises(ValueError):
        validate_config(changed, require_frozen=False)


def test_draft_cannot_be_used_as_gpu_config(draft):
    with pytest.raises(ValueError):
        validate_config(draft, require_frozen=True)


@pytest.mark.parametrize("seed,clock,prefix,expected", [
    (17, 12, 4, [4, 20, 35]),
    (29, 6, 10, [10, 26, 41]),
    (43, 2, 14, [14, 30, 45]),
])
def test_clock_horizon_derives_only_two_level_zero_events(seed, clock, prefix, expected):
    result = validate_clock_horizon({0: clock, 1: 0, 2: 0, 3: 0}, seed,
                                    [16, 64, 256, 1024, None], prefix, 16, 15)
    assert [result["first_due"], result["second_due"], result["endpoint_offset"]] == expected


def test_clock_horizon_rejects_other_memory_level_due():
    with pytest.raises(ValueError):
        validate_clock_horizon({0: 12, 1: 60, 2: 0, 3: 0}, 17,
                               [16, 64, 256, 1024, None], 4, 16, 15)
