"""CPU-only regression coverage for the X870E synthetic DEV runner contract."""
import json
import runpy
from pathlib import Path


PUBLIC = Path(__file__).resolve().parents[1]
RUNNER = PUBLIC / "tools" / "run_x870e_synthetic_dev_v2.py"
CONFIG = PUBLIC / "configs" / "x870e-synthetic-dev-v2.json"


def load_runner():
    return runpy.run_path(str(RUNNER))


def test_cpu_hash_and_qualification_schema():
    runner = load_runner()
    assert runner["cpu_contract_tests"]()["status"] == "passed"


def test_cpu_period_4_16_deferred_sleep_transaction():
    runner = load_runner()
    report = runner["cpu_state_fixture"]()
    assert report["status"] == "passed"
    assert report["restored_next_event_15_to_16"] is True
    assert report["deferred_live_counts_before_transaction"] == [4, 16]
    assert report["proposal_order"] == ["slow_level1_apply", "fast_level0_prospective_sleep"]
    assert report["candidate_counts_after_slow_then_fast_sleep"] == [0, 0]
    assert report["live_unchanged_before_replacement"] is True


def test_frozen_synthetic_config_has_disjoint_surface_roles_and_fixed_grid():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    roles = cfg["template_roles"]
    templates = roles["train"] + roles["calibration"] + roles["heldout_test_unused"]
    assert len(roles["train"]) == 4
    assert all(text == text.rstrip() for text in templates)
    assert len(set(templates)) == len(templates)
    assert cfg["candidate_lr_pairs"] == [[0.005, 0.01], [0.02, 0.04], [0.05, 0.1]]
    assert cfg["budget"]["hard_wall_seconds"] == 1200


def test_aggregate_summary_accounts_for_repeat_and_no_test_access():
    summary = json.loads((PUBLIC / "results" / "x870e-synthetic-dev-v2-summary.json").read_text(encoding="utf-8"))
    assert summary["accounting"]["total_foundation_updates_across_attempts"] == 3072
    assert summary["data"]["heldout_test_rows_generated_or_accessed"] == 0
    assert summary["protocol"]["sleep_dev_fixture"].endswith("none qualified")
    assert all(not row["qualified"] for row in summary["attempt_3_candidate_results"])
