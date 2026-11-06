"""Small CPU regression checks for v5 terminal metadata and counters."""
from pathlib import Path
import sys
import unittest
import pytest
pytest.importorskip('transformers')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_agnews_anchored_comparison_v5 as runner


class AnchoredRunnerMetadataTests(unittest.TestCase):
    def test_final_slot_counts_are_attached_before_sleep_assertion(self):
        row = {"status": "running"}
        counts = runner.attach_final_active_parameters(
            row, {"active_slots_by_bank": [0, 29], "active_parameters": 123}, True)
        self.assertEqual(counts, [0, 29])
        self.assertEqual(row["final_active_parameters"]["active_parameters"], 123)

    def test_final_slot_count_rejects_incomplete_sleep_state(self):
        with self.assertRaises(AssertionError):
            runner.attach_final_active_parameters(
                {}, {"active_slots_by_bank": [0, 28]}, True)

    def test_training_loss_stats_keep_count_sum_min_max_and_skip_missing(self):
        stats = {}
        for value in (1.5, None, 0.5, 2.0):
            runner.update_loss_stats(stats, "new_ce", value)
        self.assertEqual(stats["new_ce"], {"count": 3, "sum": 4.0, "min": 0.5, "max": 2.0})

    def test_active_slot_metadata_must_match_two_bank_contract(self):
        self.assertEqual(runner.expected_active_slots({"active_slots_by_bank": [0, 1]}), [0, 1])
        with self.assertRaises(AssertionError):
            runner.expected_active_slots({"active_slots_by_bank": [0]})


if __name__ == "__main__":
    unittest.main()
