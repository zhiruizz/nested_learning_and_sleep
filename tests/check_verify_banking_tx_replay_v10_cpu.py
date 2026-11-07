from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_banking_tx_replay_v10 as replay


class OriginalTransactionReplayTests(unittest.TestCase):
    def test_continuation_uses_original_object_then_independent_disk_restore(self):
        original = {"cursor": 3, "state": [1, 2, 3]}
        disk_snapshot = {"cursor": 3, "state": [1, 2, 3]}
        calls = []

        def continue_original(tx):
            calls.append(("original", id(tx)))
            tx["cursor"] = 6
            tx["state"].extend([4, 5, 6])
            return {"cursor": 6, "updates": 3}, {"cursor": tx["cursor"], "state": list(tx["state"])}

        def restore_from_disk(snapshot):
            calls.append(("restore", id(original)))
            return {"cursor": snapshot["cursor"], "state": list(snapshot["state"])}

        def continue_restored(tx):
            calls.append(("restored", id(tx)))
            tx["cursor"] = 6
            tx["state"].extend([4, 5, 6])
            return {"cursor": 6, "updates": 3}, {"cursor": tx["cursor"], "state": list(tx["state"])}

        def exact_equal(left, right, label):
            if left != right:
                raise AssertionError(label)

        reports = replay.original_then_restore(original, disk_snapshot,
            continue_original=continue_original, restore_from_disk=restore_from_disk,
            continue_restored=continue_restored, exact_equal=exact_equal)
        self.assertEqual(reports[0], reports[2])
        self.assertEqual(reports[1], reports[3])
        self.assertEqual(calls[0], ("original", id(original)))
        self.assertEqual(calls[1][0], "restore")
        self.assertEqual(calls[2][0], "restored")
        self.assertNotEqual(calls[2][1], id(original))
        self.assertEqual(original["cursor"], 6)

    def test_branch_divergence_fails_closed(self):
        class Mismatch(Exception):
            pass
        def assert_equal(left, right, label):
            if left != right:
                raise Mismatch(label)
        with self.assertRaises(Mismatch):
            replay.original_then_restore(object(), {},
                continue_original=lambda tx: ("A", "state-A"),
                restore_from_disk=lambda snap: object(),
                continue_restored=lambda tx: ("B", "state-B"),
                exact_equal=assert_equal)

    def test_caps_and_schema_make_scope_explicit(self):
        self.assertEqual(replay.SCHEMA, "banking-cil-v10-original-transaction-replay-proof-v1")
        self.assertEqual(replay.CAPS, {"max_wall_seconds": 1800,
            "max_forward_calls": 20000, "max_update_steps": 1000})


if __name__ == "__main__":
    unittest.main()
