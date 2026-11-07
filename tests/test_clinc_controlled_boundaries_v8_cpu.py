import copy
import json
import unittest

import torch

from tools.evaluate_clinc_controlled_boundaries_v8 import (
    ARMS, CONTEXTS_PER_EVENT, _apply_base_proposal, _proposal_from_wake, _make_context,
    checkpoint_pairs, summarize_pair, append_progress_record,
)


def _items(events=2):
    rows = []
    for arm in ARMS:
        for event in range(events):
            level = event % 2
            tick = (event + 1) * 16
            step = (event + 1) * 4
            parent = "a" * 64
            rows.append({"arm": arm, "side": "pre",
                "phase": f"pre-sleep-level{level}-event{event:03d}",
                "wake_step": step, "wake_tick": tick, "sha256": f"pre-{arm}-{event}",
                "path": f"pre-{arm}-{event}.pt", "parent_sha256": parent,
                "event": {"level": level, "global_tick": tick, "event_index": event}})
            pending = [] if event == 0 else [1]
            rows.append({"arm": arm, "side": "post",
                "phase": f"post-sleep-level{level}-event{event:03d}",
                "wake_step": step, "wake_tick": tick, "sha256": f"post-{arm}-{event}",
                "path": f"post-{arm}-{event}.pt", "parent_sha256": parent,
                "event": {"sender_level": level, "completed_event_index": event,
                          "other_due_levels_pending": pending,
                          "safe_to_resume_at_next_wake": len(pending) == 0}})
    return rows


class _Memory(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([a], dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.tensor([b], dtype=torch.float32))


class _Context(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = torch.nn.ModuleList([_Memory(2., 4.), _Memory(8., 16.)])


class BoundariesCpuTests(unittest.TestCase):
    def test_progress_persists_completed_pairs_and_refuses_duplicate_header(self):
        import tempfile
        from pathlib import Path
        output_root=Path(__file__).resolve().parents[1]/"outputs"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as tmp:
            path=Path(tmp)/"progress.jsonl"
            with self.assertRaises(FileNotFoundError):
                append_progress_record(path,{"kind":"pair"})
            append_progress_record(path,{"kind":"header","expected_pairs":32},first=True)
            append_progress_record(path,{"kind":"pair","event":0})
            self.assertEqual([json.loads(x)["kind"] for x in path.read_text().splitlines()],["header","pair"])
            with self.assertRaises(FileExistsError):
                append_progress_record(path,{"kind":"new-header"},first=True)

    def test_context_loading_uses_real_dataclass_api_without_mutating_source(self):
        from tools.clinc_tail_runtime_v8 import MutableContext
        from src.nested_sleep.experts import LowRankMemoryLevel
        original = MutableContext(torch.nn.ModuleList([LowRankMemoryLevel(2,1,2)]),
            torch.nn.Parameter(torch.ones(1,2)),torch.nn.Parameter(torch.ones(5,1)),True)
        state=original.state_dict()
        loaded=_make_context(None,state,lambda _:copy.deepcopy(original))
        self.assertFalse(hasattr(loaded,'eval'))
        self.assertTrue(all(not m.training for m in loaded.memory.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in loaded.parameters()))
        self.assertTrue(original.memory.training)
        self.assertTrue(all(p.requires_grad for p in original.parameters()))
        for a,b in zip(original.parameters(),loaded.parameters()):
            self.assertTrue(torch.equal(a,b));self.assertNotEqual(a.data_ptr(),b.data_ptr())

    def test_complete_event_pairs_and_pending_post_are_accepted(self):
        pairs = checkpoint_pairs(_items(), expected_arms=ARMS, events_per_arm=2)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs[0]["pre"]["wake_tick"], 16)
        self.assertEqual(pairs[1]["post"]["event"]["other_due_levels_pending"], [1])

    def test_simultaneous_due_events_may_share_a_wake_tick(self):
        rows = _items()
        for row in rows:
            if row["arm"] == ARMS[0] and row["phase"].endswith("event001"):
                row["wake_tick"] = 16
        pairs = checkpoint_pairs(rows, expected_arms=ARMS, events_per_arm=2)
        self.assertEqual(pairs[0]["wake_tick"], pairs[1]["wake_tick"])

    def test_missing_duplicate_and_extra_phases_are_rejected(self):
        rows = _items()
        with self.assertRaisesRegex(ValueError, "missing"):
            checkpoint_pairs(rows[:-1], expected_arms=ARMS, events_per_arm=2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checkpoint_pairs(rows + [copy.deepcopy(rows[0])], expected_arms=ARMS, events_per_arm=2)
        extra = copy.deepcopy(rows[0])
        extra["arm"] = "UNEXPECTED"
        with self.assertRaisesRegex(ValueError, "unexpected arm"):
            checkpoint_pairs(rows + [extra], expected_arms=ARMS, events_per_arm=2)

    def test_phase_sender_event_tick_and_pending_invariants_are_strict(self):
        mutations = [
            ("level", lambda x: x[1]["event"].update(sender_level=4)),
            ("index", lambda x: x[1]["event"].update(completed_event_index=7)),
            ("tick", lambda x: x[1].update(wake_tick=999)),
            ("pending", lambda x: x[1]["event"].update(safe_to_resume_at_next_wake=True)),
        ]
        for message, mutate in mutations:
            with self.subTest(message=message):
                rows = _items()
                post = next(row for row in rows if row["arm"] == ARMS[0] and row["side"] == "post" and row["phase"].endswith("event001"))
                mutate((None, post))
                with self.assertRaises(ValueError):
                    checkpoint_pairs(rows, expected_arms=ARMS, events_per_arm=2)

    def test_four_stage_summary_separates_base_reset_and_sleep_fit(self):
        metric = {}
        accuracies = {"teacher_pre": .60, "base_only_reconstruction": .70,
                      "initial_reset_expanded_reconstruction": .50, "optimized_post": .65}
        for name in CONTEXTS_PER_EVENT:
            metric[name] = {group: {"accuracy": accuracies[name], "count": 30,
                                    "invalid": 4, "predictions_sha256": name}
                            for group in ("old", "current", "seen")}
        result = summarize_pair(metric)
        self.assertAlmostEqual(result["old"]["base_only_delta_pp"], 10.)
        self.assertAlmostEqual(result["old"]["reset_expand_delta_pp"], -20.)
        self.assertAlmostEqual(result["old"]["sleep_fit_delta_pp"], 15.)
        self.assertAlmostEqual(result["old"]["total_delta_pp"], 5.)
        self.assertAlmostEqual(result["old"]["old_retention_ratio_post_over_pre"], 0.65 / 0.60)
        with self.assertRaisesRegex(ValueError, "four"):
            summarize_pair({name: metric[name] for name in CONTEXTS_PER_EVENT[:-1]})

    def test_base_only_proposal_changes_only_sender_base_in_isolated_context(self):
        original = _Context()
        clone = copy.deepcopy(original)
        wake = type("Wake", (), {"pending": {
            "memory.0.a": torch.tensor([.25]), "memory.0.b": torch.tensor([-.5])}})()
        proposal = _proposal_from_wake(original, wake, 0)
        self.assertTrue(torch.equal(proposal["memory.0.a"], torch.tensor([1.75])))
        _apply_base_proposal(clone, 0, proposal, torch)
        self.assertTrue(torch.equal(clone.memory[0].a, torch.tensor([1.75])))
        self.assertTrue(torch.equal(clone.memory[0].b, torch.tensor([4.5])))
        self.assertTrue(torch.equal(clone.memory[1].a, original.memory[1].a))
        self.assertTrue(torch.equal(original.memory[0].a, torch.tensor([2.])))


if __name__ == "__main__":
    unittest.main()
