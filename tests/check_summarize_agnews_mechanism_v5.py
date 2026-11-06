"""Pure-stdlib tests for the offline v5 evidence summarizer."""
import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "summarize_agnews_mechanism_v5.py"
spec = importlib.util.spec_from_file_location("mechanism_summary", MODULE_PATH)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def metrics(old, new, n=512):
    return {
        "all": {"examples": n, "full_vocab_two_token_greedy_exact_match": (old + new) / 2},
        "old": {"examples": n // 2, "full_vocab_two_token_greedy_exact_match": old, "class_nll": 1.0 + old},
        "new": {"examples": n // 2, "full_vocab_two_token_greedy_exact_match": new, "class_nll": 2.0 + new},
    }


class SummaryTests(unittest.TestCase):
    def test_sample_sd_uses_sample_definition(self):
        s = summary.sample_summary([0.0, 0.5, 1.0])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean"], 0.5)
        self.assertAlmostEqual(s["sample_sd"], 0.5)

    def test_component_summary_has_45_case_probe_cells_and_separate_warm_reference(self):
        rows = []
        case_pairs = [("a", 16), ("a", 112), ("b", 16), ("b", 112), ("c", 112)]
        for arm, step in case_pairs:
            for probe in summary.PROBES:
                for seed in summary.SEEDS:
                    rows.append({"arm": arm, "step": step, "probe": probe, "seed": seed,
                                 "metrics": metrics(seed / 100., step / 200.),
                                 "old_receiver_tensors_unchanged": True})
        warm = [{"seed": seed, "metrics": metrics(.9, .1)} for seed in summary.SEEDS]
        cases, refs = summary.build_component_summary({"results": rows, "warm_references": warm})
        self.assertEqual(len(cases), 5)
        self.assertEqual(sum(len(c["probes"]) for c in cases), 45)
        self.assertAlmostEqual(refs["old_em"]["mean"], .9)
        self.assertEqual(refs["old_em"]["n"], 3)

    def test_interference_keeps_128_denominator_and_12_gradient_states(self):
        runs = []
        for seed in summary.SEEDS:
            groups = {g: {"cosine": .1, "norm_a": 1., "norm_b": 2., "dot": 3., "parameter_count": 4} for g in summary.GROUPS}
            probes = [{"step": step, "tag": "post", "groups": groups} for step in (0, 1, 4, 16)]
            snapshots = []
            for step, tag in ((0, "post"), (1, "post"), (4, "prebase"), (4, "post"), (8, "post"), (16, "prebase"), (16, "post")):
                snapshots.append({"step": step, "tag": tag, "metrics": {"metrics": metrics(.5, .25, 128)}})
            runs.append({"seed": seed, "gradient_probes": probes, "snapshots": snapshots,
                         "endpoint_match_v4_step16": {k: True for k in ("model", "controller", "execution_state", "rng")}})
        grads, trajectory = summary.build_interference({"results": runs})
        self.assertEqual(len(grads), 36)
        self.assertEqual(len(trajectory), 21)
        self.assertTrue(all(x["n_all"] == 128 and x["n_old"] == 64 and x["n_new"] == 64 for x in trajectory))

    def test_duplicate_work_stage_names_are_summed_and_failed_gradient_count_rederived(self):
        work = summary._sum_work({
            "failed": {"same": {"forward_calls": 2, "forward_input_tokens": 10, "gradient_forward_input_tokens": 10}},
            "fixed": {"same": {"forward_calls": 3, "forward_input_tokens": 20, "gradient_forward_input_tokens": 20}},
        })
        self.assertEqual(work["by_stage_sum_across_sources"]["same"]["forward_calls"], 5)
        self.assertEqual(work["grand_total"]["forward_input_tokens"], 30)

        def src_work(stage, n):
            return {stage: {"forward_calls": n, "forward_input_tokens": n * 3, "gradient_forward_input_tokens": n * 3}}
        components = {"work": {}, "seconds": 1., "optimizer_updates": 0}
        cf = {"work": {}, "results": [], "type": "AssertionError"}
        inter = {"work": {}, "seconds": 2., "updates": 48, "gradient_probe_forward_calls": 66}
        iff = {"work": src_work("v5_seed17_post_base0_gradient", 1), "seconds": 3., "gradient_probe_forward_calls": 0, "attempted_wake_updates": 0}
        qual = {"work": {}, "seconds": 4., "SGD_updates": 18, "Sleep_Adam_updates": 48}
        ledger = summary.build_work_ledger(components, cf, inter, iff, qual)
        self.assertEqual(ledger["logical_cost_and_elapsed"]["failed_gradient_probe_derived_calls_from_work_stages"], 1)
        self.assertTrue(ledger["logical_cost_and_elapsed"]["failed_gradient_probe_counter_mismatch"])
        self.assertIsNone(ledger["logical_cost_and_elapsed"]["component_failed_seconds"])


if __name__ == "__main__":
    unittest.main()
