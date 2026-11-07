import copy
import hashlib
import unittest

from tools.reanalyze_clinc_readout_v9 import (
    EXPECTED_CONTEXTS,
    analyze_result,
)


def sha(ch):
    return hashlib.sha256(ch.encode("ascii")).hexdigest()


def metric(count, correct, invalid=0, delta=None):
    value = {"count": count, "correct": correct, "accuracy": correct / count, "invalid_predictions": invalid}
    if delta is not None:
        value["delta_correct_vs_v8_generation"] = delta
    return value


def fixture():
    contexts = {}
    for i, name in enumerate(EXPECTED_CONTEXTS):
        greedy0, greedy1 = i % 5, (i + 1) % 6
        greedy = {
            "task0": metric(30, greedy0, 1),
            "task1": metric(30, greedy1, 2),
            "all60": metric(60, greedy0 + greedy1, 3),
        }
        sum0, sum1 = greedy0 + 2, greedy1 + 3
        avg0, avg1 = greedy0 + 1, greedy1 + 2
        sum_readout = {
            "task0": metric(30, sum0, delta=2),
            "task1": metric(30, sum1, delta=3),
            "all60": metric(60, sum0 + sum1, delta=5),
        }
        avg_readout = {
            "task0": metric(30, avg0, delta=1),
            "task1": metric(30, avg1, delta=2),
            "all60": metric(60, avg0 + avg1, delta=3),
        }
        contexts[name] = {
            "v8_greedy_generation": greedy,
            "teacher_forced_sum": sum_readout,
            "teacher_forced_length_mean": avg_readout,
            "gold_length_mean_nll": {
                "task0": {"count": 30, "mean": 1.0 + i},
                "task1": {"count": 30, "mean": 3.0 + i},
                "all60": {"count": 60, "mean": 2.0 + i},
            },
        }
    h = {name: sha(name) for name in EXPECTED_CONTEXTS}
    return {
        "schema": "clinc-label-readout-diagnostic-v9",
        "mode": "evaluate",
        "development_only": True,
        "fixed_cohort": {"classes": 30, "rows": 60, "rows_per_class": 2, "task0_rows": 30, "task1_rows": 30,
                         "stable_id_set_sha256": sha("cohort")},
        "label_vocabulary": {"count": 150, "all_labels_as_candidates": True, "candidate_sequences": 9000,
                             "canonical_order_sha256": sha("labels")},
        "immutability": {"all_adapters_frozen": True, "backbone_versions_unchanged": True,
                          "context_hashes_before": h, "context_hashes_after": dict(h)},
        "context_map_sha256": sha("map"), "master_config_sha256": sha("cfg"),
        "source_lock_sha256": sha("source"), "training_manifest_sha256": sha("manifest"),
        "methods": {
            "candidate_scoring_work": {"backbone_forwards": 1136, "tail_calls": 4544},
            "v8_greedy_generation": {"candidate_constrained": False, "primary_reference": True,
                                     "work": {"backbone_forwards": 303, "tail_calls": 303}},
            "teacher_forced_sum": {"candidate_constrained": True, "includes_eos": True},
            "teacher_forced_length_mean": {"candidate_constrained": True, "includes_eos": True},
        },
        "work": {"backbone_forwards": 1439, "tail_calls": 4847, "optimizer_updates": 0,
                 "input_tokens": 12345, "elapsed_seconds": 10.5},
        "metrics": contexts,
    }


class ReadoutReanalysisTests(unittest.TestCase):
    def test_recomputes_accuracies_deltas_nll_and_work(self):
        report = analyze_result(fixture(), input_sha256=sha("accepted"))
        hope = report["contexts"]["HOPE_TASK1"]
        greedy = hope["readouts"]["v8_greedy_generation"]
        forced = hope["readouts"]["teacher_forced_sum"]
        self.assertEqual(forced["all60"]["delta_correct_vs_v8_generation"],
                         forced["all60"]["correct"] - greedy["all60"]["correct"])
        self.assertEqual(hope["gold_length_mean_nll"]["all60"]["mean"],
                         (hope["gold_length_mean_nll"]["task0"]["mean"] + hope["gold_length_mean_nll"]["task1"]["mean"]) / 2)
        self.assertEqual(report["work"]["candidate_tail_calls"], 4 * report["work"]["candidate_backbone_forwards"])

    def test_rejects_denominator_or_invalid_candidate_inconsistency(self):
        bad = fixture()
        bad["metrics"]["FROZEN_BASE"]["teacher_forced_sum"]["task0"]["count"] = 29
        with self.assertRaisesRegex(ValueError, "denominator/counters"):
            analyze_result(bad, input_sha256=sha("accepted"))
        bad = fixture()
        bad["metrics"]["FROZEN_BASE"]["teacher_forced_sum"]["task0"]["invalid_predictions"] = 1
        with self.assertRaisesRegex(ValueError, "has invalid predictions"):
            analyze_result(bad, input_sha256=sha("accepted"))

    def test_rejects_unweighted_nll_and_work_counter_drift(self):
        bad = fixture()
        bad["metrics"]["HOPE_TASK1"]["gold_length_mean_nll"]["all60"]["mean"] += 0.1
        with self.assertRaisesRegex(ValueError, "weighted task mean"):
            analyze_result(bad, input_sha256=sha("accepted"))
        bad = fixture()
        bad["work"]["tail_calls"] += 1
        with self.assertRaisesRegex(ValueError, "total tail-call count mismatch"):
            analyze_result(bad, input_sha256=sha("accepted"))

    def test_curated_capture_allows_new_file_hash_and_retains_raw_source_provenance(self):
        wrapped = fixture()
        wrapped["_public_capture"] = {
            "schema": "clinc-v9-public-readout-capture-v1",
            "source_artifact_sha256_raw": sha("original raw artifact"),
        }
        curated_file_sha = sha("same metrics with capture metadata and normalized paths")
        report = analyze_result(wrapped, input_sha256=curated_file_sha)
        self.assertEqual(report["input"]["sha256"], curated_file_sha)
        self.assertEqual(report["source_provenance"]["source_artifact_sha256_raw"], sha("original raw artifact"))
        self.assertEqual(report["contexts"]["HOPE_TASK1"]["readouts"]["teacher_forced_sum"]["all60"]["correct"],
                         fixture()["metrics"]["HOPE_TASK1"]["teacher_forced_sum"]["all60"]["correct"])

        wrapped["_public_capture"]["source_artifact_sha256_raw"] = "not-a-sha"
        with self.assertRaisesRegex(ValueError, "source_artifact_sha256_raw must be a SHA-256"):
            analyze_result(wrapped, input_sha256=curated_file_sha)


if __name__ == "__main__":
    unittest.main()
