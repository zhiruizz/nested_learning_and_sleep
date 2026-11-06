"""Pure-CPU checks for the offline AG News v6 analyzer."""
import json
import tempfile
import unittest
from pathlib import Path

from tools import summarize_agnews_routing_fairness_v6 as a


def test_csv_output_uses_git_canonical_lf_bytes(tmp_path):
    path = tmp_path / 'table.csv'
    a.write_csv(path, [{'name': 'context', 'score': 0.75}], ['name', 'score'])
    assert path.read_bytes() == b'name,score\ncontext,0.75\n'


def fake_metrics(total, old, new, per_class, *, em=0.8):
    out = {}
    sizes = {"all": total, "old": old, "new": new,
             **{f"class{i}": per_class for i in range(4)}}
    for group, n in sizes.items():
        out[group] = {"examples": n}
        for metric in a.METRICS:
            out[group][metric] = em if "nll" not in metric and "nats" not in metric else 0.25
    return out


class RoutingFairnessAnalyzerTests(unittest.TestCase):
    def test_sample_mean_and_sd(self):
        self.assertEqual(a.mean_sd([1, 2, 3]),
                         {"n": 3, "mean": 2.0, "sample_sd": 1.0, "values": [1.0, 2.0, 3.0]})
        self.assertEqual(a.POSTHOC_TO_PHYSICAL["warm_alone"], "constant_old")
        self.assertEqual(a.POSTHOC_TO_PHYSICAL["new_CMS_reference_alone"], "constant_new")

    def test_work_ledgers_sum_shared_stage_names(self):
        one = {"same": {"forward_calls": 2, "forward_input_tokens": 8,
                         "gradient_forward_input_tokens": 4}}
        two = {"same": {"forward_calls": 3, "forward_input_tokens": 9,
                         "gradient_forward_input_tokens": 5}}
        got = a.merge_work_ledgers({"a": one, "b": two})
        self.assertEqual(got["combined_by_stage"]["same"],
                         {"forward_calls": 5, "forward_input_tokens": 17,
                          "gradient_forward_input_tokens": 9})
        self.assertEqual(got["combined_total"], got["combined_by_stage"]["same"])

    def test_train_and_validation_denominators_are_distinct(self):
        train = fake_metrics(1024, 512, 512, 256)
        val = fake_metrics(512, 256, 256, 128)
        a.validate_metrics(train, total=1024, old=512, new=512, classes=256)
        a.validate_metrics(val, total=512, old=256, new=256, classes=128)
        with self.assertRaises(ValueError):
            a.validate_metrics(val, total=1024, old=512, new=512, classes=256)
        self.assertEqual(a.split_sizes("train"), (1024, 512, 512, 256))
        self.assertEqual(a.split_sizes("validation"), (512, 256, 256, 128))

    def test_qualification_gate_uses_global_and_each_class_thresholds(self):
        train = fake_metrics(1024, 512, 512, 256, em=0.80)
        val = fake_metrics(512, 256, 256, 128, em=0.75)
        self.assertTrue(a.gate_status(train, val)["all_tasks_pass"])
        val["class3"][a.METRICS[0]] = 0.699
        self.assertFalse(a.gate_status(train, val)["all_tasks_pass"])

    def test_endpoint_gate_ledger_is_6_raw_plus_18_routed(self):
        train = fake_metrics(1024, 512, 512, 256)
        val = fake_metrics(512, 256, 256, 128)
        rows = []
        for seed in a.SEEDS:
            for arm in ("aux_new_CE", "aux_old_gold_CE"):
                posthoc = {condition: {"train": train, "validation": val}
                           for condition in a.CONDITIONS[:-1]}
                rows.append({"seed": seed, "new_context": arm,
                    "new_branch_train": train, "new_branch_validation": val,
                    "posthoc": posthoc})
        got = a.build_gates([], rows)
        self.assertEqual(len(got), 24)
        self.assertEqual(sum(row["stage"] == "B_raw_endpoint" for row in got), 6)
        self.assertEqual(sum(row["stage"] == "A_routed_endpoint" for row in got), 18)

    def test_storage_audit_distinguishes_per_gate_and_two_gate_bytes(self):
        storage = {"schema": "agnews-routing-storage-audit-v6", "scoring_or_training": False,
            "main_seal_sha256": "seal", "all_three_gate_fits_identical": True,
            "rows": [{"seed": seed, "checkpoint_sha256": {"warm": f"audit-w{seed}", "new448": f"n{seed}"},
                      "tensor_bytes_per_context": 27396800,
                      "deduplicated_whole_tensor_storage_bytes": 28576459,
                      "distinct_tensor_count": 9} for seed in a.SEEDS],
            "v5_gate_feature_dims": [3072, 3072, 3072],
            "minimal_gate_payload_bytes_float64_mean_scale_weight": 73728,
            "v5_gate_model_sha256": ["same"] * 3}
        got = a.storage_summary(storage, {"source_sha256": {"v5_seal_sha256": "seal",
            "checkpoint_sha256_by_seed": {str(seed): {"warm": f"input-w{seed}", "v5_cms_448": f"n{seed}"}
                                         for seed in a.SEEDS}}})
        self.assertEqual(got["one_float64_gate_payload_bytes"], 73728)
        self.assertEqual(got["two_float64_gate_payload_bytes"], 147456)
        self.assertFalse(got["warm_checkpoint_file_sha_matches_a_input_by_seed"]["17"])

    def test_b_inherited_step448_validation_keeps_512_row_denominator(self):
        val = fake_metrics(512, 256, 256, 128)
        train = fake_metrics(1024, 512, 512, 256)
        source = {"driver_source_sha256": a.EXPECTED_B_SOURCE,
                  "continuation_helper_sha256": a.EXPECTED_B_HELPER}
        protocol = {"schema": "agnews-fair-continuation-protocol-v6",
            "qualification_sha256": "qsha", "planned_primary_updates": 3456,
            "source_sha256": source}
        seal = {"schema": "agnews-fair-continuation-development-seal-v6",
            "complete": True, "development_only": True,
            "confirmation_scored": False, "no_holdout_input": True,
            "protocol_sha256": "psha", "source_sha256": source,
            "runs": []}
        for seed in a.SEEDS:
            for arm in ("aux_new_CE", "aux_old_gold_CE"):
                seal["runs"].append({"seed": seed, "arm": arm, "status": "complete",
                    "primary_updates": 576, "start_step": 448,
                    "train_final": train,
                    "validation": [{"primary_step": step, "metrics": val}
                                   for step in (448, 512, 768, 1024)]})
        qualification = {"passed": True, "scoring": False}
        a.validate_b_main(protocol, seal, qualification, "psha", "qsha")

    def test_preserved_r0_failure_stays_separate_and_checks_raw_meter(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as td:
            root = Path(td)
            account = {"schema": "agnews-v6-metric-replay-failure-account-v1",
                "source_sha256": "3fdcbc79c744aef8e5ac9f4fd0de83f763e8fe5da7811a756edc5f53e5ff96dd",
                "completed_selector_pairs": 0, "actual_optimizer_updates": 0,
                "actual_ridge_fits": 0, "parity_forwards": 4,
                "parity_training_example_rows": 8, "train_rows_scored": 1024,
                "validation_rows_scored": 512, "active_seconds": 20.812,
                "process_wall_seconds": 24.891, "forward_calls": 196,
                "forward_input_tokens": 151672, "gradient_forward_calls": 0,
                "automatic_retry": False, "parity_guard_passed_before_failure": True,
                "parity_numeric_difference_not_persisted_in_failed_attempt": True,
                "partial_numeric_scores_not_persisted_in_failed_attempt": True,
                "reference_guard_relaxed": False, "confirmation_scored": False,
                "error": "parity mismatch", "evaluation_precision_issue": "FP32 aggregate rounding order"}
            failure = {"schema": "agnews-routing-fairness-failure-v6",
                "completed_pairs": 0, "confirmation_scored": False,
                "work": {"parity": {"forward_calls": 4, "forward_input_tokens": 3328,
                                     "gradient_forward_input_tokens": 0},
                         "train": {"forward_calls": 128, "forward_input_tokens": 100480,
                                   "gradient_forward_input_tokens": 0},
                         "validation": {"forward_calls": 64, "forward_input_tokens": 47864,
                                        "gradient_forward_input_tokens": 0}}}
            (root / "account.json").write_text(json.dumps(account), encoding="utf-8")
            (root / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
            got = a.load_a_initial_failure(root / "account.json")
            self.assertEqual(got["work"], {"forward_calls": 196,
                "forward_input_tokens": 151672, "gradient_forward_input_tokens": 0})
            failure["work"]["validation"]["forward_calls"] = 65
            (root / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
            with self.assertRaises(ValueError):
                a.load_a_initial_failure(root / "account.json")


if __name__ == "__main__":
    unittest.main()
