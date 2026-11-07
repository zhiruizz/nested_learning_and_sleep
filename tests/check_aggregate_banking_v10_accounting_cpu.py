"""Offline aggregation/accounting tests; no model, CUDA, or dataset imports."""
from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.aggregate_banking_v10_accounting import (
    MANIFEST_SCHEMA,
    _metric_grid,
    _test_metrics,
    aggregate_manifest,
    aggregate_work,
    load_pinned_model_config,
    normalize_work,
    summarize_values,
    _account_checkpoint,
    _is_terminal_checkpoint_relative,
)


class BankingOfflineAggregatorTests(unittest.TestCase):
    def test_terminal_checkpoint_path_accepts_exact_posix_and_windows_spellings_only(self):
        self.assertTrue(_is_terminal_checkpoint_relative("checkpoints/final-task06.pt"))
        self.assertTrue(_is_terminal_checkpoint_relative(r"checkpoints\final-task06.pt"))
        for malicious in (
            "../checkpoints/final-task06.pt",
            "checkpoints/../checkpoints/final-task06.pt",
            "/checkpoints/final-task06.pt",
            r"C:\checkpoints\final-task06.pt",
            r"C:checkpoints\final-task06.pt",
            r"\\server\share\checkpoints\final-task06.pt",
            "nested/checkpoints/final-task06.pt",
            "checkpoints//final-task06.pt",
            "checkpoints/./final-task06.pt",
            "checkpoints/final-task07.pt",
        ):
            with self.subTest(path=malicious):
                self.assertFalse(_is_terminal_checkpoint_relative(malicious))

    def test_work_histogram_verified_and_missing_cost_stays_unknown(self):
        work = normalize_work({
            "full_backbone_forward_calls": 2,
            "executed_input_tokens_including_padding": 18,
            "forward_shape_histogram": {"1x8": 1, "1x10": 1},
            "tail_calls": 3,
            "active_wall_seconds": 1.25,
        })
        self.assertEqual(work["full_backbone_forward_calls"], 2)
        self.assertEqual(work["tail_calls"], 3)
        missing_tail = normalize_work({"full_backbone_forward_calls": 1}, unknown_fields={"tail_calls"})
        self.assertIsNone(missing_tail["tail_calls"])
        total = aggregate_work([{"label": "known", "work": work},
                                {"label": "failure", "work": missing_tail}])
        self.assertEqual(total["full_backbone_forward_calls"]["total"], 3)
        self.assertIsNone(total["tail_calls"]["total"])
        self.assertEqual(total["tail_calls"]["known_sum"], 3)
        self.assertEqual(total["tail_calls"]["unknown_sources"], ["failure"])

    def test_bad_histogram_total_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not sum"):
            normalize_work({"full_backbone_forward_calls": 2,
                            "forward_shape_histogram": {"1x8": 1}})

    def test_legacy_input_tokens_remain_candidate_only_and_full_tokens_stay_unknown(self):
        legacy = normalize_work({"backbone_forwards": 5, "input_tokens": 123})
        self.assertEqual(legacy["candidate_input_tokens"], 123)
        self.assertEqual(legacy["candidate_input_tokens_source"], "legacy_input_tokens_candidate_only")
        self.assertIsNone(legacy["full_backbone_forward_tokens"])
        self.assertIsNone(legacy["executed_input_tokens_including_padding"])
        observed = normalize_work({
            "backbone_forwards": 2,
            "candidate_input_tokens": 6,
            "observed_backbone_valid_tokens": 12,
            "observed_backbone_executed_tokens_including_padding": 15,
            "observed_backbone_forward_shape_histogram": {"2x3": 1, "3x3": 1},
        })
        self.assertEqual(observed["candidate_input_tokens"], 6)
        self.assertEqual(observed["candidate_input_tokens_source"], "candidate_input_tokens")
        self.assertEqual(observed["full_backbone_forward_tokens"], 12)
        self.assertEqual(observed["executed_input_tokens_including_padding"], 15)

    def test_mean_sample_sd_and_paired_seed_differences(self):
        self.assertEqual(summarize_values([1.0, 2.0, 3.0]),
                         {"n": 3, "mean": 2.0, "sample_sd": 1.0})
        rows = []
        for seed, a, b, c in ((17, .2, .3, .4), (29, .4, .4, .5), (43, .6, .7, .8)):
            for arm, value in (("CMS_ONLY", a), ("SLEEP_ADDITIVE", b), ("GOLD_ADDITIVE", c)):
                rows.append({"seed": seed, "arm": arm, "metrics": {"accuracy": value}})
        result = _metric_grid(rows, metric_path=(), arms=("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE"))
        sleep = result["accuracy"]["paired_differences_vs_CMS_ONLY"]["SLEEP_ADDITIVE"]
        self.assertAlmostEqual(sleep["mean"], 0.0666666666667)
        self.assertAlmostEqual(sleep["sample_sd"], 0.0577350269)

    def _test_result(self, overlap=7):
        contexts = {}
        for seed in (17, 29, 43):
            for arm in ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE"):
                contexts[f"seed{seed}:{arm}"] = {
                    "teacher_forced_sum": {"count": 3080, "accuracy": .5},
                    "teacher_forced_length_mean": {"count": 3080, "accuracy": .4},
                    "greedy_generation": {"count": 3080, "accuracy": .3},
                    "gold_label_token_mean_nll": {"count": 3080, "mean": 1.7},
                }
        secondary = {}
        for name, metrics in contexts.items():
            secondary[name] = {k: {**v, "count": 3080 - overlap} for k, v in metrics.items()}
        return {"schema": "banking-cil-v10-test-evaluator-v1", "status": "complete",
            "optimizer_updates": 0, "all_contexts_frozen": True,
            "training": {"verified_contexts": [{"seed": s, "arm": a}
                for s in (17, 29, 43) for a in ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")]},
            "test_source": {"source_revision": "9d081458ff52e53cf7e848f414e6e9344e4e6696",
                "source_sha256": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
                "source_bytes": 239961, "raw_rows": 3080, "normalized_train_overlap_rows": overlap,
                "training_rows_considered": 9229, "development_rows_considered": 770,
                "train_development_rows_considered": 9999,
                "normalized_development_overlap_rows": 0,
                "normalized_train_or_development_overlap_rows": overlap,
                "train_development_disjoint_secondary_rows": 3080-overlap,
                "primary_includes_all_official_rows": True},
            "primary_raw_official_test": contexts,
            "secondary_train_development_disjoint_subset": secondary}

    def test_raw_primary_and_train_disjoint_test_remain_separate(self):
        result = _test_metrics(self._test_result())
        self.assertIn("primary_raw_official_test", result)
        self.assertIn("secondary_train_development_disjoint_subset", result)
        self.assertEqual(result["source"]["raw_rows"], 3080)
        self.assertIn("teacher_forced_sum", result["primary_raw_official_test"])

    def test_v10_runtime_scores_gold_nll_once_per_row(self):
        # Exercise the actual frozen `_score_rows` producer: each gold_nll item
        # is already a per-example mean across that row's answer tokens.
        from tools.banking_cil_v10_runtime import _score_rows
        classes = [f"label_{i}" for i in range(77)]
        rows = [{"intent_id": 2}, {"intent_id": 7}, {"intent_id": 7}]
        _metrics, gold = _score_rows({}, [2, 7, 1], [2, 7, 1], [1.0, 2.0, 3.0],
                                     rows, classes, None)
        self.assertEqual(gold["all77"]["count"], 3)
        self.assertAlmostEqual(gold["all77"]["mean_token_nll"], 2.0)

    def test_test_count_mismatch_rejected(self):
        result = self._test_result()
        result["secondary_train_development_disjoint_subset"]["seed17:CMS_ONLY"]["greedy_generation"]["count"] -= 1
        with self.assertRaisesRegex(ValueError, "unexpected row count"):
            _test_metrics(result)

    def test_sha_bound_cpu_checkpoint_context_accounting(self):
        import hashlib
        import torch
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            root = Path(td)
            context = {
                "memory": {
                    "0.a": torch.zeros(2, 3), "0.b": torch.zeros(4, 2),
                    "0.experts.slots.0.a": torch.zeros(2, 3),
                    "0.experts.slots.0.b": torch.zeros(3, 2),
                    "0.experts.slots.0.active": torch.tensor(True),
                },
                "readout_a": torch.zeros(2, 3), "readout_b": torch.zeros(3, 2),
                "readout_fp32_accumulation": False, "routing": [None],
                "trainable": {"memory.0.a": False, "memory.0.b": False,
                    "memory.0.experts.slots.0.a": False, "memory.0.experts.slots.0.b": False,
                    "readout_a": False, "readout_b": False},
            }
            cp = root / "tiny.pt"
            torch.save({"schema": "banking-cil-v10-checkpoint", "seed": 17,
                        "arm": "CMS_ONLY", "task_id": 6, "context": context,
                        "config_sha256": "cfg", "source_lock_sha256": "src",
                        "training_manifest_sha256": "data"}, cp)
            result = _account_checkpoint({"label": "tiny", "seed": 17, "arm": "CMS_ONLY",
                "task_id": 6, "checkpoint_path": str(cp),
                "checkpoint_sha256": hashlib.sha256(cp.read_bytes()).hexdigest()},
                root=root, binding={"config_sha256": "cfg", "source_lock_sha256": "src",
                                    "training_manifest_sha256": "data"})
            self.assertEqual(result["accounting"]["stored_mutable_parameters"], 38)
            self.assertEqual(result["accounting"]["corrected_per_token_unique_executed_upper"], 38)
            self.assertEqual(result["accounting"]["per_token_expert_branch_parameter_evaluations_upper"], 12)

    def test_partial_backbone_flop_estimate_is_scoped(self):
        from tools.banking_cost_estimates_v10 import frozen_backbone_matmul_estimate
        value = frozen_backbone_matmul_estimate(
            {"forward_shape_histogram": {"1x2": 1}, "executed_input_tokens_including_padding": 2},
            {"hidden_size": 4, "vocab_size": 10, "num_hidden_layers": 1,
             "intermediate_size": 6, "num_attention_heads": 2, "num_key_value_heads": 1})
        self.assertEqual(value["partial_forward_flops_estimate"], 704)
        self.assertFalse(value["measured_complete_flops"] is not None)

    def test_model_dimensions_require_matching_asset_hash(self):
        config = {"hidden_size": 4, "vocab_size": 10, "num_hidden_layers": 1,
                  "intermediate_size": 6, "num_attention_heads": 2,
                  "num_key_value_heads": 1, "extra": "ignored"}
        raw = json.dumps(config).encode()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            path = Path(td) / "config.json"
            path.write_bytes(raw)
            sha = hashlib.sha256(raw).hexdigest()
            self.assertEqual(load_pinned_model_config(path, sha)["hidden_size"], 4)
            with self.assertRaisesRegex(ValueError, "frozen asset SHA"):
                load_pinned_model_config(path, "0" * 64)

    def test_manifest_preserves_attempt_and_unknown_tail_counters(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            root = Path(td)
            foundation = {"schema": "banking-cil-v10-foundation-receipt", "status": "qualified",
                "seed": 17, "arm": "FOUNDATION_CMS_ONLY", "config_sha256": "cfg",
                "source_lock_sha256": "src", "training_manifest_sha256": "data",
                "checkpoint_sha256": "a" * 64, "test_data_included": False,
                "model_immutable_versions_unchanged": True, "metric": {"accuracy": .6},
                "mutable_parameter_counts": {"per_token_executed_mutable_parameters_upper_bound": 99},
                "work": {"full_backbone_forward_calls": 8, "tail_calls": 14,
                         "active_wall_seconds": 2.5}}
            (root / "foundation.json").write_text(json.dumps(foundation), encoding="utf-8")
            failure = {"schema": "banking-cil-v10-transfer-probe-v1", "status": "failed",
                "work": {"full_backbone_forward_calls": 3, "tail_calls": 0,
                         "physical_adamw_updates": 1, "active_wall_seconds": 1.0}, "arms": {}}
            (root / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
            ledger = {"failure_tail_counter_limitation": "tail fields were not merged; zero is unknown",
                      "known_full_backbone_forwards": 3, "known_physical_adam_steps": 1,
                      "known_backward_calls": 1, "active_wall_seconds": 1.0}
            (root / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
            manifest = {"schema": MANIFEST_SCHEMA,
                "bindings": {"config_sha256": "cfg", "source_lock_sha256": "src",
                             "training_manifest_sha256": "data"},
                "foundations": [{"seed": 17, "receipt_path": "foundation.json"}],
                "probe_attempts": [{"label": "attempt-A-failure", "failure_path": "failure.json",
                                    "ledger_path": "ledger.json"}]}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = aggregate_manifest(root / "manifest.json", root / "result.json")
            self.assertEqual(output["foundations"][0]["legacy_mutable_parameter_counts_preserved"][
                "per_token_executed_mutable_parameters_upper_bound"], 99)
            work = {row["label"]: row["work"] for row in output["work_by_source"]}
            self.assertIsNone(work["attempt-A-failure"]["tail_calls"])
            self.assertEqual(work["attempt-A-failure"]["full_backbone_forward_calls"], 3)
            self.assertEqual(work["attempt-A-failure"]["tail_backward_calls"], 1)
            self.assertEqual(output["work_totals"]["tail_calls"]["known_sum"], 14)
            self.assertIsNone(output["work_totals"]["tail_calls"]["total"])
            self.assertIn("attempt-A-failure", output["work_totals"]["tail_calls"]["unknown_sources"])


if __name__ == "__main__":
    unittest.main()
