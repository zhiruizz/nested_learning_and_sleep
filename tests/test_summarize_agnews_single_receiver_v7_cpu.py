"""Pure-stdlib synthetic schema tests for the v7 offline result validator."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import summarize_agnews_single_receiver_v7 as summary


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def metrics(split, offset=0.0):
    groups = summary.EXPECTED_COUNTS[split]
    out = {}
    for j, (group, n) in enumerate(groups.items()):
        out[group] = {"examples": n, "em": .70 + offset + j * .001,
                      "marker": .9, "class": .8, "candidate": .75,
                      "marker_nll": 1.2, "class_nll": 1.4, "nll": 1.3}
    return out


def fixture(root):
    main = root / "main"
    main.mkdir()
    config_path = root / "config.json"
    qual_path = root / "qualification.json"
    qwork_path = root / "qualification-work.json"
    cache_path = root / "cache.json"
    loss_path = root / "loss.json"
    driver = "d" * 64
    records = "r" * 64
    warm = "w" * 64
    v6 = "v" * 64
    config = {"driver_sha256": driver, "records_sha256": records,
              "warm_result_sha256": warm, "v6_seal_sha256": v6,
              "status": "frozen_development", "development_only": True,
              "seeds": list(summary.SEEDS), "arms": list(summary.ARMS), "steps": 1024,
              "evaluation_steps": list(summary.EVAL_STEPS), "main_Adam_updates_expected": 30720,
              "scope_trainable_parameters": summary.PARAM_COUNTS,
              "learning_rate": .003, "weight_decay": 0.0, "gradient_clip": 1.0,
              "elapsed_limit_seconds": 28800, "per_run_elapsed_limit_seconds": 900,
              "schedule_sha256_by_seed_arm": {},
              "numerical_qualifications": {
                  "cache": {"path": str(cache_path), "sha256": ""},
                  "loss_paths": {"path": str(loss_path), "sha256": ""}}}
    put(cache_path, {"passed": True, "confirmation_scored": False, "optimizer_updates": 0})
    put(loss_path, {"passed": True, "confirmation_scored": False, "actual_Adam_updates": 16})
    config["numerical_qualifications"]["cache"]["sha256"] = summary.sha256(cache_path)
    config["numerical_qualifications"]["loss_paths"]["sha256"] = summary.sha256(loss_path)
    put(config_path, config)
    protocol = {"schema": "agnews-single-receiver-protocol-v7", "status": "frozen_development",
                "config_sha256": summary.sha256(config_path), "driver_sha256": driver,
                "records_sha256": records, "warm_result_sha256": warm, "v6_seal_sha256": v6,
                "seeds": list(summary.SEEDS), "arms": list(summary.ARMS), "steps": 1024,
                "evaluation_steps": list(summary.EVAL_STEPS), "padded_sequence_length": 118,
                "confirmation_scored": False, "confirmation_input": False}
    put(main / "protocol.json", protocol)
    protocol_sha = summary.sha256(main / "protocol.json")
    qual = {"passed": True, "exact_model_optimizer_policy_rng": True,
            "config_sha256": summary.sha256(config_path), "driver_sha256": driver,
            "actual_Adam_updates": 12, "scoring": False, "confirmation_scored": False,
            "results": [{"scope": s, "exact_model_optimizer_policy_rng": True} for s in summary.SCOPES],
            "matmul_work_lower_bound": {}}
    put(qual_path, qual)
    runs = []
    for seed in summary.SEEDS:
        init = {"adapter_sha256": f"a{seed}", "policy_sha256": f"p{seed}"}
        for arm in summary.ARMS:
            config["schedule_sha256_by_seed_arm"][f"{seed}:{arm}"] = f"{seed:02x}" * 32
            train0, val0 = metrics("train", seed * .00001), metrics("validation", seed * .00001)
            train1, val1 = metrics("train", seed * .00001 + .01), metrics("validation", seed * .00001 + .01)
            evs = [{"step": 0, "train": train0, "validation": val0},
                   {"step": 64, "train": train0, "validation": val0},
                   {"step": 256, "train": train0, "validation": val0},
                   {"step": 1024, "train": train1, "validation": val1}]
            runs.append({"seed": seed, "arm": arm, "status": "complete", "updates": 1024,
                         "trainable_parameter_count": summary.PARAM_COUNTS[arm.rsplit("_", 1)[1]],
                         "frozen_scope_guard_passed": True, "initial": init,
                         "schedule_sha256": config["schedule_sha256_by_seed_arm"][f"{seed}:{arm}"],
                         "evaluations": evs, "train_final": train1,
                         "checkpoints": [{"step": s, "path": f"{seed}-{arm}-{s}.pt", "sha256": "x" * 64}
                                         for s in (64, 256, 1024)],
                         "parity": {"full": val1, "tail": val1},
                         "work": {"forward": {"flops": 10, "calls": 1, "ops": {}},
                                  "backward": {"flops": 20, "calls": 2, "ops": {}}},
                         "student_forward_passes": 1024, "student_backward_passes": 1024,
                         "student_input_tokens": 1000, "seconds": 1.0,
                         "peak_cuda_allocated_bytes": 10, "peak_cuda_reserved_bytes": 20,
                         "cache_file_bytes": 30, "model_tensor_bytes": 40,
                         "unique_training_examples": 20 if arm.startswith("new_only_") else 40,
                         "class_draws": {str(k): (0 if arm.startswith("new_only_") and k < 2 else
                                                  4096 if arm.startswith("new_only_") else 2048)
                                         for k in range(4)},
                         "class_unique_examples": {str(k): (0 if arm.startswith("new_only_") and k < 2 else 10)
                                                   for k in range(4)},
                         "answer_token_targets": 16384,
                         "direct_gold_answer_draws": 8192 if arm.startswith(("gold_", "new_only_")) else 0,
                         "teacher_origin_selection_draws": 8192 if arm.startswith("split") else 0})
    put(config_path, config)
    # Re-write protocol after schedule hashes are finalized; its config SHA is now final.
    protocol["config_sha256"] = summary.sha256(config_path)
    put(main / "protocol.json", protocol)
    protocol_sha = summary.sha256(main / "protocol.json")
    qual["config_sha256"] = summary.sha256(config_path)
    put(qual_path, qual)
    put(qwork_path, {"schema": "agnews-single-receiver-authoritative-matmul-work-v7",
                     "driver_sha256": driver, "optimizer_attempts": 12, "seconds": 4.0,
                     "scope": "Executed mm/addmm/bmm/mv/dot shapes, MAC=2; arithmetic lower bound.",
                     "work": {"qualification:train": {"forward": {"flops": 40, "calls": 4}}}})
    protocol["qualification_sha256"] = summary.sha256(qual_path)
    put(main / "protocol.json", protocol)
    protocol_sha = summary.sha256(main / "protocol.json")
    tq = []
    for seed in summary.SEEDS:
        wtrain = metrics("train")
        wval = metrics("validation")
        wtrain["old"]["em"] = .9
        wval["old"]["em"] = .9
        wval["class0"]["em"] = .8
        wval["class1"]["em"] = .8
        tq.append({"seed": seed, "warm_old_gate_passed": True,
                   "warm_oldteacher": {"train_old_only": {"old": wtrain["old"], "class0": wtrain["class0"], "class1": wtrain["class1"]},
                                       "validation_old_only": {"old": wval["old"], "class0": wval["class0"], "class1": wval["class1"]}},
                   "current_newteacher": {"train": metrics("train"), "validation": metrics("validation")}})
    seal = {"schema": "agnews-single-receiver-development-v7", "complete": True,
            "protocol_sha256": protocol_sha, "runs": runs, "teacher_quality_by_seed": tq,
            "common_initial_by_seed": {str(s): {"adapter_sha256": f"a{s}", "policy_sha256": f"p{s}"}
                                       for s in summary.SEEDS},
            "confirmation_input": False, "confirmation_scored": False, "optimizer_attempts": 30720}
    put(main / "development-sealed.json", seal)
    put(main / "execution-work.json", {"schema": "agnews-single-receiver-authoritative-matmul-work-v7",
                                       "driver_sha256": driver, "optimizer_attempts": 30720, "seconds": 100,
                                       "scope": "Executed mm/addmm/bmm/mv/dot shapes, MAC=2; arithmetic lower bound. Per-run local counters overlap it; never add them.",
                                       "work": {"seed17:train": {"forward": {"flops": 100, "calls": 10},
                                                                    "backward": {"flops": 200, "calls": 20}}}})
    return main, config_path, qual_path, cache_path, loss_path, qwork_path


class SummaryTests(unittest.TestCase):
    def test_recovery_attempt_bound_is_a_count_not_step_interval(self):
        count = summary._recovery_update_accounting(28 * 1024, 704, 1792, [0, 64])
        self.assertEqual(count["logical_updates"], 30720)
        self.assertEqual(count["known_observed_update_attempts"], 31168)
        self.assertEqual((count["unknown_original_update_attempts_min"],
                          count["unknown_original_update_attempts_max"]), (0, 64))
        self.assertIn("count bound", count["interpretation"])

    def test_complete_grid_summary_and_nonadditive_work(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            root = Path(td)
            main, cfg, qual, cache, loss, qwork = fixture(root)
            prior = root / "prior-qualification-failure.json"
            prior_work = root / "prior-failure-work.json"
            put(prior, {"schema": "agnews-single-receiver-qualification-failure-v7",
                        "error_type": "RuntimeError", "error": "mocked prior rejection",
                        "actual_Adam_updates": 2, "confirmation_scored": False})
            put(prior_work, {"schema": "agnews-single-receiver-authoritative-matmul-work-v7",
                             "driver_sha256": "o" * 64, "optimizer_attempts": 2, "seconds": 41.0,
                             "scope": "Executed mm/addmm/bmm/mv/dot shapes, MAC=2; arithmetic lower bound.",
                             "work": {"qualification:failed": {"forward": {"flops": 60, "calls": 6}}}})
            inputs = summary._validate_main(main, cfg, qual, cache, loss, qwork, prior, prior_work)
            analysis = summary.build_analysis(inputs)
            self.assertEqual(len(analysis["endpoints"]), 10)
            self.assertEqual(analysis["endpoints"]["split_receiver"]["validation"]["old"]["em"]["n"], 3)
            self.assertEqual(analysis["work_ledger"]["authoritative_main_work"]["flops"], 300)
            self.assertEqual(analysis["work_ledger"]["per_run_local_counter_sum_overlap_not_additive"]["flops"], 900)
            self.assertEqual(analysis["work_ledger"]["qualification_authoritative_work"]["authoritative_work"]["flops"], 40)
            self.assertEqual(analysis["work_ledger"]["prior_failure_separate_not_in_main_total"]["actual_Adam_updates"], 2)
            self.assertEqual(analysis["work_ledger"]["prior_failure_separate_not_in_main_total"]["authoritative_work"]["flops"], 60)
            self.assertEqual(len(analysis["paired_contrasts"]), 8)
            out = root / "out"
            summary._write_reports(out, analysis)
            self.assertTrue((out / "endpoint-table.md").is_file())
            self.assertTrue((out / "analysis.json").is_file())

    def test_incomplete_main_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            root = Path(td); main, cfg, qual, cache, loss, _ = fixture(root)
            seal_path = main / "development-sealed.json"
            seal = json.loads(seal_path.read_text(encoding="utf-8")); seal["runs"].pop()
            put(seal_path, seal)
            with self.assertRaisesRegex(ValueError, "exactly 30"):
                summary._validate_main(main, cfg, qual, cache, loss)

    def test_wrong_denominator_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            root = Path(td); main, cfg, qual, cache, loss, _ = fixture(root)
            seal_path = main / "development-sealed.json"
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            seal["runs"][0]["evaluations"][0]["train"]["old"]["examples"] = 256
            put(seal_path, seal)
            # The protocol remains unchanged and points to the same bytes; fail on the denominator.
            with self.assertRaisesRegex(ValueError, "wrong denominator"):
                summary._validate_main(main, cfg, qual, cache, loss)

    def test_authoritative_driver_and_qualification_must_match(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            root = Path(td); main, cfg, qual, cache, loss, _ = fixture(root)
            q = json.loads(qual.read_text(encoding="utf-8")); q["actual_Adam_updates"] = 11
            put(qual, q)
            protocol_path = main / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["qualification_sha256"] = summary.sha256(qual)
            put(protocol_path, protocol)
            seal_path = main / "development-sealed.json"
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            seal["protocol_sha256"] = summary.sha256(protocol_path)
            put(seal_path, seal)
            with self.assertRaisesRegex(ValueError, "update count"):
                summary._validate_main(main, cfg, qual, cache, loss)


if __name__ == "__main__":
    unittest.main()
