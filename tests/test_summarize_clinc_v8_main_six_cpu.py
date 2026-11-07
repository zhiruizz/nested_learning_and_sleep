import copy
import hashlib
import shutil
import tempfile
from pathlib import Path
import unittest

from tools import summarize_clinc_v8_main_six as summary


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def leaf(correct, count, classes, invalid=0):
    if not count:
        return {"correct": None, "count": 0, "class_count": 0,
                "accuracy": None, "macro_accuracy": None, "invalid": None}
    a = correct / count
    return {"correct": correct, "count": count, "class_count": classes,
            "accuracy": a, "macro_accuracy": a, "invalid": invalid}


def dev_cohorts(config):
    current_rows = {
        17: [300, 300, 299, 300, 300, 300, 299, 300, 299, 300],
        29: [300, 300, 299, 300, 300, 300, 299, 299, 300, 300],
        43: [300, 299, 299, 300, 300, 300, 300, 300, 300, 299],
    }
    seeds = {}
    for seed, rows in current_rows.items():
        tasks = {}
        old_rows = 0
        for task, current in enumerate(rows):
            tasks[str(task)] = {
                "old": {"class_count": task * 15, "count": old_rows},
                "current": {"class_count": 15, "count": current},
                "seen": {"class_count": (task + 1) * 15, "count": old_rows + current},
            }
            old_rows += current
        seeds[str(seed)] = tasks
    return {"schema": "clinc-v8-expected-clean-development-cohorts-v1",
        "model_loaded": False, "optimizer_steps": 0, "test_data_read": False,
        "total_clean_rows": 2997, "normalized_overlap_rows": 3,
        "training_manifest_sha256": config["training_manifest_sha256"], "seeds": seeds}


def derivation_fixture(cohorts):
    rows = [{"intent_id": i} for i in range(150) for _ in range(19 if i < 3 else 20)]
    low_tasks = {
        17: [2, 6, 8], 29: [2, 6, 7], 43: [1, 2, 9],
    }
    orders = {}
    for seed in summary.SEEDS:
        low_ids = iter((0, 1, 2))
        order = []
        deficient = set(low_tasks[seed])
        remaining = iter(range(3, 150))
        for task in range(10):
            block = []
            if task in deficient:
                block.append(next(low_ids))
            block.extend(next(remaining) for _ in range(15-len(block)))
            order.extend(block)
        orders[str(seed)] = order
    return orders, rows


def task_metrics(task_id, seed_i, sleep, cohort):
    oc = cohort["old"]["count"]
    cc = cohort["current"]["count"]
    old_classes = task_id * 15
    current_classes = 15
    old_correct = round(oc * (0.62 + seed_i * 0.01 - task_id * 0.01)) if oc else 0
    current_correct = round(cc * (0.75 + seed_i * 0.01 + (0.01 if sleep else 0)))
    seen_count = oc + cc
    seen_correct = old_correct + current_correct
    old = leaf(old_correct, oc, old_classes, min(oc-old_correct, oc//10) if oc else 0)
    current = leaf(current_correct, cc, current_classes, min(cc-current_correct, cc//10))
    seen = leaf(seen_correct, seen_count, old_classes + current_classes,
                (old["invalid"] or 0) + current["invalid"])
    return {"task_id": task_id, "old_count": oc, "current_count": cc,
            "seen_count": seen_count,
            "result": {"metrics": {"old": old, "current": current, "seen": seen}}}


def make_fixture():
    config = {
        "schema": "clinc-cil-v8-full-master-candidate", "status": "frozen", "seed_grid": list(summary.SEEDS), "arms": list(summary.ARMS),
        "tasks": list(summary.TASKS), "training_manifest_sha256": h("manifest"),
        "model_revision": "fixture-model@abc", "model_assets_sha256": {"model": h("asset")},
        "rank": 64, "capacity": 8, "levels": 5, "periods": [16, 64, 256, 1024, None],
        "batch_size": 8, "examples_per_task": 1500, "max_steps_per_task": 188,
        "task0_clean_accuracy_gate": 0.5,
        "initialization_seed": 1907008, "objective_version": "canonical_agreement_plus_thresholded_length_normalized_edit_v1",
        "inner_cycles_per_event": 10,
        "fixed_mix_fraction": 0.5, "gkd_calls_per_event": 20, "lti_calls_per_event": 10,
    }
    master_sha = h("master-bytes")
    source_hashes = {"runner": h("runner"), "lm": h("lm")}
    source_lock = summary.canonical_sha(source_hashes)
    cohorts = dev_cohorts(config)
    expected_rows = summary.validate_development_cohorts(cohorts, manifest_sha=config["training_manifest_sha256"])
    foundation_receipts = {}
    verified_foundations = []
    runs = []
    verified_run_files = []
    for seed_i, seed in enumerate(summary.SEEDS):
        fw = {k: 0 for k in summary.WORK_FIELDS}
        fw.update({"active_wall_seconds": 20.0 + seed_i, "full_backbone_forward_calls": 400,
                   "full_backbone_forward_tokens": 70000, "tail_calls": 400,
                   "tail_backward_calls": 188, "tail_backward_tokens": 4700,
                   "optimizer_or_lifecycle_update_steps": 188, "sleep_events": 0,
                   "executed_input_tokens_including_padding": 75000, "padded_input_tokens": 4000,
                   "sampled_answer_tokens_including_first_eos": 0})
        fcp, frec_sha = h(f"foundation-cp-{seed}"), h(f"foundation-rec-{seed}")
        fmetric = task_metrics(0, seed_i, False, expected_rows[seed][0])
        frec = {"schema": summary.FOUNDATION_SCHEMA, "status": "complete", "seed": seed,
                "master_config_sha256": master_sha, "training_manifest_sha256": config["training_manifest_sha256"],
                "source_lock_sha256": source_lock, "test_data_included": False,
                "model_revision": config["model_revision"], "model_assets_sha256": config["model_assets_sha256"],
                "acquisition_qualified": True, "tasks_completed": 1, "wake_ticks": 188,
                "pending_clocks": {"0": 12, "1": 60, "2": 188, "3": 188},
                "final_checkpoint_sha256": fcp, "work": fw, "task_metrics": [fmetric]}
        foundation_receipts[seed] = frec
        verified_foundations.append({"seed": seed, "receipt_sha256": frec_sha,
            "checkpoint_sha256": fcp, "work_counted_once": True, "work": fw})
        for arm in summary.ARMS:
            sleep = arm == "HOPE_SLEEP_FIXEDMIX"
            cp_sha = h(f"cp-{seed}-{arm}")
            metrics = [copy.deepcopy(fmetric)] + [task_metrics(t, seed_i, sleep, expected_rows[seed][t]) for t in range(1, 10)]
            events = 141 if sleep else 0
            adam = 4230 if sleep else 0
            cw = {k: 0 for k in summary.WORK_FIELDS}
            cw.update({"active_wall_seconds": 300.0 + seed_i,
                "full_backbone_forward_calls": 20000 + seed_i,
                "full_backbone_forward_tokens": 1500000 + seed_i,
                "tail_calls": 20000, "tail_backward_calls": 1692,
                "tail_backward_tokens": 44000, "optimizer_or_lifecycle_update_steps": 1692 + adam,
                "sleep_events": events, "executed_input_tokens_including_padding": 1800000,
                "padded_input_tokens": 30000, "sampled_answer_tokens_including_first_eos": adam * 2})
            run = {"schema": summary.RUN_SCHEMA, "status": "complete", "seed": seed, "arm": arm,
                "tasks_completed": 10, "acquisition_qualified": True, "test_data_included": False,
                "master_config_sha256": master_sha, "training_manifest_sha256": config["training_manifest_sha256"],
                "source_lock_sha256": source_lock, "final_checkpoint_sha256": cp_sha,
                "foundation_checkpoint_sha256": fcp, "foundation_receipt_sha256": frec_sha,
                "child_config_sha256": h(f"child-{seed}-{arm}"), "initial_mutable_sha256": h(f"init-{seed}"),
                "inner_cycles_per_event": 10, "objective_version": config["objective_version"],
                "task_metrics": metrics, "wake_ticks": 1880, "physical_wake_updates": 1880,
                "sleep_event_index": events, "physical_sleep_adamw_steps": adam,
                "work": cw, "active_wall_seconds": cw["active_wall_seconds"],
                "run_receipt_sha256": h(f"runreceipt-{seed}-{arm}"),
                "data_exposure": {"sleep_text_replay": True, "primary_train_rows": 13500,
                                  "old_intent_gold_targets_in_sleep_loss": False}}
            runs.append(run)
            verified_run_files.append({"seed": seed, "arm": arm,
                "checkpoint_sha256": cp_sha, "run_receipt_sha256": h(f"runreceipt-{seed}-{arm}")})
    seal = {"schema": summary.MAIN_SCHEMA, "complete": True, "test_data_included": False,
            "test_release_ready": True, "main_grid": [{"seed": s, "arm": a} for s in summary.SEEDS for a in summary.ARMS],
            "master_config_sha256": master_sha, "training_manifest_sha256": config["training_manifest_sha256"],
            "source_hashes": source_hashes, "source_lock_sha256": source_lock,
            "verified_foundations": verified_foundations,
            "shared_foundation_compute_counted_once": [{"seed": x["seed"], "work": x["work"],
                "foundation_checkpoint_sha256": x["checkpoint_sha256"]} for x in verified_foundations],
            "verified_run_files": verified_run_files,
            "runs": runs}
    return seal, config, master_sha, foundation_receipts, cohorts


class MainSixSummaryTests(unittest.TestCase):
    def test_development_sidecar_is_checked_against_rederived_pack_counts(self):
        _, config, _, _, cohorts = make_fixture()
        class_orders, clean_dev = derivation_fixture(cohorts)
        derived = summary.derive_development_cohort_rows(class_orders, clean_dev)
        parsed = summary.validate_development_cohorts_against_pack(
            cohorts, manifest_sha=config["training_manifest_sha256"], derived_rows=derived)
        self.assertEqual(parsed[17][9]["seen"], {"count": 2997, "class_count": 150})

        # Keep the sidecar internally consistent and the final total fixed, but
        # move one row between task cohorts. Only re-derivation catches this.
        altered = copy.deepcopy(cohorts)
        altered["seeds"]["17"]["8"]["current"]["count"] += 1
        altered["seeds"]["17"]["8"]["seen"]["count"] += 1
        altered["seeds"]["17"]["9"]["old"]["count"] += 1
        altered["seeds"]["17"]["9"]["current"]["count"] -= 1
        with self.assertRaisesRegex(ValueError, "differs from counts re-derived"):
            summary.validate_development_cohorts_against_pack(
                altered, manifest_sha=config["training_manifest_sha256"], derived_rows=derived)

    def test_exact_grid_trajectory_and_costs(self):
        seal, config, master_sha, foundations, cohorts = make_fixture()
        result = summary.build_dev_summary(seal, config, master_sha=master_sha,
            development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)
        self.assertEqual(result["status"], "complete_training_test_pending")
        self.assertEqual(result["test_payload_read"], False)
        self.assertEqual(len(result["per_seed_task_endpoints"]), 6)
        self.assertEqual(len(result["development_trajectory_stats"]), 10)
        self.assertEqual(result["costs"]["continuation_work_by_arm"]["HOPE"]["physical_wake_updates_continuation"], 5076)
        self.assertEqual(result["costs"]["continuation_work_by_arm"]["HOPE"]["sleep_events"], 0)
        self.assertEqual(result["costs"]["continuation_work_by_arm"]["HOPE_SLEEP_FIXEDMIX"]["sleep_events"], 423)
        self.assertEqual(result["costs"]["continuation_work_by_arm"]["HOPE_SLEEP_FIXEDMIX"]["physical_sleep_adamw_steps"], 12690)
        self.assertEqual([x["sleep_events"] for x in result["costs"]["continuation_work_by_arm"]["HOPE_SLEEP_FIXEDMIX"]["per_seed"]], [141, 141, 141])
        self.assertEqual([x["update_counter"] for x in result["costs"]["continuation_work_by_arm"]["HOPE_SLEEP_FIXEDMIX"]["per_seed"]], [5922, 5922, 5922])
        self.assertEqual(result["final_development_endpoints"]["HOPE"]["old"]["accuracy"]["n"], 3)
        self.assertIsNotNone(result["final_development_endpoints"]["HOPE"]["old"]["accuracy"]["sample_sd"])

    def test_rejects_partial_or_duplicate_run_grid(self):
        seal, config, master_sha, foundations, cohorts = make_fixture()
        seal["runs"].pop()
        with self.assertRaisesRegex(ValueError, "six terminal"):
            summary.build_dev_summary(seal, config, master_sha=master_sha, development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)
        seal, config, master_sha, foundations, cohorts = make_fixture()
        seal["runs"][-1] = copy.deepcopy(seal["runs"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summary.build_dev_summary(seal, config, master_sha=master_sha, development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)

    def test_rejects_shared_foundation_mismatch_and_sleep_accounting_drift(self):
        seal, config, master_sha, foundations, cohorts = make_fixture()
        seal["runs"][1]["foundation_checkpoint_sha256"] = h("different")
        with self.assertRaisesRegex(ValueError, "share the verified"):
            summary.build_dev_summary(seal, config, master_sha=master_sha, development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)
        seal, config, master_sha, foundations, cohorts = make_fixture()
        seal["runs"][1]["physical_sleep_adamw_steps"] += 1
        with self.assertRaisesRegex(ValueError, "per-run Sleep"):
            summary.build_dev_summary(seal, config, master_sha=master_sha, development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)

    def test_test_aggregate_contract_requires_all_six_global_unmasked_rows(self):
        seal, config, master_sha, _, _ = make_fixture()
        runs = []
        expected_checkpoints = {}
        for seed in summary.SEEDS:
            for arm in summary.ARMS:
                checkpoint_hash = h(f"checkpoint-{seed}-{arm}")
                receipt_hash = h(f"receipt-{seed}-{arm}")
                expected_checkpoints[(seed, arm)] = {"checkpoint_sha256": checkpoint_hash,
                                                     "run_receipt_sha256": receipt_hash}
                runs.append({"seed": seed, "arm": arm, "primary": {
                    "correct": 3000, "total": 4500,
                    "class_count": 150,
                    "accuracy": 3000 / 4500,
                    "invalid_predictions": 200,
                    "per_class": {str(i): {"total": 30, "correct": 20} for i in range(150)}},
                    "checkpoint_sha256": checkpoint_hash,
                    "run_receipt_sha256": receipt_hash})
        contract = {"schema": summary.CONTRACT_SCHEMA, "status": "frozen"}
        source_hashes = seal["source_hashes"]
        result = {"schema": summary.TEST_SCHEMA, "status": "complete", "test_data_read": True,
            "development_only": False, "primary_rows_used": 4500, "candidate_constrained": False,
            "task_mask_used": False, "checkpoint_selection_after_test": False,
            "optimizer_steps": 0, "backward_calls": 0, "evaluator_source_sha256": h("eval"),
            "evaluation_contract_sha256": h("contract"), "evaluation_contract": contract,
            "training_source_hashes": source_hashes,
            "release_gate": {"passed": True, "seal_sha256": h("seal"), "master_config_sha256": master_sha,
                "source_hashes": source_hashes, "evaluation_contract_sha256": h("contract")},
            "runs": runs}
        ok = summary.validate_test_result(result, result_sha=h("result"), seal_sha=h("seal"),
            master_sha=master_sha, source_hashes=source_hashes, evaluator_sha=h("eval"),
            contract=contract, contract_sha=h("contract"), expected_checkpoints=expected_checkpoints)
        self.assertEqual(ok["rows_per_checkpoint"], 4500)
        self.assertEqual(ok["paired_fixedmix_minus_hope"]["n"], 3)

    def test_evaluator_primary_total_schema_and_invalid_bounds(self):
        seal, _, master_sha, _, _ = make_fixture()
        runs = []
        expected_checkpoints = {}
        for seed in summary.SEEDS:
            for arm in summary.ARMS:
                checkpoint_hash = h(f"checkpoint-{seed}-{arm}")
                receipt_hash = h(f"receipt-{seed}-{arm}")
                expected_checkpoints[(seed, arm)] = {"checkpoint_sha256": checkpoint_hash,
                                                     "run_receipt_sha256": receipt_hash}
                runs.append({"seed": seed, "arm": arm, "primary": {
                    "correct": 3000, "total": 4500, "class_count": 150,
                    "accuracy": 3000 / 4500, "invalid_predictions": 200,
                    "per_class": {str(i): {"total": 30, "correct": 20} for i in range(150)}},
                    "checkpoint_sha256": checkpoint_hash,
                    "run_receipt_sha256": receipt_hash})
        contract = {"schema": summary.CONTRACT_SCHEMA, "status": "frozen"}
        source_hashes = seal["source_hashes"]
        result = {"schema": summary.TEST_SCHEMA, "status": "complete", "test_data_read": True,
            "development_only": False, "primary_rows_used": 4500, "candidate_constrained": False,
            "task_mask_used": False, "checkpoint_selection_after_test": False,
            "optimizer_steps": 0, "backward_calls": 0, "evaluator_source_sha256": h("eval"),
            "evaluation_contract_sha256": h("contract"), "evaluation_contract": contract,
            "training_source_hashes": source_hashes,
            "release_gate": {"passed": True, "seal_sha256": h("seal"), "master_config_sha256": master_sha,
                "source_hashes": source_hashes, "evaluation_contract_sha256": h("contract")},
            "runs": runs}

        def validate(candidate):
            return summary.validate_test_result(candidate, result_sha=h("result"), seal_sha=h("seal"),
                master_sha=master_sha, source_hashes=source_hashes, evaluator_sha=h("eval"),
                contract=contract, contract_sha=h("contract"), expected_checkpoints=expected_checkpoints)

        self.assertEqual(validate(result)["rows_per_checkpoint"], 4500)
        # The real evaluator calls this field `total`; no `count` alias is required.
        self.assertNotIn("count", result["runs"][0]["primary"])
        bad = copy.deepcopy(result)
        bad["runs"][0]["primary"]["invalid_predictions"] = 4501
        with self.assertRaisesRegex(ValueError, "invalid-prediction count"):
            validate(bad)
        bad = copy.deepcopy(result)
        bad["runs"][0]["primary"]["correct"] = 4301
        with self.assertRaisesRegex(ValueError, "no larger than non-invalid"):
            validate(bad)
        result["task_mask_used"] = True
        with self.assertRaisesRegex(ValueError, "task mask"):
            summary.validate_test_result(result, result_sha=h("result"), seal_sha=h("seal"),
                master_sha=master_sha, source_hashes=source_hashes, evaluator_sha=h("eval"),
                contract=contract, contract_sha=h("contract"), expected_checkpoints=expected_checkpoints)
        result["task_mask_used"] = False
        result["runs"][0]["checkpoint_sha256"] = h("wrong-checkpoint")
        with self.assertRaisesRegex(ValueError, "sealed final checkpoint"):
            summary.validate_test_result(result, result_sha=h("result"), seal_sha=h("seal"),
                master_sha=master_sha, source_hashes=source_hashes, evaluator_sha=h("eval"),
                contract=contract, contract_sha=h("contract"), expected_checkpoints=expected_checkpoints)
        result["runs"][0]["checkpoint_sha256"] = expected_checkpoints[(summary.SEEDS[0], summary.ARMS[0])]["checkpoint_sha256"]
        result["runs"][0]["primary"]["invalid_predictions"] = None
        with self.assertRaisesRegex(ValueError, "invalid-prediction count"):
            summary.validate_test_result(result, result_sha=h("result"), seal_sha=h("seal"),
                master_sha=master_sha, source_hashes=source_hashes, evaluator_sha=h("eval"),
                contract=contract, contract_sha=h("contract"), expected_checkpoints=expected_checkpoints)

    def test_markdown_and_plot_outputs_are_aggregate_only(self):
        seal, config, master_sha, foundations, cohorts = make_fixture()
        result = summary.build_dev_summary(seal, config, master_sha=master_sha,
            development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)
        markdown = summary.render_markdown(result)
        self.assertIn("not independent random initializations", markdown)
        self.assertIn("test_pending", markdown)
        base = Path(__file__).resolve().parents[1] / ".pytest-main-six-artifacts"
        base.mkdir(exist_ok=True)
        temp = Path(tempfile.mkdtemp(dir=base))
        try:
            names = summary.render_plots(result, temp)
            self.assertEqual(set(names), {"clinc-v8-main-six-development.png", "clinc-v8-main-six-development.svg"})
            self.assertTrue(all((temp / name).stat().st_size > 1000 for name in names))
        finally:
            shutil.rmtree(temp)

    def test_train_derived_denominators_invalid_counts_and_foundation_gate(self):
        seal, config, master_sha, foundations, cohorts = make_fixture()
        result = summary.build_dev_summary(seal, config, master_sha=master_sha,
            development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)
        self.assertEqual(result["protocol"]["clean_development_rows_per_seed"], 2997)

        # Per-seed clean-dev row counts are read from the frozen train-derived map.
        seal, config, master_sha, foundations, cohorts = make_fixture()
        cohorts["seeds"]["17"]["9"]["current"]["count"] -= 1
        with self.assertRaisesRegex(ValueError, "row counts do not partition"):
            summary.build_dev_summary(seal, config, master_sha=master_sha,
                development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)

        # Non-empty scopes require an actual integer invalid count and consistent partition.
        seal, config, master_sha, foundations, cohorts = make_fixture()
        seal["runs"][0]["task_metrics"][1]["result"]["metrics"]["current"]["invalid"] = None
        with self.assertRaisesRegex(ValueError, "invalid-generation count"):
            summary.build_dev_summary(seal, config, master_sha=master_sha,
                development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)
        seal, config, master_sha, foundations, cohorts = make_fixture()
        seal["runs"][0]["task_metrics"][1]["result"]["metrics"]["seen"]["invalid"] += 1
        with self.assertRaisesRegex(ValueError, "do not partition seen invalid"):
            summary.build_dev_summary(seal, config, master_sha=master_sha,
                development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)

        # A boolean receipt flag cannot substitute for the measured task-0 threshold.
        seal, config, master_sha, foundations, cohorts = make_fixture()
        item = foundations[17]["task_metrics"][0]
        item["result"]["metrics"]["current"]["accuracy"] = 0.49
        item["result"]["metrics"]["current"]["correct"] = 147
        for run in seal["runs"]:
            if run["seed"] == 17:
                run["task_metrics"][0] = copy.deepcopy(item)
        with self.assertRaisesRegex(ValueError, "below the configured acquisition gate"):
            summary.build_dev_summary(seal, config, master_sha=master_sha,
                development_cohorts=cohorts, development_cohorts_sha=h("cohorts"), foundations=foundations)


if __name__ == "__main__":
    unittest.main()
