"""Synthetic aggregate-only tests for Banking v10/v11 public reanalysis."""
from __future__ import annotations

import hashlib
import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools import reanalyze_banking_v10_v11 as re


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _allocate_correct(counts, correct_total):
    correct = [0] * len(counts)
    remaining = correct_total
    while remaining:
        progressed = False
        for i, count in enumerate(counts):
            if correct[i] < count and remaining:
                correct[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise AssertionError("correct total exceeds subgroup count")
    return correct


def _v10_candidate_metric(old_count, current_count, old_correct, current_correct):
    old_counts = [old_count // 66 + int(i < old_count % 66) for i in range(66)]
    current_counts = [current_count // 11 + int(i < current_count % 11) for i in range(11)]
    old_corrects = _allocate_correct(old_counts, old_correct)
    current_corrects = _allocate_correct(current_counts, current_correct)
    per_class = {}
    for cid, (count, correct) in enumerate(zip(old_counts + current_counts,
                                                old_corrects + current_corrects)):
        per_class[str(cid)] = {"count": count, "correct": correct,
                               "accuracy": correct / count if count else None}
    old = {"count": old_count, "correct": old_correct,
           "accuracy": old_correct / old_count if old_count else None}
    current = {"count": current_count, "correct": current_correct,
               "accuracy": current_correct / current_count if current_count else None}
    total_count, total_correct = old_count + current_count, old_correct + current_correct
    total = {"count": total_count, "correct": total_correct,
             "accuracy": total_correct / total_count}
    task_groups = {"old": old, "current": current, "seen": total,
                   "unseen": {"count": 0, "correct": 0, "accuracy": None}}
    class_accuracies = [item["accuracy"] for item in per_class.values() if item["count"]]
    return {**total, "per_class": per_class,
            "class_macro_accuracy": sum(class_accuracies) / len(class_accuracies),
            "task_breakdown": {"6": {"teacher_forced_sum": task_groups,
                                       "teacher_forced_length_mean": task_groups}}}


def _v10_secondary_metric_with_zero_old_class():
    old_counts = [0] + [40] * 65
    current_counts = [37] * 4 + [36] * 7
    old_corrects = _allocate_correct(old_counts, 1300)
    current_corrects = _allocate_correct(current_counts, 200)
    per_class = {}
    for cid, (count, correct) in enumerate(zip(old_counts + current_counts,
                                                old_corrects + current_corrects)):
        per_class[str(cid)] = {"count": count, "correct": correct,
                               "accuracy": correct / count if count else None}
    old = {"count": 2600, "correct": 1300, "accuracy": 0.5}
    current = {"count": 400, "correct": 200, "accuracy": 0.5}
    seen = {"count": 3000, "correct": 1500, "accuracy": 0.5}
    task_groups = {"old": old, "current": current, "seen": seen,
                   "unseen": {"count": 0, "correct": 0, "accuracy": None}}
    macro = sum(x["accuracy"] for x in per_class.values() if x["count"]) / 76
    return {**seen, "per_class": per_class, "class_macro_accuracy": macro,
            "task_breakdown": {"6": {"teacher_forced_sum": task_groups,
                                       "teacher_forced_length_mean": task_groups}}}


def _v10_context(sum_metric, mean_metric, greedy, nll):
    """Match frozen evaluate_rows: task_breakdown is a context-level sibling."""
    task6 = {}
    for name, metric in (("teacher_forced_sum", sum_metric),
                         ("teacher_forced_length_mean", mean_metric)):
        nested = metric.pop("task_breakdown")
        task6[name] = nested["6"][name]
    return {"teacher_forced_sum": sum_metric,
            "teacher_forced_length_mean": mean_metric,
            "greedy_generation": greedy,
            "gold_label_token_mean_nll": nll,
            "task_breakdown": {"6": task6}}


def _install_v10_task6_metric(context, name, metric):
    nested = metric.pop("task_breakdown")
    context[name] = metric
    context.setdefault("task_breakdown", {}).setdefault("6", {})[name] = nested["6"][name]


def _synthetic_inputs():
    runs = []
    verified = []
    primary = {}
    secondary = {}
    for seed in re.SEEDS:
        for i, arm in enumerate(re.V10_ARMS):
            cp, receipt = _sha(f"cp-{seed}-{arm}"), _sha(f"receipt-{seed}-{arm}")
            runs.append({"seed": seed, "arm": arm, "status": "complete", "tasks_completed": 7,
                         "config_sha256": _sha("config"), "source_lock_sha256": _sha("source"),
                         "training_manifest_sha256": _sha("manifest"),
                         "final_checkpoint_sha256": cp, "run_receipt_sha256": receipt,
                         "endpoint_metrics": {"opaque": 1}, "work": {"backbone_forwards": 10},
                         "run_dir": "do-not-follow/" + str(seed) + arm,
                         "final_checkpoint": "also-do-not-follow.pt"})
            verified.append({"seed": seed, "arm": arm, "checkpoint_sha256": cp,
                             "receipt_sha256": receipt})
            name = f"seed{seed}:{arm}"
            # Gradually changing accuracies make mean, sample SD, and paired contrasts testable.
            accuracy = 0.30 + 0.05 * i + 0.02 * re.SEEDS.index(seed)
            raw_old_correct = round(2640 * accuracy)
            raw_current_correct = round(440 * accuracy)
            secondary_old_correct = round(2640 * 0.5)
            secondary_current_correct = round(360 * 0.5)
            raw_sum = _v10_candidate_metric(2640, 440,
                raw_old_correct, raw_current_correct)
            raw_mean = _v10_candidate_metric(2640, 440, 1320, 220)
            secondary_sum = _v10_candidate_metric(2640, 360,
                secondary_old_correct, secondary_current_correct)
            secondary_mean = _v10_candidate_metric(2640, 360, 1320, 180)
            primary[name] = _v10_context(
                raw_sum, raw_mean,
                {"count": 3080, "correct": 924, "accuracy": 0.3},
                {"count": 3080, "mean": 1.2})
            secondary[name] = _v10_context(
                secondary_sum, secondary_mean,
                {"count": 3000, "correct": 900, "accuracy": 0.3},
                {"count": 3000, "mean": 1.3})
    seal = {"schema": "banking-cil-v10-training-seal-v1", "complete": True,
            "seeds": list(re.SEEDS), "arms": list(re.V10_ARMS), "runs": runs,
            "config_sha256": _sha("config"), "source_lock_sha256": _sha("source"),
            "training_manifest_sha256": _sha("manifest"), "test_data_included": False,
            "official_test_read": False}
    evaluation = {"schema": re.V10_SCHEMA, "status": "complete",
                  "optimizer_updates": 0, "all_contexts_frozen": True,
                  "backbone_unchanged": True,
                  "training": {"config_sha256": _sha("config"),
                               "source_lock_sha256": _sha("source"),
                               "training_manifest_sha256": _sha("manifest"),
                               "seal_sha256": "SET_AFTER_SERIALIZE",
                               "verified_contexts": verified},
                  "test_source": {"source_revision": re.EXPECTED_TEST_REVISION,
                                  "source_sha256": re.EXPECTED_TEST_SHA256,
                                  "source_bytes": re.EXPECTED_TEST_BYTES,
                                  "raw_rows": re.EXPECTED_TEST_ROWS, "classes": 77,
                                  "rows_per_class": 40, "primary_includes_all_official_rows": True,
                                  "train_development_disjoint_secondary_rows": 3000},
                  "primary_raw_official_test": primary,
                  "secondary_train_development_disjoint_subset": secondary,
                  "secondary_metrics_reused_raw_primary": False,
                  "work": {"optimizer_updates": 0, "backbone_forwards": 123}}
    v11 = {"schema": re.V11_SCHEMA, "status": "complete", "optimizer_updates": 0,
           "contexts_unchanged": True, "frozen_backbone_unchanged": True,
           "protocol": {"official_test_accessed": False, "seeds": list(re.SEEDS),
                        "arms": list(re.V11_ARMS), "contexts_per_seed": 5,
                        "rows_per_seed": 220,
                        "candidate_metric_primary": "global_77_sum_log_likelihood"},
           "frozen_inputs": {"v10_grid_seal_sha256": "SET_AFTER_SERIALIZE",
                             "v10_config_sha256": _sha("config"),
                             "training_manifest_sha256": _sha("manifest")},
           "development": {}, "work": {"optimizer_updates": 0}}
    for seed in re.SEEDS:
        candidates, greedy = {}, {}
        for i, arm in enumerate(re.V11_ARMS):
            candidates[arm] = {}
            for metric in ("teacher_forced_sum", "teacher_forced_length_mean"):
                candidates[arm][metric] = {}
                for group, n in (("all77", 220), ("old", 110), ("current", 110)):
                    correct = min(n, 40 + i * 5 + re.SEEDS.index(seed))
                    candidates[arm][metric][group] = {"count": n, "correct": correct,
                                                     "accuracy": correct / n}
            greedy[arm] = {"count": 220, "correct": 50 + i, "accuracy": (50 + i) / 220,
                           "invalid_predictions": 0, "candidate_constrained": False}
        v11["development"][str(seed)] = {"cohort_rows": 220, "context_count": 5,
            "candidate_metric_primary": "global_77_sum_log_likelihood",
            "candidate_metrics": candidates, "greedy_generation_metrics": greedy,
            "work": {"optimizer_updates": 0}}
    return seal, evaluation, v11


class ReanalysisTests(unittest.TestCase):
    def _write_inputs(self, root: Path):
        seal, evaluation, v11 = _synthetic_inputs()
        seal_path = root / "training-seal.json"
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
        seal_sha = hashlib.sha256(seal_path.read_bytes()).hexdigest()
        evaluation["training"]["seal_sha256"] = seal_sha
        v11["frozen_inputs"]["v10_grid_seal_sha256"] = seal_sha
        test_path = root / "evaluation.json"
        v11_path = root / "development-evaluation.json"
        test_path.write_text(json.dumps(evaluation), encoding="utf-8")
        v11_path.write_text(json.dumps(v11), encoding="utf-8")
        return seal_path, test_path, v11_path

    def test_sample_sd_paired_contrasts_and_population_separation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            seal, test, v11 = self._write_inputs(Path(td))
            test_size = test.stat().st_size
            result = re.analyze(seal, test, v11)
        self.assertEqual(result["payloads_downloaded"], False)
        self.assertEqual(result["inputs"]["v10_test"]["bytes"], test_size)
        v10 = result["summaries"]["v10"]["populations"]
        raw = v10["official_test_raw"]["teacher_forced_sum/all"]["arm_mean_sample_sd"]
        self.assertEqual(raw["CMS_ONLY"]["n"], 3)
        self.assertAlmostEqual(raw["CMS_ONLY"]["sample_sd"], 0.02, places=3)
        contrast = v10["official_test_raw"]["teacher_forced_sum/all"]["paired_differences"][
            "SLEEP_ADDITIVE-minus-CMS_ONLY"]
        self.assertAlmostEqual(contrast["mean"], 0.05)
        self.assertAlmostEqual(contrast["sample_sd"], 0.0)
        self.assertEqual(len(result["paired_rows"]), 207)
        self.assertIn("development_task01", result["summaries"]["v11"]["populations"])
        self.assertNotIn("development_task01", result["summaries"]["v10"]["populations"])

    def test_wrong_denominator_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            seal, test, v11 = self._write_inputs(Path(td))
            obj = json.loads(test.read_text(encoding="utf-8"))
            obj["primary_raw_official_test"]["seed17:CMS_ONLY"]["teacher_forced_sum"]["count"] = 3079
            test.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 3080"):
                re.analyze(seal, test, v11)

    def test_final_task6_subgroups_are_emitted_with_distinct_cross_sectional_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            seal, test, v11 = self._write_inputs(Path(td))
            result = re.analyze(seal, test, v11)
            csv_path = Path(td) / "summary.csv"
            re.write_outputs(result, Path(td) / "summary.json", csv_path)
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
        rows = [r for r in result["rows"] if r["version"] == "v10" and
                r["population"] == "official_test_raw" and r["seed"] == 17 and
                r["arm"] == "CMS_ONLY" and r["metric"] == "teacher_forced_sum"]
        by_subgroup = {r["subgroup"]: r for r in rows}
        self.assertEqual(by_subgroup["all"]["n"], 3080)
        self.assertEqual(by_subgroup["final_old66"]["n"], 2640)
        self.assertEqual(by_subgroup["final_current11"]["n"], 440)
        self.assertEqual(by_subgroup["final_old66"]["correct"] +
                         by_subgroup["final_current11"]["correct"],
                         by_subgroup["all"]["correct"])
        self.assertEqual(by_subgroup["final_current11"]["partition_source"],
                         "frozen_evaluator.task_breakdown.6")
        self.assertIn("not before/after forgetting", by_subgroup["final_old66"]["interpretation"])
        secondary = [r for r in result["rows"] if r["version"] == "v10" and
                     r["population"] == "official_test_train_development_disjoint" and
                     r["seed"] == 17 and r["arm"] == "CMS_ONLY" and
                     r["metric"] == "teacher_forced_sum"]
        secondary_by_group = {r["subgroup"]: r for r in secondary}
        self.assertEqual(secondary_by_group["final_old66"]["n"] +
                         secondary_by_group["final_current11"]["n"],
                         secondary_by_group["all"]["n"])
        self.assertEqual(secondary_by_group["final_current11"]["n"], 360)
        self.assertEqual(len(result["paired_rows"]), 207)
        self.assertTrue(any(r["subgroup"] == "all" and r["population"] == "official_test_raw"
                            for r in csv_rows))
        csv_final = next(r for r in csv_rows if r["version"] == "v10" and
                         r["population"] == "official_test_raw" and r["seed"] == "17" and
                         r["arm"] == "CMS_ONLY" and r["metric"] == "teacher_forced_sum" and
                         r["subgroup"] == "final_old66")
        self.assertIn("not before/after forgetting", csv_final["interpretation"])

    def test_final_task6_group_correct_must_reconcile_to_all(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            seal, test, v11 = self._write_inputs(Path(td))
            obj = json.loads(test.read_text(encoding="utf-8"))
            context = obj["primary_raw_official_test"]["seed17:CMS_ONLY"]
            old = context["task_breakdown"]["6"]["teacher_forced_sum"]["old"]
            old["correct"] += 1
            old["accuracy"] = old["correct"] / old["count"]
            test.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "old/current counts or correct do not sum to all"):
                re.analyze(seal, test, v11)

    def test_final_task6_group_count_must_reconcile_to_all(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            seal, test, v11 = self._write_inputs(Path(td))
            obj = json.loads(test.read_text(encoding="utf-8"))
            context = obj["primary_raw_official_test"]["seed17:CMS_ONLY"]
            old = context["task_breakdown"]["6"]["teacher_forced_sum"]["old"]
            old["count"] -= 1
            old["accuracy"] = old["correct"] / old["count"]
            test.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "old/current counts or correct do not sum to all"):
                re.analyze(seal, test, v11)

    def test_secondary_zero_count_class_is_valid_when_all_partitions_reconcile(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            seal, test, v11 = self._write_inputs(Path(td))
            obj = json.loads(test.read_text(encoding="utf-8"))
            metrics = obj["secondary_train_development_disjoint_subset"]["seed17:CMS_ONLY"]
            _install_v10_task6_metric(
                metrics, "teacher_forced_sum", _v10_secondary_metric_with_zero_old_class())
            _install_v10_task6_metric(
                metrics, "teacher_forced_length_mean", _v10_secondary_metric_with_zero_old_class())
            test.write_text(json.dumps(obj), encoding="utf-8")
            result = re.analyze(seal, test, v11)
        rows = [r for r in result["rows"] if r["version"] == "v10" and
                r["population"] == "official_test_train_development_disjoint" and
                r["seed"] == 17 and r["arm"] == "CMS_ONLY" and
                r["metric"] == "teacher_forced_sum"]
        groups = {r["subgroup"]: r for r in rows}
        self.assertEqual(groups["all"]["n"], 3000)
        self.assertEqual(groups["final_old66"]["n"], 2600)
        self.assertEqual(groups["final_current11"]["n"], 400)
        self.assertEqual(groups["final_old66"]["correct"] + groups["final_current11"]["correct"],
                         groups["all"]["correct"])

    def test_per_class_correct_must_reconcile_to_all_and_missing_schema_is_rejected(self):
        for edit in ("perclass", "missing_task6", "wrongly_nested_task6"):
            with self.subTest(edit=edit), tempfile.TemporaryDirectory(
                    dir=Path(__file__).resolve().parents[1]) as td:
                seal, test, v11 = self._write_inputs(Path(td))
                obj = json.loads(test.read_text(encoding="utf-8"))
                context = obj["primary_raw_official_test"]["seed17:CMS_ONLY"]
                metric = context["teacher_forced_sum"]
                if edit == "perclass":
                    item = metric["per_class"]["0"]
                    item["correct"] += 1
                    item["accuracy"] = item["correct"] / item["count"]
                    seen = [v["accuracy"] for v in metric["per_class"].values()]
                    metric["class_macro_accuracy"] = sum(seen) / len(seen)
                    expected = "per-class correct totals do not sum to all-cohort correct"
                elif edit == "missing_task6":
                    del context["task_breakdown"]
                    expected = "missing task_breakdown"
                else:
                    metric["task_breakdown"] = context.pop("task_breakdown")
                    expected = "missing task_breakdown"
                test.write_text(json.dumps(obj), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    re.analyze(seal, test, v11)

    def test_missing_seed_arm_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            seal, test, v11 = self._write_inputs(Path(td))
            obj = json.loads(test.read_text(encoding="utf-8"))
            del obj["primary_raw_official_test"]["seed43:GOLD_ADDITIVE"]
            test.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "context grid mismatch"):
                re.analyze(seal, test, v11)

    def test_v11_old_current_denominators_are_enforced(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            seal, test, v11 = self._write_inputs(Path(td))
            obj = json.loads(v11.read_text(encoding="utf-8"))
            obj["development"]["17"]["candidate_metrics"]["LTI_NO_REWARD"][
                "teacher_forced_sum"]["old"]["count"] = 109
            v11.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 110"):
                re.analyze(seal, test, v11)

    def test_v11_config_and_manifest_must_match_supplied_seal(self):
        for field in ("v10_config_sha256", "training_manifest_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory(
                    dir=Path(__file__).resolve().parents[1]) as td:
                seal, test, v11 = self._write_inputs(Path(td))
                obj = json.loads(v11.read_text(encoding="utf-8"))
                obj["frozen_inputs"][field] = _sha("wrong-" + field)
                v11.write_text(json.dumps(obj), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "differs from supplied training seal"):
                    re.analyze(seal, test, v11)

    def test_nonzero_v11_per_seed_updates_rejected_when_field_is_present(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            seal, test, v11 = self._write_inputs(Path(td))
            obj = json.loads(v11.read_text(encoding="utf-8"))
            obj["development"]["29"]["work"]["optimizer_updates"] = 1
            v11.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed 29 work ledger reports optimizer updates"):
                re.analyze(seal, test, v11)


if __name__ == "__main__":
    unittest.main()
