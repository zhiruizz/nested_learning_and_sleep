import unittest

from tools import reanalyze_clinc_v8_main as analyzer


def _stat(values):
    import statistics
    return {"n": 3, "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values), "per_seed": list(values)}


def _test_result(correct_by_pair, *, raw_sha, public_sha):
    rows = []
    for seed in analyzer.SEEDS:
        for arm in analyzer.ARMS:
            correct = correct_by_pair[(seed, arm)]
            q, rem = divmod(correct, 150)
            per_class = {str(i): {"total": 30, "correct": q + int(i < rem)}
                         for i in range(150)}
            rows.append({"seed": seed, "arm": arm,
                "primary": {"total": 4500, "correct": correct,
                    "accuracy": correct / 4500, "per_class": per_class}})
    return {"schema": "clinc-v8-final-test-evaluation-v1", "status": "complete",
        "test_data_read": True, "development_only": False,
        "primary_rows_used": 4500, "candidate_constrained": False,
        "task_mask_used": False, "checkpoint_selection_after_test": False,
        "optimizer_steps": 0, "backward_calls": 0, "runs": rows,
        "_public_capture": {"schema": "clinc-v8-public-development-capture-reference-v1",
            "source_artifact_sha256_raw": raw_sha, "raw_input_sha256": raw_sha,
            "public_artifact_sha256": public_sha}}


def _summary():
    values = {arm: {} for arm in analyzer.ARMS}
    for arm_index, arm in enumerate(analyzer.ARMS):
        for scope_index, scope in enumerate(analyzer.SCOPES):
            rows = []
            for seed_index, seed in enumerate(analyzer.SEEDS):
                count = 10 + seed_index
                correct = 1 + seed_index + arm_index + scope_index
                rows.append({"seed": seed, "correct": correct, "count": count,
                    "accuracy": correct / count, "macro_accuracy": correct / count})
            rates = [row["accuracy"] for row in rows]
            values[arm][scope] = {"per_seed": rows, "accuracy": _stat(rates)}
    paired = {}
    trajectory_pair = {}
    for scope in analyzer.SCOPES:
        deltas = [values[analyzer.ARMS[1]][scope]["per_seed"][i]["accuracy"] -
                  values[analyzer.ARMS[0]][scope]["per_seed"][i]["accuracy"]
                  for i in range(3)]
        paired[scope] = _stat(deltas)
        trajectory_pair[scope] = _stat(deltas)
    hope = {seed: 1000 + i * 100 for i, seed in enumerate(analyzer.SEEDS)}
    mix = {seed: 1500 + i * 100 for i, seed in enumerate(analyzer.SEEDS)}
    raw_sha, public_sha = "c" * 64, "b" * 64
    test = _test_result({(s, "HOPE"): hope[s] for s in analyzer.SEEDS} |
                        {(s, "HOPE_SLEEP_FIXEDMIX"): mix[s] for s in analyzer.SEEDS},
                        raw_sha=raw_sha, public_sha=public_sha)
    test_rates = {arm: [((hope if arm == "HOPE" else mix)[s]) / 4500 for s in analyzer.SEEDS]
                  for arm in analyzer.ARMS}
    test_by_arm = {arm: {"accuracy": _stat(test_rates[arm])} for arm in analyzer.ARMS}
    test_deltas = [test_rates[analyzer.ARMS[1]][i] - test_rates[analyzer.ARMS[0]][i]
                   for i in range(3)]
    test_summary = {"status": "complete", "rows_per_checkpoint": 4500,
        "by_arm": test_by_arm, "paired_fixedmix_minus_hope": _stat(test_deltas),
        "per_seed": [{"seed": s, "hope_accuracy": hope[s] / 4500,
            "fixedmix_accuracy": mix[s] / 4500,
            "fixedmix_minus_hope": (mix[s] - hope[s]) / 4500} for s in analyzer.SEEDS],
        "input_artifact_sha256": raw_sha}
    return {"schema": "clinc-v8-main-six-aggregate-v1", "status": "complete_training_and_test",
        "test_status": "complete_all4500_global_unmasked", "test_payload_read": False,
        "claim_scope": "synthetic paired example", "_public_capture": {
            "schema": "clinc-v8-public-development-capture-reference-v1",
            "source_artifact_sha256_raw": "a" * 64},
        "final_development_endpoints": values,
        "final_development_paired_differences_fixedmix_minus_hope": paired,
        "development_trajectory_stats": [{"task_id": 9, "by_arm": values}],
        "paired_descriptive_differences_fixedmix_minus_hope": [
            {"task_id": 9, **trajectory_pair}],
        "final_test": test_summary}, test, public_sha


class TestReanalyzeClincMainCpu(unittest.TestCase):
    def test_recomputes_dev_and_test_global_stats_from_counts(self):
        summary, test, test_sha = _summary()
        output = analyzer.reanalyze(summary, test_result=test,
            test_result_sha=test_sha, summary_sha="d" * 64)
        self.assertEqual(output["status"], "complete")
        self.assertEqual(output["development_final_task9"]["by_scope"]["old"]["HOPE"]["accuracy"]["n"], 3)
        self.assertAlmostEqual(output["final_test_global"]["by_arm"]["HOPE"]["accuracy"]["mean"],
            (1000 + 1100 + 1200) / 3 / 4500)
        self.assertEqual(output["final_test_global"]["by_arm"]["HOPE_SLEEP_FIXEDMIX"]["per_seed_counts"][2]["correct"], 1700)

    def test_rejects_stored_final_dev_mean_mismatch(self):
        summary, test, test_sha = _summary()
        summary["final_development_endpoints"]["HOPE"]["old"]["accuracy"]["mean"] += 0.01
        with self.assertRaisesRegex(ValueError, "stored aggregate mismatch"):
            analyzer.reanalyze(summary, test_result=test, test_result_sha=test_sha)

    def test_rejects_duplicate_or_incomplete_test_grid(self):
        summary, test, test_sha = _summary()
        test["runs"][0] = dict(test["runs"][1])
        with self.assertRaisesRegex(ValueError, "duplicate or unexpected"):
            analyzer.reanalyze(summary, test_result=test, test_result_sha=test_sha)

    def test_rejects_capture_sha_mismatch_and_wrong_total_field(self):
        summary, test, test_sha = _summary()
        test["_public_capture"]["source_artifact_sha256_raw"] = "e" * 64
        test["_public_capture"]["raw_input_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "raw result SHA differs"):
            analyzer.reanalyze(summary, test_result=test, test_result_sha=test_sha)
        summary, test, test_sha = _summary()
        test["runs"][0]["primary"]["total"] = 4499
        with self.assertRaisesRegex(ValueError, "total=4500"):
            analyzer.reanalyze(summary, test_result=test, test_result_sha=test_sha)


if __name__ == "__main__":
    unittest.main()
