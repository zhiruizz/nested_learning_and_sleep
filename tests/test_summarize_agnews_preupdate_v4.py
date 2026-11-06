"""Synthetic regression tests for the offline v4 aggregate analyzer."""
import importlib.util
import pathlib
import tempfile
import unittest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "summarize_agnews_preupdate_v4.py"
SPEC = importlib.util.spec_from_file_location("summarize_agnews_preupdate_v4", MODULE_PATH)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def metrics(value):
    return {g: {A.EM: value, A.NLL: 1.0, "candidate_accuracy": value,
                "marker_accuracy": 1.0, "examples": 256} for g in A.GROUPS}


def valid_runs():
    runs = []
    for seed in A.SEEDS:
        for arm in A.ARMS:
            runs.append({
                "seed": seed, "arm": arm, "status": "complete", "primary_updates": 112,
                "extra_replay_updates": 336 if arm == "warm_extra_gold_replay_P4" else 0,
                "sleep_Adam_updates": 672 if arm in ("warm_pre_sleep_P4", "warm_post_sleep_P4") else 24 if arm == "warm_pre_sleep_P64" else 0,
                "validation": [{"primary_step": step, "metrics": {g: metrics(.5)[g] for g in A.GROUPS}} for step in A.STEPS],
                "checkpoints": [{"primary_step": step} for step in A.STEPS], "work": {},
            })
    return runs


def warm_fixture(early_teacher_drop=15.0):
    teacher = metrics(.95)
    stage_records = []
    for seed in A.SEEDS:
        va_value = .95 - early_teacher_drop / 100
        validation = metrics(va_value)
        train = metrics(.90)
        pass_record = {"budget": 24, "seed": seed, "qualified": False,
            "train": train, "validation": validation,
            "receiver_disabled_validation": metrics(.5),
            "teacher_max_drop_old_and_each_class_pp": early_teacher_drop}
        stage_records.append(pass_record)
    final_records = []
    qualified = []
    for seed in A.SEEDS:
        final_records.append({"budget": 1536, "seed": seed, "qualified": True,
            "train": metrics(.90), "validation": metrics(.90),
            "receiver_disabled_validation": metrics(.5),
            "teacher_max_drop_old_and_each_class_pp": 5.0})
        qualified.append({"seed": seed, "warm_updates": 1536, "train": {"old":metrics(.9)["old"],"class0":metrics(.9)["class0"],"class1":metrics(.9)["class1"]},
            "validation": {"old":metrics(.9)["old"],"class0":metrics(.9)["class0"],"class1":metrics(.9)["class1"]}})
    return {"all3_qualified": True, "phaseB_permitted": True, "confirmation_scored": False,
        "chosen_common_budget": 1536,
        "stages": [{"budget":24,"all_qualified":False,"records":stage_records},
                   {"budget":1536,"all_qualified":True,"records":final_records}],
        "teachers": {str(s): {"train":metrics(.95),"validation":metrics(.95)} for s in A.SEEDS},
        "initial_controls": {str(s): {"reset_without_KD_train":metrics(.2),
            "reset_without_KD_validation":metrics(.3),
            "teacher":{"train":metrics(.95),"validation":metrics(.95)}} for s in A.SEEDS},
        "qualified_models": qualified}


class AnalyzerRegressionTests(unittest.TestCase):
    def test_aggregate_rejects_wrong_total_and_accepts_2016_primary(self):
        seal = {"complete":True,"confirmation_scored":False,"no_holdout_input":True,"runs":valid_runs()}
        parsed = A.aggregate_runs(seal)
        self.assertEqual(len(parsed),18)
        seal["runs"][0]["primary_updates"] = 111
        with self.assertRaises(AssertionError):
            A.aggregate_runs(seal)

    def test_teacher_relative_drop_can_fail_when_absolute_gates_pass(self):
        out = A.warm_analysis(warm_fixture())
        seed17 = out["stages"][0]["seeds"][0]
        self.assertTrue(seed17["gates"]["old_train_ge_80pct"])
        self.assertTrue(seed17["gates"]["old_validation_ge_75pct"])
        self.assertFalse(seed17["gates"]["teacher_relative_drop_old_le_5pp"])
        self.assertAlmostEqual(seed17["teacher_relative_drops_pp"]["old"],15.0)

    def test_initial_controls_preserve_teacher_train_and_validation(self):
        out = A.warm_analysis(warm_fixture())
        control = out["initial_controls_em"]["17"]
        self.assertEqual(control["teacher"]["train"]["old"],.95)
        self.assertEqual(control["teacher"]["validation"]["class0"],.95)
        self.assertEqual(control["reset_without_KD_train"]["old"],.2)

    def test_method_path_warm_tax_zero_for_foundation_and_perseed_for_warmarms(self):
        runs=valid_runs(); by_key={(r["seed"],r["arm"]):r for r in runs}
        warm={"work":{},"successful_or_attempted_main_Adam_updates":4608,"extra_ghost_Adam_updates":2}
        seal={"runs":runs,"work":{}}
        ledger=A.work_ledger(warm,seal,by_key)
        paths=ledger["method_path_per_seed"]
        self.assertEqual(paths["foundation_CMS_reference"]["17"]["warm_tax_Adam_updates"],0)
        self.assertEqual(paths["warm_CMS_no_sleep"]["17"]["warm_tax_Adam_updates"],1536)
        self.assertEqual(ledger["method_path_all3_seeds"]["warm_CMS_no_sleep"]["warm_tax_Adam_updates"],4608)
        self.assertEqual(ledger["phaseB"]["primary_SGD_updates"],2016)
        self.assertEqual(ledger["phaseB"]["extra_gold_replay_SGD_updates"],1008)
        self.assertEqual(ledger["phaseB"]["total_SGD_updates"],3024)
        self.assertEqual(ledger["method_path_all3_seeds"]["foundation_CMS_reference"]["warm_tax_Adam_updates"],0)

    def test_work_ledger_rejects_wrong_total_update_count(self):
        runs=valid_runs(); by_key={(r["seed"],r["arm"]):r for r in runs}
        runs[0]["extra_replay_updates"] = 335
        with self.assertRaises(AssertionError):
            A.work_ledger({"work":{},"successful_or_attempted_main_Adam_updates":4608,"extra_ghost_Adam_updates":2},
                          {"runs":runs,"work":{}},by_key)

    def test_work_counter_residue_is_rejected(self):
        runs=valid_runs(); by_key={(r["seed"],r["arm"]):r for r in runs}
        warm={"work":{},"successful_or_attempted_main_Adam_updates":4608,"extra_ghost_Adam_updates":2}
        seal={"runs":runs,"work":{"orphan":{"forward_calls":1,"forward_input_tokens":2,"gradient_forward_input_tokens":0}}}
        with self.assertRaises(AssertionError):
            A.work_ledger(warm,seal,by_key)

    def test_n3_mean_and_sample_sd(self):
        self.assertEqual(A.mean_sd([1.0,2.0,3.0]),{"n":3,"mean":2.0,"sample_sd":1.0})

    def test_output_text_uses_lf_newlines(self):
        with tempfile.TemporaryDirectory(dir=MODULE_PATH.parents[1]) as d:
            p=pathlib.Path(d)/"x.txt"
            A.write_text_lf(p,"a\nb\n")
            self.assertEqual(p.read_bytes(),b"a\nb\n")


if __name__ == "__main__":
    unittest.main()
