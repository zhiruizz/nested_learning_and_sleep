"""CPU-only protocol and deterministic auxiliary-schedule checks for v6."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_agnews_fair_continuation_v6 as driver
import qualify_agnews_fair_resume_v6 as qualifier
import agnews_fair_continuation_v6 as helper


class FairContinuationV6Tests(unittest.TestCase):
    def test_bounded_continuation_budget_and_eval_points(self):
        self.assertEqual(driver.END_STEP - driver.START_STEP, 576)
        self.assertEqual(driver.EVAL_STEPS, (448, 512, 768, 1024))
        self.assertEqual(2 * 3 * (driver.END_STEP - driver.START_STEP), 3456)
        self.assertEqual(3456 * 2, 6912)  # student forward and backward passes
        self.assertEqual(3456 * 4, 13824)  # auxiliary target examples/prefix draws
        self.assertEqual(3 * 576 * 4, 6912)  # each auxiliary class-source arm
        qualifier_updates = len(qualifier.ARMS) * (2 + 2 * 2)
        self.assertEqual(qualifier_updates, 12)
        self.assertEqual(qualifier_updates * 2, 24)  # new and auxiliary student passes
        self.assertEqual(qualifier_updates * 4, 48)  # targets per stream
        self.assertEqual(qualifier.ARMS, (("aux_new_CE", (2, 3)), ("aux_old_gold_CE", (0, 1))))

    def test_aux_schedule_is_fixed_balanced_train_only_and_disjoint_by_source_role(self):
        records = []
        for cls in range(4):
            for j in range(10):
                records.append({"role": "train", "class_id": cls, "record_id": f"train-{cls}-{j}"})
            records.append({"role": "validation", "class_id": cls, "record_id": f"val-{cls}"})
        pack = {"records": records}
        rows = records
        old_a = driver.aux_schedule(pack, 29, rows, (0, 1))
        old_b = driver.aux_schedule(pack, 29, rows, (0, 1))
        new = driver.aux_schedule(pack, 29, rows, (2, 3))
        self.assertEqual(old_a, old_b)
        self.assertNotEqual(driver.schedule_hash(old_a), driver.schedule_hash(new))
        self.assertEqual(list(old_a), list(range(449, 1025)))
        self.assertEqual(len(old_a), 576)
        for schedule, expected in ((old_a, (0, 1)), (new, (2, 3))):
            for indices in schedule.values():
                selected = [records[i] for i in indices]
                self.assertEqual(len(selected), 4)
                self.assertTrue(all(row["role"] == "train" for row in selected))
                self.assertEqual([sum(row["class_id"] == cls for row in selected) for cls in expected], [2, 2])

    def test_helper_declares_truthful_aux_class_contract(self):
        import inspect
        self.assertEqual(inspect.signature(helper.anchored_step).parameters["aux_classes"].default, (0, 1))
        self.assertEqual(driver.ARMS, (("aux_new_CE", (2, 3)), ("aux_old_gold_CE", (0, 1))))

    def test_qualification_and_main_bridge_on_full_aux_schedule(self):
        records=[]
        for cls in range(4):
            for j in range(10): records.append({"role":"train","class_id":cls,"i":j})
            records.append({"role":"validation","class_id":cls,"i":10})
        pack={"records":records}
        for arm,classes in qualifier.ARMS:
            qfull=qualifier.build_aux_schedule(records,17,classes)
            mainfull=driver.aux_schedule(pack,17,records,classes)
            self.assertEqual(qfull,mainfull)
            self.assertEqual(qualifier.schedule_hash(qfull),driver.schedule_hash(mainfull))
            self.assertEqual(qualifier.prefix_schedule_hash(qfull),
                driver.schedule_hash({s:mainfull[s] for s in (449,450,451,452)}))

    def test_qualification_shared_hashes_do_not_conflate_driver_sources(self):
        main_sources={"driver_source_sha256":"main-driver","common_sha256":"common",
            "records_sha256":"data"}
        qplan={"qualifier_source_sha256":"qual-driver","common_sha256":"common",
            "records_sha256":"data"}
        self.assertTrue(driver.qualification_shared_sources_match(qplan,main_sources,"qual-driver"))
        self.assertFalse(driver.qualification_shared_sources_match(qplan,main_sources,"different-qual-driver"))
        qplan["records_sha256"]="other-data"
        self.assertFalse(driver.qualification_shared_sources_match(qplan,main_sources,"qual-driver"))


if __name__ == "__main__":
    unittest.main()
