import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import prepare_banking_v11_launch_materials as prep
import run_banking_cil_v10 as v10
import run_banking_lti_ablation_v11 as v11


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class V11LaunchMaterialsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = json.loads((ROOT / "configs" / "banking_v10_frozen_k10.json").read_text(encoding="utf-8"))
        cls.base = v10.validate_config(raw, phase="main")

    def test_child_config_is_frozen_bounded_and_inherits_fixed_reward_weights(self):
        hashes = {"tools/run_banking_lti_ablation_v11.py": sha("runner")}
        cfg = prep.build_v11_config(self.base,
            v10_config_sha256=v10._config_sha(self.base),
            v10_source_lock_sha256=v11.canonical_sha(self.base["source_hashes"]),
            actual_v11_source_hashes=hashes,
            baseline_training_seal_sha256=sha("real-seal-placeholder-for-unit-contract"))
        self.assertEqual(cfg["seeds"], [17, 29, 43])
        self.assertEqual(cfg["arms"], ["LTI_NO_REWARD", "LTI_CANONICAL_TEACHER"])
        self.assertEqual(cfg["lti_teacher_validity_policy"], "canonical_teacher_required_for_reward")
        self.assertEqual(cfg["task_id"], 1)
        self.assertEqual(cfg["epochs"], 2)
        self.assertEqual(cfg["batch_size"], 8)
        self.assertEqual(cfg["candidate_batch_size"], 32)
        self.assertEqual(cfg["max_wall_seconds"], 1800)
        self.assertEqual(cfg["max_forward_calls"], 30000)
        self.assertEqual(cfg["max_update_steps"], 3000)
        self.assertEqual(cfg["lti_semantic_weight"], self.base["lti_semantic_weight"])
        self.assertEqual(cfg["lti_edit_weight"], self.base["lti_edit_weight"])
        self.assertNotIn("lti_empty_teacher_policy", cfg)
        self.assertFalse(cfg["test_data_included"])
        invalid = dict(cfg, lti_teacher_validity_policy="old_empty_teacher_gate")
        with self.assertRaisesRegex(ValueError, "canonical-teacher reward policy"):
            v11.validate_config(invalid, self.base, actual_source_hashes=hashes,
                actual_seal_sha256=sha("real-seal-placeholder-for-unit-contract"),
                actual_manifest_sha256=self.base["training_manifest_sha256"])

    def test_six_output_directories_are_fresh_children(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="v11-grid-") as temp:
            root = Path(temp)
            grid = prep._contract_grid(root)
            self.assertEqual(len(grid), 6)
            self.assertEqual({seed for seed, _, _ in grid}, {17, 29, 43})
            self.assertEqual({arm for _, arm, _ in grid}, set(prep.ARMS))
            self.assertTrue(all(path.parent == root and not path.exists() for _, _, path in grid))
            grid[0][2].mkdir()
            with self.assertRaises(ValueError):
                prep._contract_grid(root)

    def test_output_root_must_already_exist(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="v11-empty-") as temp:
            absent = Path(temp) / "not-created-by-builder"
            with self.assertRaises(FileNotFoundError):
                prep._contract_grid(absent)


if __name__ == "__main__":
    unittest.main()
