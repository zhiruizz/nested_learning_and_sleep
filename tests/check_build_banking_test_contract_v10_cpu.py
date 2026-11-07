from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import evaluate_banking_test_v10 as evaluator
import build_banking_test_contract_v10 as builder
import run_banking_cil_v10 as driver
import banking_cil_common_v10 as common


def h(ch: str) -> str:
    return hashlib.sha256(ch.encode()).hexdigest()


class TestBankingTestContractBuilder(unittest.TestCase):
    def setUp(self):
        self.config = {
            "_logical_sha256": h("logical"),
            "source_hashes": {"tools/a.py": h("source")},
            "model_revision": "locked-model-rev",
            "model_assets_sha256": {"weights.safetensors": h("model")},
            "max_sequence_length": 128,
        }
        self.refs = {str(seed): {"receipt_sha256": h(f"receipt-{seed}"),
                                 "checkpoint_sha256": h(f"checkpoint-{seed}")}
                     for seed in (17, 29, 43)}

    def make(self, refs=None):
        return builder.make_contract(
            evaluator_sha256=h("evaluator"), config=self.config,
            config_file_sha256=h("config-file"), source_lock_sha256=h("source-lock"),
            training_manifest_sha256=h("train-dev"), training_seal_sha256=h("seal"),
            main_release_sha256=h("release"), foundation_refs=refs or self.refs,
            evaluator=evaluator)

    def test_contract_matches_exact_frozen_evaluator_schema(self):
        contract = self.make()
        self.assertEqual(set(contract), {
            "schema", "evaluator_sha256", "training_config_sha256",
            "training_config_file_sha256", "source_lock_sha256", "source_hashes",
            "model_revision", "model_assets_sha256", "training_manifest_sha256",
            "training_seal_sha256", "main_release_sha256", "foundation_refs",
            "official_test", "evaluation_policy", "training_manifest_excludes_test"})
        self.assertEqual(contract["official_test"]["sha256"], evaluator.EXPECTED_TEST_SHA256)
        self.assertEqual(contract["evaluation_policy"]["optimizer_updates"], 0)
        self.assertEqual(evaluator.MAX_FORWARDS, 600_000)
        self.assertEqual(contract["evaluation_policy"]["max_backbone_forwards"],
                         evaluator.MAX_FORWARDS)
        self.assertEqual(
            contract["evaluation_policy"]["secondary_subset"],
            "exclude_normalized_text_overlap_with_training_and_development",
        )
        self.assertEqual(contract["evaluation_policy"]["secondary_overlap_roles"],
                         ["train", "development"])
        self.assertEqual(contract["evaluation_policy"]["secondary_overlap_rule"],
                         "exact_common_normalize_text_match_in_either_role")
        self.assertEqual(contract["foundation_refs"], self.refs)
        # The builder pins metadata only; no test CSV path or payload is accepted.
        self.assertNotIn("test_csv", contract)

    def test_incomplete_foundation_map_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make({"17": self.refs["17"], "29": self.refs["29"]})

    def test_malformed_foundation_hash_is_rejected(self):
        refs = dict(self.refs)
        refs["43"] = {"receipt_sha256": "x" * 64,
                       "checkpoint_sha256": self.refs["43"]["checkpoint_sha256"]}
        with self.assertRaises(ValueError):
            self.make(refs)

    def test_build_passes_validated_logical_hash_to_real_evaluator_validator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project_root = root / "project"
            model_dir = root / "model"
            training_root = root / "main"
            foundation_root = root / "foundations"
            for path in (project_root, model_dir, training_root, foundation_root):
                path.mkdir()
            for seed in (17, 29, 43):
                (foundation_root / f"seed{seed}").mkdir()

            config_path = root / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            manifest_path = root / "training-manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            manifest_sha = builder.sha256_file(manifest_path)
            source_hashes = {"tools/fake.py": h("source")}
            source_lock_sha = h("source-lock")
            logical_sha = h("logical-config")
            cfg = {"source_hashes": source_hashes,
                   "model_revision": "locked-model-revision",
                   "model_assets_sha256": {"weights.safetensors": h("model")},
                   "training_manifest_sha256": manifest_sha,
                   "max_sequence_length": 128}
            self.assertNotIn("_logical_sha256", cfg)

            refs = {str(seed): {"receipt_sha256": h(f"foundation-receipt-{seed}"),
                                "checkpoint_sha256": h(f"foundation-checkpoint-{seed}")}
                    for seed in (17, 29, 43)}
            release_path = root / "main-release.json"
            release_path.write_text(json.dumps({
                "schema": "banking-cil-v10-main-release-v1", "status": "ready",
                "config_sha256": logical_sha, "source_lock_sha256": source_lock_sha,
                "training_manifest_sha256": manifest_sha,
                "foundations": {
                    seed: {"run_dir": str(foundation_root / f"seed{seed}"), **ref}
                    for seed, ref in refs.items()},
            }), encoding="utf-8")
            seal_path = root / "training-seal.json"
            seal_rows = []
            for seed in (17, 29, 43):
                for arm in evaluator.ARMS:
                    seal_rows.append({
                        "seed": seed, "arm": arm, "status": "complete",
                        "tasks_completed": 7,
                        "final_checkpoint": "checkpoints/final-task06.pt",
                        "final_checkpoint_sha256": h(f"cp-{seed}-{arm}"),
                        "run_receipt_sha256": h(f"receipt-{seed}-{arm}"),
                        "config_sha256": logical_sha, "source_lock_sha256": source_lock_sha,
                        "training_manifest_sha256": manifest_sha,
                        "run_dir": str(training_root / f"seed{seed}-{arm}"),
                        "endpoint_metrics": {}, "work": {},
                    })
            seal_path.write_text(json.dumps({
                "schema": "banking-cil-v10-training-seal-v1", "complete": True,
                "config_sha256": logical_sha, "source_lock_sha256": source_lock_sha,
                "training_manifest_sha256": manifest_sha, "test_data_included": False,
                "official_test_read": False, "runs": seal_rows,
            }), encoding="utf-8")
            args = SimpleNamespace(
                output=root / "new-contract.json", project_root=project_root,
                task_root=ROOT, config=config_path, model_dir=model_dir,
                training_manifest=manifest_path, training_root=training_root,
                seal=seal_path, main_release_artifact=release_path,
                foundation_root=foundation_root)

            # External source/data and sealed CP checks are stubbed. The actual
            # build() caller, make_contract(), and evaluator validator all run.
            with patch.object(driver, "validate_config", return_value=cfg), \
                 patch.object(driver, "_config_sha", return_value=logical_sha), \
                 patch.object(common, "validate_manifest"), \
                 patch.object(driver, "_source_prepare",
                              return_value=(source_hashes, source_lock_sha, {},
                                            [{}] * 9229, [{}] * 770)), \
                 patch.object(driver, "_validate_foundation_release"), \
                 patch.object(evaluator, "verify_run_artifacts", return_value=[]):
                result = builder.build(args)

            built = json.loads(Path(result["contract_path"]).read_text(encoding="utf-8"))
            self.assertEqual(built["training_config_sha256"], logical_sha)
            self.assertNotIn("_logical_sha256", cfg)


if __name__ == "__main__":
    unittest.main()
