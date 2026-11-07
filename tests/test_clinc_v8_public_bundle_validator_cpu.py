"""Synthetic public-bundle integrity tests; no CLINC rows or models."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import hashlib

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "clinc_public_bundle_validator", ROOT / "tools" / "validate_clinc_v8_public_bundle.py")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class PublicBundleValidatorTests(unittest.TestCase):
    def make_bundle(self, root: Path):
        rel = "results/clinc-v8-development/development-summary.json"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b'{"schema":"toy-development-summary"}\n'
        path.write_bytes(payload)
        capture = {
            "schema": "clinc-v8-public-development-capture-v1",
            "test_data_included": False,
            "raw_dataset_rows_included": False,
            "model_weights_or_checkpoints_included": False,
            "full_grid_status": "in_progress; no completed six-run result in this release",
            "source_artifacts": [{"public_artifact": rel,
                                  "public_artifact_sha256": hashlib.sha256(payload).hexdigest()}],
        }
        capture_path = root / VALIDATOR.CAPTURE
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        capture_path.write_text(json.dumps(capture), encoding="utf-8")
        return rel, capture

    def test_accepts_hashed_development_only_capture(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            self.make_bundle(root)
            result = VALIDATOR.validate_bundle(root)
            self.assertTrue(result["passed"])
            self.assertEqual(result["verified_artifacts"], 1)
            self.assertFalse(result["test_data_included"])

    def test_rejects_modified_published_artifact(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            rel, _ = self.make_bundle(root)
            (root / rel).write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                VALIDATOR.validate_bundle(root)

    def test_rejects_capture_that_claims_test_data_or_completed_grid(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            self.make_bundle(root)
            path = root / VALIDATOR.CAPTURE
            capture = json.loads(path.read_text(encoding="utf-8"))
            capture["test_data_included"] = True
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test-data exclusion"):
                VALIDATOR.validate_bundle(root)

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            self.make_bundle(root)
            path = root / VALIDATOR.CAPTURE
            capture = json.loads(path.read_text(encoding="utf-8"))
            capture["full_grid_status"] = "complete"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overstates"):
                VALIDATOR.validate_bundle(root)

if __name__ == "__main__":
    unittest.main()
