#!/usr/bin/env python3
"""Validate byte hashes and development-only claims in a staged CLINC bundle.

This public-side checker consumes only the published capture manifest and
published artifacts. It never opens datasets, predictions, model files, or
anything outside the supplied bundle root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CAPTURE = "results/clinc-v8-development/public-capture.json"
FORBIDDEN_SUFFIXES = {".jsonl", ".parquet", ".pt", ".pth", ".safetensors"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(root: Path) -> dict:
    root = root.resolve()
    capture_path = root / CAPTURE
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    if capture.get("schema") != "clinc-v8-public-development-capture-v1":
        raise ValueError("unexpected public capture schema")
    if capture.get("test_data_included") is not False:
        raise ValueError("capture does not affirm test-data exclusion")
    if capture.get("raw_dataset_rows_included") is not False:
        raise ValueError("capture does not affirm dataset-row exclusion")
    if capture.get("model_weights_or_checkpoints_included") is not False:
        raise ValueError("capture does not affirm checkpoint exclusion")
    artifacts = capture.get("source_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("capture source artifact list is absent")
    checked = set()
    for record in artifacts:
        relative = record.get("public_artifact")
        expected = record.get("public_artifact_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("malformed artifact capture record")
        path = (root / relative).resolve()
        if root not in path.parents or path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"unsafe or private artifact type: {relative}")
        if relative in checked:
            raise ValueError(f"duplicate public artifact record: {relative}")
        checked.add(relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"published artifact hash mismatch: {relative}")
    if "results/clinc-v8-development/public-capture.json" in checked:
        raise ValueError("capture must not recursively hash itself")
    if capture.get("full_grid_status") != "in_progress; no completed six-run result in this release":
        raise ValueError("development capture overstates full-grid completion")
    return {"schema": "clinc-v8-public-bundle-validation-v1", "passed": True,
            "verified_artifacts": len(checked), "test_data_included": False,
            "full_grid_status": capture["full_grid_status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_bundle(args.bundle_root), sort_keys=True))


if __name__ == "__main__":
    main()
