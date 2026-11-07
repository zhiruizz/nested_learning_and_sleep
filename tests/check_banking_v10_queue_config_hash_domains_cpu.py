"""Regression fixture documenting the two distinct V10 config hash domains."""
from __future__ import annotations

import hashlib
import json


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return _sha(raw)


def test_launch_record_file_sha_and_semantic_sha_are_distinct_bindings():
    # A launch record's config_sha256 hashes the JSON file bytes; its
    # config_semantic_sha256 hashes the normalized config object. They are
    # expected to differ and must be compared to their matching contract fields.
    file_bytes = b'{\n  "status": "frozen", "learning_rate": 0.001\n}\n'
    config = json.loads(file_bytes.decode("utf-8"))
    contract = {
        "config_file_sha256": _sha(file_bytes),
        "config_semantic_sha256": _canonical_sha(config),
    }
    launch_record = {
        "config_sha256": _sha(file_bytes),
        "config_semantic_sha256": _canonical_sha(config),
    }
    assert launch_record["config_sha256"] == contract["config_file_sha256"]
    assert launch_record["config_semantic_sha256"] == contract["config_semantic_sha256"]
    assert launch_record["config_sha256"] != contract["config_semantic_sha256"]

