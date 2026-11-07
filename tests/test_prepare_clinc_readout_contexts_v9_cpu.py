from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_clinc_readout_contexts_v9.py"
spec = importlib.util.spec_from_file_location("prepare_clinc_readout_contexts_v9_test", MODULE_PATH)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)


def test_receipt_artifact_path_must_stay_under_run_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    artifact = root / "checkpoints" / "task.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"fixture")
    assert builder._safe_child_path(root, "checkpoints/task.pt") == artifact.resolve()
    with pytest.raises(ValueError, match="escapes"):
        builder._safe_child_path(root, "../outside.pt")
    with pytest.raises(ValueError, match="relative"):
        builder._safe_child_path(root, str(artifact.resolve()))


def test_task1_boundary_requires_one_unique_matching_checkpoint(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "task01-next0189-safe.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"task1 boundary fixture")
    row = {"event": "safe_checkpoint", "task_id": 1, "step": 188,
           "path": "checkpoints/task01-next0189-safe.pt", "sha256": "a" * 64}
    found, path = builder._unique_task1_checkpoint([__import__("json").dumps(row)], run)
    assert found == row
    assert path == checkpoint.resolve()
    with pytest.raises(ValueError, match="found 2"):
        builder._unique_task1_checkpoint([__import__("json").dumps(row)] * 2, run)
    escaped = dict(row, path="../../outside.pt")
    with pytest.raises(ValueError, match="escapes"):
        builder._unique_task1_checkpoint([__import__("json").dumps(escaped)], run)
