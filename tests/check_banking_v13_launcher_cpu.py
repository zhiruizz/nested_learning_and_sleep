from pathlib import Path

import pytest

from tools.launch_banking_sleep_lifecycle_v13 import ARMS, SEEDS, validate_output_paths


def test_v13_launcher_grid_is_explicit():
    assert SEEDS == (17, 29, 43)
    assert ARMS == ("CMS_ONLY", "B1_HIGH", "B1_LOW", "B32_HIGH", "B32_LOW", "B32_LOW_DOSE")


def test_output_paths_accept_fresh_disjoint_siblings(tmp_path):
    protected = tmp_path / "source"
    protected.mkdir()
    out_parent = tmp_path / "results"
    out_parent.mkdir()
    rec_parent = tmp_path / "records"
    rec_parent.mkdir()
    out, rec = validate_output_paths(out_parent / "run", rec_parent / "launch.json", [protected])
    assert out == (out_parent / "run").resolve()
    assert rec == (rec_parent / "launch.json").resolve()


def test_output_paths_reject_source_overlap(tmp_path):
    protected = tmp_path / "source"
    protected.mkdir()
    out_parent = tmp_path / "source" / "outputs"
    out_parent.mkdir()
    rec_parent = tmp_path / "records"
    rec_parent.mkdir()
    with pytest.raises(ValueError):
        validate_output_paths(out_parent / "run", rec_parent / "launch.json", [protected])


def test_output_paths_reject_existing_or_overlapping_records(tmp_path):
    base = tmp_path / "results"
    base.mkdir()
    protected = tmp_path / "source"
    protected.mkdir()
    out = base / "run"
    record = out / "launch.json"
    with pytest.raises(ValueError):
        validate_output_paths(out, record, [protected])
    out.mkdir()
    with pytest.raises(FileExistsError):
        validate_output_paths(out, base / "launch.json", [protected])


def test_output_parent_must_exist(tmp_path):
    with pytest.raises(ValueError):
        validate_output_paths(tmp_path / "missing" / "run", tmp_path / "record.json", [])
