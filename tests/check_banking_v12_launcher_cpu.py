import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import launch_banking_optimizer_boundary_v12 as launcher


def _args(tmp_path, *, expected_hash=None, output_dir=None, launch_record=None):
    config = tmp_path / "v12.json"
    config.write_text(json.dumps({
        "cuda_allocator_conf": "garbage_collection_threshold:0.7",
        "cuda_memory_fraction": 0.8,
    }), encoding="utf-8")
    v10_config = tmp_path / "v10.json"
    v10_config.write_text("{}", encoding="utf-8")
    project = tmp_path / "project"
    model = tmp_path / "model"
    pack = tmp_path / "pack"
    foundation = tmp_path / "foundations"
    for path in (project, model, pack, foundation):
        path.mkdir()
    # Keep outputs outside the protected training-pack parent: the launcher
    # rejects any overlap with that source tree, not just the pack directory.
    out_parent = tmp_path.parent / f"out-{tmp_path.name}"
    record_parent = tmp_path.parent / f"record-{tmp_path.name}"
    out_parent.mkdir()
    record_parent.mkdir()
    output_dir = output_dir or (out_parent / "run")
    launch_record = launch_record or (record_parent / "launch.json")
    values = [
        "--config", str(config), "--v10-config", str(v10_config),
        "--project-root", str(project), "--model-path", str(model),
        "--training-pack", str(pack), "--foundation-root", str(foundation),
        "--output-dir", str(output_dir), "--launch-record", str(launch_record),
        "--seed", "17",
    ]
    if expected_hash is not None:
        values += ["--expected-config-sha256", expected_hash]
    return values, config, project, output_dir, launch_record


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_expected_config_hash_is_required_before_path_or_runner_work(tmp_path):
    args, _, _, _, launch_record = _args(tmp_path)
    with pytest.raises(SystemExit):
        launcher.main(args)
    assert not launch_record.exists()


def test_config_hash_mismatch_writes_failed_record_without_importing_runner(tmp_path, monkeypatch):
    args, _, _, output_dir, launch_record = _args(tmp_path, expected_hash="0" * 64)

    def forbidden_import(_name):
        raise AssertionError("runner import must occur only after config hash validation")

    monkeypatch.setattr(launcher.importlib, "import_module", forbidden_import)
    with pytest.raises(ValueError, match="config SHA-256 differs"):
        launcher.main(args)
    state = json.loads(launch_record.read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["error"]["type"] == "ValueError"
    assert state["automatic_retries"] == 0
    assert state["test_payload_accessed"] is False
    assert state["runner_callback_attempted"] is False
    assert state["gpu_policy_applied"] is False
    assert "launcher_wall_seconds_including_preflight" in state
    assert not output_dir.exists()


def test_output_inside_protected_project_root_is_rejected(tmp_path):
    project_out = tmp_path / "project" / "future-output"
    args, _, project, _, launch_record = _args(
        tmp_path, expected_hash="0" * 64, output_dir=project_out)
    assert project_out.is_relative_to(project)
    with pytest.raises(ValueError, match="overlaps a protected"):
        launcher.main(args)
    assert not launch_record.exists()
    assert not project_out.exists()


def test_output_directory_and_launch_record_must_not_alias(tmp_path):
    aliased_path = tmp_path.parent / f"aliased-{tmp_path.name}"
    args, _, _, _, _ = _args(
        tmp_path, expected_hash="0" * 64,
        output_dir=aliased_path, launch_record=aliased_path)
    with pytest.raises(ValueError, match="output.*record|overlap"):
        launcher.main(args)
    assert not aliased_path.exists()


def test_cpu_preflight_exception_is_persisted_before_any_cuda_policy(tmp_path, monkeypatch):
    args, config, _, output_dir, launch_record = _args(
        tmp_path, expected_hash=None)
    expected_hash = _sha(config)
    args += ["--expected-config-sha256", expected_hash]
    runner_name = "run_banking_optimizer_boundary_v12"
    runner = types.ModuleType(runner_name)
    runner.__file__ = str(TOOLS / "run_banking_optimizer_boundary_v12.py")

    def fail_preflight(_args):
        raise RuntimeError("intentional CPU-preflight sentinel")

    runner._preflight = fail_preflight
    monkeypatch.setitem(sys.modules, runner_name, runner)
    with pytest.raises(RuntimeError, match="intentional CPU-preflight sentinel"):
        launcher.main(args)
    state = json.loads(launch_record.read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["error"]["type"] == "RuntimeError"
    assert state["runner_callback_attempted"] is False
    assert state["gpu_policy_applied"] is False
    assert state["test_payload_accessed"] is False
    assert not output_dir.exists()
