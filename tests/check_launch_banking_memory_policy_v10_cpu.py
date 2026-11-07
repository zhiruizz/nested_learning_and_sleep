from __future__ import annotations

import hashlib
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "launch_banking_memory_policy_v10.py"
SPEC = importlib.util.spec_from_file_location("banking_memory_policy_launcher", MODULE_PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class FakeCuda:
    def __init__(self):
        self.initialized = False
        self.queued = []
        self.calls = []

    def is_initialized(self):
        return self.initialized

    def _lazy_call(self, callback):
        self.queued.append(callback)

    def set_per_process_memory_fraction(self, fraction, *, device):
        self.calls.append((fraction, device))


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


def test_policy_is_queued_not_applied_early_and_applies_exact_fraction():
    torch = FakeTorch()
    state = launcher._install_memory_policy(torch, "cuda:0")
    assert state == {"registered": True, "called": False, "error": None}
    assert torch.cuda.is_initialized() is False
    assert torch.cuda.calls == []
    assert len(torch.cuda.queued) == 1
    torch.cuda.queued[0]()
    assert state["called"] is True
    assert torch.cuda.calls == [(0.8, "cuda:0")]


def test_policy_refuses_already_initialized_cuda():
    torch = FakeTorch()
    torch.cuda.initialized = True
    with pytest.raises(RuntimeError, match="initialized before"):
        launcher._install_memory_policy(torch, "cuda:0")
    assert not torch.cuda.queued


def test_torch_lazy_call_keeps_cuda_uninitialized_in_cpu_process():
    code = r'''
import json, torch
called=[]
before=torch.cuda.is_initialized()
torch.cuda._lazy_call(lambda: called.append("called"))
after=torch.cuda.is_initialized()
print(json.dumps({"before":before,"after":after,"called":called}))
'''
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {"before": False, "after": False, "called": []}


def _fake_contract(tmp_path: Path, queue_root: Path):
    output = queue_root / "foundations" / "seed17"
    record = queue_root / "launcher-records" / "seed17.json"
    return {
        "script_key": "run_banking_cil_v10.py",
        "argv": ["--phase", "foundation", "--seed", "17", "--output-dir", str(output)],
        "launch_record_path": str(record),
    }


def test_queue_preflight_requires_fresh_private_unique_outputs(tmp_path, monkeypatch):
    queue_root = tmp_path / "queue"
    cpath = tmp_path / "contract.json"
    cpath.write_text("{}", encoding="utf-8")
    contract = _fake_contract(tmp_path, queue_root)
    monkeypatch.setattr(launcher, "_load_contract", lambda *_: contract)
    monkeypatch.setattr(launcher, "_verify_frozen_inputs", lambda _: {"preflight": True})
    queue = {
        "schema": "banking-v10-memory-policy-queue-v1",
        "launcher_sha256": launcher.sha256_file(MODULE_PATH),
        "jobs": [{"contract_path": str(cpath), "contract_sha256": hashlib.sha256(b"{}").hexdigest()}],
        "minimum_free_before_start_bytes": 0,
    }
    prepared = launcher.validate_queue(queue, queue_root, task_root=ROOT)
    assert len(prepared) == 1
    assert prepared[0]["expected_receipt"]["status"] == "qualified"
    assert not queue_root.exists()


def test_queue_rejects_output_outside_queue_root(tmp_path, monkeypatch):
    queue_root = tmp_path / "queue"
    cpath = tmp_path / "contract.json"
    cpath.write_text("{}", encoding="utf-8")
    contract = _fake_contract(tmp_path, queue_root)
    contract["argv"][-1] = str(tmp_path / "outside" / "seed17")
    monkeypatch.setattr(launcher, "_load_contract", lambda *_: contract)
    monkeypatch.setattr(launcher, "_verify_frozen_inputs", lambda _: {"preflight": True})
    queue = {
        "schema": "banking-v10-memory-policy-queue-v1",
        "launcher_sha256": launcher.sha256_file(MODULE_PATH),
        "jobs": [{"contract_path": str(cpath), "contract_sha256": hashlib.sha256(b"{}").hexdigest()}],
        "minimum_free_before_start_bytes": 0,
    }
    with pytest.raises(ValueError, match="under the fresh queue root"):
        launcher.validate_queue(queue, queue_root, task_root=ROOT)


def test_launcher_contract_rejects_changed_allocator_policy(tmp_path):
    contract = {
        "schema": launcher.SCHEMA,
        "launcher_sha256": "x", "project_root": "x", "task_root": "x", "model_path": "x",
        "training_pack": "x", "config_path": "x", "config_file_sha256": "x",
        "config_semantic_sha256": "x", "training_manifest_sha256": "x", "source_lock_sha256": "x",
        "source_hashes": {}, "script_key": "run_banking_cil_v10.py", "script_path": "x",
        "script_sha256": "x", "argv": ["--phase", "foundation"], "launch_record_path": "x",
        "allocator_conf": "garbage_collection_threshold:0.9", "memory_fraction": 0.8, "device": "cuda:0",
        "torch_version": launcher.REQUIRED_TORCH_VERSION,
    }
    path = tmp_path / "bad-contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed policy"):
        launcher._load_contract(path, launcher.sha256_file(path))


def test_queue_child_command_uses_fresh_launch_subcommand():
    item = {"contract_path": r"D:\private\contract.json", "contract_sha256": "a" * 64}
    cmd = launcher._child_command(item, MODULE_PATH, python_executable="python-test")
    assert cmd == ["python-test", str(MODULE_PATH.resolve()), "launch", "--contract",
                   item["contract_path"], "--contract-sha256", item["contract_sha256"]]


def test_input_gate_failure_happens_before_torch_import_and_side_effects(tmp_path, monkeypatch):
    record = tmp_path / "records" / "attempt.json"
    fake = {"launch_record_path": str(record)}
    monkeypatch.setattr(launcher, "_load_contract", lambda *_: fake)
    monkeypatch.setattr(launcher, "_verify_frozen_inputs",
                        lambda _: (_ for _ in ()).throw(ValueError("source mismatch")))
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("torch imported before source gate")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="source mismatch"):
        launcher.launch(tmp_path / "unused.json", "0" * 64)
    assert not record.exists()


def test_queue_requires_nonexisting_parent_and_unique_logs(tmp_path, monkeypatch):
    root = tmp_path / "missing-parent" / "queue"
    cpath = tmp_path / "contract.json"
    cpath.write_text("{}", encoding="utf-8")
    contract = _fake_contract(tmp_path, root)
    monkeypatch.setattr(launcher, "_load_contract", lambda *_: contract)
    monkeypatch.setattr(launcher, "_verify_frozen_inputs", lambda _: {"preflight": True})
    queue = {
        "schema": "banking-v10-memory-policy-queue-v1",
        "launcher_sha256": launcher.sha256_file(MODULE_PATH),
        "jobs": [{"contract_path": str(cpath), "contract_sha256": hashlib.sha256(b"{}").hexdigest()}],
        "minimum_free_before_start_bytes": 0,
    }
    with pytest.raises(FileNotFoundError, match="parent must already exist"):
        launcher.validate_queue(queue, root, task_root=ROOT)
    assert not root.exists()


def test_direct_launch_marks_failed_gate_receipt_as_failed(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    record_path = tmp_path / "receipt" / "launch.json"
    foundation = {
        "schema": "banking-cil-v10-foundation-receipt", "status": "failed_gate",
        "seed": 17, "arm": "FOUNDATION_CMS_ONLY", "task_id": 0,
        "config_sha256": "b" * 64, "source_lock_sha256": "c" * 64,
        "training_manifest_sha256": "d" * 64, "model_immutable_versions_unchanged": True,
        "test_data_included": False,
    }
    (output / "foundation-receipt.json").write_text(json.dumps(foundation), encoding="utf-8")
    contract = {"script_key": "run_banking_cil_v10.py", "argv": [
        "--phase", "foundation", "--seed", "17", "--output-dir", str(output)],
        "launch_record_path": str(record_path), "torch_version": launcher.REQUIRED_TORCH_VERSION,
        "script_sha256": "f" * 64}
    validated = {
        "output_dir": output, "cfg": {"device": "cuda:0", "model_revision": "model@rev"},
        "config_sha256": "a" * 64, "config_semantic_sha256": "b" * 64,
        "source_lock_sha256": "c" * 64, "source_hashes": {},
        "manifest_sha256": "d" * 64, "launcher_sha256": "e" * 64,
        "script_path": Path("target.py"), "script_sha256": "f" * 64,
        "argv": contract["argv"], "phase": "foundation", "train_rows": 1, "development_rows": 1,
    }
    fake_cuda = FakeCuda()
    fake_torch = SimpleNamespace(__version__=launcher.REQUIRED_TORCH_VERSION, cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(launcher, "_load_contract", lambda *_: contract)
    monkeypatch.setattr(launcher, "_verify_frozen_inputs", lambda _: validated)
    monkeypatch.setattr(launcher.runpy, "run_path", lambda *args, **kwargs: None)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    with pytest.raises(ValueError, match="foundation receipt field 'status'"):
        launcher.launch(tmp_path / "contract.json", "0" * 64)
    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["exit_code"] == 1
    assert "target_terminal_artifact" not in persisted


def _write_probe_fixture(tmp_path: Path, *, include_sleep_parity: bool):
    output = tmp_path / "probe-output"
    output.mkdir()
    foundation_dir = tmp_path / "foundation"
    foundation_dir.mkdir()
    checkpoint = foundation_dir / "final.pt"
    checkpoint.write_bytes(b"fixture checkpoint bytes")
    cp_sha = launcher.sha256_file(checkpoint)
    foundation_receipt = {
        "status": "qualified", "seed": 17, "arm": "FOUNDATION_CMS_ONLY",
        "source_lock_sha256": "s" * 64, "config_sha256": "c" * 64,
        "training_manifest_sha256": "m" * 64, "checkpoint_sha256": cp_sha,
        "checkpoint": "final.pt", "test_data_included": False,
        "model_immutable_versions_unchanged": True,
    }
    receipt_path = foundation_dir / "foundation-receipt.json"
    receipt_path.write_text(json.dumps(foundation_receipt), encoding="utf-8")
    names = ["SLEEP_ADDITIVE", "GOLD_ADDITIVE", "SLEEP_FIXEDMIX", "GOLD_FIXEDMIX"]
    arms = {name: {"opportunity_count": 30, "actual_stage_metrics": {"3": {}, "30": {}}}
            for name in names}
    if include_sleep_parity:
        arms["SLEEP_ADDITIVE"]["disk_resume_equivalence"] = {
            "exact_model_adam_tx_rng_process_rng": True,
            "continuous_snapshot_sha256": "r" * 64,
            "restored_snapshot_sha256": "r" * 64,
        }
    result = {
        "schema": "banking-cil-v10-transfer-probe-v1", "status": "complete", "seed": 17,
        "task_id": 0, "development_only": True, "test_data_read": False,
        "banking_driver_source_lock_sha256": "s" * 64, "config_sha256": "c" * 64,
        "training_manifest_sha256": "m" * 64, "model_immutable_versions_unchanged": True,
        "foundation_receipt_sha256": launcher.sha256_file(receipt_path),
        "foundation_checkpoint_sha256": cp_sha, "arms": arms,
    }
    (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (output / "progress.json").write_text(json.dumps({"status": "complete", "completed_arms": names}),
                                         encoding="utf-8")
    contract = {"script_key": "probe_banking_transfer_v10.py", "argv": [
        "--phase", "run", "--foundation-dir", str(foundation_dir)]}
    validated = {"output_dir": output, "source_lock_sha256": "s" * 64,
                 "config_semantic_sha256": "c" * 64, "manifest_sha256": "m" * 64}
    return contract, validated


def test_probe_terminal_accepts_real_arm_specific_resume_evidence(tmp_path):
    contract, validated = _write_probe_fixture(tmp_path, include_sleep_parity=True)
    accepted = launcher._validate_target_terminal(contract, validated)
    assert accepted["receipt"]["status"] == "complete"
    assert accepted["checkpoint_sha256"] == "" or len(accepted["checkpoint_sha256"]) == 64


def test_probe_terminal_rejects_missing_sleep_resume_parity(tmp_path):
    contract, validated = _write_probe_fixture(tmp_path, include_sleep_parity=False)
    with pytest.raises(ValueError, match="disk recovery parity"):
        launcher._validate_target_terminal(contract, validated)


def test_direct_launch_fails_if_valid_target_never_applies_cuda_policy(tmp_path, monkeypatch):
    contract, validated = _write_probe_fixture(tmp_path, include_sleep_parity=True)
    record_path = tmp_path / "launcher-records" / "attempt.json"
    contract.update({"launch_record_path": str(record_path),
                     "torch_version": launcher.REQUIRED_TORCH_VERSION,
                     "script_sha256": "f" * 64})
    validated.update({"cfg": {"device": "cuda:0", "model_revision": "model@rev"},
                      "config_sha256": "a" * 64, "launcher_sha256": "e" * 64,
                      "source_hashes": {},
                      "script_path": Path("probe.py"), "script_sha256": "f" * 64,
                      "argv": contract["argv"], "phase": "run", "train_rows": 1,
                      "development_rows": 1})
    fake_torch = SimpleNamespace(__version__=launcher.REQUIRED_TORCH_VERSION, cuda=FakeCuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(launcher, "_load_contract", lambda *_: contract)
    monkeypatch.setattr(launcher, "_verify_frozen_inputs", lambda _: validated)
    monkeypatch.setattr(launcher.runpy, "run_path", lambda *args, **kwargs: None)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    with pytest.raises(RuntimeError, match="without applying"):
        launcher.launch(tmp_path / "contract.json", "0" * 64)
    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["memory_fraction_callback_registered_before_cuda_init"] is True
    assert persisted["memory_fraction_callback_called"] is False
