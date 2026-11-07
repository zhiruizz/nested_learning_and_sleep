import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import orchestrate_banking_v11_canonical_teacher as orchestrator


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_paths_projection_never_uses_test_csv(tmp_path):
    paths = {key: str(tmp_path / key) for key in orchestrator.PATH_KEYS}
    paths["test_csv"] = str(tmp_path / "must-not-open-test.csv")
    source = tmp_path / "paths.json"
    _write_json(source, paths)
    loaded = orchestrator.load_path_map(source)
    assert "test_csv" not in loaded
    argv = orchestrator.prepare_argv(loaded, task_root=tmp_path)
    assert str(tmp_path / "test_eval_dir" / "evaluation.json") in argv
    assert str(tmp_path / "must-not-open-test.csv") not in argv


def test_fixed_order_is_seed_then_arm():
    assert orchestrator.grid_keys() == (
        (17, "LTI_NO_REWARD"), (17, "LTI_CANONICAL_TEACHER"),
        (29, "LTI_NO_REWARD"), (29, "LTI_CANONICAL_TEACHER"),
        (43, "LTI_NO_REWARD"), (43, "LTI_CANONICAL_TEACHER"))


def test_validates_six_contracts_and_rejects_hash_or_identity_mismatch(tmp_path):
    contract_dir, output_root = tmp_path / "contracts", tmp_path / "run-root"
    contract_dir.mkdir(); output_root.mkdir()
    advertised = {}
    for seed, arm in orchestrator.grid_keys():
        path = contract_dir / f"{orchestrator.run_name(seed, arm)}.json"
        contract = {"seed": seed, "arm": arm,
            "output_dir": str(output_root / orchestrator.run_name(seed, arm)),
            "launch_record_path": str(output_root / "launch-records" / f"{orchestrator.run_name(seed, arm)}.json")}
        _write_json(path, contract)
        advertised[str(path)] = orchestrator.sha256_file(path)
    bundle = {"schema": orchestrator.CONTRACT_BUNDLE_SCHEMA, "status": "contracts_ready",
        "contract_count": 6, "contracts": advertised, "no_model_loaded": True,
        "optimizer_updates": 0, "test_payload_read": False, "launcher_invoked": False}
    assert len(orchestrator.validate_contract_bundle(bundle, contract_dir, output_root)) == 6
    bundle["contracts"][str(contract_dir / "seed17-lti-no-reward.json")] = "0" * 64
    with pytest.raises(ValueError, match="contract hash mismatch"):
        orchestrator.validate_contract_bundle(bundle, contract_dir, output_root)


def _complete_job(tmp_path):
    output = tmp_path / "seed17-lti-no-reward"
    record_path = tmp_path / "launch-records" / "seed17-lti-no-reward.json"
    contract = {"output_dir": str(output), "launch_record_path": str(record_path),
        "seed": 17, "arm": "LTI_NO_REWARD", "config_sha256": "a" * 64,
        "v11_source_lock_sha256": "b" * 64, "v10_source_lock_sha256": "c" * 64,
        "training_manifest_sha256": "d" * 64, "baseline_seal_sha256": "e" * 64,
        "foundation_checkpoint_sha256": "f" * 64}
    cp = output / "checkpoints" / "task01-final.pt"
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b"synthetic checkpoint bytes")
    cp_sha = orchestrator.sha256_file(cp)
    receipt = {"schema": "banking-cil-v11-run-receipt-v1", "status": "complete",
        "seed": 17, "arm": "LTI_NO_REWARD", "task_id": 1, "epochs": 2,
        "config_sha256": contract["config_sha256"], "source_lock_sha256": contract["v11_source_lock_sha256"],
        "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "baseline_training_seal_sha256": contract["baseline_seal_sha256"],
        "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
        "comparison_v10_sleep_task1_checkpoint_sha256": "1" * 64,
        "checkpoint": "checkpoints/task01-final.pt", "checkpoint_sha256": cp_sha,
        "task1_wake_ticks": 200, "test_data_included": False, "test_data_read": False,
        "model_immutable_versions_unchanged": True}
    receipt_path = output / "run-receipt.json"
    _write_json(receipt_path, receipt)
    receipt_sha = orchestrator.sha256_file(receipt_path)
    protocol = {"schema": "banking-cil-v11-protocol-v1", "status": "complete",
        "config_sha256": contract["config_sha256"], "source_lock_sha256": contract["v11_source_lock_sha256"],
        "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "baseline_training_seal_sha256": contract["baseline_seal_sha256"],
        "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
        "comparison_v10_sleep_task1_checkpoint_sha256": receipt["comparison_v10_sleep_task1_checkpoint_sha256"],
        "seed": 17, "arm": "LTI_NO_REWARD", "task_id": 1, "epochs": 2,
        "test_data_included": False, "test_data_read": False,
        "checkpoint_sha256": cp_sha, "run_receipt_sha256": receipt_sha}
    _write_json(output / "protocol.json", protocol)
    record = {"schema": "banking-v11-memory-policy-launch-record-v1", "status": "complete",
        "exit_code": 0, "contract_sha256": "9" * 64,
        "config_sha256": contract["config_sha256"], "source_lock_sha256": contract["v11_source_lock_sha256"],
        "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "baseline_seal_sha256": contract["baseline_seal_sha256"],
        "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
        "terminal_receipt": {"path": str(receipt_path), "sha256": receipt_sha,
                             "checkpoint_sha256": cp_sha},
        "preflight": {"comparison_v10_sleep_task1_checkpoint_sha256":
            receipt["comparison_v10_sleep_task1_checkpoint_sha256"]},
        "memory_fraction_callback_registered_before_cuda_init": True,
        "memory_fraction_callback_called": True, "memory_fraction_callback_error": None,
        "cuda_initialized_at_exit": True,
        "test_data_accessed_by_launcher": False}
    _write_json(record_path, record)
    return contract, record


def test_validates_receipt_protocol_checkpoint_and_launch_receipt(tmp_path):
    contract, record = _complete_job(tmp_path)
    assert orchestrator.validate_completed_job(contract, record["contract_sha256"])["status"] == "complete"


def test_uses_actual_launcher_no_test_field_and_rejects_true(tmp_path):
    contract, record = _complete_job(tmp_path)
    assert "test_data_accessed_by_launcher" in record
    assert "official_test_accessed_by_launcher" not in record
    record["test_data_accessed_by_launcher"] = True
    _write_json(Path(contract["launch_record_path"]), record)
    with pytest.raises(ValueError, match="memory launcher terminal record"):
        orchestrator.validate_completed_job(contract, record["contract_sha256"])


def test_rejects_test_access_and_terminal_hash_tampering(tmp_path):
    contract, record = _complete_job(tmp_path)
    receipt_path = Path(contract["output_dir"]) / "run-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["test_data_read"] = True
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="receipt does not match"):
        orchestrator.validate_completed_job(contract, record["contract_sha256"])


def test_serial_execution_stops_at_first_failure_without_retrying():
    jobs = [{"seed": seed, "arm": arm} for seed, arm in orchestrator.grid_keys()]
    called = []
    def fake(job):
        called.append((job["seed"], job["arm"]))
        if len(called) == 2:
            raise RuntimeError("synthetic launcher failure")
        return {"status": "complete"}
    with pytest.raises(RuntimeError, match="synthetic launcher failure"):
        orchestrator.execute_serial(jobs, fake)
    assert called == [(17, "LTI_NO_REWARD"), (17, "LTI_CANONICAL_TEACHER")]


def _write_prepared_contract_set(root):
    contract_dir, output_root = root / "contracts", root / "runs"
    contract_dir.mkdir(); output_root.mkdir()
    rows = []
    for seed, arm in orchestrator.grid_keys():
        name = orchestrator.run_name(seed, arm)
        contract_path = contract_dir / f"{name}.json"
        output = output_root / name
        record = output_root / "launch-records" / f"{name}.json"
        contract = {"seed": seed, "arm": arm, "output_dir": str(output),
            "launch_record_path": str(record), "v10_test_evaluation_sha256": "a" * 64,
            "baseline_seal_sha256": "b" * 64, "config_sha256": "c" * 64}
        _write_json(contract_path, contract)
        rows.append({"seed": seed, "arm": arm, "contract_path": str(contract_path),
            "contract_sha256": orchestrator.sha256_file(contract_path),
            "output_dir": str(output), "launch_record_path": str(record)})
    summary = {"schema": "banking-v11-serial-contract-preparation-v1", "status": "ready",
        "contract_count": 6, "jobs_launched": 0, "contracts": rows,
        "official_test_evaluation_sha256": "a" * 64,
        "v10_seal_sha256": "b" * 64, "v11_config_sha256": "c" * 64}
    _write_json(contract_dir / "serial-contract-bundle.json", summary)
    return contract_dir, output_root


def test_contract_edit_after_prepare_is_rejected_before_any_child(tmp_path, monkeypatch):
    contract_dir, output_root = _write_prepared_contract_set(tmp_path)
    edited = contract_dir / "seed17-lti-no-reward.json"
    contract = json.loads(edited.read_text(encoding="utf-8"))
    contract["arm"] = "LTI_CANONICAL_TEACHER"
    _write_json(edited, contract)

    values = {key: str(tmp_path / key) for key in orchestrator.PATH_KEYS}
    values["task_root"] = str(ROOT)
    values["v11_contract_dir"] = str(contract_dir)
    values["v11_train_root"] = str(output_root)
    path_file = tmp_path / "paths.json"
    _write_json(path_file, values)
    children = []
    monkeypatch.setattr(orchestrator.subprocess, "run", lambda *a, **k: children.append(a))
    with pytest.raises(ValueError, match="contract changed after preparation"):
        orchestrator.run_grid(path_file, "python")
    assert children == []
    assert not (contract_dir / "serial-run-status.json").exists()


def test_load_jobs_requires_exact_prepared_bundle_sha_map(tmp_path):
    contract_dir, output_root = _write_prepared_contract_set(tmp_path)
    paths = {"v11_contract_dir": contract_dir, "v11_train_root": output_root}
    assert len(orchestrator.load_contract_jobs(paths)) == 6
    # An extra contract-like JSON file is rejected before any job process.
    _write_json(contract_dir / "seed99-unplanned.json", {"seed": 99})
    with pytest.raises(ValueError, match="exactly the six expected contract files"):
        orchestrator.load_contract_jobs(paths)
