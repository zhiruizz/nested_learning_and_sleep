from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import launch_banking_memory_policy_v10 as frozen
import launch_banking_checked_queue_v10 as checked


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path):
    out = tmp_path / "seed17-cms_only"
    out.mkdir()
    checkpoint = out / "final.pt"
    checkpoint.write_bytes(b"tiny-cpu-fixture-checkpoint")
    receipt = {
        "schema": "banking-cil-v10-run-receipt", "status": "complete",
        "seed": 17, "arm": "CMS_ONLY", "tasks_completed": 7,
        "config_sha256": "b" * 64, "source_lock_sha256": "c" * 64,
        "training_manifest_sha256": "d" * 64,
        "model_immutable_versions_unchanged": True, "test_data_included": False,
        "final_checkpoint": "final.pt", "final_checkpoint_sha256": sha(checkpoint.read_bytes()),
    }
    receipt_path = out / "run-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    record_path = tmp_path / "launcher-record.json"
    contract = {
        "script_key": "run_banking_cil_v10.py",
        "script_sha256": "e" * 64,
        "launcher_sha256": "f" * 64,
        "config_file_sha256": "a" * 64,
        "config_semantic_sha256": "b" * 64,
        "source_lock_sha256": "c" * 64,
        "source_hashes": {"tools/run.py": "e" * 64},
        "training_manifest_sha256": "d" * 64,
        "allocator_conf": frozen.ALLOCATOR_POLICY,
        "memory_fraction": frozen.MEMORY_FRACTION,
    }
    target = frozen._validate_target_terminal(
        {**contract, "argv": ["--phase", "main", "--seed", "17", "--arm", "CMS_ONLY"]},
        {"output_dir": out, "config_semantic_sha256": "b" * 64,
         "source_lock_sha256": "c" * 64, "manifest_sha256": "d" * 64})
    record = {
        "schema": "banking-v10-memory-policy-launch-record-v1",
        "status": "complete", "exit_code": 0, "contract_sha256": "9" * 64,
        "launcher_sha256": contract["launcher_sha256"],
        "script_key": contract["script_key"], "script_sha256": contract["script_sha256"],
        # c353 semantics: raw file hash and canonical config hash are distinct fields.
        "config_sha256": contract["config_file_sha256"],
        "config_semantic_sha256": contract["config_semantic_sha256"],
        "source_lock_sha256": contract["source_lock_sha256"],
        "source_hashes": contract["source_hashes"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "model_revision": "fixed-model-rev", "allocator_conf": contract["allocator_conf"],
        "memory_fraction": contract["memory_fraction"],
        "memory_fraction_callback_registered_before_cuda_init": True,
        "memory_fraction_callback_called": True, "memory_fraction_callback_error": None,
        "cuda_initialized_before_target": False, "cuda_initialized_at_exit": True,
        "test_data_accessed_by_launcher": False, "automatic_retries": 0,
        "target_terminal_artifact": {
            "path": str(receipt_path.resolve()), "sha256": sha(receipt_path.read_bytes()),
            "checkpoint_sha256": receipt["final_checkpoint_sha256"],
        },
    }
    item = {"contract": {**contract, "argv": ["--phase", "main", "--seed", "17", "--arm", "CMS_ONLY"]},
            "launch_record": str(record_path)}
    return item, record, target


def test_success_record_uses_raw_and_semantic_config_hash_domains(tmp_path):
    item, record, target = _fixture(tmp_path)
    summary = checked.validate_success_record(
        contract=item["contract"], contract_sha256="9" * 64, record=record,
        target=target, expected_model_revision="fixed-model-rev")
    assert summary["seed"] == 17 and summary["arm"] == "CMS_ONLY"
    assert summary["receipt_status"] == "complete"
    assert summary["checkpoint_sha256"] == target["checkpoint_sha256"]

    swapped = dict(record)
    swapped["config_sha256"], swapped["config_semantic_sha256"] = (
        swapped["config_semantic_sha256"], swapped["config_sha256"])
    with pytest.raises(ValueError, match="config_sha256"):
        checked.validate_success_record(
            contract=item["contract"], contract_sha256="9" * 64, record=swapped,
            target=target, expected_model_revision="fixed-model-rev")


def test_rejects_terminal_checkpoint_or_receipt_mismatch(tmp_path):
    item, record, target = _fixture(tmp_path)
    corrupted = dict(record)
    corrupted["target_terminal_artifact"] = dict(record["target_terminal_artifact"])
    corrupted["target_terminal_artifact"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="terminal artifact"):
        checked.validate_success_record(
            contract=item["contract"], contract_sha256="9" * 64, record=corrupted,
            target=target, expected_model_revision="fixed-model-rev")


def test_exact_nine_job_grid_requires_outputs_directly_under_root(tmp_path):
    root = tmp_path / "fresh-root"
    prepared = []
    for seed in (17, 29, 43):
        for arm in ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE"):
            name = f"seed{seed}-{arm.lower()}"
            contract = {"script_key": "run_banking_cil_v10.py",
                        "argv": ["--phase", "main", "--seed", str(seed), "--arm", arm]}
            prepared.append({"contract": contract, "output_dir": str(root / name),
                             "launch_record": str(root / "launcher-records" / f"{name}.json")})
    checked.validate_nine_job_queue({"schema": checked.QUEUE_SCHEMA}, prepared, root)
    prepared[0]["output_dir"] = str(root / "nested" / "seed17-cms_only")
    with pytest.raises(ValueError, match="direct queue-root child"):
        checked.validate_nine_job_queue({"schema": checked.QUEUE_SCHEMA}, prepared, root)


def test_serial_queue_stops_at_first_child_failure_without_retry():
    called = []

    def run_one(item):
        called.append(item)
        if item == "first":
            raise RuntimeError("simulated child failure")
        return item

    with pytest.raises(RuntimeError, match="simulated child failure"):
        checked._run_serially(["first", "second", "third"], run_one)
    assert called == ["first"]


def test_allocator_setting_is_preserved_or_conflict_fails_closed():
    policy = "garbage_collection_threshold:0.7"
    assert checked._child_environment({}, policy)["PYTORCH_CUDA_ALLOC_CONF"] == policy
    assert checked._child_environment({"PYTORCH_CUDA_ALLOC_CONF": policy}, policy)[
        "PYTORCH_CUDA_ALLOC_CONF"] == policy
    with pytest.raises(ValueError, match="refusing to overwrite"):
        checked._child_environment({"PYTORCH_CUDA_ALLOC_CONF": "other:setting"}, policy)
