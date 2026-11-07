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
import collect_banking_main_queue_receipts_v10 as collector
import launch_banking_memory_policy_v10 as launcher


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    path.write_text(raw, encoding="utf-8")
    return sha_bytes(path.read_bytes())


def make_fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    queue_root = tmp_path / "queue-root"
    queue_root.mkdir()
    config_path = ROOT / "configs" / "banking_v10_frozen_k10.json"
    config_file_sha = collector.sha256_file(config_path)
    # These are intentionally distinct: file SHA versus canonical semantic SHA.
    assert config_file_sha == "785ded5cc9d205dc3cbc6d25179abf6fd0d589ce6943a42626aa1692b54dfd9d"
    semantic_sha = "f65c55587c3795b83d0d62695dbfc0501aa2927576e492d6185dd18e5558e7e4"
    source_hashes = {"tools/run_banking_cil_v10.py": "a" * 64}
    training_manifest_sha = "b" * 64
    source_lock = "c" * 64
    project = tmp_path / "project"; project.mkdir()
    model = tmp_path / "model"; model.mkdir()
    pack = tmp_path / "train-dev.json"; pack.write_text("{}", encoding="utf-8")
    foundation = tmp_path / "foundation"; foundation.mkdir()
    release = tmp_path / "release.json"; release.write_text("{}", encoding="utf-8")
    script_path = TOOLS / "run_banking_cil_v10.py"
    launcher_path = TOOLS / "launch_banking_memory_policy_v10.py"
    contracts = []
    first_expected = None
    for seed in (17, 29, 43):
        for arm in ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE"):
            stem = f"seed{seed}-{arm.lower()}"
            output = queue_root / stem
            record_path = queue_root / "launcher-records" / f"{stem}.json"
            argv = ["--phase", "main", "--project-root", str(project),
                    "--model-path", str(model), "--training-pack", str(pack),
                    "--config", str(config_path), "--output-dir", str(output),
                    "--seed", str(seed), "--arm", arm,
                    "--foundation-dir", str(foundation), "--release-artifact", str(release)]
            contract = {
                "schema": "banking-v10-memory-policy-launch-v1",
                "launcher_sha256": collector.sha256_file(launcher_path),
                "project_root": str(project), "task_root": str(ROOT),
                "model_path": str(model), "training_pack": str(pack),
                "config_path": str(config_path), "config_file_sha256": config_file_sha,
                "config_semantic_sha256": semantic_sha,
                "training_manifest_sha256": training_manifest_sha,
                "source_lock_sha256": source_lock, "source_hashes": source_hashes,
                "script_key": "run_banking_cil_v10.py", "script_path": str(script_path),
                "script_sha256": collector.sha256_file(script_path), "argv": argv,
                "launch_record_path": str(record_path),
                "allocator_conf": launcher.ALLOCATOR_POLICY, "memory_fraction": .8,
                "device": "cuda:0", "torch_version": launcher.REQUIRED_TORCH_VERSION,
            }
            contract_path = tmp_path / "contracts" / f"{stem}.json"
            contract_sha = dump(contract_path, contract)
            contracts.append({"contract_path": str(contract_path), "contract_sha256": contract_sha})
            if seed == 17 and arm == "CMS_ONLY":
                first_expected = (contract, contract_path, contract_sha, output, record_path)
    queue = {"schema": "banking-v10-memory-policy-queue-v1",
             "launcher_sha256": collector.sha256_file(launcher_path),
             "queue_root": str(queue_root), "jobs": contracts,
             "minimum_free_before_start_bytes": 1}
    queue_path = tmp_path / "main-queue.json"
    queue_sha = dump(queue_path, queue)
    contract, contract_path, contract_sha, output, record_path = first_expected
    output.mkdir()
    (output / "checkpoints").mkdir()
    cp_path = output / "checkpoints" / "final-task06.pt"
    cp_path.write_bytes(b"small fixture; collector must not hash checkpoint bytes")
    cp_sha = collector.sha256_file(cp_path)
    receipt = {"schema": "banking-cil-v10-run-receipt", "status": "complete",
        "seed": 17, "arm": "CMS_ONLY", "tasks_completed": 7,
        "config_sha256": semantic_sha, "source_lock_sha256": source_lock,
        "training_manifest_sha256": training_manifest_sha,
        "model_immutable_versions_unchanged": True, "test_data_included": False,
        "final_checkpoint": "checkpoints/final-task06.pt", "final_checkpoint_sha256": cp_sha}
    receipt_path = output / "run-receipt.json"
    receipt_sha = dump(receipt_path, receipt)
    record = {"schema": "banking-v10-memory-policy-launch-record-v1", "status": "complete",
        "exit_code": 0, "contract_sha256": contract_sha,
        "launcher_sha256": contract["launcher_sha256"], "script_key": contract["script_key"],
        "script_sha256": contract["script_sha256"],
        "config_sha256": config_file_sha, "config_semantic_sha256": semantic_sha,
        "source_lock_sha256": source_lock, "source_hashes": source_hashes,
        "training_manifest_sha256": training_manifest_sha,
        "memory_fraction_callback_registered_before_cuda_init": True,
        "memory_fraction_callback_called": True, "memory_fraction_callback_error": None,
        "cuda_initialized_at_exit": True, "test_data_accessed_by_launcher": False,
        "target_terminal_artifact": {"path": str(receipt_path), "sha256": receipt_sha,
                                     "checkpoint_sha256": cp_sha}}
    record_sha = dump(record_path, record)
    status = {"schema": "banking-v10-memory-policy-queue-status-v1", "status": "stopped",
        "queue_sha256": queue_sha, "launcher_sha256": contract["launcher_sha256"],
        "automatic_retries": 0, "jobs": [{"job": "job-01-run_banking_cil_v10.py",
            "status": "stopped", "contract_sha256": contract_sha}],
        "test_data_accessed_by_queue": False}
    status_path = queue_root / "queue-status.json"
    status_sha = dump(status_path, status)
    return {"queue_root": queue_root, "queue_path": queue_path, "queue_sha": queue_sha,
            "status_path": status_path, "status_sha": status_sha,
            "first_contract_sha": contract_sha, "first_receipt_sha": receipt_sha,
            "first_record_sha": record_sha, "first_record_path": record_path,
            "first_receipt_path": receipt_path, "first_cp_path": cp_path,
            "queue_status_raw": status_path.read_bytes()}


def collect(fixture, output):
    return collector.collect_snapshot(queue_path=fixture["queue_path"],
        queue_sha256=fixture["queue_sha"], queue_root=fixture["queue_root"],
        queue_status_sha256=fixture["status_sha"],
        first_contract_sha256=fixture["first_contract_sha"],
        first_receipt_sha256=fixture["first_receipt_sha"],
        first_launch_record_sha256=fixture["first_record_sha"],
        output_path=output)


def test_collects_first_completed_and_eight_pending_without_mutating_queue(tmp_path):
    fixture = make_fixture(tmp_path)
    snapshot = collect(fixture, tmp_path / "snapshot.json")
    assert snapshot["runs"][0]["state"] == "complete_receipt_verified_checkpoint_bytes_deferred"
    assert [item["state"] for item in snapshot["runs"][1:]] == ["pending"] * 8
    assert snapshot["runs"][0]["checkpoint_bytes_hashed"] is False
    assert snapshot["runs"][0]["target_terminal_artifact_sha256"] == fixture["first_receipt_sha"]
    assert fixture["status_path"].read_bytes() == fixture["queue_status_raw"]
    saved = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    assert "snapshot_file_sha256" not in saved


def test_raw_file_and_semantic_hashes_are_checked_in_their_own_domains(tmp_path):
    fixture = make_fixture(tmp_path)
    record = json.loads(fixture["first_record_path"].read_text(encoding="utf-8"))
    contract_path = next(Path(job["contract_path"]) for job in json.loads(
        fixture["queue_path"].read_text(encoding="utf-8"))["jobs"][:1])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert record["config_sha256"] == contract["config_file_sha256"]
    assert record["config_semantic_sha256"] == contract["config_semantic_sha256"]
    assert record["config_sha256"] != record["config_semantic_sha256"]
    snapshot = collect(fixture, tmp_path / "snapshot.json")
    assert snapshot["runs"][0]["state"].startswith("complete_")


def test_swapped_hash_domain_is_integrity_error_and_first_run_is_not_counted_complete(tmp_path):
    fixture = make_fixture(tmp_path)
    record = json.loads(fixture["first_record_path"].read_text(encoding="utf-8"))
    record["config_sha256"], record["config_semantic_sha256"] = (
        record["config_semantic_sha256"], record["config_sha256"])
    fixture["first_record_sha"] = dump(fixture["first_record_path"], record)
    # Make the explicit first-run hash match the tampered bytes; semantic validation must still fail.
    with pytest.raises(ValueError, match="preexisting first scientific run"):
        collect(fixture, tmp_path / "snapshot.json")


def test_launch_record_write_window_is_reported_running_not_failure(tmp_path):
    fixture = make_fixture(tmp_path)
    contract_path = next(Path(job["contract_path"]) for job in json.loads(
        fixture["queue_path"].read_text(encoding="utf-8"))["jobs"][:1])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    item = {"contract": contract, "job": {"contract_sha256": fixture["first_contract_sha"]},
        "contract_path": contract_path,
        "seed": 17, "arm": "CMS_ONLY", "output": fixture["first_receipt_path"].parent,
        "record_path": fixture["first_record_path"], "receipt_path": fixture["first_receipt_path"],
        "expected_receipt": {"schema": "banking-cil-v10-run-receipt", "status": "complete",
                             "seed": 17, "arm": "CMS_ONLY", "tasks_completed": 7}}
    record = json.loads(fixture["first_record_path"].read_text(encoding="utf-8"))
    record["status"] = "launching"
    dump(fixture["first_record_path"], record)
    state = collector._classify(item, launcher_sha256=contract["launcher_sha256"])
    assert state["state"] == "running"
