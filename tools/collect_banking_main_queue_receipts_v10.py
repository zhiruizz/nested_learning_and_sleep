#!/usr/bin/env python3
"""Read-only snapshot of per-run status for an original V10 nine-job queue.

This collector never launches work, changes queue status, opens test data, or
hashes large checkpoint files. A completed state requires the queue contract,
target receipt, launcher record, and cross-bound receipt/checkpoint metadata to
agree; final sealing remains responsible for hashing checkpoint bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


SEEDS = (17, 29, 43)
ARMS = ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")
QUEUE_SCHEMA = "banking-v10-memory-policy-queue-v1"
STATUS_SCHEMA = "banking-v10-memory-policy-queue-status-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path != root and root in path.parents


def _load_queue_contracts(queue_path: Path, expected_queue_sha: str,
                          queue_root: Path, launcher_path: Path):
    if sha256_file(queue_path) != expected_queue_sha:
        raise ValueError("original queue bytes differ from the supplied SHA")
    queue = _read_json(queue_path)
    if set(queue) != {"schema", "launcher_sha256", "queue_root", "jobs",
                      "minimum_free_before_start_bytes"}:
        raise ValueError("queue fields differ from the frozen V10 queue schema")
    if queue["schema"] != QUEUE_SCHEMA or Path(queue["queue_root"]).resolve() != queue_root.resolve():
        raise ValueError("queue schema/root binding mismatch")
    if sha256_file(launcher_path) != queue["launcher_sha256"]:
        raise ValueError("frozen V10 launcher source hash changed")
    jobs = queue["jobs"]
    expected_grid = [(seed, arm) for seed in SEEDS for arm in ARMS]
    if not isinstance(jobs, list) or len(jobs) != 9:
        raise ValueError("collector requires the original complete nine-contract queue")
    if type(queue["minimum_free_before_start_bytes"]) is not int or queue["minimum_free_before_start_bytes"] < 0:
        raise ValueError("queue minimum-free field is malformed")
    task_root = Path(__file__).resolve().parents[1]
    tools_dir = task_root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import launch_banking_memory_policy_v10 as launcher
    if Path(launcher.__file__).resolve() != launcher_path.resolve():
        raise ValueError("loaded V10 launcher differs from queue-pinned launcher")

    contracts = []
    paths_seen = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or set(job) != {"contract_path", "contract_sha256"}:
            raise ValueError(f"queue job {index} has invalid fields")
        contract_path = Path(job["contract_path"]).expanduser().resolve(strict=True)
        if contract_path in paths_seen:
            raise ValueError("queue repeats a contract path")
        paths_seen.add(contract_path)
        contract = launcher._load_contract(contract_path, job["contract_sha256"])
        script, argv = launcher._parse_argv(contract, task_root)
        seed = int(launcher._flag_value(argv, "--seed"))
        arm = launcher._flag_value(argv, "--arm")
        phase = launcher._flag_value(argv, "--phase")
        if (seed, arm) != expected_grid[index] or phase != "main":
            raise ValueError("queue job order/grid differs from the frozen nine-run plan")
        if contract["launcher_sha256"] != queue["launcher_sha256"]:
            raise ValueError("job contract and queue use different launcher hashes")
        config_path = Path(contract["config_path"]).expanduser().resolve(strict=True)
        if sha256_file(config_path) != contract["config_file_sha256"]:
            raise ValueError("frozen config bytes differ from a queue contract")
        output = launcher._output_dir_from_contract(contract)
        record = Path(contract["launch_record_path"]).expanduser().resolve()
        receipt, expected_receipt = launcher._receipt_for_contract(contract)
        if not _inside(output, queue_root) or output.parent != queue_root.resolve():
            raise ValueError("job output must be a direct child of the queue root")
        if not _inside(record, queue_root) or not _inside(receipt, queue_root):
            raise ValueError("job record/receipt must remain under the queue root")
        contracts.append({"index": index, "job": job, "contract": contract,
                          "contract_path": contract_path, "script_path": script,
                          "seed": seed, "arm": arm, "output": output,
                          "record_path": record, "receipt_path": receipt,
                          "expected_receipt": expected_receipt})
    return queue, contracts, launcher


def _classify(item: Mapping[str, Any], *, launcher_sha256: str) -> dict:
    contract = item["contract"]
    receipt_path: Path = item["receipt_path"]
    record_path: Path = item["record_path"]
    output: Path = item["output"]
    record_exists, receipt_exists = record_path.is_file(), receipt_path.is_file()
    base = {"seed": item["seed"], "arm": item["arm"],
            "contract_sha256": item["job"]["contract_sha256"],
            "contract_file_sha256": sha256_file(item["contract_path"]),
            "output_dir": str(output), "receipt_path": str(receipt_path),
            "launch_record_path": str(record_path),
            "receipt_present": receipt_exists, "launch_record_present": record_exists,
            "checkpoint_bytes_hashed": False}
    if not record_exists and not receipt_exists:
        return {**base, "state": "pending"}
    if not record_exists:
        return {**base, "state": "running_or_pending_launcher_record",
                "detail": "target receipt appeared before launch record"}

    record_sha = sha256_file(record_path)
    try:
        record = _read_json(record_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {**base, "state": "running_record_write_window",
                "launch_record_sha256": record_sha}
    state = record.get("status")
    if state == "launching":
        return {**base, "state": "running", "launch_record_sha256": record_sha,
                "target_receipt_present": receipt_exists}
    if state != "complete" or record.get("exit_code") != 0:
        return {**base, "state": "launcher_stopped_or_failed_review",
                "launch_record_sha256": record_sha,
                "launcher_status": state, "exit_code": record.get("exit_code")}
    if not receipt_exists:
        return {**base, "state": "terminal_record_missing_receipt_review",
                "launch_record_sha256": record_sha}

    receipt_sha = sha256_file(receipt_path)
    receipt = _read_json(receipt_path)
    expected = item["expected_receipt"]
    if any(receipt.get(key) != value for key, value in expected.items()):
        return {**base, "state": "integrity_error", "detail": "target terminal identity mismatch",
                "launch_record_sha256": record_sha, "receipt_sha256": receipt_sha}
    if (receipt.get("tasks_completed") != 7 or
            receipt.get("model_immutable_versions_unchanged") is not True or
            receipt.get("test_data_included") is not False):
        return {**base, "state": "integrity_error", "detail": "target receipt completion/isolation mismatch",
                "launch_record_sha256": record_sha, "receipt_sha256": receipt_sha}

    target = record.get("target_terminal_artifact")
    if not isinstance(target, Mapping):
        return {**base, "state": "integrity_error", "detail": "launcher lacks target terminal artifact record",
                "launch_record_sha256": record_sha, "receipt_sha256": receipt_sha}
    record_bindings = (
        record.get("schema"), record.get("contract_sha256"), record.get("launcher_sha256"),
        record.get("script_key"), record.get("script_sha256"),
        record.get("config_sha256"), record.get("config_semantic_sha256"),
        record.get("source_lock_sha256"), record.get("training_manifest_sha256"),
        record.get("source_hashes"),
        record.get("memory_fraction_callback_registered_before_cuda_init"),
        record.get("memory_fraction_callback_called"), record.get("memory_fraction_callback_error"),
        record.get("cuda_initialized_at_exit"), record.get("test_data_accessed_by_launcher"))
    expected_bindings = (
        "banking-v10-memory-policy-launch-record-v1", item["job"]["contract_sha256"],
        launcher_sha256, contract["script_key"], contract["script_sha256"],
        contract["config_file_sha256"], contract["config_semantic_sha256"],
        contract["source_lock_sha256"], contract["training_manifest_sha256"],
        contract["source_hashes"],
        True, True, None, True, False)
    if record_bindings != expected_bindings:
        return {**base, "state": "integrity_error", "detail": "launcher record source/config/policy mismatch",
                "launch_record_sha256": record_sha, "receipt_sha256": receipt_sha}
    if (Path(str(target.get("path", ""))).resolve() != receipt_path.resolve() or
            target.get("sha256") != receipt_sha or
            target.get("checkpoint_sha256") != receipt.get("final_checkpoint_sha256")):
        return {**base, "state": "integrity_error", "detail": "target artifact hashes do not match receipt",
                "launch_record_sha256": record_sha, "receipt_sha256": receipt_sha}
    cp_relative = receipt.get("final_checkpoint")
    if not isinstance(cp_relative, str) or Path(cp_relative).is_absolute() or ".." in Path(cp_relative).parts:
        return {**base, "state": "integrity_error", "detail": "invalid final checkpoint path",
                "launch_record_sha256": record_sha, "receipt_sha256": receipt_sha}
    checkpoint = (output / cp_relative).resolve()
    if not _inside(checkpoint, output) or not checkpoint.is_file():
        return {**base, "state": "integrity_error", "detail": "final checkpoint missing/outside run directory",
                "launch_record_sha256": record_sha, "receipt_sha256": receipt_sha}
    # The run receipt is small and is hash-checked; the large checkpoint's
    # claimed SHA is cross-bound but its bytes are reserved for final sealing.
    return {**base, "state": "complete_receipt_verified_checkpoint_bytes_deferred",
            "launch_record_sha256": record_sha, "receipt_sha256": receipt_sha,
            "receipt_status": receipt.get("status"),
            "target_terminal_artifact_sha256": target.get("sha256"),
            "checkpoint_sha256": receipt.get("final_checkpoint_sha256"),
            "checkpoint_file_present": True}


def collect_snapshot(*, queue_path: Path, queue_sha256: str, queue_root: Path,
                     queue_status_sha256: str, first_contract_sha256: str,
                     first_receipt_sha256: str, first_launch_record_sha256: str,
                     output_path: Path, launcher_path: Path | None = None) -> dict:
    queue_path = queue_path.resolve(strict=True)
    queue_root = queue_root.resolve(strict=True)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError("snapshot output must be fresh; refusing overwrite")
    launcher_path = (launcher_path or (Path(__file__).resolve().parents[1] / "tools" /
                     "launch_banking_memory_policy_v10.py")).resolve(strict=True)
    status_path = queue_root / "queue-status.json"
    if not status_path.is_file() or sha256_file(status_path) != queue_status_sha256:
        raise ValueError("original queue status bytes differ from the supplied preserved-status SHA")
    queue, contracts, launcher = _load_queue_contracts(queue_path, queue_sha256,
                                                        queue_root, launcher_path)
    status = _read_json(status_path)
    if (status.get("schema"), status.get("queue_sha256"), status.get("launcher_sha256")) != (
            STATUS_SCHEMA, queue_sha256, queue["launcher_sha256"]):
        raise ValueError("original queue status does not correspond to this queue")
    if contracts[0]["job"]["contract_sha256"] != first_contract_sha256:
        raise ValueError("explicit completed-run contract is not the first original queue job")

    first_record = contracts[0]["record_path"]
    first_receipt = contracts[0]["receipt_path"]
    if not first_record.is_file() or sha256_file(first_record) != first_launch_record_sha256:
        raise ValueError("completed first launch record is missing or differs from explicit SHA")
    if not first_receipt.is_file() or sha256_file(first_receipt) != first_receipt_sha256:
        raise ValueError("completed first target receipt is missing or differs from explicit SHA")
    first = _classify(contracts[0], launcher_sha256=queue["launcher_sha256"])
    if first["state"] != "complete_receipt_verified_checkpoint_bytes_deferred":
        raise ValueError("the preexisting first scientific run is not a verified complete target")

    # The queue and contract hashes preserve the frozen plan. This is a status
    # reader, not another launch preflight; it does not load the model stack.
    run_states = [first]
    for item in contracts[1:]:
        run_states.append(_classify(item, launcher_sha256=queue["launcher_sha256"]))

    if sha256_file(status_path) != queue_status_sha256:
        raise RuntimeError("original queue status changed during read-only collection")
    snapshot = {
        "schema": "banking-v10-readonly-main-queue-snapshot-v1",
        "status": "snapshot_only",
        "queue_path": str(queue_path), "queue_sha256": queue_sha256,
        "queue_root": str(queue_root), "original_queue_status_path": str(status_path),
        "original_queue_status_sha256": queue_status_sha256,
        "original_queue_status": status.get("status"),
        "original_queue_status_preserved": True,
        "launcher_sha256": queue["launcher_sha256"],
        "first_completed_contract_sha256": first_contract_sha256,
        "first_completed_receipt_sha256": first_receipt_sha256,
        "first_completed_launch_record_sha256": first_launch_record_sha256,
        "runs": run_states,
        "counts": {state: sum(run["state"] == state for run in run_states)
                   for state in sorted({run["state"] for run in run_states})},
        "checkpoint_bytes_hashed": False,
        "launch_performed": False,
        "test_payload_read": False,
        "cuda_initialized": False,
        "created_unix": time.time(),
    }
    snapshot["snapshot_path"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    with output_path.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return {**snapshot, "snapshot_file_sha256": sha256_file(output_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--queue-status-sha256", required=True)
    parser.add_argument("--first-contract-sha256", required=True)
    parser.add_argument("--first-receipt-sha256", required=True)
    parser.add_argument("--first-launch-record-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher", type=Path)
    args = parser.parse_args(argv)
    result = collect_snapshot(queue_path=args.queue, queue_sha256=args.queue_sha256,
        queue_root=args.queue_root, queue_status_sha256=args.queue_status_sha256,
        first_contract_sha256=args.first_contract_sha256,
        first_receipt_sha256=args.first_receipt_sha256,
        first_launch_record_sha256=args.first_launch_record_sha256,
        output_path=args.output, launcher_path=args.launcher)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
