#!/usr/bin/env python3
"""Run the frozen nine-job Banking V10 main queue with corrected receipt checks.

This parent-only runner is for future reproducible runs. It reuses the frozen
c353 single-child launcher and does not modify it or the original queue JSON.
Each child runs serially beneath a fresh queue root; a failure stops the queue
without retries. The current manual continuation is not launched through this
helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable, Mapping


STATUS_NAME = "checked-queue-status.json"
QUEUE_SCHEMA = "banking-v10-memory-policy-queue-v1"
EXPECTED_GRID = {(seed, arm) for seed in (17, 29, 43)
                 for arm in ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        raise FileExistsError(tmp)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")
    with tmp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)
    if sha256_file(path) != hashlib.sha256(raw).hexdigest():
        raise IOError("checked queue status readback hash mismatch")


def validate_nine_job_queue(queue: Mapping, prepared: list[dict], root: Path) -> None:
    """Narrow the frozen launcher's 1..9 generic queue to the exact main grid."""
    if queue.get("schema") != QUEUE_SCHEMA or len(prepared) != 9:
        raise ValueError("future reproducible queue must have the frozen nine-job schema")
    observed = set()
    for item in prepared:
        contract = item["contract"]
        argv = contract["argv"]
        if contract["script_key"] != "run_banking_cil_v10.py":
            raise ValueError("checked queue permits only frozen main-training children")
        if "--phase" not in argv or argv[argv.index("--phase") + 1] != "main":
            raise ValueError("every queue child must be a main-phase run")
        seed = int(argv[argv.index("--seed") + 1])
        arm = argv[argv.index("--arm") + 1]
        pair = (seed, arm)
        if pair in observed or pair not in EXPECTED_GRID:
            raise ValueError("queue has duplicate or unexpected seed/arm")
        observed.add(pair)
        output = Path(item["output_dir"]).resolve()
        record = Path(item["launch_record"]).resolve()
        if output.parent != root or root not in record.parents:
            raise ValueError("each child output must be a direct queue-root child")
        if output.exists() or record.exists():
            raise FileExistsError("fresh queue child output/record already exists")
    if observed != EXPECTED_GRID:
        raise ValueError("queue does not contain the exact 3-seed x 3-arm grid")


def validate_success_record(*, contract: Mapping, contract_sha256: str,
                            record: Mapping, target: Mapping,
                            expected_model_revision: str) -> dict:
    """Validate c353 record using distinct raw-file and canonical config hashes."""
    expected = {
        "schema": "banking-v10-memory-policy-launch-record-v1",
        "status": "complete",
        "exit_code": 0,
        "contract_sha256": contract_sha256,
        "launcher_sha256": contract["launcher_sha256"],
        "script_key": contract["script_key"],
        "script_sha256": contract["script_sha256"],
        "config_sha256": contract["config_file_sha256"],
        "config_semantic_sha256": contract["config_semantic_sha256"],
        "source_lock_sha256": contract["source_lock_sha256"],
        "source_hashes": contract["source_hashes"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "model_revision": expected_model_revision,
        "allocator_conf": contract["allocator_conf"],
        "memory_fraction": contract["memory_fraction"],
        "memory_fraction_callback_registered_before_cuda_init": True,
        "memory_fraction_callback_called": True,
        "memory_fraction_callback_error": None,
        "cuda_initialized_before_target": False,
        "cuda_initialized_at_exit": True,
        "test_data_accessed_by_launcher": False,
        "automatic_retries": 0,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"launcher record mismatch: {key}")
    receipt = target["receipt"]
    expected_terminal = {
        "path": str(Path(target["artifact_path"]).resolve()),
        "sha256": sha256_file(target["artifact_path"]),
        "checkpoint_sha256": target.get("checkpoint_sha256"),
    }
    if record.get("target_terminal_artifact") != expected_terminal:
        raise ValueError("launcher terminal artifact binding differs from verified target receipt")
    if receipt.get("test_data_included") is not False:
        raise ValueError("training receipt must affirm official test exclusion")
    return {"receipt_sha256": expected_terminal["sha256"],
            "checkpoint_sha256": expected_terminal["checkpoint_sha256"],
            "receipt_status": receipt.get("status"),
            "seed": receipt.get("seed"), "arm": receipt.get("arm")}


def verify_completed_child(item: Mapping, *, launcher, validated: Mapping,
                           contract_sha256: str) -> dict:
    """Use c353 terminal verifier, then validate its launcher record exactly."""
    target = launcher._validate_target_terminal(item["contract"], dict(validated))
    record_path = Path(item["launch_record"])
    record = _read_json(record_path)
    summary = validate_success_record(contract=item["contract"],
                                      contract_sha256=contract_sha256,
                                      record=record, target=target,
                                      expected_model_revision=validated["cfg"]["model_revision"])
    summary["launch_record_sha256"] = sha256_file(record_path)
    return summary


def _run_serially(items: list[dict], run_one: Callable) -> list[dict]:
    """Run one callback at a time and stop immediately on its first exception."""
    results = []
    for item in items:
        results.append(run_one(item))
    return results


def _child_environment(parent_env: Mapping[str, str], allocator_policy: str) -> dict[str, str]:
    prior = parent_env.get("PYTORCH_CUDA_ALLOC_CONF")
    if prior not in (None, allocator_policy):
        raise ValueError("conflicting process allocator setting; refusing to overwrite it")
    env = dict(parent_env)
    env["PYTORCH_CUDA_ALLOC_CONF"] = allocator_policy
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _write_status(root: Path, status: dict) -> None:
    path = root / STATUS_NAME
    if path.exists():
        # Later updates are written through an exclusive, unique partial name.
        raise FileExistsError(path)
    _atomic_json(path, status)


def _replace_owned_status(root: Path, status: dict) -> None:
    path = root / STATUS_NAME
    tmp = root / (STATUS_NAME + ".update.partial")
    if tmp.exists():
        raise FileExistsError(tmp)
    raw = (json.dumps(status, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")
    with tmp.open("xb") as stream:
        stream.write(raw); stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)
    if sha256_file(path) != hashlib.sha256(raw).hexdigest():
        raise IOError("checked queue status update readback mismatch")


def run_checked_queue(queue_path: str | Path, expected_queue_sha256: str, *,
                      launcher_path: str | Path | None = None,
                      task_root: str | Path | None = None,
                      python_executable: str | None = None,
                      popen_run: Callable = subprocess.run) -> dict:
    queue_file = Path(queue_path).expanduser().resolve(strict=True)
    if sha256_file(queue_file) != expected_queue_sha256:
        raise ValueError("queue file SHA-256 mismatch")
    queue = _read_json(queue_file)
    task = Path(task_root).expanduser().resolve(strict=True) if task_root else Path(__file__).resolve().parents[1]
    launcher_file = (Path(launcher_path).expanduser().resolve(strict=True) if launcher_path
                     else (task / "tools" / "launch_banking_memory_policy_v10.py").resolve(strict=True))
    if launcher_file != (task / "tools" / "launch_banking_memory_policy_v10.py").resolve(strict=True):
        raise ValueError("queue must reuse the frozen c353 launcher in task_root/tools")
    if queue.get("launcher_sha256") != sha256_file(launcher_file):
        raise ValueError("queue does not pin the frozen c353 launcher bytes")
    root = Path(queue.get("queue_root", "")).expanduser().resolve()
    if set(queue) != {"schema", "launcher_sha256", "queue_root", "jobs", "minimum_free_before_start_bytes"}:
        raise ValueError("unexpected frozen queue fields")
    import importlib.util
    spec = importlib.util.spec_from_file_location("frozen_banking_v10_launcher", launcher_file)
    if spec is None or spec.loader is None:
        raise ImportError("could not load frozen c353 launcher")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    queue_for_validation = {k: v for k, v in queue.items() if k != "queue_root"}
    prepared = launcher.validate_queue(queue_for_validation, root, task_root=task)
    validate_nine_job_queue(queue, prepared, root)
    if shutil.disk_usage(root.parent).free < queue["minimum_free_before_start_bytes"]:
        raise RuntimeError("insufficient free space before checked queue start")
    import os
    env = _child_environment(os.environ, launcher.ALLOCATOR_POLICY)
    root.mkdir(parents=False, exist_ok=False)
    status = {"schema": "banking-v10-checked-queue-status-v1", "status": "running",
              "queue_path": str(queue_file), "queue_sha256": expected_queue_sha256,
              "launcher_sha256": sha256_file(launcher_file), "pid": os.getpid(),
              "automatic_retries": 0, "jobs": [], "started_unix": time.time(),
              "test_data_accessed_by_parent": False}
    _write_status(root, status)
    current_record = None
    def run_one(item):
        nonlocal current_record
        index = prepared.index(item)
        if shutil.disk_usage(root.parent).free < queue["minimum_free_before_start_bytes"]:
            raise RuntimeError("free-space reserve fell below the configured floor")
        contract_path = Path(item["contract_path"])
        contract = launcher._load_contract(contract_path, item["contract_sha256"])
        validated = launcher._verify_frozen_inputs(contract)
        record = {"job_index": index, "seed": int(contract["argv"][contract["argv"].index("--seed") + 1]),
                  "arm": contract["argv"][contract["argv"].index("--arm") + 1],
                  "status": "running", "contract_sha256": item["contract_sha256"],
                  "output_dir": item["output_dir"], "started_unix": time.time()}
        current_record = record
        status["jobs"].append(record)
        _replace_owned_status(root, status)
        stdout_path, stderr_path = root / f"job-{index+1:02d}.stdout.log", root / f"job-{index+1:02d}.stderr.log"
        command = launcher._child_command(item, launcher_file, python_executable)
        started = time.monotonic()
        with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
            child = popen_run(command, cwd=task, env=env, stdout=stdout, stderr=stderr, check=False)
        record["returncode"] = child.returncode
        record["elapsed_wall_seconds"] = time.monotonic() - started
        if child.returncode != 0:
            raise RuntimeError(f"child process returned {child.returncode}")
        verified = verify_completed_child(item, launcher=launcher, validated=validated,
                                          contract_sha256=item["contract_sha256"])
        record.update(status="complete", **verified, ended_unix=time.time())
        _replace_owned_status(root, status)
        current_record = None
        return record

    try:
        _run_serially(prepared, run_one)
        status.update(status="complete", ended_unix=time.time())
        _replace_owned_status(root, status)
    except BaseException as exc:
        if current_record is not None:
            current_record.update(status="stopped", error=f"{type(exc).__name__}: {exc}",
                                  ended_unix=time.time())
        status.update(status="stopped", ended_unix=time.time(),
                      error=f"{type(exc).__name__}: {exc}")
        # Do not overwrite the last durable status if a partial update is present.
        _replace_owned_status(root, status)
        raise
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--task-root", type=Path)
    args = parser.parse_args(argv)
    result = run_checked_queue(args.queue, args.queue_sha256,
                               launcher_path=args.launcher, task_root=args.task_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
