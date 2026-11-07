#!/usr/bin/env python3
"""Run one source-pinned Banking V10 command with a process-local CUDA policy.

This wrapper does not edit any V10 source, config, registry, or machine setting.
It verifies the frozen source/config/model/train-dev assets on CPU first, then
sets the allocator environment and queues the memory-fraction setter for the
first CUDA initialization. The wrapped command still owns its normal gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
from typing import Any


SCHEMA = "banking-v10-memory-policy-launch-v1"
ALLOCATOR_POLICY = "garbage_collection_threshold:0.7"
MEMORY_FRACTION = 0.8
REQUIRED_TORCH_VERSION = "2.8.0+cu128"
ALLOWED_SCRIPTS = {
    "run_banking_cil_v10.py",
    "probe_banking_transfer_v10.py",
    "evaluate_banking_test_v10.py",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=True)


def _load_contract(path: str | Path, expected_sha256: str) -> dict:
    contract_path = _resolved(path)
    if sha256_file(contract_path) != expected_sha256:
        raise ValueError("launcher contract SHA-256 mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("schema") != SCHEMA:
        raise ValueError("unsupported launch contract")
    if not isinstance(contract.get("device"), str) or not contract["device"].startswith("cuda:"):
        raise ValueError("memory policy launcher requires an explicit CUDA device")
    required = {
        "schema", "launcher_sha256", "project_root", "task_root", "model_path",
        "training_pack", "config_path", "config_file_sha256", "source_lock_sha256",
        "config_semantic_sha256", "training_manifest_sha256", "source_hashes",
        "script_key", "script_path", "script_sha256", "argv",
        "launch_record_path", "allocator_conf", "memory_fraction", "device", "torch_version",
    }
    if set(contract) != required:
        raise ValueError(f"launch contract keys differ: missing={sorted(required-set(contract))}, "
                         f"extra={sorted(set(contract)-required)}")
    if contract["allocator_conf"] != ALLOCATOR_POLICY or contract["memory_fraction"] != MEMORY_FRACTION:
        raise ValueError("allocator policy is not the reviewed fixed policy")
    if contract["torch_version"] != REQUIRED_TORCH_VERSION:
        raise ValueError("launcher is qualified only for the reviewed PyTorch build")
    if contract["script_key"] not in ALLOWED_SCRIPTS:
        raise ValueError("script is not in the launcher allowlist")
    return contract


def _parse_argv(contract: dict, task_root: Path) -> tuple[Path, list[str]]:
    script_path = _resolved(contract["script_path"])
    expected = (task_root / "tools" / contract["script_key"]).resolve(strict=True)
    if script_path != expected:
        raise ValueError("script path is outside the pinned tools directory")
    if sha256_file(script_path) != contract["script_sha256"]:
        raise ValueError("wrapped script SHA-256 mismatch")
    argv = contract["argv"]
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        raise ValueError("contract argv must be a nonempty list of strings")
    # All supported commands must execute a work/evaluation phase, never a
    # preflight-only command that might be misreported as a launched run.
    for flag in ("--phase", "--mode"):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 >= len(argv) or argv[index + 1] == "preflight":
                raise ValueError("preflight-only commands cannot use the GPU policy launcher")
    if contract["script_key"] == "run_banking_cil_v10.py":
        if _flag_value(argv, "--phase") not in ("foundation", "main"):
            raise ValueError("V10 driver launcher permits foundation or main only")
    elif contract["script_key"] == "probe_banking_transfer_v10.py":
        if _flag_value(argv, "--phase") not in (None, "run"):
            raise ValueError("transfer probe must use its run phase")
    elif contract["script_key"] == "evaluate_banking_test_v10.py":
        if _flag_value(argv, "--mode") != "evaluate":
            raise ValueError("test evaluator launcher must use evaluate mode")
    return script_path, argv


def _flag_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"missing value after {flag}")
    return argv[index + 1]


def _verify_argv_bindings(contract: dict, project_root: Path, model_path: Path,
                          training_pack: Path, config_path: Path) -> None:
    argv = contract["argv"]
    expected_values = {
        "--project-root": str(project_root),
        "--model-path": str(model_path),
        "--model-dir": str(model_path),
        "--training-pack": str(training_pack),
        "--training-manifest": str(training_pack),
        "--training-root": str(training_pack.parent),
        "--config": str(config_path),
    }
    for flag, expected in expected_values.items():
        if flag in argv and str(_resolved(_flag_value(argv, flag))) != expected:
            raise ValueError(f"argv {flag} does not match the pinned path")
    required_flags = {
        "run_banking_cil_v10.py": ("--project-root", "--model-path", "--training-pack", "--config"),
        "probe_banking_transfer_v10.py": ("--project-root", "--model-path", "--training-pack", "--config"),
        "evaluate_banking_test_v10.py": ("--project-root", "--model-dir", "--training-manifest", "--config"),
    }[contract["script_key"]]
    absent = [flag for flag in required_flags if flag not in argv]
    if absent:
        raise ValueError(f"wrapped command is missing required pinned path arguments: {absent}")


def _verify_frozen_inputs(contract: dict) -> dict:
    project_root = _resolved(contract["project_root"])
    task_root = _resolved(contract["task_root"])
    model_path = _resolved(contract["model_path"])
    training_pack = _resolved(contract["training_pack"])
    config_path = _resolved(contract["config_path"])
    launcher_path = Path(__file__).resolve(strict=True)
    if sha256_file(launcher_path) != contract["launcher_sha256"]:
        raise ValueError("launcher SHA-256 mismatch")
    if not project_root.is_dir() or not model_path.is_dir() or not training_pack.is_file():
        raise FileNotFoundError("pinned project/model/train-dev inputs are unavailable")
    if sha256_file(config_path) != contract["config_file_sha256"]:
        raise ValueError("frozen config file SHA-256 mismatch")
    script_path, argv = _parse_argv(contract, task_root)
    _verify_argv_bindings(contract, project_root, model_path, training_pack, config_path)
    output_dir = _output_dir_from_contract(contract)
    record_path = Path(contract["launch_record_path"]).expanduser().resolve()
    if output_dir == record_path.parent or output_dir in record_path.parents:
        raise ValueError("launcher record may not be inside the target run directory")
    for target in (output_dir, record_path):
        for protected in (project_root, model_path, training_pack.parent):
            if target == protected or protected in target.parents:
                raise ValueError("output/receipt may not be inside source, model, or data trees")
    if output_dir.exists() or record_path.exists():
        raise FileExistsError("target output or launcher receipt already exists")

    # Import only the frozen driver and its stdlib/data helpers here. This path
    # runs before importing torch and uses the same source/model/manifest gates
    # as the driver itself.
    tools_dir = task_root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import run_banking_cil_v10 as driver

    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    phase = _flag_value(argv, "--phase")
    driver_phase = phase if contract["script_key"] == "run_banking_cil_v10.py" else None
    cfg = driver.validate_config(raw_cfg, phase=driver_phase)
    if cfg.get("status") != "frozen":
        raise ValueError("candidate/missing config cannot launch")
    if cfg.get("device") != contract["device"]:
        raise ValueError("contract device differs from frozen config")
    if driver._config_sha(cfg) != contract["config_semantic_sha256"]:
        raise ValueError("frozen semantic config SHA-256 mismatch")
    if cfg.get("training_manifest_sha256") != contract["training_manifest_sha256"]:
        raise ValueError("contract train/dev manifest SHA-256 differs from frozen config")
    if cfg.get("source_hashes") != contract["source_hashes"]:
        raise ValueError("contract source hash map differs from frozen config")
    hashes, source_lock_sha, manifest, train_rows, dev_rows = driver._source_prepare(
        project_root, task_root, model_path, training_pack, cfg)
    if source_lock_sha != contract["source_lock_sha256"]:
        raise ValueError("frozen source lock SHA-256 mismatch")
    return {
        "driver": driver,
        "cfg": cfg,
        "source_hashes": hashes,
        "source_lock_sha256": source_lock_sha,
        "project_root": project_root,
        "task_root": task_root,
        "model_path": model_path,
        "training_pack": training_pack,
        "config_path": config_path,
        "config_sha256": sha256_file(config_path),
        "config_semantic_sha256": driver._config_sha(cfg),
        "manifest_sha256": cfg["training_manifest_sha256"],
        "train_rows": len(train_rows),
        "development_rows": len(dev_rows),
        "script_path": script_path,
        "output_dir": output_dir,
        "launch_record_path": record_path,
        "argv": argv,
        "launcher_sha256": sha256_file(launcher_path),
        "phase": phase or _flag_value(argv, "--mode"),
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        raise FileExistsError(tmp)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    if sha256_file(path) != hashlib.sha256(raw.encode("utf-8")).hexdigest():
        raise IOError("launch receipt readback hash mismatch")


def _replace_json(path: Path, value: dict) -> None:
    """Atomically replace a file owned by this wrapper (e.g. queue status)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    if sha256_file(path) != hashlib.sha256(raw.encode("utf-8")).hexdigest():
        raise IOError("atomic JSON readback hash mismatch")


def _install_memory_policy(torch_module, device: str) -> dict:
    if torch_module.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the memory policy could be queued")
    state = {"registered": False, "called": False, "error": None}

    def apply_fraction_on_cuda_init():
        state["called"] = True
        try:
            torch_module.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device=device)
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            raise

    # PyTorch _lazy_init marks its thread-local init guard before draining the
    # queue, so set_per_process_memory_fraction's nested _lazy_init is a no-op.
    torch_module.cuda._lazy_call(apply_fraction_on_cuda_init)
    state["registered"] = True
    if torch_module.cuda.is_initialized():
        raise RuntimeError("queueing the CUDA policy unexpectedly initialized CUDA")
    return state


def _output_dir_from_contract(contract: dict) -> Path:
    argv = contract["argv"]
    value = _flag_value(argv, "--output-dir") or _flag_value(argv, "--out-dir")
    if value is None:
        raise ValueError("queue job must have a pinned output directory")
    return Path(value).expanduser().resolve()


def _receipt_for_contract(contract: dict) -> tuple[Path, dict]:
    argv = contract["argv"]
    script = contract["script_key"]
    output_dir = _output_dir_from_contract(contract)
    if script == "run_banking_cil_v10.py":
        phase = _flag_value(argv, "--phase")
        if phase == "foundation":
            return output_dir / "foundation-receipt.json", {
                "schema": "banking-cil-v10-foundation-receipt", "status": "qualified",
                "seed": int(_flag_value(argv, "--seed")), "arm": "FOUNDATION_CMS_ONLY",
                "task_id": 0,
            }
        if phase == "main":
            return output_dir / "run-receipt.json", {
                "schema": "banking-cil-v10-run-receipt", "status": "complete",
                "seed": int(_flag_value(argv, "--seed")), "arm": _flag_value(argv, "--arm"),
                "tasks_completed": 7,
            }
    raise ValueError("serial queue currently permits only frozen V10 foundation/main jobs")


def _validate_target_terminal(contract: dict, validated: dict) -> dict:
    """Require a successful, source-bound target artifact before marking launch complete."""
    output_dir = validated["output_dir"]
    expected_common = {
        "config_sha256": validated["config_semantic_sha256"],
        "source_lock_sha256": validated["source_lock_sha256"],
        "training_manifest_sha256": validated["manifest_sha256"],
    }
    if contract["script_key"] == "run_banking_cil_v10.py":
        phase = _flag_value(contract["argv"], "--phase")
        seed_value = _flag_value(contract["argv"], "--seed")
        if seed_value is None:
            raise ValueError("training launch contract must pin its seed")
        seed = int(seed_value)
        if phase == "foundation":
            path = output_dir / "foundation-receipt.json"
            receipt = json.loads(path.read_text(encoding="utf-8"))
            required = {**expected_common, "schema": "banking-cil-v10-foundation-receipt",
                        "status": "qualified", "seed": seed, "arm": "FOUNDATION_CMS_ONLY",
                        "task_id": 0, "model_immutable_versions_unchanged": True,
                        "test_data_included": False}
            _assert_fields(receipt, required, "foundation receipt")
            _verify_receipt_checkpoint(output_dir, receipt, "checkpoint", "checkpoint_sha256")
            return {"artifact_path": str(path), "artifact_sha256": sha256_file(path),
                    "checkpoint_sha256": receipt["checkpoint_sha256"], "receipt": receipt}
        if phase == "main":
            path = output_dir / "run-receipt.json"
            receipt = json.loads(path.read_text(encoding="utf-8"))
            required = {**expected_common, "schema": "banking-cil-v10-run-receipt",
                        "status": "complete", "seed": seed,
                        "arm": _flag_value(contract["argv"], "--arm"),
                        "tasks_completed": 7, "model_immutable_versions_unchanged": True,
                        "test_data_included": False}
            _assert_fields(receipt, required, "main run receipt")
            _verify_receipt_checkpoint(output_dir, receipt, "final_checkpoint", "final_checkpoint_sha256")
            return {"artifact_path": str(path), "artifact_sha256": sha256_file(path),
                    "checkpoint_sha256": receipt["final_checkpoint_sha256"], "receipt": receipt}
    if contract["script_key"] == "probe_banking_transfer_v10.py":
        path = output_dir / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        required = {"schema": "banking-cil-v10-transfer-probe-v1", "status": "complete",
                    "seed": 17, "task_id": 0, "development_only": True, "test_data_read": False,
                    "banking_driver_source_lock_sha256": validated["source_lock_sha256"],
                    "config_sha256": validated["config_semantic_sha256"],
                    "training_manifest_sha256": validated["manifest_sha256"],
                    "model_immutable_versions_unchanged": True}
        _assert_fields(result, required, "transfer probe result")
        expected_arms = {"SLEEP_ADDITIVE", "GOLD_ADDITIVE", "SLEEP_FIXEDMIX", "GOLD_FIXEDMIX"}
        progress_path = output_dir / "progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (set(result.get("arms", {})) != expected_arms or progress.get("status") != "complete" or
            set(progress.get("completed_arms", [])) != expected_arms):
            raise ValueError("transfer probe did not finish its fixed 30-call four-arm contract")
        for name, arm in result["arms"].items():
            if (arm.get("opportunity_count") != 30 or
                "30" not in arm.get("actual_stage_metrics", {})):
                raise ValueError(f"probe arm {name} lacks the fixed 30-opportunity endpoint")
            if name == "SLEEP_ADDITIVE":
                parity = arm.get("disk_resume_equivalence", {})
                if (parity.get("exact_model_adam_tx_rng_process_rng") is not True or
                    parity.get("continuous_snapshot_sha256") != parity.get("restored_snapshot_sha256")):
                    raise ValueError("additive Sleep arm lacks exact 3-to-6 disk recovery parity")
        argv = contract["argv"]
        foundation_dir = _resolved(_flag_value(argv, "--foundation-dir"))
        foundation_path = foundation_dir / "foundation-receipt.json"
        foundation = json.loads(foundation_path.read_text(encoding="utf-8"))
        if (foundation.get("status"), foundation.get("seed"), foundation.get("arm"),
            foundation.get("source_lock_sha256"), foundation.get("config_sha256"),
            foundation.get("training_manifest_sha256"), foundation.get("checkpoint_sha256"),
            foundation.get("test_data_included"), foundation.get("model_immutable_versions_unchanged"),
            result.get("foundation_receipt_sha256"), result.get("foundation_checkpoint_sha256")) != (
                "qualified", 17, "FOUNDATION_CMS_ONLY", validated["source_lock_sha256"],
                validated["config_semantic_sha256"], validated["manifest_sha256"],
                foundation.get("checkpoint_sha256"), False, True, sha256_file(foundation_path),
                foundation.get("checkpoint_sha256")):
            raise ValueError("transfer probe result is not bound to the qualified seed-17 foundation")
        cp_path = (foundation_dir / foundation["checkpoint"]).resolve()
        if foundation_dir not in cp_path.parents or sha256_file(cp_path) != foundation["checkpoint_sha256"]:
            raise ValueError("transfer probe foundation checkpoint bytes changed")
        return {"artifact_path": str(path), "artifact_sha256": sha256_file(path),
                "checkpoint_sha256": result["foundation_checkpoint_sha256"], "receipt": result}
    if contract["script_key"] == "evaluate_banking_test_v10.py":
        path = output_dir / "evaluation.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        training = result.get("training", {})
        required = {"schema": "banking-cil-v10-test-evaluator-v1", "status": "complete",
                    "optimizer_updates": 0, "all_contexts_frozen": True, "backbone_unchanged": True}
        _assert_fields(result, required, "official test evaluation")
        for key, value in expected_common.items():
            if training.get(key) != value:
                raise ValueError(f"evaluation training binding mismatch: {key}")
        if training.get("training_manifest_sha256") != validated["manifest_sha256"]:
            raise ValueError("evaluation manifest binding mismatch")
        return {"artifact_path": str(path), "artifact_sha256": sha256_file(path), "receipt": result}
    raise ValueError("no terminal receipt validator for wrapped script")


def _assert_fields(value: dict, expected: dict, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"{label} field {key!r} mismatch")


def _verify_receipt_checkpoint(output_dir: Path, receipt: dict, path_key: str, sha_key: str) -> None:
    relative = receipt.get(path_key)
    expected_sha = receipt.get(sha_key)
    if not isinstance(relative, str) or not isinstance(expected_sha, str):
        raise ValueError("terminal receipt lacks checkpoint path/hash")
    checkpoint = (output_dir / relative).resolve()
    if output_dir.resolve() not in checkpoint.parents or not checkpoint.is_file():
        raise ValueError("terminal checkpoint path escapes output or is missing")
    if sha256_file(checkpoint) != expected_sha:
        raise ValueError("terminal checkpoint bytes differ from receipt")


def _child_command(item: dict, launcher_path: str | Path, python_executable: str | None = None) -> list[str]:
    return [python_executable or sys.executable, str(Path(launcher_path).resolve()), "launch",
            "--contract", item["contract_path"], "--contract-sha256", item["contract_sha256"]]


def validate_queue(queue: dict, queue_root: str | Path, *, task_root: str | Path) -> list[dict]:
    """Validate all finite queue contracts before creating output or a child."""
    if not isinstance(queue, dict) or set(queue) != {
        "schema", "launcher_sha256", "jobs", "minimum_free_before_start_bytes"
    } or queue.get("schema") != "banking-v10-memory-policy-queue-v1":
        raise ValueError("unsupported queue schema/fields")
    launcher_sha = sha256_file(Path(__file__).resolve())
    if queue["launcher_sha256"] != launcher_sha:
        raise ValueError("queue launcher SHA-256 mismatch")
    jobs = queue["jobs"]
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 9:
        raise ValueError("finite queue must contain one to nine jobs")
    minimum_free = queue["minimum_free_before_start_bytes"]
    if type(minimum_free) is not int or minimum_free < 0:
        raise ValueError("minimum free bytes must be a nonnegative integer")
    root = Path(queue_root).expanduser().resolve()
    task = _resolved(task_root)
    if root.exists():
        raise FileExistsError("queue root must be fresh")
    if not root.parent.is_dir():
        raise FileNotFoundError("queue root parent must already exist")
    prepared = []
    outputs, records = set(), set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or set(job) != {"contract_path", "contract_sha256"}:
            raise ValueError(f"queue job {index} has invalid fields")
        contract = _load_contract(job["contract_path"], job["contract_sha256"])
        _verify_frozen_inputs(contract)
        output = _output_dir_from_contract(contract)
        record = Path(contract["launch_record_path"]).expanduser().resolve()
        if root not in output.parents or root not in record.parents:
            raise ValueError("every job output and launcher receipt must be under the fresh queue root")
        if output.exists() or record.exists():
            raise FileExistsError("queue output or launcher receipt already exists")
        receipt, expected = _receipt_for_contract(contract)
        if receipt.exists() or not (root == receipt.parent or root in receipt.parents):
            raise ValueError("wrapped command receipt must be a fresh child of queue root")
        if output in outputs or record in records:
            raise ValueError("queue jobs must have unique output and launcher receipt paths")
        outputs.add(output); records.add(record)
        prepared.append({"contract_path": str(_resolved(job["contract_path"])),
                         "contract_sha256": job["contract_sha256"], "contract": contract,
                         "output_dir": str(output), "launch_record": str(record),
                         "receipt_path": str(receipt), "expected_receipt": expected})
    if shutil.disk_usage(root.parent).free < minimum_free:
        raise RuntimeError("insufficient disk space for finite queue")
    return prepared


def launch_queue(queue_path: str | Path, expected_queue_sha256: str) -> int:
    queue_file = _resolved(queue_path)
    if sha256_file(queue_file) != expected_queue_sha256:
        raise ValueError("queue file SHA-256 mismatch")
    queue = json.loads(queue_file.read_text(encoding="utf-8"))
    queue_root = Path(queue.get("queue_root", ""))
    # Keep queue_root external to the queue JSON's hashed fields by requiring
    # the root explicitly in the queue object and checking its type here.
    if set(queue) != {"schema", "launcher_sha256", "queue_root", "jobs", "minimum_free_before_start_bytes"}:
        raise ValueError("queue fields differ from the required finite queue schema")
    queue_for_validation = {k: v for k, v in queue.items() if k != "queue_root"}
    task_root = Path(__file__).resolve().parents[1]
    prepared = validate_queue(queue_for_validation, queue_root, task_root=task_root)
    queue_env = os.environ.copy()
    prior_allocator = queue_env.get("PYTORCH_CUDA_ALLOC_CONF")
    if prior_allocator not in (None, ALLOCATOR_POLICY):
        raise ValueError("conflicting process allocator setting; refusing to overwrite it")
    root = queue_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / "queue-status.json"
    status = {"schema": "banking-v10-memory-policy-queue-status-v1", "status": "running",
              "queue_sha256": expected_queue_sha256, "launcher_sha256": sha256_file(Path(__file__).resolve()),
              "pid": os.getpid(), "automatic_retries": 0, "jobs": [], "started_unix": time.time(),
              "test_data_accessed_by_queue": False}
    _replace_json(status_path, status)
    queue_env["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR_POLICY
    queue_env["PYTHONUNBUFFERED"] = "1"
    for index, item in enumerate(prepared):
        # Recheck the source/config/model/data binding immediately before each
        # child; this parent process never imports torch or starts CUDA.
        contract = _load_contract(item["contract_path"], item["contract_sha256"])
        label = f"job-{index + 1:02d}-{contract['script_key']}"
        job_record = {"job": label, "status": "running", "output_dir": item["output_dir"],
                      "contract_sha256": item["contract_sha256"], "started_unix": time.time()}
        status["jobs"].append(job_record); _replace_json(status_path, status)
        stdout_path, stderr_path = root / f"{label}.stdout.log", root / f"{label}.stderr.log"
        cmd = _child_command(item, Path(__file__).resolve())
        started = time.monotonic()
        try:
            _verify_frozen_inputs(contract)
            with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
                child = subprocess.run(cmd, cwd=task_root, env=queue_env, stdout=stdout, stderr=stderr,
                                       check=False)
            job_record["returncode"] = child.returncode
            job_record["elapsed_wall_seconds"] = time.monotonic() - started
            if child.returncode != 0:
                raise RuntimeError(f"child returned {child.returncode}")
            receipt_path = Path(item["receipt_path"])
            if not receipt_path.is_file():
                raise RuntimeError("target receipt is missing")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected = item["expected_receipt"]
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise RuntimeError("target receipt does not match its expected terminal identity")
            contract = item["contract"]
            if (receipt.get("config_sha256"), receipt.get("source_lock_sha256"),
                receipt.get("training_manifest_sha256")) != (
                    contract["config_semantic_sha256"], contract["source_lock_sha256"],
                    contract["training_manifest_sha256"]):
                raise RuntimeError("target receipt source/config/data binding mismatch")
            launch_record_path = Path(item["launch_record"])
            launch_record = json.loads(launch_record_path.read_text(encoding="utf-8"))
            if (launch_record.get("status"), launch_record.get("exit_code"),
                launch_record.get("memory_fraction_callback_registered_before_cuda_init"),
                launch_record.get("memory_fraction_callback_called"),
                launch_record.get("source_lock_sha256"), launch_record.get("config_sha256")) != (
                    "complete", 0, True, True, contract["source_lock_sha256"],
                    contract["config_semantic_sha256"]):
                raise RuntimeError("launcher receipt does not prove policy application and frozen bindings")
            job_record.update(status="complete", receipt_path=str(receipt_path),
                              receipt_sha256=sha256_file(receipt_path), receipt_status=receipt.get("status"),
                              launch_record_sha256=sha256_file(launch_record_path))
            _replace_json(status_path, status)
        except BaseException as exc:
            job_record.update(status="stopped", error=f"{type(exc).__name__}: {exc}",
                              elapsed_wall_seconds=time.monotonic() - started)
            status.update(status="stopped", ended_unix=time.time())
            _replace_json(status_path, status)
            raise
    status.update(status="complete", ended_unix=time.time())
    _replace_json(status_path, status)
    return 0


def launch(contract_path: str | Path, expected_contract_sha256: str) -> int:
    contract = _load_contract(contract_path, expected_contract_sha256)
    record_path = Path(contract["launch_record_path"]).expanduser().resolve()
    if record_path.exists():
        raise FileExistsError(record_path)
    # All source, config, model, and data checks are completed before torch is
    # imported, CUDA policy is set, or runpy can invoke the target script.
    validated = _verify_frozen_inputs(contract)
    allocator_before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if allocator_before not in (None, ALLOCATOR_POLICY):
        raise ValueError("conflicting process allocator setting; refusing to overwrite it")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR_POLICY

    import torch
    if torch.__version__ != contract["torch_version"]:
        raise RuntimeError(f"PyTorch build mismatch: expected {contract['torch_version']}, got {torch.__version__}")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the memory policy could be queued")
    # Recheck immediately before registration/target execution to close the
    # CPU-preflight-to-launch gap without invoking any CUDA operation.
    validated = _verify_frozen_inputs(contract)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized while rechecking launch inputs")
    callback_state = _install_memory_policy(torch, validated["cfg"]["device"])

    started_wall = time.monotonic()
    record = {
        "schema": "banking-v10-memory-policy-launch-record-v1",
        "status": "launching",
        "contract_sha256": expected_contract_sha256,
        "launcher_sha256": validated["launcher_sha256"],
        "script_key": contract["script_key"],
        "script_sha256": contract["script_sha256"],
        "config_sha256": validated["config_sha256"],
        "config_semantic_sha256": validated["config_semantic_sha256"],
        "source_lock_sha256": validated["source_lock_sha256"],
        "source_hashes": validated["source_hashes"],
        "training_manifest_sha256": validated["manifest_sha256"],
        "model_revision": validated["cfg"]["model_revision"],
        "torch_version": torch.__version__,
        "allocator_conf": ALLOCATOR_POLICY,
        "allocator_conf_before": allocator_before,
        "memory_fraction": MEMORY_FRACTION,
        "memory_fraction_callback_registered_before_cuda_init": callback_state["registered"],
        "cuda_initialized_before_target": False,
        "memory_fraction_callback_called": False,
        "device": validated["cfg"]["device"],
        "argv": validated["argv"],
        "phase": validated["phase"],
        "train_rows": validated["train_rows"],
        "development_rows": validated["development_rows"],
        "test_data_accessed_by_launcher": False,
        "automatic_retries": 0,
        "started_unix": time.time(),
    }
    # Pre-create the immutable target via exclusive open. The target command
    # writes its own separate output; this receipt is not a source permission.
    _atomic_json(record_path, record)
    exit_code = 0
    caught: BaseException | None = None
    try:
        try:
            sys.argv = [str(validated["script_path"]), *validated["argv"]]
            runpy.run_path(str(validated["script_path"]), run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            exit_code = int(code) if isinstance(code, int) else (0 if code is None else 1)
            if exit_code:
                caught = exc
        except BaseException as exc:
            exit_code = 1
            caught = exc
        if caught is None and exit_code == 0:
            try:
                target_artifact = _validate_target_terminal(contract, validated)
                if (not callback_state["called"] or callback_state["error"] is not None or
                    not torch.cuda.is_initialized()):
                    raise RuntimeError("target completed without applying the queued CUDA memory policy")
                record["target_terminal_artifact"] = {
                    "path": target_artifact["artifact_path"],
                    "sha256": target_artifact["artifact_sha256"],
                    "checkpoint_sha256": target_artifact.get("checkpoint_sha256"),
                }
            except BaseException as exc:
                exit_code = 1
                caught = exc
    finally:
        record.update({
            "status": "complete" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "elapsed_wall_seconds": time.monotonic() - started_wall,
            "memory_fraction_callback_called": callback_state["called"],
            "memory_fraction_callback_error": callback_state["error"],
            "cuda_initialized_at_exit": bool(torch.cuda.is_initialized()),
            "ended_unix": time.time(),
        })
        # Atomic replace is allowed only for this launcher's own record.
        tmp = record_path.with_name(record_path.name + ".update.partial")
        final_raw = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(final_raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, record_path)
        if sha256_file(record_path) != hashlib.sha256(final_raw.encode("utf-8")).hexdigest():
            raise IOError("updated launch record readback hash mismatch")
    if caught is not None:
        raise caught
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("launch")
    one.add_argument("--contract", type=Path, required=True)
    one.add_argument("--contract-sha256", required=True)
    queue = sub.add_parser("queue")
    queue.add_argument("--queue", type=Path, required=True)
    queue.add_argument("--queue-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "launch":
        return launch(args.contract, args.contract_sha256)
    return launch_queue(args.queue, args.queue_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
