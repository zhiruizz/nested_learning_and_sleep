"""Prepare and serially launch the frozen six-run Banking V11 grid.

Subcommands are intentionally explicit: ``prepare`` verifies the completed
V10 official-test aggregate and builds six one-run contracts; ``run`` launches
only fresh jobs, one child process at a time, and stops on the first failure;
``prepare-dev-eval`` requires all six complete jobs and builds (but does not
launch) the strict 15-context development reference/contract. This tool never
opens the official test CSV or rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

SEEDS = (17, 29, 43)
ARMS = ("LTI_NO_REWARD", "LTI_CANONICAL_TEACHER")
TEST_EVAL_SCHEMA = "banking-cil-v10-test-evaluator-v1"
CONTRACT_BUNDLE_SCHEMA = "banking-cil-v11-launch-contract-bundle-v1"
EXPECTED_CONTRACT_FILES = tuple(
    f"seed{seed}-{arm.lower().replace('_', '-')}.json"
    for seed in SEEDS for arm in ARMS)
PATH_KEYS = {
    "task_root", "project_root", "model_dir", "foundation_root", "main_release",
    "seal_json", "training_main_root", "training_manifest", "v10_config",
    "v11_config", "v11_contract_dir", "v11_train_root", "test_eval_dir",
    "v11_reference", "v11_dev_memory_contract", "v11_dev_preflight_dir",
    "v11_dev_eval_dir", "v11_dev_launch_record",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping, *, fresh: bool = True) -> str:
    if fresh and path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if fresh:
        os.rename(tmp, path)
    else:
        os.replace(tmp, path)
    if sha256_file(path) != hashlib.sha256(raw).hexdigest():
        raise IOError(f"atomic readback hash mismatch: {path}")
    return hashlib.sha256(raw).hexdigest()


def load_path_map(path: Path) -> dict[str, Path]:
    raw = read_json(path)
    missing = PATH_KEYS - set(raw)
    if missing:
        raise ValueError(f"path map missing required keys: {sorted(missing)}")
    # Deliberately project only allowed keys. In particular, ignore its
    # official-test CSV path; this orchestrator accepts aggregate JSON only.
    return {key: Path(raw[key]).expanduser() for key in PATH_KEYS}


def grid_keys() -> tuple[tuple[int, str], ...]:
    return tuple((seed, arm) for seed in SEEDS for arm in ARMS)


def run_name(seed: int, arm: str) -> str:
    return f"seed{seed}-{arm.lower().replace('_', '-')}"


def _expected_contract_paths(contract_dir: Path) -> tuple[Path, ...]:
    return tuple(contract_dir / name for name in EXPECTED_CONTRACT_FILES)


def _assert_completed_test_aggregate(evaluation_path: Path) -> dict:
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"completed V10 evaluation aggregate missing: {evaluation_path}")
    value = read_json(evaluation_path)
    if value.get("schema") != TEST_EVAL_SCHEMA or value.get("status") != "complete":
        raise ValueError("V11 contract preparation requires a completed V10 official-test aggregate")
    # Do not read test examples or test payloads. The strict existing material
    # builder verifies all nine context metrics and source/checkpoint lineage.
    if value.get("optimizer_updates") != 0 or value.get("all_contexts_frozen") is not True or \
            value.get("backbone_unchanged") is not True:
        raise ValueError("V10 evaluation aggregate lacks immutable zero-update attestations")
    return value


def prepare_argv(paths: Mapping[str, Path], *, task_root: Path) -> list[str]:
    evaluation = paths["test_eval_dir"] / "evaluation.json"
    return [
        str(task_root / "tools" / "prepare_banking_v11_launch_materials.py"),
        "build-contracts",
        "--project-root", str(paths["project_root"]),
        "--model-path", str(paths["model_dir"]),
        "--training-pack", str(paths["training_manifest"]),
        "--v10-config", str(paths["v10_config"]),
        "--baseline-seal", str(paths["seal_json"]),
        "--baseline-runs-root", str(paths["training_main_root"]),
        "--v11-config", str(paths["v11_config"]),
        "--main-release", str(paths["main_release"]),
        "--foundation-root", str(paths["foundation_root"]),
        "--v10-test-evaluation", str(evaluation),
        "--output-root", str(paths["v11_train_root"]),
        "--contract-dir", str(paths["v11_contract_dir"]),
    ]


def validate_contract_bundle(bundle: Mapping, contract_dir: Path,
                             output_root: Path) -> list[dict]:
    if (bundle.get("schema"), bundle.get("status"), bundle.get("contract_count")) != (
            CONTRACT_BUNDLE_SCHEMA, "contracts_ready", 6):
        raise ValueError("contract preparation did not return the fixed six-job bundle")
    if (bundle.get("no_model_loaded") is not True or bundle.get("optimizer_updates") != 0 or
            bundle.get("test_payload_read") is not False or bundle.get("launcher_invoked") is not False):
        raise ValueError("contract builder violated its CPU-only/no-payload contract")
    advertised = bundle.get("contracts")
    if not isinstance(advertised, Mapping) or len(advertised) != 6:
        raise ValueError("contract bundle is missing six path hashes")
    wanted_paths = _expected_contract_paths(contract_dir)
    if set(map(str, wanted_paths)) != set(advertised):
        raise ValueError("contract bundle file set differs from the predeclared six-run grid")
    rows = []
    for seed, arm in grid_keys():
        path = contract_dir / f"{run_name(seed, arm)}.json"
        digest = advertised[str(path)]
        if sha256_file(path) != digest:
            raise ValueError(f"contract hash mismatch for {path.name}")
        contract = read_json(path)
        if (contract.get("seed"), contract.get("arm")) != (seed, arm):
            raise ValueError(f"contract identity mismatch: {path.name}")
        expected_out = output_root.resolve() / run_name(seed, arm)
        if Path(contract.get("output_dir", "")).resolve() != expected_out:
            raise ValueError(f"contract output path mismatch: {path.name}")
        rows.append({"seed": seed, "arm": arm, "contract_path": str(path.resolve()),
                     "contract_sha256": digest,
                     "output_dir": str(expected_out),
                     "launch_record_path": str(Path(contract["launch_record_path"]).resolve())})
    return rows


def validate_completed_job(contract: Mapping, contract_sha256: str) -> dict:
    output = Path(contract["output_dir"]).resolve()
    record_path = Path(contract["launch_record_path"]).resolve()
    if (output / "failure.json").exists() or not output.is_dir():
        raise ValueError("run is absent or has a failure marker")
    receipt_path = output / "run-receipt.json"
    protocol_path = output / "protocol.json"
    checkpoint_path = output / "checkpoints" / "task01-final.pt"
    if not all(path.is_file() for path in (receipt_path, protocol_path, checkpoint_path, record_path)):
        raise ValueError("existing run is incomplete; automatic retry is forbidden")
    receipt = read_json(receipt_path)
    protocol = read_json(protocol_path)
    launch_record = read_json(record_path)
    expected = {
        "schema": "banking-cil-v11-run-receipt-v1", "status": "complete",
        "seed": contract["seed"], "arm": contract["arm"], "task_id": 1, "epochs": 2,
        "config_sha256": contract["config_sha256"],
        "source_lock_sha256": contract["v11_source_lock_sha256"],
        "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "baseline_training_seal_sha256": contract["baseline_seal_sha256"],
        "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
        "checkpoint": "checkpoints/task01-final.pt",
        "test_data_included": False, "test_data_read": False,
        "model_immutable_versions_unchanged": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("existing run receipt does not match its frozen contract")
    comparison_sha = receipt.get("comparison_v10_sleep_task1_checkpoint_sha256")
    if not isinstance(comparison_sha, str) or len(comparison_sha) != 64:
        raise ValueError("receipt lacks its same-seed V10 Sleep comparison checkpoint")
    receipt_sha = sha256_file(receipt_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    if (receipt.get("checkpoint_sha256") != checkpoint_sha or
            type(receipt.get("task1_wake_ticks")) is not int or receipt["task1_wake_ticks"] <= 0):
        raise ValueError("terminal checkpoint hash/tick does not match the run receipt")
    protocol_expected = {
        "schema": "banking-cil-v11-protocol-v1", "status": "complete",
        "config_sha256": contract["config_sha256"],
        "source_lock_sha256": contract["v11_source_lock_sha256"],
        "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "baseline_training_seal_sha256": contract["baseline_seal_sha256"],
        "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
        "seed": contract["seed"], "arm": contract["arm"],
        "task_id": 1, "epochs": 2, "test_data_included": False, "test_data_read": False,
    }
    if any(protocol.get(key) != value for key, value in protocol_expected.items()):
        raise ValueError("run protocol does not match its frozen contract")
    if protocol.get("comparison_v10_sleep_task1_checkpoint_sha256") != comparison_sha:
        raise ValueError("receipt/protocol paired V10 Sleep checkpoint differs")
    preflight = launch_record.get("preflight")
    if (not isinstance(preflight, Mapping) or
            preflight.get("comparison_v10_sleep_task1_checkpoint_sha256") != comparison_sha):
        raise ValueError("paired V10 Sleep checkpoint is not bound to launcher CPU preflight")
    if protocol.get("checkpoint_sha256") != checkpoint_sha or protocol.get("run_receipt_sha256") != receipt_sha:
        raise ValueError("run protocol endpoint hashes do not match actual files")
    if (launch_record.get("schema") != "banking-v11-memory-policy-launch-record-v1" or
            launch_record.get("status") != "complete" or launch_record.get("exit_code") != 0 or
            launch_record.get("contract_sha256") != contract_sha256 or
            launch_record.get("config_sha256") != contract["config_sha256"] or
            launch_record.get("source_lock_sha256") != contract["v11_source_lock_sha256"] or
            launch_record.get("v10_source_lock_sha256") != contract["v10_source_lock_sha256"] or
            launch_record.get("training_manifest_sha256") != contract["training_manifest_sha256"] or
            launch_record.get("baseline_seal_sha256") != contract["baseline_seal_sha256"] or
            launch_record.get("foundation_checkpoint_sha256") != contract["foundation_checkpoint_sha256"] or
            launch_record.get("terminal_receipt", {}).get("sha256") != receipt_sha or
            Path(launch_record.get("terminal_receipt", {}).get("path", "")).resolve() != receipt_path.resolve() or
            launch_record.get("terminal_receipt", {}).get("checkpoint_sha256") != checkpoint_sha or
            launch_record.get("memory_fraction_callback_registered_before_cuda_init") is not True or
            launch_record.get("memory_fraction_callback_called") is not True or
            launch_record.get("memory_fraction_callback_error") is not None or
            launch_record.get("cuda_initialized_at_exit") is not True or
            launch_record.get("test_data_accessed_by_launcher") is not False):
        raise ValueError("memory launcher terminal record is incomplete or incorrectly bound")
    return {"status": "complete", "receipt_sha256": receipt_sha,
            "checkpoint_sha256": checkpoint_sha, "launch_record_sha256": sha256_file(record_path),
            "output_dir": str(output)}


def execute_serial(jobs: list[dict], run_one: Callable[[dict], Mapping]) -> list[dict]:
    """Run in declared order, never retry and stop immediately on an exception."""
    results = []
    for job in jobs:
        result = run_one(job)
        if result.get("status") not in ("complete", "already_complete"):
            raise RuntimeError(f"job did not complete: {job['seed']} {job['arm']}")
        results.append(dict(result))
    return results


def prepare(paths_file: Path, python: str) -> dict:
    paths = load_path_map(paths_file)
    task_root = paths["task_root"].resolve(strict=True)
    if task_root != Path(__file__).resolve().parents[1]:
        raise ValueError("path map task_root must point to this source tree")
    contract_dir = paths["v11_contract_dir"].resolve()
    output_root = paths["v11_train_root"].resolve()
    evaluation_path = paths["test_eval_dir"] / "evaluation.json"
    _assert_completed_test_aggregate(evaluation_path)
    if contract_dir.exists():
        raise FileExistsError(f"contract directory must be fresh: {contract_dir}")
    if output_root.exists():
        raise FileExistsError(f"V11 output root must be fresh: {output_root}")
    if not contract_dir.parent.is_dir():
        raise FileNotFoundError(contract_dir.parent)
    # The existing builder requires output-root to exist and validates that
    # each child job path is fresh. Create only the declared empty root.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    argv = prepare_argv(paths, task_root=task_root)
    completed = subprocess.run([python, *argv], cwd=task_root, text=True,
                               capture_output=True, check=False)
    if completed.returncode != 0:
        failure = {"schema": "banking-v11-serial-preparation-v1", "status": "failed",
                   "returncode": completed.returncode,
                   "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                   "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
                   "contracts_written": contract_dir.exists(), "jobs_launched": 0,
                   "test_rows_read_by_orchestrator": False}
        atomic_json(contract_dir.parent / "v11-contract-preparation-failure.json", failure)
        raise RuntimeError("V11 build-contracts failed; no jobs were launched")
    try:
        bundle = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise ValueError("prepare helper did not return its JSON bundle") from exc
    contracts = validate_contract_bundle(bundle, contract_dir, output_root)
    result = {"schema": "banking-v11-serial-contract-preparation-v1", "status": "ready",
        "contract_bundle_schema": bundle["schema"], "contract_count": len(contracts),
        "contracts": contracts, "official_test_evaluation_sha256": bundle["official_test_evaluation_sha256"],
        "v10_seal_sha256": bundle["v10_seal_sha256"], "v11_config_sha256": bundle["config_sha256"],
        "no_model_loaded": True, "optimizer_updates": 0, "test_payload_read": False,
        "jobs_launched": 0, "prepared_unix": time.time()}
    summary_path = contract_dir / "serial-contract-bundle.json"
    result["summary_sha256"] = atomic_json(summary_path, result)
    return result


def load_contract_jobs(paths: Mapping[str, Path]) -> list[dict]:
    contract_dir = paths["v11_contract_dir"].resolve(strict=True)
    output_root = paths["v11_train_root"].resolve(strict=True)
    summary = read_json(contract_dir / "serial-contract-bundle.json")
    if (summary.get("schema"), summary.get("status"), summary.get("contract_count"),
            summary.get("jobs_launched")) != (
            "banking-v11-serial-contract-preparation-v1", "ready", 6, 0):
        raise ValueError("prepared contract summary is missing or invalid")
    summary_rows = summary.get("contracts")
    if not isinstance(summary_rows, list) or len(summary_rows) != 6:
        raise ValueError("prepared summary must bind exactly six contracts")
    expected_paths = {str(path.resolve()) for path in _expected_contract_paths(contract_dir)}
    allowed_json = expected_paths | {str((contract_dir / "serial-contract-bundle.json").resolve()),
                                     str((contract_dir / "serial-run-status.json").resolve())}
    actual_json = {str(path.resolve()) for path in contract_dir.glob("*.json")}
    if actual_json - allowed_json or not expected_paths.issubset(actual_json):
        raise ValueError("contract directory does not contain exactly the six expected contract files")
    summary_by_path = {}
    for row in summary_rows:
        if not isinstance(row, Mapping) or set(row) != {
                "seed", "arm", "contract_path", "contract_sha256", "output_dir", "launch_record_path"}:
            raise ValueError("prepared contract row schema mismatch")
        key = str(Path(row["contract_path"]).resolve())
        if key in summary_by_path:
            raise ValueError("prepared summary contains a duplicate contract path")
        summary_by_path[key] = dict(row)
    if set(summary_by_path) != expected_paths:
        raise ValueError("prepared summary contract paths differ from the exact six-run grid")
    jobs = []
    for seed, arm in grid_keys():
        contract_path = contract_dir / f"{run_name(seed, arm)}.json"
        contract = read_json(contract_path)
        digest = sha256_file(contract_path)
        prepared = summary_by_path[str(contract_path.resolve())]
        if prepared["contract_sha256"] != digest:
            raise ValueError(f"contract changed after preparation: {contract_path.name}")
        if (contract.get("seed"), contract.get("arm")) != (seed, arm):
            raise ValueError(f"contract grid mismatch: {contract_path.name}")
        expected_output = output_root / run_name(seed, arm)
        expected_record = output_root / "launch-records" / f"{run_name(seed, arm)}.json"
        if (Path(contract.get("output_dir", "")).resolve() != expected_output or
                Path(contract.get("launch_record_path", "")).resolve() != expected_record):
            raise ValueError(f"contract points outside the expected run grid: {contract_path.name}")
        if (prepared["seed"], prepared["arm"],
                Path(prepared["output_dir"]).resolve(), Path(prepared["launch_record_path"]).resolve()) != (
                seed, arm, expected_output, expected_record):
            raise ValueError(f"prepared summary identity/path differs from contract body: {contract_path.name}")
        if (contract.get("v10_test_evaluation_sha256") != summary.get("official_test_evaluation_sha256") or
                contract.get("baseline_seal_sha256") != summary.get("v10_seal_sha256") or
                contract.get("config_sha256") != summary.get("v11_config_sha256")):
            raise ValueError(f"contract common input binding differs from prepared summary: {contract_path.name}")
        jobs.append({"seed": seed, "arm": arm, "contract_path": contract_path,
                     "contract": contract, "contract_sha256": digest})
    return jobs


def run_grid(paths_file: Path, python: str) -> dict:
    paths = load_path_map(paths_file)
    task_root = paths["task_root"].resolve(strict=True)
    jobs = load_contract_jobs(paths)
    contract_dir = paths["v11_contract_dir"].resolve()
    status_path = contract_dir / "serial-run-status.json"
    if status_path.exists():
        raise FileExistsError("serial status exists; automatic relaunch/resume is forbidden")
    status = {"schema": "banking-v11-serial-run-status-v1", "status": "running",
        "jobs": [], "no_automatic_retry": True, "official_test_accessed_by_orchestrator": False,
        "started_unix": time.time()}
    logs_dir = contract_dir / "serial-logs"
    if logs_dir.exists():
        raise FileExistsError("serial log directory exists; refusing an implicit retry")
    atomic_json(status_path, status)
    logs_dir.mkdir(exist_ok=False)
    def launch_one(job):
        contract = job["contract"]
        output = Path(contract["output_dir"]).resolve()
        record = Path(contract["launch_record_path"]).resolve()
        if output.exists() or record.exists():
            try:
                verified = validate_completed_job(contract, job["contract_sha256"])
            except Exception as exc:
                result = {"seed": job["seed"], "arm": job["arm"],
                          "status": "failed_existing", "failure": str(exc)}
                status["jobs"].append(result)
                atomic_json(status_path, status, fresh=False)
                raise
            result = {"seed": job["seed"], "arm": job["arm"],
                      "status": "already_complete", **verified}
            status["jobs"].append(result)
            atomic_json(status_path, status, fresh=False)
            return result
        command = [python, str(task_root / "tools" / "launch_banking_memory_policy_v11.py"),
                   "--contract", str(job["contract_path"]),
                   "--contract-sha256", job["contract_sha256"]]
        logbase = logs_dir / run_name(job["seed"], job["arm"])
        started = time.monotonic()
        completed = subprocess.run(command, cwd=task_root, text=True,
                                   capture_output=True, check=False)
        logbase.with_suffix(".stdout.txt").write_text(completed.stdout, encoding="utf-8")
        logbase.with_suffix(".stderr.txt").write_text(completed.stderr, encoding="utf-8")
        work = {"returncode": completed.returncode,
                "elapsed_wall_seconds": time.monotonic() - started,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest()}
        if completed.returncode != 0:
            result = {"seed": job["seed"], "arm": job["arm"], "status": "failed",
                      "process": work, "failure": f"child launcher exited {completed.returncode}"}
            status["jobs"].append(result)
            atomic_json(status_path, status, fresh=False)
            raise RuntimeError(result["failure"])
        try:
            verified = validate_completed_job(contract, job["contract_sha256"])
        except Exception as exc:
            result = {"seed": job["seed"], "arm": job["arm"], "status": "failed",
                      "process": work, "failure": f"{type(exc).__name__}: {exc}"}
            status["jobs"].append(result)
            atomic_json(status_path, status, fresh=False)
            raise
        result = {"seed": job["seed"], "arm": job["arm"],
                  "status": "complete", "process": work, **verified}
        status["jobs"].append(result)
        atomic_json(status_path, status, fresh=False)
        return result
    try:
        results = execute_serial(jobs, launch_one)
        status.update({"status": "complete", "jobs": results,
                       "completed_unix": time.time()})
        atomic_json(status_path, status, fresh=False)
        return status
    except BaseException as exc:
        status.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}",
                       "completed_jobs": sum(row.get("status") in ("complete", "already_complete")
                                               for row in status["jobs"]),
                       "jobs_recorded": len(status["jobs"]), "failed_unix": time.time()})
        atomic_json(status_path, status, fresh=False)
        raise


def prepare_dev_evaluation(paths_file: Path, python: str) -> dict:
    paths = load_path_map(paths_file)
    jobs = load_contract_jobs(paths)
    # No new V11 job is started here. Every six terminal receipt/checkpoint
    # bytes and each matching launch record must already be complete.
    verified = []
    for job in jobs:
        verified.append({"seed": job["seed"], "arm": job["arm"],
                         **validate_completed_job(job["contract"], job["contract_sha256"])})
    ref_path = paths["v11_reference"].resolve()
    contract_path = paths["v11_dev_memory_contract"].resolve()
    if ref_path.exists() or contract_path.exists():
        raise FileExistsError("development reference/contract output must be fresh")
    task_root = paths["task_root"].resolve(strict=True)
    builder = task_root / "tools" / "build_banking_v11_dev_reference.py"
    args = [python, str(builder), "--task-root", str(task_root),
        "--project-root", str(paths["project_root"]), "--model-dir", str(paths["model_dir"]),
        "--manifest", str(paths["training_manifest"]), "--v10-config", str(paths["v10_config"]),
        "--v10-seal", str(paths["seal_json"]), "--v10-release", str(paths["main_release"]),
        "--v10-runs-root", str(paths["training_main_root"]), "--v11-config", str(paths["v11_config"]),
        "--v11-runs-root", str(paths["v11_train_root"]), "--foundation-root", str(paths["foundation_root"]),
        "--output", str(ref_path)]
    completed = subprocess.run(args, cwd=task_root, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError("strict 15-context reference CPU preflight failed; dev evaluation not launched")
    ref_receipt = json.loads(completed.stdout.strip().splitlines()[-1])
    if (ref_receipt.get("status"), ref_receipt.get("contexts"), ref_receipt.get("official_test_accessed"),
            ref_receipt.get("model_loaded"), ref_receipt.get("optimizer_updates")) != (
            "reference_ready", 15, False, False, 0):
        raise ValueError("reference builder receipt failed strict 15-context CPU-only gate")
    reference = read_json(ref_path)
    launcher_sources = {
        "tools/launch_banking_v11_dev_evaluation.py",
        "tools/evaluate_banking_lti_ablation_v11.py",
        "tools/evaluate_banking_test_v10.py",
        "tools/launch_banking_memory_policy_v10.py",
        "tools/launch_banking_memory_policy_v11.py",
        "tools/run_banking_lti_ablation_v11.py",
        "tools/build_banking_v11_dev_reference.py",
    }
    source_hashes = {relative: sha256_file(task_root / relative) for relative in launcher_sources}
    v10_cfg = read_json(paths["v10_config"])
    contract = {
        "schema": "banking-v11-dev-memory-evaluation-contract-v1", "status": "frozen",
        "task_root": str(task_root), "project_root": str(paths["project_root"].resolve()),
        "model_dir": str(paths["model_dir"].resolve()),
        "training_pack_root": str(paths["training_manifest"].resolve().parent),
        "training_manifest_path": str(paths["training_manifest"].resolve()),
        "v10_config_path": str(paths["v10_config"].resolve()),
        "v11_config_path": str(paths["v11_config"].resolve()),
        "v10_grid_seal_path": str(paths["seal_json"].resolve()),
        "v10_release_artifact_path": str(paths["main_release"].resolve()),
        "v10_runs_root": str(paths["training_main_root"].resolve()),
        "v11_runs_root": str(paths["v11_train_root"].resolve()),
        "foundation_root": str(paths["foundation_root"].resolve()),
        "reference_path": str(ref_path),
        "preflight_output_dir": str(paths["v11_dev_preflight_dir"].resolve()),
        "evaluation_output_dir": str(paths["v11_dev_eval_dir"].resolve()),
        "record_path": str(paths["v11_dev_launch_record"].resolve()),
        "device": v10_cfg["device"], "reference_sha256": sha256_file(ref_path),
        "training_manifest_sha256": sha256_file(paths["training_manifest"]),
        "v10_config_file_sha256": sha256_file(paths["v10_config"]),
        "v11_config_file_sha256": sha256_file(paths["v11_config"]),
        "v10_grid_seal_sha256": sha256_file(paths["seal_json"]),
        "v10_release_sha256": sha256_file(paths["main_release"]),
        "source_hashes": source_hashes,
        "model_assets_sha256": reference["model_assets_sha256"],
    }
    import launch_banking_v11_dev_evaluation as dev_launcher
    encoded = (json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    contract_sha = hashlib.sha256(encoded).hexdigest()
    dev_launcher._validate_contract(contract, task_root=task_root,
        expected_contract_sha256=contract_sha, actual_contract_sha256=contract_sha)
    if contract_path.exists() or not contract_path.parent.is_dir():
        raise FileExistsError("development memory-contract output must be fresh under an existing parent")
    with contract_path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if sha256_file(contract_path) != contract_sha:
        raise IOError("development memory-contract readback SHA mismatch")
    return {"schema": "banking-v11-dev-evaluation-preparation-v1", "status": "ready_for_review",
        "reference_path": str(ref_path), "reference_sha256": sha256_file(ref_path),
        "reference_context_count": 15, "preflight_output_dir": contract["preflight_output_dir"],
        "evaluation_output_dir": contract["evaluation_output_dir"],
        "contract_path": str(contract_path), "contract_sha256": contract_sha,
        "verified_training_contexts": len(verified), "dev_evaluator_launched": False,
        "official_test_accessed_by_orchestrator": False, "optimizer_updates": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "prepare-dev-eval"):
        command = sub.add_parser(name)
        command.add_argument("--paths", type=Path, default=Path(__file__).resolve().parents[1] / "banking_v10_v11_paths.json")
        command.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.paths, args.python)
    elif args.command == "run":
        result = run_grid(args.paths, args.python)
    else:
        result = prepare_dev_evaluation(args.paths, args.python)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
