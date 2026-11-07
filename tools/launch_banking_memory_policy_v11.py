#!/usr/bin/env python3
"""Launch one source-bound V11 task-1 ablation with the reviewed process-local CUDA policy.

The launcher performs V11's real CPU/source/config/seal/foundation preflight
before setting ``PYTORCH_CUDA_ALLOC_CONF`` or queuing CUDA initialization.
It launches exactly one seed/arm; it contains no model, batching, optimizer,
or numerical changes and has no test-data input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

SCHEMA = "banking-v11-memory-policy-launch-v1"
POLICY = "garbage_collection_threshold:0.7"
MEMORY_FRACTION = 0.8
TORCH_VERSION = "2.8.0+cu128"
V11_SCRIPT = "run_banking_lti_ablation_v11.py"
CONTRACT_KEYS = {
    "schema", "launcher_sha256", "policy_launcher_v10_sha256", "project_root",
    "task_root", "model_path", "training_pack", "v10_config_path", "config_path",
    "baseline_seal", "baseline_runs_root", "foundation_dir", "v10_test_evaluation",
    "v10_test_evaluation_sha256", "v10_test_evaluator_sha256", "output_dir", "seed",
    "arm", "script_key", "script_sha256", "launch_record_path", "allocator_conf",
    "memory_fraction", "device", "torch_version", "config_sha256",
    "v11_source_lock_sha256", "v10_source_lock_sha256", "training_manifest_sha256",
    "baseline_seal_sha256", "foundation_checkpoint_sha256",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_contract(path: Path, expected_sha: str) -> dict:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError("V11 launcher contract is missing or its SHA-256 differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != CONTRACT_KEYS:
        raise ValueError("V11 launcher contract has an invalid schema or key set")
    if value["schema"] != SCHEMA:
        raise ValueError("unsupported V11 memory-policy contract")
    if value["allocator_conf"] != POLICY or value["memory_fraction"] != MEMORY_FRACTION:
        raise ValueError("V11 launcher only accepts the reviewed fixed memory policy")
    if value["torch_version"] != TORCH_VERSION or not str(value["device"]).startswith("cuda:"):
        raise ValueError("V11 launcher requires the reviewed PyTorch build and explicit CUDA device")
    if value["script_key"] != V11_SCRIPT:
        raise ValueError("V11 launcher allowlist permits only the V11 runner")
    for key in ("launcher_sha256", "policy_launcher_v10_sha256", "script_sha256", "config_sha256",
                "v11_source_lock_sha256", "v10_source_lock_sha256", "training_manifest_sha256",
                "baseline_seal_sha256", "foundation_checkpoint_sha256",
                "v10_test_evaluation_sha256", "v10_test_evaluator_sha256"):
        item = value[key]
        if not isinstance(item, str) or len(item) != 64 or any(c not in "0123456789abcdef" for c in item):
            raise ValueError(f"contract field {key} must be a lowercase SHA-256")
    return value


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=True)


def _load_v11(task_root: Path):
    tools_dir = task_root / "tools"
    for entry in (tools_dir, task_root):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import run_banking_lti_ablation_v11 as v11
    return v11


def _preflight_contract(contract: Mapping[str, Any], *, task_root: Path,
                        v11_module=None) -> tuple[dict, Any]:
    """Run the real V11 CPU gates and compare every contract binding."""
    if Path(contract["task_root"]).resolve() != task_root.resolve():
        raise ValueError("contract task root differs from this V11 source tree")
    project_root = _resolved(contract["project_root"])
    model_path = _resolved(contract["model_path"])
    training_pack = _resolved(contract["training_pack"])
    v10_config = _resolved(contract["v10_config_path"])
    config = _resolved(contract["config_path"])
    baseline_seal = _resolved(contract["baseline_seal"])
    baseline_runs = _resolved(contract["baseline_runs_root"])
    foundation_dir = _resolved(contract["foundation_dir"])
    test_evaluation = _resolved(contract["v10_test_evaluation"])
    output_dir = Path(contract["output_dir"]).expanduser().resolve()
    record_path = Path(contract["launch_record_path"]).expanduser().resolve()
    script = (task_root / "tools" / V11_SCRIPT).resolve(strict=True)
    policy_launcher = (task_root / "tools" / "launch_banking_memory_policy_v10.py").resolve(strict=True)
    if sha256_file(script) != contract["script_sha256"]:
        raise ValueError("V11 runner source SHA differs from launch contract")
    if sha256_file(policy_launcher) != contract["policy_launcher_v10_sha256"]:
        raise ValueError("reviewed V10 memory-policy helper source SHA differs")
    if sha256_file(Path(__file__).resolve()) != contract["launcher_sha256"]:
        raise ValueError("V11 launcher source SHA differs from contract")
    v10_config_obj = json.loads(v10_config.read_text(encoding="utf-8"))
    if v10_config_obj.get("device") != contract["device"]:
        raise ValueError("launch contract device differs from frozen V10 configuration")
    if output_dir.exists() or record_path.exists():
        raise FileExistsError("V11 output and launch-record paths must be fresh")
    args = SimpleNamespace(config=config, v10_config=v10_config,
        project_root=project_root, model_path=model_path, training_pack=training_pack,
        baseline_seal=baseline_seal, baseline_runs_root=baseline_runs,
        foundation_dir=foundation_dir, output_dir=output_dir,
        seed=int(contract["seed"]), arm=str(contract["arm"]))
    v11 = v11_module or _load_v11(task_root)
    if str(contract["arm"]) not in tuple(v11.ARMS):
        raise ValueError("V11 contract arm is not in the runner's frozen arm grid")
    test_evaluation_binding = _validate_completed_v10_test(contract, test_evaluation,
        task_root=task_root, v11_module=v11, v10_config=v10_config)
    result = v11._preflight(args)
    # The frozen V11 producer's legacy result key named
    # ``training_manifest_sha256`` actually carries the train JSONL digest.
    # Keep the contract's manifest-file SHA distinct and validate both values
    # against the pack metadata before comparing the launch bindings.
    pack_manifest_sha = sha256_file(training_pack)
    if pack_manifest_sha != contract["training_manifest_sha256"]:
        raise ValueError("training pack manifest bytes differ from launch contract")
    pack_manifest = json.loads(training_pack.read_text(encoding="utf-8"))
    files = pack_manifest.get("files") if isinstance(pack_manifest, Mapping) else None
    train_record = files.get("train") if isinstance(files, Mapping) else None
    train_file_sha = train_record.get("sha256") if isinstance(train_record, Mapping) else None
    if not _is_sha256(train_file_sha):
        raise ValueError("training pack manifest has no valid train-file SHA-256")
    if result.get("training_manifest_sha256") != train_file_sha:
        raise ValueError("V11 preflight train-file digest differs from pack manifest metadata")
    if sha256_file(training_pack) != pack_manifest_sha:
        raise ValueError("training pack manifest changed during CPU preflight")
    result = dict(result)
    result["training_file_sha256"] = train_file_sha
    result["training_manifest_sha256"] = pack_manifest_sha
    protected_roots = (project_root, model_path, training_pack.parent,
                       foundation_dir, baseline_runs)
    v11.output_path_guard(record_path, protected_roots=protected_roots)
    if (record_path == output_dir or record_path in output_dir.parents or
            output_dir in record_path.parents):
        raise ValueError("launch record must be separate from the run output directory")
    expected = {
        "config_sha256": contract["config_sha256"],
        "source_lock_sha256": contract["v11_source_lock_sha256"],
        "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "baseline_seal_sha256": contract["baseline_seal_sha256"],
        "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
    }
    actual = {
        "config_sha256": result["config_sha256"],
        "source_lock_sha256": result["source_lock_sha256"],
        "v10_source_lock_sha256": result["v10_source_lock_sha256"],
        "training_manifest_sha256": result["training_manifest_sha256"],
        "baseline_seal_sha256": result["baseline_seal_sha256"],
        "foundation_checkpoint_sha256": result["foundation_checkpoint_sha256"],
    }
    if actual != expected:
        raise ValueError("V11 CPU preflight bindings differ from launch contract")
    if (result.get("seed"), result.get("arm"), result.get("test_data_included"),
        result.get("model_loaded"), result.get("optimizer_updates")) != (
            int(contract["seed"]), str(contract["arm"]), False, False, 0):
        raise ValueError("V11 preflight did not prove the expected no-test/no-model state")
    result["v10_test_evaluation_binding"] = test_evaluation_binding
    return result, v11


def _validate_completed_v10_test(contract: Mapping[str, Any], evaluation_path: Path, *,
                                 task_root: Path, v11_module, v10_config: Path) -> dict:
    """Require a byte-bound, complete official-test evaluation for all nine V10 runs."""
    if sha256_file(evaluation_path) != contract["v10_test_evaluation_sha256"]:
        raise ValueError("completed V10 official-test evaluation bytes differ from contract")
    evaluator_path = task_root / "tools" / "evaluate_banking_test_v10.py"
    if sha256_file(evaluator_path) != contract["v10_test_evaluator_sha256"]:
        raise ValueError("V10 official-test evaluator source differs from contract")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict) or (evaluation.get("schema"), evaluation.get("status")) != (
            "banking-cil-v10-test-evaluator-v1", "complete"):
        raise ValueError("V11 launch requires a completed official-test evaluation artifact")
    driver = v11_module._load_driver()
    cfg = driver.validate_config(json.loads(v10_config.read_text(encoding="utf-8")), phase="main")
    training = evaluation.get("training", {})
    for key, expected in (("config_sha256", driver._config_sha(cfg)),
                          ("source_lock_sha256", contract["v10_source_lock_sha256"]),
                          ("training_manifest_sha256", contract["training_manifest_sha256"]),
                          ("seal_sha256", contract["baseline_seal_sha256"])):
        if training.get(key) != expected:
            raise ValueError(f"official-test evaluation training binding mismatch: {key}")
    if training.get("evaluator_sha256") != contract["v10_test_evaluator_sha256"]:
        raise ValueError("official-test result was not produced by the pinned evaluator")
    seal_path = _resolved(contract["baseline_seal"])
    if sha256_file(seal_path) != contract["baseline_seal_sha256"]:
        raise ValueError("official-test baseline seal bytes differ from contract")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected_grid = {(seed, arm) for seed in (17, 29, 43)
                     for arm in ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")}
    if (seal.get("schema"), seal.get("complete"), seal.get("config_sha256"),
            seal.get("source_lock_sha256"), seal.get("training_manifest_sha256"),
            seal.get("test_data_included"), seal.get("official_test_read")) != (
            "banking-cil-v10-training-seal-v1", True, driver._config_sha(cfg),
            contract["v10_source_lock_sha256"], contract["training_manifest_sha256"], False, False):
        raise ValueError("official-test baseline seal is incomplete or has mismatched source bindings")
    seal_rows = seal.get("runs")
    if not isinstance(seal_rows, list) or len(seal_rows) != 9:
        raise ValueError("official-test baseline seal must contain exactly nine V10 runs")
    row_keys = {"seed", "arm", "status", "tasks_completed", "final_checkpoint",
        "final_checkpoint_sha256", "run_receipt_sha256", "config_sha256", "source_lock_sha256",
        "training_manifest_sha256", "run_dir", "endpoint_metrics", "work"}
    seal_by_grid = {}
    for row in seal_rows:
        if not isinstance(row, Mapping) or set(row) != row_keys:
            raise ValueError("official-test seal rows do not match frozen V10 schema")
        key = (row["seed"], row["arm"])
        if key in seal_by_grid or key not in expected_grid:
            raise ValueError("official-test seal contains a duplicate or unexpected seed/arm")
        if (row["status"], row["tasks_completed"], row["final_checkpoint"],
                row["config_sha256"], row["source_lock_sha256"], row["training_manifest_sha256"]) != (
                "complete", 7, "checkpoints/final-task06.pt", driver._config_sha(cfg),
                contract["v10_source_lock_sha256"], contract["training_manifest_sha256"]):
            raise ValueError("official-test seal row is incomplete or has mismatched bindings")
        if not _is_sha256(row["final_checkpoint_sha256"]) or not _is_sha256(row["run_receipt_sha256"]):
            raise ValueError("official-test seal row is missing checkpoint/receipt SHA-256")
        seal_by_grid[key] = row
    if set(seal_by_grid) != expected_grid:
        raise ValueError("official-test baseline seal does not cover the exact nine-run grid")

    verified = training.get("verified_contexts")
    if not isinstance(verified, list) or any(not isinstance(row, Mapping) for row in verified):
        raise ValueError("official-test evaluation verified-context list is malformed")
    actual_grid = {(row.get("seed"), row.get("arm")) for row in verified}
    if len(verified) != 9 or actual_grid != expected_grid:
        raise ValueError("official-test evaluation did not verify the complete nine-run V10 grid")
    if len(actual_grid) != len(verified):
        raise ValueError("official-test evaluation has duplicate seed/arm contexts")
    for row in verified:
        if set(row) != {"seed", "arm", "checkpoint_sha256", "receipt_sha256"}:
            raise ValueError("official-test verified-context row differs from frozen evaluator schema")
        if not _is_sha256(row["checkpoint_sha256"]) or not _is_sha256(row["receipt_sha256"]):
            raise ValueError("official-test verified-context hash is malformed")
        key = (row["seed"], row["arm"])
        sealed = seal_by_grid[key]
        if (row.get("checkpoint_sha256"), row.get("receipt_sha256")) != (
                sealed["final_checkpoint_sha256"], sealed["run_receipt_sha256"]):
            raise ValueError("official-test verified-context hashes do not match the baseline seal")

    primary = evaluation.get("primary_raw_official_test")
    if not isinstance(primary, Mapping) or set(primary) != {
            f"seed{seed}:{arm}" for seed, arm in expected_grid}:
        raise ValueError("official-test primary metrics must contain exactly all nine seed/arm contexts")
    metric_names = ("teacher_forced_sum", "teacher_forced_length_mean", "greedy_generation",
                    "gold_label_token_mean_nll")
    for name, context_metrics in primary.items():
        if not isinstance(context_metrics, Mapping) or not set(metric_names).issubset(context_metrics):
            raise ValueError(f"official-test context {name} is missing frozen evaluator metrics")
        for metric_name in metric_names[:3]:
            metric = context_metrics[metric_name]
            if not isinstance(metric, Mapping) or metric.get("count") != 3080:
                raise ValueError(f"official-test context {name} metric {metric_name} must cover 3,080 rows")
        nll = context_metrics["gold_label_token_mean_nll"]
        if not isinstance(nll, Mapping) or nll.get("count") != 3080:
            raise ValueError(f"official-test context {name} NLL count must cover 3,080 per-example means")
        if (type(nll.get("mean")) not in (int, float) or
                not math.isfinite(float(nll["mean"]))):
            raise ValueError(f"official-test context {name} must have finite per-example NLL mean")
        for metric_name in metric_names[:3]:
            metric = context_metrics[metric_name]
            correct = metric.get("correct")
            if type(correct) is not int or not 0 <= correct <= 3080:
                raise ValueError(f"official-test context {name} metric {metric_name} has invalid correct count")
        greedy = context_metrics["greedy_generation"]
        invalid = greedy.get("invalid_predictions")
        if type(invalid) is not int or not 0 <= invalid <= 3080:
            raise ValueError(f"official-test context {name} has invalid generation invalid-count")
    source = evaluation.get("test_source", {})
    if (source.get("source_revision"), source.get("source_sha256"), source.get("source_bytes"),
        source.get("raw_rows"), source.get("classes"), source.get("rows_per_class"),
        source.get("primary_includes_all_official_rows")) != (
            "9d081458ff52e53cf7e848f414e6e9344e4e6696",
            "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
            239961, 3080, 77, 40, True):
        raise ValueError("official-test evaluation does not report the pinned complete test cohort")
    if (evaluation.get("optimizer_updates") != 0 or evaluation.get("all_contexts_frozen") is not True or
            evaluation.get("backbone_unchanged") is not True or
            evaluation.get("raw_prompts_predictions_and_token_ids_saved") is not False):
        raise ValueError("official-test evaluation safety/immutability fields are invalid")
    if not isinstance(evaluation.get("primary_raw_official_test"), Mapping):
        raise ValueError("official-test evaluation is missing primary metric payload")
    return {"artifact_sha256": contract["v10_test_evaluation_sha256"],
            "evaluator_sha256": contract["v10_test_evaluator_sha256"],
            "training_seal_sha256": training["seal_sha256"],
            "verified_context_count": 9, "official_test_rows": 3080,
            "metrics_per_context": list(metric_names),
            "seal_crossbinding_verified": True}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _install_reviewed_policy(torch_module, device: str, launcher_module) -> dict:
    if torch_module.__version__ != TORCH_VERSION:
        raise RuntimeError(f"PyTorch version mismatch: {torch_module.__version__}")
    if torch_module.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before the V11 memory policy could be queued")
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if current not in (None, POLICY):
        raise ValueError("conflicting allocator environment; refusing to overwrite")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = POLICY
    state = launcher_module._install_memory_policy(torch_module, device)
    if not state.get("registered") or torch_module.cuda.is_initialized():
        raise RuntimeError("deferred memory policy was not queued before CUDA initialization")
    return state


def _atomic_record(path: Path, value: dict, *, replace: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        raise FileExistsError(tmp)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if sha256_file(path) != expected:
        raise IOError("V11 launch-record readback hash mismatch")
    return expected


def launch(contract_path: str | Path, expected_contract_sha256: str) -> int:
    contract_path = Path(contract_path).resolve(strict=True)
    contract = _load_contract(contract_path, expected_contract_sha256)
    task_root = Path(__file__).resolve().parents[1]
    # Run all source, config, seal, foundation and checkpoint checks before
    # importing CUDA launch machinery or changing allocator environment.
    preflight, v11 = _preflight_contract(contract, task_root=task_root)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during CPU preflight")
    policy_launcher = _load_v10_policy_launcher(task_root)
    callback = _install_reviewed_policy(torch, str(contract["device"]), policy_launcher)
    record_path = Path(contract["launch_record_path"]).expanduser().resolve()
    record = {"schema": "banking-v11-memory-policy-launch-record-v1", "status": "launching",
        "contract_sha256": expected_contract_sha256,
        "launcher_sha256": contract["launcher_sha256"],
        "policy_launcher_v10_sha256": contract["policy_launcher_v10_sha256"],
        "script_sha256": contract["script_sha256"],
        "config_sha256": contract["config_sha256"],
        "source_lock_sha256": contract["v11_source_lock_sha256"],
        "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "baseline_seal_sha256": contract["baseline_seal_sha256"],
        "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
        "v10_test_evaluation_sha256": contract["v10_test_evaluation_sha256"],
        "v10_test_evaluation_binding": preflight["v10_test_evaluation_binding"],
        "seed": int(contract["seed"]), "arm": str(contract["arm"]),
        "allocator_conf": POLICY, "memory_fraction": MEMORY_FRACTION,
        "memory_fraction_callback_registered_before_cuda_init": True,
        "memory_fraction_callback_called": False,
        "cuda_initialized_before_target": False, "test_data_accessed_by_launcher": False,
        "preflight": preflight, "started_unix": time.time()}
    _atomic_record(record_path, record, replace=False)
    args = ["--phase", "run", "--config", str(Path(contract["config_path"]).resolve()),
        "--v10-config", str(Path(contract["v10_config_path"]).resolve()),
        "--project-root", str(Path(contract["project_root"]).resolve()),
        "--model-path", str(Path(contract["model_path"]).resolve()),
        "--training-pack", str(Path(contract["training_pack"]).resolve()),
        "--baseline-seal", str(Path(contract["baseline_seal"]).resolve()),
        "--baseline-runs-root", str(Path(contract["baseline_runs_root"]).resolve()),
        "--foundation-dir", str(Path(contract["foundation_dir"]).resolve()),
        "--output-dir", str(Path(contract["output_dir"]).resolve()),
        "--seed", str(contract["seed"]), "--arm", str(contract["arm"])]
    started = time.monotonic()
    exit_code = 1
    failure: str | None = None
    try:
        sys.argv = [str(task_root / "tools" / V11_SCRIPT), *args]
        try:
            runpy.run_path(str(task_root / "tools" / V11_SCRIPT), run_name="__main__")
        except SystemExit as exc:
            exit_code = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            if exit_code:
                failure = f"SystemExit: {exc.code}"
        if exit_code == 0:
            output = Path(contract["output_dir"]).resolve()
            receipt_path = output / "run-receipt.json"
            if not receipt_path.is_file():
                raise RuntimeError("V11 terminal run receipt is missing")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected = {"schema": "banking-cil-v11-run-receipt-v1", "status": "complete",
                "seed": int(contract["seed"]), "arm": str(contract["arm"]), "task_id": 1,
                "config_sha256": contract["config_sha256"],
                "source_lock_sha256": contract["v11_source_lock_sha256"],
                "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
                "training_manifest_sha256": contract["training_manifest_sha256"],
                "baseline_training_seal_sha256": contract["baseline_seal_sha256"],
                "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
                "comparison_v10_sleep_task1_checkpoint_sha256":
                    preflight["comparison_v10_sleep_task1_checkpoint_sha256"],
                "epochs": 2, "test_data_included": False, "test_data_read": False,
                "model_immutable_versions_unchanged": True}
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise RuntimeError("V11 receipt fails source/grid/test isolation binding")
            if int(receipt.get("task1_wake_ticks", 0)) <= 0:
                raise RuntimeError("V11 terminal receipt has no completed task-1 training ticks")
            cp_relative = Path(receipt["checkpoint"])
            checkpoint = (output / cp_relative).resolve()
            if output not in checkpoint.parents or sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
                raise RuntimeError("V11 terminal checkpoint path/hash is invalid")
            record["terminal_receipt"] = {"path": str(receipt_path),
                "sha256": sha256_file(receipt_path), "checkpoint_sha256": receipt["checkpoint_sha256"]}
            if not callback.get("called") or callback.get("error") or not torch.cuda.is_initialized():
                raise RuntimeError("V11 target did not apply the deferred CUDA policy")
    except BaseException as exc:
        exit_code = 1
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        record.update({"status": "complete" if exit_code == 0 else "failed",
            "exit_code": exit_code, "failure": failure,
            "elapsed_wall_seconds": time.monotonic() - started,
            "memory_fraction_callback_called": bool(callback.get("called")),
            "memory_fraction_callback_error": callback.get("error"),
            "cuda_initialized_at_exit": bool(torch.cuda.is_initialized()),
            "ended_unix": time.time()})
        _atomic_record(record_path, record, replace=True)
    if exit_code:
        raise RuntimeError(f"V11 memory-policy launch failed: {failure}")
    return 0


def _load_v10_policy_launcher(task_root: Path):
    tools_dir = task_root / "tools"
    for entry in (tools_dir, task_root):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import launch_banking_memory_policy_v10 as launcher
    return launcher


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args(argv)
    return launch(args.contract, args.contract_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
