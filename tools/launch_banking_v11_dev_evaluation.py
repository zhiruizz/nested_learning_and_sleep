"""Source-bound same-process V11 dev evaluator with the reviewed memory policy.

The wrapper runs the evaluator's CPU preflight first, queues the reviewed
allocator policy, and then invokes evaluation with the same input arguments and
a fresh output directory. It never reads the official test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Mapping

SCHEMA = "banking-v11-dev-memory-evaluation-contract-v1"
RECORD_SCHEMA = "banking-v11-dev-memory-evaluation-record-v1"
POLICY = "garbage_collection_threshold:0.7"
MEMORY_FRACTION = 0.8
TORCH_VERSION = "2.8.0+cu128"
EXPECTED_SOURCES = {
    "tools/launch_banking_v11_dev_evaluation.py",
    "tools/evaluate_banking_lti_ablation_v11.py",
    "tools/evaluate_banking_test_v10.py",
    "tools/launch_banking_memory_policy_v10.py",
    "tools/launch_banking_memory_policy_v11.py",
    "tools/run_banking_lti_ablation_v11.py",
    "tools/build_banking_v11_dev_reference.py",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _is_sha(value) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(ch in "0123456789abcdef" for ch in value.lower()))


def _load_evaluator(task_root: Path):
    tools_dir = task_root / "tools"
    for item in (tools_dir, task_root):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    import evaluate_banking_lti_ablation_v11 as evaluator
    return evaluator


def verify_source_hashes(task_root: Path, source_hashes: Mapping) -> None:
    """Check every locked source, including this launcher, before any run."""
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != EXPECTED_SOURCES:
        raise ValueError("contract must pin the evaluator, both memory helpers, V11 runner, and reference builder")
    for relative, expected in source_hashes.items():
        if not _is_sha(expected):
            raise ValueError(f"invalid source SHA-256: {relative}")
        path = Path(task_root) / relative
        if sha256_file(path) != expected:
            raise ValueError(f"pinned source changed: {relative}")


def _validate_contract(contract: Mapping, *, task_root: Path,
                       expected_contract_sha256: str,
                       actual_contract_sha256: str) -> tuple[dict, object]:
    required = {
        "schema", "status", "task_root", "project_root", "model_dir",
        "training_pack_root", "training_manifest_path", "v10_config_path",
        "v11_config_path", "v10_grid_seal_path", "v10_release_artifact_path",
        "v10_runs_root", "v11_runs_root", "foundation_root", "reference_path",
        "preflight_output_dir", "evaluation_output_dir", "record_path", "device",
        "reference_sha256", "training_manifest_sha256",
        "v10_config_file_sha256", "v11_config_file_sha256", "v10_grid_seal_sha256",
        "v10_release_sha256", "source_hashes", "model_assets_sha256",
    }
    if set(contract) != required or contract.get("schema") != SCHEMA:
        raise ValueError("V11 dev memory contract schema/field set mismatch")
    if contract.get("status") != "frozen":
        raise ValueError("V11 dev evaluation requires a frozen contract")
    if (not _is_sha(expected_contract_sha256) or
            actual_contract_sha256 != expected_contract_sha256):
        raise ValueError("contract SHA does not match its launch argument")
    if Path(contract["task_root"]).resolve() != task_root.resolve():
        raise ValueError("contract task root differs from the running source tree")
    for key in ("reference_sha256", "training_manifest_sha256", "v10_config_file_sha256",
                "v11_config_file_sha256", "v10_grid_seal_sha256", "v10_release_sha256"):
        if not _is_sha(contract.get(key)):
            raise ValueError(f"contract {key} must be SHA-256")
    source_hashes = contract["source_hashes"]
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != EXPECTED_SOURCES:
        raise ValueError("contract must pin the evaluator, both memory helpers, V11 runner, and reference builder")
    if any(not _is_sha(value) for value in source_hashes.values()):
        raise ValueError("contract source map contains an invalid SHA-256")
    if not isinstance(contract["model_assets_sha256"], Mapping) or not contract["model_assets_sha256"] or any(
            not _is_sha(value) for value in contract["model_assets_sha256"].values()):
        raise ValueError("contract model asset map is malformed")

    paths = {key: Path(contract[key]).expanduser().resolve() for key in (
        "project_root", "model_dir", "training_pack_root", "training_manifest_path",
        "v10_config_path", "v11_config_path", "v10_grid_seal_path",
        "v10_release_artifact_path", "v10_runs_root", "v11_runs_root",
        "foundation_root", "reference_path", "preflight_output_dir",
        "evaluation_output_dir", "record_path")}
    for key in ("project_root", "model_dir", "training_pack_root", "v10_runs_root",
                "v11_runs_root", "foundation_root"):
        if not paths[key].is_dir():
            raise FileNotFoundError(f"contract {key} is not an existing directory")
    for key in ("training_manifest_path", "v10_config_path", "v11_config_path",
                "v10_grid_seal_path", "v10_release_artifact_path", "reference_path"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"contract {key} is not an existing file")
    if paths["preflight_output_dir"] == paths["evaluation_output_dir"]:
        raise ValueError("preflight and evaluation outputs must be separate directories")
    for key in ("preflight_output_dir", "evaluation_output_dir", "record_path"):
        if paths[key].exists():
            raise FileExistsError(f"contract output is not fresh: {key}")
        if not paths[key].parent.is_dir():
            raise FileNotFoundError(f"contract output parent must already exist: {key}")
    if contract["device"] != "cuda:0":
        raise ValueError("frozen V11 dev evaluation contract must use cuda:0")

    verify_source_hashes(task_root, source_hashes)
    checks = {
        "reference_sha256": paths["reference_path"],
        "training_manifest_sha256": paths["training_manifest_path"],
        "v10_config_file_sha256": paths["v10_config_path"],
        "v11_config_file_sha256": paths["v11_config_path"],
        "v10_grid_seal_sha256": paths["v10_grid_seal_path"],
        "v10_release_sha256": paths["v10_release_artifact_path"],
    }
    for field, path in checks.items():
        if sha256_file(path) != contract[field]:
            raise ValueError(f"contract input bytes mismatch: {field}")
    if _json(paths["v10_config_path"]).get("device") != contract["device"]:
        raise ValueError("contract device differs from the frozen V10 configuration")
    reference = _json(paths["reference_path"])
    evaluator = _load_evaluator(task_root)
    evaluator.validate_reference(reference)
    if (reference["evaluator_sha256"] != source_hashes["tools/evaluate_banking_lti_ablation_v11.py"] or
            reference["model_assets_sha256"] != contract["model_assets_sha256"]):
        raise ValueError("reference evaluator/model asset pins differ from the contract")
    return paths, evaluator


def evaluator_argv(contract: Mapping, *, mode: str, output_dir: Path) -> list[str]:
    if mode not in ("preflight", "evaluate"):
        raise ValueError("unsupported evaluator mode")
    pairs = (
        ("--mode", mode), ("--reference", contract["reference_path"]),
        ("--v10-config", contract["v10_config_path"]),
        ("--v11-config", contract["v11_config_path"]),
        ("--training-manifest", contract["training_manifest_path"]),
        ("--v10-grid-seal", contract["v10_grid_seal_path"]),
        ("--v10-release-artifact", contract["v10_release_artifact_path"]),
        ("--v10-runs-root", contract["v10_runs_root"]),
        ("--v11-runs-root", contract["v11_runs_root"]),
        ("--foundation-root", contract["foundation_root"]),
        ("--training-pack-root", contract["training_pack_root"]),
        ("--project-root", contract["project_root"]),
        ("--model-dir", contract["model_dir"]),
        ("--output-dir", str(output_dir)),
    )
    return [item for pair in pairs for item in pair]


def _run_evaluator(evaluator_path: Path, argv: list[str]) -> None:
    prior = sys.argv
    try:
        sys.argv = [str(evaluator_path), *argv]
        try:
            runpy.run_path(str(evaluator_path), run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            if code not in (None, 0):
                raise RuntimeError(f"evaluator exited with status {code}") from exc
    finally:
        sys.argv = prior


def validate_preflight_receipt(value: Mapping, *, reference: Mapping) -> None:
    if (value.get("schema") != "banking-lti-ablation-v11-dev-evaluator-v1" or
            value.get("status") != "pass" or value.get("mode") != "preflight" or
            value.get("context_count") != 15 or value.get("v10_terminal_checkpoint_count") != 9 or
            value.get("cpu_checkpoint_deserialization") is not True or
            value.get("model_loaded") is not False or value.get("cuda_initialized") is not False or
            value.get("official_test_accessed") is not False or value.get("optimizer_updates") != 0 or
            value.get("reference_sha256") != reference.get("_actual_reference_sha256") or
            value.get("evaluator_sha256") != reference["evaluator_sha256"]):
        raise ValueError("CPU preflight receipt does not match the locked 15-context contract")


def validate_evaluation_result(value: Mapping, *, contract: Mapping,
                               reference: Mapping, callback: Mapping) -> None:
    from evaluate_banking_lti_ablation_v11 import ARMS, SEEDS, TASK01_ROWS
    protocol = value.get("protocol")
    expected_inputs = {
        "reference_sha256": contract["reference_sha256"],
        "evaluator_sha256": contract["source_hashes"]["tools/evaluate_banking_lti_ablation_v11.py"],
        "v10_config_sha256": reference["v10_config_sha256"],
        "v11_config_sha256": reference["v11_config_sha256"],
        "v10_grid_seal_sha256": contract["v10_grid_seal_sha256"],
        "v10_release_sha256": contract["v10_release_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
    }
    if (value.get("schema") != "banking-lti-ablation-v11-dev-evaluator-v1" or
            value.get("status") != "complete" or
            not isinstance(protocol, Mapping) or protocol.get("seeds") != list(SEEDS) or
            protocol.get("contexts_per_seed") != 5 or protocol.get("arms") != list(ARMS) or
            protocol.get("rows_per_seed") != TASK01_ROWS or
            protocol.get("official_test_accessed") is not False or
            protocol.get("raw_examples_predictions_and_token_ids_saved") is not False or
            value.get("optimizer_updates") != 0 or value.get("contexts_unchanged") is not True or
            value.get("frozen_backbone_unchanged") is not True or
            value.get("observed_model_logits_calls", 0) <= 0):
        raise ValueError("development evaluation does not satisfy the frozen 3x5 protocol")
    if value.get("frozen_inputs") != expected_inputs:
        raise ValueError("development result frozen-input hashes differ from contract")
    development = value.get("development")
    if not isinstance(development, Mapping) or set(development) != {str(seed) for seed in SEEDS}:
        raise ValueError("development result is missing seed summaries")
    for seed in SEEDS:
        row = development[str(seed)]
        if (row.get("cohort_rows") != TASK01_ROWS or row.get("context_count") != 5 or
                set(row.get("candidate_metrics", {})) != set(ARMS) or
                set(row.get("greedy_generation_metrics", {})) != set(ARMS)):
            raise ValueError(f"development result seed {seed} has incomplete 5-arm metrics")
    if not callback.get("registered") or not callback.get("called") or callback.get("error"):
        raise ValueError("deferred memory fraction callback did not complete successfully")


def _atomic_record(path: Path, payload: Mapping, *, replace: bool) -> str:
    if path.exists() and not replace:
        raise FileExistsError(path)
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(tmp, path)
    else:
        os.rename(tmp, path)
    if sha256_file(path) != hashlib.sha256(raw).hexdigest():
        raise IOError("launch record readback SHA mismatch")
    return hashlib.sha256(raw).hexdigest()


def launch(contract_path: str | Path, expected_contract_sha256: str) -> int:
    contract_path = Path(contract_path).resolve(strict=True)
    contract = _json(contract_path)
    task_root = Path(__file__).resolve().parents[1]
    paths, evaluator = _validate_contract(contract, task_root=task_root,
        expected_contract_sha256=expected_contract_sha256,
        actual_contract_sha256=sha256_file(contract_path))
    reference = _json(paths["reference_path"])
    reference["_actual_reference_sha256"] = contract["reference_sha256"]
    evaluator_path = task_root / "tools" / "evaluate_banking_lti_ablation_v11.py"
    callback = {"registered": False, "called": False, "error": None}
    phase = "preflight"
    record = {"schema": RECORD_SCHEMA, "status": "running",
        "contract_sha256": expected_contract_sha256,
        "reference_sha256": contract["reference_sha256"],
        "evaluator_sha256": contract["source_hashes"]["tools/evaluate_banking_lti_ablation_v11.py"],
        "v10_memory_launcher_sha256": contract["source_hashes"]["tools/launch_banking_memory_policy_v10.py"],
        "v11_memory_launcher_sha256": contract["source_hashes"]["tools/launch_banking_memory_policy_v11.py"],
        "preflight_output_dir": str(paths["preflight_output_dir"]),
        "evaluation_output_dir": str(paths["evaluation_output_dir"]),
        "memory_fraction_callback_registered": False,
        "memory_fraction_callback_called": False,
        "official_test_accessed_by_launcher": False,
        "started_unix": time.time()}
    started = time.monotonic()
    try:
        preflight_argv = evaluator_argv(contract, mode="preflight",
                                        output_dir=paths["preflight_output_dir"])
        _run_evaluator(evaluator_path, preflight_argv)
        preflight_receipt = _json(paths["preflight_output_dir"] / "preflight.json")
        validate_preflight_receipt(preflight_receipt, reference=reference)
        import torch
        if torch.__version__ != TORCH_VERSION or torch.cuda.is_initialized():
            raise RuntimeError("preflight must finish on pinned PyTorch with CUDA uninitialized")
        v11_launcher = sys.modules.get("launch_banking_memory_policy_v11")
        if v11_launcher is None:
            sys.path.insert(0, str(task_root / "tools"))
            import launch_banking_memory_policy_v11 as v11_launcher
        if sha256_file(Path(v11_launcher.__file__)) != contract["source_hashes"]["tools/launch_banking_memory_policy_v11.py"]:
            raise ValueError("reviewed V11 memory launcher source changed")
        v10_launcher = sys.modules.get("launch_banking_memory_policy_v10")
        if v10_launcher is None:
            import launch_banking_memory_policy_v10 as v10_launcher
        if sha256_file(Path(v10_launcher.__file__)) != contract["source_hashes"]["tools/launch_banking_memory_policy_v10.py"]:
            raise ValueError("reviewed V10 memory helper source changed")
        callback = v11_launcher._install_reviewed_policy(torch, contract["device"], v10_launcher)
        record.update({"preflight_receipt_sha256": sha256_file(paths["preflight_output_dir"] / "preflight.json"),
                       "memory_fraction_callback_registered": bool(callback.get("registered")),
                       "memory_fraction_callback_called": bool(callback.get("called")),
                       "allocator_conf": POLICY, "memory_fraction": MEMORY_FRACTION,
                       "cuda_initialized_before_evaluation": False})
        _atomic_record(paths["record_path"], record, replace=False)

        phase = "evaluation"
        evaluate_argv = evaluator_argv(contract, mode="evaluate",
                                       output_dir=paths["evaluation_output_dir"])
        # Prove that inputs are the same and only mode/output-dir differ.
        def normalize(argv):
            pairs = dict(zip(argv[::2], argv[1::2]))
            pairs["--mode"] = "<MODE>"
            pairs["--output-dir"] = "<OUTPUT>"
            return pairs
        if normalize(preflight_argv) != normalize(evaluate_argv):
            raise RuntimeError("evaluator argv changed outside mode and fresh output")
        _run_evaluator(evaluator_path, evaluate_argv)
        result_path = paths["evaluation_output_dir"] / "development-evaluation.json"
        result = _json(result_path)
        validate_evaluation_result(result, contract=contract,
                                   reference=reference, callback=callback)
        # Recheck every helper/source and model asset after evaluation.
        for relative, expected in contract["source_hashes"].items():
            if sha256_file(task_root / relative) != expected:
                raise ValueError(f"source changed during evaluation: {relative}")
        for field, path_key in (
                ("reference_sha256", "reference_path"),
                ("training_manifest_sha256", "training_manifest_path"),
                ("v10_config_file_sha256", "v10_config_path"),
                ("v11_config_file_sha256", "v11_config_path"),
                ("v10_grid_seal_sha256", "v10_grid_seal_path"),
                ("v10_release_sha256", "v10_release_artifact_path")):
            if sha256_file(paths[path_key]) != contract[field]:
                raise ValueError(f"frozen input changed during evaluation: {field}")
        v10 = sys.modules.get("run_banking_cil_v10")
        if v10 is None:
            import run_banking_cil_v10 as v10
        v10._verify_model_assets(paths["model_dir"], contract["model_assets_sha256"])
        import run_banking_lti_ablation_v11 as v11
        sources = v11.candidate_source_hashes(paths["project_root"])
        if sources != reference["v11_source_hashes"]:
            raise ValueError("V11 training source map changed during evaluation")
        record.update({"status": "complete", "evaluation_result_sha256": sha256_file(result_path),
                       "memory_fraction_callback_called": bool(callback.get("called")),
                       "memory_fraction_callback_error": callback.get("error"),
                       "cuda_initialized_at_exit": bool(torch.cuda.is_initialized()),
                       "official_test_accessed": False, "optimizer_updates": 0})
        if not torch.cuda.is_initialized():
            raise RuntimeError("CUDA did not initialize after policy queueing")
        _atomic_record(paths["record_path"], record, replace=True)
        return 0
    except BaseException as exc:
        record.update({"status": "failed", "phase": phase,
                       "failure": f"{type(exc).__name__}: {exc}",
                       "memory_fraction_callback_registered": bool(callback.get("registered")),
                       "memory_fraction_callback_called": bool(callback.get("called")),
                       "memory_fraction_callback_error": callback.get("error"),
                       "elapsed_wall_seconds": time.monotonic() - started,
                       "official_test_accessed": False, "optimizer_updates": 0})
        if not paths["record_path"].exists():
            _atomic_record(paths["record_path"], record, replace=False)
        else:
            _atomic_record(paths["record_path"], record, replace=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args(argv)
    return launch(args.contract, args.contract_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
