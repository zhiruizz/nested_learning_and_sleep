#!/usr/bin/env python3
"""Build CPU-verified V11 config and per-run launch contracts from real V10 artifacts.

No model/GPU or official-test payload reader is used. ``freeze-config`` needs
the completed V10 main seal and validates its real run/receipt/checkpoint bytes.
``build-contracts`` additionally requires the completed official-test result
and V10 release/foundation artifacts before writing six one-run contracts.
This helper never launches work and never invents missing artifact hashes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

TASK_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = TASK_ROOT / "tools"
for _entry in (TOOLS_DIR, TASK_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import launch_banking_memory_policy_v11 as memory_launcher
import run_banking_lti_ablation_v11 as v11

SEEDS = (17, 29, 43)
ARMS = ("LTI_NO_REWARD", "LTI_CANONICAL_TEACHER")
MAX_WALL_SECONDS = 1800
MAX_FORWARD_CALLS = 30000
MAX_UPDATE_STEPS = 3000
TORCH_VERSION = memory_launcher.TORCH_VERSION


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping) -> str:
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent directory must already exist: {path.parent}")
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                         allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    expected = v11.sha256_bytes(payload.encode("utf-8"))
    if memory_launcher.sha256_file(path) != expected:
        raise IOError("atomic config/contract readback SHA mismatch")
    return expected


def build_v11_config(base_cfg: Mapping[str, Any], *, v10_config_sha256: str,
                     v10_source_lock_sha256: str, actual_v11_source_hashes: Mapping[str, str],
                     baseline_training_seal_sha256: str) -> dict:
    """Create the predeclared child config, deriving shared values from V10."""
    cfg = {
        "schema": v11.SCHEMA,
        "status": "frozen",
        "v10_config_sha256": v10_config_sha256,
        "v10_source_lock_sha256": v10_source_lock_sha256,
        "v10_source_hashes": dict(base_cfg["source_hashes"]),
        "source_hashes": dict(actual_v11_source_hashes),
        "baseline_training_seal_sha256": baseline_training_seal_sha256,
        "training_manifest_sha256": base_cfg["training_manifest_sha256"],
        "model_revision": base_cfg["model_revision"],
        "model_assets_sha256": dict(base_cfg["model_assets_sha256"]),
        "initialization_seed": base_cfg["initialization_seed"],
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "task_id": 1,
        "epochs": base_cfg["epochs_per_task"],
        "batch_size": base_cfg["batch_size"],
        "sleep_lr": base_cfg["sleep_lr"],
        "sleep_weight_decay": base_cfg["sleep_weight_decay"],
        "sleep_clip_norm": base_cfg["sleep_clip_norm"],
        # These define the unchanged Eq.4 proxy weights consumed by the
        # callback. Keeping them in the frozen child config makes the reward
        # semantics explicit and prevents a later base-config edit from
        # silently changing the ablation.
        "lti_semantic_weight": base_cfg["lti_semantic_weight"],
        "lti_edit_weight": base_cfg["lti_edit_weight"],
        "lti_teacher_validity_policy": "canonical_teacher_required_for_reward",
        "objective_version": "v10_eq4_reward_term_ablation_with_kl_retained_v1",
        "test_data_included": False,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "max_forward_calls": MAX_FORWARD_CALLS,
        "max_update_steps": MAX_UPDATE_STEPS,
        "candidate_batch_size": base_cfg["candidate_batch_size"],
    }
    v11.validate_config(cfg, base_cfg,
        actual_source_hashes=actual_v11_source_hashes,
        actual_seal_sha256=baseline_training_seal_sha256,
        actual_manifest_sha256=base_cfg["training_manifest_sha256"])
    return cfg


def _collect_frozen_v10(project_root: Path, model_path: Path, training_pack: Path,
                        v10_config_path: Path, baseline_seal_path: Path,
                        baseline_runs_root: Path) -> dict:
    driver = v11._load_driver()
    base_raw = _json(v10_config_path)
    base_cfg = driver.validate_config(base_raw, phase="main")
    if base_cfg.get("status") != "frozen":
        raise ValueError("V10 config must be frozen before V11 config preparation")
    actual_v11_sources = v11.candidate_source_hashes(project_root)
    source_hashes, source_lock, manifest, train_rows, dev_rows = driver._source_prepare(
        project_root, TASK_ROOT, model_path, training_pack, base_cfg)
    if source_hashes != base_cfg["source_hashes"]:
        raise ValueError("current V10 source files differ from frozen V10 config")
    seal_sha = driver._sha256_file(baseline_seal_path)
    baseline = v11.validate_baseline_seal(baseline_seal_path, baseline_runs_root,
        expected_seal_sha256=seal_sha,
        config_sha256=driver._config_sha(base_cfg), source_lock_sha256=source_lock,
        manifest_sha256=base_cfg["training_manifest_sha256"], source_hashes=source_hashes)
    # `_source_prepare` calls V10 `_load_rows`, which validates the SHA of the
    # manifest file itself against training_manifest_sha256 and validates both
    # role files. `manifest.files.train.sha256` is a different digest (the
    # train JSONL), so it must not be compared to the manifest digest.
    return {"driver": driver, "base_cfg": base_cfg, "v10_config_sha256": driver._config_sha(base_cfg),
        "v10_source_lock_sha256": source_lock, "source_hashes": source_hashes,
        "v11_source_hashes": actual_v11_sources, "manifest": manifest,
        "baseline_seal_sha256": seal_sha, "baseline": baseline,
        "train_rows": len(train_rows), "development_rows": len(dev_rows)}


def freeze_config(args) -> dict:
    project_root = args.project_root.resolve(strict=True)
    values = _collect_frozen_v10(project_root, args.model_path.resolve(strict=True),
        args.training_pack.resolve(strict=True), args.v10_config.resolve(strict=True),
        args.baseline_seal.resolve(strict=True), args.baseline_runs_root.resolve(strict=True))
    child = build_v11_config(values["base_cfg"],
        v10_config_sha256=values["v10_config_sha256"],
        v10_source_lock_sha256=values["v10_source_lock_sha256"],
        actual_v11_source_hashes=values["v11_source_hashes"],
        baseline_training_seal_sha256=values["baseline_seal_sha256"])
    output = args.output_config.resolve()
    config_file_sha = _atomic_json(output, child)
    return {"schema": "banking-cil-v11-config-preparation-v1", "status": "config_frozen",
        "config_path": str(output), "config_file_sha256": config_file_sha,
        "config_semantic_sha256": v11._config_hash(child),
        "baseline_training_seal_sha256": values["baseline_seal_sha256"],
        "source_lock_sha256": v11.canonical_sha(values["v11_source_hashes"]),
        "training_manifest_sha256": values["base_cfg"]["training_manifest_sha256"],
        "train_rows": values["train_rows"], "development_rows": values["development_rows"],
        "official_test_read": False, "model_loaded": False, "optimizer_updates": 0}


def _foundation_paths(release_path: Path, foundation_root: Path, *, cfg: Mapping,
                      v10_config_sha256: str, v10_source_lock_sha256: str,
                      manifest_sha256: str, baseline: Mapping) -> dict[int, dict]:
    driver = v11._load_driver()
    release = _json(release_path)
    if (release.get("schema"), release.get("status"), release.get("config_sha256"),
        release.get("source_lock_sha256"), release.get("training_manifest_sha256")) != (
            "banking-cil-v10-main-release-v1", "ready", v10_config_sha256,
            v10_source_lock_sha256, manifest_sha256):
        raise ValueError("V10 main release is incomplete or has different frozen bindings")
    if memory_launcher.sha256_file(release_path) != baseline["release_artifact_sha256"]:
        raise ValueError("V10 main release bytes do not match the sealed training parent")
    refs = release.get("foundations")
    if not isinstance(refs, Mapping) or set(refs) != {str(seed) for seed in SEEDS}:
        raise ValueError("V10 release must include the actual three seed foundations")
    out = {}
    for seed in SEEDS:
        ref = refs[str(seed)]
        run_dir = Path(ref["run_dir"]).resolve(strict=True)
        if foundation_root.resolve() not in run_dir.parents:
            raise ValueError("V10 foundation directory is outside the declared foundation root")
        receipt_path = run_dir / "foundation-receipt.json"
        receipt_sha = memory_launcher.sha256_file(receipt_path)
        receipt = _json(receipt_path)
        if (receipt_sha, receipt.get("checkpoint_sha256")) != (
            ref.get("receipt_sha256"), ref.get("checkpoint_sha256")):
            raise ValueError(f"foundation seed {seed} receipt/checkpoint hashes differ from release")
        driver._validate_foundation_release(release_path, cfg, v10_config_sha256,
            v10_source_lock_sha256, manifest_sha256, seed, run_dir)
        if (receipt.get("schema"), receipt.get("status"), receipt.get("seed"),
            receipt.get("task_id"), receipt.get("source_lock_sha256"),
            receipt.get("config_sha256"), receipt.get("training_manifest_sha256"),
            receipt.get("test_data_included")) != (
                "banking-cil-v10-foundation-receipt", "qualified", seed, 0,
                v10_source_lock_sha256, v10_config_sha256, manifest_sha256, False):
            raise ValueError(f"foundation seed {seed} receipt has invalid bindings")
        out[seed] = {"run_dir": run_dir, "receipt_sha256": receipt_sha,
                     "checkpoint_sha256": ref["checkpoint_sha256"]}
    for seed in SEEDS:
        if baseline["foundation_checkpoint_sha256_by_seed"][seed] != out[seed]["checkpoint_sha256"] or \
                baseline["foundation_receipt_sha256_by_seed"][seed] != out[seed]["receipt_sha256"]:
            raise ValueError(f"foundation seed {seed} differs between V10 release and sealed main grid")
    return out


def _contract_grid(output_root: Path, *, seeds=SEEDS, arms=ARMS) -> list[tuple[int, str, Path]]:
    root = output_root.resolve(strict=True)
    grid = [(seed, arm, root / f"seed{seed}-{arm.lower().replace('_', '-')}")
            for seed in seeds for arm in arms]
    paths = [row[2] for row in grid]
    if len(set(paths)) != len(paths) or any(path.parent != root or path.exists() for path in paths):
        raise ValueError("V11 run output root must have six fresh direct-child paths")
    return grid


def build_contracts(args) -> dict:
    project_root = args.project_root.resolve(strict=True)
    model_path = args.model_path.resolve(strict=True)
    training_pack = args.training_pack.resolve(strict=True)
    v10_config_path = args.v10_config.resolve(strict=True)
    baseline_seal_path = args.baseline_seal.resolve(strict=True)
    baseline_runs_root = args.baseline_runs_root.resolve(strict=True)
    release_path = args.main_release.resolve(strict=True)
    foundation_root = args.foundation_root.resolve(strict=True)
    evaluation_path = args.v10_test_evaluation.resolve(strict=True)
    config_path = args.v11_config.resolve(strict=True)
    child = _json(config_path)
    values = _collect_frozen_v10(project_root, model_path, training_pack, v10_config_path,
        baseline_seal_path, baseline_runs_root)
    v11.validate_config(child, values["base_cfg"],
        actual_source_hashes=values["v11_source_hashes"],
        actual_seal_sha256=values["baseline_seal_sha256"],
        actual_manifest_sha256=values["base_cfg"]["training_manifest_sha256"])
    if child["baseline_training_seal_sha256"] != values["baseline_seal_sha256"]:
        raise ValueError("frozen V11 child config differs from actual V10 seal")
    foundations = _foundation_paths(release_path, foundation_root, cfg=values["base_cfg"],
        v10_config_sha256=values["v10_config_sha256"],
        v10_source_lock_sha256=values["v10_source_lock_sha256"],
        manifest_sha256=values["base_cfg"]["training_manifest_sha256"],
        baseline=values["baseline"])
    evaluation_binding_contract = {
        "v10_test_evaluation": str(evaluation_path),
        "v10_test_evaluation_sha256": memory_launcher.sha256_file(evaluation_path),
        "v10_test_evaluator_sha256": memory_launcher.sha256_file(TOOLS_DIR / "evaluate_banking_test_v10.py"),
        "v10_source_lock_sha256": values["v10_source_lock_sha256"],
        "training_manifest_sha256": values["base_cfg"]["training_manifest_sha256"],
        "baseline_seal_sha256": values["baseline_seal_sha256"],
        "baseline_seal": str(baseline_seal_path),
    }
    test_binding = memory_launcher._validate_completed_v10_test(
        evaluation_binding_contract, evaluation_path, task_root=TASK_ROOT,
        v11_module=v11, v10_config=v10_config_path)
    config_semantic_sha = v11._config_hash(child)
    v11_lock = v11.canonical_sha(values["v11_source_hashes"])
    driver_sha = memory_launcher.sha256_file(TOOLS_DIR / "launch_banking_memory_policy_v10.py")
    launcher_sha = memory_launcher.sha256_file(TOOLS_DIR / "launch_banking_memory_policy_v11.py")
    script_sha = memory_launcher.sha256_file(Path(v11.__file__).resolve())
    output_root = args.output_root.resolve(strict=True)
    grid = _contract_grid(output_root)
    contracts_dir = args.contract_dir.resolve()
    if contracts_dir.exists() or not contracts_dir.parent.is_dir():
        raise ValueError("contract directory must be a fresh child of an existing parent")
    generated = []
    for seed, arm, run_output in grid:
        contract = {
            "schema": memory_launcher.SCHEMA,
            "launcher_sha256": launcher_sha,
            "policy_launcher_v10_sha256": driver_sha,
            "project_root": str(project_root), "task_root": str(TASK_ROOT),
            "model_path": str(model_path), "training_pack": str(training_pack),
            "v10_config_path": str(v10_config_path), "config_path": str(config_path),
            "baseline_seal": str(baseline_seal_path), "baseline_runs_root": str(baseline_runs_root),
            "foundation_dir": str(foundations[seed]["run_dir"]),
            "v10_test_evaluation": str(evaluation_path),
            "v10_test_evaluation_sha256": evaluation_binding_contract["v10_test_evaluation_sha256"],
            "v10_test_evaluator_sha256": evaluation_binding_contract["v10_test_evaluator_sha256"],
            "output_dir": str(run_output), "seed": seed, "arm": arm,
            "script_key": memory_launcher.V11_SCRIPT, "script_sha256": script_sha,
            "launch_record_path": str(output_root / "launch-records" / f"seed{seed}-{arm.lower().replace('_', '-')}.json"),
            "allocator_conf": memory_launcher.POLICY,
            "memory_fraction": memory_launcher.MEMORY_FRACTION,
            "device": values["base_cfg"]["device"], "torch_version": TORCH_VERSION,
            "config_sha256": config_semantic_sha,
            "v11_source_lock_sha256": v11_lock,
            "v10_source_lock_sha256": values["v10_source_lock_sha256"],
            "training_manifest_sha256": values["base_cfg"]["training_manifest_sha256"],
            "baseline_seal_sha256": values["baseline_seal_sha256"],
            "foundation_checkpoint_sha256": foundations[seed]["checkpoint_sha256"],
        }
        if set(contract) != memory_launcher.CONTRACT_KEYS:
            raise RuntimeError("internal V11 launcher contract schema drift")
        generated.append((contracts_dir / f"seed{seed}-{arm.lower().replace('_', '-')}.json", contract))
    contracts_dir.mkdir()
    hashes = {str(path): _atomic_json(path, payload) for path, payload in generated}
    return {"schema": "banking-cil-v11-launch-contract-bundle-v1", "status": "contracts_ready",
        "contracts": hashes, "contract_count": len(hashes),
        "official_test_evaluation_sha256": evaluation_binding_contract["v10_test_evaluation_sha256"],
        "official_test_binding": test_binding, "v10_seal_sha256": values["baseline_seal_sha256"],
        "config_sha256": config_semantic_sha, "no_model_loaded": True,
        "optimizer_updates": 0, "test_payload_read": False, "launcher_invoked": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    freeze = sub.add_parser("freeze-config")
    contracts = sub.add_parser("build-contracts")
    for command in (freeze, contracts):
        command.add_argument("--project-root", type=Path, required=True)
        command.add_argument("--model-path", type=Path, required=True)
        command.add_argument("--training-pack", type=Path, required=True)
        command.add_argument("--v10-config", type=Path, required=True)
        command.add_argument("--baseline-seal", type=Path, required=True)
        command.add_argument("--baseline-runs-root", type=Path, required=True)
    freeze.add_argument("--output-config", type=Path, required=True)
    contracts.add_argument("--v11-config", type=Path, required=True)
    contracts.add_argument("--main-release", type=Path, required=True)
    contracts.add_argument("--foundation-root", type=Path, required=True)
    contracts.add_argument("--v10-test-evaluation", type=Path, required=True)
    contracts.add_argument("--output-root", type=Path, required=True)
    contracts.add_argument("--contract-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze_config(args) if args.phase == "freeze-config" else build_contracts(args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
