"""Build a portable, hash-bound four-context map for the v9 readout probe.

This reads only frozen v8 metadata and artifact hashes. It never deserializes a
checkpoint, loads a model, or opens CLINC examples (including test data).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_child_path(root: Path, relative: str) -> Path:
    """Resolve an artifact path declared by a receipt; reject absolute/traversal."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("receipt artifact path must be a nonempty relative path")
    rel = Path(relative)
    if rel.is_absolute() or rel.drive:
        raise ValueError("receipt artifact path must be relative")
    base = root.resolve()
    candidate = (base / rel).resolve()
    if candidate == base or base not in candidate.parents:
        raise ValueError("receipt artifact path escapes its declared root")
    return candidate


def _unique_task1_checkpoint(lines, run_dir: Path) -> tuple[dict, Path]:
    """Find the one locked task-1 step-188 safe checkpoint without loading it."""
    matches = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("event") == "safe_checkpoint" and row.get("task_id") == 1
                and row.get("step") == 188):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"expected one task-1 step-188 safe checkpoint, found {len(matches)}")
    row = matches[0]
    if not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64:
        raise ValueError("task-1 checkpoint event lacks a SHA256")
    path = _safe_child_path(run_dir, row.get("path"))
    return row, path


def _import_full(project_root: Path):
    tools_dir = project_root / "tools"
    src_dir = project_root / "src"
    if not (tools_dir / "run_clinc_cil_v8_full.py").is_file():
        raise FileNotFoundError("project tools/run_clinc_cil_v8_full.py not found")
    sys.path.insert(0, str(tools_dir.resolve()))
    sys.path.insert(0, str(src_dir.resolve()))
    name = "_clinc_full_for_context_map"
    spec = importlib.util.spec_from_file_location(name, tools_dir / "run_clinc_cil_v8_full.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot import the project's frozen full runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != (tools_dir / "run_clinc_cil_v8_full.py").resolve():
        raise RuntimeError("full runner import escaped the explicit project root")
    return module


def _git_provenance(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root,
                                     stderr=subprocess.STDOUT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root,
                                         stderr=subprocess.STDOUT, text=True).strip())
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit.lower()):
        raise ValueError("could not record a full local Git commit")
    return commit, dirty


def _verify_foundation(full, grid_root: Path, *, master_sha: str, manifest_sha: str,
                       source_lock_sha: str, cfg: dict, seal: dict) -> tuple[Path, Path, dict]:
    foundation_dir = grid_root / "foundations" / "seed17"
    receipt_path = foundation_dir / "foundation-receipt.json"
    receipt = _load(receipt_path)
    expected_child = full._child_config_sha(master_sha, "foundation", 17, None)
    expected = {
        "schema": full.FOUNDATION_SCHEMA, "status": "complete", "seed": 17,
        "acquisition_qualified": True, "test_data_included": False,
        "master_config_sha256": master_sha, "training_manifest_sha256": manifest_sha,
        "source_lock_sha256": source_lock_sha, "child_config_sha256": expected_child,
        "model_revision": cfg["model_revision"],
        "model_assets_sha256": cfg["model_assets_sha256"], "tasks_completed": 1,
        "wake_ticks": 188,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"seed-17 foundation receipt binding mismatch: {key}")
    accuracy = float(receipt.get("task0_clean_current_accuracy", -1))
    if not math.isfinite(accuracy) or accuracy < cfg["task0_clean_accuracy_gate"]:
        raise ValueError("seed-17 task-0 foundation did not pass the frozen acquisition gate")
    tasks = receipt.get("task_metrics")
    if not isinstance(tasks, list) or len(tasks) != 1 or tasks[0].get("task_id") != 0:
        raise ValueError("foundation receipt must contain only task-0 metrics")
    if sorted(int(v) for v in receipt.get("pending_clocks", {}).values()) != [12, 60, 188, 188]:
        raise ValueError("foundation terminal clocks differ from the frozen task-0 boundary")
    cp = _safe_child_path(foundation_dir, receipt.get("final_checkpoint", ""))
    cp_sha = _sha(cp)
    if cp_sha != receipt.get("final_checkpoint_sha256"):
        raise ValueError("foundation checkpoint file hash mismatch")
    sealed = [x for x in seal.get("verified_foundations", []) if x.get("seed") == 17]
    if len(sealed) != 1 or sealed[0].get("checkpoint_sha256") != cp_sha:
        raise ValueError("seal does not uniquely bind the seed-17 foundation checkpoint")
    seal_receipt = Path(sealed[0].get("receipt_path", "")).resolve()
    if seal_receipt != receipt_path.resolve() or sealed[0].get("receipt_sha256") != _sha(receipt_path):
        raise ValueError("seal does not bind the actual seed-17 foundation receipt")
    return cp, receipt_path, receipt


def _verify_continuation(full, grid_root: Path, arm: str, name: str, foundation_sha: str,
                         *, master_sha: str, manifest_sha: str, source_lock_sha: str,
                         cfg: dict, seal: dict) -> dict:
    run_dir = grid_root / "runs" / "seed17" / arm
    receipt_path = run_dir / "run-receipt.json"
    metrics_path = run_dir / "metrics.jsonl"
    receipt = _load(receipt_path)
    expected_child = full._child_config_sha(master_sha, "continuation", 17, arm)
    expected = {
        "schema": full.RECEIPT_SCHEMA, "status": "complete", "seed": 17, "arm": arm,
        "tasks_completed": 10, "acquisition_qualified": True, "test_data_included": False,
        "master_config_sha256": master_sha, "training_manifest_sha256": manifest_sha,
        "source_lock_sha256": source_lock_sha, "child_config_sha256": expected_child,
        "foundation_checkpoint_sha256": foundation_sha,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"{name} terminal receipt binding mismatch: {key}")
    tasks = receipt.get("task_metrics")
    if not isinstance(tasks, list) or [x.get("task_id") for x in tasks] != list(range(10)):
        raise ValueError(f"{name} receipt must contain all ten completed task metrics")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing metrics file for {name}")
    receipt_sha = _sha(receipt_path)
    event, checkpoint = _unique_task1_checkpoint(metrics_path.read_text(encoding="utf-8").splitlines(), run_dir)
    if event["path"].replace("/", "\\") != "checkpoints\\task01-next0189-safe.pt":
        raise ValueError(f"{name} task-1 checkpoint is not the locked step-188 boundary")
    checkpoint_sha = _sha(checkpoint)
    if checkpoint_sha != event["sha256"]:
        raise ValueError(f"{name} task-1 checkpoint byte hash mismatch")
    sealed = [r for r in seal.get("runs", []) if r.get("seed") == 17 and r.get("arm") == arm]
    if len(sealed) != 1 or sealed[0].get("final_checkpoint_sha256") != receipt.get("final_checkpoint_sha256"):
        raise ValueError(f"seal does not uniquely bind {name} completed run")
    verified = [r for r in seal.get("verified_run_files", [])
                if r.get("seed") == 17 and r.get("arm") == arm]
    if len(verified) != 1 or verified[0].get("run_receipt_sha256") != receipt_sha:
        raise ValueError(f"seal does not bind the actual {name} receipt bytes")
    return {
        "name": name, "kind": "checkpoint", "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha, "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha, "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": _sha(metrics_path), "expected_mode": "continuation",
        "expected_arm": arm, "expected_tasks_completed": 2,
        "expected_child_config_sha256": expected_child,
        "expected_foundation_checkpoint_sha256": foundation_sha,
    }


def build_context_map(*, project_root: Path, master_config: Path, manifest: Path,
                      grid_root: Path, model_path: Path) -> dict:
    project_root = project_root.resolve()
    master_config, manifest = master_config.resolve(), manifest.resolve()
    grid_root, model_path = grid_root.resolve(), model_path.resolve()
    full = _import_full(project_root)
    cfg = full.validate_master_config(_load(master_config))
    master_sha, manifest_sha = _sha(master_config), _sha(manifest)
    if cfg["training_manifest_sha256"] != manifest_sha:
        raise ValueError("training manifest bytes do not match the frozen master config")
    full._verify_asset_hashes(model_path, cfg["model_assets_sha256"])
    source_hashes = full._source_hashes(project_root, project_root / "tools")
    source_lock_sha = full.canonical_sha(source_hashes)
    seal_path = grid_root / "main-six-seal.json"
    seal = _load(seal_path)
    if seal.get("schema") != full.SEAL_SCHEMA or seal.get("complete") is not True or seal.get("test_data_included") is not False:
        raise ValueError("grid is not a complete test-isolated v8 main seal")
    for key, expected in (("master_config_sha256", master_sha),
                          ("training_manifest_sha256", manifest_sha),
                          ("source_lock_sha256", source_lock_sha)):
        if seal.get(key) != expected:
            raise ValueError(f"main seal source/config binding mismatch: {key}")
    if seal.get("source_hashes") != source_hashes:
        raise ValueError("current frozen v8 source files differ from the seal")
    full.validate_main_grid_receipts(seal.get("runs", []), master_sha=master_sha,
        manifest_sha=manifest_sha, source_lock_sha=source_lock_sha,
        inner_cycles_per_event=cfg["inner_cycles_per_event"], objective_version=cfg["objective_version"])

    foundation_cp, foundation_receipt_path, foundation = _verify_foundation(
        full, grid_root, master_sha=master_sha, manifest_sha=manifest_sha,
        source_lock_sha=source_lock_sha, cfg=cfg, seal=seal)
    foundation_sha = _sha(foundation_cp)
    frow = {"name": "TASK0_FOUNDATION", "kind": "checkpoint",
        "checkpoint_path": str(foundation_cp.resolve()), "checkpoint_sha256": foundation_sha,
        "receipt_path": str(foundation_receipt_path.resolve()),
        "receipt_sha256": _sha(foundation_receipt_path), "expected_mode": "foundation",
        "expected_arm": None, "expected_tasks_completed": 1,
        "expected_child_config_sha256": full._child_config_sha(master_sha, "foundation", 17, None),
        "expected_foundation_checkpoint_sha256": None}
    contexts = [{"name": "FROZEN_BASE", "kind": "zero_adapter"}, frow]
    contexts.append(_verify_continuation(full, grid_root, "HOPE", "HOPE_TASK1", foundation_sha,
        master_sha=master_sha, manifest_sha=manifest_sha, source_lock_sha=source_lock_sha, cfg=cfg, seal=seal))
    contexts.append(_verify_continuation(full, grid_root, "HOPE_SLEEP_FIXEDMIX", "SLEEP_TASK1", foundation_sha,
        master_sha=master_sha, manifest_sha=manifest_sha, source_lock_sha=source_lock_sha, cfg=cfg, seal=seal))
    commit, dirty = _git_provenance(project_root)
    seal_sha = _sha(seal_path)
    return {
        "schema": "clinc-v9-readout-contexts-v1", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True, "official_test_used_for_selection": False,
        "test_payload_access_permitted": False, "seed": 17,
        "master_config_path": str(master_config), "master_config_sha256": master_sha,
        "training_manifest_sha256": manifest_sha, "source_lock_sha256": source_lock_sha,
        "frozen_training_source_hashes": source_hashes, "v8_public_commit": commit,
        "v8_public_tree_dirty": dirty, "v8_main_seal_sha256": seal_sha,
        "model_revision": cfg["model_revision"], "model_assets_sha256": cfg["model_assets_sha256"],
        "selection": {"class_order_seed": 17, "tasks": [0, 1], "rows_per_class": 2,
            "total_rows": 60, "policy": "Stable ID sort per class on normalized train-disjoint development; do not choose by predictions"},
        "readouts": ["v8_free_generation", "all150_candidate_sum_log_likelihood",
                     "all150_candidate_length_mean_log_likelihood"],
        "limits": {"max_full_backbone_forwards": 10000, "max_wall_seconds": 1800,
                   "max_optimizer_steps": 0}, "contexts": contexts,
        "scope": "Exploratory readout diagnosis on existing validation; candidate labels and added inference compute are extra access, not a replacement for frozen v8 primary scores.",
        "provenance": {"local_commit": commit, "local_tree_dirty": dirty,
            "raw_main_seal_sha256": seal_sha, "main_seal_path": str(seal_path),
            "foundation_task0_accuracy": foundation["task0_clean_current_accuracy"],
            "foundation_checkpoint_sha256": foundation_sha,
            "optimizer_or_test_payload_read": False},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--master-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--grid-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite context map: {out}")
    result = build_context_map(project_root=args.project_root, master_config=args.master_config,
        manifest=args.manifest, grid_root=args.grid_root, model_path=args.model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # Exclusive creation prevents a race from replacing an existing map.
    with out.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"status": "prepared", "output": str(out),
        "contexts": [row["name"] for row in result["contexts"]],
        "seal_sha256": result["v8_main_seal_sha256"], "source_lock_sha256": result["source_lock_sha256"]},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
