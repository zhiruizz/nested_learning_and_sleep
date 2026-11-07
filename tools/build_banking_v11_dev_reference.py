"""Build the strict 15-context V11 development reference from sealed artifacts.

This is a CPU-only artifact binder. It does not read the official test set,
load a model, score examples, or initialize CUDA. The evaluator's locked-input
preflight is rerun before the output file is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _v11_run_dir(root: Path, seed: int, arm: str) -> Path:
    return root / f"seed{seed}-{arm.lower().replace('_', '-')}"


def assemble_context_rows(*, v10_root: Path, v11_root: Path,
                          baseline: Mapping, foundation_refs: Mapping,
                          seal_sha256: str) -> list[dict]:
    """Create the exact evaluator row schema from already verified indexes.

    Artifact bytes and semantics are checked by the evaluator's subsequent
    verify_context_artifacts/validate_locked_inputs pass.
    """
    from evaluate_banking_lti_ablation_v11 import ARMS, SEEDS, V10_ARMS, V11_ARMS

    if set(foundation_refs) != {str(seed) for seed in SEEDS}:
        raise ValueError("foundation references must cover all three seeds")
    rows = []
    for seed in SEEDS:
        foundation = foundation_refs[str(seed)]
        if set(foundation) != {"receipt_sha256", "checkpoint_sha256"}:
            raise ValueError("malformed foundation reference")
        for arm in ARMS:
            family = "v10" if arm in V10_ARMS else "v11"
            if family == "v10":
                run = baseline["runs"].get((seed, arm))
                if not isinstance(run, Mapping):
                    raise ValueError(f"missing sealed V10 run seed={seed} arm={arm}")
                run_dir = Path(run["run_dir"]).resolve()
                root = v10_root.resolve()
                cp_rel = str(run["task1_checkpoint"])
                cp_sha = run.get("task1_checkpoint_sha256")
                receipt = read_json(run_dir / "run-receipt.json")
                task_rows = receipt.get("task_metrics")
                step = _task1_step(task_rows)
                comparison_sha = None
            else:
                run_dir = _v11_run_dir(v11_root.resolve(), seed, arm)
                root = v11_root.resolve()
                receipt_path = run_dir / "run-receipt.json"
                protocol_path = run_dir / "protocol.json"
                cp_rel = "checkpoints/task01-final.pt"
                cp_path = run_dir / cp_rel
                if not (receipt_path.is_file() and protocol_path.is_file() and cp_path.is_file()):
                    raise FileNotFoundError(f"incomplete V11 context: seed={seed} arm={arm}")
                receipt = read_json(receipt_path)
                task_rows = receipt.get("task_metrics")
                step = _task1_step(task_rows)
                cp_sha = sha256_file(cp_path)
                comparison_sha = baseline["runs"][(seed, "SLEEP_ADDITIVE")]["task1_checkpoint_sha256"]
            if not root.is_dir() or root not in run_dir.parents:
                raise ValueError("run directory must be a child of its declared family root")
            protocol_path = run_dir / "protocol.json"
            receipt_path = run_dir / "run-receipt.json"
            cp_path = run_dir / cp_rel
            if not all(path.is_file() for path in (protocol_path, receipt_path, cp_path)):
                raise FileNotFoundError(f"missing fixed context artifacts: seed={seed} arm={arm}")
            row = {
                "seed": seed, "family": family, "arm": arm,
                "run_dir": _relative(root, run_dir),
                "protocol_path": _relative(root, protocol_path),
                "protocol_sha256": sha256_file(protocol_path),
                "receipt_path": _relative(root, receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "checkpoint_path": _relative(root, cp_path),
                "checkpoint_sha256": sha256_file(cp_path),
                "task_id": 1, "step": step,
                "foundation_receipt_sha256": foundation["receipt_sha256"],
                "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
                "baseline_training_seal_sha256": seal_sha256,
                "comparison_v10_sleep_task1_checkpoint_sha256": comparison_sha,
            }
            if cp_sha != row["checkpoint_sha256"]:
                raise ValueError("V10 baseline endpoint checkpoint hash mismatch")
            rows.append(row)
    return rows


def _task1_step(task_rows) -> int:
    if not isinstance(task_rows, list):
        raise ValueError("task endpoint metrics are missing")
    matches = [row for row in task_rows if isinstance(row, Mapping) and row.get("task_id") == 1]
    if len(matches) != 1:
        raise ValueError("expected exactly one task-1 endpoint metric")
    row = matches[0]
    step = row.get("global_wake_tick")
    if type(step) is not int or step < 1:
        raise ValueError("task-1 endpoint has an invalid global wake tick")
    return step


def _path_map_sha(path: Path) -> str:
    return sha256_file(path)


def build_reference(*, task_root: Path, project_root: Path, model_dir: Path,
                    manifest_path: Path, v10_config_path: Path,
                    v10_seal_path: Path, v10_release_path: Path,
                    v10_runs_root: Path, v11_config_path: Path,
                    v11_runs_root: Path, foundation_root: Path) -> dict:
    """Verify source/config/seal bindings and build the frozen 3x5 reference."""
    task_root, project_root = task_root.resolve(), project_root.resolve()
    tools_path = task_root / "tools"
    for candidate in (task_root, tools_path):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import evaluate_banking_lti_ablation_v11 as evaluator
    import run_banking_cil_v10 as v10
    import run_banking_lti_ablation_v11 as v11

    manifest_sha = sha256_file(manifest_path)
    manifest = read_json(manifest_path)
    import banking_cil_common_v10 as common
    common.validate_manifest(manifest)
    v10_raw = read_json(v10_config_path)
    v10_cfg = v10.validate_config(v10_raw, phase="main")
    v10_sources = v10._source_hashes(project_root, task_root)
    v10._verify_source_lock(v10_cfg, v10_sources)
    v10_cfg_sha = v10._config_sha(v10_cfg)
    v10_lock = v10._canonical_sha(v10_sources)

    v11_sources = v11.candidate_source_hashes(project_root)
    evaluator.verify_source_map(v11_sources, project_root=project_root, task_root=task_root)
    seal_sha = sha256_file(v10_seal_path)
    v11_raw = read_json(v11_config_path)
    v11_cfg = v11.validate_config(v11_raw, v10_cfg,
        actual_source_hashes=v11_sources, actual_seal_sha256=seal_sha,
        actual_manifest_sha256=manifest_sha)
    v11_cfg_sha = v11._config_hash(v11_cfg)
    v11_lock = v11.canonical_sha(v11_sources)
    release_sha = sha256_file(v10_release_path)
    release = read_json(v10_release_path)
    foundation_refs = {}
    for seed in (17, 29, 43):
        record = release.get("foundations", {}).get(str(seed))
        if not isinstance(record, Mapping):
            raise ValueError(f"release is missing qualified foundation seed {seed}")
        foundation_refs[str(seed)] = {
            "receipt_sha256": record.get("receipt_sha256"),
            "checkpoint_sha256": record.get("checkpoint_sha256"),
        }
    baseline = v11.validate_baseline_seal(Path(v10_seal_path), Path(v10_runs_root),
        expected_seal_sha256=seal_sha, config_sha256=v10_cfg_sha,
        source_lock_sha256=v10_lock, manifest_sha256=manifest_sha,
        source_hashes=v10_sources)
    rows = assemble_context_rows(v10_root=Path(v10_runs_root), v11_root=Path(v11_runs_root),
        baseline=baseline, foundation_refs=foundation_refs, seal_sha256=seal_sha)
    model_assets = v10_cfg["model_assets_sha256"]
    reference = {
        "schema": evaluator.REFERENCE_SCHEMA, "status": "frozen",
        "seeds": list(evaluator.SEEDS), "arms": list(evaluator.ARMS),
        "model_revision": v10_cfg["model_revision"],
        "model_assets_sha256": model_assets,
        "training_manifest_sha256": manifest_sha,
        "evaluation_policy": {
            "cohort": "all_10_development_rows_per_class_in_seed_task0_and_task1_blocks",
            "rows_per_seed": evaluator.TASK01_ROWS, "classes_per_seed": evaluator.CLASSES_PER_TASK * 2,
            "primary": "global_77_sum_log_likelihood",
            "secondary": ["global_77_length_mean_log_likelihood", "unconstrained_greedy_generation"],
            "candidate_label_mask": None, "max_sequence_length": evaluator.MAX_SEQ,
            "candidate_batch_size": evaluator.CANDIDATE_BATCH,
            "max_new_tokens": evaluator.MAX_NEW_TOKENS,
            "max_backbone_forwards": evaluator.MAX_BACKBONE_FORWARDS,
            "max_wall_seconds": evaluator.MAX_WALL_SECONDS, "optimizer_updates": 0,
        },
        "v10_config_file_sha256": sha256_file(v10_config_path), "v10_config_sha256": v10_cfg_sha,
        "v10_source_lock_sha256": v10_lock, "v10_source_hashes": v10_sources,
        "v10_grid_seal_sha256": seal_sha, "v10_release_sha256": release_sha,
        "foundation_refs": foundation_refs,
        "v11_config_file_sha256": sha256_file(v11_config_path), "v11_config_sha256": v11_cfg_sha,
        "v11_source_lock_sha256": v11_lock, "v11_source_hashes": v11_sources,
        "evaluator_sha256": sha256_file(Path(evaluator.__file__)), "contexts": rows,
    }
    evaluator.validate_reference(reference)
    # This CPU-only gate checks configs, source maps, all foundations, V10 seal,
    # all 15 artifact hashes/lineage fields, and deserializes the bound CPs.
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before reference preflight")
    evaluator.validate_locked_inputs(reference=reference,
        evaluator_path=Path(evaluator.__file__), v10_config_path=v10_config_path,
        v11_config_path=v11_config_path, manifest_path=manifest_path,
        v10_seal_path=v10_seal_path, v10_release_path=v10_release_path,
        v10_runs_root=v10_runs_root, v11_runs_root=v11_runs_root,
        foundation_root=foundation_root, project_root=project_root,
        model_dir=model_dir, task_root=task_root, torch=torch)
    if torch.cuda.is_initialized():
        raise RuntimeError("reference preflight initialized CUDA unexpectedly")
    return reference


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "project-root", "model-dir", "manifest",
                 "v10-config", "v10-seal", "v10-release", "v10-runs-root",
                 "v11-config", "v11-runs-root", "foundation-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite reference output: {out}")
    if not out.parent.is_dir():
        raise FileNotFoundError(f"output parent must already exist: {out.parent}")
    ref = build_reference(task_root=args.task_root, project_root=args.project_root,
        model_dir=args.model_dir, manifest_path=args.manifest,
        v10_config_path=args.v10_config, v10_seal_path=args.v10_seal,
        v10_release_path=args.v10_release, v10_runs_root=args.v10_runs_root,
        v11_config_path=args.v11_config, v11_runs_root=args.v11_runs_root,
        foundation_root=args.foundation_root)
    data = (json.dumps(ref, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")
    # Exclusive create ensures an existing/competing output cannot be replaced.
    with out.open("xb") as stream:
        stream.write(data)
        stream.flush()
    print(json.dumps({"status": "reference_ready", "output": str(out),
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "contexts": len(ref["contexts"]), "official_test_accessed": False,
                      "model_loaded": False, "optimizer_updates": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
