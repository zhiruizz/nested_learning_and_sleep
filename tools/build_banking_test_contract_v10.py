#!/usr/bin/env python3
"""Build a frozen Banking77 test-evaluation contract after the exact 9-run seal.

This CPU-only builder validates train/dev inputs, main release, receipts and
checkpoint bytes, then writes the evaluator's contract. It never reads or
downloads the official test payload and never loads PyTorch/CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(ch in "0123456789abcdef" for ch in value.lower()))


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _expected_policy(evaluator) -> dict:
    return {
        "primary": "global_77_sum_log_likelihood",
        "secondary": ["global_77_length_mean_log_likelihood", "unconstrained_greedy_generation"],
        "include_greedy": True,
        "max_sequence_length": evaluator.EVAL_MAX_SEQUENCE_LENGTH,
        "candidate_batch_size": evaluator.EVAL_CANDIDATE_BATCH_SIZE,
        "max_new_tokens": evaluator.EVAL_GREEDY_MAX_NEW_TOKENS,
        "primary_includes_every_official_row": True,
        "secondary_subset": "exclude_normalized_text_overlap_with_training_and_development",
        "secondary_overlap_roles": ["train", "development"],
        "secondary_overlap_rule": "exact_common_normalize_text_match_in_either_role",
        "max_backbone_forwards": evaluator.MAX_FORWARDS,
        "max_wall_seconds": evaluator.MAX_WALL_SECONDS,
        "optimizer_updates": 0,
    }


def make_contract(*, evaluator_sha256: str, config: Mapping,
                  config_file_sha256: str, source_lock_sha256: str,
                  training_manifest_sha256: str, training_seal_sha256: str,
                  main_release_sha256: str, foundation_refs: Mapping[str, Mapping],
                  evaluator) -> dict:
    """Pure constructor for the evaluator's exact, fixed test contract."""
    if not all(_is_sha256(x) for x in (
            evaluator_sha256, config_file_sha256, source_lock_sha256,
            training_manifest_sha256, training_seal_sha256, main_release_sha256)):
        raise ValueError("all contract hashes must be 64-character SHA-256 strings")
    if set(foundation_refs) != {"17", "29", "43"}:
        raise ValueError("contract requires exactly three qualified foundation references")
    refs = {}
    for seed in ("17", "29", "43"):
        row = foundation_refs[seed]
        if set(row) != {"receipt_sha256", "checkpoint_sha256"}:
            raise ValueError("foundation reference fields differ from evaluator schema")
        if any(not _is_sha256(row[k]) for k in row):
            raise ValueError("foundation receipt/checkpoint references must be SHA-256")
        refs[seed] = dict(row)
    contract = {
        "schema": evaluator.CONTRACT_SCHEMA,
        "evaluator_sha256": evaluator_sha256,
        "training_config_sha256": config["_logical_sha256"],
        "training_config_file_sha256": config_file_sha256,
        "source_lock_sha256": source_lock_sha256,
        "source_hashes": config["source_hashes"],
        "model_revision": config["model_revision"],
        "model_assets_sha256": config["model_assets_sha256"],
        "training_manifest_sha256": training_manifest_sha256,
        "training_seal_sha256": training_seal_sha256,
        "main_release_sha256": main_release_sha256,
        "foundation_refs": refs,
        "official_test": {
            "revision": evaluator.EXPECTED_TEST_REVISION,
            "bytes": evaluator.EXPECTED_TEST_BYTES,
            "sha256": evaluator.EXPECTED_TEST_SHA256,
            "rows": evaluator.EXPECTED_TEST_ROWS,
            "classes": 77,
            "rows_per_class": evaluator.EXPECTED_ROWS_PER_CLASS,
        },
        "evaluation_policy": _expected_policy(evaluator),
        "training_manifest_excludes_test": True,
    }
    # Assert the artifact is byte-schema compatible before writing it.
    evaluator.validate_evaluation_contract(
        contract, evaluator_sha256=evaluator_sha256, config=config,
        config_file_sha256=config_file_sha256,
        source_lock_sha256=source_lock_sha256,
        training_manifest_sha256=training_manifest_sha256,
        training_seal_sha256=training_seal_sha256,
        main_release_sha256=main_release_sha256,
        foundation_refs=refs)
    return contract


def build(args) -> dict:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("contract output already exists; refusing overwrite")
    project_root = args.project_root.expanduser().resolve(strict=True)
    task_root = args.task_root.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    manifest_path = args.training_manifest.expanduser().resolve(strict=True)
    training_root = args.training_root.expanduser().resolve(strict=True)
    seal_path = args.seal.expanduser().resolve(strict=True)
    release_path = args.main_release_artifact.expanduser().resolve(strict=True)
    foundation_root = args.foundation_root.expanduser().resolve(strict=True)
    evaluator_path = (task_root / "tools" / "evaluate_banking_test_v10.py").resolve(strict=True)
    if str(task_root / "tools") not in sys.path:
        sys.path.insert(0, str(task_root / "tools"))
    import run_banking_cil_v10 as driver
    import banking_cil_common_v10 as common
    import evaluate_banking_test_v10 as evaluator

    raw_cfg = _read_json(config_path)
    cfg = driver.validate_config(raw_cfg, phase="main")
    cfg_sha = driver._config_sha(cfg)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != cfg["training_manifest_sha256"]:
        raise ValueError("train/dev manifest bytes differ from frozen training config")
    manifest = _read_json(manifest_path)
    common.validate_manifest(manifest)
    source_hashes, source_lock, _manifest, train_rows, dev_rows = driver._source_prepare(
        project_root, task_root, model_dir, manifest_path, cfg)
    if len(train_rows) != 9229 or len(dev_rows) != 770:
        raise ValueError("locked Banking77 train/dev partition has unexpected row counts")
    if source_hashes != cfg["source_hashes"]:
        raise ValueError("verified source graph differs from frozen config")

    release_sha = sha256_file(release_path)
    release = _read_json(release_path)
    foundations = release.get("foundations")
    if not isinstance(foundations, dict) or set(foundations) != {"17", "29", "43"}:
        raise ValueError("main release must bind all three foundation receipts")
    if (release.get("schema"), release.get("status"), release.get("config_sha256"),
        release.get("source_lock_sha256"), release.get("training_manifest_sha256")) != (
            "banking-cil-v10-main-release-v1", "ready", cfg_sha, source_lock, manifest_sha):
        raise ValueError("main release source/config/data binding mismatch")
    refs = {}
    for seed in (17, 29, 43):
        record = foundations[str(seed)]
        run_dir = Path(record["run_dir"]).expanduser().resolve(strict=True)
        if run_dir != foundation_root / f"seed{seed}":
            raise ValueError(f"foundation seed {seed} is outside the fixed foundation root")
        # The frozen driver's validator checks qualification gates, receipt bytes,
        # checkpoint bytes, and the actual release parent without deserializing.
        driver._validate_foundation_release(
            release_path, cfg, cfg_sha, source_lock, manifest_sha, seed, run_dir)
        refs[str(seed)] = {
            "receipt_sha256": record["receipt_sha256"],
            "checkpoint_sha256": record["checkpoint_sha256"],
        }

    seal_sha = sha256_file(seal_path)
    seal = _read_json(seal_path)
    rows = evaluator.validate_training_seal(
        seal, config_sha256=cfg_sha, source_lock_sha256=source_lock,
        training_manifest_sha256=manifest_sha)
    # Verify each completed run's protocol, receipt, foundation parent and final
    # checkpoint bytes. This is stdlib file verification; evaluator later performs
    # its own CPU checkpoint deserialization before opening test CSV bytes.
    evaluator.verify_run_artifacts(
        rows, training_root, config_sha256=cfg_sha,
        source_lock_sha256=source_lock, training_manifest_sha256=manifest_sha,
        foundation_refs=refs, source_hashes=source_hashes,
        main_release_sha256=release_sha)

    contract = make_contract(
        evaluator_sha256=sha256_file(evaluator_path),
        # validate_config returns the semantic config without this derived key;
        # the evaluator validator expects it alongside (not inside) that config.
        config={**cfg, "_logical_sha256": cfg_sha},
        config_file_sha256=sha256_file(config_path),
        source_lock_sha256=source_lock,
        training_manifest_sha256=manifest_sha,
        training_seal_sha256=seal_sha, main_release_sha256=release_sha,
        foundation_refs=refs, evaluator=evaluator)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    return {"contract_path": str(output), "contract_sha256": sha256_file(output),
            "evaluator_sha256": sha256_file(evaluator_path),
            "training_seal_sha256": seal_sha, "main_release_sha256": release_sha,
            "foundation_refs": refs, "test_payload_read": False, "cuda_initialized": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--main-release-artifact", type=Path, required=True)
    parser.add_argument("--foundation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
