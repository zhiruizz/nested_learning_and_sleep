#!/usr/bin/env python3
"""Build, but never launch, nine source-pinned Banking V10 main contracts.

The builder requires an already verified main-release artifact and binds the
nine serial jobs to its exact path and bytes. It creates only a fresh contract
bundle in the C task workspace; the finite queue's run directory remains a
fresh private D-path chosen by the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


SEEDS = (17, 29, 43)
ARMS = ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")
QUEUE_SCHEMA = "banking-v10-memory-policy-queue-v1"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_job_contracts(*, config: dict, config_path: Path, project_root: Path,
                         task_root: Path, model_path: Path, training_pack: Path,
                         release_path: Path, release_sha256: str, queue_root: Path,
                         contract_dir: Path, launcher_path: Path, driver_path: Path,
                         source_lock_sha256: str, config_semantic_sha256: str,
                         source_hashes: dict, training_manifest_sha256: str) -> tuple[list[dict], dict]:
    """Return deterministic main-job contracts and queue JSON without writing."""
    if config.get("status") != "frozen" or config.get("protocol_stage") is None:
        raise ValueError("main contracts require a frozen V10 config")
    queue_root = queue_root.expanduser().resolve()
    contract_dir = contract_dir.expanduser().resolve()
    if queue_root.exists():
        raise FileExistsError("queue root must be fresh")
    if contract_dir.exists():
        raise FileExistsError("contract bundle directory must be fresh")
    if not queue_root.parent.is_dir():
        raise FileNotFoundError("queue root parent must already exist")
    if not release_path.is_file() or sha256_file(release_path) != release_sha256:
        raise ValueError("main release artifact is missing or changed")
    if set(config.get("source_hashes", {})) != set(source_hashes):
        raise ValueError("validated source hash map does not match frozen config")
    if source_hashes != config["source_hashes"]:
        raise ValueError("validated source bytes differ from frozen config")

    launcher_sha = sha256_file(launcher_path)
    driver_sha = sha256_file(driver_path)
    jobs: list[dict] = []
    queue_jobs: list[dict] = []
    output_paths: set[Path] = set()
    record_paths: set[Path] = set()
    release_obj = json.loads(release_path.read_text(encoding="utf-8"))
    for seed in SEEDS:
        # Use the exact released path; layout is an input, not inferred from the
        # release artifact's directory.
        foundation = release_obj.get("foundations", {}).get(str(seed), {})
        released_dir = Path(foundation.get("run_dir", "")).resolve()
        if not foundation.get("run_dir") or not released_dir.is_dir():
            raise ValueError(f"release lacks the actual foundation directory for seed {seed}")
        for arm in ARMS:
            stem = f"seed{seed}-{arm.lower()}"
            # The frozen trainer creates output_dir with parents=False, so its
            # immediate parent must already be the queue root created by the
            # serial queue runner. Keep each run a direct child; nesting under
            # runs/seedN would make the first child fail before model loading.
            out_dir = (queue_root / f"seed{seed}-{arm.lower()}").resolve()
            record_path = (queue_root / "launcher-records" / f"{stem}.json").resolve()
            if out_dir.parent != queue_root or queue_root not in record_path.parents:
                raise ValueError("all run outputs and launcher records must be under the queue root")
            if out_dir in output_paths or record_path in record_paths:
                raise ValueError("duplicate queue output path")
            output_paths.add(out_dir); record_paths.add(record_path)
            argv = ["--phase", "main", "--project-root", str(project_root.resolve()),
                    "--model-path", str(model_path.resolve()), "--training-pack", str(training_pack.resolve()),
                    "--config", str(config_path.resolve()), "--output-dir", str(out_dir),
                    "--seed", str(seed), "--arm", arm, "--foundation-dir", str(released_dir),
                    "--release-artifact", str(release_path.resolve())]
            contract = {
                "schema": "banking-v10-memory-policy-launch-v1",
                "launcher_sha256": launcher_sha,
                "project_root": str(project_root.resolve()),
                "task_root": str(task_root.resolve()),
                "model_path": str(model_path.resolve()),
                "training_pack": str(training_pack.resolve()),
                "config_path": str(config_path.resolve()),
                "config_file_sha256": sha256_file(config_path),
                "config_semantic_sha256": config_semantic_sha256,
                "training_manifest_sha256": training_manifest_sha256,
                "source_lock_sha256": source_lock_sha256,
                "source_hashes": source_hashes,
                "script_key": driver_path.name,
                "script_path": str(driver_path.resolve()),
                "script_sha256": driver_sha,
                "argv": argv,
                "launch_record_path": str(record_path),
                "allocator_conf": "garbage_collection_threshold:0.7",
                "memory_fraction": 0.8,
                "device": config["device"],
                "torch_version": "2.8.0+cu128",
            }
            # Validate the driver's explicit main arguments and contract shape
            # without touching model/data payloads or invoking the launcher.
            if arm not in config["arms"] or int(argv[argv.index("--seed") + 1]) != seed:
                raise ValueError("job arm/seed is not in the frozen config")
            jobs.append({"name": stem, "contract": contract})
            contract_path = (contract_dir / f"{stem}.json").resolve()
            queue_jobs.append({"contract_path": str(contract_path), "contract_sha256": None})
    queue = {
        "schema": QUEUE_SCHEMA,
        "launcher_sha256": launcher_sha,
        "queue_root": str(queue_root),
        "jobs": queue_jobs,
        "minimum_free_before_start_bytes": None,
    }
    return jobs, queue


def build_bundle(args) -> dict:
    release_path = args.main_release.expanduser().resolve(strict=True)
    if sha256_file(release_path) != args.main_release_sha256:
        raise ValueError("main release SHA-256 mismatch")
    config_path = args.config.expanduser().resolve(strict=True)
    project_root = args.project_root.expanduser().resolve(strict=True)
    task_root = args.task_root.expanduser().resolve(strict=True)
    model_path = args.model_path.expanduser().resolve(strict=True)
    training_pack = args.training_pack.expanduser().resolve(strict=True)
    launcher_path = args.launcher.expanduser().resolve(strict=True)
    driver_path = (task_root / "tools" / "run_banking_cil_v10.py").resolve(strict=True)
    contract_dir = args.contract_dir.expanduser().resolve()
    queue_root = args.queue_root.expanduser().resolve()
    if args.minimum_free_before_start_bytes < 0:
        raise ValueError("minimum free bytes must be nonnegative")
    if queue_root.exists() or contract_dir.exists():
        raise FileExistsError("queue root and contract bundle must both be fresh")

    if str(task_root / "tools") not in sys.path:
        sys.path.insert(0, str(task_root / "tools"))
    import run_banking_cil_v10 as driver
    import launch_banking_memory_policy_v10 as launcher
    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = driver.validate_config(raw_cfg, phase="main")
    source_hashes, source_lock_sha, manifest, train, dev = driver._source_prepare(
        project_root, task_root, model_path, training_pack, cfg)
    if launcher_path != Path(launcher.__file__).resolve():
        raise ValueError("requested launcher path is not the reviewed helper loaded in this process")
    if launcher.sha256_file(launcher_path) != launcher.sha256_file(Path(launcher.__file__)):
        raise ValueError("launcher bytes changed during contract construction")
    if len(train) != 9229 or len(dev) != 770:
        raise ValueError("unexpected locked Banking77 train/dev partition")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (release.get("schema"), release.get("status")) != ("banking-cil-v10-main-release-v1", "ready"):
        raise ValueError("main release artifact is not ready")
    if (release.get("config_sha256"), release.get("source_lock_sha256"),
        release.get("training_manifest_sha256")) != (
            driver._config_sha(cfg), source_lock_sha, cfg["training_manifest_sha256"]):
        raise ValueError("main release config/source/data bindings differ from current frozen inputs")
    for seed in SEEDS:
        foundation_dir = Path(release.get("foundations", {}).get(str(seed), {}).get("run_dir", "")).resolve()
        driver._validate_foundation_release(release_path, cfg, driver._config_sha(cfg), source_lock_sha,
                                            cfg["training_manifest_sha256"], seed, foundation_dir)

    jobs, queue = build_job_contracts(
        config=cfg, config_path=config_path, project_root=project_root, task_root=task_root,
        model_path=model_path, training_pack=training_pack, release_path=release_path,
        release_sha256=args.main_release_sha256, queue_root=queue_root, contract_dir=contract_dir,
        launcher_path=launcher_path, driver_path=driver_path, source_lock_sha256=source_lock_sha,
        config_semantic_sha256=driver._config_sha(cfg), source_hashes=source_hashes,
        training_manifest_sha256=cfg["training_manifest_sha256"])
    queue["minimum_free_before_start_bytes"] = args.minimum_free_before_start_bytes

    # Stage all bundle files in a fresh sibling, then publish with one rename.
    staging = contract_dir.with_name(contract_dir.name + ".staging")
    if staging.exists():
        raise FileExistsError("staging bundle already exists; inspect before continuing")
    staging.mkdir(parents=True, exist_ok=False)
    contract_hashes = []
    for item in jobs:
        path = staging / f"{item['name']}.json"
        raw = json.dumps(item["contract"], ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(raw)
        digest = sha256_file(path)
        contract_hashes.append(digest)
    for index, item in enumerate(queue["jobs"]):
        item["contract_path"] = str((contract_dir / Path(item["contract_path"]).name).resolve())
        item["contract_sha256"] = contract_hashes[index]
    queue_path = staging / "main-queue.json"
    queue_raw = json.dumps(queue, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with queue_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(queue_raw)
    bundle = {
        "schema": "banking-v10-main-memory-queue-bundle-v1",
        "status": "contracts_built_not_launched",
        "main_release_path": str(release_path),
        "main_release_sha256": args.main_release_sha256,
        "config_path": str(config_path),
        "config_file_sha256": sha256_file(config_path),
        "config_semantic_sha256": driver._config_sha(cfg),
        "source_lock_sha256": source_lock_sha,
        "training_manifest_sha256": cfg["training_manifest_sha256"],
        "model_revision": cfg["model_revision"],
        "launcher_sha256": sha256_file(launcher_path),
        "driver_sha256": sha256_file(driver_path),
        "contract_files": [f"{item['name']}.json" for item in jobs],
        "contract_sha256s": contract_hashes,
        "queue_file": "main-queue.json",
        "queue_sha256": sha256_file(queue_path),
        "job_count": 9,
        "jobs": [{"seed": seed, "arm": arm} for seed in SEEDS for arm in ARMS],
        "test_data_included": False,
        "launch_performed": False,
    }
    bundle_path = staging / "bundle.json"
    with bundle_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    staging.rename(contract_dir)
    return {"bundle_path": str(contract_dir / "bundle.json"), "bundle_sha256": sha256_file(contract_dir / "bundle.json"),
            "queue_path": str(contract_dir / "main-queue.json"), "queue_sha256": sha256_file(contract_dir / "main-queue.json"),
            "contract_dir": str(contract_dir), "jobs": 9, "launch_performed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-release", type=Path, required=True)
    parser.add_argument("--main-release-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-pack", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--minimum-free-before-start-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    result = build_bundle(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
