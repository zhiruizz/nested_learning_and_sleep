from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools" / "build_banking_memory_queue_v10.py"
SPEC = importlib.util.spec_from_file_location("banking_main_queue_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "banking_memory_policy_launcher_for_builder", ROOT / "tools" / "launch_banking_memory_policy_v10.py")
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
LAUNCHER_SPEC.loader.exec_module(launcher)


def _inputs(tmp_path: Path):
    project = tmp_path / "project"; project.mkdir()
    task_root = ROOT
    model = tmp_path / "model"; model.mkdir()
    pack = tmp_path / "pack.json"; pack.write_text("{}", encoding="utf-8")
    cfg_path = tmp_path / "cfg.json"; cfg_path.write_text("{}", encoding="utf-8")
    release = tmp_path / "release.json"
    foundations = {}
    for seed in (17, 29, 43):
        path = tmp_path / f"foundation-{seed}"; path.mkdir()
        foundations[str(seed)] = {"run_dir": str(path)}
    release.write_text(json.dumps({"foundations": foundations}), encoding="utf-8")
    queue_root = tmp_path / "private" / "main-run"; queue_root.parent.mkdir()
    contract_dir = tmp_path / "contracts"
    script = ROOT / "tools" / "run_banking_cil_v10.py"
    source_hashes = {"tools/run_banking_cil_v10.py": "a" * 64}
    config = {"status": "frozen", "protocol_stage": "probe_only",
              "arms": ["CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE"],
              "source_hashes": source_hashes, "device": "cuda:0"}
    return {
        "config": config, "config_path": cfg_path, "project_root": project, "task_root": task_root,
        "model_path": model, "training_pack": pack, "release_path": release,
        "release_sha256": builder.sha256_file(release), "queue_root": queue_root,
        "contract_dir": contract_dir, "launcher_path": ROOT / "tools" / "launch_banking_memory_policy_v10.py",
        "driver_path": script, "source_lock_sha256": "b" * 64,
        "config_semantic_sha256": "c" * 64, "source_hashes": source_hashes,
        "training_manifest_sha256": "d" * 64,
    }


def test_builds_exact_nine_private_serial_main_jobs_without_creating_outputs(tmp_path):
    args = _inputs(tmp_path)
    jobs, queue = builder.build_job_contracts(**args)
    assert len(jobs) == 9
    assert [(x["contract"]["argv"][x["contract"]["argv"].index("--seed") + 1],
             x["contract"]["argv"][x["contract"]["argv"].index("--arm") + 1]) for x in jobs] == [
        (str(seed), arm) for seed in (17, 29, 43) for arm in
        ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")]
    assert queue["schema"] == "banking-v10-memory-policy-queue-v1"
    assert len(queue["jobs"]) == 9
    # The frozen V10 driver uses output_dir.mkdir(parents=False). The queue
    # runner creates queue_root once, so all per-run directories must be its
    # immediate children for the first child to start successfully.
    args["queue_root"].mkdir()
    for item in jobs:
        contract = item["contract"]
        script_path, argv = launcher._parse_argv(contract, args["task_root"])
        assert script_path == args["driver_path"]
        out = Path(argv[argv.index("--output-dir") + 1])
        rec = Path(contract["launch_record_path"])
        assert out.parent == args["queue_root"]
        assert args["queue_root"] in rec.parents
        assert not out.exists() and not rec.exists()
        # Mirror the driver's non-recursive creation without launching it.
        out.mkdir(parents=False)
    assert len(list(args["queue_root"].iterdir())) == 9
    assert not args["contract_dir"].exists()


def test_refuses_changed_release_bytes_before_contract_creation(tmp_path):
    args = _inputs(tmp_path)
    args["release_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="release artifact is missing or changed"):
        builder.build_job_contracts(**args)
    assert not args["contract_dir"].exists()
    assert not args["queue_root"].exists()


def test_refuses_source_map_that_does_not_match_frozen_config(tmp_path):
    args = _inputs(tmp_path)
    args["source_hashes"] = {"tools/run_banking_cil_v10.py": "e" * 64}
    with pytest.raises(ValueError, match="source bytes differ"):
        builder.build_job_contracts(**args)
    assert not args["contract_dir"].exists()
