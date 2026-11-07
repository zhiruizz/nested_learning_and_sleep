"""Immediate serial execution of a frozen CLINC grid; never retries or resumes."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master-config", required=True, type=Path)
    p.add_argument("--training-pack", required=True, type=Path)
    p.add_argument("--project-root", required=True, type=Path)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--grid-root", required=True, type=Path)
    p.add_argument("--freeze-receipt", required=True, type=Path)
    a = p.parse_args()
    runner = Path(__file__).with_name("run_clinc_cil_v8_full.py").resolve()
    spec = importlib.util.spec_from_file_location("full_queue_protocol", runner)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    master = m.validate_master_config(json.loads(a.master_config.read_text()))
    freeze = json.loads(a.freeze_receipt.read_text())
    grid = a.grid_root.resolve()
    foundations = grid / "foundations"
    runs = grid / "runs"
    log_path = grid / "queue.jsonl"
    if log_path.exists():
        raise FileExistsError("refusing to restart queue; inspect any failed child explicitly")
    grid.mkdir(parents=True, exist_ok=True)

    def source_guard():
        if m.sha256_file(a.master_config) != freeze["master_config_sha256"]:
            raise ValueError("frozen master changed")
        actual = m._source_hashes(a.project_root.resolve(), runner.parent)
        if actual != freeze["source_hashes"] or m.canonical_sha(actual) != freeze["source_lock_sha256"]:
            raise ValueError("reviewed training source changed")
        if m.sha256_file(a.training_pack) != master["training_manifest_sha256"]:
            raise ValueError("training pack changed")

    def record(value):
        value.update(utc=datetime.now(timezone.utc).isoformat(), test_data_accessed=False)
        with log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(value, sort_keys=True) + "\n")
            f.flush()
        print(json.dumps(value, sort_keys=True), flush=True)

    base = [sys.executable, str(runner), "--master-config", str(a.master_config.resolve()),
            "--training-pack", str(a.training_pack.resolve()), "--project-root", str(a.project_root.resolve())]

    def child(mode, seed=None, arm=None):
        source_guard()
        command = base + ["--mode", mode]
        if mode == "seal":
            command += ["--foundation-dir", str(foundations), "--run-root", str(runs),
                        "--output-path", str(grid / "main-six-seal.json")]
            label = "seal"
        else:
            out = foundations / f"seed{seed}" if mode == "foundation" else runs / f"seed{seed}" / arm
            if out.exists():
                raise FileExistsError(f"refusing any existing child output: {out}")
            command += ["--model-path", str(a.model_path.resolve()), "--seed", str(seed), "--output-dir", str(out)]
            if mode == "continuation":
                command += ["--arm", arm, "--foundation-dir", str(foundations)]
            label = f"{mode}-seed{seed}" + ("-" + arm if arm else "")
        record({"event": "child_start", "mode": mode, "seed": seed, "arm": arm})
        with (grid / (label + ".stdout.log")).open("xb") as stdout, (grid / (label + ".stderr.log")).open("xb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        record({"event": "child_end", "mode": mode, "seed": seed, "arm": arm, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{label} failed; no automatic retry, inspect its failure/logs")

    try:
        source_guard()
        # Seed 17 was deliberately launched and inspected by root before this
        # queue. Require its actual qualified terminal receipt and bytes.
        receipt_path = foundations / "seed17" / "foundation-receipt.json"
        first = json.loads(receipt_path.read_text())
        if first.get("status") != "complete" or first.get("acquisition_qualified") is not True:
            raise ValueError("seed17 has no qualified terminal foundation")
        cp = m._safe_child_path(receipt_path.parent, first["final_checkpoint"])
        if m.sha256_file(cp) != first["final_checkpoint_sha256"]:
            raise ValueError("seed17 foundation checkpoint changed")
        record({"event": "verified_completed_seed17_foundation", "checkpoint_sha256": first["final_checkpoint_sha256"]})
        for seed in (29, 43):
            child("foundation", seed)
        source_guard()
        m.load_and_verify_foundation_grid(foundations,
            master_sha=freeze["master_config_sha256"], manifest_sha=master["training_manifest_sha256"],
            source_lock_sha=freeze["source_lock_sha256"], model_revision=master["model_revision"],
            model_assets_sha256=master["model_assets_sha256"])
        record({"event": "three_foundations_verified"})
        for seed in m.SEEDS:
            for arm in m.ARMS:
                child("continuation", seed, arm)
        child("seal")
        record({"event": "queue_complete", "main_runs": 6, "test_release_ready": True})
    except BaseException as exc:
        record({"event": "queue_failed", "error_type": type(exc).__name__, "error": str(exc), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
