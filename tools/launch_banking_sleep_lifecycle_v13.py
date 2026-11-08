"""Launch one reviewed V13 seed/arm after strict CPU-only preflight."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


ALLOCATOR_CONF = "garbage_collection_threshold:0.7"
MEMORY_FRACTION = 0.8
SEEDS = (17, 29, 43)
ARMS = ("CMS_ONLY", "B1_HIGH", "B1_LOW", "B32_HIGH", "B32_LOW", "B32_LOW_DOSE")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_paths(output_dir, launch_record, protected_roots):
    out, record = Path(output_dir).resolve(), Path(launch_record).resolve()
    if out == record or out in record.parents or record in out.parents:
        raise ValueError("scientific output and launch record must be disjoint")
    if out.exists() or record.exists():
        raise FileExistsError("fresh output and launch record required")
    if not out.parent.is_dir() or not record.parent.is_dir():
        raise ValueError("explicit output and launch-record parent directories must exist")
    for protected in map(lambda p: Path(p).resolve(), protected_roots):
        for candidate in (out, record):
            if candidate == protected or candidate in protected.parents or protected in candidate.parents:
                raise ValueError("output or launch record overlaps a protected source/asset tree")
    return out, record


def _record(path, value):
    import run_banking_cil_v10 as v10
    v10._atomic_json(Path(path), value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "v10-config", "project-root", "model-path", "training-pack",
                 "foundation-root", "release-artifact", "output-dir", "launch-record"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args(argv)
    tools_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(tools_root))
    for name in ("config", "v10_config", "project_root", "model_path", "training_pack",
                 "foundation_root", "release_artifact"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    output, launch_record = validate_output_paths(args.output_dir, args.launch_record,
        (args.project_root, args.model_path, args.training_pack.parent, args.foundation_root))
    args.output_dir, args.launch_record = output, launch_record
    if not args.expected_config_sha256 or len(args.expected_config_sha256) != 64:
        raise ValueError("expected config SHA-256 must be supplied")
    started = time.monotonic()
    state = {"schema": "banking-v13-launch-record-v1", "status": "cpu_preflight_started",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": args.seed, "arm": args.arm, "expected_config_sha256": args.expected_config_sha256,
        "observed_config_sha256": sha256(args.config), "launcher_sha256": sha256(__file__),
        "output_dir": str(args.output_dir), "automatic_retries": 0, "test_data_read": False,
        "process_local_memory_fraction": MEMORY_FRACTION,
        "process_local_allocator_conf": ALLOCATOR_CONF,
        "runner_callback_attempted": False, "gpu_policy_applied": False}
    _record(args.launch_record, state)
    try:
        if sha256(args.config) != args.expected_config_sha256:
            raise ValueError("V13 frozen config SHA-256 mismatch")
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must be uninitialized before CPU preflight")
        state["cuda_uninitialized_before_preflight"] = True
        runner = importlib.import_module("run_banking_sleep_lifecycle_v13")
        if Path(runner.__file__).resolve() != tools_root / "run_banking_sleep_lifecycle_v13.py":
            raise RuntimeError("V13 runner import escaped pinned tools directory")
        state["runner_sha256"] = sha256(runner.__file__)
        if torch.cuda.is_initialized():
            raise RuntimeError("runner import initialized CUDA")
        state["preflight"] = runner._preflight(args)
        if torch.cuda.is_initialized():
            raise RuntimeError("V13 CPU preflight initialized CUDA")
        state["cuda_uninitialized_after_preflight"] = True
        if sha256(args.config) != args.expected_config_sha256:
            raise RuntimeError("V13 config changed during CPU preflight")
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        if (cfg.get("cuda_allocator_conf"), cfg.get("cuda_memory_fraction")) != (ALLOCATOR_CONF, MEMORY_FRACTION):
            raise ValueError("V13 config does not pin the reviewed process-local CUDA policy")
        configured = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        if configured not in (None, "", ALLOCATOR_CONF):
            raise RuntimeError("conflicting CUDA allocator policy already exists in process")
        state["status"] = "cpu_preflight_passed"
        _record(args.launch_record, state)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR_CONF
        if torch.__version__ != "2.8.0+cu128" or torch.cuda.is_initialized():
            raise RuntimeError("expected reviewed PyTorch 2.8.0+cu128 with CUDA still uninitialized")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device=0)
        state.update(status="running", gpu_policy_applied=True, runner_callback_attempted=True)
        _record(args.launch_record, state)
        result = runner._run(args, state["preflight"])
        if not isinstance(result, dict) or result.get("status") != "complete":
            raise RuntimeError("V13 runner did not return a complete result")
        state.update(status="complete", result=result)
    except BaseException as exc:
        state.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        state["launcher_wall_seconds_including_preflight"] = time.monotonic() - started
        state["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _record(args.launch_record, state)
    print(json.dumps({"status": state["status"], "seed": args.seed, "arm": args.arm,
                      "launch_record": str(args.launch_record), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
