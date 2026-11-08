"""Launch one reviewed V12 seed after its real CPU preflight.

The CUDA policy is process-local and matches the previously validated memory
policy. No test-data path, automatic retry, output overwrite or system change.
"""
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, value):
    # State changes replace this launcher's own fresh record, not an old run.
    from run_banking_cil_v10 import _atomic_json
    _atomic_json(path, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "v10-config", "project-root", "model-path", "training-pack",
                 "foundation-root", "output-dir", "launch-record"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    args = parser.parse_args(argv)
    tools_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(tools_root))
    for name in ("config", "v10_config", "project_root", "model_path", "training_pack", "foundation_root"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    args.launch_record = args.launch_record.resolve()
    if (args.output_dir == args.launch_record or args.output_dir.is_relative_to(args.launch_record)
            or args.launch_record.is_relative_to(args.output_dir)):
        raise ValueError("scientific output and launch record must be disjoint")
    if args.output_dir.exists() or args.launch_record.exists():
        raise FileExistsError("fresh scientific output and launch record required")
    if not args.output_dir.parent.is_dir() or not args.launch_record.parent.is_dir():
        raise ValueError("explicit output and record parents must already exist")
    for protected in (args.project_root, args.model_path, args.training_pack.parent, args.foundation_root):
        for output in (args.output_dir, args.launch_record):
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise ValueError("output/record overlaps a protected source or asset root")
    policy = "garbage_collection_threshold:0.7"
    started = time.monotonic()
    state = {"schema": "banking-v12-launch-record-v1", "status": "preflight_started",
             "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "seed": args.seed, "config_sha256": args.expected_config_sha256,
             "observed_config_sha256": sha256(args.config),
             "launcher_sha256": sha256(__file__), "output_dir": str(args.output_dir),
             "automatic_retries": 0, "test_payload_accessed": False,
             "process_local_memory_fraction": 0.8, "process_local_allocator_conf": policy,
             "runner_callback_attempted": False, "gpu_policy_applied": False}
    record(args.launch_record, state)
    try:
        if sha256(args.config) != args.expected_config_sha256:
            raise ValueError("frozen V12 config SHA-256 differs")
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must be uninitialized before CPU preflight")
        state["cpu_preflight_cuda_uninitialized_before"] = True
        runner = importlib.import_module("run_banking_optimizer_boundary_v12")
        if Path(runner.__file__).resolve() != tools_root / "run_banking_optimizer_boundary_v12.py":
            raise RuntimeError("V12 runner import escaped the pinned tools directory")
        state["runner_sha256"] = sha256(runner.__file__)
        if torch.cuda.is_initialized():
            raise RuntimeError("V12 runner import initialized CUDA before CPU preflight")
        # The real preflight verifies source/config/model/pack/foundation bindings.
        state["preflight"] = runner._preflight(args)
        if torch.cuda.is_initialized():
            raise RuntimeError("V12 CPU preflight initialized CUDA")
        state["cpu_preflight_cuda_uninitialized_after"] = True
        if sha256(args.config) != args.expected_config_sha256:
            raise RuntimeError("V12 config changed during preflight")
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        if (cfg["cuda_allocator_conf"], cfg["cuda_memory_fraction"]) != (policy, 0.8):
            raise ValueError("unreviewed memory policy")
        if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") not in (None, "", policy):
            raise RuntimeError("conflicting process-local CUDA allocator policy")
        state.update(status="cpu_preflight_passed")
        record(args.launch_record, state)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = policy
        if torch.__version__ != "2.8.0+cu128" or torch.cuda.is_initialized():
            raise RuntimeError("expected reviewed PyTorch build and uninitialized CUDA process")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_per_process_memory_fraction(0.8, device=0)
        state.update(status="running", gpu_policy_applied=True, runner_callback_attempted=True)
        record(args.launch_record, state)
        result = runner._run(args, state["preflight"])
        if result.get("status") != "complete":
            raise RuntimeError("V12 callback did not return a complete receipt")
        state.update(status="complete", result=result)
    except BaseException as exc:
        state.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        state["launcher_wall_seconds_including_preflight"] = time.monotonic() - started
        state["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record(args.launch_record, state)
    print(json.dumps({"status": state["status"], "seed": args.seed,
                      "launch_record": str(args.launch_record), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
