#!/usr/bin/env python3
"""Bounded diagnostic replay of the R1 sparse-Sleep guard failure.

This wrapper imports and runs the frozen R1 driver without changing its math.
It records the failing SleepTransaction inputs and saves a CPU-only mutable
transaction snapshot when the existing guard rejects the loss. It is a replay
diagnostic, not a repair or an alternate training method.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


EXPECTED_GUARD = "sleep requires finite differentiable scalar loss"
WAKE_TICK_CAP = 64
MAX_PHYSICAL_SLEEP_ADAM_STEPS = 40
MAX_FORWARD_CALLS = 3000


class WakeTickCap(RuntimeError):
    """Stops before any 65th wake update can be applied."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with tmp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        if sha256_file(path) != hashlib.sha256(data).hexdigest():
            raise IOError("diagnostic JSON readback SHA mismatch")
    finally:
        if tmp.exists():
            tmp.unlink()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _finite_loss_description(loss, torch):
    tensor = isinstance(loss, torch.Tensor)
    scalar = tensor and loss.ndim == 0
    finite = bool(torch.isfinite(loss).item()) if scalar else False
    value = None
    if scalar and finite:
        value = float(loss.detach().cpu().item())
    return {"is_tensor": tensor, "is_scalar": scalar, "is_finite": finite,
            "requires_grad": bool(getattr(loss, "requires_grad", False)),
            "value": value}


def run(args) -> dict:
    task_root = Path(__file__).resolve().parents[1]
    r1_path = task_root / "tools" / "run_clinc_cil_v8.py"
    lifecycle_path = task_root / "tools" / "clinc_lifecycle_v8.py"
    common_path = task_root / "tools" / "clinc_cil_common_v8.py"
    runtime_path = task_root / "tools" / "clinc_tail_runtime_v8.py"
    cfg_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    pack_path = Path(args.training_pack).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output_dir}")
    if not cfg_path.is_file() or not r1_path.is_file() or not lifecycle_path.is_file():
        raise FileNotFoundError("diagnostic config, frozen R1 runner, or lifecycle helper missing")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    expected_overrides = {"arms": ["HOPE_SLEEP"], "max_steps_per_task": 64,
                          "eval_full_development": False, "max_wall_seconds": 600,
                          "max_forward_calls": 3000, "max_update_steps": 200}
    for key, expected in expected_overrides.items():
        if cfg.get(key) != expected:
            raise ValueError(f"diagnostic config must pin {key}={expected!r}")
    if cfg.get("tasks") != [0, 1] or cfg.get("seed") != 17:
        raise ValueError("diagnostic config must retain R1 seed-17 task schedule")
    if cfg.get("periods") != [16, 64, 256, 1024, None]:
        raise ValueError("diagnostic config may not change R1 periods")
    if cfg.get("sleep_prompt_policy") != "seen_replay":
        raise ValueError("diagnostic config must retain the R1 seen-replay prompt policy")
    if output_dir == project_root or project_root in output_dir.parents:
        raise ValueError("diagnostic outputs must remain outside public project")
    if output_dir == model_path or model_path in output_dir.parents:
        raise ValueError("diagnostic outputs must remain outside model assets")
    if output_dir == pack_path.parent or pack_path.parent in output_dir.parents:
        raise ValueError("diagnostic output must remain outside training-data directory")

    # R1 itself creates output_dir only after all preflight checks.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    task_tools = task_root / "tools"
    if str(task_tools) in sys.path:
        sys.path.remove(str(task_tools))
    sys.path.insert(0, str(task_tools))
    r1 = load_module(r1_path, "clinc_v8_frozen_r1_sparse_diagnostic")
    lifecycle = load_module(lifecycle_path, "clinc_lifecycle_v8")
    import torch

    source_paths = [Path(__file__).resolve(), r1_path, lifecycle_path, common_path, runtime_path,
                    project_root / "src" / "nested_sleep" / "lm.py",
                    project_root / "src" / "nested_sleep" / "experts.py",
                    project_root / "src" / "nested_sleep" / "accounting.py",
                    project_root / "src" / "nested_sleep" / "native.py",
                    cfg_path]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"diagnostic provenance files missing: {missing}")
    source_pins = {str(path.resolve()): sha256_file(path) for path in source_paths}

    evidence = {"accepted_wake_ticks": 0, "wake_tick_attempts": 0,
                "physical_sleep_adamw_step_attempts": 0, "physical_sleep_adamw_steps": 0,
                "last_route_audit": None,
                "route_audits": [], "failed_optimize": None,
                "snapshot_path": None, "snapshot_sha256": None,
                "current_arm": None, "error": None}
    budgets = []
    output_created = False
    run_start = time.monotonic()

    # Count actual optimizer.step calls without modifying their inputs or math.
    original_adam_step = torch.optim.AdamW.step
    def counted_adam_step(optimizer, *step_args, **step_kwargs):
        if evidence["physical_sleep_adamw_step_attempts"] >= MAX_PHYSICAL_SLEEP_ADAM_STEPS:
            raise RuntimeError("diagnostic physical Sleep AdamW step cap exceeded")
        evidence["physical_sleep_adamw_step_attempts"] += 1
        result = original_adam_step(optimizer, *step_args, **step_kwargs)
        evidence["physical_sleep_adamw_steps"] += 1
        return result
    torch.optim.AdamW.step = counted_adam_step

    original_budget = r1.Budget
    class TrackingBudget(original_budget):
        def __init__(self, config):
            super().__init__(config)
            budgets.append(self)
    r1.Budget = TrackingBudget

    original_record_tick = lifecycle.WakeController.record_tick
    def capped_record_tick(controller, gradients, *, token_count=1):
        evidence["wake_tick_attempts"] += 1
        if int(controller.ticks) >= WAKE_TICK_CAP:
            raise WakeTickCap("diagnostic stopped before the 65th wake update")
        result = original_record_tick(controller, gradients, token_count=token_count)
        evidence["accepted_wake_ticks"] += 1
        return result
    lifecycle.WakeController.record_tick = capped_record_tick

    original_optimize = lifecycle.SleepTransaction.optimize
    def audited_optimize(tx, loss, kind, tokens):
        description = _finite_loss_description(loss, torch)
        try:
            return original_optimize(tx, loss, kind, tokens)
        except BaseException as exc:
            if str(exc) == "sleep requires finite differentiable scalar loss":
                snap_path = output_dir / "sleep-transaction-snapshot.pt"
                snapshot_info = None
                try:
                    # This snapshot is owned CPU mutable/optimizer/RNG state only;
                    # it contains no backbone, prompt, or token sequence tensors.
                    snapshot = tx.snapshot()
                    torch.save(snapshot, snap_path)
                    snapshot_info = {"path": str(snap_path), "sha256": sha256_file(snap_path),
                                     "bytes": snap_path.stat().st_size}
                    evidence["snapshot_path"] = snapshot_info["path"]
                    evidence["snapshot_sha256"] = snapshot_info["sha256"]
                except BaseException as snapshot_exc:
                    snapshot_info = {"error_type": type(snapshot_exc).__name__,
                                     "error": str(snapshot_exc)}
                evidence["failed_optimize"] = {
                    "wake_ticks": evidence["accepted_wake_ticks"],
                    "sender": int(tx.sender), "receiver": int(tx.receiver),
                    "receiver_slot": int(tx.slot), "kind": str(kind),
                    "tokens_argument": int(tokens) if type(tokens) is int else None,
                    "loss": description, "allowed_parameter_names": sorted(tx.allowed_names),
                    "transaction_counters_before_guard": dict(tx.counters),
                    "snapshot": snapshot_info, "guard_error": str(exc)}
            raise
    lifecycle.SleepTransaction.optimize = audited_optimize

    original_audit = r1._audited_student_logits
    def audited_student_logits(*call_args, **call_kwargs):
        result = original_audit(*call_args, **call_kwargs)
        route_info = result[1] if isinstance(result, tuple) and len(result) == 2 else None
        if isinstance(route_info, dict):
            summary = {key: route_info.get(key) for key in
                       ("receiver_selected_answer_tokens", "answer_tokens",
                        "receiver_answer_coverage", "route_score_abs_max")}
            evidence["last_route_audit"] = summary
            evidence["route_audits"].append(summary)
        return result
    r1._audited_student_logits = audited_student_logits

    try:
        run_args = argparse.Namespace(project_root=str(project_root), model_path=str(model_path),
                                      training_pack=str(pack_path), output_dir=str(output_dir),
                                      config=str(cfg_path))
        try:
            r1.run(run_args)
            result_status = "guard_not_reproduced"
        except BaseException as exc:
            evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
            result_status = "reproduced_guard_failure" if str(exc) == EXPECTED_GUARD and evidence["failed_optimize"] else "other_failure"
        output_created = output_dir.is_dir()
        failure_path = output_dir / "failure.json"
        failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.is_file() else None
        if failure is None:
            failure_status = "no_runner_failure_artifact"
        else:
            failure_status = failure.get("status")
        work = failure.get("work") if isinstance(failure, dict) else None
        if evidence["physical_sleep_adamw_step_attempts"] > MAX_PHYSICAL_SLEEP_ADAM_STEPS:
            raise RuntimeError("physical Sleep AdamW calls exceeded diagnostic cap")
        diagnostic = {
            "schema": "clinc-v8-sparse-sleep-failure-diagnostic-v1",
            "status": result_status,
            "runner_failure_status": failure_status,
            "expected_failure": EXPECTED_GUARD,
            "config": {"path": str(cfg_path), "sha256": source_pins[str(cfg_path.resolve())],
                       "values": expected_overrides,
                       "config_sha256": r1._canonical_sha({k: v for k, v in r1.validate_config(cfg).items()
                                                            if k != "derived_steps_per_epoch"})},
            "source_pins": source_pins,
            "r1_protocol_sha256": sha256_file(output_dir / "protocol.json") if (output_dir / "protocol.json").is_file() else None,
            "r1_failure_sha256": sha256_file(failure_path) if failure_path.is_file() else None,
            "r1_failure_work": work,
            "routing_evidence": {"accepted_wake_ticks": evidence["accepted_wake_ticks"],
                "wake_tick_attempts_including_cap": evidence["wake_tick_attempts"],
                "physical_sleep_adamw_step_attempts": evidence["physical_sleep_adamw_step_attempts"],
                "physical_sleep_adamw_steps": evidence["physical_sleep_adamw_steps"],
                "physical_adam_steps_before": 0,
                "physical_adam_steps_after": evidence["physical_sleep_adamw_steps"],
                "last_answer_route_coverage": evidence["last_route_audit"],
                "route_audit_call_count": len(evidence["route_audits"]),
                "failed_optimize": evidence["failed_optimize"]},
            "snapshot": {"path": evidence["snapshot_path"], "sha256": evidence["snapshot_sha256"]},
            "elapsed_seconds": time.monotonic() - run_start,
            "budget_limits": {"wake_updates": WAKE_TICK_CAP,
                              "physical_sleep_adamw_steps": MAX_PHYSICAL_SLEEP_ADAM_STEPS,
                              "full_backbone_forwards": MAX_FORWARD_CALLS,
                              "wall_seconds": cfg["max_wall_seconds"]},
            "test_data_accessed": False,
            "mathematics_changed": False}
        atomic_json(output_dir / "routing-diagnostic.json", diagnostic)
        return diagnostic
    finally:
        torch.optim.AdamW.step = original_adam_step
        r1.Budget = original_budget
        lifecycle.WakeController.record_tick = original_record_tick
        lifecycle.SleepTransaction.optimize = original_optimize
        r1._audited_student_logits = original_audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-pack", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"status": result["status"], "accepted_wake_ticks": result["routing_evidence"]["accepted_wake_ticks"],
                      "physical_sleep_adamw_steps": result["routing_evidence"]["physical_sleep_adamw_steps"],
                      "output_dir": str(Path(args.output_dir).resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
