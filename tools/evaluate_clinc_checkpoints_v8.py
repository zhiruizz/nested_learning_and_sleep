#!/usr/bin/env python3
"""Read-only rescoring of completed R1 CLINC checkpoints on training/dev only.

This utility repairs the R1 row-alignment issue by recomputing task-0 metrics
with the R2 evaluator. It never reads a test manifest, updates parameters, or
resumes training. The checkpoint files are treated as trusted local artifacts
only after their path and SHA-256 match a completed R1 receipt.
"""
from __future__ import annotations

import argparse
from dataclasses import is_dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback


SCRIPT_SCHEMA = "clinc-v8-checkpoint-rescore-v1"
MAX_FORWARD_CALLS = 5000
MAX_WALL_SECONDS = 1800


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with tmp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        if sha256_file(path) != hashlib.sha256(payload).hexdigest():
            raise IOError("JSON readback hash mismatch")
    finally:
        if tmp.exists():
            tmp.unlink()


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_completed_receipts(run_dir: Path, protocol: dict, arms: list[str]) -> dict[str, dict]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"R1 metrics file missing: {metrics_path}")
    known_arms = protocol["config"].get("arms")
    if not isinstance(known_arms, list) or not known_arms:
        raise ValueError("R1 protocol has no arm list")
    if any(arm not in known_arms for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError("requested arms must be unique members of the R1 protocol")

    receipts: dict[str, dict] = {}
    raw = metrics_path.read_bytes()
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith((b"\n", b"\r")):
                break  # Ignore only an incomplete trailing write from a live run.
            raise ValueError(f"malformed R1 metrics JSONL at line {index + 1}")
        if row.get("event") != "run_complete":
            continue
        receipt = row.get("receipt")
        arm = row.get("arm")
        if not isinstance(receipt, dict) or not isinstance(arm, str):
            raise ValueError("malformed run_complete record")
        if receipt.get("arm") != arm:
            raise ValueError("run_complete arm mismatch")
        if arm in receipts:
            raise ValueError(f"duplicate run_complete record for arm {arm}")
        receipts[arm] = receipt

    missing = [arm for arm in arms if arm not in receipts]
    if missing:
        raise RuntimeError(f"requested arms are not complete yet: {missing}")
    return {arm: receipts[arm] for arm in arms}


def verify_protocol(protocol: dict, run_dir: Path, project_root: Path,
                    model_path: Path, training_pack: Path, r1_script: Path) -> dict:
    if protocol.get("schema") != "clinc-cil-v8-pilot-protocol":
        raise ValueError("unexpected R1 protocol schema")
    if protocol.get("test_access") != "none":
        raise ValueError("R1 protocol does not certify test exclusion")
    cfg = protocol.get("config")
    if not isinstance(cfg, dict) or cfg.get("seed") != 17 or cfg.get("tasks") != [0, 1]:
        raise ValueError("R1 protocol config is not the expected seed-17 two-task pilot")
    if protocol.get("training_manifest_sha256") != cfg.get("training_manifest_sha256"):
        raise ValueError("R1 protocol/config training-manifest hashes differ")
    if sha256_file(training_pack) != protocol["training_manifest_sha256"]:
        raise ValueError("training pack SHA256 differs from R1 protocol")
    declared_model = protocol.get("model_path")
    if not isinstance(declared_model, str) or os.path.normcase(os.path.abspath(declared_model)) != os.path.normcase(str(model_path)):
        raise ValueError("--model-path does not match the R1 protocol model path")
    assets = protocol.get("model_assets_sha256")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("R1 protocol lacks model asset hashes")
    checked_assets = {}
    for relative, expected in assets.items():
        candidate = (model_path / relative).resolve()
        if model_path not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(f"locked model asset missing/outside model directory: {relative}")
        actual = sha256_file(candidate)
        if actual.lower() != str(expected).lower():
            raise ValueError(f"model asset hash mismatch: {relative}")
        checked_assets[str(relative)] = actual

    source_hashes = protocol.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("R1 protocol lacks source hashes")
    if canonical_sha(source_hashes) != protocol.get("source_lock_sha256"):
        raise ValueError("R1 source lock digest does not match source_hashes")
    verified_sources = {}
    for source_name, expected in source_hashes.items():
        source = Path(source_name).resolve()
        if not source.is_file() or sha256_file(source).lower() != str(expected).lower():
            raise ValueError(f"R1 source no longer matches its lock: {source_name}")
        verified_sources[str(source)] = str(expected).lower()

    protocol_model = protocol.get("model_path")
    if not Path(protocol_model).is_dir():
        raise FileNotFoundError(f"R1 model directory missing: {protocol_model}")
    return {"config": cfg, "model_assets_sha256": checked_assets,
            "r1_source_hashes_verified": verified_sources,
            "training_manifest_sha256": protocol["training_manifest_sha256"],
            "protocol_sha256": sha256_file(run_dir / "protocol.json"),
            "r1_driver_sha256": sha256_file(r1_script)}


def safe_checkpoint_path(run_dir: Path, relative: str) -> Path:
    path = (run_dir / relative).resolve()
    if run_dir not in path.parents or not path.is_file():
        raise ValueError("checkpoint path is missing or escapes the R1 run directory")
    return path


def _hash_text_list(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def run(args) -> dict:
    started = time.monotonic()
    output_dir = Path(args.output_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    training_pack = Path(args.training_pack).resolve()
    task_root = Path(__file__).resolve().parents[1]
    r1_script = task_root / "tools" / "run_clinc_cil_v8.py"
    r2_script = task_root / "tools" / "run_clinc_cil_v8_r2.py"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if output_dir == project_root or project_root in output_dir.parents:
        raise ValueError("output directory must be outside the public project")
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("output directory must be outside the R1 run directory")
    if output_dir == model_path or model_path in output_dir.parents:
        raise ValueError("output directory must be outside model assets")
    if not (run_dir / "protocol.json").is_file():
        raise FileNotFoundError("R1 protocol.json is required")
    if not r1_script.is_file() or not r2_script.is_file():
        raise FileNotFoundError("R1/R2 evaluator source files missing")
    output_dir.mkdir(parents=True, exist_ok=False)

    counters = None
    try:
        protocol_path = run_dir / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        pinned = verify_protocol(protocol, run_dir, project_root, model_path, training_pack, r1_script)
        arms = list(args.arms) if args.arms else list(protocol["config"]["arms"])
        receipts = read_completed_receipts(run_dir, protocol, arms)

        # Check each complete receipt and load only its hash-verified final state to CPU.
        import torch
        cpu_states = {}
        cp_metadata = {}
        for arm, receipt in receipts.items():
            if receipt.get("status") not in ("complete", "stopped_task0_acquisition_gate") or receipt.get("tasks_completed") != 1:
                raise ValueError(f"arm {arm} lacks an intact terminal task-0 training snapshot")
            checkpoint = safe_checkpoint_path(run_dir, receipt["final_checkpoint"])
            checkpoint_hash = sha256_file(checkpoint)
            if checkpoint_hash.lower() != str(receipt.get("final_checkpoint_sha256", "")).lower():
                raise ValueError(f"arm {arm} final checkpoint SHA256 mismatch")
            # This is an own-run checkpoint, hash-verified and loaded on CPU only.
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(state, dict) or state.get("schema") != "clinc-cil-v8-final-state":
                raise ValueError(f"arm {arm} final checkpoint schema mismatch")
            if state.get("arm") != arm or state.get("seed") != protocol["config"]["seed"]:
                raise ValueError(f"arm {arm} checkpoint identity mismatch")
            if not isinstance(state.get("context"), dict) or not is_dataclass(state.get("wake")):
                raise ValueError(f"arm {arm} checkpoint lacks context/wake state")
            cpu_states[arm] = state
            cp_metadata[arm] = {"path": str(checkpoint.relative_to(run_dir)), "sha256": checkpoint_hash,
                                "schema": state["schema"], "tasks_completed": receipt["tasks_completed"]}

        # Import the frozen R2 evaluator and helpers from this task workspace.
        task_tools = task_root / "tools"
        if str(task_tools) in sys.path:
            sys.path.remove(str(task_tools))
        sys.path.insert(0, str(task_tools))
        r2 = load_module(r2_script, "clinc_cil_v8_r2_rescore")
        r1_schema = load_module(r1_script, "clinc_cil_v8_r1_schema_only")
        checked_cfg = r1_schema.validate_config(dict(pinned["config"]))
        actual_config_sha = r2._canonical_sha({k: v for k, v in checked_cfg.items()
                                                if k != "derived_steps_per_epoch"})
        if actual_config_sha != protocol.get("config_sha256"):
            raise ValueError("R1 protocol config does not match its config SHA256")
        from clinc_cil_common_v8 import (make_training_view, iter_role, validate_role_partition,
                                         task_class_ids)
        import clinc_cil_common_v8 as common_module
        import clinc_tail_runtime_v8 as tail_runtime
        from clinc_tail_runtime_v8 import clone_mutable_context, causal_token_ce

        if Path(common_module.__file__).resolve() != task_tools / "clinc_cil_common_v8.py":
            raise RuntimeError("common helper import path mismatch")
        if Path(tail_runtime.__file__).resolve() != task_tools / "clinc_tail_runtime_v8.py":
            raise RuntimeError("tail runtime import path mismatch")
        if r2._sha256(r2_script) != sha256_file(r2_script):
            raise RuntimeError("R2 source hash calculation mismatch")

        manifest = json.loads(training_pack.read_text(encoding="utf-8"))
        if manifest.get("test_data_included") is not False:
            raise ValueError("training pack must explicitly exclude test data")
        view = make_training_view(manifest)
        train_rows = list(iter_role(view, "train", training_pack.parent))
        dev_rows = list(iter_role(view, "development", training_pack.parent))
        partition = validate_role_partition(train_rows, dev_rows)
        train_norm = {r2.normalize_text(row["text"]) for row in train_rows}
        duplicate_dev = [row for row in dev_rows if r2.normalize_text(row["text"]) in train_norm]
        clean_dev = [row for row in dev_rows if r2.normalize_text(row["text"]) not in train_norm]
        if len(duplicate_dev) != protocol["config"]["expected_normalized_train_dev_duplicates"]:
            raise ValueError("normalized train/dev duplicate count differs from R1 protocol")
        class_order = view.class_orders[str(protocol["config"]["seed"])]
        task0_ids = set(task_class_ids(class_order, 0))
        clean_task0 = sorted((r for r in clean_dev if int(r["intent_id"]) in task0_ids), key=lambda r: r["id"])
        task0_train = []
        for class_id in class_order[:15]:
            group = sorted((r for r in train_rows if int(r["intent_id"]) == class_id), key=lambda r: r["id"])
            if len(group) < 2:
                raise ValueError(f"task-0 class {class_id} has fewer than two train rows")
            task0_train.extend(group[:2])
        if len(clean_task0) == 0 or len(task0_train) != 30:
            raise ValueError("unexpected task-0 evaluation selection size")
        selection = {"clean_task0_development_rows": len(clean_task0),
                     "clean_development_duplicates_removed": len(duplicate_dev),
                     "task0_train_rows": len(task0_train), "train_rows": len(train_rows),
                     "development_rows": len(dev_rows), "partition": partition,
                     "task0_train_id_list_sha256": _hash_text_list([r["id"] for r in task0_train]),
                     "task0_clean_development_id_list_sha256": _hash_text_list([r["id"] for r in clean_task0]),
                     "test_data_read": False}

        cfg = dict(protocol["config"])
        cfg["max_forward_calls"] = MAX_FORWARD_CALLS
        cfg["max_wall_seconds"] = MAX_WALL_SECONDS
        cfg.setdefault("batch_size", 8)
        random.seed(cfg["initialization_seed"])
        import numpy as np
        np.random.seed(cfg["initialization_seed"])
        torch.manual_seed(cfg["initialization_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg["initialization_seed"])
        torch, tokenizer, model, immutable_versions_fn = r2._load_common(project_root, model_path, cfg["device"], cfg)
        experts_module = sys.modules.get("nested_sleep.experts")
        if experts_module is None or project_root / "src" not in Path(experts_module.__file__).resolve().parents:
            raise RuntimeError("nested_sleep.experts was not loaded from the requested public source tree")
        lm_module = sys.modules.get("nested_sleep.lm")
        if lm_module is None or project_root / "src" not in Path(lm_module.__file__).resolve().parents:
            raise RuntimeError("nested_sleep.lm was not loaded from the requested public source tree")
        frozen_before = immutable_versions_fn(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        budget = r2.Budget(cfg)
        budget.started = started
        counters = budget
        original_logits = model.logits
        original_tail = tail_runtime.tail_logits

        def counted_logits(tokens, mask=None):
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_logits(tokens, mask)

        def counted_tail(tail_model, context, cache):
            grad_on = torch.is_grad_enabled() and any(p.requires_grad for p in context.parameters())
            budget.tail(cache, grad_on)
            return original_tail(tail_model, context, cache)

        model.logits = counted_logits
        tail_runtime.tail_logits = counted_tail
        results = {}
        source_hashes = {"evaluator": sha256_file(Path(__file__).resolve()),
                         "r1_driver": sha256_file(r1_script),
                         "r2_evaluator": sha256_file(r2_script),
                         "common": sha256_file(task_tools / "clinc_cil_common_v8.py"),
                         "tail_runtime": sha256_file(task_tools / "clinc_tail_runtime_v8.py"),
                         "lifecycle": sha256_file(task_tools / "clinc_lifecycle_v8.py")}
        source_paths = {"evaluator": str(Path(__file__).resolve()),
                        "r1_driver": str(r1_script.resolve()), "r2_evaluator": str(r2_script.resolve()),
                        "common": str((task_tools / "clinc_cil_common_v8.py").resolve()),
                        "tail_runtime": str((task_tools / "clinc_tail_runtime_v8.py").resolve()),
                        "lifecycle": str((task_tools / "clinc_lifecycle_v8.py").resolve()),
                        "nested_sleep.lm": str(Path(lm_module.__file__).resolve()),
                        "nested_sleep.experts": str(Path(experts_module.__file__).resolve())}
        for arm, state in cpu_states.items():
            budget.check()
            context = clone_mutable_context(model)
            context.load_state_dict(state["context"], strict=True)
            for parameter in context.parameters():
                parameter.requires_grad_(False)
            r1_metrics_mark = "eval_invalid_rown_alignment; recomputed_with_r2"
            clean_metrics = r2._evaluate(model, context, clean_task0, class_order, 0,
                                         tokenizer, view, cfg, torch, cfg["device"], budget)
            train_greedy = r2._evaluate(model, context, task0_train, class_order, 0,
                                        tokenizer, view, cfg, torch, cfg["device"], budget)

            ce_sum = 0.0
            answer_tokens = 0
            token_correct = 0
            with torch.no_grad():
                for start in range(0, len(task0_train), cfg["batch_size"]):
                    rows = task0_train[start:start + cfg["batch_size"]]
                    _, ids, mask, labels = r2._batch(rows, tokenizer, view, cfg, torch, cfg["device"])
                    cache = tail_runtime.capture(model, ids, mask, device=cfg["device"])
                    logits = tail_runtime.tail_logits(model, context, cache)
                    answer_mask = labels.ne(-100)
                    loss, count = causal_token_ce(logits, ids, answer_mask)
                    ce_sum += float(loss.cpu()) * count
                    answer_tokens += count
                    valid = answer_mask[:, 1:]
                    pred = logits[:, :-1].argmax(-1)
                    gold = ids[:, 1:]
                    token_correct += int(((pred == gold) & valid).sum().item())

            results[arm] = {"seed": int(cfg["seed"]), "checkpoint": cp_metadata[arm],
                            "r1_metrics_status": r1_metrics_mark,
                            "task0_full_clean_development_seen15": clean_metrics,
                            "task0_fixed_train_two_per_class_greedy": train_greedy,
                            "task0_fixed_train_two_per_class_teacher_forced": {
                                "rows": len(task0_train), "answer_tokens": answer_tokens,
                                "mean_full_vocab_ce_nats_per_token": ce_sum / answer_tokens if answer_tokens else None,
                                "token_accuracy": token_correct / answer_tokens if answer_tokens else None}}
            del context

        unchanged = immutable_versions_fn(model) == frozen_before
        if not unchanged:
            raise RuntimeError("frozen backbone changed during rescoring")
        result = {"schema": SCRIPT_SCHEMA, "status": "complete", "training_only": True,
                  "model_revision": protocol.get("model_revision"), "source_revision": protocol.get("source_revision"),
                  "r1_protocol_sha256": pinned["protocol_sha256"],
                  "r1_source_lock_sha256": protocol["source_lock_sha256"],
                  "r1_source_hashes_verified": pinned["r1_source_hashes_verified"],
                  "r2_source_sha256": source_hashes["r2_evaluator"], "evaluation_source_hashes": source_hashes,
                  "evaluation_source_paths": source_paths,
                  "training_manifest_sha256": pinned["training_manifest_sha256"],
                  "model_assets_sha256": pinned["model_assets_sha256"], "selection": selection,
                  "arms": results, "work": budget.as_dict(),
                  "wall_seconds": time.monotonic() - started, "wall_limit_seconds": MAX_WALL_SECONDS,
                  "forward_limit": MAX_FORWARD_CALLS, "optimizer_steps": 0,
                  "immutable_backbone_unchanged": unchanged,
                  "interpretation": "R1 task-0 development metrics are invalidated for row alignment and replaced by this fixed clean-development rescore; no test set or future-class evaluation was accessed."}
        atomic_json(output_dir / "evaluation.json", result)
        return result
    except BaseException as exc:
        failure = {"schema": SCRIPT_SCHEMA, "status": "failed", "error_type": type(exc).__name__,
                   "error": str(exc), "elapsed_seconds": time.monotonic() - started,
                   "work": counters.as_dict() if counters is not None else None,
                   "optimizer_steps": 0, "test_data_read": False}
        atomic_json(output_dir / "failure.json", failure)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, help="public NL-Sleep project root")
    parser.add_argument("--model-path", required=True, help="local model directory matching R1 protocol")
    parser.add_argument("--run-dir", required=True, help="completed R1 run directory with protocol.json/metrics.jsonl")
    parser.add_argument("--training-pack", required=True, help="R1 training-only manifest JSON")
    parser.add_argument("--output-dir", required=True, help="new private evaluation output directory")
    parser.add_argument("--arms", nargs="*", default=None, help="optional completed R1 arms; defaults to protocol arms")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"status": result["status"], "arms": list(result["arms"]),
                      "output_dir": str(Path(args.output_dir).resolve()),
                      "full_backbone_forward_calls": result["work"]["full_backbone_forward_calls"]}, sort_keys=True))


if __name__ == "__main__":
    main()
