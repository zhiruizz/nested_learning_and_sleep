#!/usr/bin/env python3
"""Standalone recovery proof: original transaction continuation vs disk restore.

This is a bounded, development-only follow-up to the archived V10 transfer
probe. It runs one seed-17 SLEEP_ADDITIVE transaction to opportunity 3, saves
and reloads its exact snapshot, continues the *original transaction object*
through calls 4-6, then restores the disk snapshot and replays calls 4-6. It
does not read development/test labels, produce a readout, or alter V10 sources.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for _p in (ROOT, TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools import probe_banking_transfer_v10 as probe

SCHEMA = "banking-cil-v10-original-transaction-replay-proof-v1"
CAPS = {"max_wall_seconds": 1800, "max_forward_calls": 20000,
        "max_update_steps": 1000}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.partial")
    try:
        with tmp.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        if path.read_bytes() != raw:
            raise IOError("atomic recovery-proof JSON readback mismatch")
        return hashlib.sha256(raw).hexdigest()
    finally:
        if tmp.exists():
            tmp.unlink()


def original_then_restore(original_tx, snapshot, *, continue_original,
                          restore_from_disk, continue_restored, exact_equal):
    """Run both branches from a snapshot, preserving identity of original_tx."""
    original_identity = id(original_tx)
    original_result, original_state = continue_original(original_tx)
    if id(original_tx) != original_identity:
        raise AssertionError("uninterrupted branch did not keep the original transaction object")
    restored_tx = restore_from_disk(snapshot)
    if restored_tx is original_tx:
        raise AssertionError("disk restore must reconstruct a separate transaction object")
    restored_result, restored_state = continue_restored(restored_tx)
    exact_equal(original_result, restored_result, "opportunities 4-6 reports")
    exact_equal(original_state, restored_state, "opportunities 4-6 transaction/process state")
    return original_result, original_state, restored_result, restored_state


def _preflight(args):
    checked = probe._preflight_inputs(args)
    output = Path(args.out_dir).resolve()
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent must already exist: {output.parent}")
    if output.exists():
        raise FileExistsError(output)
    checked["output_dir"] = output
    return checked


def _source_receipt(checked) -> dict:
    paths = [Path(__file__).resolve(), Path(probe.__file__).resolve(),
             TOOLS / "launch_banking_memory_policy_v11.py",
             TOOLS / "launch_banking_memory_policy_v10.py"]
    return {path.relative_to(ROOT).as_posix(): _sha_file(path) for path in paths}


def run(args) -> dict:
    checked = _preflight(args)
    if args.phase == "preflight":
        receipt_path = args.preflight_receipt.resolve()
        if receipt_path.exists():
            raise FileExistsError(receipt_path)
        checked["driver"]._validate_output_path(receipt_path,
            checked["project_root"], checked["model_path"])
        if receipt_path == checked["output_dir"] or checked["output_dir"] in receipt_path.parents:
            raise ValueError("preflight receipt must be outside the fresh run output directory")
        _atomic_json(receipt_path, {
            "schema": SCHEMA + ".preflight", "status": "preflight_pass",
            "input_preflight": checked["summary"], "source_sha256": _source_receipt(checked),
            "memory_policy": {"allocator_conf": "garbage_collection_threshold:0.7",
                "per_process_fraction": 0.8, "queued_before_cuda_init": False},
            "model_loaded": False, "cuda_initialized": False,
            "optimizer_updates": 0, "test_data_read": False})
        return {"schema": SCHEMA + ".preflight", "status": "preflight_pass",
                "model_loaded": False, "optimizer_updates": 0}

    out = checked["output_dir"]
    started = time.monotonic()
    source_shas = _source_receipt(checked)
    budget = None
    counted_work = {"tail_calls": 0, "tail_calls_with_grad": 0,
                    "tail_calls_without_grad": 0}
    model = None
    original_logits = original_tail = None
    result = {"phase": "load_model", "physical_updates": 0}
    out.mkdir(parents=False, exist_ok=False)
    try:
        torch = checked["torch"]
        policy = __import__("tools.launch_banking_memory_policy_v11", fromlist=["*"])
        v10_policy = __import__("tools.launch_banking_memory_policy_v10", fromlist=["*"])
        policy_state = policy._install_reviewed_policy(torch, checked["cfg"]["device"], v10_policy)
        if torch.cuda.is_initialized() or not policy_state.get("registered"):
            raise RuntimeError("memory fraction policy was not deferred before CUDA initialization")

        driver, cfg = checked["driver"], checked["cfg"]
        budget_cfg = dict(checked["budget_cfg"])
        budget_cfg.update(CAPS)
        budget = driver.Budget(budget_cfg)
        budget.started = time.monotonic()
        loaded_torch, tokenizer, model, immutable_versions = driver._load_model(
            checked["project_root"], checked["model_path"], cfg)
        if loaded_torch is not torch:
            raise RuntimeError("V10 loader returned a different torch module")
        model.requires_grad_(False)
        frozen_versions = immutable_versions(model)
        original_logits = model.logits

        def counted_logits(tokens, mask=None):
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_logits(tokens, mask)
        model.logits = counted_logits

        tail_runtime = checked["tail_runtime"]
        original_tail = tail_runtime.tail_logits
        def counted_tail_logits(tail_model, context, cache):
            active = torch.is_grad_enabled() and any(p.requires_grad for p in context.parameters())
            budget.check()
            counted_work["tail_calls"] += 1
            counted_work["tail_calls_with_grad"] += int(active)
            counted_work["tail_calls_without_grad"] += int(not active)
            return original_tail(tail_model, context, cache)
        tail_runtime.tail_logits = counted_tail_logits

        common, runtime = checked["common"], checked["runtime"]
        state = checked["foundation_state"]
        driver._restore_rng(state["rng"], torch)
        live = checked["clone_mutable_context"](model)
        live.load_state_dict(state["context"])
        base_names = set(__import__("clinc_lifecycle_v8")._base_names(0))
        params = dict(live.named_parameters())
        prospective = {name: params[name].detach().clone() for name in base_names}
        tx = checked["AdditiveSleepTransaction"](live, 0, prospective,
            lr=cfg["sleep_lr"], weight_decay=cfg["sleep_weight_decay"],
            clip_norm=cfg["sleep_clip_norm"], seed=int(cfg["initialization_seed"]) + probe.SEED)
        schedule = probe.select_old_prompt_schedule(checked["train_rows"],
            checked["manifest"]["class_orders"][str(probe.SEED)])
        _ = probe.safe_schedule_receipt(schedule)  # deterministic schedule integrity check
        budget.sleep_events += 1
        result = {"phase": "first_triplet", "physical_updates": 0}
        triplet_0_3 = probe._run_triplet(model, tx, schedule[0:3], runtime=runtime,
            common=common, driver=driver, cfg=cfg, tokenizer=tokenizer,
            classes=tuple(checked["manifest"]["classes"]), torch=torch,
            device=next(model.parameters()).device, budget=budget, arm="SLEEP_ADDITIVE")
        snapshot3 = tx.snapshot()
        snapshot_hash = probe._tree_hash(snapshot3, torch)
        envelope = {"schema": "banking-cil-v10-original-tx-replay-checkpoint-v1",
            "next_call": 3, "seed": probe.SEED, "task_id": probe.TASK_ID,
            "config_sha256": checked["config_sha"],
            "banking_driver_source_lock_sha256": checked["source_lock_sha"],
            "probe_source_lock_sha256": checked["probe_source_lock_sha"],
            "verifier_source_sha256": source_shas[Path(__file__).resolve().relative_to(ROOT).as_posix()],
            "foundation_checkpoint_sha256": checked["foundation_sha"],
            "foundation_receipt_sha256": checked["receipt_sha"],
            "training_manifest_sha256": cfg["training_manifest_sha256"],
            "snapshot_sha256": snapshot_hash, "transaction_snapshot": snapshot3}
        cp_path = out / "private-checkpoints" / "original-tx-next-call-0003.pt"
        cp_sha = probe._atomic_torch(cp_path, envelope, torch)
        cp_loaded = torch.load(cp_path, map_location="cpu", weights_only=False)
        if (_sha_file(cp_path) != cp_sha or cp_loaded.get("schema") != envelope["schema"] or
                probe._tree_hash(cp_loaded["transaction_snapshot"], torch) != snapshot_hash):
            raise RuntimeError("on-disk transaction snapshot failed readback verification")

        where = next(model.parameters()).device
        classes = tuple(checked["manifest"]["classes"])
        def continue_original(original_tx):
            reports = probe._run_triplet(model, original_tx, schedule[3:6], runtime=runtime,
                common=common, driver=driver, cfg=cfg, tokenizer=tokenizer, classes=classes,
                torch=torch, device=where, budget=budget, arm="SLEEP_ADDITIVE")
            return reports, probe._tree_hash(original_tx.snapshot(), torch)
        def restore_from_disk(saved):
            return checked["AdditiveSleepTransaction"].from_snapshot(live, saved["transaction_snapshot"])
        def continue_restored(restored_tx):
            reports = probe._run_triplet(model, restored_tx, schedule[3:6], runtime=runtime,
                common=common, driver=driver, cfg=cfg, tokenizer=tokenizer, classes=classes,
                torch=torch, device=where, budget=budget, arm="SLEEP_ADDITIVE")
            return reports, probe._tree_hash(restored_tx.snapshot(), torch)
        original_report, original_hash, restored_report, restored_hash = original_then_restore(
            tx, cp_loaded, continue_original=continue_original,
            restore_from_disk=restore_from_disk, continue_restored=continue_restored,
            exact_equal=probe._same_exact)
        if original_hash != restored_hash:
            raise RuntimeError("original continuation and disk replay states differ")
        # The budget is incremented by each physical update as it happens.
        # Snapshot counters are cumulative from opportunity 0 and therefore
        # must not be summed across the two mutually exclusive branches.
        updates = int(budget.adamw_updates)
        expected_updates = 9
        if updates != expected_updates:
            raise RuntimeError(f"expected {expected_updates} physical AdamW updates across three triplets; got {updates}")
        if immutable_versions(model) != frozen_versions:
            raise RuntimeError("immutable model guard failed")
        work = probe._probe_work_snapshot(budget, counted_work)
        proof = {"schema": SCHEMA, "status": "complete", "seed": probe.SEED,
            "task_id": probe.TASK_ID, "development_only": True, "test_data_read": False,
            "official_test_data_loaded": False, "readout_performed": False,
            "config_sha256": checked["config_sha"],
            "banking_driver_source_lock_sha256": checked["source_lock_sha"],
            "probe_source_lock_sha256": checked["probe_source_lock_sha"],
            "source_sha256": source_shas,
            "foundation_checkpoint_sha256": checked["foundation_sha"],
            "foundation_receipt_sha256": checked["receipt_sha"],
            "training_manifest_sha256": cfg["training_manifest_sha256"],
            "schedule_sha256": probe._canonical_sha(probe.safe_schedule_receipt(schedule)),
            "completed_calls": 6, "physical_updates_across_all_branches": updates,
            "snapshot_at_next_call": 3, "snapshot_sha256": snapshot_hash,
            "checkpoint_relative_path": cp_path.relative_to(out).as_posix(),
            "checkpoint_sha256": cp_sha,
            "original_transaction_object_continued": True,
            "disk_transaction_restored_separately": True,
            "exact_reports_and_transaction_process_rng_state": True,
            "original_branch_state_sha256": original_hash,
            "restored_branch_state_sha256": restored_hash,
            "triplet_0_3": triplet_0_3,
            "original_calls_4_6": original_report,
            "restored_calls_4_6": restored_report,
            "work": work, "elapsed_seconds": time.monotonic() - started,
            "memory_policy": {"allocator_conf": "garbage_collection_threshold:0.7",
                "memory_fraction": 0.8,
                "registered_before_cuda_initialization": bool(policy_state["registered"])}}
        _atomic_json(out / "result.json", proof)
        return proof
    except BaseException as exc:
        if out.exists():
            failure = {"schema": SCHEMA, "status": "failed", "phase": result.get("phase"),
                "error_type": type(exc).__name__, "error": str(exc),
                "work": probe._probe_work_snapshot(budget, counted_work) if budget is not None else None,
                "test_data_read": False, "official_test_data_loaded": False,
                "readout_performed": False, "source_sha256": source_shas,
                "elapsed_seconds": time.monotonic() - started}
            if not (out / "failure.json").exists():
                _atomic_json(out / "failure.json", failure)
        raise
    finally:
        if model is not None and original_logits is not None:
            model.logits = original_logits
        if original_tail is not None:
            checked["tail_runtime"].tail_logits = original_tail
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "model-path", "training-pack", "config", "foundation-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "run"), default="run")
    parser.add_argument("--preflight-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "preflight" and args.preflight_receipt is None:
        parser.error("--preflight-receipt is required with --phase preflight")
    value = run(args)
    print(json.dumps({k: value.get(k) for k in
        ("schema", "status", "seed", "completed_calls", "physical_updates_across_all_branches",
         "model_loaded", "test_data_read")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
