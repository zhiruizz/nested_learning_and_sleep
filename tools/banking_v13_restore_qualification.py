"""One-time B32_HIGH seed-17 V13 snapshot/replay qualification callback.

This is engineering work only. It snapshots after requests 0:32, lets the
original transaction finish 32:64, then replays 32:64 from the serialized
snapshot in an isolated transaction. The extra replay costs one 32-request
microbatch/one optimizer attempt and is counted separately by the caller.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"qualification artifact already exists: {path.name}")
    raw = _canonical_bytes(value)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Allow only public numeric batch metrics; reject arbitrary callback data."""
    allowed = {"request_count", "gkd_opportunities", "lti_opportunities",
               "backward_calls", "token_count", "physical_adamw_steps",
               "skipped_zero_gradient", "loss_mean", "preclip_grad_norm",
               "parameter_update_l2"}
    if set(value) - allowed:
        raise ValueError("unexpected nonaggregate fields in qualification batch metrics")
    result = {}
    for key, item in value.items():
        if type(item) not in (bool, int, float):
            raise TypeError("qualification batch metrics must be finite scalar values")
        if type(item) is float and not np.isfinite(item):
            raise ValueError("nonfinite qualification batch metric")
        result[key] = item
    return result


def _rng_from_snapshot(snapshot, torch) -> None:
    payload = snapshot.payload
    torch.set_rng_state(payload.cpu_rng.detach().cpu().clone())
    if payload.cuda_rng is not None:
        torch.cuda.set_rng_state_all([item.detach().cpu().clone() for item in payload.cuda_rng])
    random.setstate(deepcopy(payload.python_rng))
    np.random.set_state(deepcopy(payload.numpy_rng))


class B32RestoreQualification:
    """Callable ``on_batch_committed`` hook for the first B32_HIGH event only.

    The objective factory must construct a fresh closure over the supplied
    restored transaction. It must preserve the regular V13 objective and may
    not write raw prompts, token IDs, labels, or sampled text to this receipt.
    """

    def __init__(self, *, selected: Sequence[tuple[str, Any, str]], tx,
                 objective_builder_factory: Callable[[Any], Callable[..., Mapping[str, Any]]],
                 budget, output_dir: Path, torch, seed: int, arm: str):
        if type(seed) is not int or seed != 17 or arm != "B32_HIGH":
            raise ValueError("restore qualification is restricted to seed 17 / B32_HIGH")
        if len(selected) < 64 or not callable(objective_builder_factory):
            raise ValueError("qualification needs a 64-request stream and objective factory")
        self.selected = selected
        self.tx = tx
        self.objective_builder_factory = objective_builder_factory
        self.budget = budget
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError("qualification output directory must be fresh")
        self.torch = torch
        self.seed, self.arm = seed, arm
        self.snapshot_path = self.output_dir / "after32-private.pt"
        self.receipt_path = self.output_dir / "restore-qualification.json"
        self.failure_path = self.output_dir / "failure.json"
        self.snapshot_file_sha256 = None
        self.snapshot_state_sha256 = None
        self.original_second_metrics = None
        self.done = False
        self._v12 = None

    def _hash_tools(self):
        if self._v12 is None:
            import run_banking_optimizer_boundary_v12 as v12
            self._v12 = v12
        return self._v12

    def __call__(self, batch_index: int, next_request_offset: int,
                 next_gkd_ordinal: int, batch_result: Mapping[str, Any]) -> None:
        if self.done:
            raise RuntimeError("V13 restore qualification callback already completed")
        if batch_index == 0 and next_request_offset == 32:
            try:
                self._save_after32(next_gkd_ordinal)
            except BaseException as exc:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                if not self.failure_path.exists():
                    _write_json_new(self.failure_path, {
                        "schema": "banking-v13-b32-restore-qualification-failure-v1",
                        "status": "failed", "seed": self.seed, "arm": self.arm,
                        "event_index": 0, "next_request_offset": 32,
                        "error_type": type(exc).__name__,
                    })
                raise
            return
        if batch_index == 1 and next_request_offset == 64:
            self._qualify_after64(next_gkd_ordinal, batch_result)
            return
        raise RuntimeError("qualification callback received an unexpected event/batch boundary")

    def _save_after32(self, next_gkd_ordinal: int) -> None:
        if next_gkd_ordinal != 22 or self.snapshot_file_sha256 is not None:
            raise RuntimeError("unexpected GKD cursor or duplicate after-32 qualification")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        v12 = self._hash_tools()
        snapshot = self.tx.snapshot()
        expected_state_sha = v12.stable_state_sha(snapshot, self.torch)
        envelope = {"schema": "banking-v13-b32-restore-private-snapshot-v1",
                    "seed": self.seed, "arm": self.arm,
                    "next_request_offset": 32, "next_gkd_ordinal": 22,
                    "snapshot": snapshot}
        self.snapshot_file_sha256 = v12._atomic_torch(self.snapshot_path, envelope, self.torch)
        reread_sha = hashlib.sha256(self.snapshot_path.read_bytes()).hexdigest()
        if reread_sha != self.snapshot_file_sha256:
            raise RuntimeError("qualification checkpoint file hash changed after write")
        loaded = self.torch.load(self.snapshot_path, map_location="cpu", weights_only=False)
        if loaded.get("schema") != envelope["schema"] or (loaded.get("seed"), loaded.get("arm")) != (17, "B32_HIGH"):
            raise RuntimeError("qualification checkpoint envelope mismatch")
        self.snapshot_state_sha256 = v12.stable_state_sha(loaded["snapshot"], self.torch)
        if self.snapshot_state_sha256 != expected_state_sha:
            raise RuntimeError("serialized request-32 snapshot content differs")

    def _qualify_after64(self, next_gkd_ordinal: int,
                         batch_result: Mapping[str, Any]) -> None:
        if self.snapshot_file_sha256 is None or next_gkd_ordinal != 43:
            raise RuntimeError("after-64 qualification lacks the request-32 snapshot/cursor")
        original_metrics = _metric_projection(batch_result)
        self.original_second_metrics = original_metrics
        torch = self.torch
        v12 = self._hash_tools()
        original_post = self.tx.snapshot()
        original_post_sha = v12.stable_state_sha(original_post, torch)
        budget_before = self.budget.as_dict() if self.budget is not None else None

        # The original transaction has already completed requests 32:64.
        # Restoring from the read-back file changes process RNGs; put them back
        # to the original post-64 state in the finally block below.
        try:
            loaded = torch.load(self.snapshot_path, map_location="cpu", weights_only=False)
            if hashlib.sha256(self.snapshot_path.read_bytes()).hexdigest() != self.snapshot_file_sha256:
                raise RuntimeError("saved request-32 checkpoint changed before restore")
            saved = loaded["snapshot"]
            if v12.stable_state_sha(saved, torch) != self.snapshot_state_sha256:
                raise RuntimeError("read-back request-32 snapshot content changed")
            from banking_cil_v13_batching import BatchableAdditiveSleepTransactionV13
            restored = BatchableAdditiveSleepTransactionV13.from_snapshot(
                deepcopy(self.tx.live), saved)
            objective_builder = self.objective_builder_factory(restored)
            from banking_sleep_v13_runtime import run_batched_sleep_event
            replay = run_batched_sleep_event(
                self.selected[:64], restored, microbatch_size=32,
                objective_builder=objective_builder, budget=self.budget,
                request_offset=32, gkd_ordinal_start=22)
            if (replay["processed_opportunities"] != 32 or
                    replay["gkd_ordinal_start"] != 22 or
                    replay["physical_adamw_steps"] != 1 or len(replay["batches"]) != 1):
                raise RuntimeError("restored branch did not replay exactly one 32-request batch")
            restored_metrics = _metric_projection(replay["batches"][0])
            restored_post = restored.snapshot()
            restored_post_sha = v12.stable_state_sha(restored_post, torch)
            metrics_sha_equal = (v12.stable_state_sha(original_metrics, torch) ==
                                 v12.stable_state_sha(restored_metrics, torch))
            state_sha_equal = original_post_sha == restored_post_sha
            budget_after = self.budget.as_dict() if self.budget is not None else None
            extra_work = None
            if budget_before is not None and budget_after is not None:
                extra_work = {key: budget_after[key] - value
                              for key, value in budget_before.items()
                              if key in budget_after and
                              isinstance(value, (int, float)) and
                              isinstance(budget_after[key], (int, float))}
            if not metrics_sha_equal or not state_sha_equal:
                raise RuntimeError("original and disk-restored requests 32:64 diverged")
            receipt = {
                "schema": "banking-v13-b32-restore-qualification-v1",
                "status": "pass", "seed": self.seed, "arm": self.arm,
                "event_index": 0,
                "snapshot_file_sha256": self.snapshot_file_sha256,
                "snapshot_content_sha256": self.snapshot_state_sha256,
                "original_post_range_content_sha256": original_post_sha,
                "restored_post_range_content_sha256": restored_post_sha,
                "original_request_range": [32, 64],
                "restored_request_range": [32, 64],
                "original_batch_metrics": original_metrics,
                "restored_batch_metrics": restored_metrics,
                "batch_metrics_content_equal": metrics_sha_equal,
                "full_snapshot_content_equal": state_sha_equal,
                "extra_request_evaluations": 32,
                "extra_physical_adamw_steps": 1,
                "extra_work_delta": extra_work,
                "global_rng_restored_to_original_post64": True,
            }
        except BaseException as exc:
            _write_json_new(self.failure_path, {
                "schema": "banking-v13-b32-restore-qualification-failure-v1",
                "status": "failed", "seed": self.seed, "arm": self.arm,
                "event_index": 0, "next_request_offset": 64,
                "snapshot_file_sha256": self.snapshot_file_sha256,
                "error_type": type(exc).__name__,
                "original_post_range_content_sha256": original_post_sha,
                "global_rng_restored_to_original_post64": True,
            })
            raise
        finally:
            _rng_from_snapshot(original_post, torch)

        _write_json_new(self.receipt_path, receipt)
        self.done = True


def make_b32_restore_qualification_callback(**kwargs) -> B32RestoreQualification:
    """Convenience factory for the V13 first-event B32_HIGH qualification."""
    return B32RestoreQualification(**kwargs)
