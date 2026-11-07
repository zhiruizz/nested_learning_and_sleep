#!/usr/bin/env python3
"""Candidate full CLINC V8 runner; intentionally does not read the test role.

One process runs exactly one foundation seed, one continuation seed/arm, or a
read-only six-run seal aggregation. The main comparison is the paired grid
{17,29,43} x {HOPE, HOPE_SLEEP_FIXEDMIX}. The fixed mix is an explicit routing
intervention, not natural Sleep and not a paper-faithful HOPE reproduction.

The master config must be reviewed and marked ``status: frozen`` before any
training invocation. This candidate leaves optimizer/reward values in that
external config so they can be frozen only after the task probes are reviewed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback
from typing import Mapping, Sequence


SCHEMA = "clinc-cil-v8-full-master-candidate"
SEEDS = (17, 29, 43)
ARMS = ("HOPE", "HOPE_SLEEP_FIXEDMIX")
TASKS = tuple(range(10))
PERIODS = (16, 64, 256, 1024, None)
RECEIPT_SCHEMA = "clinc-cil-v8-full-run-receipt"
FOUNDATION_SCHEMA = "clinc-cil-v8-common-foundation-receipt"
CHECKPOINT_SCHEMA = "clinc-cil-v8-full-safe-checkpoint"
SEAL_SCHEMA = "clinc-cil-v8-main-six-seal"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value, *, refuse_existing: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                         allow_nan=False).encode("utf-8")
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if refuse_existing:
            # Windows rename is atomic and fails if the destination exists;
            # on POSIX an exclusive hard-link install has the same property.
            if os.name == "nt":
                os.rename(tmp, path)
            else:
                os.link(tmp, path)
                tmp.unlink()
        else:
            os.replace(tmp, path)
        actual = sha256_file(path)
        if actual != hashlib.sha256(payload).hexdigest():
            raise IOError("atomic JSON readback SHA mismatch")
        return actual
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_torch(path: Path, value, torch) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    tmp = Path(raw)
    try:
        torch.save(value, tmp)
        with tmp.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(tmp)
        os.replace(tmp, path)
        if sha256_file(path) != digest:
            raise IOError("atomic checkpoint readback SHA mismatch")
        check = torch.load(path, map_location="cpu", weights_only=False)
        if check.get("schema") != value.get("schema"):
            raise IOError("checkpoint readback schema mismatch")
        return digest
    finally:
        if tmp.exists():
            tmp.unlink()


def _hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def validate_master_config(cfg: Mapping) -> dict:
    """Validate the full experiment contract before model loading."""
    if cfg.get("schema") != SCHEMA:
        raise ValueError("master config schema mismatch")
    if cfg.get("status") != "frozen":
        raise ValueError("candidate config is not frozen; review probes before training")
    if cfg.get("seed_grid") != list(SEEDS) or cfg.get("arms") != list(ARMS):
        raise ValueError("master grid must be exactly the locked three-seed/two-arm comparison")
    if cfg.get("tasks") != list(TASKS):
        raise ValueError("master config must cover tasks 0 through 9")
    required = {
        "source_revision", "model_revision", "model_assets_sha256", "training_manifest_sha256",
        "expected_normalized_train_dev_duplicates", "initialization_seed", "rank", "capacity",
        "levels", "periods", "batch_size", "examples_per_task", "epochs_per_task",
        "max_steps_per_task", "base_lr", "fast_lr", "sleep_lr", "sleep_weight_decay",
        "sleep_clip_norm", "gkd_calls_per_event", "lti_calls_per_event", "max_new_tokens",
        "lti_edit_distance_z0", "lti_semantic_weight", "lti_edit_weight", "temperature",
        "max_sequence_length", "task0_clean_accuracy_gate", "fixed_mix_fraction",
        "budgets", "resume_policy", "evaluation_policy", "inner_cycles_per_event", "objective_version",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"master config lacks keys: {sorted(missing)}")
    if cfg["rank"] != 64 or cfg["capacity"] != 8 or cfg["levels"] != 5:
        raise ValueError("full-run architecture is locked to rank 64, capacity 8, five levels")
    if cfg["periods"] != list(PERIODS):
        raise ValueError("periods must be [16,64,256,1024,None]")
    if cfg["batch_size"] != 8 or cfg["examples_per_task"] != 1500 or cfg["epochs_per_task"] != 1:
        raise ValueError("task schedule must be one pass over 1,500 rows at batch size 8")
    if cfg["max_steps_per_task"] != 188:
        raise ValueError("each task must have the 188-update schedule, including its final four-row batch")
    if cfg["initialization_seed"] != 1907008:
        raise ValueError("shared backbone/mutable initialization seed is fixed at 1907008")
    if cfg["task0_clean_accuracy_gate"] != 0.5:
        raise ValueError("common foundation gate must be the fixed 0.5 clean-development threshold")
    if cfg["fixed_mix_fraction"] != 0.5:
        raise ValueError("fixed-route arm must use the controlled lifecycle's 0.5 mix")
    if cfg["resume_policy"] != "manual_safe_checkpoint_only":
        raise ValueError("resume policy must be explicit safe-checkpoint-only")
    if cfg["evaluation_policy"] != "one_clean_seen_sweep_at_each_task_end":
        raise ValueError("evaluation schedule must use one clean seen-class sweep at task end")
    if cfg["objective_version"] != "canonical_agreement_plus_thresholded_length_normalized_edit_v1":
        raise ValueError("full-run LTI edit objective version is not explicitly locked")
    for key in ("base_lr", "fast_lr", "sleep_lr", "sleep_clip_norm", "temperature"):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(float(cfg["sleep_weight_decay"])) or cfg["sleep_weight_decay"] < 0:
        raise ValueError("sleep_weight_decay must be finite and nonnegative")
    k = cfg["inner_cycles_per_event"]
    if k not in (1, 10) or cfg["gkd_calls_per_event"] != 2 * k or cfg["lti_calls_per_event"] != k:
        raise ValueError("inner cycles must be 1 or 10, repeated as [GKD,GKD,LTI] triplets")
    if type(cfg["max_new_tokens"]) is not int or cfg["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(cfg["lti_edit_distance_z0"]) is not int or cfg["lti_edit_distance_z0"] < 1:
        raise ValueError("lti_edit_distance_z0 must be a positive integer")
    if cfg["lti_semantic_weight"] != 0.5 or cfg["lti_edit_weight"] != 0.5:
        raise ValueError("local LTI proxy weights must be locked at 0.5/0.5")
    if cfg["max_sequence_length"] != 128:
        raise ValueError("max sequence length must remain 128")
    if not _hex64(cfg["training_manifest_sha256"]):
        raise ValueError("training manifest SHA must be hex64")
    if not isinstance(cfg["model_assets_sha256"], dict) or not cfg["model_assets_sha256"]:
        raise ValueError("model assets need an explicit hash map")
    if any(not _hex64(value) for value in cfg["model_assets_sha256"].values()):
        raise ValueError("model asset hashes must be SHA256 hex")
    if type(cfg["expected_normalized_train_dev_duplicates"]) is not int or cfg["expected_normalized_train_dev_duplicates"] < 0:
        raise ValueError("expected duplicate count must be a nonnegative integer")
    if type(cfg["initialization_seed"]) is not int:
        raise ValueError("initialization seed must be integer")
    if not isinstance(cfg["source_revision"], str) or not cfg["source_revision"]:
        raise ValueError("source revision is required")
    if any(token in cfg["source_revision"].casefold() for token in ("tbd", "placeholder", "candidate")):
        raise ValueError("frozen master source_revision cannot be a placeholder")
    if not isinstance(cfg["model_revision"], str) or not cfg["model_revision"]:
        raise ValueError("model revision is required")
    budgets = cfg["budgets"]
    for key, expected in (("foundation", {"max_wall_seconds": 1800, "max_forward_calls": 10000, "max_update_steps": 500}),
                          ("continuation", {"max_wall_seconds": 7200, "max_forward_calls": 100000, "max_update_steps": 7000})):
        if not isinstance(budgets.get(key), dict):
            raise ValueError(f"missing {key} budget")
        for field, minimum in expected.items():
            if type(budgets[key].get(field)) is not int or budgets[key][field] < minimum:
                raise ValueError(f"{key} {field} must be at least {minimum}")
    return dict(cfg)


def _child_config_sha(master_sha: str, mode: str, seed: int, arm: str | None) -> str:
    return canonical_sha({"master_config_sha256": master_sha, "mode": mode,
                          "seed": int(seed), "arm": arm})


def mode_task_range(mode: str, first_task: int) -> range:
    """Return only the legal training tasks for a foundation or continuation."""
    if mode == "foundation":
        if first_task not in (0, 1):
            raise ValueError("foundation cursor must be task 0 or the completed-task-0 cursor 1")
        return range(first_task, 1)
    if mode == "continuation":
        if not 1 <= first_task <= 10:
            raise ValueError("continuation cursor must be task 1 through 10")
        return range(first_task, 10)
    raise ValueError("unknown training mode")


def assert_foundation_terminal_state(*, wake_ticks: int, pending_clocks: Mapping,
                                     task_metrics: Sequence[Mapping], cursor: Mapping) -> None:
    """Reject any foundation that trained beyond task 0 or ended mid-clock."""
    if wake_ticks != 188:
        raise ValueError("task-0 foundation must contain exactly 188 wake updates")
    if sorted(int(value) for value in pending_clocks.values()) != [12, 60, 188, 188]:
        raise ValueError("task-0 foundation must end at pending clocks 12/60/188/188")
    if len(task_metrics) != 1 or int(task_metrics[0].get("task_id", -1)) != 0:
        raise ValueError("foundation receipt must contain exactly the task-0 metric")
    if int(cursor.get("task_id", -1)) != 1 or int(cursor.get("tasks_completed", -1)) != 1:
        raise ValueError("foundation cursor must mark task 0 complete and next task 1")


def terminal_receipt_name(mode: str) -> str:
    if mode == "foundation":
        return "foundation-receipt.json"
    if mode == "continuation":
        return "run-receipt.json"
    raise ValueError("terminal marker only exists for training modes")


def refuse_terminal_run(run_dir: Path, mode: str) -> None:
    """Prevent resume or repeat invocation after either successful terminal receipt."""
    for name in ("foundation-receipt.json", "run-receipt.json"):
        if (run_dir / name).exists():
            raise ValueError(f"completed run is immutable; terminal marker exists: {name}")


def task_schedule_ids(train_rows: Sequence[Mapping], class_order: Sequence[int], seed: int,
                      task_id: int, examples_per_task: int = 1500) -> list[str]:
    """Deterministic schedule using training rows only; indices are role-local IDs."""
    from clinc_cil_common_v8 import shuffled_ids, task_class_ids
    allowed = set(task_class_ids(class_order, task_id))
    task_rows = [row for row in train_rows if int(row["intent_id"]) in allowed]
    if len(task_rows) < examples_per_task:
        raise ValueError(f"task {task_id} has fewer than {examples_per_task} train rows")
    result = shuffled_ids(task_rows, seed, task_id)[:examples_per_task]
    if len(result) != examples_per_task or len(set(result)) != examples_per_task:
        raise ValueError("training schedule is incomplete or duplicates row IDs")
    return result


def task_schedule_hash(row_ids: Sequence[str]) -> str:
    return canonical_sha(list(row_ids))


def class_balanced_replay_pools(train_rows: Sequence[Mapping], class_order: Sequence[int], task_id: int,
                                *, per_class: int = 4) -> dict:
    """Rebuild old/current text pools from train metadata, never labels in loss."""
    from clinc_cil_common_v8 import task_class_ids
    current_ids = set(task_class_ids(class_order, task_id))
    old_ids = set(class_order[:task_id * 15])
    def pool(class_ids):
        selected = []
        for class_id in sorted(class_ids):
            rows = sorted((r for r in train_rows if int(r["intent_id"]) == class_id),
                          key=lambda r: str(r["id"]))
            if len(rows) < per_class:
                raise ValueError(f"class {class_id} lacks replay rows")
            selected.extend(rows[:per_class])
        return selected
    old_pool, current_pool = pool(old_ids), pool(current_ids)
    if task_id > 0 and (not old_pool or not current_pool):
        raise ValueError("Sleep replay requires both prior-task and current-task pools")
    return {"old": old_pool, "current": current_pool}


def pool_ids_hash(pools: Mapping) -> str:
    return canonical_sha({key: [str(row["id"]) for row in value]
                          for key, value in pools.items() if key in ("old", "current")})


def select_full_sleep_prompts(pools: Mapping, *, event: int, task_id: int,
                              gkd_calls: int = 2, lti_calls: int = 1) -> list[tuple[str, Mapping, str]]:
    """Generalize the frozen probe's round-robin to every previously seen class."""
    if set(pools) != {"old", "current"} or event < 0 or task_id < 1:
        raise ValueError("selector needs old/current pools and positive task id")
    grouped = {}
    for role in ("old", "current"):
        unique = {str(row["id"]): row for row in pools[role]}
        by_class = {}
        for row_id, row in unique.items():
            by_class.setdefault(int(row["intent_id"]), []).append((row_id, row))
        classes = []
        for class_id in sorted(by_class):
            rows = [row for _, row in sorted(by_class[class_id])]
            if len(rows) < 4:
                raise ValueError(f"{role} class {class_id} has fewer than four unique pool IDs")
            classes.append(rows[:4])
        grouped[role] = classes
    if len(grouped["current"]) != 15 or len(grouped["old"]) != task_id * 15:
        raise ValueError("role pools do not match current/previous CIL class counts")
    if int(gkd_calls) != 2 * int(lti_calls) or int(lti_calls) not in (1, 10):
        raise ValueError("selector is locked to one or ten repeated [GKD,GKD,LTI] triplets")
    kinds = [kind for _ in range(int(lti_calls)) for kind in ("gkd", "gkd", "lti")]
    result = []
    for call, kind in enumerate(kinds):
        g = len(kinds) * int(event) + call
        role = "old" if (g + int(task_id)) % 2 == 0 else "current"
        exposure = g // 2
        nclasses = len(grouped[role])
        class_index = exposure % nclasses
        example_index = (exposure // nclasses) % 4
        result.append((kind, grouped[role][class_index][example_index], role))
    return result


def summarize_sleep_exposure(pools: Mapping, selected: Sequence[tuple[str, Mapping, str]]) -> dict:
    result = {role: {"calls": 0, "unique_ids": [], "classes_covered": [],
                     "class_count": len({int(row["intent_id"]) for row in pools[role]})}
              for role in ("old", "current")}
    for _kind, row, role in selected:
        result[role]["calls"] += 1
        result[role]["unique_ids"].append(str(row["id"]))
        result[role]["classes_covered"].append(int(row["intent_id"]))
    for role, item in result.items():
        item["unique_ids"] = sorted(set(item["unique_ids"]))
        item["unique_id_count"] = len(item["unique_ids"])
        item["classes_covered"] = sorted(set(item["classes_covered"]))
        item["classes_covered_count"] = len(item["classes_covered"])
        item["class_coverage_fraction"] = item["classes_covered_count"] / item["class_count"] if item["class_count"] else None
    return result


def plain_task_exposure(value: Mapping) -> dict:
    """JSON-safe copy of accumulated role/class coverage state."""
    out = {}
    for role, row in value.items():
        out[role] = {key: (sorted(item) if isinstance(item, set) else deepcopy(item))
                     for key, item in row.items()}
        covered = out[role].get("classes_covered", [])
        unique_ids = out[role].get("unique_ids", [])
        out[role]["selected_class_count"] = len(covered)
        out[role]["unique_id_count"] = len(unique_ids)
        denom = int(out[role].get("class_count", 0))
        out[role]["class_coverage_fraction"] = len(covered) / denom if denom else None
    return out


def paper_edit_similarity(left: Sequence[int], right: Sequence[int], *, z0: int) -> float:
    """Candidate's thresholded length-normalized edit component (not full LTI)."""
    if type(z0) is not int or z0 < 1:
        raise ValueError("z0 must be positive")
    a, b = list(map(int, left)), list(map(int, right))
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    distance = row[-1]
    if distance > z0:
        return 0.0
    denominator = max(len(a), len(b))
    return 1.0 if denominator == 0 else 1.0 - distance / denominator


def full_lti_reward(teacher_text: str, student_text: str, *, teacher_tokens: Sequence[int],
                    student_tokens: Sequence[int], z0: int, canonical_labels: Sequence[str], r2) -> float:
    """Local canonical-label equivalence plus corrected thresholded edit proxy.

    This remains a local proxy; it is not the paper's trained/frozen reward model.
    """
    t, s = r2.normalize_text(teacher_text), r2.normalize_text(student_text)
    labels = {r2.normalize_text(x) for x in canonical_labels}
    semantic = float(bool(t) and t == s and t in labels)
    edit = paper_edit_similarity(teacher_tokens, student_tokens, z0=z0)
    return 0.5 * semantic + 0.5 * edit


def run_full_controlled_sleep_event(model, tx, selected, tokenizer, view, cfg,
                                    torch, device, budget, r2, probe,
                                    tail_runtime, last_event):
    """Full-run GKD/LTI step, adapting only class selection and edit proxy."""
    from clinc_tail_runtime_v8 import capture, causal_token_ce, causal_token_kl
    outputs = []
    gkd_index = 0
    valid_labels = {r2.normalize_text(x) for x in view.classes}
    for kind, row, role in selected:
        budget.check()
        prompt = r2._prefix(row, tokenizer, view, cfg)
        owner = "teacher" if kind == "lti" or gkd_index % 2 == 0 else "student"
        if kind == "gkd":
            gkd_index += 1
        owner_context = tx.teacher_context() if owner == "teacher" else tx.student_context()
        sampled_result, sampled_answer = r2._sample_tokens(model, owner_context, prompt,
            cfg["max_new_tokens"], cfg["temperature"], tx.generator, tokenizer.eos_token_id,
            torch, device, budget)
        if kind == "gkd":
            sequence = sampled_result.token_ids
            answer_mask = sampled_result.answer_mask()
            cache = capture(model, sequence, torch.ones_like(sequence), device=device)
            with torch.no_grad():
                teacher_logits = tail_runtime.tail_logits(model, tx.teacher_context(), cache)
            student_logits, route_stats = probe._audited_student_logits(model, tx.student_context(),
                cache, tx.receiver, tx.slot, answer_mask, tail_runtime)
            loss, token_count = causal_token_kl(student_logits, teacher_logits, answer_mask, temperature=1.0)
            reward = None
        else:
            teacher_ids = sampled_answer[:-1] if sampled_answer and sampled_answer[-1] == tokenizer.eos_token_id else sampled_answer
            prefix_len = (int(torch.randint(0, len(teacher_ids), (1,), generator=tx.generator,
                             device=tx.generator.device).item()) if teacher_ids else 0)
            student_prefix = prompt + teacher_ids[:prefix_len]
            eos_emitted = bool(sampled_answer and sampled_answer[-1] == tokenizer.eos_token_id)
            suffix_budget = max(1, len(teacher_ids) - prefix_len + int(eos_emitted))
            student_result, student_answer = r2._sample_tokens(model, tx.student_context(),
                student_prefix, max(1, min(cfg["max_new_tokens"], suffix_budget)),
                cfg["temperature"], tx.generator, tokenizer.eos_token_id, torch, device, budget)
            sequence = student_result.token_ids
            answer_mask = student_result.answer_mask()
            cache = capture(model, sequence, torch.ones_like(sequence), device=device)
            student_logits, route_stats = probe._audited_student_logits(model, tx.student_context(),
                cache, tx.receiver, tx.slot, answer_mask, tail_runtime)
            with torch.no_grad():
                teacher_logits = tail_runtime.tail_logits(model, tx.teacher_context(), cache)
            kl, token_count = causal_token_kl(student_logits, teacher_logits, answer_mask, temperature=1.0)
            teacher_text = r2._decode_answer(teacher_ids, tokenizer, tokenizer.eos_token_id)
            student_suffix = student_answer[:-1] if student_answer and student_answer[-1] == tokenizer.eos_token_id else student_answer
            student_full = teacher_ids[:prefix_len] + student_suffix
            student_text = r2._decode_answer(student_full, tokenizer, tokenizer.eos_token_id)
            reward = full_lti_reward(teacher_text, student_text, teacher_tokens=teacher_ids,
                student_tokens=student_full, z0=cfg["lti_edit_distance_z0"],
                canonical_labels=view.classes, r2=r2)
            student_ce, ce_tokens = causal_token_ce(student_logits, sequence, answer_mask)
            loss, token_count = 0.5 * kl + 0.5 * reward * student_ce, token_count + ce_tokens
        loss_value = float(loss.detach().float().cpu())
        row_sha = hashlib.sha256(str(row["id"]).encode()).hexdigest()
        if not bool(torch.isfinite(loss)) or not loss.requires_grad:
            last_event.update({"loss": loss_value, "loss_requires_grad": bool(loss.requires_grad),
                               "route": route_stats, "kind": kind, "row_id_sha256": row_sha})
            raise RuntimeError("controlled Sleep produced a nonfinite or nondifferentiable loss")
        budget.backward(torch.ones((1, token_count), dtype=torch.long, device=device))
        budget.update()
        previous_zero = tx.counters["zero_gradient_steps"]
        tx.optimize(loss, kind, max(1, int(token_count)))
        if tx.counters["zero_gradient_steps"] != previous_zero:
            last_event.update({"loss": loss_value, "route": route_stats, "kind": kind,
                               "row_id_sha256": row_sha, "zero_gradient_step": True})
            raise RuntimeError("controlled receiver optimizer observed zero gradient")
        sampled_text = r2._decode_answer(sampled_answer, tokenizer, tokenizer.eos_token_id)
        outputs.append({"kind": kind, "sample_owner": owner, "loss": loss_value,
            "token_count": int(token_count), "kl_weight": 1.0 if kind == "gkd" else 0.5,
            "lti_reward_proxy": reward, "sampled_label_valid": r2.normalize_text(sampled_text) in valid_labels,
            "sampled_label_matches_prompt_gold_diagnostic": r2.normalize_text(sampled_text) ==
                r2.normalize_text(view.classes[int(row["intent_id"])]),
            "prompt_id_sha256": row_sha, "prompt_role": role,
            "prompt_class_id": int(row["intent_id"]), "natural_vs_controlled_route": route_stats})
        last_event.update({"loss": loss_value, "kind": kind, "route": route_stats,
                           "row_id_sha256": row_sha, "prompt_role": role,
                           "prompt_class_id": int(row["intent_id"])})
    return outputs


def checkpoint_cursor(*, mode: str, task_id: int, next_step: int, task_count: int,
                      schedule_sha256: str, replay_ids_sha256: str) -> dict:
    if mode not in ("foundation", "continuation") or task_id not in TASKS or next_step < 1:
        raise ValueError("invalid checkpoint cursor")
    if not _hex64(schedule_sha256) or not _hex64(replay_ids_sha256):
        raise ValueError("cursor requires schedule and replay hashes")
    return {"mode": mode, "task_id": task_id, "next_step": next_step,
            "tasks_completed": task_count, "schedule_sha256": schedule_sha256,
            "replay_ids_sha256": replay_ids_sha256,
            "safe_point": "post_all_due_events"}


def validate_checkpoint_payload(payload: Mapping, *, master_config_sha256: str,
                                child_config_sha256: str, manifest_sha256: str,
                                source_lock_sha256: str, seed: int, arm: str | None) -> None:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("safe checkpoint schema mismatch")
    bindings = {"master_config_sha256": master_config_sha256,
                "child_config_sha256": child_config_sha256,
                "training_manifest_sha256": manifest_sha256,
                "source_lock_sha256": source_lock_sha256,
                "seed": seed, "arm": arm}
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint binding mismatch: {key}")
    cursor = payload.get("cursor")
    required_cursor = {"mode", "task_id", "next_step", "tasks_completed", "schedule_sha256",
                       "replay_ids_sha256", "safe_point"}
    if not isinstance(cursor, dict) or set(cursor) != required_cursor:
        raise ValueError("checkpoint cursor schema mismatch")
    if cursor["safe_point"] != "post_all_due_events":
        raise ValueError("checkpoint is not at a safe post-event boundary")
    if not _hex64(cursor["schedule_sha256"]) or not _hex64(cursor["replay_ids_sha256"]):
        raise ValueError("checkpoint cursor hashes malformed")
    for key in ("context", "wake", "rng", "work_cumulative", "physical_sleep_adamw_steps",
                "physical_wake_updates_run", "sleep_event_index", "sleep_exposure"):
        if key not in payload:
            raise ValueError(f"checkpoint lacks {key}")


def validate_foundation_receipts(receipts: Mapping[int, Mapping], *, master_sha: str,
                                 manifest_sha: str, source_lock_sha: str,
                                 model_revision: str, model_assets_sha256: Mapping) -> None:
    if set(map(int, receipts)) != set(SEEDS):
        raise ValueError("all three qualified task-0 foundations must be present")
    for seed in SEEDS:
        receipt = receipts.get(seed, receipts.get(str(seed)))
        if not isinstance(receipt, Mapping) or receipt.get("schema") != FOUNDATION_SCHEMA:
            raise ValueError(f"seed {seed} foundation receipt missing/schema mismatch")
        if receipt.get("status") != "complete" or receipt.get("acquisition_qualified") is not True:
            raise ValueError(f"seed {seed} common foundation did not pass its gate")
        if receipt.get("test_data_included") is not False:
            raise ValueError(f"seed {seed} foundation receipt does not affirm test-data isolation")
        for key, expected in (("seed", seed), ("master_config_sha256", master_sha),
                              ("training_manifest_sha256", manifest_sha),
                              ("source_lock_sha256", source_lock_sha),
                              ("child_config_sha256", _child_config_sha(master_sha, "foundation", seed, None)),
                              ("model_revision", model_revision),
                              ("model_assets_sha256", dict(model_assets_sha256))):
            if receipt.get(key) != expected:
                raise ValueError(f"seed {seed} foundation binding mismatch: {key}")
        if not _hex64(receipt.get("final_checkpoint_sha256")):
            raise ValueError(f"seed {seed} foundation checkpoint hash malformed")
        if float(receipt.get("task0_clean_current_accuracy", -1)) < 0.5:
            raise ValueError(f"seed {seed} foundation gate metric failed")
        if receipt.get("tasks_completed") != 1:
            raise ValueError(f"seed {seed} foundation must contain exactly one completed task")
        metrics = receipt.get("task_metrics")
        if not isinstance(metrics, list) or len(metrics) != 1 or metrics[0].get("task_id") != 0:
            raise ValueError(f"seed {seed} foundation must contain only task-0 metrics")
        if int(receipt.get("wake_ticks", -1)) != 188:
            raise ValueError(f"seed {seed} foundation wake count must equal 188")
        if sorted(int(value) for value in receipt.get("pending_clocks", {}).values()) != [12, 60, 188, 188]:
            raise ValueError(f"seed {seed} foundation pending clocks are not the task-0 terminal state")


def validate_main_grid_receipts(receipts: Sequence[Mapping], *, master_sha: str,
                                manifest_sha: str, source_lock_sha: str,
                                inner_cycles_per_event: int | None = None,
                                objective_version: str | None = None) -> None:
    """Pure grid gate; filesystem byte hashes are verified separately by seal_main_grid."""
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    actual = {(int(item.get("seed", -1)), str(item.get("arm", ""))) for item in receipts}
    if len(receipts) != 6 or actual != expected:
        raise ValueError("main comparison must contain exactly six unique seed/arm receipts")
    for item in receipts:
        if item.get("schema") != RECEIPT_SCHEMA or item.get("status") != "complete":
            raise ValueError("every main-grid run must have a terminal successful receipt")
        if item.get("tasks_completed") != 10:
            raise ValueError("every main-grid run must complete all ten tasks")
        task_metrics = item.get("task_metrics")
        if not isinstance(task_metrics, list) or [x.get("task_id") for x in task_metrics] != list(TASKS):
            raise ValueError("every main-grid run must retain task metrics for tasks 0 through 9")
        if item.get("acquisition_qualified") is not True or item.get("test_data_included") is not False:
            raise ValueError("main-grid receipt must affirm task-0 qualification and no test-data use")
        if not _hex64(item.get("foundation_checkpoint_sha256")):
            raise ValueError("run receipt foundation checkpoint hash malformed")
        if not _hex64(item.get("foundation_receipt_sha256")):
            raise ValueError("run receipt foundation receipt hash malformed")
        if inner_cycles_per_event is not None and item.get("inner_cycles_per_event") != inner_cycles_per_event:
            raise ValueError("run receipt inner-cycle count differs from the frozen master")
        if objective_version is not None and item.get("objective_version") != objective_version:
            raise ValueError("run receipt objective version differs from the frozen master")
        for key, expected_value in (("master_config_sha256", master_sha),
                                    ("training_manifest_sha256", manifest_sha),
                                    ("source_lock_sha256", source_lock_sha)):
            if item.get(key) != expected_value:
                raise ValueError(f"run receipt {key} mismatch")
        if not _hex64(item.get("final_checkpoint_sha256")):
            raise ValueError("run receipt final checkpoint hash malformed")
    by_seed = {}
    for item in receipts:
        by_seed.setdefault(int(item["seed"]), []).append(item)
    for seed, paired in by_seed.items():
        if len({item["foundation_checkpoint_sha256"] for item in paired}) != 1:
            raise ValueError(f"seed {seed} treatment arms do not share one foundation checkpoint")
        if len({item["foundation_receipt_sha256"] for item in paired}) != 1:
            raise ValueError(f"seed {seed} treatment arms do not share one foundation receipt")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_hashes(project_root: Path, tools_dir: Path) -> dict:
    files = {
        "candidate_runner": tools_dir / "run_clinc_cil_v8_full.py",
        "r2_runner": tools_dir / "run_clinc_cil_v8_r2.py",
        "common": tools_dir / "clinc_cil_common_v8.py",
        "lifecycle": tools_dir / "clinc_lifecycle_v8.py",
        "controlled_lifecycle": tools_dir / "clinc_controlled_routing_v8.py",
        "controlled_probe": tools_dir / "run_clinc_controlled_probe_v8.py",
        "tail_runtime": tools_dir / "clinc_tail_runtime_v8.py",
    }
    for key in ("lm", "experts", "accounting", "native"):
        files[f"public_{key}"] = project_root / "src" / "nested_sleep" / f"{key}.py"
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source-lock files missing: {missing}")
    return {key: sha256_file(path) for key, path in sorted(files.items())}


def _safe_child_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("checkpoint path escapes run directory")
    return path


def load_and_verify_foundation_grid(foundation_root: Path, *, master_sha: str,
                                    manifest_sha: str, source_lock_sha: str,
                                    model_revision: str, model_assets_sha256: Mapping) -> tuple[dict, list[dict]]:
    receipts = {}
    verified = []
    for seed in SEEDS:
        seed_dir = foundation_root / f"seed{seed}"
        receipt_path = seed_dir / "foundation-receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"missing foundation receipt: {receipt_path}")
        receipt = _load_json(receipt_path)
        receipts[seed] = receipt
    validate_foundation_receipts(receipts, master_sha=master_sha,
        manifest_sha=manifest_sha, source_lock_sha=source_lock_sha,
        model_revision=model_revision, model_assets_sha256=model_assets_sha256)
    for seed in SEEDS:
        seed_dir = foundation_root / f"seed{seed}"
        receipt_path = seed_dir / "foundation-receipt.json"
        receipt = receipts[seed]
        checkpoint_path = _safe_child_path(seed_dir, receipt.get("final_checkpoint", ""))
        checkpoint_sha = receipt["final_checkpoint_sha256"]
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_sha:
            raise ValueError(f"seed {seed} foundation checkpoint bytes mismatch")
        verified.append({"seed": seed,
            "receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
            "work_counted_once": True, "work": receipt.get("work")})
    return receipts, verified


def seal_main_grid(args) -> dict:
    """Aggregate exactly six completed run receipts and verify checkpoint bytes.

    This routine never reads test metadata, rows, labels, or invokes a test reader.
    """
    run_root = Path(args.run_root).resolve()
    config_path = Path(args.master_config).resolve()
    pack_path = Path(args.training_pack).resolve()
    project_root = Path(args.project_root).resolve()
    if not args.foundation_dir:
        raise ValueError("seal mode requires --foundation-dir with all three task-0 foundations")
    foundation_root = Path(args.foundation_dir).resolve()
    output_path = Path(args.output_path).resolve()
    cfg = validate_master_config(_load_json(config_path))
    master_sha = sha256_file(config_path)
    manifest_sha = sha256_file(pack_path)
    if manifest_sha != cfg["training_manifest_sha256"]:
        raise ValueError("training pack bytes do not match the frozen master manifest SHA")
    source_hashes = _source_hashes(project_root, Path(__file__).resolve().parent)
    source_lock_sha = canonical_sha(source_hashes)
    foundations, verified_foundations = load_and_verify_foundation_grid(foundation_root,
        master_sha=master_sha, manifest_sha=manifest_sha, source_lock_sha=source_lock_sha,
        model_revision=cfg["model_revision"], model_assets_sha256=cfg["model_assets_sha256"])
    rows = []
    verified_files = []
    for seed in SEEDS:
        for arm in ARMS:
            run_dir = run_root / f"seed{seed}" / arm
            receipt_path = run_dir / "run-receipt.json"
            if not receipt_path.is_file():
                raise FileNotFoundError(f"missing terminal run receipt: {receipt_path}")
            receipt = _load_json(receipt_path)
            if receipt.get("seed") != seed or receipt.get("arm") != arm:
                raise ValueError("receipt path and seed/arm metadata disagree")
            if receipt.get("child_config_sha256") != _child_config_sha(master_sha, "continuation", seed, arm):
                raise ValueError("run receipt child config hash mismatch")
            foundation = foundations[seed]
            foundation_receipt_sha = sha256_file(foundation_root / f"seed{seed}" / "foundation-receipt.json")
            if receipt.get("foundation_checkpoint_sha256") != foundation["final_checkpoint_sha256"]:
                raise ValueError(f"run receipt does not bind to seed {seed}'s verified common foundation")
            if receipt.get("foundation_receipt_sha256") != foundation_receipt_sha:
                raise ValueError(f"run receipt does not bind to seed {seed}'s actual foundation receipt bytes")
            cp_path = _safe_child_path(run_dir, receipt.get("final_checkpoint", ""))
            if not cp_path.is_file() or sha256_file(cp_path) != receipt.get("final_checkpoint_sha256"):
                raise ValueError(f"final checkpoint bytes mismatch for seed {seed} {arm}")
            receipt_sha = sha256_file(receipt_path)
            row = dict(receipt)
            row["run_receipt_path"] = str(receipt_path.relative_to(run_root))
            row["run_receipt_sha256"] = receipt_sha
            rows.append(row)
            verified_files.append({"seed": seed, "arm": arm,
                                   "checkpoint_path": str(cp_path.relative_to(run_root)),
                                   "checkpoint_sha256": receipt["final_checkpoint_sha256"],
                                   "run_receipt_path": str(receipt_path.relative_to(run_root)),
                                   "run_receipt_sha256": receipt_sha})
    validate_main_grid_receipts(rows, master_sha=master_sha,
                                manifest_sha=manifest_sha, source_lock_sha=source_lock_sha,
                                inner_cycles_per_event=cfg["inner_cycles_per_event"],
                                objective_version=cfg["objective_version"])
    seal = {"schema": SEAL_SCHEMA, "complete": True, "test_data_included": False,
            "main_grid": [{"seed": s, "arm": a} for s in SEEDS for a in ARMS],
            "master_config_path": str(config_path), "master_config_sha256": master_sha,
            "training_manifest_sha256": manifest_sha,
            "source_hashes": source_hashes, "source_lock_sha256": source_lock_sha,
            "verified_foundations": verified_foundations,
            "shared_foundation_compute_counted_once": [
                {"seed": item["seed"], "work": item["work"], "foundation_checkpoint_sha256": item["checkpoint_sha256"]}
                for item in verified_foundations],
            "verified_run_files": verified_files, "runs": rows,
            "test_release_ready": True,
            "test_release_note": "Artifact integrity and exact main grid verified; no test rows were read."}
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite seal: {output_path}")
    _atomic_json(output_path, seal, refuse_existing=True)
    return seal


def _read_inputs(pack_path: Path, cfg: dict):
    import clinc_cil_common_v8 as common
    if pack_path.exists() is False:
        raise FileNotFoundError(f"training-only pack missing: {pack_path}")
    manifest = _load_json(pack_path)
    view = common.make_training_view(manifest)
    train_rows = list(common.iter_role(view, "train", pack_path.parent))
    dev_rows = list(common.iter_role(view, "development", pack_path.parent))
    partition = common.validate_role_partition(train_rows, dev_rows)
    if len(train_rows) != 15000 or len(dev_rows) != 3000:
        raise ValueError("locked in-scope pack must contain 15,000 train and 3,000 development rows")
    if cfg["expected_normalized_train_dev_duplicates"] < 0:
        raise ValueError("invalid normalized duplicate count")
    r2 = _load_module(Path(__file__).resolve().parent / "run_clinc_cil_v8_r2.py", "_clinc_r2_for_full")
    train_norm = {r2.normalize_text(row["text"]) for row in train_rows}
    duplicates = [row for row in dev_rows if r2.normalize_text(row["text"]) in train_norm]
    if len(duplicates) != cfg["expected_normalized_train_dev_duplicates"]:
        raise ValueError("normalized train/development duplicate count differs from master config")
    clean_dev = [row for row in dev_rows if r2.normalize_text(row["text"]) not in train_norm]
    return view, train_rows, dev_rows, clean_dev, partition, len(duplicates), r2


def _runtime_cfg(master: dict, seed: int, *, max_mode: str, arm: str | None) -> dict:
    cfg = dict(master)
    cfg.update({"seed": int(seed), "arms": [arm] if arm else ["HOPE"],
                "periods": tuple(master["periods"]), "device": master.get("device", "cuda:0"),
                "temperature": float(master["temperature"]), "eval_examples_per_class": 2,
                "max_new_tokens": int(master["max_new_tokens"]),
                "max_sequence_length": int(master["max_sequence_length"]),
                "eval_full_development": False, "full_development_scope": "seen_classes",
                "endpoint_full_clean": True, "task0_acquisition_gate": float(master["task0_clean_accuracy_gate"]),
                "max_wall_seconds": int(master["budgets"][max_mode]["max_wall_seconds"]),
                "max_forward_calls": int(master["budgets"][max_mode]["max_forward_calls"]),
                "max_update_steps": int(master["budgets"][max_mode]["max_update_steps"]),
                "checkpoint_every_steps": 64, "initialization_seed": int(master["initialization_seed"]),
                "derived_steps_per_epoch": 188})
    return cfg


def _seed_everything(torch, seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _restore_rng(state: Mapping, torch):
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu().clone())
    if state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([x.cpu().clone() for x in state["torch_cuda"]])


def _rng_state(torch) -> dict:
    import numpy as np
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state().clone(),
            "torch_cuda": [x.cpu().clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None}


def _active_slots(context) -> list[int]:
    return [sum(bool(slot.active) for slot in bank.slots) for bank in context.banks()]


def _write_protocol(run_dir: Path, data: dict):
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError(f"refusing to overwrite protocol: {protocol_path}")
    _atomic_json(protocol_path, data)


def _save_safe_checkpoint(run_dir: Path, *, torch, context, wake, mode: str, seed: int,
                          arm: str | None, task_id: int, next_step: int, task_count: int,
                          schedule_sha: str, replay_sha: str, master_sha: str, child_sha: str,
                          manifest_sha: str, source_lock_sha: str, foundation_sha: str | None,
                          work_cumulative: dict, physical_sleep_adamw_steps: int,
                          physical_wake_updates_run: int, sleep_event_index: int,
                          sleep_exposure: dict,
                          task_metrics: list[dict], init_sha: str) -> tuple[Path, str]:
    cursor = checkpoint_cursor(mode=mode, task_id=task_id, next_step=next_step,
                               task_count=task_count, schedule_sha256=schedule_sha,
                               replay_ids_sha256=replay_sha)
    state = {"schema": CHECKPOINT_SCHEMA, "mode": mode, "seed": seed, "arm": arm,
             "master_config_sha256": master_sha, "child_config_sha256": child_sha,
             "training_manifest_sha256": manifest_sha, "source_lock_sha256": source_lock_sha,
             "foundation_checkpoint_sha256": foundation_sha,
             "context": context.state_dict(), "wake": wake.snapshot(), "rng": _rng_state(torch),
             "cursor": cursor, "work_cumulative": deepcopy(work_cumulative),
             "physical_sleep_adamw_steps": int(physical_sleep_adamw_steps),
             "physical_wake_updates_run": int(physical_wake_updates_run),
             "sleep_event_index": int(sleep_event_index),
             "sleep_exposure": deepcopy(sleep_exposure),
             "task_metrics": deepcopy(task_metrics), "initial_mutable_sha256": init_sha,
             "checkpoint_boundary": "after_all_due_events", "saved_unix_time": time.time()}
    path = run_dir / "checkpoints" / f"task{task_id:02d}-next{next_step:04d}-safe.pt"
    digest = _atomic_torch(path, state, torch)
    return path, digest


def _load_safe_checkpoint(path: Path, *, torch, expected: dict):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload, master_config_sha256=expected["master_config_sha256"],
        child_config_sha256=expected["child_config_sha256"],
        manifest_sha256=expected["training_manifest_sha256"], source_lock_sha256=expected["source_lock_sha256"],
        seed=expected["seed"], arm=expected["arm"])
    if payload.get("mode") != expected["mode"]:
        raise ValueError("checkpoint mode mismatch")
    if payload.get("foundation_checkpoint_sha256") != expected.get("foundation_checkpoint_sha256"):
        raise ValueError("checkpoint foundation binding mismatch")
    return payload


def _safe_point_resume_check(payload: dict, *, train_rows: Sequence[Mapping], view,
                             expected_arm: str | None, cfg: dict) -> None:
    cursor = payload["cursor"]
    task_id, next_step = int(cursor["task_id"]), int(cursor["next_step"])
    if expected_arm != payload.get("arm"):
        raise ValueError("resume arm mismatch")
    ids = task_schedule_ids(train_rows, view.class_orders[str(payload["seed"])],
                            payload["seed"], task_id, cfg["examples_per_task"])
    if task_schedule_hash(ids) != cursor["schedule_sha256"]:
        raise ValueError("resume schedule hash mismatch")
    pools = class_balanced_replay_pools(train_rows, view.class_orders[str(payload["seed"])], task_id)
    if pool_ids_hash(pools) != cursor["replay_ids_sha256"]:
        raise ValueError("resume replay-pool reconstruction mismatch")
    if next_step > math.ceil(len(ids) / cfg["batch_size"]) + 1:
        raise ValueError("resume next-step cursor outside task schedule")


def _combine_work(previous: Mapping | None, current: Mapping) -> dict:
    if not previous:
        return deepcopy(dict(current))
    out = deepcopy(dict(previous))
    for key, value in current.items():
        old = out.get(key)
        if isinstance(value, (int, float)) and isinstance(old, (int, float)):
            out[key] = old + value
        elif isinstance(value, dict) and isinstance(old, dict):
            out[key] = _combine_work(old, value)
        else:
            out[key] = value
    return out


def _verify_asset_hashes(model_path: Path, assets: Mapping[str, str]):
    for relative, expected in assets.items():
        path = (model_path / relative).resolve()
        if model_path.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(f"model asset missing/outside root: {relative}")
        if sha256_file(path) != expected:
            raise ValueError(f"model asset hash mismatch: {relative}")


def run_training(args) -> dict:
    """Run one task-0 foundation or one seed/arm continuation."""
    cfg_path = Path(args.master_config).resolve()
    pack_path = Path(args.training_pack).resolve()
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    run_dir = Path(args.output_dir).resolve()
    if args.mode not in ("foundation", "continuation"):
        raise ValueError("training mode must be foundation or continuation")
    seed = int(args.seed)
    if seed not in SEEDS:
        raise ValueError("seed must be one of 17, 29, 43")
    arm = None if args.mode == "foundation" else args.arm
    if args.mode == "foundation" and arm is not None:
        raise ValueError("foundation mode has no treatment arm")
    if args.mode == "continuation" and arm not in ARMS:
        raise ValueError(f"continuation arm must be one of {ARMS}")
    master = validate_master_config(_load_json(cfg_path))
    master_sha = sha256_file(cfg_path)
    child_sha = _child_config_sha(master_sha, args.mode, seed, arm)
    manifest_sha = sha256_file(pack_path)
    if manifest_sha != master["training_manifest_sha256"]:
        raise ValueError("training pack SHA differs from frozen master config")
    if not project_root.is_dir() or not model_path.is_dir():
        raise FileNotFoundError("public project or local model directory missing")
    _verify_asset_hashes(model_path, master["model_assets_sha256"])

    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    source_hashes = _source_hashes(project_root, tools_dir)
    source_lock_sha = canonical_sha(source_hashes)
    view, train_rows, dev_rows, clean_dev, partition, dup_count, r2 = _read_inputs(pack_path, master)
    if len(view.classes) != 150 or set(view.class_orders) != {str(s) for s in SEEDS}:
        raise ValueError("training view does not include all locked class orders")
    order = view.class_orders[str(seed)]
    max_mode = "foundation" if args.mode == "foundation" else "continuation"
    cfg = _runtime_cfg(master, seed, max_mode=max_mode, arm=arm)
    cfg["training_manifest_sha256"] = manifest_sha
    cfg["expected_normalized_train_dev_duplicates"] = dup_count
    cfg["source_revision"] = master["source_revision"]
    cfg["model_revision"] = master["model_revision"]
    cfg["model_assets_sha256"] = master["model_assets_sha256"]
    cfg["rank"], cfg["capacity"], cfg["levels"] = master["rank"], master["capacity"], master["levels"]
    cfg["base_lr"], cfg["fast_lr"] = master["base_lr"], master["fast_lr"]
    cfg["sleep_lr"], cfg["sleep_weight_decay"], cfg["sleep_clip_norm"] = master["sleep_lr"], master["sleep_weight_decay"], master["sleep_clip_norm"]
    cfg["gkd_calls_per_event"], cfg["lti_calls_per_event"] = master["gkd_calls_per_event"], master["lti_calls_per_event"]
    cfg["inner_cycles_per_event"] = master["inner_cycles_per_event"]
    cfg["lti_edit_distance_z0"] = master["lti_edit_distance_z0"]
    cfg["lti_semantic_weight"], cfg["lti_edit_weight"] = master["lti_semantic_weight"], master["lti_edit_weight"]
    cfg["sleep_prompt_policy"] = "seen_replay"
    cfg["examples_per_task"], cfg["batch_size"], cfg["max_steps_per_task"] = 1500, 8, 188

    foundation_root = Path(args.foundation_dir).resolve() if args.foundation_dir else None
    foundation_receipts = None
    foundation_cp_path = None
    foundation_cp_sha = None
    foundation_receipt_sha = None
    foundation_task_metrics = []
    if args.mode == "continuation":
        if foundation_root is None:
            raise ValueError("continuation requires --foundation-dir containing all three qualified task-0 receipts")
        foundation_receipts, _verified_foundations = load_and_verify_foundation_grid(
            foundation_root, master_sha=master_sha, manifest_sha=manifest_sha,
            source_lock_sha=source_lock_sha, model_revision=master["model_revision"],
            model_assets_sha256=master["model_assets_sha256"])
        receipt = foundation_receipts[seed]
        foundation_cp_path = _safe_child_path(foundation_root / f"seed{seed}", receipt["final_checkpoint"])
        if sha256_file(foundation_cp_path) != receipt["final_checkpoint_sha256"]:
            raise ValueError("foundation checkpoint actual bytes mismatch")
        foundation_cp_sha = receipt["final_checkpoint_sha256"]
        foundation_receipt_sha = sha256_file(foundation_root / f"seed{seed}" / "foundation-receipt.json")
        foundation_task_metrics = receipt.get("task_metrics", [])

    # Successful terminal receipts are immutable. Refuse either resume or a
    # second invocation before loading the model or doing any work.
    if run_dir.exists():
        refuse_terminal_run(run_dir, args.mode)
        if not args.resume_checkpoint:
            raise FileExistsError(f"refusing to reuse existing run directory without --resume-checkpoint: {run_dir}")
    elif args.resume_checkpoint:
        raise FileNotFoundError("resume run directory does not exist")

    resume_path = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    if resume_path is None:
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        if not run_dir.is_dir():
            raise ValueError("resume requires an existing incomplete run directory")
        refuse_terminal_run(run_dir, args.mode)
        protocol_path = run_dir / "protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError("resume run lacks protocol.json")
        protocol = _load_json(protocol_path)
        if protocol.get("master_config_sha256") != master_sha or protocol.get("child_config_sha256") != child_sha:
            raise ValueError("existing run protocol binding differs from requested resume")
        if protocol.get("source_lock_sha256") != source_lock_sha or protocol.get("training_manifest_sha256") != manifest_sha:
            raise ValueError("existing run source/data binding differs")
        if protocol.get("foundation_checkpoint_sha256") != foundation_cp_sha or protocol.get("foundation_receipt_sha256") != foundation_receipt_sha:
            raise ValueError("existing run foundation receipt/checkpoint binding differs")

    data_meta = {"train_rows": len(train_rows), "development_rows": len(dev_rows),
                 "clean_development_rows": len(clean_dev), "normalized_overlap_rows_excluded": dup_count,
                 "partition": partition, "test_data_included": False}
    init_seed = int(master["initialization_seed"])
    _seed_everything(__import__("torch"), init_seed)
    run_started = time.monotonic()
    phase = "model_load"
    budget = None
    model = context = wake = tx = None
    active_adam_attempts = active_adam_completed = 0
    physical_steps_start = 0
    latest_safe_path = None
    latest_safe_sha = None
    original_adam_step = None
    try:
        torch, tokenizer, model, immutable_versions = r2._load_common(project_root, model_path, cfg["device"], cfg)
        if cfg["device"].startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        frozen_versions = immutable_versions(model)
        model.requires_grad_(False)
        from clinc_tail_runtime_v8 import clone_mutable_context, capture, tail_logits
        from clinc_lifecycle_v8 import WakeController
        from clinc_controlled_routing_v8 import ControlledSleepTransaction, ControlledWakeController
        probe = _load_module(tools_dir / "run_clinc_controlled_probe_v8.py", "_clinc_controlled_probe_for_full")

        budget = r2.Budget(cfg)
        budget.started = run_started
        original_logits = model.logits
        def counted_logits(tokens, mask=None):
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_logits(tokens, mask)
        model.logits = counted_logits

        import clinc_tail_runtime_v8 as tail_runtime
        original_tail = tail_runtime.tail_logits
        def counted_tail(tail_model, tail_context, cache):
            active = torch.is_grad_enabled() and any(p.requires_grad for p in tail_context.parameters())
            budget.tail(cache, active)
            return original_tail(tail_model, tail_context, cache)
        tail_runtime.tail_logits = counted_tail

        context = clone_mutable_context(model)
        init_sha = r2._mutable_context_sha(context)
        controller_cls = ControlledWakeController if arm == "HOPE_SLEEP_FIXEDMIX" else WakeController
        wake = controller_cls(context, periods=cfg["periods"], base_lr=cfg["base_lr"], fast_lr=cfg["fast_lr"])
        wake.prepare_wake()
        task_metrics = list(foundation_task_metrics)
        physical_sleep_adamw_steps = 0
        physical_wake_updates_run = 0
        sleep_event_index = 0
        sleep_exposure = {"by_task": {}, "by_role_class": {},
                          "unique_ids_by_role": {"old": [], "current": []}}
        cumulative_work = {}
        cursor = {"task_id": 0 if args.mode == "foundation" else 1, "next_step": 1,
                  "tasks_completed": 0 if args.mode == "foundation" else 1}
        if resume_path is None:
            if args.mode == "continuation":
                foundation_state = torch.load(foundation_cp_path, map_location="cpu", weights_only=False)
                expected_foundation = {"master_config_sha256": master_sha,
                    "child_config_sha256": _child_config_sha(master_sha, "foundation", seed, None),
                    "training_manifest_sha256": manifest_sha, "source_lock_sha256": source_lock_sha,
                    "seed": seed, "arm": None, "mode": "foundation",
                    "foundation_checkpoint_sha256": None}
                validate_checkpoint_payload(foundation_state, master_config_sha256=expected_foundation["master_config_sha256"],
                    child_config_sha256=expected_foundation["child_config_sha256"], manifest_sha256=manifest_sha,
                    source_lock_sha256=source_lock_sha, seed=seed, arm=None)
                if foundation_state.get("mode") != "foundation":
                    raise ValueError("foundation checkpoint mode mismatch")
                context.load_state_dict(foundation_state["context"])
                wake.restore(foundation_state["wake"])
                context = wake.context
                _restore_rng(foundation_state["rng"], torch)
                init_sha = r2._mutable_context_sha(context)
                if init_sha != foundation_receipts[seed].get("foundation_context_sha256"):
                    raise ValueError("restored task-0 foundation state differs from its terminal receipt")
                task_metrics = deepcopy(foundation_state["task_metrics"])
                physical_sleep_adamw_steps = int(foundation_state["physical_sleep_adamw_steps"])
                # Task 0 is shared and counted once in the experiment seal.
                cumulative_work = {}
                # Foundation cursor is task 0 complete; continuation begins at task 1.
            else:
                cursor = {"task_id": 0, "next_step": 1, "tasks_completed": 0}
        else:
            payload = _load_safe_checkpoint(resume_path, torch=torch, expected={
                "master_config_sha256": master_sha, "child_config_sha256": child_sha,
                "training_manifest_sha256": manifest_sha, "source_lock_sha256": source_lock_sha,
                "seed": seed, "arm": arm, "mode": args.mode,
                "foundation_checkpoint_sha256": foundation_cp_sha})
            latest_safe_path = resume_path
            latest_safe_sha = sha256_file(resume_path)
            cursor = payload["cursor"]
            _safe_point_resume_check(payload, train_rows=train_rows, view=view, expected_arm=arm, cfg=cfg)
            context.load_state_dict(payload["context"])
            wake.restore(payload["wake"])
            context = wake.context
            _restore_rng(payload["rng"], torch)
            init_sha = payload["initial_mutable_sha256"]
            task_metrics = deepcopy(payload["task_metrics"])
            physical_sleep_adamw_steps = int(payload["physical_sleep_adamw_steps"])
            physical_wake_updates_run = int(payload["physical_wake_updates_run"])
            sleep_event_index = int(payload["sleep_event_index"])
            sleep_exposure = deepcopy(payload["sleep_exposure"])
            cumulative_work = deepcopy(payload["work_cumulative"])
            if (run_dir / "metrics.jsonl").is_file():
                metrics_file = (run_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n", buffering=1)
            else:
                metrics_file = (run_dir / "metrics.jsonl").open("w", encoding="utf-8", newline="\n", buffering=1)
        if resume_path is None:
            metrics_file = (run_dir / "metrics.jsonl").open("w", encoding="utf-8", newline="\n", buffering=1)
            if args.mode == "foundation":
                _write_protocol(run_dir, {"schema": "clinc-cil-v8-full-run-protocol", "mode": args.mode,
                    "seed": seed, "arm": None, "master_config_sha256": master_sha,
                    "child_config_sha256": child_sha, "training_manifest_sha256": manifest_sha,
                    "source_hashes": source_hashes, "source_lock_sha256": source_lock_sha,
                    "data": data_meta, "test_data_included": False,
                    "initialization_seed": init_seed, "task0_gate": master["task0_clean_accuracy_gate"],
                    "method_note": "Task 0 is a common no-Sleep HOPE foundation; it is shared state, not treatment exposure."})
            else:
                _write_protocol(run_dir, {"schema": "clinc-cil-v8-full-run-protocol", "mode": args.mode,
                    "seed": seed, "arm": arm, "master_config_sha256": master_sha,
                    "child_config_sha256": child_sha, "training_manifest_sha256": manifest_sha,
                    "source_hashes": source_hashes, "source_lock_sha256": source_lock_sha,
                    "foundation_checkpoint_sha256": foundation_cp_sha,
                    "foundation_receipt_sha256": foundation_receipt_sha,
                    "foundation_task0_metric": foundation_receipts[seed]["task0_clean_current_accuracy"],
                    "data": data_meta, "test_data_included": False,
                    "initialization_seed": init_seed,
                    "method_note": "Fixed 0.5 routing is an explicit engineering ablation; task 0 is common HOPE warmup."})
        else:
            metrics_file.write(json.dumps({"event": "manual_resume", "checkpoint_path": str(resume_path),
                "checkpoint_sha256": sha256_file(resume_path), "cursor": cursor,
                "restored_wake_ticks": wake.ticks, "test_data_included": False}, sort_keys=True) + "\n")

        # Count actual AdamW step attempts and successful returns, including
        # failures inside a controlled transaction.
        original_adam_step = torch.optim.AdamW.step
        def counted_adam_step(optimizer, *step_args, **step_kwargs):
            nonlocal active_adam_attempts, active_adam_completed
            active_adam_attempts += 1
            result = original_adam_step(optimizer, *step_args, **step_kwargs)
            active_adam_completed += 1
            return result
        torch.optim.AdamW.step = counted_adam_step

        task_first = int(cursor["task_id"])
        first_resume_step = int(cursor["next_step"])
        if first_resume_step == 189:
            task_first += 1
            first_resume_step = 1
            if args.mode == "foundation" and task_first == 1:
                cursor = {"task_id": 1, "next_step": 1, "tasks_completed": 1}
        if args.mode == "foundation" and int(cursor["tasks_completed"]) == 1:
            task_first = 1
        for task_id in mode_task_range(args.mode, task_first):
            phase = f"task_{task_id}_setup"
            task_ids = set(view.class_orders[str(seed)][task_id * 15:(task_id + 1) * 15])
            schedule_ids = task_schedule_ids(train_rows, order, seed, task_id, cfg["examples_per_task"])
            schedule_sha = task_schedule_hash(schedule_ids)
            row_by_id = {str(row["id"]): row for row in train_rows if int(row["intent_id"]) in task_ids}
            schedule_rows = [row_by_id[row_id] for row_id in schedule_ids]
            pools = class_balanced_replay_pools(train_rows, order, task_id)
            replay_sha = pool_ids_hash(pools)
            sleep_pool = pools
            first_step = first_resume_step if task_id == task_first else 1
            task_wake_steps = 0
            task_sleep_events = 0
            task_ce_sum = 0.0
            task_ce_count = 0
            metrics_file.write(json.dumps({"event": "task_start", "seed": seed, "arm": arm,
                "task_id": task_id, "schedule_sha256": schedule_sha, "replay_pool_ids_sha256": replay_sha,
                "start_step": first_step, "train_rows": len(schedule_rows),
                "active_slots_by_level": _active_slots(context), "test_data_included": False}, sort_keys=True) + "\n")

            for step, start in enumerate(range(0, len(schedule_rows), cfg["batch_size"]), 1):
                if step < first_step:
                    continue
                phase = f"task_{task_id}_wake_step_{step}"
                budget.check()
                wake.prepare_wake()
                rows = schedule_rows[start:start + cfg["batch_size"]]
                _data, ids, mask, labels = r2._batch(rows, tokenizer, view, cfg, torch, cfg["device"])
                cache = capture(model, ids, mask, device=cfg["device"])
                for parameter in context.parameters():
                    parameter.grad = None
                logits = tail_runtime.tail_logits(model, context, cache)
                targets = labels[:, 1:].reshape(-1)
                valid = targets.ne(-100)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).float()[valid], targets[valid])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("wake CE became nonfinite")
                loss.backward()
                budget.backward(mask * labels.ne(-100).long())
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in context.parameters() if p.requires_grad], float(cfg["sleep_clip_norm"]),
                    error_if_nonfinite=True)
                grads = {name: parameter.grad.detach().clone() for name, parameter in context.named_parameters()
                         if parameter.grad is not None}
                budget.update()
                wake.record_tick(grads, token_count=int(labels.ne(-100).sum().item()))
                for parameter in context.parameters():
                    parameter.grad = None
                task_wake_steps += 1
                physical_wake_updates_run += 1
                task_ce_sum += float(loss.detach().float().cpu())
                task_ce_count += 1
                due = sorted(wake.proposals())
                metrics_file.write(json.dumps({"event": "wake_update", "seed": seed, "arm": arm,
                    "task_id": task_id, "step": step, "global_wake_tick": wake.ticks,
                    "ce": float(loss.detach().float().cpu()), "grad_norm_before_clip": float(grad_norm.detach().float().cpu()),
                    "due_levels": due, "active_slots_by_level": _active_slots(context)}, sort_keys=True) + "\n")

                for level in due:
                    phase = f"task_{task_id}_step_{step}_due_level_{level}"
                    proposal = wake.proposals()[level]
                    if arm == "HOPE_SLEEP_FIXEDMIX" and level < len(PERIODS) - 1:
                        tx = ControlledSleepTransaction(context, level, proposal, lr=cfg["sleep_lr"],
                            weight_decay=cfg["sleep_weight_decay"], clip_norm=cfg["sleep_clip_norm"],
                            seed=seed * 1_000_000 + task_id * 10_000 + wake.ticks * 10 + level)
                        last_event = {"seed": seed, "task_id": task_id, "wake_tick": wake.ticks,
                                      "sender_level": level, "receiver_level": tx.receiver,
                                      "receiver_slot": tx.slot, "phase": "controlled_sleep"}
                        selected = select_full_sleep_prompts(sleep_pool, event=sleep_event_index,
                            task_id=task_id, gkd_calls=cfg["gkd_calls_per_event"],
                            lti_calls=cfg["lti_calls_per_event"])
                        outputs = run_full_controlled_sleep_event(model, tx, selected, tokenizer,
                            view, cfg, torch, cfg["device"], budget, r2, probe,
                            tail_runtime, last_event)
                        exposure = summarize_sleep_exposure(sleep_pool, selected)
                        task_exposure = sleep_exposure["by_task"].setdefault(str(task_id),
                            {role: {"calls": 0, "classes_covered": set(), "unique_ids": set(),
                                    "class_count": item["class_count"]}
                             for role, item in exposure.items()})
                        for role, item in exposure.items():
                            task_exposure[role]["calls"] += item["calls"]
                            task_exposure[role]["classes_covered"].update(item["classes_covered"])
                            task_exposure[role]["unique_ids"].update(item["unique_ids"])
                            for row in sleep_pool[role]:
                                row_key = f"{role}:{int(row['intent_id'])}"
                                sleep_exposure["by_role_class"][row_key] = sleep_exposure["by_role_class"].get(row_key, 0)
                        for role, item in exposure.items():
                            task_exposure[role]["selected_classes"] = sorted(task_exposure[role]["classes_covered"])
                            task_exposure[role]["selected_class_count"] = len(task_exposure[role]["classes_covered"])
                            task_exposure[role]["unique_id_count"] = len(task_exposure[role]["unique_ids"])
                            task_exposure[role]["class_coverage_fraction"] = (
                                len(task_exposure[role]["classes_covered"]) / item["class_count"] if item["class_count"] else None)
                        for _kind, row, role in selected:
                            row_key = f"{role}:{int(row['intent_id'])}"
                            sleep_exposure["by_role_class"][row_key] += 1
                        tx_counters = dict(tx.counters)
                        physical_sleep_adamw_steps += tx_counters["adamw_steps"]
                        new_context = tx.commit()
                        wake.bind_context(new_context)
                        wake.consume(level)
                        context = new_context
                        task_sleep_events += 1
                        sleep_event_index += 1
                        budget.sleep_events += 1
                        metrics_file.write(json.dumps({"event": "sleep_commit", "seed": seed, "arm": arm,
                            "task_id": task_id, "step": step, "global_wake_tick": wake.ticks,
                            "sender_level": level, "receiver_level": tx.receiver, "receiver_slot": tx.slot,
                            "sleep_updates": tx_counters["adamw_steps"], "sleep_calls": outputs,
                            "sleep_event_index": sleep_event_index - 1, "prompt_exposure": exposure,
                            "controlled_mix_fraction": 0.5, "teacher_label_use": False,
                            "last_event_audit": last_event, "active_slots_by_level": _active_slots(context)}, sort_keys=True) + "\n")
                        tx = None
                    else:
                        wake.apply(level, proposal)
                        context = wake.context

                # A checkpoint is safe only after every due transaction for the
                # just-completed wake step has committed/applied.
                is_task_end = step == len(range(0, len(schedule_rows), cfg["batch_size"]))
                if step % 64 == 0 and not is_task_end:
                    phase = f"task_{task_id}_safe_checkpoint"
                    budget.check()
                    next_task = task_id + 1 if is_task_end else task_id
                    next_step = 1 if is_task_end else step + 1
                    completed_count = task_id + 1 if is_task_end else task_id
                    cp_path, cp_sha = _save_safe_checkpoint(run_dir, torch=torch, context=context, wake=wake,
                        mode=args.mode, seed=seed, arm=arm, task_id=task_id, next_step=next_step,
                        task_count=completed_count, schedule_sha=schedule_sha, replay_sha=replay_sha,
                        master_sha=master_sha, child_sha=child_sha, manifest_sha=manifest_sha,
                        source_lock_sha=source_lock_sha, foundation_sha=foundation_cp_sha,
                        work_cumulative=_combine_work(cumulative_work, budget.as_dict()),
                        physical_sleep_adamw_steps=physical_sleep_adamw_steps,
                        physical_wake_updates_run=physical_wake_updates_run,
                        sleep_event_index=sleep_event_index, sleep_exposure=sleep_exposure,
                        task_metrics=task_metrics, init_sha=init_sha)
                    latest_safe_path, latest_safe_sha = cp_path, cp_sha
                    metrics_file.write(json.dumps({"event": "safe_checkpoint", "task_id": task_id,
                        "step": step, "next_task_id": next_task, "next_step": next_step,
                        "path": str(cp_path.relative_to(run_dir)), "sha256": cp_sha,
                        "wake_ticks": wake.ticks, "pending_levels": dict(wake.clocks),
                        "work": budget.as_dict()}, sort_keys=True) + "\n")
                if step >= 188:
                    break

            if task_wake_steps != (188 - first_step + 1 if task_id == task_first else 188):
                raise RuntimeError(f"task {task_id} wake step count mismatch: {task_wake_steps}")
            qualified = None
            phase = f"task_{task_id}_clean_development_evaluation"
            budget.check()
            seen_dev = [row for row in clean_dev if int(row["intent_id"]) in set(order[:(task_id + 1) * 15])]
            result = r2._evaluate(model, context, seen_dev, order, task_id, tokenizer, view, cfg,
                torch, cfg["device"], budget, include_unseen=False)
            metric = result["metrics"]
            item = {"task_id": task_id, "result": result,
                    "evaluation_ids_sha256": canonical_sha([row["id"] for row in seen_dev]),
                    "train_dev_overlap_excluded": dup_count,
                    "old_count": metric["old"]["count"], "current_count": metric["current"]["count"],
                    "seen_count": metric["seen"]["count"],
                    "sleep_exposure": plain_task_exposure(sleep_exposure["by_task"].get(str(task_id), {}))}
            task_metrics.append(item)
            metrics_file.write(json.dumps({"event": "task_end_clean_development", "seed": seed,
                "arm": arm, **item, "test_data_included": False}, sort_keys=True) + "\n")
            if args.mode == "foundation" and task_id == 0:
                acc = metric["current"]["accuracy"]
                qualified = acc is not None and float(acc) >= master["task0_clean_accuracy_gate"]
                metrics_file.write(json.dumps({"event": "task0_foundation_gate", "threshold": .5,
                    "current_clean_accuracy": acc, "passed": qualified}, sort_keys=True) + "\n")
            # End-of-task CP is written after the clean evaluation, so resume
            # cannot skip or duplicate the fixed boundary metric.
            phase = f"task_{task_id}_end_safe_checkpoint"
            cp_path, cp_sha = _save_safe_checkpoint(run_dir, torch=torch, context=context, wake=wake,
                mode=args.mode, seed=seed, arm=arm, task_id=task_id, next_step=189,
                task_count=task_id + 1, schedule_sha=schedule_sha, replay_sha=replay_sha,
                master_sha=master_sha, child_sha=child_sha, manifest_sha=manifest_sha,
                source_lock_sha=source_lock_sha, foundation_sha=foundation_cp_sha,
                work_cumulative=_combine_work(cumulative_work, budget.as_dict()),
                physical_sleep_adamw_steps=physical_sleep_adamw_steps,
                physical_wake_updates_run=physical_wake_updates_run,
                sleep_event_index=sleep_event_index, sleep_exposure=sleep_exposure,
                task_metrics=task_metrics, init_sha=init_sha)
            latest_safe_path, latest_safe_sha = cp_path, cp_sha
            metrics_file.write(json.dumps({"event": "safe_checkpoint", "task_id": task_id,
                "step": 188, "next_task_id": task_id + 1, "next_step": 189,
                "path": str(cp_path.relative_to(run_dir)), "sha256": cp_sha,
                "wake_ticks": wake.ticks, "pending_levels": dict(wake.clocks),
                "work": budget.as_dict()}, sort_keys=True) + "\n")
            if args.mode == "foundation" and task_id == 0 and not qualified:
                raise RuntimeError("common task-0 clean-development foundation gate failed")
            cursor = {"task_id": task_id + 1, "next_step": 1, "tasks_completed": task_id + 1}
            metrics_file.flush()

        if args.mode == "foundation":
            assert_foundation_terminal_state(wake_ticks=wake.ticks, pending_clocks=wake.clocks,
                task_metrics=task_metrics, cursor=cursor)
            foundation_acc = task_metrics[0]["result"]["metrics"]["current"]["accuracy"]
            if foundation_acc is None or foundation_acc < 0.5:
                raise RuntimeError("task-0 foundation is not qualified")
        if immutable_versions(model) != frozen_versions:
            raise RuntimeError("frozen backbone changed during full-run candidate")
        final_task = 0 if args.mode == "foundation" else 9
        schedule_ids = task_schedule_ids(train_rows, order, seed, final_task, cfg["examples_per_task"])
        pools = class_balanced_replay_pools(train_rows, order, final_task)
        final_cursor = checkpoint_cursor(mode=args.mode, task_id=final_task, next_step=189,
            task_count=1 if args.mode == "foundation" else 10,
            schedule_sha256=task_schedule_hash(schedule_ids), replay_ids_sha256=pool_ids_hash(pools))
        # Every final task boundary has already written a post-event checkpoint.
        if latest_safe_path is None or latest_safe_sha != sha256_file(latest_safe_path):
            raise RuntimeError("terminal safe checkpoint missing or changed")
        budget_work = budget.as_dict()
        cumulative_work = _combine_work(cumulative_work, budget_work)
        task_metrics = [m for m in task_metrics if m.get("task_id") is not None]
        if args.mode == "foundation":
            foundation_receipt = {"schema": FOUNDATION_SCHEMA, "status": "complete", "seed": seed,
                "master_config_sha256": master_sha, "child_config_sha256": child_sha,
                "training_manifest_sha256": manifest_sha, "source_lock_sha256": source_lock_sha,
                "model_revision": master["model_revision"], "model_assets_sha256": master["model_assets_sha256"],
                "final_checkpoint": str(latest_safe_path.relative_to(run_dir)),
                "final_checkpoint_sha256": latest_safe_sha, "tasks_completed": 1,
                "task0_clean_current_accuracy": float(foundation_acc), "acquisition_qualified": True,
                "task_metrics": task_metrics, "initial_mutable_sha256": init_sha,
                "foundation_context_sha256": r2._mutable_context_sha(context),
                "wake_ticks": wake.ticks, "pending_clocks": dict(wake.clocks),
                "active_slots_by_level": _active_slots(context), "work": cumulative_work,
                "physical_sleep_adamw_steps": 0, "cuda_peak_allocated_bytes":
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "active_wall_seconds": time.monotonic() - run_started, "test_data_included": False,
                "foundation_semantics": "common task-0 HOPE warmup; no Sleep exposure"}
            _atomic_json(run_dir / "foundation-receipt.json", foundation_receipt, refuse_existing=True)
            metrics_file.write(json.dumps({"event": "foundation_complete", "receipt": foundation_receipt}, sort_keys=True) + "\n")
            metrics_file.flush()
            return foundation_receipt

        receipt = {"schema": RECEIPT_SCHEMA, "status": "complete", "seed": seed, "arm": arm,
            "tasks_completed": 10, "master_config_sha256": master_sha,
            "child_config_sha256": child_sha, "training_manifest_sha256": manifest_sha,
            "source_lock_sha256": source_lock_sha,
            "foundation_checkpoint_sha256": foundation_cp_sha,
            "foundation_receipt_sha256": foundation_receipt_sha,
            "acquisition_qualified": True, "common_foundation_task0": True,
            "task_metrics": task_metrics, "final_checkpoint": str(latest_safe_path.relative_to(run_dir)),
            "final_checkpoint_sha256": latest_safe_sha, "wake_ticks": wake.ticks,
            "pending_clocks": dict(wake.clocks), "physical_wake_updates": wake.ticks,
            "physical_sleep_adamw_steps": physical_sleep_adamw_steps,
            "inner_cycles_per_event": master["inner_cycles_per_event"],
            "objective_version": master["objective_version"],
            "sleep_event_index": sleep_event_index,
            "sleep_exposure": {"by_role_class": sleep_exposure["by_role_class"],
                                "by_task": {task: plain_task_exposure(value)
                                            for task, value in sleep_exposure["by_task"].items()}},
            "adamw_step_attempts_this_process": active_adam_attempts,
            "adamw_step_completions_this_process": active_adam_completed,
            "work": cumulative_work, "initial_mutable_sha256": init_sha,
            "active_slots_by_level": _active_slots(context),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "active_wall_seconds": time.monotonic() - run_started,
            "data_exposure": {"primary_train_rows": 13500, "sleep_text_replay": True,
                              "old_intent_gold_targets_in_sleep_loss": False,
                              "intent_metadata_used_for_balanced_pool_selection": True},
            "test_data_included": False,
            "claim_scope": "development-only local controlled-routing ablation; not paper-faithful HOPE Sleep"}
        if receipt["physical_sleep_adamw_steps"] > 141 * 3 * int(master["inner_cycles_per_event"]):
            raise RuntimeError("continuation Sleep optimizer count exceeded the full-period plan")
        _atomic_json(run_dir / "run-receipt.json", receipt, refuse_existing=True)
        metrics_file.write(json.dumps({"event": "run_complete", "receipt": receipt}, sort_keys=True) + "\n")
        metrics_file.flush()
        return receipt
    except BaseException as exc:
        forensic = None
        if model is not None and context is not None and wake is not None:
            try:
                partial = {"schema": "clinc-cil-v8-failure-forensics", "phase": phase,
                    "seed": seed, "arm": arm, "mode": args.mode,
                    "master_config_sha256": master_sha, "child_config_sha256": child_sha,
                    "training_manifest_sha256": manifest_sha, "source_lock_sha256": source_lock_sha,
                    "context": context.state_dict(), "wake": wake.snapshot(), "rng": _rng_state(__import__("torch")),
                    "cursor": locals().get("cursor"), "latest_safe_checkpoint": None if latest_safe_path is None else str(latest_safe_path),
                    "latest_safe_checkpoint_sha256": latest_safe_sha,
                    "physical_sleep_adamw_steps": locals().get("physical_sleep_adamw_steps", 0),
                    "active_tx_counters": dict(tx.counters) if tx is not None else None,
                    "active_tx_snapshot": tx.snapshot() if tx is not None else None,
                    "work": budget.as_dict() if budget is not None else None}
                fpath = run_dir / "failure-forensics.pt"
                fdigest = _atomic_torch(fpath, partial, __import__("torch"))
                forensic = {"path": str(fpath.relative_to(run_dir)), "sha256": fdigest}
            except BaseException as forensic_error:
                forensic = {"error_type": type(forensic_error).__name__, "error": str(forensic_error)}
        failure = {"schema": "clinc-cil-v8-full-failure", "status": "failed",
            "mode": args.mode, "seed": seed, "arm": arm, "phase": phase,
            "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_seconds": time.monotonic() - run_started,
            "latest_safe_checkpoint": None if latest_safe_path is None else str(latest_safe_path),
            "latest_safe_checkpoint_sha256": latest_safe_sha,
            "active_adamw_step_attempts": active_adam_attempts,
            "active_adamw_step_completions": active_adam_completed,
            "work": budget.as_dict() if budget is not None else None,
            "forensics": forensic, "test_data_included": False,
            "automatic_retry": False}
        try:
            _atomic_json(run_dir / "failure.json", failure)
        except BaseException:
            pass
        raise
    finally:
        if original_adam_step is not None:
            __import__("torch").optim.AdamW.step = original_adam_step
        if "metrics_file" in locals() and not metrics_file.closed:
            metrics_file.close()
        if "original_tail" in locals():
            import clinc_tail_runtime_v8 as tail_runtime
            tail_runtime.tail_logits = original_tail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("foundation", "continuation", "seal"))
    parser.add_argument("--master-config", required=True)
    parser.add_argument("--training-pack", required=True, help="training-only manifest; no test role")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-path", help="required for foundation/continuation")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--foundation-dir", help="root with seed17/29/43 foundation receipts")
    parser.add_argument("--output-dir", help="new run directory or existing dir with explicit safe resume")
    parser.add_argument("--resume-checkpoint", help="manual resume from a safe post-event checkpoint")
    parser.add_argument("--run-root", help="root containing seedXX/ARM run directories for seal mode")
    parser.add_argument("--output-path", help="new seal JSON path for seal mode")
    args = parser.parse_args(argv)
    if args.mode == "seal":
        if not args.run_root or not args.output_path:
            parser.error("seal mode requires --run-root and --output-path")
        result = seal_main_grid(args)
        print(json.dumps({"status": "main_six_sealed", "runs": len(result["runs"]),
                          "test_data_included": False, "output_path": str(Path(args.output_path).resolve())}, sort_keys=True))
        return
    if not args.seed or not args.output_dir or not args.model_path:
        parser.error("training modes require --seed, --model-path, and --output-dir")
    if args.mode == "foundation" and args.arm is not None:
        parser.error("foundation mode does not accept --arm")
    if args.mode == "continuation" and args.arm is None:
        parser.error("continuation mode requires --arm")
    result = run_training(args)
    print(json.dumps({"status": result["status"], "seed": result["seed"],
                      "arm": result.get("arm"), "tasks_completed": result["tasks_completed"],
                      "test_data_included": False}, sort_keys=True))


if __name__ == "__main__":
    main()
