#!/usr/bin/env python3
"""Bounded, development-only two-task CLINC pilot for the v8 tail runtime.

This is a CMS-inspired last-MLP experiment, not a native HOPE reproduction.
It reads only the locked training/development pack. There is intentionally no
test-manifest argument or test reader in this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import unicodedata
from typing import Iterable, Sequence


DEFAULT_PERIODS = (16, 64, 256, 1024, None)
VALID_ARMS = {"HOPE", "HOPE_SLEEP", "EXPAND_CE"}


def normalize_text(text: str) -> str:
    """Locked duplicate key: NFKC, casefold, underscore/space equivalence."""
    value = unicodedata.normalize("NFKC", str(text)).casefold().replace("_", " ")
    return " ".join(value.split())


def levenshtein_similarity(left: str, right: str) -> float:
    """Character-level normalized edit similarity, with a defined empty case."""
    a, b = normalize_text(left), normalize_text(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nxt = [i]
        for j, cb in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = nxt
    return 1.0 - row[-1] / max(len(a), len(b))


def token_edit_similarity(left: Sequence[int], right: Sequence[int], *, z0: int) -> float:
    """Bounded normalized edit proxy over token IDs, with explicit cutoff z0."""
    if type(z0) is not int or z0 < 1:
        raise ValueError("z0 must be a positive integer")
    a, b = list(map(int, left)), list(map(int, right))
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    distance = row[-1]
    return max(0.0, 1.0 - distance / z0)


def lti_reward(teacher_text: str, student_text: str, *, teacher_tokens: Sequence[int],
               student_tokens: Sequence[int], z0: int, canonical_labels: Sequence[str] = (),
               semantic_weight: float = 0.5, edit_weight: float = 0.5) -> float:
    """Local label-agreement/edit proxy; not the paper's learned reward model."""
    t, s = normalize_text(teacher_text), normalize_text(student_text)
    canonical = {normalize_text(x) for x in canonical_labels}
    semantic = float(bool(t) and t == s and (not canonical or t in canonical))
    edit = token_edit_similarity(teacher_tokens, student_tokens, z0=z0)
    if not math.isclose(semantic_weight + edit_weight, 1.0) or min(semantic_weight, edit_weight) < 0:
        raise ValueError("LTI proxy weights must be nonnegative and sum to one")
    return semantic_weight * semantic + edit_weight * edit


def fixed_eval_rows(rows: Sequence[dict], *, per_class: int | None = None) -> list[dict]:
    """Choose stable ID-sorted rows per class; no model score enters selection."""
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(int(row["intent_id"]), []).append(dict(row))
    out = []
    for class_id in sorted(groups):
        group = sorted(groups[class_id], key=lambda r: str(r["id"]))
        if per_class is not None:
            if per_class < 1:
                raise ValueError("per_class must be positive or None")
            group = group[:per_class]
        out.extend(group)
    return out


def task_groups(rows: Sequence[dict], class_order: Sequence[int], task_id: int) -> dict[str, list[dict]]:
    """Partition rows into seen-old/current-new by class-order task membership."""
    if not 0 <= task_id < 10 or len(class_order) != 150:
        raise ValueError("invalid task/order")
    seen = set(class_order[: (task_id + 1) * 15])
    current = set(class_order[task_id * 15 : (task_id + 1) * 15])
    old_rows, new_rows = [], []
    for row in rows:
        cls = int(row["intent_id"])
        if cls in current:
            new_rows.append(row)
        elif cls in seen:
            old_rows.append(row)
    return {"old": old_rows, "new": new_rows}


def simulate_slot_peaks(ticks: int, periods: Sequence[int | None]) -> list[int]:
    """Conservative deterministic receiver-bank occupancy under due-event order."""
    if ticks < 0 or not periods or periods[-1] is not None:
        raise ValueError("invalid tick count or terminal period")
    clocks = {i: 0 for i, period in enumerate(periods) if period is not None}
    active = [0] * len(periods)
    active[0] = 1
    peaks = list(active)
    for _ in range(ticks):
        for level in clocks:
            clocks[level] += 1
        for level in sorted(i for i, clock in clocks.items() if clock == periods[i]):
            active[level] = 0
            if level + 1 < len(active):
                active[level + 1] += 1
            # At a shared boundary the fresh receiver exists before the next
            # sender is pruned. Count this transient occupancy as well.
            peaks = [max(old, now) for old, now in zip(peaks, active)]
            clocks[level] = 0
        if active[0] == 0:
            active[0] = 1  # prepare_wake reactivates fast slot next tick.
        peaks = [max(old, now) for old, now in zip(peaks, active)]
    return peaks


def validate_config(cfg: dict) -> dict:
    """Validate every scientific and resource bound before model loading."""
    required = {
        "schema", "seed", "tasks", "arms", "rank", "capacity", "levels",
        "periods", "batch_size", "examples_per_task", "epochs_per_task", "max_steps_per_task",
        "base_lr", "fast_lr", "sleep_lr", "sleep_weight_decay", "sleep_clip_norm", "sleep_prompt_policy",
        "gkd_calls_per_event", "lti_calls_per_event", "max_new_tokens",
        "lti_edit_distance_z0", "lti_semantic_weight", "lti_edit_weight", "temperature", "eval_examples_per_class",
        "eval_full_development", "full_development_scope", "endpoint_full_clean", "checkpoint_every_steps", "max_wall_seconds",
        "max_forward_calls", "max_update_steps", "device", "max_sequence_length",
        "initialization_seed", "task0_acquisition_gate", "expected_normalized_train_dev_duplicates",
        "training_manifest_sha256", "model_revision", "model_assets_sha256", "source_revision",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    if cfg["schema"] != "clinc-cil-v8-pilot":
        raise ValueError("config schema mismatch")
    if cfg["seed"] != 17 or cfg["tasks"] != [0, 1]:
        raise ValueError("this driver is frozen to development-only seed 17, tasks [0,1]")
    if type(cfg["initialization_seed"]) is not int:
        raise ValueError("initialization_seed must be an explicit integer")
    if not isinstance(cfg["source_revision"], str) or not cfg["source_revision"]:
        raise ValueError("source_revision must be explicitly pinned")
    if not isinstance(cfg["model_revision"], str) or not cfg["model_revision"]:
        raise ValueError("model_revision must be explicitly pinned")
    if not isinstance(cfg["model_assets_sha256"], dict) or not cfg["model_assets_sha256"]:
        raise ValueError("model_assets_sha256 must pin local model files")
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v.lower())
           for v in cfg["model_assets_sha256"].values()):
        raise ValueError("every model asset hash must be lowercase/uppercase SHA256 hex")
    if not isinstance(cfg["training_manifest_sha256"], str) or len(cfg["training_manifest_sha256"]) != 64:
        raise ValueError("training_manifest_sha256 must be SHA256 hex")
    if type(cfg["expected_normalized_train_dev_duplicates"]) is not int or cfg["expected_normalized_train_dev_duplicates"] < 0:
        raise ValueError("expected duplicate count must be a nonnegative integer")
    if cfg["rank"] != 64 or cfg["levels"] != 5:
        raise ValueError("v8 pilot requires rank64, five memory levels")
    if not 1 <= int(cfg["capacity"]) <= 64:
        raise ValueError("capacity must be within [1,64]")
    arms = cfg["arms"]
    if not isinstance(arms, list) or not arms or len(set(arms)) != len(arms) or not set(arms) <= VALID_ARMS:
        raise ValueError(f"arms must be a nonempty unique subset of {sorted(VALID_ARMS)}")
    periods = cfg["periods"]
    if periods != [16, 64, 256, 1024, None]:
        raise ValueError("period schedule must be exactly [16,64,256,1024,None]")
    if cfg["batch_size"] != 8 or cfg["examples_per_task"] < 1:
        raise ValueError("batch_size must be 8 and examples_per_task positive")
    if cfg["epochs_per_task"] != 1:
        raise ValueError("this bounded pilot is frozen to one pass per selected task")
    steps = math.ceil(cfg["examples_per_task"] / cfg["batch_size"])
    if not 1 <= cfg["max_steps_per_task"] <= steps:
        raise ValueError("max_steps_per_task must be within the configured epoch schedule")
    for key in ("base_lr", "fast_lr", "sleep_lr", "sleep_clip_norm", "temperature"):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(float(cfg["sleep_weight_decay"])) or float(cfg["sleep_weight_decay"]) < 0:
        raise ValueError("sleep_weight_decay must be finite and nonnegative")
    if cfg["sleep_prompt_policy"] not in ("current", "seen_replay"):
        raise ValueError("sleep_prompt_policy must be current or seen_replay")
    for key in ("gkd_calls_per_event", "lti_calls_per_event", "max_new_tokens", "eval_examples_per_class",
                "checkpoint_every_steps", "max_wall_seconds", "max_forward_calls", "max_update_steps",
                "max_sequence_length"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["lti_edit_distance_z0"]) is not int or cfg["lti_edit_distance_z0"] < 1:
        raise ValueError("lti_edit_distance_z0 must be a positive integer")
    if cfg["lti_semantic_weight"] != 0.5 or cfg["lti_edit_weight"] != 0.5:
        raise ValueError("local LTI proxy weights are frozen at 0.5/0.5")
    if not 0 <= float(cfg["task0_acquisition_gate"]) <= 1:
        raise ValueError("task0_acquisition_gate must be in [0,1]")
    if type(cfg["eval_full_development"]) is not bool:
        raise ValueError("eval_full_development must be boolean")
    if cfg["full_development_scope"] != "seen_classes":
        raise ValueError("secondary full-development evaluation is limited to seen classes")
    if cfg["endpoint_full_clean"] is not True:
        raise ValueError("pilot endpoints must include full clean-development evaluation")
    if cfg["device"] not in ("cuda:0", "cuda:1", "cpu"):
        raise ValueError("device must be explicitly cuda:0, cuda:1 or cpu")
    cfg = dict(cfg)
    cfg["derived_steps_per_epoch"] = steps
    cfg["periods"] = tuple(periods)
    peaks = simulate_slot_peaks(len(cfg["tasks"]) * cfg["max_steps_per_task"], cfg["periods"])
    cfg["derived_slot_peaks"] = peaks
    if any(peak > cfg["capacity"] for peak in peaks):
        raise ValueError(f"configured capacity cannot cover the fixed update schedule: {peaks}")
    return cfg


class Budget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.started = time.monotonic()
        self.forward_calls = 0
        self.forward_tokens = 0
        self.executed_input_tokens = 0
        self.padded_input_tokens = 0
        self.forward_shape_histogram = {}
        self.tail_calls = 0
        self.tail_grad_calls = 0
        self.tail_no_grad_calls = 0
        self.tail_shape_histogram = {}
        self.tail_grad_shape_histogram = {}
        self.tail_no_grad_shape_histogram = {}
        self.generated_valid_tokens = 0
        self.generated_placeholder_tokens = 0
        self.backward_calls = 0
        self.backward_tokens = 0
        self.update_steps = 0
        self.sleep_events = 0

    def check(self):
        if time.monotonic() - self.started > self.cfg["max_wall_seconds"]:
            raise RuntimeError("hard wall-time budget exceeded")
        if self.forward_calls > self.cfg["max_forward_calls"]:
            raise RuntimeError("hard forward-call budget exceeded")
        if self.update_steps > self.cfg["max_update_steps"]:
            raise RuntimeError("hard update-step budget exceeded")

    def forward(self, tokens):
        self.forward_calls += 1
        valid = int(tokens.sum().item())
        executed = int(tokens.numel())
        self.forward_tokens += valid
        self.executed_input_tokens += executed
        self.padded_input_tokens += executed - valid
        shape = f"{tokens.shape[0]}x{tokens.shape[1]}"
        self.forward_shape_histogram[shape] = self.forward_shape_histogram.get(shape, 0) + 1
        self.check()

    def tail(self, cache, grad_enabled):
        self.tail_calls += 1
        self.tail_grad_calls += int(grad_enabled)
        self.tail_no_grad_calls += int(not grad_enabled)
        shape = f"{cache.y.shape[0]}x{cache.y.shape[1]}"
        self.tail_shape_histogram[shape] = self.tail_shape_histogram.get(shape, 0) + 1
        selected = self.tail_grad_shape_histogram if grad_enabled else self.tail_no_grad_shape_histogram
        selected[shape] = selected.get(shape, 0) + 1

    def backward(self, tokens):
        self.backward_calls += 1
        self.backward_tokens += int(tokens.sum().item())
        self.check()

    def update(self, amount=1):
        self.update_steps += amount
        self.check()

    def as_dict(self):
        return {"full_backbone_forward_calls": self.forward_calls,
                "full_backbone_forward_tokens": self.forward_tokens,
                "executed_input_tokens_including_padding": self.executed_input_tokens,
                "padded_input_tokens": self.padded_input_tokens,
                "forward_input_shape_histogram": dict(sorted(self.forward_shape_histogram.items())),
                "tail_calls": self.tail_calls, "tail_grad_enabled_calls": self.tail_grad_calls,
                "tail_no_grad_calls": self.tail_no_grad_calls,
                "tail_shape_histogram": dict(sorted(self.tail_shape_histogram.items())),
                "tail_grad_shape_histogram": dict(sorted(self.tail_grad_shape_histogram.items())),
                "tail_no_grad_shape_histogram": dict(sorted(self.tail_no_grad_shape_histogram.items())),
                "sampled_answer_tokens_including_first_eos": self.generated_valid_tokens,
                "forced_post_eos_placeholders": self.generated_placeholder_tokens,
                "tail_backward_calls": self.backward_calls,
                "tail_backward_tokens": self.backward_tokens,
                "optimizer_or_lifecycle_update_steps": self.update_steps,
                "sleep_events": self.sleep_events,
                "active_wall_seconds": time.monotonic() - self.started}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _mutable_context_sha(context) -> str:
    """Stable digest over the small mutable adapter context, not the backbone."""
    h = hashlib.sha256()
    state = context.state_dict()
    for name, tensor in sorted(state["memory"].items()):
        h.update(name.encode("utf-8"))
        value = tensor.detach().contiguous().cpu()
        h.update(str(value.dtype).encode("ascii"))
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    for name in ("readout_a", "readout_b"):
        value = state[name].detach().contiguous().cpu()
        h.update(name.encode("ascii"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    h.update(_canonical_sha({"routing": state["routing"], "trainable": state["trainable"],
                             "readout_fp32_accumulation": state["readout_fp32_accumulation"]}).encode("ascii"))
    return h.hexdigest()


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
        if _sha256(path) != hashlib.sha256(payload).hexdigest():
            raise IOError("atomic JSON readback hash mismatch")
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_torch(path: Path, value, torch):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, tmp_name)
        with open(tmp_name, "r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        expected = _sha256(Path(tmp_name))
        os.replace(tmp_name, path)
        digest = _sha256(path)
        if digest != expected:
            raise IOError("checkpoint readback hash mismatch")
        check = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(check, dict) or check.get("schema") != value.get("schema"):
            raise IOError("checkpoint readback schema mismatch")
        return digest
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _rng_state(torch):
    import numpy as np
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state().clone(),
            "torch_cuda": [x.cpu().clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None}


def _save_checkpoint(root, arm, task_id, step, context, wake, torch, cfg, manifest_sha, source_hashes):
    state = {"schema": "clinc-cil-v8-checkpoint", "seed": cfg["seed"], "arm": arm,
             "task_id": task_id, "step": step, "context": context.state_dict(),
             "wake": wake.snapshot(), "rng": _rng_state(torch),
             "config_sha256": _canonical_sha({k: v for k, v in cfg.items() if k != "derived_steps_per_epoch"}),
             "training_manifest_sha256": manifest_sha, "source_hashes": source_hashes}
    path = root / "checkpoints" / arm / f"task{task_id:02d}-step{step:05d}.pt"
    return path, _atomic_torch(path, state, torch)


def _load_common(project_root, model_path, device, cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source_dir = project_root / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"public source directory missing: {source_dir}")
    for value in (str(source_dir),):
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    from nested_sleep.lm import LlamaCMSLM, promote_mutable_to_fp32, immutable_versions
    lm_path = Path(sys.modules["nested_sleep.lm"].__file__).resolve()
    if source_dir.resolve() not in lm_path.parents:
        raise RuntimeError(f"public source binding failed; nested_sleep.lm loaded from {lm_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("local tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager").eval()
    backbone.requires_grad_(False)
    model = LlamaCMSLM(backbone, rank=cfg["rank"], capacity=cfg["capacity"], levels=cfg["levels"])
    promote_mutable_to_fp32(model)
    model.eval()
    return torch, tokenizer, model, immutable_versions


def _batch(rows, tokenizer, view, cfg, torch, device):
    from clinc_cil_common_v8 import tokenize_batch
    data = tokenize_batch(rows, tokenizer, view.classes, tokenizer.eos_token_id,
        tokenizer.pad_token_id, cfg["max_sequence_length"], prepend_bos=False)
    ids = torch.tensor(data["input_ids"], dtype=torch.long, device=device)
    mask = torch.tensor(data["attention_mask"], dtype=torch.long, device=device)
    labels = torch.tensor(data["labels"], dtype=torch.long, device=device)
    return data, ids, mask, labels


def _prefix(row, tokenizer, view, cfg):
    from clinc_cil_common_v8 import tokenize_prompt
    item = tokenize_prompt(row["text"], tokenizer, view.classes, tokenizer.eos_token_id,
                           cfg["max_sequence_length"], prepend_bos=False)
    return item["input_ids"][:item["prompt_length"]]


def _decode_answer(ids, tokenizer, eos):
    values = list(map(int, ids))
    if eos in values:
        values = values[:values.index(eos)]
    return tokenizer.decode(values, skip_special_tokens=True)


def _greedy_predictions(model, context, rows, tokenizer, view, cfg, torch, device, budget):
    from clinc_cil_common_v8 import tokenize_prompt
    from clinc_tail_runtime_v8 import capture, tail_logits
    # Group by exact prompt length so the model never sees padded capture rows.
    groups = {}
    for original_index, row in enumerate(rows):
        item = tokenize_prompt(row["text"], tokenizer, view.classes, tokenizer.eos_token_id,
                               cfg["max_sequence_length"], prepend_bos=False)
        prefix = item["input_ids"][:item["prompt_length"]]
        groups.setdefault(len(prefix), []).append((original_index, row, prefix))
    maximum = max(len(tokenizer.encode(label, add_special_tokens=False)) for label in view.classes) + 1
    # Length batching is an execution detail. Restore the caller's row order
    # before metrics select seen/old/current subsets by their original row IDs.
    predictions, gold = [None] * len(rows), [None] * len(rows)
    for prompt_len in sorted(groups):
        group = groups[prompt_len]
        for start in range(0, len(group), cfg["batch_size"]):
            chunk = group[start:start + cfg["batch_size"]]
            ids = torch.tensor([x[2] for x in chunk], dtype=torch.long, device=device)
            active = torch.ones_like(ids, dtype=torch.long)
            finished = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            emitted = [[] for _ in chunk]
            for _ in range(maximum):
                budget.check()
                cache = capture(model, ids, active, device=device)
                logits = tail_logits(model, context, cache)[:, -1].float()
                next_ids = logits.argmax(-1)
                for i, token in enumerate(next_ids.tolist()):
                    if not bool(finished[i]):
                        emitted[i].append(int(token))
                        if token == tokenizer.eos_token_id:
                            finished[i] = True
                if bool(finished.all()):
                    break
                ids = torch.cat([ids, next_ids[:, None]], dim=1)
                active = torch.cat([active, (~finished).long()[:, None]], dim=1)
            for (original_index, row, _), answer in zip(chunk, emitted):
                predictions[original_index] = _decode_answer(answer, tokenizer, tokenizer.eos_token_id)
                gold[original_index] = int(row["intent_id"])
    from clinc_cil_common_v8 import score_predictions
    return predictions, gold, score_predictions(predictions, gold, view.classes, view.class_orders[str(cfg["seed"])])


def _evaluate(model, context, dev_rows, class_order, task_id, tokenizer, view, cfg, torch, device, budget,
              *, include_unseen=False):
    seen = set(class_order[: (task_id + 1) * 15])
    rows = list(dev_rows) if include_unseen else [r for r in dev_rows if int(r["intent_id"]) in seen]
    groups = task_groups([r for r in rows if int(r["intent_id"]) in seen], class_order, task_id)
    with torch.no_grad():
        preds, gold, _ = _greedy_predictions(model, context, rows, tokenizer, view, cfg, torch, device, budget)
    def summarize(subset):
        keys = {str(r["id"]) for r in subset}
        chosen = [(p, int(r["intent_id"])) for p, r in zip(preds, rows) if str(r["id"]) in keys]
        if not chosen:
            return {"count": 0, "accuracy": None, "invalid": None}
        from clinc_cil_common_v8 import score_predictions
        metric = score_predictions([x[0] for x in chosen], [x[1] for x in chosen], view.classes)
        macro = [v["accuracy"] for v in metric["per_class"].values() if v["accuracy"] is not None]
        return {"count": len(chosen), "accuracy": metric["accuracy"],
                "macro_accuracy": sum(macro) / len(macro) if macro else None,
                "class_count": len(macro), "invalid": metric["invalid_predictions"],
                "predictions_sha256": _canonical_sha([x[0] for x in chosen])}
    seen_rows = [r for r in rows if int(r["intent_id"]) in seen]
    results = {"seen": summarize(seen_rows), "old": summarize(groups["old"]), "current": summarize(groups["new"])}
    if include_unseen:
        unseen_rows = [r for r in rows if int(r["intent_id"]) not in seen]
        results["unseen"] = summarize(unseen_rows)
        results["all_development"] = summarize(rows)
    return {"task_id": task_id, "metrics": results,
            "evaluation_scope": "all_development" if include_unseen else "seen_classes"}


def _sleep_prompt_batch(pool, count, event, kind, task_id):
    if not pool:
        raise RuntimeError("sleep prompt pool is empty")
    # The pool is preordered by a fixed per-event schedule (and, for replay,
    # interleaves old/current prompts exactly 50:50). Do not randomize it here.
    try:
        call = int(kind.rsplit("-", 1)[1])
    except (ValueError, IndexError):
        call = 0
    offset = (event * max(1, count) + call + task_id) % len(pool)
    return [pool[(offset + i) % len(pool)] for i in range(count)]


def _sample_tokens(model, context, prompt_ids, count, temperature, generator, eos, torch, device, budget):
    from clinc_tail_runtime_v8 import sample_with_context
    prefix = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = sample_with_context(model, context, prefix, count, generator=generator,
        temperature=temperature, eos_token_id=eos, return_info=True)
    budget.generated_valid_tokens += int(result.actual_generated_tokens - result.executed_extra_tokens)
    budget.generated_placeholder_tokens += int(result.executed_extra_tokens)
    # The model.logits wrapper counts every full-backbone capture call.
    answer = result.token_ids[0, result.prefix_length:].tolist()
    if eos in answer:
        answer = answer[:answer.index(eos) + 1]
    return result, answer


def _audited_student_logits(model, context, cache, receiver_level, receiver_slot, answer_mask):
    """Temporarily audit only the receiving expert; restore runtime buffers."""
    from copy import deepcopy
    import clinc_tail_runtime_v8 as tail_runtime
    banks = context.banks()
    if receiver_level >= len(banks):
        raise ValueError("receiver level outside context")
    saved = [(bank, bool(bank.audit_routes), bank.last_route_audit, list(bank.audit_history),
              deepcopy(bank.last_routes)) for bank in banks]
    try:
        for i, bank in enumerate(banks):
            bank.audit_routes = i == receiver_level
        logits = tail_runtime.tail_logits(model, context, cache)
        audit = banks[receiver_level].last_route_audit
        coverage = None
        selected_count = 0
        answer_count = int(answer_mask.sum().item())
        if audit is not None and answer_count:
            selected = audit["selected"].reshape(-1)
            mask = answer_mask.reshape(-1).to(selected.device).bool()
            selected_count = int((selected[mask] == receiver_slot).sum().item())
            coverage = selected_count / answer_count
        return logits, {"receiver_selected_answer_tokens": selected_count,
                        "answer_tokens": answer_count, "receiver_answer_coverage": coverage,
                        "route_score_abs_max": None if audit is None else audit.get("score_abs_max")}
    finally:
        for bank, audit_routes, last_audit, history, routes in saved:
            bank.audit_routes = audit_routes
            bank.last_route_audit = last_audit
            bank.audit_history = history
            bank.last_routes = routes


def _run_sleep_event(model, tx, row_pool, tokenizer, view, cfg, torch, device, budget, task_id, event):
    from clinc_tail_runtime_v8 import capture, tail_logits, causal_token_kl, causal_token_ce
    gen = tx.generator
    outputs = []
    methods = []
    for i in range(max(cfg["gkd_calls_per_event"], cfg["lti_calls_per_event"])):
        if i < cfg["gkd_calls_per_event"]:
            methods.append("gkd")
        if i < cfg["lti_calls_per_event"]:
            methods.append("lti")
    gkd_index = 0
    for call, kind in enumerate(methods):
        budget.check()
        row = _sleep_prompt_batch(row_pool, 1, event, f"{kind}-{call}", task_id)[0]
        prompt = _prefix(row, tokenizer, view, cfg)
        owner = "teacher" if kind == "lti" or gkd_index % 2 == 0 else "student"
        if kind == "gkd":
            gkd_index += 1
        owner_context = tx.teacher_context() if owner == "teacher" else tx.student_context()
        sampled_result, sampled_answer = _sample_tokens(model, owner_context, prompt,
            cfg["max_new_tokens"], cfg["temperature"], gen, tokenizer.eos_token_id, torch, device, budget)
        if kind == "gkd":
            seq = sampled_result.token_ids
            answer_mask = sampled_result.answer_mask()
            fullmask = torch.ones_like(seq, dtype=torch.long)
            cache = capture(model, seq, fullmask, device=device)
            with torch.no_grad():
                tlogits = tail_logits(model, tx.teacher_context(), cache)
            slogits, route_stats = _audited_student_logits(model, tx.student_context(), cache,
                tx.receiver, tx.slot, answer_mask)
            kl, token_count = causal_token_kl(slogits, tlogits, answer_mask, temperature=1.0)
            loss = kl
            reward = None
        else:
            # Student continues from a random proper prefix of teacher answer,
            # excluding the final EOS as a continuation source.
            teacher_ids = sampled_answer[:-1] if sampled_answer and sampled_answer[-1] == tokenizer.eos_token_id else sampled_answer
            prefix_len = (int(torch.randint(0, len(teacher_ids), (1,), generator=gen, device=gen.device).item())
                          if teacher_ids else 0)
            student_prefix = prompt + teacher_ids[:prefix_len]
            eos_emitted = bool(sampled_answer and sampled_answer[-1] == tokenizer.eos_token_id)
            suffix_budget = max(1, len(teacher_ids) - prefix_len + int(eos_emitted))
            student_result, student_answer = _sample_tokens(model, tx.student_context(), student_prefix,
                max(1, min(cfg["max_new_tokens"], suffix_budget)),
                cfg["temperature"], gen, tokenizer.eos_token_id, torch, device, budget)
            student_seq = student_result.token_ids
            student_mask = student_result.answer_mask()
            student_cache = capture(model, student_seq, torch.ones_like(student_seq), device=device)
            student_logits, route_stats = _audited_student_logits(model, tx.student_context(), student_cache,
                tx.receiver, tx.slot, student_mask)
            with torch.no_grad():
                teacher_logits = tail_logits(model, tx.teacher_context(), student_cache)
            kl, token_count = causal_token_kl(student_logits, teacher_logits, student_mask, temperature=1.0)
            # Reward compares the full teacher answer with supplied-prefix plus
            # the student's suffix; old gold labels are never read by this loss.
            teacher_text = _decode_answer(teacher_ids, tokenizer, tokenizer.eos_token_id)
            student_suffix = student_answer[:-1] if student_answer and student_answer[-1] == tokenizer.eos_token_id else student_answer
            student_full_tokens = teacher_ids[:prefix_len] + student_suffix
            student_text = _decode_answer(student_full_tokens, tokenizer, tokenizer.eos_token_id)
            reward = lti_reward(teacher_text, student_text, teacher_tokens=teacher_ids,
                                student_tokens=student_full_tokens, z0=cfg["lti_edit_distance_z0"],
                                canonical_labels=view.classes,
                                semantic_weight=cfg["lti_semantic_weight"], edit_weight=cfg["lti_edit_weight"])
            student_logprob, student_count = causal_token_ce(student_logits, student_seq, student_mask)
            # Maximizing answer log probability contributes negative CE; this
            # sign keeps the implemented scalar equal to .5 KL - .5 r log p.
            loss = 0.5 * kl + 0.5 * reward * student_logprob
            token_count += student_count
        budget.backward(torch.ones((1, token_count), dtype=torch.long, device=device))
        budget.update()  # Reserve a bounded receiver update before AdamW.step.
        tx.optimize(loss, kind, max(1, token_count))
        teacher_text = _decode_answer(sampled_answer, tokenizer, tokenizer.eos_token_id)
        teacher_norm = normalize_text(teacher_text)
        valid_labels = {normalize_text(x) for x in view.classes}
        prompt_task = next(i for i in range(10) if int(row["intent_id"]) in set(view.class_orders[str(cfg["seed"])][i * 15:(i + 1) * 15]))
        outputs.append({"kind": kind, "token_count": token_count,
                        "loss": float(loss.detach().float().cpu()),
                        "gkd_kl_weight": 1.0 if kind == "gkd" else 0.5,
                        "objective_note": "alternating GKD/LTI optimizer steps; approximate lambda mix, not one combined update",
                        "sampled_output_label_valid": teacher_norm in valid_labels,
                        "sampled_output_matches_prompt_gold_diagnostic": teacher_norm == normalize_text(view.classes[int(row["intent_id"])]),
                        "sampled_answer_tokens": len(sampled_answer),
                        "sampled_first_token_eos": bool(sampled_answer and sampled_answer[0] == tokenizer.eos_token_id),
                        "student_receiver_route": route_stats,
                        "prompt_id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest(),
                        "prompt_source_task": prompt_task,
                        "prompt_is_prior_task_replay": prompt_task < task_id,
                        "sample_owner": owner,
                        "lti_reward_proxy": reward if kind == "lti" else None})
    return outputs


def _run_expand_ce_event(model, tx, row_pool, tokenizer, view, cfg, torch, device, budget, task_id, event):
    """Capacity-matched but explicitly gold-supervised receiver control."""
    from clinc_tail_runtime_v8 import capture, tail_logits
    count = cfg["gkd_calls_per_event"] + cfg["lti_calls_per_event"]
    outputs = []
    for call, row in enumerate(_sleep_prompt_batch(row_pool, count, event, "expand-ce", task_id)):
        data, ids, mask, labels = _batch([row], tokenizer, view, cfg, torch, device)
        cache = capture(model, ids, mask, device=device)
        logits = tail_logits(model, tx.student_context(), cache)
        targets = labels[:, 1:].reshape(-1)
        valid = targets.ne(-100)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float()[valid], targets[valid])
        token_count = int(valid.sum().item())
        budget.backward(mask * labels.ne(-100).long())
        budget.update()  # Reserve before the receiver's AdamW.step.
        tx.optimize(loss, "gkd", max(1, token_count))  # Lifecycle counter bucket; objective remains gold CE.
        outputs.append({"objective": "gold_label_CE", "token_count": token_count,
                        "loss": float(loss.detach().float().cpu()), "uses_gold_intent": True,
                        "prompt_id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest()})
    return outputs


def _source_hashes(project_root: Path, task_root: Path):
    paths = [project_root / "src" / "nested_sleep" / "lm.py",
             project_root / "src" / "nested_sleep" / "experts.py",
             project_root / "src" / "nested_sleep" / "accounting.py",
             project_root / "src" / "nested_sleep" / "native.py",
             task_root / "tools" / "clinc_cil_common_v8.py",
             task_root / "tools" / "clinc_tail_runtime_v8.py",
             task_root / "tools" / "clinc_lifecycle_v8.py",
             Path(__file__).resolve()]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"source files missing: {missing}")
    return {str(p.resolve()): _sha256(p) for p in paths}


_RESUME_SCIENCE_FIELDS = (
    "seed", "initialization_seed", "tasks", "rank", "capacity", "levels", "periods",
    "epochs_per_task", "examples_per_task", "batch_size", "max_steps_per_task",
    "base_lr", "fast_lr", "sleep_lr", "sleep_weight_decay", "sleep_clip_norm",
    "sleep_prompt_policy", "gkd_calls_per_event", "lti_calls_per_event",
    "max_new_tokens", "lti_edit_distance_z0", "lti_semantic_weight",
    "lti_edit_weight", "temperature", "max_sequence_length",
    "expected_normalized_train_dev_duplicates", "task0_acquisition_gate",
)


def _plain(value):
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _resume_config_differences(parent_cfg, current_cfg):
    """Return immutable scientific mismatches; evaluation/resources may differ."""
    return {key: {"parent": _plain(parent_cfg.get(key)), "current": _plain(current_cfg.get(key))}
            for key in _RESUME_SCIENCE_FIELDS
            if _plain(parent_cfg.get(key)) != _plain(current_cfg.get(key))}


def _validate_resume_contract(parent_protocol, parent_receipts, qualification, *, cfg,
                              current_r2_source_sha256, training_manifest_sha256,
                              manifest_source_sha256, parent_protocol_sha256):
    """Validate the external corrected task-0 gate and all source/data bindings.

    Qualification is the independently generated corrected rescore artifact:
    `clinc-v8-checkpoint-rescore-v1`. Its task-0 clean metric and checkpoint
    fields are read directly from the frozen rescore schema, never from R1's
    invalid acquisition-gate record.
    """
    if parent_protocol.get("schema") != "clinc-cil-v8-pilot-protocol":
        raise ValueError("resume parent protocol schema mismatch")
    if parent_protocol.get("training_manifest_sha256") != training_manifest_sha256:
        raise ValueError("resume parent training manifest SHA mismatch")
    if parent_protocol.get("manifest_source_sha256") != manifest_source_sha256:
        raise ValueError("resume parent manifest source mismatch")
    if parent_protocol.get("model_revision") != cfg["model_revision"] or \
       parent_protocol.get("model_assets_sha256") != cfg["model_assets_sha256"]:
        raise ValueError("resume parent model revision/assets mismatch")
    parent_cfg = parent_protocol.get("config")
    if not isinstance(parent_cfg, dict):
        raise ValueError("resume parent protocol lacks config")
    differences = _resume_config_differences(parent_cfg, cfg)
    if differences:
        raise ValueError(f"resume changes locked science parameters: {differences}")
    q = qualification
    if q.get("schema") != "clinc-v8-checkpoint-rescore-v1" or q.get("status") != "complete":
        raise ValueError("corrected task-0 rescore schema/status mismatch")
    if q.get("training_only") is not True:
        raise ValueError("resume qualification is not bound to training/development-only data")
    selection = q.get("selection")
    if not isinstance(selection, dict) or selection.get("test_data_read") is not False:
        raise ValueError("resume qualification must explicitly attest that test data was not read")
    expected_bindings = {
        "r1_protocol_sha256": parent_protocol_sha256,
        "r1_source_lock_sha256": parent_protocol.get("source_lock_sha256"),
        "r2_source_sha256": current_r2_source_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "model_assets_sha256": cfg["model_assets_sha256"],
        "model_revision": cfg["model_revision"],
    }
    for field, expected in expected_bindings.items():
        if q.get(field) != expected:
            raise ValueError(f"resume rescore binding mismatch: {field}")
    verified_r1_sources = q.get("r1_source_hashes_verified")
    if not isinstance(verified_r1_sources, dict) or verified_r1_sources != parent_protocol.get("source_hashes"):
        raise ValueError("rescore did not verify the exact archived R1 source hash map")
    if _canonical_sha(verified_r1_sources) != parent_protocol.get("source_lock_sha256"):
        raise ValueError("rescore R1 source hash map does not reproduce the archived source lock")
    arms_data = q.get("arms")
    if not isinstance(arms_data, dict) or set(arms_data) != set(cfg["arms"]):
        raise ValueError("rescore must contain exactly the requested resume arms")
    receipt_by_arm = {str(item.get("arm")): item for item in parent_receipts}
    if not set(cfg["arms"]).issubset(receipt_by_arm):
        raise ValueError("requested resume arm is absent from parent run_complete receipts")
    out = {}
    for arm in cfg["arms"]:
        receipt = receipt_by_arm[arm]
        if receipt.get("status") not in ("complete", "stopped_task0_acquisition_gate") or \
           receipt.get("tasks_completed") != 1 or receipt.get("ticks") != cfg["max_steps_per_task"]:
            raise ValueError(f"parent arm {arm} is not a completed task-0 checkpoint")
        cp_sha = receipt.get("final_checkpoint_sha256")
        if not isinstance(cp_sha, str) or len(cp_sha) != 64:
            raise ValueError(f"parent arm {arm} has invalid final checkpoint hash")
        entry = arms_data[arm]
        checkpoint = entry.get("checkpoint")
        metric_group = entry.get("task0_full_clean_development_seen15")
        current = (metric_group.get("metrics", {}).get("current", {})
                   if isinstance(metric_group, dict) else {})
        if not isinstance(checkpoint, dict) or checkpoint.get("tasks_completed") != 1:
            raise ValueError(f"rescore arm {arm} is not bound to a task-0-only checkpoint")
        if checkpoint.get("sha256") != cp_sha:
            raise ValueError(f"qualification arm {arm} checkpoint hash mismatch")
        if current.get("count") != 300:
            raise ValueError(f"rescore arm {arm} did not score all 300 clean seen task-0 rows")
        try:
            accuracy = float(current["accuracy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"rescore arm {arm} lacks a valid current-class accuracy") from exc
        if not math.isfinite(accuracy) or accuracy < float(cfg["task0_acquisition_gate"]):
            raise ValueError(f"corrected clean-development acquisition gate failed for {arm}")
        if receipt.get("common_init_sha256") is None:
            raise ValueError(f"parent arm {arm} lacks common initialization binding")
        out[arm] = {"receipt": receipt, "qualification": entry,
                    "task0_clean_accuracy": accuracy}
    return out


def _assert_context_states_equal(left, right, torch):
    if set(left) != set(right):
        raise ValueError("checkpoint top-level context and wake snapshot schema differ")
    if set(left["memory"]) != set(right["memory"]):
        raise ValueError("checkpoint context/wake memory keys differ")
    for key in left["memory"]:
        if not torch.equal(left["memory"][key].cpu(), right["memory"][key].cpu()):
            raise ValueError(f"checkpoint context/wake tensor mismatch: {key}")
    for key in ("readout_a", "readout_b"):
        if not torch.equal(left[key].cpu(), right[key].cpu()):
            raise ValueError(f"checkpoint context/wake tensor mismatch: {key}")
    for key in ("readout_fp32_accumulation", "routing", "trainable"):
        if _plain(left[key]) != _plain(right[key]):
            raise ValueError(f"checkpoint context/wake metadata mismatch: {key}")


def _restore_parent_task0_checkpoint(context, wake, checkpoint_path: Path, *, receipt,
                                     qualification, cfg, torch):
    """Load a source-bound task-0 state, restore clocks/pending state and all RNGs."""
    actual_sha = _sha256(checkpoint_path)
    q_checkpoint = qualification.get("checkpoint", {})
    if actual_sha != receipt["final_checkpoint_sha256"] or actual_sha != q_checkpoint.get("sha256"):
        raise ValueError("resume checkpoint content hash mismatch")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("schema") != "clinc-cil-v8-final-state":
        raise ValueError("resume checkpoint schema mismatch")
    if state.get("arm") != receipt.get("arm") or state.get("seed") != cfg["seed"]:
        raise ValueError("resume checkpoint arm/seed mismatch")
    if state.get("wake") is None or not isinstance(state.get("rng"), dict):
        raise ValueError("resume checkpoint lacks wake/RNG state")
    _assert_context_states_equal(state["context"], state["wake"].context_state, torch)
    context.load_state_dict(state["context"])
    loaded_context_sha = _mutable_context_sha(context)
    wake.restore(state["wake"])
    if _mutable_context_sha(context) != loaded_context_sha:
        raise ValueError("wake restore changed serialized mutable context")
    expected_clocks = {i: cfg["max_steps_per_task"] % int(period)
                       for i, period in enumerate(cfg["periods"]) if period is not None}
    if wake.ticks != cfg["max_steps_per_task"] or wake.clocks != expected_clocks:
        raise ValueError(f"task-0 wake cursor/pending clocks mismatch: {wake.ticks}, {wake.clocks}")
    rng = state["rng"]
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(rng) != required_rng:
        raise ValueError("resume checkpoint RNG schema mismatch")
    import numpy as np
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch_cpu"].cpu().clone())
    if rng["torch_cuda"] is not None:
        if not torch.cuda.is_available() or len(rng["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA RNG topology differs from parent checkpoint")
        torch.cuda.set_rng_state_all([x.cpu().clone() for x in rng["torch_cuda"]])
    elif cfg["device"].startswith("cuda"):
        raise ValueError("CUDA resume requested but parent checkpoint lacks CUDA RNG states")
    return {"checkpoint_sha256": actual_sha, "context_sha256": loaded_context_sha,
            "restored_ticks": wake.ticks, "restored_clocks": dict(wake.clocks)}


def _load_resume_inputs(resume_run_dir: Path, qualification_path: Path, *, cfg,
                        current_source_lock_sha256, current_r2_source_sha256, training_manifest_sha256,
                        manifest_source_sha256):
    """Read and validate R1 protocol/receipts/CP bytes plus the R2 correction."""
    if not resume_run_dir.is_dir() or not qualification_path.is_file():
        raise FileNotFoundError("resume directory or corrected qualification file is missing")
    protocol_path = resume_run_dir / "protocol.json"
    metrics_path = resume_run_dir / "metrics.jsonl"
    if not protocol_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("resume parent lacks protocol.json or metrics.jsonl")
    protocol_sha = _sha256(protocol_path)
    parent_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    records = []
    with metrics_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "run_complete" and isinstance(record.get("receipt"), dict):
                records.append(record["receipt"])
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    matched = _validate_resume_contract(parent_protocol, records, qualification,
        cfg=cfg, current_r2_source_sha256=current_r2_source_sha256,
        training_manifest_sha256=training_manifest_sha256,
        manifest_source_sha256=manifest_source_sha256,
        parent_protocol_sha256=protocol_sha)
    for arm, item in matched.items():
        relative = Path(item["receipt"].get("final_checkpoint", ""))
        cp_path = (resume_run_dir / relative).resolve()
        if resume_run_dir.resolve() not in cp_path.parents or not cp_path.is_file():
            raise ValueError(f"resume checkpoint path escapes parent run or is absent for {arm}")
        if _sha256(cp_path) != item["receipt"]["final_checkpoint_sha256"]:
            raise ValueError(f"resume checkpoint bytes do not match run_complete receipt for {arm}")
        item["checkpoint_path"] = str(cp_path)
    return {"parent_run_dir": str(resume_run_dir.resolve()),
            "parent_protocol_sha256": protocol_sha,
            "parent_source_lock_sha256": parent_protocol.get("source_lock_sha256"),
            "qualification_path": str(qualification_path.resolve()),
            "qualification_sha256": _sha256(qualification_path),
            "r2_source_lock_sha256": current_source_lock_sha256,
            "training_manifest_sha256": training_manifest_sha256,
            "manifest_source_sha256": manifest_source_sha256,
            "arms": matched}


def run(args) -> dict:
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    pack_path = Path(args.training_pack).resolve()
    output_dir = Path(args.output_dir).resolve()
    cfg_path = Path(args.config).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {model_path}")
    if not pack_path.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {pack_path}")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config does not exist: {cfg_path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if output_dir == project_root or project_root in output_dir.parents:
        raise ValueError("outputs must be written outside the public project tree")
    if output_dir == model_path or model_path in output_dir.parents:
        raise ValueError("outputs must not be written inside the local model asset directory")
    cfg = validate_config(json.loads(cfg_path.read_text(encoding="utf-8")))
    if _sha256(pack_path) != cfg["training_manifest_sha256"]:
        raise ValueError("training manifest SHA256 differs from config lock")
    for relative, expected_sha in cfg["model_assets_sha256"].items():
        asset = (model_path / relative).resolve()
        if model_path not in asset.parents or not asset.is_file() or _sha256(asset) != expected_sha:
            raise ValueError(f"model asset hash mismatch: {relative}")
    source_hashes = _source_hashes(project_root, Path(__file__).resolve().parents[1])
    source_hash = _canonical_sha(source_hashes)
    task_tools = Path(__file__).resolve().parent
    if str(task_tools) in sys.path:
        sys.path.remove(str(task_tools))
    sys.path.insert(0, str(task_tools))
    manifest_bytes = pack_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    from clinc_cil_common_v8 import make_training_view, iter_role, validate_role_partition, shuffled_ids, task_class_ids, normalize_prediction
    view = make_training_view(manifest)
    # The manifest only contains train/dev. All role file digests are checked
    # before model loading, and exact ID partition is verified.
    pack_root = pack_path.parent
    train_rows = list(iter_role(view, "train", pack_root))
    dev_rows = list(iter_role(view, "development", pack_root))
    partition = validate_role_partition(train_rows, dev_rows)
    train_norm = {normalize_text(r["text"]) for r in train_rows}
    duplicate_dev = [r for r in dev_rows if normalize_text(r["text"]) in train_norm]
    clean_dev = [r for r in dev_rows if normalize_text(r["text"]) not in train_norm]
    if len(duplicate_dev) != cfg["expected_normalized_train_dev_duplicates"]:
        raise ValueError("normalized train/dev duplicate count differs from configured expectation")
    data_note = {"train_rows": len(train_rows), "development_rows": len(dev_rows),
                 "normalized_train_dev_duplicate_rows": len(duplicate_dev),
                 "clean_development_rows": len(clean_dev), "partition": partition,
                 "test_data_included": False}
    if len(train_rows) != 15000 or len(dev_rows) != 3000:
        raise ValueError("pilot expects locked CLINC train=15000 and development=3000")
    # Check selected tasks have the expected 100 examples/class before creating output.
    order = view.class_orders[str(cfg["seed"])]
    for task_id in cfg["tasks"]:
        ids = task_class_ids(order, task_id)
        subset = [r for r in train_rows if int(r["intent_id"]) in set(ids)]
        if len(subset) < cfg["examples_per_task"]:
            raise ValueError(f"task {task_id} has too few training examples")
    resume_run_dir = getattr(args, "resume_run_dir", None)
    resume_qualification = getattr(args, "resume_qualification", None)
    if bool(resume_run_dir) != bool(resume_qualification):
        raise ValueError("--resume-run-dir and --resume-qualification must be supplied together")
    resume_metadata = None
    if resume_run_dir:
        if cfg["arms"] != ["HOPE"]:
            raise ValueError("task-0 resume is currently limited to the corrected HOPE checkpoint arm")
        resume_metadata = _load_resume_inputs(Path(resume_run_dir).resolve(), Path(resume_qualification).resolve(),
            cfg=cfg, current_source_lock_sha256=source_hash,
            current_r2_source_sha256=_sha256(Path(__file__).resolve()),
            training_manifest_sha256=_sha256(pack_path), manifest_source_sha256=view.source_sha256)
    output_dir.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    budget = None
    try:
        import numpy as np
        random.seed(cfg["initialization_seed"])
        np.random.seed(cfg["initialization_seed"])
        import torch
        torch.manual_seed(cfg["initialization_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg["initialization_seed"])
        torch, tokenizer, model, immutable_versions = _load_common(project_root, model_path, cfg["device"], cfg)
        if cfg["device"].startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model_frozen_versions = immutable_versions(model)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(False)
        from clinc_tail_runtime_v8 import clone_mutable_context, capture, tail_logits, causal_token_ce, parameter_counts
        from clinc_lifecycle_v8 import WakeController, SleepTransaction
        import clinc_cil_common_v8 as common_module
        import clinc_lifecycle_v8 as lifecycle_module
        if Path(common_module.__file__).resolve() != task_tools / "clinc_cil_common_v8.py":
            raise RuntimeError("clinc common helper import path mismatch")
        if Path(lifecycle_module.__file__).resolve() != task_tools / "clinc_lifecycle_v8.py":
            raise RuntimeError("clinc lifecycle helper import path mismatch")
        original_logits = model.logits
        budget = Budget(cfg)
        budget.started = started  # Hard wall cap includes local model loading after output creation.
        def counted_logits(tokens, mask=None):
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_logits(tokens, mask)
        model.logits = counted_logits
        import clinc_tail_runtime_v8 as tail_runtime
        original_tail_logits = tail_runtime.tail_logits
        def counted_tail_logits(tail_model, context, cache):
            active = torch.is_grad_enabled() and any(p.requires_grad for p in context.parameters())
            budget.tail(cache, active)
            return original_tail_logits(tail_model, context, cache)
        tail_runtime.tail_logits = counted_tail_logits
        # The wake loop also holds a local imported alias; bind it to the same
        # counter so every tail evaluation is included exactly once.
        tail_logits = counted_tail_logits
        metrics_path = output_dir / "metrics.jsonl"
        receipts = []
        cfg_hash = _canonical_sha({k: v for k, v in cfg.items() if k != "derived_steps_per_epoch"})
        _atomic_json(output_dir / "protocol.json", {"schema": "clinc-cil-v8-pilot-protocol",
            "config": cfg, "config_sha256": cfg_hash, "training_manifest_sha256": _sha256(pack_path),
            "manifest_source_sha256": view.source_sha256, "source_hashes": source_hashes,
            "source_lock_sha256": source_hash, "source_revision": cfg["source_revision"],
            "model_path": str(model_path), "model_revision": cfg["model_revision"],
            "model_assets_sha256": cfg["model_assets_sha256"],
            "full_development_scope": cfg["full_development_scope"],
            "resume": resume_metadata,
            "data": data_note, "method_limitations": [
                "LlamaCMSLM last-MLP augmentation, not native HOPE", "LTI reward is a local teacher/student label-text proxy, not a learned reward model",
                "seen-replay prompts expose old input text but not old gold labels to sleep loss"],
            "eval_clean_rule": "NFKC+casefold+underscore/space normalized exact duplicates against train excluded",
            "eval_selection": "ID-sorted first N development rows per class; no model-score selection",
            "test_access": "none"})
        with metrics_path.open("w", encoding="utf-8", newline="\n", buffering=1) as metrics:
            common_init_sha = None
            for arm in cfg["arms"]:
                context = clone_mutable_context(model)
                arm_init_sha = _mutable_context_sha(context)
                if common_init_sha is None:
                    common_init_sha = arm_init_sha
                elif arm_init_sha != common_init_sha:
                    raise RuntimeError("arms did not receive identical mutable common initialization")
                wake = WakeController(context, periods=cfg["periods"], base_lr=cfg["base_lr"], fast_lr=cfg["fast_lr"])
                wake.prepare_wake()
                # Tokenized prompt pools are text-only and bounded at four/class.
                seen_prompt_rows: dict[int, list[dict]] = {}
                resume_arm = resume_metadata["arms"].get(arm) if resume_metadata else None
                resume_state = None
                if resume_arm is not None:
                    parent_receipt = resume_arm["receipt"]
                    if parent_receipt.get("common_init_sha256") != arm_init_sha:
                        raise ValueError(f"parent common initialization differs for resumed arm {arm}")
                    resume_state = _restore_parent_task0_checkpoint(
                        context, wake, Path(resume_arm["checkpoint_path"]), receipt=parent_receipt,
                        qualification=resume_arm["qualification"], cfg=cfg, torch=torch)
                    wake.prepare_wake()
                    prior_ids = set(task_class_ids(order, 0))
                    prior_rows = [row for row in train_rows if int(row["intent_id"]) in prior_ids]
                    for cls in sorted(prior_ids):
                        seen_prompt_rows[cls] = sorted(
                            (row for row in prior_rows if int(row["intent_id"]) == cls),
                            key=lambda row: row["id"])[:4]
                run_start = time.monotonic()
                arm_work_start = budget.as_dict()
                acquisition_pass = True
                completed_tasks = 1 if resume_arm is not None else 0
                sleep_adam_steps = 0
                run_sleep_events = 0
                wake_steps = 0
                metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "event": "run_start",
                    "common_init_sha256": arm_init_sha,
                    "parameter_counts": parameter_counts(context),
                    "resume_parent_checkpoint_sha256": None if resume_arm is None else resume_state["checkpoint_sha256"],
                    "resume_parent_protocol_sha256": None if resume_metadata is None else resume_metadata["parent_protocol_sha256"],
                    "resume_qualification_sha256": None if resume_metadata is None else resume_metadata["qualification_sha256"],
                    "resume_task0_clean_accuracy": None if resume_arm is None else resume_arm["task0_clean_accuracy"],
                    "resume_parent_work_arm_delta": None if resume_arm is None else parent_receipt.get("work_arm_delta"),
                    "resume_parent_active_seconds": None if resume_arm is None else parent_receipt.get("active_seconds"),
                    "resume_start_task": 1 if resume_arm is not None else 0,
                    "physical_attempt_wake_updates_at_start": 0,
                    "logical_cumulative_wake_ticks_at_start": 0 if resume_arm is None else resume_state["restored_ticks"]}, sort_keys=True) + "\n")
                for task_id in cfg["tasks"]:
                    if resume_arm is not None and task_id < 1:
                        continue
                    task_ids = set(task_class_ids(order, task_id))
                    task_train = [r for r in train_rows if int(r["intent_id"]) in task_ids]
                    by_id = {r["id"]: r for r in task_train}
                    schedule = shuffled_ids(task_train, cfg["seed"], task_id)[:cfg["examples_per_task"]]
                    schedule_rows = [by_id[rid] for rid in schedule]
                    current_pool = []
                    for cls in sorted(task_ids):
                        current_pool.extend(sorted((r for r in task_train if int(r["intent_id"]) == cls),
                                                   key=lambda r: r["id"])[:4])
                    if cfg["sleep_prompt_policy"] == "seen_replay" and task_id:
                        old_pool = [r for cls in sorted(seen_prompt_rows) for r in seen_prompt_rows[cls]]
                        sleep_pool = [row for pair in zip(
                            (old_pool[i % len(old_pool)] for i in range(max(len(old_pool), len(current_pool)))),
                            (current_pool[i % len(current_pool)] for i in range(max(len(old_pool), len(current_pool)))))
                                      for row in pair]
                    else:
                        old_pool, sleep_pool = [], current_pool
                    eval_clean_rows = [r for r in clean_dev if int(r["intent_id"]) in set(order[:(task_id + 1) * 15])]
                    eval_subset = fixed_eval_rows(eval_clean_rows, per_class=cfg["eval_examples_per_class"])
                    baseline = _evaluate(model, context, eval_subset, order, task_id, tokenizer, view, cfg,
                                         torch, cfg["device"], budget)
                    metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": task_id,
                        "event": "task_start_eval_clean_subset", "result": baseline,
                        "training_schedule_ids_sha256": _canonical_sha(schedule),
                        "evaluation_ids_sha256": _canonical_sha([r["id"] for r in eval_subset])}, sort_keys=True) + "\n")
                    if cfg["eval_full_development"]:
                        full_metric = _evaluate(model, context, dev_rows, order, task_id, tokenizer, view, cfg,
                                                torch, cfg["device"], budget, include_unseen=False)
                        metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": task_id,
                            "event": "task_start_eval_full_development", "scope": cfg["full_development_scope"],
                            "result": full_metric}, sort_keys=True) + "\n")
                    task_updates = 0
                    sleep_events = 0
                    loss_rows = []
                    grad_rows = []
                    for step, start in enumerate(range(0, len(schedule_rows), cfg["batch_size"]), 1):
                        budget.check()
                        wake.prepare_wake()
                        rows = schedule_rows[start:start + cfg["batch_size"]]
                        batch_data, ids, mask, labels = _batch(rows, tokenizer, view, cfg, torch, cfg["device"])
                        cache = capture(model, ids, mask, device=cfg["device"])
                        for parameter in context.parameters():
                            parameter.grad = None
                        # Set grad=None before backward so no prior clipped
                        # wake gradient can accumulate into this minibatch.
                        logits = tail_logits(model, context, cache)
                        answer_mask = labels.ne(-100)
                        targets = labels[:, 1:].reshape(-1)
                        shifted = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
                        valid = targets.ne(-100)
                        loss = torch.nn.functional.cross_entropy(shifted[valid], targets[valid])
                        loss.backward()
                        budget.backward(mask * answer_mask.long())
                        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in context.parameters() if p.requires_grad], 1.0,
                                                                   error_if_nonfinite=True)
                        allowed = {n: p.grad.detach().clone() for n, p in context.named_parameters() if p.grad is not None}
                        budget.update()  # Reserve before the fast parameter/pending-state update.
                        wake.record_tick(allowed, token_count=int(answer_mask.sum().item()))
                        for parameter in context.parameters():
                            parameter.grad = None
                        task_updates += 1
                        wake_steps += 1
                        loss_rows.append(float(loss.detach().float().cpu()))
                        grad_rows.append(float(grad_norm.detach().float().cpu()))
                        metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": task_id,
                            "step": step, "event": "wake_update", "ce": loss_rows[-1],
                            "global_grad_norm_before_clip": grad_rows[-1],
                            "answer_tokens": int(answer_mask.sum().item()), "active_bank_slots": [
                                sum(bool(slot.active) for slot in bank.slots) for bank in context.banks()],
                            "last_forward_routes_prompt_plus_answer": [
                                [{"slot": x.get("slot"), "tokens": x.get("tokens"), "kind": x.get("kind")}
                                 for x in bank.last_routes] for bank in context.banks()]}, sort_keys=True) + "\n")
                        for level in sorted(wake.proposals()):
                            # Reacquire after each prior apply/commit so proposal
                            # tensors always match the current context identity.
                            proposal = wake.proposals()[level]
                            if arm in ("HOPE_SLEEP", "EXPAND_CE") and level < len(cfg["periods"]) - 1:
                                tx = SleepTransaction(context, level, proposal, lr=cfg["sleep_lr"],
                                    weight_decay=cfg["sleep_weight_decay"],
                                    clip_norm=cfg["sleep_clip_norm"], seed=cfg["seed"] + task_id * 10000 + step * 100 + level)
                                sleep_rows = sleep_pool
                                if cfg["sleep_prompt_policy"] == "seen_replay" and task_id:
                                    # Fixed 50:50 old/current prompts without old labels in loss.
                                    if len(old_pool) < 1:
                                        raise RuntimeError("seen_replay requested without prior-task prompts")
                                    sleep_rows = [old_pool[(step + j) % len(old_pool)] if j % 2 == 0 else
                                                  current_pool[(step + j) % len(current_pool)]
                                                  for j in range(2 * max(cfg["gkd_calls_per_event"], cfg["lti_calls_per_event"]))]
                                if arm == "HOPE_SLEEP":
                                    event_losses = _run_sleep_event(model, tx, sleep_rows, tokenizer, view, cfg,
                                        torch, cfg["device"], budget, task_id, sleep_events)
                                else:
                                    event_losses = _run_expand_ce_event(model, tx, sleep_rows, tokenizer, view, cfg,
                                        torch, cfg["device"], budget, task_id, sleep_events)
                                sleep_adam_steps += tx.counters["adamw_steps"]
                                new_context = tx.commit()
                                wake.bind_context(new_context)
                                wake.consume(level)
                                context = new_context
                                sleep_events += 1
                                run_sleep_events += 1
                                budget.sleep_events += 1
                                metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": task_id,
                                    "step": step, "event": "sleep_commit", "level": level,
                                    "receiver_slot": tx.slot, "sleep_updates": tx.counters["adamw_steps"],
                                    "losses": event_losses, "teacher_label_use": arm == "EXPAND_CE",
                                    "prompt_policy": cfg["sleep_prompt_policy"]}, sort_keys=True) + "\n")
                            else:
                                wake.apply(level, proposal)
                                context = wake.context
                        task_steps = min(math.ceil(len(schedule_rows) / cfg["batch_size"]), cfg["max_steps_per_task"])
                        if step % cfg["checkpoint_every_steps"] == 0 or step == task_steps:
                            budget.check()
                            path, digest = _save_checkpoint(output_dir, arm, task_id, step, context, wake,
                                                            torch, cfg, _sha256(pack_path), source_hashes)
                            budget.check()
                            metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": task_id,
                                "step": step, "event": "checkpoint", "path": str(path.relative_to(output_dir)),
                                "sha256": digest, "work": budget.as_dict()}, sort_keys=True) + "\n")
                        if step >= cfg["max_steps_per_task"]:
                            break
                    # Preserve at most four text-only replay prompts per class.
                    for cls in sorted(task_ids):
                        seen_prompt_rows[cls] = sorted((r for r in task_train if int(r["intent_id"]) == cls),
                                                       key=lambda r: r["id"])[:4]
                    # Fixed full-clean-development endpoint and separate full-dev report.
                    clean_seen = [r for r in clean_dev if int(r["intent_id"]) in set(order[:(task_id + 1) * 15])]
                    full_clean_eval = _evaluate(model, context, clean_seen, order, task_id, tokenizer, view, cfg,
                                                torch, cfg["device"], budget)
                    metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": task_id,
                        "event": "task_end_eval_clean_full", "result": full_clean_eval,
                        "wake_ticks": wake.ticks, "sleep_events": sleep_events,
                        "wake_loss_last": None,
                        "pending_levels": {str(level): wake.clocks[level] for level in wake.clocks if wake.clocks[level]},
                        "wake_ce_mean": sum(loss_rows) / len(loss_rows) if loss_rows else None,
                        "wake_grad_norm_mean": sum(grad_rows) / len(grad_rows) if grad_rows else None,
                        "wake_update_count_task": task_updates}, sort_keys=True) + "\n")
                    completed_tasks += 1
                    if cfg["eval_full_development"]:
                        full_dev_eval = _evaluate(model, context, dev_rows, order, task_id, tokenizer, view, cfg,
                                                  torch, cfg["device"], budget, include_unseen=False)
                        metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": task_id,
                            "event": "task_end_eval_full_development", "scope": cfg["full_development_scope"],
                            "result": full_dev_eval}, sort_keys=True) + "\n")
                    if task_id == 0:
                        acquisition_pass = (full_clean_eval["metrics"]["current"]["accuracy"] is not None and
                                             full_clean_eval["metrics"]["current"]["accuracy"] >= cfg["task0_acquisition_gate"])
                        metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": task_id,
                            "event": "task0_development_acquisition_gate", "gate": cfg["task0_acquisition_gate"],
                            "passed": acquisition_pass, "current_clean_accuracy": full_clean_eval["metrics"]["current"]["accuracy"]}, sort_keys=True) + "\n")
                        if not acquisition_pass:
                            break
                state = {"schema": "clinc-cil-v8-final-state", "arm": arm, "seed": cfg["seed"],
                         "context": context.state_dict(), "wake": wake.snapshot(), "rng": _rng_state(torch)}
                checkpoint = output_dir / "checkpoints" / arm / "final.pt"
                digest = _atomic_torch(checkpoint, state, torch)
                work_end = budget.as_dict()
                work_delta = {}
                for key, value in work_end.items():
                    start_value = arm_work_start.get(key)
                    if isinstance(value, (int, float)) and isinstance(start_value, (int, float)):
                        work_delta[key] = value - start_value
                    elif isinstance(value, dict):
                        old = arm_work_start.get(key, {})
                        work_delta[key] = {k: v - old.get(k, 0) for k, v in value.items() if isinstance(v, int)}
                all_tasks_done = completed_tasks == len(cfg["tasks"])
                receipt = {"arm": arm, "seed": cfg["seed"],
                    "status": "complete" if all_tasks_done else "stopped_task0_acquisition_gate",
                    "tasks_completed": completed_tasks,
                    "final_checkpoint": str(checkpoint.relative_to(output_dir)), "final_checkpoint_sha256": digest,
                    "ticks": wake.ticks, "pending_levels": {str(k): v for k, v in wake.clocks.items() if v},
                    "sleep_events": work_delta.get("sleep_events", 0), "active_seconds": time.monotonic() - run_start,
                    "work_global_cumulative": work_end, "work_arm_delta": work_delta,
                    "physical_sleep_adamw_steps": sleep_adam_steps,
                    "wake_steps": wake_steps, "common_init_sha256": arm_init_sha,
                    "physical_attempt_wake_updates": wake_steps,
                    "logical_cumulative_wake_ticks": wake.ticks,
                    "resume_parent_checkpoint_sha256": None if resume_arm is None else resume_state["checkpoint_sha256"],
                    "resume_parent_protocol_sha256": None if resume_metadata is None else resume_metadata["parent_protocol_sha256"],
                    "resume_qualification_sha256": None if resume_metadata is None else resume_metadata["qualification_sha256"],
                    "resume_parent_r1_source_lock_sha256": None if resume_metadata is None else resume_metadata["parent_source_lock_sha256"],
                    "resume_task0_clean_accuracy": None if resume_arm is None else resume_arm["task0_clean_accuracy"],
                    "resume_parent_work_arm_delta": None if resume_arm is None else parent_receipt.get("work_arm_delta"),
                    "resume_parent_active_seconds": None if resume_arm is None else parent_receipt.get("active_seconds"),
                    "physical_attempt_start_task": 1 if resume_arm is not None else 0,
                    "parameter_counts": parameter_counts(context),
                    "task0_acquisition_gate_passed": acquisition_pass}
                receipts.append(receipt)
                metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "event": "run_complete",
                                          "receipt": receipt}, sort_keys=True) + "\n")
                metrics.flush()
        final = {"schema": "clinc-cil-v8-pilot-seal",
            "complete": len(receipts) == len(cfg["arms"]) and all(x["tasks_completed"] == len(cfg["tasks"]) for x in receipts),
            "all_requested_runs_terminated": len(receipts) == len(cfg["arms"]),
            "test_data_included": False, "seed": cfg["seed"], "tasks": cfg["tasks"],
            "arms_expected": cfg["arms"], "arms_completed": len(receipts), "runs": receipts,
            "config_sha256": cfg_hash, "training_manifest_sha256": _sha256(pack_path),
            "source_lock_sha256": source_hash, "elapsed_seconds": time.monotonic() - started,
            "work": budget.as_dict(), "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "immutable_base_unchanged": immutable_versions(model) == model_frozen_versions,
            "claim_scope": "development-only seed17 tasks0-1 pilot; no test-set access"}
        if not final["immutable_base_unchanged"]:
            raise RuntimeError("frozen backbone changed during pilot")
        _atomic_json(output_dir / "development-sealed.json", final)
        return final
    except BaseException as exc:
        failure = {"schema": "clinc-cil-v8-pilot-failure", "status": "failed", "error_type": type(exc).__name__,
            "error": str(exc), "elapsed_seconds": time.monotonic() - started,
            "work": budget.as_dict() if budget is not None else None,
            "known_completed_metrics": str(output_dir / "metrics.jsonl"),
            "test_data_included": False}
        _atomic_json(output_dir / "failure.json", failure)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-pack", required=True, help="training-only manifest JSON; no test input accepted")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-run-dir", help="R1 run directory with a qualifying task-0 final checkpoint")
    parser.add_argument("--resume-qualification", help="independently corrected checkpoint-rescore JSON")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"status": "complete", "arms_completed": result["arms_completed"],
                      "output_dir": str(Path(args.output_dir).resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
