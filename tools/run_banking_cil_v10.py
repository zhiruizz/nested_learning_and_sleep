#!/usr/bin/env python3
"""Finite staged Banking77 continual-learning driver (train/dev only).

This is a local frozen-base CMS/additive-receiver study, not native HOPE or a
Sleep-paper reproduction.  It has no official-test reader.  Each invocation
does one auditable phase: metadata preflight, one seed foundation, one
seed/arm continuation, or a read-only nine-run seal check.
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
from collections import defaultdict
from typing import Sequence


SCHEMA = "banking-cil-v10-driver-v1"
PACK_SCHEMA = "banking-cil-train-dev-manifest-v10"
SEEDS = (17, 29, 43)
ARMS = ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")
NUM_CLASSES = 77
TASKS = 7
CLASSES_PER_TASK = 11
PROMPT_HEAD = "Classify the intent of this user request.\nRequest: "
PROMPT_TAIL = "\nIntent:"


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    return value


def _config_sha(cfg):
    return _canonical_sha({k: _plain(v) for k, v in cfg.items() if k != "phase"})


def normalize_prediction(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold().strip().replace("_", " ")
    return " ".join(value.split())


def class_task_map(class_order: Sequence[int]) -> dict[int, int]:
    if len(class_order) != NUM_CLASSES or set(map(int, class_order)) != set(range(NUM_CLASSES)):
        raise ValueError("class order must be a permutation of 0..76")
    return {int(cid): i // CLASSES_PER_TASK for i, cid in enumerate(class_order)}


def task_class_ids(class_order: Sequence[int], task_id: int) -> tuple[int, ...]:
    class_task_map(class_order)
    if type(task_id) is not int or not 0 <= task_id < TASKS:
        raise ValueError("task id must be in 0..6")
    return tuple(int(x) for x in class_order[task_id * CLASSES_PER_TASK:(task_id + 1) * CLASSES_PER_TASK])


def task_rows(rows: Sequence[dict], class_order: Sequence[int], task_id: int) -> list[dict]:
    allowed = set(task_class_ids(class_order, task_id))
    return [row for row in rows if int(row["intent_id"]) in allowed]


def seen_rows(rows: Sequence[dict], class_order: Sequence[int], task_id: int) -> list[dict]:
    allowed = set(class_order[: (task_id + 1) * CLASSES_PER_TASK])
    return [row for row in rows if int(row["intent_id"]) in allowed]


def deterministic_epoch_ids(rows: Sequence[dict], *, seed: int, task_id: int, epoch: int) -> list[str]:
    """Stable permutation independent of Python's process hash randomization."""
    ids = [str(row["id"]) for row in rows]
    random.Random(f"banking-cil-v10:{seed}:{task_id}:{epoch}").shuffle(ids)
    return ids


def make_sleep_request_schedule(old_by_class: dict[int, Sequence[dict]],
                                current_by_class: dict[int, Sequence[dict]],
                                *, event_index: int, task_id: int,
                                method_kinds: Sequence[str]) -> list[tuple[str, dict, str]]:
    """Same deterministic, role/class-cycling requests for Sleep and GOLD.

    The selector uses class/task membership to provide balanced coverage; Sleep
    loss still receives sampled text only and never gold target tokens.
    """
    if type(event_index) is not int or event_index < 0 or not method_kinds:
        raise ValueError("invalid event/method schedule")
    old_classes = sorted(int(x) for x in old_by_class)
    current_classes = sorted(int(x) for x in current_by_class)
    if task_id == 0:
        old_classes = []
    if not current_classes or any(not old_by_class[c] for c in old_classes) or any(not current_by_class[c] for c in current_classes):
        raise ValueError("sleep request role pools must be nonempty for present classes")
    all_roles = [("old", old_classes), ("current", current_classes)]
    serial_start = event_index * len(method_kinds)
    selected = []
    for call, kind in enumerate(method_kinds):
        if kind not in ("gkd", "lti"):
            raise ValueError("method schedule may contain only gkd/lti")
        serial = serial_start + call
        if not old_classes:
            role, classes = "current", current_classes
            within = serial
        else:
            role = "old" if (serial + task_id) % 2 == 0 else "current"
            classes = old_classes if role == "old" else current_classes
            within = serial // 2
        class_id = classes[within % len(classes)]
        pool = old_by_class[class_id] if role == "old" else current_by_class[class_id]
        row_index = (within // len(classes)) % len(pool)
        selected.append((kind, dict(pool[row_index]), role))
    return selected


def sleep_method_schedule(cfg: dict) -> list[str]:
    gkd, lti = int(cfg["gkd_calls_per_event"]), int(cfg["lti_calls_per_event"])
    if gkd != 2 * lti:
        raise ValueError("frozen main event schedule must repeat [GKD,GKD,LTI]")
    return [kind for _ in range(lti) for kind in ("gkd", "gkd", "lti")]


def build_prompt_ids(text: str, tokenizer, classes: Sequence[str], eos_token_id: int,
                     max_sequence_length: int) -> list[int]:
    """Gold-independent segmented prompt with target room for all 77 labels."""
    import banking_cil_common_v10 as common
    item = common.tokenize_prompt(text, tokenizer, classes, eos_token_id, max_sequence_length)
    return [int(x) for x in item["input_ids"][:item["prompt_length"]]]


def tokenized_training_row(row: dict, tokenizer, classes: Sequence[str], eos_token_id: int,
                           max_sequence_length: int) -> dict:
    import banking_cil_common_v10 as common
    item = common.tokenize_row(row, tokenizer, classes, eos_token_id, max_sequence_length)
    return {"ids": [int(x) for x in item["input_ids"]],
            "labels": [int(x) for x in item["labels"]],
            "prompt_length": int(item["prompt_length"]), "intent_id": int(row["intent_id"])}


def validate_config(cfg: dict, *, phase: str | None = None) -> dict:
    required = {
        "schema", "status", "source_revision", "source_hashes", "model_revision", "model_assets_sha256",
        "training_manifest_sha256", "initialization_seed", "seeds", "tasks", "classes_per_task",
        "arms", "rank", "capacity", "levels", "periods", "batch_size", "foundation_epochs",
        "epochs_per_task", "base_lr", "fast_lr", "sleep_lr", "sleep_weight_decay", "sleep_clip_norm",
        "temperature", "max_new_tokens", "lti_edit_distance_z0", "lti_semantic_weight", "lti_edit_weight",
        "gkd_calls_per_event", "lti_calls_per_event", "eval_examples_per_class", "max_sequence_length",
        "checkpoint_every_steps",
        "task0_global_candidate_gate", "task0_class_macro_gate", "foundation_max_wall_seconds",
        "foundation_max_forward_calls", "foundation_max_update_steps", "main_max_wall_seconds",
        "main_max_forward_calls", "main_max_update_steps", "device", "sleep_prompt_policy", "candidate_batch_size",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"config missing required fields: {sorted(missing)}")
    if cfg["schema"] != SCHEMA:
        raise ValueError("config schema mismatch")
    if cfg["status"] not in ("candidate", "frozen"):
        raise ValueError("config status must be candidate or frozen")
    selected_phase = phase or cfg.get("phase")
    if selected_phase in ("foundation", "main", "seal") and cfg["status"] != "frozen":
        raise ValueError("GPU training and sealing require a frozen config")
    if cfg["tasks"] != TASKS or cfg["classes_per_task"] != CLASSES_PER_TASK:
        raise ValueError("this driver is frozen to seven tasks of eleven classes")
    if tuple(cfg["seeds"]) != SEEDS:
        raise ValueError("seeds must be exactly [17,29,43]")
    if cfg["arms"] != list(ARMS):
        raise ValueError("main comparison must include CMS_ONLY, SLEEP_ADDITIVE, GOLD_ADDITIVE")
    if cfg["rank"] != 64 or cfg["levels"] != 5 or cfg["capacity"] != 8:
        raise ValueError("v10 capacity/architecture must be rank64, five levels, capacity8")
    if cfg["periods"] != [16, 64, 256, 1024, None]:
        raise ValueError("periods are locked to minibatches [16,64,256,1024,None]")
    if cfg["batch_size"] != 8 or cfg["foundation_epochs"] != 2 or cfg["epochs_per_task"] != 2:
        raise ValueError("batch8 and two fixed epochs are required by this config")
    if cfg["temperature"] != 0.8 or cfg["lti_edit_distance_z0"] != 8:
        raise ValueError("temperature .8 and token-edit z0=8 are locked for the probe comparison")
    if cfg["max_new_tokens"] < 10:
        raise ValueError("Banking77 longest label plus EOS requires at least 10 generated tokens")
    if cfg["sleep_prompt_policy"] != "balanced_seen_class_cycle":
        raise ValueError("sleep prompts must use the fixed old/current class-cycling selector")
    if not math.isclose(float(cfg["lti_semantic_weight"]), .5) or not math.isclose(float(cfg["lti_edit_weight"]), .5):
        raise ValueError("local LTI proxy weights are fixed .5/.5")
    if type(cfg["initialization_seed"]) is not int:
        raise ValueError("initialization_seed must be explicitly pinned")
    for name in ("base_lr", "fast_lr", "sleep_lr", "sleep_clip_norm"):
        if not math.isfinite(float(cfg[name])) or float(cfg[name]) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(float(cfg["sleep_weight_decay"])) or float(cfg["sleep_weight_decay"]) < 0:
        raise ValueError("sleep_weight_decay must be finite and nonnegative")
    for name in ("gkd_calls_per_event", "lti_calls_per_event", "eval_examples_per_class", "max_sequence_length",
                 "checkpoint_every_steps", "foundation_max_wall_seconds", "foundation_max_forward_calls",
                 "foundation_max_update_steps", "main_max_wall_seconds", "main_max_forward_calls",
                 "main_max_update_steps", "candidate_batch_size"):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["gkd_calls_per_event"] != 2 * cfg["lti_calls_per_event"]:
        raise ValueError("main sleep calls must repeat the fixed [GKD,GKD,LTI] cycle")
    if cfg["device"] not in ("cuda:0", "cuda:1", "cpu"):
        raise ValueError("device must be explicitly cuda:0/cuda:1/cpu")
    if cfg["candidate_batch_size"] > 32:
        raise ValueError("77-way candidate batch size is capped at 32")
    for name in ("task0_global_candidate_gate", "task0_class_macro_gate"):
        if not 0 <= float(cfg[name]) <= 1:
            raise ValueError(f"{name} must lie in [0,1]")
    if not isinstance(cfg["training_manifest_sha256"], str) or len(cfg["training_manifest_sha256"]) != 64:
        raise ValueError("training_manifest_sha256 must be pinned")
    if not isinstance(cfg["source_hashes"], dict) or not cfg["source_hashes"]:
        raise ValueError("source_hashes lock required")
    for name, digest in cfg["source_hashes"].items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("malformed source hash map")
    if not isinstance(cfg["model_assets_sha256"], dict) or not cfg["model_assets_sha256"]:
        raise ValueError("model asset hashes required")
    for name, digest in cfg["model_assets_sha256"].items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("malformed model hash map")
    cfg = dict(cfg)
    cfg["phase"] = phase or cfg.get("phase", "foundation")
    if cfg["phase"] not in ("preflight", "foundation", "main", "seal"):
        raise ValueError("unknown phase")
    return cfg


def _budget_config(cfg, phase):
    prefix = "foundation" if phase == "foundation" else "main"
    return {**cfg, "max_wall_seconds": cfg[f"{prefix}_max_wall_seconds"],
            "max_forward_calls": cfg[f"{prefix}_max_forward_calls"],
            "max_update_steps": cfg[f"{prefix}_max_update_steps"]}


class Budget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.started = time.monotonic()
        self.full_forward_calls = 0
        self.full_forward_tokens = 0
        self.executed_input_tokens = 0
        self.padded_input_tokens = 0
        self.backward_calls = 0
        self.backward_tokens = 0
        self.tail_calls = 0
        self.tail_grad_calls = 0
        self.tail_no_grad_calls = 0
        self.tail_shape_histogram = {}
        self.wake_updates = 0
        self.adamw_updates = 0
        self.update_attempts = 0
        self.generated_tokens = 0
        self.sleep_events = 0
        self.shape_histogram = {}

    def check(self):
        if time.monotonic() - self.started > self.cfg["max_wall_seconds"]:
            raise RuntimeError("hard wall-time budget exceeded")
        if self.full_forward_calls > self.cfg["max_forward_calls"]:
            raise RuntimeError("hard full-backbone forward budget exceeded")
        if self.wake_updates + self.update_attempts > self.cfg["max_update_steps"]:
            raise RuntimeError("hard update-attempt budget exceeded")

    def forward(self, mask):
        self.full_forward_calls += 1
        valid = int(mask.sum().item())
        total = int(mask.numel())
        self.full_forward_tokens += valid
        self.executed_input_tokens += total
        self.padded_input_tokens += total - valid
        shape = f"{mask.shape[0]}x{mask.shape[1]}"
        self.shape_histogram[shape] = self.shape_histogram.get(shape, 0) + 1
        self.check()

    def backward(self, mask):
        self.backward_calls += 1
        self.backward_tokens += int(mask.sum().item())
        self.check()

    def update(self, amount=1):
        if type(amount) is not int or amount < 1:
            raise ValueError("update reservation must be positive integer")
        self.update_attempts += amount
        if self.wake_updates + self.update_attempts > self.cfg["max_update_steps"]:
            raise RuntimeError("hard update-attempt budget exceeded")
        self.check()

    def tail(self, cache, grad_enabled):
        self.tail_calls += 1
        self.tail_grad_calls += int(bool(grad_enabled))
        self.tail_no_grad_calls += int(not bool(grad_enabled))
        shape = f"{cache.y.shape[0]}x{cache.y.shape[1]}"
        self.tail_shape_histogram[shape] = self.tail_shape_histogram.get(shape, 0) + 1

    def as_dict(self):
        return {"full_backbone_forward_calls": self.full_forward_calls,
                "full_backbone_forward_tokens": self.full_forward_tokens,
                "executed_input_tokens_including_padding": self.executed_input_tokens,
                "padded_input_tokens": self.padded_input_tokens,
                "forward_shape_histogram": dict(sorted(self.shape_histogram.items())),
                "tail_backward_calls": self.backward_calls,
                "tail_backward_tokens": self.backward_tokens,
                "tail_calls": self.tail_calls,
                "tail_grad_enabled_calls": self.tail_grad_calls,
                "tail_no_grad_calls": self.tail_no_grad_calls,
                "tail_shape_histogram": dict(sorted(self.tail_shape_histogram.items())),
                "wake_updates": self.wake_updates,
                "physical_adamw_updates": self.adamw_updates,
                "update_attempts": self.update_attempts,
                "sleep_events": self.sleep_events,
                "sampled_output_tokens": self.generated_tokens,
                "active_wall_seconds": time.monotonic() - self.started}


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
        if _sha256_file(path) != hashlib.sha256(payload).hexdigest():
            raise IOError("JSON atomic readback mismatch")
    finally:
        if os.path.exists(name): os.unlink(name)


def _atomic_torch(path: Path, value, torch):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, name)
        with open(name, "r+b") as stream:
            stream.flush(); os.fsync(stream.fileno())
        expected = _sha256_file(name)
        os.replace(name, path)
        if _sha256_file(path) != expected:
            raise IOError("checkpoint atomic readback mismatch")
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if loaded.get("schema") != value.get("schema"):
            raise IOError("checkpoint deserialization readback mismatch")
        return expected
    finally:
        if os.path.exists(name): os.unlink(name)


def _rng_state(torch):
    import numpy as np
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state().clone(),
            "torch_cuda": [x.cpu().clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None}


def _restore_rng(state, torch):
    import numpy as np
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("checkpoint RNG schema mismatch")
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu().clone())
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology mismatch")
        torch.cuda.set_rng_state_all([x.cpu().clone() for x in state["torch_cuda"]])


def _encode(tokenizer, text):
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]


def _batch(rows, tokenizer, classes, eos, pad, max_length, torch, device):
    import banking_cil_common_v10 as common
    batch = common.tokenize_batch(rows, tokenizer, classes, eos, pad, max_length)
    ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
    mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
    labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
    return ids, mask, labels


def _load_rows(manifest_path: Path, expected_manifest_sha: str):
    import banking_cil_common_v10 as common
    if _sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("training pack manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    common.validate_manifest(manifest)
    view = common.make_training_view(manifest)
    train = list(common.iter_role(view, "train", manifest_path.parent))
    development = list(common.iter_role(view, "development", manifest_path.parent))
    partition = common.validate_role_partition(train, development)
    if len(train) != 9229 or len(development) != 770 or partition["rows_by_role"] != {"train": 9229, "development": 770}:
        raise ValueError("unexpected locked Banking train/dev partition")
    return manifest, train, development


def _source_hashes(project_root: Path, task_root: Path):
    source_root = project_root.resolve()
    task_root = task_root.resolve()
    candidates = {p.relative_to(source_root).as_posix(): p
                 for p in sorted((source_root / "src" / "nested_sleep").glob("*.py"))}
    tool_names = [
                   ("run_banking_cil_v10.py", "banking_cil_v10_routing.py", "banking_cil_v10_lifecycle.py",
                    "banking_cil_v10_runtime.py", "banking_cil_common_v10.py",
                    "clinc_lifecycle_v8.py", "clinc_tail_runtime_v8.py")]
    tool_names = list(tool_names[0])
    candidates.update({f"tools/{name}": task_root / "tools" / name for name in tool_names})
    missing = [key for key, p in candidates.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"source files missing: {missing}")
    return {key: _sha256_file(path) for key, path in sorted(candidates.items())}


def _verify_source_lock(cfg, actual):
    if cfg["source_hashes"] != actual:
        expected_keys, actual_keys = set(cfg["source_hashes"]), set(actual)
        raise ValueError(f"source lock mismatch; missing={sorted(actual_keys-expected_keys)}, extra={sorted(expected_keys-actual_keys)}, changed={[k for k in sorted(actual_keys & expected_keys) if actual[k] != cfg['source_hashes'][k]]}")
    if cfg.get("source_revision") is None:
        raise ValueError("source_revision pin is required")
    return _canonical_sha(actual)


def _verify_model_assets(model_dir: Path, hashes: dict):
    for relative, expected in hashes.items():
        path = (model_dir / relative).resolve()
        if model_dir.resolve() not in path.parents or not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"model asset hash mismatch: {relative}")


def _load_model(project_root: Path, model_dir: Path, cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = project_root / "src"
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))
    from nested_sleep.lm import LlamaCMSLM, promote_mutable_to_fp32, immutable_versions
    if source.resolve() not in Path(sys.modules["nested_sleep.lm"].__file__).resolve().parents:
        raise RuntimeError("nested_sleep.lm import escaped the pinned project")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map=cfg["device"], attn_implementation="eager").eval()
    backbone.requires_grad_(False)
    model = LlamaCMSLM(backbone, rank=cfg["rank"], capacity=cfg["capacity"], levels=cfg["levels"])
    promote_mutable_to_fp32(model)
    model.eval()
    return torch, tokenizer, model, immutable_versions


def _verify_loaded_project_modules(project_root: Path):
    source = (project_root / "src").resolve()
    bad = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("nested_sleep"):
            continue
        path = getattr(module, "__file__", None)
        if path and source not in Path(path).resolve().parents:
            bad[name] = str(Path(path).resolve())
    if bad:
        raise RuntimeError(f"nested_sleep modules imported outside pinned project tree: {bad}")


def _install_tail_counter(budget):
    import clinc_tail_runtime_v8 as tail_runtime
    if getattr(tail_runtime, "_banking_v10_counter", False):
        raise RuntimeError("tail counter already installed in this process")
    original = tail_runtime.tail_logits
    def counted(model, context, cache):
        budget.tail(cache, __import__("torch").is_grad_enabled())
        return original(model, context, cache)
    tail_runtime.tail_logits = counted
    tail_runtime._banking_v10_counter = True
    return tail_runtime, original


def _remove_tail_counter(runtime, original):
    if runtime is not None and original is not None:
        runtime.tail_logits = original
        runtime._banking_v10_counter = False


def _init_rng(seed, torch):
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _restore_context_wake(context, wake, state, torch):
    if state.get("schema") != "banking-cil-v10-checkpoint":
        raise ValueError("checkpoint schema mismatch")
    context.load_state_dict(state["context"])
    wake.restore(state["wake"])
    _restore_rng(state["rng"], torch)


def _save_checkpoint(path, *, seed, arm, task_id, step, context, wake, torch, cfg,
                     config_sha, source_lock, manifest_sha, extra=None):
    payload = {"schema": "banking-cil-v10-checkpoint", "seed": seed, "arm": arm,
        "task_id": task_id, "step": step, "context": context.state_dict(), "wake": wake.snapshot(),
        "rng": _rng_state(torch), "config_sha256": config_sha, "source_lock_sha256": source_lock,
        "training_manifest_sha256": manifest_sha, "extra": extra or {}}
    return _atomic_torch(path, payload, torch)


def _restore_checkpoint(path, context, wake, *, expected_sha, seed, arm, cfg, torch):
    if _sha256_file(path) != expected_sha:
        raise ValueError("checkpoint bytes differ from receipt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != "banking-cil-v10-checkpoint" or (state.get("seed"), state.get("arm")) != (seed, arm):
        raise ValueError("checkpoint identity mismatch")
    _restore_context_wake(context, wake, state, torch)
    return state


def _validate_foundation_checkpoint_cpu(path, expected_sha, *, seed, cfg, source_lock, manifest_sha):
    """Inspect the small adapter/wake checkpoint before loading the 3B model."""
    if _sha256_file(path) != expected_sha:
        raise ValueError("foundation checkpoint bytes changed before model load")
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"schema", "seed", "arm", "task_id", "step", "context", "wake", "rng",
                "config_sha256", "source_lock_sha256", "training_manifest_sha256", "extra"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("foundation checkpoint schema mismatch")
    if (state["schema"], state["seed"], state["arm"], state["task_id"],
        state["config_sha256"], state["source_lock_sha256"], state["training_manifest_sha256"]) != (
            "banking-cil-v10-checkpoint", seed, "FOUNDATION_CMS_ONLY", 0,
            _config_sha(cfg), source_lock, manifest_sha):
        raise ValueError("foundation checkpoint identity/binding mismatch before model load")
    if not isinstance(state["step"], int) or state["step"] <= 0:
        raise ValueError("foundation checkpoint cursor missing")
    wake = state["wake"]
    if not hasattr(wake, "ticks") or wake.ticks != state["step"] or not isinstance(state["context"], dict):
        raise ValueError("foundation checkpoint wake cursor/context mismatch")
    if not isinstance(state["rng"], dict) or set(state["rng"]) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("foundation checkpoint RNG schema mismatch")
    wake_state = getattr(wake, "context_state", None)
    if not isinstance(wake_state, dict) or set(wake_state) != set(state["context"]):
        raise ValueError("foundation context/wake snapshot schemas differ")
    if set(state["context"].get("memory", {})) != set(wake_state.get("memory", {})):
        raise ValueError("foundation context/wake memory keys differ")
    for name, value in state["context"].get("memory", {}).items():
        if not torch.equal(value.cpu(), wake_state["memory"][name].cpu()):
            raise ValueError(f"foundation context/wake memory differs: {name}")
    for name in ("readout_a", "readout_b"):
        if not torch.equal(state["context"][name].cpu(), wake_state[name].cpu()):
            raise ValueError(f"foundation context/wake readout differs: {name}")
    return {"step": state["step"], "wake_ticks": wake.ticks,
            "checkpoint_sha256": expected_sha, "cpu_deserialized": True}


def _make_batch(rows, tokenizer, classes, cfg, torch, device):
    return _batch(rows, tokenizer, classes, tokenizer.eos_token_id, tokenizer.pad_token_id,
                  cfg["max_sequence_length"], torch, device)


def _candidate_eval(runtime, model, context, rows, tokenizer, classes, cfg, torch, device, budget):
    import banking_cil_common_v10 as common
    import clinc_tail_runtime_v8 as tail_runtime
    return runtime.score_candidates(model, {"context": context}, rows, tokenizer, classes,
        common, tail_runtime, torch=torch, device=device, max_sequence_length=cfg["max_sequence_length"],
        batch_size=cfg["candidate_batch_size"], budget=budget, include_greedy=False)


def _source_prepare(project_root, task_root, model_path, pack_path, cfg):
    hashes = _source_hashes(project_root, task_root)
    lock_sha = _verify_source_lock(cfg, hashes)
    _verify_model_assets(model_path, cfg["model_assets_sha256"])
    manifest, train, dev = _load_rows(pack_path, cfg["training_manifest_sha256"])
    return hashes, lock_sha, manifest, train, dev


def _preflight(args, cfg):
    project_root = Path(args.project_root).resolve()
    task_root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model_path).resolve()
    pack_path = Path(args.training_pack).resolve()
    if not project_root.is_dir() or not model_path.is_dir() or not pack_path.is_file():
        raise FileNotFoundError("project, model assets, and train/dev pack must exist")
    hashes, source_lock_sha, manifest, train, dev = _source_prepare(project_root, task_root, model_path, pack_path, cfg)
    if cfg.get("model_revision") is None:
        raise ValueError("model_revision pin missing")
    output = Path(args.output_dir).resolve()
    _validate_output_path(output, project_root, model_path)
    info = {"status": "preflight_pass", "phase": cfg["phase"], "source_lock_sha256": source_lock_sha,
        "source_hashes": hashes, "training_manifest_sha256": cfg["training_manifest_sha256"],
        "train_rows": len(train), "development_rows": len(dev), "class_count": len(manifest["classes"]),
        "test_data_included": False, "model_loaded": False, "optimizer_updates": 0}
    if args.receipt:
        receipt = Path(args.receipt).resolve()
        if receipt.exists(): raise FileExistsError(receipt)
        _atomic_json(receipt, info)
    return info


def _validate_output_path(output: Path, project_root: Path, model_path: Path):
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    out, project, model = output.resolve(), project_root.resolve(), model_path.resolve()
    for protected in (project, model):
        if out == protected or protected in out.parents or out in protected.parents:
            raise ValueError("output must be separate from the public project and model asset trees")


def _train_tick(model, context, wake, rows, tokenizer, classes, cfg, torch, device, budget):
    from clinc_tail_runtime_v8 import capture, tail_logits, causal_token_ce
    wake.prepare_wake()
    ids, mask, labels = _make_batch(rows, tokenizer, classes, cfg, torch, device)
    cache = capture(model, ids, mask, device=device)
    logits = tail_logits(model, context, cache)
    loss, target_count = causal_token_ce(logits, ids, labels.ne(-100))
    budget.backward(labels.ne(-100).long())
    context_parameters = dict(context.named_parameters())
    allowed = [p for p in context_parameters.values() if p.requires_grad]
    if not allowed:
        raise RuntimeError("wake step has no trainable parameters")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(allowed, 1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise RuntimeError("wake gradient norm must be positive and finite")
    gradients = {name: parameter.grad.detach().clone() for name, parameter in context_parameters.items()
                 if parameter.requires_grad and parameter.grad is not None}
    for parameter in context_parameters.values(): parameter.grad = None
    wake.record_tick(gradients, token_count=max(1, target_count))
    budget.wake_updates += 1
    budget.check()
    return {"loss": float(loss.detach().float().cpu()), "answer_tokens": target_count,
            "wake_gradient_norm": float(norm.detach().float().cpu())}


def _due_events(model, context, wake, arm, pools, tokenizer, classes, cfg, torch, device, budget, runtime,
                *, event_counts, selector_event_index, seed, task_id, tx_seed, only_level=None):
    due = sorted(wake.proposals())
    if only_level is not None:
        if only_level not in due: raise ValueError("requested event level is not due")
        due = [only_level]
    records = []
    for level in due:
        proposal = wake.proposals()[level]
        if arm == "CMS_ONLY":
            wake.apply(level, proposal)
            records.append({"level": level, "mode": "apply_without_sleep"})
            continue
        event = event_counts[level]
        selected = make_sleep_request_schedule(pools["old"], pools["current"], event_index=selector_event_index,
            task_id=task_id, method_kinds=sleep_method_schedule(cfg))
        from banking_cil_v10_lifecycle import AdditiveSleepTransaction
        tx = AdditiveSleepTransaction(context, level, proposal, lr=cfg["sleep_lr"],
            weight_decay=cfg["sleep_weight_decay"], clip_norm=cfg["sleep_clip_norm"],
            seed=tx_seed + level + event * 1009)
        before_adam = tx.counters["adamw_steps"]
        if arm == "SLEEP_ADDITIVE":
            import banking_cil_common_v10 as common
            import clinc_tail_runtime_v8 as tail_runtime
            result = runtime.run_sleep_event(model, tx, selected, tokenizer, classes, common,
                cfg, torch, device, budget, tail_runtime, sample_tokens=_sample_tokens,
                lti_reward=_lti_reward_adapter, causal_token_kl=tail_runtime.causal_token_kl,
                causal_token_ce=tail_runtime.causal_token_ce)
        elif arm == "GOLD_ADDITIVE":
            import banking_cil_common_v10 as common
            import clinc_tail_runtime_v8 as tail_runtime
            outcomes = []
            for _kind, row, _role in selected:
                def apply_loss(loss, tokens):
                    tx.optimize(loss, "gold", max(1, int(tokens)))
                outcomes.append(runtime.run_gold_replay_step(model, tx.student_context(), row,
                    tokenizer, classes, common, cfg, torch, device, budget, tail_runtime,
                    apply_loss=apply_loss))
            result = {"steps": len(outcomes), "loss_mean": sum(x["loss"] for x in outcomes) / max(1, len(outcomes)),
                      "gold_tokens": sum(x["token_count"] for x in outcomes), "per_step": outcomes}
        else:
            raise ValueError(f"unknown arm {arm}")
        next_context = tx.commit()
        wake.bind_context(next_context)
        wake.consume(level)
        physical = tx.counters["adamw_steps"] - before_adam
        budget.adamw_updates += physical
        budget.sleep_events += 1
        budget.check()
        event_counts[level] += 1
        records.append({"level": level, "event_index": event, "physical_adamw_steps": physical,
                        "global_selector_event_index": selector_event_index,
                        "request_ids_sha256": _canonical_sha([x[1]["id"] for x in selected]),
                        "requests_by_role": {role: sum(item[2] == role for item in selected)
                                              for role in ("old", "current")},
                        "objectives": result, "receiver": tx.slot,
                        "tx_counters": dict(tx.counters)})
        context = wake.context
        selector_event_index += 1
    return context, records, selector_event_index


def _sample_tokens(model, context, prefix_ids, count, temperature, generator, eos, torch, device, budget):
    from clinc_tail_runtime_v8 import sample_with_context
    prefix = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    result = sample_with_context(model, context, prefix, count, generator=generator,
        temperature=temperature, eos_token_id=eos, return_info=True)
    budget.generated_tokens += int(result.actual_generated_tokens)
    answer = result.token_ids[0, result.prefix_length:].tolist()
    if eos in answer: answer = answer[:answer.index(eos) + 1]
    return result, answer


def _lti_reward_adapter(teacher_text, student_text, *, teacher_tokens, student_tokens, z0,
                        canonical_labels):
    """Use the shared runtime's thresholded, length-normalized Eq. 4 proxy."""
    import banking_cil_v10_runtime as runtime
    return runtime.local_lti_reward(teacher_text, student_text,
        teacher_tokens=teacher_tokens, student_tokens=student_tokens, z0=z0,
        canonical_labels=canonical_labels, semantic_weight=0.5, edit_weight=0.5)


def _save_protocol(output_dir, *, phase, cfg, config_sha, lock_sha, source_hashes, manifest_sha,
                   seed=None, arm=None, parent=None):
    protocol = {"schema": "banking-cil-v10-protocol", "phase": phase, "config": _plain(cfg),
        "config_sha256": config_sha, "source_lock_sha256": lock_sha, "source_hashes": source_hashes,
        "training_manifest_sha256": manifest_sha, "seed": seed, "arm": arm,
        "test_data_included": False, "test_reader_available": False, "parent": parent or {}}
    _atomic_json(output_dir / "protocol.json", protocol)
    return protocol


def _append_jsonl(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _group_sleep_pools(train_rows, class_order, task_id):
    import banking_cil_common_v10 as common
    old_classes = tuple(int(x) for x in class_order[:task_id * CLASSES_PER_TASK])
    current_classes = task_class_ids(class_order, task_id)
    by_class = defaultdict(list)
    for row in train_rows:
        by_class[int(row["intent_id"])].append(row)
    for values in by_class.values():
        values.sort(key=lambda row: str(row["id"]))
    old = {cid: by_class[cid] for cid in old_classes if by_class[cid]}
    current = {cid: by_class[cid] for cid in current_classes if by_class[cid]}
    if task_id > 0 and (set(old) != set(old_classes) or set(current) != set(current_classes)):
        raise ValueError("sleep prompt pool lacks a seen class")
    return {"old": old, "current": current}


def _validate_foundation_release(release_path: Path, cfg, cfg_sha, source_lock_sha,
                                 manifest_sha, seed, foundation_dir: Path):
    if not release_path.is_file():
        raise FileNotFoundError("main release artifact is required before continuation")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema") != "banking-cil-v10-main-release-v1" or release.get("status") != "ready":
        raise ValueError("main release artifact schema/status mismatch")
    for key, expected in (("config_sha256", cfg_sha), ("source_lock_sha256", source_lock_sha),
                          ("training_manifest_sha256", manifest_sha)):
        if release.get(key) != expected:
            raise ValueError(f"main release binding mismatch: {key}")
    foundations = release.get("foundations")
    if not isinstance(foundations, dict) or set(foundations) != {str(x) for x in SEEDS}:
        raise ValueError("release must bind all three qualified foundation receipts")
    for seed_key, record in foundations.items():
        if not isinstance(record, dict) or record.get("status") != "qualified":
            raise ValueError(f"foundation seed {seed_key} is not qualified")
        if record.get("source_lock_sha256") != source_lock_sha or record.get("config_sha256") != cfg_sha:
            raise ValueError(f"foundation seed {seed_key} source/config mismatch")
        if record.get("training_manifest_sha256") != manifest_sha:
            raise ValueError(f"foundation seed {seed_key} data mismatch")
        if not isinstance(record.get("checkpoint_sha256"), str) or len(record["checkpoint_sha256"]) != 64:
            raise ValueError(f"foundation seed {seed_key} checkpoint hash missing")
        if not isinstance(record.get("receipt_sha256"), str) or len(record["receipt_sha256"]) != 64:
            raise ValueError(f"foundation seed {seed_key} receipt hash missing")
        run_dir_value = record.get("run_dir")
        if not isinstance(run_dir_value, str):
            raise ValueError(f"foundation seed {seed_key} run_dir missing")
        run_dir = Path(run_dir_value).resolve()
        receipt_path = run_dir / "foundation-receipt.json"
        if not receipt_path.is_file() or _sha256_file(receipt_path) != record["receipt_sha256"]:
            raise ValueError(f"foundation seed {seed_key} receipt bytes mismatch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("schema"), receipt.get("status"), receipt.get("seed"), receipt.get("arm"),
            receipt.get("task_id"), receipt.get("source_lock_sha256"),
            receipt.get("config_sha256"), receipt.get("training_manifest_sha256"),
            receipt.get("checkpoint_sha256"), receipt.get("model_immutable_versions_unchanged")) != (
                "banking-cil-v10-foundation-receipt", "qualified", int(seed_key),
                "FOUNDATION_CMS_ONLY", 0, source_lock_sha, cfg_sha, manifest_sha,
                record["checkpoint_sha256"], True):
            raise ValueError(f"foundation seed {seed_key} receipt contents mismatch")
        metric = receipt.get("metric")
        gate = receipt.get("gate")
        if (not isinstance(metric, dict) or metric.get("count") != 110 or
                metric.get("classes_evaluated") != CLASSES_PER_TASK or
                not isinstance(metric.get("accuracy"), (int, float)) or
                not isinstance(metric.get("task_class_macro_accuracy"), (int, float))):
            raise ValueError(f"foundation seed {seed_key} has incomplete task-0 gate metrics")
        if (metric["accuracy"] < cfg["task0_global_candidate_gate"] or
                metric["task_class_macro_accuracy"] < cfg["task0_class_macro_gate"]):
            raise ValueError(f"foundation seed {seed_key} does not meet the frozen task-0 gates")
        if (not isinstance(gate, dict) or gate.get("passed") is not True or
                gate.get("global_candidate_sum_accuracy") != cfg["task0_global_candidate_gate"] or
                gate.get("task0_class_macro_accuracy") != cfg["task0_class_macro_gate"]):
            raise ValueError(f"foundation seed {seed_key} receipt gate declaration mismatch")
        cp_path = (run_dir / str(receipt.get("checkpoint", ""))).resolve()
        if run_dir not in cp_path.parents or not cp_path.is_file() or _sha256_file(cp_path) != record["checkpoint_sha256"]:
            raise ValueError(f"foundation seed {seed_key} checkpoint bytes mismatch")
    probe = release.get("transfer_probe")
    if not isinstance(probe, dict) or probe.get("status") != "complete" or probe.get("resume_parity") is not True:
        raise ValueError("the predeclared 3-vs-30 transfer/resume probe must be complete and restore-exact")
    if probe.get("source_lock_sha256") != release.get("probe_source_lock_sha256"):
        raise ValueError("transfer probe source-lock binding missing")
    probe_artifact = probe.get("artifact_path")
    if not isinstance(probe_artifact, str) or not isinstance(probe.get("artifact_sha256"), str):
        raise ValueError("transfer probe artifact path/hash missing")
    probe_path = Path(probe_artifact).resolve()
    if not probe_path.is_file() or _sha256_file(probe_path) != probe["artifact_sha256"]:
        raise ValueError("transfer probe artifact bytes mismatch")
    selected = foundations[str(seed)]
    if selected.get("run_dir") is None:
        raise ValueError("release lacks the selected seed foundation path")
    current_dir = Path(selected["run_dir"]).resolve()
    if current_dir != foundation_dir.resolve():
        raise ValueError("provided foundation directory differs from released foundation path")
    receipt_path = current_dir / "foundation-receipt.json"
    if not receipt_path.is_file() or _sha256_file(receipt_path) != selected["receipt_sha256"]:
        raise ValueError("released foundation receipt bytes changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("checkpoint_sha256") != selected["checkpoint_sha256"] or receipt.get("status") != "qualified":
        raise ValueError("released foundation checkpoint/gate mismatch")
    return release, receipt


def _run_foundation(args, cfg):
    seed = args.seed
    if seed not in SEEDS: raise ValueError("foundation seed must be 17,29,43")
    project_root = Path(args.project_root).resolve(); model_path = Path(args.model_path).resolve()
    pack_path = Path(args.training_pack).resolve(); output_dir = Path(args.output_dir).resolve()
    _validate_output_path(output_dir, project_root, model_path)
    task_root = Path(__file__).resolve().parents[1]
    source_hashes, lock_sha, manifest, train_rows, dev_rows = _source_prepare(project_root, task_root, model_path, pack_path, cfg)
    cfg_sha = _config_sha(cfg)
    _init_rng(cfg["initialization_seed"], __import__("torch"))
    output_dir.mkdir(parents=False)
    started = time.monotonic()
    budget = Budget(_budget_config(cfg, "foundation")); budget.started = started
    arm = "FOUNDATION_CMS_ONLY"
    tail_runtime = original_tail = None
    try:
        torch, tokenizer, model, immutable_versions = _load_model(project_root, model_path, cfg)
        _verify_loaded_project_modules(project_root)
        original_logits = model.logits
        def counted(tokens, mask=None):
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_logits(tokens, mask)
        model.logits = counted
        frozen_versions = immutable_versions(model)
        for _, parameter in model.named_parameters(): parameter.requires_grad_(False)
        from clinc_tail_runtime_v8 import clone_mutable_context, parameter_counts
        tail_runtime, original_tail = _install_tail_counter(budget)
        from banking_cil_v10_lifecycle import AdditiveWakeController
        context = clone_mutable_context(model)
        wake = AdditiveWakeController(context, periods=cfg["periods"], base_lr=cfg["base_lr"], fast_lr=cfg["fast_lr"])
        order = manifest["class_orders"][str(seed)]
        rows = task_rows(train_rows, order, 0)
        metrics = []
        step = 0
        for epoch in range(cfg["foundation_epochs"]):
            by_id = {r["id"]: r for r in rows}
            shuffled = deterministic_epoch_ids(rows, seed=seed, task_id=0, epoch=epoch)
            for start in range(0, len(shuffled), cfg["batch_size"]):
                budget.check()
                chunk = [by_id[x] for x in shuffled[start:start + cfg["batch_size"]]]
                step += 1
                metrics.append(_train_tick(model, wake.context, wake, chunk, tokenizer, manifest["classes"], cfg,
                                           torch, next(model.parameters()).device, budget))
                _atomic_json(output_dir / "progress.json", {"seed": seed, "phase": "foundation",
                    "status": "running", "steps": step, "task_id": 0, "work": budget.as_dict()})
                # Foundation is CMS-only; apply every due absolute proposal and carry other pending clocks.
                for level in sorted(wake.proposals()): wake.apply(level, wake.proposals()[level])
                if step % cfg.get("checkpoint_every_steps", 64) == 0:
                    cp = output_dir / "checkpoints" / f"task00-step{step:06d}.pt"
                    if not cp.exists(): _save_checkpoint(cp, seed=seed, arm=arm, task_id=0, step=step,
                        context=wake.context, wake=wake, torch=torch, cfg=cfg, config_sha=cfg_sha,
                        source_lock=lock_sha, manifest_sha=cfg["training_manifest_sha256"])
        import banking_cil_v10_runtime as runtime
        eval_rows = [r for r in dev_rows if int(r["intent_id"]) in set(task_class_ids(order, 0))]
        scored = _candidate_eval(runtime, model, wake.context, eval_rows, tokenizer, manifest["classes"],
                                 cfg, torch, next(model.parameters()).device, budget)
        metric = _extract_candidate_metric(scored)
        macro = metric["task_class_macro_accuracy"]
        gate_pass = metric["accuracy"] >= cfg["task0_global_candidate_gate"] and macro is not None and macro >= cfg["task0_class_macro_gate"]
        final_path = output_dir / "checkpoints" / "foundation-final.pt"
        cp_sha = _save_checkpoint(final_path, seed=seed, arm=arm, task_id=0, step=step,
            context=wake.context, wake=wake, torch=torch, cfg=cfg, config_sha=cfg_sha,
            source_lock=lock_sha, manifest_sha=cfg["training_manifest_sha256"],
            extra={"foundation_gate": metric, "foundation_gate_pass": gate_pass})
        receipt = {"schema": "banking-cil-v10-foundation-receipt", "status": "qualified" if gate_pass else "failed_gate",
            "seed": seed, "arm": arm, "task_id": 0, "steps": step, "checkpoint": str(final_path.relative_to(output_dir)),
            "checkpoint_sha256": cp_sha, "config_sha256": cfg_sha, "source_lock_sha256": lock_sha,
            "training_manifest_sha256": cfg["training_manifest_sha256"], "metric": metric,
            "gate": {"global_candidate_sum_accuracy": cfg["task0_global_candidate_gate"],
                     "task0_class_macro_accuracy": cfg["task0_class_macro_gate"], "passed": gate_pass},
            "work": budget.as_dict(), "test_data_included": False,
            "model_immutable_versions_unchanged": immutable_versions(model) == frozen_versions,
            "mutable_parameter_counts": parameter_counts(wake.context)}
        if receipt["model_immutable_versions_unchanged"] is not True:
            raise RuntimeError("frozen model changed during foundation")
        _atomic_json(output_dir / "foundation-receipt.json", receipt)
        _atomic_json(output_dir / "progress.json", {"status": receipt["status"], "steps": step, "work": budget.as_dict()})
        _remove_tail_counter(tail_runtime, original_tail)
        return receipt
    except Exception as exc:
        _remove_tail_counter(tail_runtime, original_tail)
        _atomic_json(output_dir / "failure.json", {"phase": "foundation", "seed": seed,
            "error_type": type(exc).__name__, "error": str(exc), "work": budget.as_dict(),
            "test_data_included": False})
        raise


def _extract_candidate_metric(result, context_index=0):
    if not isinstance(result, dict) or result.get("schema") != "banking-cil-v10-runtime-v1":
        raise ValueError("candidate scorer schema mismatch")
    names = list(result.get("candidate_metrics", {}))
    if not 0 <= context_index < len(names):
        raise ValueError("candidate scorer omitted the requested context")
    metrics = result["candidate_metrics"][names[context_index]]
    primary = metrics.get("teacher_forced_sum")
    if not isinstance(primary, dict) or not isinstance(primary.get("all77"), dict):
        raise ValueError("candidate scorer omitted global all-77 SUM metrics")
    all_metric = primary["all77"]
    count, correct = int(all_metric.get("count", 0)), int(all_metric.get("correct", -1))
    if count < 1 or not 0 <= correct <= count:
        raise ValueError("invalid all-77 candidate counts")
    per_class = primary.get("per_class")
    if not isinstance(per_class, dict):
        raise ValueError("candidate scorer omitted per-class counts")
    active = []
    for item in per_class.values():
        n = int(item.get("count", 0))
        if n:
            c = int(item.get("correct", -1))
            if not 0 <= c <= n:
                raise ValueError("invalid per-class candidate counts")
            active.append(c / n)
    gold_table = metrics.get("gold_label_token_mean_nll")
    if not isinstance(gold_table, dict) or not isinstance(gold_table.get("all77"), dict):
        raise ValueError("candidate scorer omitted global gold-label NLL")
    return {"count": count, "correct": correct, "accuracy": correct / count,
            "task_class_macro_accuracy": sum(active) / len(active) if active else None,
            "classes_evaluated": len(active), "gold_label_token_mean_nll": gold_table["all77"]}


def _load_foundation_ref(foundation_dir: Path, seed: int, cfg, config_sha, lock_sha, torch):
    receipt_path = foundation_dir / "foundation-receipt.json"
    if not receipt_path.is_file(): raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "qualified" or receipt.get("seed") != seed:
        raise ValueError("foundation is not qualified for this seed")
    if receipt.get("training_manifest_sha256") != cfg["training_manifest_sha256"]:
        raise ValueError("foundation data lock mismatch")
    if receipt.get("source_lock_sha256") != lock_sha:
        raise ValueError("foundation source lock mismatch")
    cp = (foundation_dir / receipt["checkpoint"]).resolve()
    if foundation_dir.resolve() not in cp.parents or not cp.is_file() or _sha256_file(cp) != receipt.get("checkpoint_sha256"):
        raise ValueError("foundation checkpoint is absent or changed")
    return receipt, cp


def _run_main(args, cfg):
    if args.seed not in SEEDS or args.arm not in ARMS:
        raise ValueError("main phase requires a seed in {17,29,43} and one frozen arm")
    if args.foundation_dir is None or args.release_artifact is None:
        raise ValueError("main phase requires --foundation-dir and --release-artifact")
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    pack_path = Path(args.training_pack).resolve()
    output_dir = Path(args.output_dir).resolve()
    _validate_output_path(output_dir, project_root, model_path)
    task_root = Path(__file__).resolve().parents[1]
    source_hashes, lock_sha, manifest, train_rows, dev_rows = _source_prepare(
        project_root, task_root, model_path, pack_path, cfg)
    cfg_sha = _config_sha(cfg)
    manifest_sha = cfg["training_manifest_sha256"]
    release, foundation_receipt = _validate_foundation_release(
        Path(args.release_artifact).resolve(), cfg, cfg_sha, lock_sha, manifest_sha,
        args.seed, Path(args.foundation_dir).resolve())
    if foundation_receipt["seed"] != args.seed:
        raise ValueError("foundation seed mismatch")
    if args.arm not in ARMS:
        raise ValueError("arm not in frozen 3-arm grid")
    foundation_cp = (Path(args.foundation_dir).resolve() / foundation_receipt["checkpoint"]).resolve()
    cpu_checkpoint_preflight = _validate_foundation_checkpoint_cpu(
        foundation_cp, foundation_receipt["checkpoint_sha256"], seed=args.seed, cfg=cfg,
        source_lock=lock_sha, manifest_sha=manifest_sha)

    phase_cfg = _budget_config(cfg, "main")
    budget = Budget(phase_cfg)
    output_dir.mkdir(parents=False, exist_ok=False)
    _save_protocol(output_dir, phase="main", cfg=cfg, config_sha=cfg_sha, lock_sha=lock_sha,
                   source_hashes=source_hashes, manifest_sha=manifest_sha, seed=args.seed,
                   arm=args.arm, parent={"foundation_receipt_sha256": _sha256_file(Path(args.foundation_dir) / "foundation-receipt.json"),
                                         "release_artifact_sha256": _sha256_file(args.release_artifact),
                                         "foundation_checkpoint_sha256": foundation_receipt["checkpoint_sha256"],
                                         "cpu_checkpoint_preflight": cpu_checkpoint_preflight})
    metrics_path = output_dir / "metrics.jsonl"
    started = time.monotonic()
    budget.started = started
    tail_runtime = original_tail = None
    try:
        _init_rng(cfg["initialization_seed"], __import__("torch"))
        torch, tokenizer, model, immutable_versions = _load_model(project_root, model_path, cfg)
        _verify_loaded_project_modules(project_root)
        if cfg["device"].startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        original_logits = model.logits
        def counted(tokens, mask=None):
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_logits(tokens, mask)
        model.logits = counted
        frozen_versions = immutable_versions(model)
        for _, parameter in model.named_parameters():
            parameter.requires_grad_(False)
        import clinc_tail_runtime_v8 as tail_runtime
        tail_runtime, original_tail = _install_tail_counter(budget)
        from banking_cil_v10_lifecycle import AdditiveWakeController
        context = tail_runtime.clone_mutable_context(model)
        wake = AdditiveWakeController(context, periods=cfg["periods"], base_lr=cfg["base_lr"],
                                      fast_lr=cfg["fast_lr"])
        foundation_state = _restore_checkpoint(foundation_cp, context, wake,
            expected_sha=foundation_receipt["checkpoint_sha256"], seed=args.seed,
            arm="FOUNDATION_CMS_ONLY", cfg=cfg, torch=torch)
        if (foundation_state.get("config_sha256"), foundation_state.get("source_lock_sha256"),
            foundation_state.get("training_manifest_sha256"), foundation_state.get("task_id")) != (
                cfg_sha, lock_sha, manifest_sha, 0):
            raise ValueError("foundation checkpoint protocol binding/cursor mismatch")
        task_totals = {0: int(foundation_state["step"])}
        if int(foundation_state["step"]) != wake.ticks:
            raise ValueError("foundation checkpoint step and restored wake cursor differ")
        order = tuple(int(x) for x in manifest["class_orders"][str(args.seed)])
        event_counts = {i: 0 for i, period in enumerate(cfg["periods"]) if period is not None}
        selector_event_index = 0
        task_metrics = []
        event_records = []
        step_metrics = 0
        run_context = wake.context
        for task_id in range(1, TASKS):
            current_rows = task_rows(train_rows, order, task_id)
            by_id = {str(row["id"]): row for row in current_rows}
            pools = _group_sleep_pools(train_rows, order, task_id)
            task_step = 0
            for epoch in range(cfg["epochs_per_task"]):
                ids = deterministic_epoch_ids(current_rows, seed=args.seed, task_id=task_id, epoch=epoch)
                for start in range(0, len(ids), cfg["batch_size"]):
                    budget.check()
                    rows = [by_id[row_id] for row_id in ids[start:start + cfg["batch_size"]]]
                    tick_metrics = _train_tick(model, wake.context, wake, rows, tokenizer,
                        manifest["classes"], cfg, torch, next(model.parameters()).device, budget)
                    task_step += 1
                    step_metrics += 1
                    task_totals[task_id] = task_totals.get(task_id, 0) + 1
                    _append_jsonl(metrics_path, {"event": "wake_tick", "seed": args.seed,
                        "arm": args.arm, "task_id": task_id, "task_step": task_step,
                        "global_wake_tick": wake.ticks, **tick_metrics})
                    due = sorted(wake.proposals())
                    for level in due:
                        if args.arm != "CMS_ONLY":
                            event = event_counts[level]
                            pre_path = output_dir / "checkpoints" / (
                                f"pre-sleep-task{task_id:02d}-tick{wake.ticks:06d}-event{selector_event_index:05d}-level{level}.pt")
                            pre_sha = _save_checkpoint(pre_path, seed=args.seed, arm=args.arm, task_id=task_id,
                                step=wake.ticks, context=wake.context, wake=wake, torch=torch, cfg=cfg,
                                config_sha=cfg_sha, source_lock=lock_sha, manifest_sha=manifest_sha,
                                extra={"phase": "before_sleep", "level": level,
                                       "event_index": event, "global_selector_event_index": selector_event_index})
                        run_context, records, selector_event_index = _due_events(
                            model, wake.context, wake, args.arm, pools, tokenizer, manifest["classes"],
                            cfg, torch, next(model.parameters()).device, budget,
                            __import__("banking_cil_v10_runtime"), event_counts=event_counts,
                            selector_event_index=selector_event_index, seed=args.seed, task_id=task_id,
                            tx_seed=cfg["initialization_seed"] + args.seed * 100003 + task_id * 1009,
                            only_level=level)
                        event_records.extend(records)
                        for record in records:
                            _append_jsonl(metrics_path, {"event": "sleep_or_boundary", "seed": args.seed,
                                "arm": args.arm, "task_id": task_id, "global_wake_tick": wake.ticks,
                                **record})
                            if args.arm != "CMS_ONLY":
                                post_path = output_dir / "checkpoints" / (
                                    f"post-sleep-task{task_id:02d}-tick{wake.ticks:06d}-event{record['global_selector_event_index']:05d}-level{record['level']}.pt")
                                post_sha = _save_checkpoint(post_path, seed=args.seed, arm=args.arm, task_id=task_id,
                                    step=wake.ticks, context=wake.context, wake=wake, torch=torch, cfg=cfg,
                                    config_sha=cfg_sha, source_lock=lock_sha, manifest_sha=manifest_sha,
                                    extra={"phase": "after_sleep_commit", "event": record})
                                record["post_checkpoint_sha256"] = post_sha
                    if step_metrics % cfg["checkpoint_every_steps"] == 0:
                        cp = output_dir / "checkpoints" / f"task{task_id:02d}-step{wake.ticks:06d}.pt"
                        if not cp.exists():
                            _save_checkpoint(cp, seed=args.seed, arm=args.arm, task_id=task_id,
                                step=wake.ticks, context=wake.context, wake=wake, torch=torch, cfg=cfg,
                                config_sha=cfg_sha, source_lock=lock_sha, manifest_sha=manifest_sha,
                                extra={"event_counts": dict(event_counts),
                                       "selector_event_index": selector_event_index})
                    if step_metrics % 16 == 0:
                        _atomic_json(output_dir / "progress.json", {"status": "running", "seed": args.seed,
                            "arm": args.arm, "task_id": task_id, "task_step": task_step,
                            "global_wake_tick": wake.ticks, "event_counts": dict(event_counts),
                            "selector_event_index": selector_event_index, "work": budget.as_dict()})
            seen = set(order[:(task_id + 1) * CLASSES_PER_TASK])
            eval_rows = []
            grouped_dev = defaultdict(list)
            for row in dev_rows:
                cid = int(row["intent_id"])
                if cid in seen:
                    grouped_dev[cid].append(row)
            for cid in sorted(grouped_dev):
                eval_rows.extend(sorted(grouped_dev[cid], key=lambda r: str(r["id"]))[:cfg["eval_examples_per_class"]])
            import banking_cil_v10_runtime as runtime
            eval_result = _candidate_eval(runtime, model, wake.context, eval_rows, tokenizer,
                manifest["classes"], cfg, torch, next(model.parameters()).device, budget)
            summary = _extract_candidate_metric(eval_result)
            task_record = {"task_id": task_id, "global_wake_tick": wake.ticks,
                           "evaluation_rows": len(eval_rows), "candidate_metrics": summary,
                           "evaluation_scope": "seen_class_fixed_id_sorted_sample",
                           "event_counts": dict(event_counts), "work": budget.as_dict()}
            task_metrics.append(task_record)
            _append_jsonl(metrics_path, {"event": "task_endpoint", "seed": args.seed,
                                         "arm": args.arm, **task_record})
            task_cp = output_dir / "checkpoints" / f"task{task_id:02d}-final.pt"
            task_cp_sha = _save_checkpoint(task_cp, seed=args.seed, arm=args.arm, task_id=task_id,
                step=wake.ticks, context=wake.context, wake=wake, torch=torch, cfg=cfg,
                config_sha=cfg_sha, source_lock=lock_sha, manifest_sha=manifest_sha,
                extra={"task_metrics": task_record, "event_counts": dict(event_counts),
                       "selector_event_index": selector_event_index})
            task_record["checkpoint_sha256"] = task_cp_sha
        # Fixed full-development endpoint: every 77-class label is a candidate, no task mask.
        import banking_cil_v10_runtime as runtime
        endpoint = _candidate_eval(runtime, model, wake.context, dev_rows, tokenizer,
            manifest["classes"], cfg, torch, next(model.parameters()).device, budget)
        endpoint_summary = _extract_candidate_metric(endpoint)
        final_cp = output_dir / "checkpoints" / "final-task06.pt"
        final_cp_sha = _save_checkpoint(final_cp, seed=args.seed, arm=args.arm, task_id=6,
            step=wake.ticks, context=wake.context, wake=wake, torch=torch, cfg=cfg,
            config_sha=cfg_sha, source_lock=lock_sha, manifest_sha=manifest_sha,
            extra={"endpoint_metrics": endpoint_summary, "task_metrics": task_metrics,
                   "event_counts": dict(event_counts), "selector_event_index": selector_event_index})
        unchanged = immutable_versions(model) == frozen_versions
        if not unchanged:
            raise RuntimeError("frozen backbone changed during continuation")
        receipt = {"schema": "banking-cil-v10-run-receipt", "status": "complete",
            "seed": args.seed, "arm": args.arm, "tasks_completed": TASKS,
            "foundation_checkpoint_sha256": foundation_receipt["checkpoint_sha256"],
            "final_checkpoint": str(final_cp.relative_to(output_dir)),
            "final_checkpoint_sha256": final_cp_sha, "config_sha256": cfg_sha,
            "source_lock_sha256": lock_sha, "training_manifest_sha256": manifest_sha,
            "task_metrics": task_metrics, "endpoint_metrics": endpoint_summary,
            "wake_ticks": wake.ticks, "event_counts": dict(event_counts),
            "selector_event_index": selector_event_index, "work": budget.as_dict(),
            "model_immutable_versions_unchanged": unchanged, "test_data_included": False}
        _atomic_json(output_dir / "run-receipt.json", receipt)
        receipt["run_receipt_sha256"] = _sha256_file(output_dir / "run-receipt.json")
        _atomic_json(output_dir / "progress.json", {"status": "complete", "seed": args.seed,
            "arm": args.arm, "tasks_completed": TASKS, "work": budget.as_dict()})
        _remove_tail_counter(tail_runtime, original_tail)
        return receipt
    except Exception as exc:
        _remove_tail_counter(tail_runtime, original_tail)
        _atomic_json(output_dir / "failure.json", {"schema": "banking-cil-v10-failure",
            "phase": "main", "seed": args.seed, "arm": args.arm,
            "error_type": type(exc).__name__, "error": str(exc), "work": budget.as_dict(),
            "completed_task_endpoints": task_metrics if "task_metrics" in locals() else [],
            "event_counts": event_counts if "event_counts" in locals() else {},
            "test_data_included": False})
        raise


def _nested_state_equal(left, right, torch):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_nested_state_equal(left[k], right[k], torch) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_state_equal(a, b, torch) for a, b in zip(left, right))
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    return left == right


def _validate_terminal_checkpoint_cpu(path, expected_sha, *, seed, arm, cfg_sha, lock_sha,
                                      manifest_sha, periods, torch):
    if _sha256_file(path) != expected_sha:
        raise ValueError("terminal checkpoint bytes changed before seal CPU validation")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or set(state) != {
            "schema", "seed", "arm", "task_id", "step", "context", "wake", "rng",
            "config_sha256", "source_lock_sha256", "training_manifest_sha256", "extra"}:
        raise ValueError("terminal checkpoint schema mismatch")
    if (state["schema"], state["seed"], state["arm"], state["task_id"],
        state["config_sha256"], state["source_lock_sha256"], state["training_manifest_sha256"]) != (
            "banking-cil-v10-checkpoint", seed, arm, 6, cfg_sha, lock_sha, manifest_sha):
        raise ValueError("terminal checkpoint identity/binding mismatch")
    wake = state["wake"]
    wake_fields = ("context_state", "pending", "clocks", "token_counts", "ticks", "periods",
                   "base_lr", "fast_lr", "bank_states")
    if not all(hasattr(wake, name) for name in wake_fields):
        raise ValueError("terminal wake snapshot schema mismatch")
    if type(state["step"]) is not int or state["step"] < 1 or wake.ticks != state["step"]:
        raise ValueError("terminal checkpoint step/wake cursor mismatch")
    if tuple(wake.periods) != tuple(periods):
        raise ValueError("terminal wake periods mismatch")
    if not isinstance(state["rng"], dict) or set(state["rng"]) != {
            "python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("terminal checkpoint RNG schema mismatch")
    if not isinstance(state["context"], dict) or not _nested_state_equal(
            state["context"], wake.context_state, torch):
        raise ValueError("terminal context and wake snapshots are not identical")
    if _sha256_file(path) != expected_sha:
        raise ValueError("terminal checkpoint changed during CPU validation")
    return {"checkpoint_sha256": expected_sha, "task_id": state["task_id"],
            "step": state["step"], "wake_ticks": wake.ticks, "cpu_deserialized": True}


def _seal(args, cfg):
    if not args.run_dir or len(args.run_dir) != 9:
        raise ValueError("seal phase requires exactly nine --run-dir inputs")
    project_root = Path(args.project_root).resolve(); model_path = Path(args.model_path).resolve()
    pack_path = Path(args.training_pack).resolve(); output_dir = Path(args.output_dir).resolve()
    _validate_output_path(output_dir, project_root, model_path)
    task_root = Path(__file__).resolve().parents[1]
    source_hashes, lock_sha, manifest, _train_rows, _dev_rows = _source_prepare(
        project_root, task_root, model_path, pack_path, cfg)
    cfg_sha = _config_sha(cfg); manifest_sha = cfg["training_manifest_sha256"]
    if args.release_artifact is None:
        raise ValueError("seal phase requires the source-bound --release-artifact")
    release_path = Path(args.release_artifact).resolve()
    release_sha = _sha256_file(release_path)
    preliminary = json.loads(release_path.read_text(encoding="utf-8"))
    first_foundation = preliminary.get("foundations", {}).get(str(SEEDS[0]), {})
    if not isinstance(first_foundation, dict) or not isinstance(first_foundation.get("run_dir"), str):
        raise ValueError("release artifact omits foundation path for the first seed")
    release, _first_receipt = _validate_foundation_release(
        release_path, cfg, cfg_sha, lock_sha, manifest_sha, SEEDS[0], Path(first_foundation["run_dir"]))
    released_foundations = {int(seed): {"receipt_sha256": record["receipt_sha256"],
        "checkpoint_sha256": record["checkpoint_sha256"], "run_dir": str(Path(record["run_dir"]).resolve())}
        for seed, record in release["foundations"].items()}
    seen = set(); run_rows = []
    terminal_payloads = []
    for value in args.run_dir:
        run_dir = Path(value).resolve()
        if not run_dir.is_dir(): raise FileNotFoundError(run_dir)
        protocol_path = run_dir / "protocol.json"
        receipt_path = run_dir / "run-receipt.json"
        if not protocol_path.is_file() or not receipt_path.is_file():
            raise ValueError(f"main run is not sealed: {run_dir}")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        key = (receipt.get("seed"), receipt.get("arm"))
        if key in seen or receipt.get("status") != "complete" or receipt.get("tasks_completed") != TASKS:
            raise ValueError("duplicate or incomplete seed/arm run")
        seen.add(key)
        if key[0] not in released_foundations:
            raise ValueError("main run seed has no released foundation")
        if (receipt.get("config_sha256"), receipt.get("source_lock_sha256"), receipt.get("training_manifest_sha256")) != (
            cfg_sha, lock_sha, manifest_sha):
            raise ValueError("main run source/config/data binding mismatch")
        foundation = released_foundations[key[0]]
        parent = protocol.get("parent")
        if (protocol.get("schema"), protocol.get("phase"), protocol.get("seed"), protocol.get("arm"),
            protocol.get("config_sha256"), protocol.get("training_manifest_sha256"),
            protocol.get("source_hashes"), protocol.get("source_lock_sha256")) != (
                "banking-cil-v10-protocol", "main", key[0], key[1], cfg_sha, manifest_sha, source_hashes, lock_sha):
            raise ValueError("main run protocol identity/source binding mismatch")
        if (not isinstance(parent, dict) or parent.get("foundation_receipt_sha256") != foundation["receipt_sha256"] or
                parent.get("foundation_checkpoint_sha256") != foundation["checkpoint_sha256"] or
                parent.get("release_artifact_sha256") != release_sha):
            raise ValueError("main protocol parent does not match the released foundation")
        if (receipt.get("foundation_checkpoint_sha256") != foundation["checkpoint_sha256"] or
                receipt.get("model_immutable_versions_unchanged") is not True or
                receipt.get("test_data_included") is not False):
            raise ValueError("main run receipt foundation/model/test binding mismatch")
        task_records = receipt.get("task_metrics")
        if not isinstance(task_records, list) or [x.get("task_id") for x in task_records] != list(range(1, TASKS)):
            raise ValueError("main receipt must contain unique task endpoints 1 through 6 in order")
        for task_record in task_records:
            expected_rows = int(cfg["eval_examples_per_class"]) * CLASSES_PER_TASK * (int(task_record["task_id"]) + 1)
            candidate = task_record.get("candidate_metrics")
            if (task_record.get("evaluation_rows") != expected_rows or not isinstance(candidate, dict) or
                    candidate.get("count") != expected_rows or
                    candidate.get("classes_evaluated") != CLASSES_PER_TASK * (int(task_record["task_id"]) + 1)):
                raise ValueError("main task endpoint candidate counts do not match the seen-class protocol")
            task_id = int(task_record["task_id"])
            task_cp = run_dir / "checkpoints" / f"task{task_id:02d}-final.pt"
            task_sha = task_record.get("checkpoint_sha256")
            if (not isinstance(task_sha, str) or len(task_sha) != 64 or not task_cp.is_file() or
                    _sha256_file(task_cp) != task_sha):
                raise ValueError("main task endpoint checkpoint bytes do not match receipt")
        cp_rel = Path(receipt["final_checkpoint"])
        cp = (run_dir / cp_rel).resolve()
        if run_dir not in cp.parents or not cp.is_file() or _sha256_file(cp) != receipt.get("final_checkpoint_sha256"):
            raise ValueError("main final checkpoint bytes mismatch")
        endpoint = receipt.get("endpoint_metrics")
        if not isinstance(endpoint, dict) or endpoint.get("count") != len(_dev_rows) or endpoint.get("classes_evaluated") != NUM_CLASSES:
            raise ValueError("main final endpoint must cover all 770 development rows and all 77 classes")
        terminal_payloads.append((cp, receipt["final_checkpoint_sha256"], key[0], key[1]))
        rec_sha = _sha256_file(receipt_path)
        run_rows.append({"seed": key[0], "arm": key[1], "status": receipt["status"],
            "tasks_completed": receipt["tasks_completed"], "run_dir": str(run_dir),
            "final_checkpoint": cp_rel.as_posix(), "final_checkpoint_sha256": receipt["final_checkpoint_sha256"],
            "run_receipt_sha256": rec_sha, "config_sha256": cfg_sha,
            "source_lock_sha256": lock_sha, "training_manifest_sha256": manifest_sha,
            "endpoint_metrics": receipt["endpoint_metrics"], "work": receipt["work"]})
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if seen != expected:
        raise ValueError(f"seal requires the exact 3x3 seed/arm grid; got {sorted(seen)}")
    import torch
    terminal_audit = [_validate_terminal_checkpoint_cpu(cp, sha, seed=seed, arm=arm,
        cfg_sha=cfg_sha, lock_sha=lock_sha, manifest_sha=manifest_sha,
        periods=cfg["periods"], torch=torch)
        for cp, sha, seed, arm in terminal_payloads]
    run_rows.sort(key=lambda x: (SEEDS.index(int(x["seed"])), ARMS.index(x["arm"])))
    seal = {"schema": "banking-cil-v10-training-seal-v1", "complete": True,
        "seeds": list(SEEDS), "arms": list(ARMS), "runs": run_rows,
        "config_sha256": cfg_sha, "source_lock_sha256": lock_sha,
        "training_manifest_sha256": manifest_sha, "test_data_included": False,
        "official_test_read": False, "release_artifact_sha256": release_sha,
        "terminal_checkpoint_cpu_preflight": terminal_audit}
    output_dir.mkdir(parents=False, exist_ok=False)
    _atomic_json(output_dir / "training-seal.json", seal)
    return seal


def run(args):
    cfg = validate_config(json.loads(Path(args.config).read_text(encoding="utf-8")), phase=args.phase)
    if args.phase == "preflight": return _preflight(args, cfg)
    if args.phase == "foundation": return _run_foundation(args, cfg)
    if args.phase == "main": return _run_main(args, cfg)
    if args.phase == "seal": return _seal(args, cfg)
    raise ValueError(args.phase)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "foundation", "main", "seal"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-pack", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--foundation-dir", type=Path)
    parser.add_argument("--release-artifact", type=Path)
    parser.add_argument("--run-dir", action="append", type=Path,
                        help="repeat exactly nine times for --phase seal")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
