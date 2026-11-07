"""Development-only CLINC label-readout diagnostic (not the v8 primary metric).

This module intentionally separates pure sequence/scoring helpers from the
optional model runner.  The latter is only entered with explicit local paths;
there is no test reader, checkpoint discovery, training, or retry path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCHEMA = "clinc-label-readout-diagnostic-v9"
MAX_FW = 10_000
MAX_SECONDS = 1_800.0
MAX_UPDATES = 0
CONTEXT_NAMES = ("FROZEN_BASE", "TASK0_FOUNDATION", "HOPE_TASK1", "SLEEP_TASK1")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_candidate_sequences(prompt_ids: Sequence[int], label_token_ids: Sequence[Sequence[int]],
                              eos_token_id: int) -> list[list[int]]:
    """Make prefix + canonical label + EOS candidates without using row labels."""
    prefix = [int(x) for x in prompt_ids]
    if not prefix:
        raise ValueError("prompt prefix cannot be empty")
    result = []
    for label in label_token_ids:
        answer = [int(x) for x in label]
        if not answer:
            raise ValueError("candidate label tokenization cannot be empty")
        result.append(prefix + answer + [int(eos_token_id)])
    return result


def teacher_forced_candidate_scores(logits, input_ids, prompt_lengths: Sequence[int],
                                    answer_lengths: Sequence[int]):
    """Return sum and length-mean log-likelihood for answer tokens plus EOS.

    ``logits[:, t]`` predicts ``input_ids[:, t+1]``.  Only targets beginning at
    each prompt boundary are scored; prompt likelihood is excluded. This is
    length-aware and performs no constrained decoding or label masking.
    """
    import torch
    import torch.nn.functional as F

    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != (input_ids.shape[0], input_ids.shape[1] - 1):
        raise ValueError("logits must align with next-token positions in input_ids")
    if len(prompt_lengths) != input_ids.shape[0] or len(answer_lengths) != input_ids.shape[0]:
        raise ValueError("per-row prompt/answer lengths mismatch")
    targets = input_ids[:, 1:].to(dtype=torch.long)
    logp = F.log_softmax(logits.float(), dim=-1)
    selected = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    sums, means = [], []
    width = selected.shape[1]
    for i, (p, n) in enumerate(zip(prompt_lengths, answer_lengths)):
        p, n = int(p), int(n)
        if p < 1 or n < 1 or p + n > input_ids.shape[1]:
            raise ValueError("invalid prompt or answer boundary")
        # target sequence indices p..p+n-1 correspond to logits p-1..p+n-2.
        start, stop = p - 1, p + n - 1
        if stop > width:
            raise ValueError("answer targets exceed logits")
        values = selected[i, start:stop]
        if values.numel() != n:
            raise ValueError("answer mask length mismatch")
        total = values.sum()
        sums.append(total)
        means.append(total / n)
    return torch.stack(sums), torch.stack(means)


def ordered_candidate_predictions(scores, class_names: Sequence[str]) -> list[str]:
    """Argmax with deterministic first-canonical-label tie breaking."""
    if scores.ndim != 2 or scores.shape[1] != len(class_names):
        raise ValueError("score matrix and canonical label count mismatch")
    indices = scores.argmax(dim=1).tolist()
    return [str(class_names[int(i)]) for i in indices]


def fixed_clean_development_rows(train_rows: Iterable[Mapping], dev_rows: Iterable[Mapping],
                                 class_order: Sequence[int], *, per_class: int = 2,
                                 normalize_text=None):
    """Choose stable-ID clean rows for task 0 and task 1, two per class by default."""
    from clinc_cil_common_v8 import task_class_ids
    if per_class != 2:
        raise ValueError("v9 diagnostic cohort is fixed at two rows per class")
    normalize_text = normalize_text or (lambda x: " ".join(str(x).casefold().split()))
    train_text = {normalize_text(r["text"]) for r in train_rows}
    selected = []
    required = set(task_class_ids(class_order, 0)) | set(task_class_ids(class_order, 1))
    buckets = defaultdict(list)
    for row in dev_rows:
        if int(row["intent_id"]) not in required:
            continue
        normalized = normalize_text(row["text"])
        if normalized in train_text:
            continue
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            raise ValueError("development rows require stable IDs")
        buckets[int(row["intent_id"])].append(row)
    for cid in sorted(required):
        candidates = sorted(buckets[cid], key=lambda r: str(r["id"]))
        if len(candidates) < per_class:
            raise ValueError(f"class {cid} has fewer than two clean development rows")
        selected.extend(candidates[:per_class])
    if len(selected) != 60 or len({r["id"] for r in selected}) != 60:
        raise ValueError("fixed diagnostic cohort must contain 60 unique rows")
    return selected


def label_tokenizer_audit(tokenizer, classes: Sequence[str], *, max_sequence_length: int = 128):
    """Aggregate-only audit for candidate labels and prompt truncation policy."""
    from clinc_cil_common_v8 import tokenize_prompt
    lengths = [len(tokenizer.encode(x, add_special_tokens=False)) for x in classes]
    decoded = [tokenizer.decode(tokenizer.encode(x, add_special_tokens=False), skip_special_tokens=True)
               for x in classes]
    roundtrip = [a == b for a, b in zip(decoded, classes)]
    # A synthetic overlong body exercises the common prefix's 128-token truncation.
    probe = tokenize_prompt("x " * (max_sequence_length * 4), tokenizer, classes,
                            tokenizer.eos_token_id, max_sequence_length, prepend_bos=False)
    if len(probe["input_ids"]) > max_sequence_length:
        raise ValueError("common tokenizer exceeded maximum length")
    return {
        "class_count": len(classes), "label_token_length_min": min(lengths),
        "label_token_length_max": max(lengths),
        "labels_at_max_length": sum(n == max(lengths) for n in lengths),
        "roundtrip_exact_count": sum(roundtrip), "roundtrip_mismatch_count": len(classes) - sum(roundtrip),
        "max_candidate_answer_tokens_including_eos": max(lengths) + 1,
        "labels_over_eight_tokens": sum(n > 8 for n in lengths),
        "labels_over_eight_tokens_including_eos": sum(n + 1 > 8 for n in lengths),
        "prepend_bos": False, "prompt_probe_input_tokens": len(probe["input_ids"]),
        "prompt_probe_truncated": len(probe["input_ids"]) == max_sequence_length,
        "would_truncate_prefix_if_129_tokens": max_sequence_length == 128,
        "max_sequence_length": max_sequence_length,
    }


class Budget:
    def __init__(self, max_forwards: int = MAX_FW, max_seconds: float = MAX_SECONDS, *, started=None):
        self.max_forwards, self.max_seconds = int(max_forwards), float(max_seconds)
        self.forwards = self.tail_calls = self.input_tokens = 0
        self.started = time.monotonic() if started is None else float(started)

    def check(self):
        self.check_wall()
        if self.forwards >= self.max_forwards:
            raise RuntimeError("v9 diagnostic hard budget exceeded")

    def check_wall(self):
        if time.monotonic() - self.started > self.max_seconds:
            raise RuntimeError("v9 diagnostic hard wall-clock budget exceeded")

    def capture(self, fn, *args, **kwargs):
        self.check()
        result = fn(*args, **kwargs)
        self.forwards += 1
        self.input_tokens += int(args[1].numel()) if len(args) > 1 else 0
        return result

    def tail(self, fn, *args, **kwargs):
        self.check_wall()
        result = fn(*args, **kwargs)
        self.tail_calls += 1
        return result

    def summary(self):
        return {"backbone_forwards": self.forwards, "tail_calls": self.tail_calls,
                "input_tokens": self.input_tokens,
                "elapsed_seconds": time.monotonic() - self.started,
                "optimizer_updates": 0, "max_backbone_forwards": self.max_forwards,
                "max_elapsed_seconds": self.max_seconds}


def score_sequence_chunk(model, contexts: Mapping[str, object], sequences: Sequence[Sequence[int]],
                         prompt_lengths: Sequence[int], answer_lengths: Sequence[int],
                         *, torch, device, budget: Budget):
    """One frozen-backbone capture shared by context-specific tail rescoring."""
    from clinc_tail_runtime_v8 import capture, tail_logits
    if len(set(len(x) for x in sequences)) != 1:
        raise ValueError("chunk sequences must have equal lengths; group by exact length")
    ids = torch.tensor(sequences, dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    cache = budget.capture(capture, model, ids, mask, device=device)
    results = {}
    for name, context in contexts.items():
        if name == "FROZEN_BASE":
            logits = budget.tail(_base_logits_from_cache, model, cache)
        else:
            logits = budget.tail(tail_logits, model, context, cache)
        sums, means = teacher_forced_candidate_scores(logits[:, :-1, :], ids,
                                                      prompt_lengths, answer_lengths)
        results[name] = {"sum": sums.detach().cpu(), "mean": means.detach().cpu()}
    return results


def _base_logits_from_cache(model, cache):
    """Evaluate the unaugmented Llama final MLP output from the shared capture."""
    layer = model.backbone.model.layers[-1]
    final_norm = model.backbone.model.norm
    head = model.backbone.lm_head
    return head(final_norm(cache.residual + cache.y))


def validate_context_map(document: Mapping, *, master_sha: str, manifest_sha: str,
                         source_lock_sha: str) -> dict[str, dict]:
    """Validate the explicit four-context map before any checkpoint deserialization."""
    if document.get("schema") != "clinc-v9-readout-contexts-v1" or document.get("seed") != 17:
        raise ValueError("context map schema/seed mismatch")
    for key, expected in (("master_config_sha256", master_sha),
                          ("training_manifest_sha256", manifest_sha),
                          ("source_lock_sha256", source_lock_sha)):
        if document.get(key) != expected:
            raise ValueError(f"context map binding mismatch: {key}")
    rows = document.get("contexts")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("context map must contain exactly four contexts")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("name") in result:
            raise ValueError("malformed or duplicate context row")
        name = row["name"]
        if name not in CONTEXT_NAMES:
            raise ValueError(f"unexpected context name: {name}")
        if row.get("kind") == "zero_adapter":
            if name != "FROZEN_BASE" or set(row) != {"name", "kind"}:
                raise ValueError("zero adapter must be the path-free FROZEN_BASE row")
        elif row.get("kind") == "checkpoint":
            required = {"name", "kind", "checkpoint_path", "checkpoint_sha256", "receipt_path",
                        "receipt_sha256", "expected_mode", "expected_arm", "expected_tasks_completed",
                        "expected_child_config_sha256", "expected_foundation_checkpoint_sha256"}
            if name in ("HOPE_TASK1", "SLEEP_TASK1"):
                required |= {"metrics_path", "metrics_sha256"}
            if set(row) != required:
                raise ValueError(f"context row {name} fields differ from contract")
            if not row.get("checkpoint_path") or not row.get("receipt_path"):
                raise ValueError(f"context row {name} lacks explicit local artifact paths")
            for key in ("checkpoint_sha256", "receipt_sha256", "expected_child_config_sha256"):
                value = row.get(key)
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"context row {name} has malformed {key}")
        else:
            raise ValueError(f"invalid context kind for {name}")
        result[name] = dict(row)
    if set(result) != set(CONTEXT_NAMES):
        raise ValueError("context map is missing one of the four required contexts")
    if result["TASK0_FOUNDATION"].get("expected_mode") != "foundation" or result["TASK0_FOUNDATION"].get("expected_arm") is not None or result["TASK0_FOUNDATION"].get("expected_tasks_completed") != 1:
        raise ValueError("foundation context contract mismatch")
    for name, arm in (("HOPE_TASK1", "HOPE"), ("SLEEP_TASK1", "HOPE_SLEEP_FIXEDMIX")):
        row = result[name]
        if row.get("expected_mode") != "continuation" or row.get("expected_arm") != arm or row.get("expected_tasks_completed") != 2:
            raise ValueError(f"{name} continuation contract mismatch")
        if row["expected_foundation_checkpoint_sha256"] != result["TASK0_FOUNDATION"]["checkpoint_sha256"]:
            raise ValueError(f"{name} is not paired to the declared task-0 foundation")
    return result


def _read_context_records(contexts: Mapping[str, dict], *, project_root: Path, master: Mapping,
                          master_sha: str, manifest_sha: str, source_lock_sha: str):
    """Hash and validate all declared receipts/checkpoints before torch.load."""
    from run_clinc_cil_v8_full import _child_config_sha, validate_checkpoint_payload
    validated = {}
    for name, row in contexts.items():
        if row["kind"] == "zero_adapter":
            validated[name] = None
            continue
        cp = Path(row["checkpoint_path"]).resolve()
        receipt_path = Path(row["receipt_path"]).resolve()
        for path, sha in ((cp, row["checkpoint_sha256"]), (receipt_path, row["receipt_sha256"])):
            if not path.is_file() or sha256_file(path) != sha:
                raise ValueError(f"{name} artifact missing or byte hash mismatch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_arm = row["expected_arm"]
        expected_mode = row["expected_mode"]
        expected_child = _child_config_sha(master_sha, expected_mode, 17, expected_arm)
        if row["expected_child_config_sha256"] != expected_child:
            raise ValueError(f"{name} declared child config hash is not canonical")
        if receipt.get("seed") != 17 or receipt.get("master_config_sha256") != master_sha or receipt.get("training_manifest_sha256") != manifest_sha or receipt.get("source_lock_sha256") != source_lock_sha:
            raise ValueError(f"{name} receipt source/data/config binding mismatch")
        if receipt.get("child_config_sha256") != expected_child:
            raise ValueError(f"{name} receipt child-config binding mismatch")
        if expected_mode == "foundation":
            if receipt.get("schema") != "clinc-cil-v8-common-foundation-receipt" or receipt.get("status") != "complete" or receipt.get("acquisition_qualified") is not True or receipt.get("test_data_included") is not False or receipt.get("tasks_completed") != 1 or receipt.get("final_checkpoint_sha256") != row["checkpoint_sha256"]:
                raise ValueError("foundation receipt does not bind the expected task-0 checkpoint")
        else:
            if receipt.get("schema") != "clinc-cil-v8-full-run-receipt" or receipt.get("status") != "complete" or receipt.get("test_data_included") is not False or receipt.get("arm") != expected_arm or receipt.get("tasks_completed") != 10:
                raise ValueError(f"{name} terminal run receipt mismatch")
            if receipt.get("foundation_checkpoint_sha256") != row["expected_foundation_checkpoint_sha256"]:
                raise ValueError(f"{name} receipt does not bind the declared foundation")
            metrics = Path(row["metrics_path"]).resolve()
            if not metrics.is_file() or sha256_file(metrics) != row["metrics_sha256"]:
                raise ValueError(f"{name} metrics.jsonl missing or byte hash mismatch")
            found = []
            for line in metrics.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item.get("event") == "safe_checkpoint" and item.get("task_id") == 1 and item.get("step") == 188:
                    found.append(item)
            if len(found) != 1 or found[0].get("sha256") != row["checkpoint_sha256"] or found[0].get("path") != f"checkpoints\\task01-next0189-safe.pt":
                raise ValueError(f"{name} metrics do not identify the locked task-1 end checkpoint")
            task_metrics = receipt.get("task_metrics")
            if not isinstance(task_metrics, list) or not any(m.get("task_id") == 1 for m in task_metrics):
                raise ValueError(f"{name} terminal receipt omits task-1 metrics")
        if row["expected_tasks_completed"] not in (1, 2):
            raise ValueError(f"{name} checkpoint task count malformed")
        # Digest is checked before this trusted checkpoint is ever deserialized.
        validated[name] = {"path": cp, "receipt": receipt, "expected": {
            "master_config_sha256": master_sha, "child_config_sha256": expected_child,
            "training_manifest_sha256": manifest_sha, "source_lock_sha256": source_lock_sha,
            "seed": 17, "arm": expected_arm, "mode": expected_mode,
            "foundation_checkpoint_sha256": row["expected_foundation_checkpoint_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "expected_tasks_completed": row["expected_tasks_completed"]}}
    return validated


def _zero_context(context, torch):
    """Construct a literal all-zero adapter state, independent of learned CPs."""
    with torch.no_grad():
        for parameter in context.parameters():
            parameter.zero_()
            parameter.requires_grad_(False)
    for bank in context.banks():
        bank.route_override = None
        bank.dream = False
        bank.last_routes = []
        bank.last_route_audit = None
        bank.audit_history = []
        bank.generator = None
    return context


def _load_context_states(model, validated, *, torch):
    from clinc_tail_runtime_v8 import clone_mutable_context
    import run_clinc_cil_v8_full as full
    loaded = {}
    for name in CONTEXT_NAMES:
        context = clone_mutable_context(model)
        if name == "FROZEN_BASE":
            context = _zero_context(context, torch)
        else:
            row = validated[name]
            if sha256_file(row["path"]) != row["expected"]["checkpoint_sha256"]:
                raise ValueError(f"{name} checkpoint changed after pre-load verification")
            payload = row.get("preflight_payload")
            if payload is None:
                payload = full._load_safe_checkpoint(row["path"], torch=torch, expected=row["expected"])
            validate_context_checkpoint_boundary(name, payload, row)
            context.load_state_dict(payload["context"])
            for parameter in context.parameters():
                parameter.requires_grad_(False)
        loaded[name] = context
    return loaded


def validate_context_checkpoint_boundary(name: str, payload: Mapping, record: Mapping) -> None:
    """Enforce the real end-of-task safe checkpoint cursor before model use."""
    mode = "foundation" if name == "TASK0_FOUNDATION" else "continuation"
    expected_task_id = 0 if mode == "foundation" else 1
    expected_count = 1 if mode == "foundation" else 2
    cursor = payload.get("cursor")
    if not isinstance(cursor, Mapping):
        raise ValueError(f"{name} checkpoint lacks safe cursor")
    if (payload.get("mode") != mode or cursor.get("mode") != mode or
        cursor.get("task_id") != expected_task_id or cursor.get("next_step") != 189 or
        cursor.get("tasks_completed") != expected_count or
        cursor.get("safe_point") != "post_all_due_events" or
        record.get("expected", {}).get("expected_tasks_completed") != expected_count):
        raise ValueError(f"{name} checkpoint is not the expected task-end safe point")


def _preflight_checkpoint_payloads(validated: Mapping[str, Mapping], *, torch) -> dict[str, dict]:
    """CPU-deserialize and validate every declared checkpoint before model loading."""
    import run_clinc_cil_v8_full as full
    payloads = {}
    for name in ("TASK0_FOUNDATION", "HOPE_TASK1", "SLEEP_TASK1"):
        record = validated[name]
        if sha256_file(record["path"]) != record["expected"]["checkpoint_sha256"]:
            raise ValueError(f"{name} checkpoint changed before CPU deserialization")
        payload = full._load_safe_checkpoint(record["path"], torch=torch, expected=record["expected"])
        validate_context_checkpoint_boundary(name, payload, record)
        payloads[name] = payload
    return payloads


def _generation_predictions(model, context_name, context, rows, tokenizer, view, cfg, torch, device, budget):
    from clinc_cil_common_v8 import tokenize_prompt, decode_label_prefix
    from clinc_tail_runtime_v8 import capture, tail_logits
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        item = tokenize_prompt(row["text"], tokenizer, view.classes, tokenizer.eos_token_id,
                               cfg["max_sequence_length"], prepend_bos=False)
        prefix = item["input_ids"][:item["prompt_length"]]
        groups[len(prefix)].append((i, row, prefix))
    max_new = max(len(tokenizer.encode(label, add_special_tokens=False)) for label in view.classes) + 1
    predictions = [None] * len(rows)
    for width, bucket in sorted(groups.items()):
        for start in range(0, len(bucket), cfg["batch_size"]):
            chunk = bucket[start:start + cfg["batch_size"]]
            ids = torch.tensor([x[2] for x in chunk], dtype=torch.long, device=device)
            active = torch.ones_like(ids)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            emitted = [[] for _ in chunk]
            for _ in range(max_new):
                cache = budget.capture(capture, model, ids, active, device=device)
                if context_name == "FROZEN_BASE":
                    step_logits = budget.tail(_base_logits_from_cache, model, cache)
                else:
                    step_logits = budget.tail(tail_logits, model, context, cache)
                logits = step_logits[:, -1].float()
                next_ids = logits.argmax(-1)
                for j, token in enumerate(next_ids.tolist()):
                    if not bool(done[j]):
                        emitted[j].append(int(token))
                        done[j] |= int(token) == int(tokenizer.eos_token_id)
                if bool(done.all()):
                    break
                ids = torch.cat((ids, next_ids[:, None]), dim=1)
                active = torch.cat((active, (~done).long()[:, None]), dim=1)
            for (original, _row, _prefix), answer in zip(chunk, emitted):
                predictions[original] = decode_label_prefix(answer, tokenizer, tokenizer.eos_token_id)
    return predictions


def _aggregate_metric(correct_by_row, invalid_by_row, rows, class_order):
    from clinc_cil_common_v8 import task_for_class
    groups = {"task0": [], "task1": [], "all60": list(range(len(rows)))}
    for i, row in enumerate(rows):
        task = task_for_class(int(row["intent_id"]), class_order)
        if task in (0, 1):
            groups[f"task{task}"].append(i)
    out = {}
    for name, indices in groups.items():
        n = len(indices)
        out[name] = {"count": n, "correct": sum(bool(correct_by_row[i]) for i in indices),
                     "accuracy": sum(bool(correct_by_row[i]) for i in indices) / n if n else None,
                     "invalid_predictions": sum(bool(invalid_by_row[i]) for i in indices)}
    return out


def _evaluate_candidates(model, contexts, rows, tokenizer, view, cfg, torch, device, budget, on_progress):
    from clinc_cil_common_v8 import tokenize_prompt, normalize_prediction
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in view.classes]
    # Intermediate scores are in memory only. The output stores aggregates, never IDs/text/predictions.
    sums = {k: [[None] * len(view.classes) for _ in rows] for k in contexts}
    means = {k: [[None] * len(view.classes) for _ in rows] for k in contexts}
    candidates = []
    for row_i, row in enumerate(rows):
        prompt = tokenize_prompt(row["text"], tokenizer, view.classes, tokenizer.eos_token_id,
                                 cfg["max_sequence_length"], prepend_bos=False)
        prefix = prompt["input_ids"][:prompt["prompt_length"]]
        seqs = build_candidate_sequences(prefix, labels, tokenizer.eos_token_id)
        for class_i, seq in enumerate(seqs):
            candidates.append((row_i, class_i, seq, len(prefix), len(labels[class_i]) + 1))
    grouped = defaultdict(list)
    for item in candidates:
        grouped[len(item[2])].append(item)
    for width, bucket in sorted(grouped.items()):
        for start in range(0, len(bucket), cfg["batch_size"]):
            chunk = bucket[start:start + cfg["batch_size"]]
            scored = score_sequence_chunk(model, contexts, [x[2] for x in chunk],
                                          [x[3] for x in chunk], [x[4] for x in chunk],
                                          torch=torch, device=device, budget=budget)
            for key in contexts:
                for j, item in enumerate(chunk):
                    sums[key][item[0]][item[1]] = float(scored[key]["sum"][j])
                    means[key][item[0]][item[1]] = float(scored[key]["mean"][j])
            on_progress(width, start + len(chunk), len(bucket))
    predictions = {"teacher_forced_sum": {}, "teacher_forced_length_mean": {}, "gold_length_mean_nll": {}}
    for key in contexts:
        predictions["teacher_forced_sum"][key] = [view.classes[max(range(150), key=lambda j: sums[key][i][j])] for i in range(len(rows))]
        predictions["teacher_forced_length_mean"][key] = [view.classes[max(range(150), key=lambda j: means[key][i][j])] for i in range(len(rows))]
        predictions["gold_length_mean_nll"][key] = [-means[key][i][int(row["intent_id"])] for i, row in enumerate(rows)]
    return predictions


def _summary_correct(predictions: Sequence[str], rows: Sequence[Mapping], classes: Sequence[str]):
    from clinc_cil_common_v8 import normalize_prediction
    mapping = {normalize_prediction(label): i for i, label in enumerate(classes)}
    correct, invalid = [], []
    for prediction, row in zip(predictions, rows):
        parsed = mapping.get(normalize_prediction(prediction))
        invalid.append(parsed is None)
        correct.append(parsed == int(row["intent_id"]))
    return correct, invalid


def _aggregate_values(values: Sequence[float], rows: Sequence[Mapping], class_order: Sequence[int]):
    from clinc_cil_common_v8 import task_for_class
    groups = {"task0": [], "task1": [], "all60": list(range(len(rows)))}
    for i, row in enumerate(rows):
        task = task_for_class(int(row["intent_id"]), class_order)
        if task in (0, 1): groups[f"task{task}"].append(i)
    out = {}
    for name, indexes in groups.items():
        xs = [float(values[i]) for i in indexes]
        out[name] = {"count": len(xs), "mean": sum(xs) / len(xs) if xs else None}
    return out


def _evaluate(args):
    evaluation_started = time.monotonic()
    import clinc_cil_common_v8 as common
    import run_clinc_cil_v8_full as full
    import run_clinc_cil_v8_r2 as r2

    master = full.validate_master_config(json.loads(args.master_config.read_text(encoding="utf-8")))
    master_sha, pack_sha = sha256_file(args.master_config), sha256_file(args.manifest)
    if pack_sha != master["training_manifest_sha256"]:
        raise ValueError("training pack SHA differs from locked master configuration")
    full._verify_asset_hashes(args.model_path, master["model_assets_sha256"])
    tool_dir = Path(__file__).resolve().parent
    frozen_tools_dir = args.project_root / "tools"
    source_lock = full.canonical_sha(full._source_hashes(args.project_root, frozen_tools_dir))
    doc = json.loads(args.contexts.read_text(encoding="utf-8"))
    ctx_rows = validate_context_map(doc, master_sha=master_sha, manifest_sha=pack_sha, source_lock_sha=source_lock)
    validated = _read_context_records(ctx_rows, project_root=args.project_root, master=master,
                                      master_sha=master_sha, manifest_sha=pack_sha, source_lock_sha=source_lock)
    import torch
    preflight_payloads = _preflight_checkpoint_payloads(validated, torch=torch)
    for name, payload in preflight_payloads.items():
        validated[name]["preflight_payload"] = payload
    # All CP/source/config/data checks have passed before any checkpoint is deserialized.
    view, train_rows, dev_rows, _clean, _partition, _duplicates, r2 = full._read_inputs(args.manifest, master)
    order = view.class_orders["17"]
    rows = fixed_clean_development_rows(train_rows, dev_rows, order, normalize_text=r2.normalize_text)
    cohort_sha = hashlib.sha256("\n".join(sorted(str(r["id"]) for r in rows)).encode()).hexdigest()
    cfg = full._runtime_cfg(master, 17, max_mode="continuation", arm="HOPE_SLEEP_FIXEDMIX")
    cfg.update({"rank": master["rank"], "capacity": master["capacity"], "levels": master["levels"],
                "max_sequence_length": master["max_sequence_length"], "max_new_tokens": master["max_new_tokens"],
                "batch_size": master["batch_size"], "device": args.device})
    torch, tokenizer, model, immutable_versions = r2._load_common(args.project_root, args.model_path, args.device, cfg)
    model.eval(); model.requires_grad_(False)
    contexts = _load_context_states(model, validated, torch=torch)
    for context in contexts.values():
        for parameter in context.parameters(): parameter.requires_grad_(False)
    initial_context_hashes = {name: r2._mutable_context_sha(context) for name, context in contexts.items()}
    initial_immutable = immutable_versions(model)
    from clinc_tail_runtime_v8 import clone_mutable_context
    baseline = _zero_context(clone_mutable_context(model), torch)
    # Reuse the baseline only if the model's freshly cloned context is exactly the declared zero adapter.
    if r2._mutable_context_sha(baseline) != initial_context_hashes["FROZEN_BASE"]:
        raise ValueError("zero adapter changed during model/context initialization")
    # The wall cap covers source/data/checkpoint validation and model setup as well
    # as inference. Forward/token counts still begin at the first scored call.
    budget = Budget(started=evaluation_started)
    try:
        generation = {}
        generation_start_fw, generation_start_tail = budget.forwards, budget.tail_calls
        for name, context in contexts.items():
            preds = _generation_predictions(model, name, context, rows, tokenizer, view, cfg,
                                            torch, args.device, budget)
            generation[name] = preds
        generation_work = {"backbone_forwards": budget.forwards - generation_start_fw,
                           "tail_calls": budget.tail_calls - generation_start_tail}
        ranking_start_fw, ranking_start_tail = budget.forwards, budget.tail_calls
        def progress(width, done, total):
            _append_event(args.out_dir / "progress.jsonl", "candidate_scoring",
                          sequence_length=width, chunk_done=done, length_group_size=total,
                          forwards=budget.forwards, tail_calls=budget.tail_calls)
        ranking = _evaluate_candidates(model, contexts, rows, tokenizer, view, cfg,
                                       torch, args.device, budget, progress)
        if budget.forwards > MAX_FW or budget.summary()["elapsed_seconds"] > MAX_SECONDS:
            raise RuntimeError("v9 diagnostic exceeded hard budget")
        # Recompute all frozen/context fingerprints before writing a success record.
        final_context_hashes = {name: r2._mutable_context_sha(context) for name, context in contexts.items()}
        final_immutable = immutable_versions(model)
        if initial_context_hashes != final_context_hashes or initial_immutable != final_immutable:
            raise RuntimeError("evaluation mutated a frozen context or backbone state")
        budget.check_wall()
        if budget.forwards > MAX_FW:
            raise RuntimeError("v9 diagnostic exceeded backbone-forward cap")
        groups = {"task0": [], "task1": [], "all60": list(range(len(rows)))}
        from clinc_cil_common_v8 import task_for_class, normalize_prediction
        id_to_task = {cid: task_for_class(cid, order) for cid in range(150)}
        for i, row in enumerate(rows):
            task = id_to_task[int(row["intent_id"])]
            if task in (0, 1): groups[f"task{task}"].append(i)
        metrics = {}
        for name in contexts:
            gen_correct, gen_invalid = _summary_correct(generation[name], rows, view.classes)
            metrics[name] = {"v8_greedy_generation": _aggregate_metric(gen_correct, gen_invalid, rows, order)}
            for metric_key, pred_key in (("teacher_forced_sum", "teacher_forced_sum"),
                                         ("teacher_forced_length_mean", "teacher_forced_length_mean")):
                correct, invalid = _summary_correct(ranking[pred_key][name], rows, view.classes)
                metrics[name][metric_key] = _aggregate_metric(correct, invalid, rows, order)
            metrics[name]["gold_length_mean_nll"] = _aggregate_values(ranking["gold_length_mean_nll"][name], rows, order)
            for metric_key in ("teacher_forced_sum", "teacher_forced_length_mean"):
                for group in metrics[name][metric_key]:
                    metrics[name][metric_key][group]["delta_correct_vs_v8_generation"] = (
                        metrics[name][metric_key][group]["correct"] - metrics[name]["v8_greedy_generation"][group]["correct"])
        output = {"schema": SCHEMA, "mode": "evaluate", "development_only": True,
                  "context_map_sha256": sha256_file(args.contexts), "master_config_sha256": master_sha,
                  "training_manifest_sha256": pack_sha, "source_lock_sha256": source_lock,
                  "fixed_cohort": {"rows": 60, "classes": 30, "rows_per_class": 2,
                                   "task0_rows": 30, "task1_rows": 30, "stable_id_set_sha256": cohort_sha},
                  "label_vocabulary": {"count": 150, "canonical_order_sha256": hashlib.sha256("\n".join(view.classes).encode()).hexdigest(),
                                       "all_labels_as_candidates": True, "candidate_sequences": 9000,
                                       "label_dictionary_disclosure": "All 150 canonical labels were scored as candidates; this diagnostic intentionally exposes the label set and adds 150-way inference per example. It is not the v8 primary metric."},
                  "methods": {"v8_greedy_generation": {"primary_reference": True, "candidate_constrained": False,
                                                           "work": generation_work},
                              "teacher_forced_sum": {"candidate_constrained": True, "includes_eos": True},
                              "teacher_forced_length_mean": {"candidate_constrained": True, "includes_eos": True},
                              "candidate_scoring_work": {"backbone_forwards": budget.forwards - ranking_start_fw,
                                                         "tail_calls": budget.tail_calls - ranking_start_tail}},
                  "metrics": metrics, "work": budget.summary(),
                  "immutability": {"context_hashes_before": initial_context_hashes,
                                   "context_hashes_after": final_context_hashes,
                                   "backbone_versions_unchanged": True,
                                   "all_adapters_frozen": True}}
        _atomic_json(args.out_dir / "result.json", output)
        return output
    except BaseException as exc:
        failure = {"schema": SCHEMA, "mode": "evaluate", "phase": "inference_or_aggregation",
                   "exception_type": type(exc).__name__, "message": "diagnostic failed; see local stderr",
                   "work": budget.summary(), "optimizer_updates": 0,
                   "context_hashes_at_start": initial_context_hashes}
        if not (args.out_dir / "failure.json").exists():
            _atomic_json(args.out_dir / "failure.json", failure)
        raise


def _append_event(path: Path, phase: str, **fields):
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"phase": phase, **fields}, sort_keys=True, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    if path.exists():
        temp.unlink(missing_ok=True)
        raise FileExistsError(path)
    os.replace(temp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("token-audit", "checkpoint-preflight", "evaluate"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--manifest", type=Path, required=True, help="v8 train/development-only manifest")
    parser.add_argument("--master-config", type=Path, help="frozen full-grid master config (evaluate mode)")
    parser.add_argument("--contexts", type=Path, help="explicit, source-bound four-context checkpoint map")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists; no overwrite or retry")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    progress = args.out_dir / "progress.jsonl"

    def event(phase, **fields):
        with progress.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"phase": phase, **fields}, sort_keys=True, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())

    try:
        # Import only public train/dev helpers. They do not expose a test reader here.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import clinc_cil_common_v8 as common
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        view = common.make_training_view(manifest)
        if len(view.classes) != 150:
            raise ValueError("expected the canonical 150-label CLINC vocabulary")
        if args.mode == "token-audit":
            if args.model_path is None:
                raise ValueError("token-audit mode requires --model-path")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer requires EOS")
            report = {"schema": SCHEMA, "mode": "token-audit", "manifest_sha256": sha256_file(args.manifest),
                      "tokenizer": label_tokenizer_audit(tokenizer, view.classes)}
            _atomic_json(args.out_dir / "token-audit.json", report)
            event("complete", mode="token-audit")
            return 0
        if args.mode == "checkpoint-preflight":
            if args.master_config is None or args.contexts is None:
                raise ValueError("checkpoint-preflight requires frozen master config and context map")
            import run_clinc_cil_v8_full as full
            master = full.validate_master_config(json.loads(args.master_config.read_text(encoding="utf-8")))
            master_sha, pack_sha = sha256_file(args.master_config), sha256_file(args.manifest)
            if pack_sha != master["training_manifest_sha256"]:
                raise ValueError("training pack SHA differs from locked master configuration")
            source_lock = full.canonical_sha(full._source_hashes(args.project_root, args.project_root / "tools"))
            doc = json.loads(args.contexts.read_text(encoding="utf-8"))
            rows = validate_context_map(doc, master_sha=master_sha, manifest_sha=pack_sha,
                                        source_lock_sha=source_lock)
            validated = _read_context_records(rows, project_root=args.project_root, master=master,
                                              master_sha=master_sha, manifest_sha=pack_sha,
                                              source_lock_sha=source_lock)
            import torch
            payloads = _preflight_checkpoint_payloads(validated, torch=torch)
            preflight = {"schema": SCHEMA, "mode": "checkpoint-preflight", "cpu_only": True,
                         "test_data_accessed": False, "optimizer_updates": 0,
                         "source_lock_sha256": source_lock,
                         "contexts": {name: {"checkpoint_sha256": validated[name]["expected"]["checkpoint_sha256"],
                                              "cursor": {k: payloads[name]["cursor"][k] for k in
                                                         ("mode", "task_id", "next_step", "tasks_completed", "safe_point")}}
                                      for name in payloads}}
            _atomic_json(args.out_dir / "checkpoint-preflight.json", preflight)
            event("complete", mode="checkpoint-preflight", checkpoint_count=3)
            return 0
        if args.contexts is None or args.master_config is None:
            raise ValueError("evaluate mode requires explicit context map and frozen master config")
        if args.model_path is None:
            raise ValueError("evaluate mode requires --model-path")
        event("preflight_started", mode="evaluate")
        report = _evaluate(args)
        event("complete", mode="evaluate", backbone_forwards=report["work"]["backbone_forwards"],
              tail_calls=report["work"]["tail_calls"])
        return 0
    except BaseException as exc:
        failure = {"schema": SCHEMA, "phase": "failed", "exception_type": type(exc).__name__,
                   "message": "diagnostic failed; see local stderr for details",
                   "optimizer_updates": 0}
        if not (args.out_dir / "failure.json").exists():
            _atomic_json(args.out_dir / "failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
