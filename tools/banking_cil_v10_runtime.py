"""Banking77-specific label scoring and Sleep-event runtime helpers (v10).

This is a local, controlled CLINC-v8-runtime reuse for Banking77; it is not a
claim of native/paper-faithful HOPE.  Class count is checked explicitly as 77.
All candidate scoring uses the public segmented prompt convention, no BOS, and
all 77 labels without a task mask.  Returned records contain aggregates only.
"""
from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import defaultdict
from typing import Mapping, Sequence

SCHEMA = "banking-cil-v10-runtime-v1"
NUM_CLASSES = 77


def _validate_classes(classes: Sequence[str]) -> tuple[str, ...]:
    values = tuple(classes)
    if len(values) != NUM_CLASSES or len(set(values)) != NUM_CLASSES:
        raise ValueError("Banking77 runtime requires exactly 77 unique canonical classes")
    if any(not isinstance(x, str) or not x or x != x.strip() for x in values):
        raise ValueError("canonical classes must be nonempty trimmed strings")
    return values


def _normalized_label(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip().replace("_", " ")
    return " ".join(value.split())


def paper_edit_similarity(left: Sequence[int], right: Sequence[int], *, z0: int) -> float:
    """Exact full-v8 thresholded, length-normalized token edit similarity."""
    if type(z0) is not int or z0 < 1:
        raise ValueError("z0 must be a positive integer")
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


def local_lti_reward(teacher_text: str, student_text: str, *,
                     teacher_tokens: Sequence[int], student_tokens: Sequence[int],
                     z0: int, canonical_labels: Sequence[str],
                     semantic_weight: float = 0.5, edit_weight: float = 0.5) -> float:
    """Local label-equivalence + full-v8 edit proxy; not a trained reward model."""
    if not math.isfinite(float(semantic_weight)) or not math.isfinite(float(edit_weight)):
        raise ValueError("reward weights must be finite")
    if float(semantic_weight) < 0 or float(edit_weight) < 0:
        raise ValueError("reward weights must be nonnegative")
    teacher = _normalized_label(teacher_text)
    student = _normalized_label(student_text)
    labels = {_normalized_label(x) for x in _validate_classes(canonical_labels)}
    semantic = float(bool(teacher) and teacher == student and teacher in labels)
    edit = paper_edit_similarity(teacher_tokens, student_tokens, z0=z0)
    return float(semantic_weight) * semantic + float(edit_weight) * edit


def _append_count(budget, mask):
    if budget is None:
        return
    check = getattr(budget, "check", None)
    if callable(check):
        check()


def _class_task_map(class_order: Sequence[int] | None):
    if class_order is None:
        return None
    order = tuple(int(x) for x in class_order)
    if len(order) != NUM_CLASSES or set(order) != set(range(NUM_CLASSES)):
        raise ValueError("class_order must be a permutation of the 77 Banking77 IDs")
    return {class_id: index // 11 for index, class_id in enumerate(order)}


def teacher_forced_scores(logits, input_ids, prompt_lengths: Sequence[int], answer_lengths: Sequence[int]):
    """Scores only answer tokens plus EOS: summed and token-mean log likelihood."""
    import torch.nn.functional as F
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != (input_ids.shape[0], input_ids.shape[1] - 1):
        raise ValueError("logits must align to next-token positions")
    if len(prompt_lengths) != input_ids.shape[0] or len(answer_lengths) != input_ids.shape[0]:
        raise ValueError("per-row prompt/answer lengths mismatch")
    targets = input_ids[:, 1:].long()
    logp = F.log_softmax(logits.float(), dim=-1)
    selected = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    sums, means = [], []
    for i, (prefix_n, answer_n) in enumerate(zip(prompt_lengths, answer_lengths)):
        prefix_n, answer_n = int(prefix_n), int(answer_n)
        if prefix_n < 1 or answer_n < 1 or prefix_n + answer_n > input_ids.shape[1]:
            raise ValueError("invalid prefix or answer boundary")
        start, stop = prefix_n - 1, prefix_n + answer_n - 1
        values = selected[i, start:stop]
        if values.numel() != answer_n:
            raise ValueError("candidate answer mask length mismatch")
        total = values.sum()
        sums.append(total)
        means.append(total / answer_n)
    return torch_stack(sums), torch_stack(means)


def torch_stack(values):
    # Kept lazy so metadata-only imports do not import torch.
    import torch
    return torch.stack(values)


def _candidate_sequence(prefix: Sequence[int], label: Sequence[int], eos: int) -> list[int]:
    if not prefix or not label:
        raise ValueError("prefix and canonical label encoding must be nonempty")
    return [int(x) for x in prefix] + [int(x) for x in label] + [int(eos)]


def _score_rows(contexts, per_row_sum, per_row_mean, gold_nll, rows, classes, class_order):
    task_map = _class_task_map(class_order)
    groups = {"all77": list(range(len(rows)))}
    if task_map is not None:
        groups.update({"old": [], "seen": [], "current": [], "unseen": []})
        current_task = getattr(rows, "task_id", None)
        if current_task is None:
            current_task = 0
        seen_classes = {cid for cid, task in task_map.items() if task <= current_task}
        current_classes = {cid for cid, task in task_map.items() if task == current_task}
        for i, row in enumerate(rows):
            cid = int(row["intent_id"])
            if cid in current_classes:
                groups["current"].append(i)
            elif cid in seen_classes:
                groups["old"].append(i)
            if cid in seen_classes:
                groups["seen"].append(i)
            else:
                groups["unseen"].append(i)
    mapping = {_normalized_label(name): i for i, name in enumerate(classes)}
    metrics = {}
    for name, pred_ids in (("teacher_forced_sum", per_row_sum),
                           ("teacher_forced_length_mean", per_row_mean)):
        result = {}
        for group, indices in groups.items():
            n = len(indices)
            correct = sum(int(int(rows[i]["intent_id"]) == int(pred_ids[i])) for i in indices)
            result[group] = {"count": n, "correct": correct,
                             "accuracy": correct / n if n else None,
                             "invalid_predictions": 0}
        result["per_class"] = {}
        for cid in range(NUM_CLASSES):
            total = sum(int(r["intent_id"]) == cid for r in rows)
            correct = sum(int(r["intent_id"]) == cid and int(pred_ids[i]) == cid
                          for i, r in enumerate(rows))
            result["per_class"][str(cid)] = {"count": total, "correct": correct,
                                              "accuracy": correct / total if total else None}
        metrics[name] = result
    gold = {}
    for group, indices in groups.items():
        values = [float(gold_nll[i]) for i in indices]
        gold[group] = {"count": len(values), "mean_token_nll": sum(values) / len(values) if values else None}
    return metrics, gold


def score_candidates(model, contexts: Mapping[str, object], rows: Sequence[Mapping], tokenizer,
                     classes: Sequence[str], common, tail_runtime, *, torch, device,
                     max_sequence_length: int = 128, batch_size: int = 8, budget=None,
                     class_order: Sequence[int] | None = None, task_id: int | None = None,
                     include_greedy: bool = False, max_new_tokens: int = 32) -> dict:
    """Score global all-77 label candidates; share every capture across contexts.

    SUM log-likelihood is primary, length-mean is a secondary diagnostic, and
    gold-label token-mean NLL is emitted as a format/probability diagnostic.
    No candidate is removed by task visibility. Optional greedy decoding is an
    additional unconstrained metric and uses the same gold-independent prefix.
    """
    classes = _validate_classes(classes)
    if not rows or batch_size < 1 or max_sequence_length < 1:
        raise ValueError("nonempty rows and positive sequence/batch sizes required")
    if set(contexts) == set():
        raise ValueError("at least one named context is required")
    task_map = _class_task_map(class_order)
    if task_id is not None and (task_map is None or type(task_id) is not int or not 0 <= task_id < 7):
        raise ValueError("task breakdown requires a valid 77-class order and task ID")
    if task_id is not None:
        # Attach study-bound visibility only for reporting; candidate enumeration remains global.
        rows = _RowsWithTask(rows, task_id)
    if include_greedy:
        required_new_tokens = max(len(tokenizer.encode(label, add_special_tokens=False)) + 1 for label in classes)
        if max_new_tokens < required_new_tokens:
            raise ValueError("greedy cap is shorter than the longest canonical label plus EOS")
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in classes]
    if any(not ids for ids in label_ids):
        raise ValueError("canonical label tokenized to empty sequence")
    candidate_records = []
    prefixes = []
    for row_i, row in enumerate(rows):
        prompt = common.tokenize_prompt(row["text"], tokenizer, classes, tokenizer.eos_token_id,
                                        max_sequence_length)
        prefix = prompt["input_ids"][:prompt["prompt_length"]]
        prefixes.append(prefix)
        for class_i, answer in enumerate(label_ids):
            seq = _candidate_sequence(prefix, answer, tokenizer.eos_token_id)
            candidate_records.append((row_i, class_i, seq, len(prefix), len(answer) + 1))
    by_length = defaultdict(list)
    for record in candidate_records:
        by_length[len(record[2])].append(record)
    sums = {name: [[0.0] * NUM_CLASSES for _ in rows] for name in contexts}
    means = {name: [[0.0] * NUM_CLASSES for _ in rows] for name in contexts}
    captures = tails = input_token_count = 0
    for width, group in sorted(by_length.items()):
        for start in range(0, len(group), batch_size):
            chunk = group[start:start + batch_size]
            _append_count(budget, None)
            ids = torch.tensor([x[2] for x in chunk], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            with torch.no_grad():
                cache = tail_runtime.capture(model, ids, mask, device=device)
            captures += 1
            input_token_count += int(ids.numel())
            for name, context in contexts.items():
                with torch.no_grad():
                    logits = tail_runtime.tail_logits(model, context, cache)
                tails += 1
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("nonfinite logits while scoring Banking77 candidates")
                chunk_sum, chunk_mean = teacher_forced_scores(
                    logits[:, :-1, :], ids,
                    [x[3] for x in chunk], [x[4] for x in chunk])
                for j, (row_i, class_i, *_rest) in enumerate(chunk):
                    sums[name][row_i][class_i] = float(chunk_sum[j].detach().cpu())
                    means[name][row_i][class_i] = float(chunk_mean[j].detach().cpu())
                    if not math.isfinite(sums[name][row_i][class_i]) or not math.isfinite(means[name][row_i][class_i]):
                        raise ValueError("nonfinite candidate likelihood")
    context_scores = {}
    context_metrics = {}
    for name in contexts:
        sum_ids = [max(range(NUM_CLASSES), key=lambda cid: sums[name][i][cid]) for i in range(len(rows))]
        mean_ids = [max(range(NUM_CLASSES), key=lambda cid: means[name][i][cid]) for i in range(len(rows))]
        gold_nll = [-means[name][i][int(row["intent_id"])] for i, row in enumerate(rows)]
        primary, secondary_gold = _score_rows(contexts, sum_ids, mean_ids, gold_nll,
                                              rows, classes, class_order if task_id is not None else None)
        context_metrics[name] = {**primary, "gold_label_token_mean_nll": secondary_gold}
        context_scores[name] = {"candidate_count_per_example": NUM_CLASSES,
                                "gold_correct_sum": sum(int(x == int(r["intent_id"])) for x, r in zip(sum_ids, rows)),
                                "gold_correct_length_mean": sum(int(x == int(r["intent_id"])) for x, r in zip(mean_ids, rows))}
    greedy_metrics = None
    greedy_work = {"backbone_forwards": 0, "tail_calls": 0, "generated_tokens": 0}
    if include_greedy:
        greedy_metrics, greedy_work = _greedy_all_contexts(model, contexts, rows, prefixes, tokenizer,
                                                           classes, tail_runtime, torch, device,
                                                           max_new_tokens, batch_size, budget)
    return {"schema": SCHEMA, "candidate_metric_primary": "global_77_sum_log_likelihood",
            "classes": NUM_CLASSES, "candidate_sequences": len(rows) * NUM_CLASSES,
            "label_dictionary_sha256": hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest(),
            "candidate_metrics": context_metrics, "candidate_correct_counts": context_scores,
            "greedy_generation_metrics": greedy_metrics,
            "work": {"candidate_backbone_forwards": captures,
                     "candidate_tail_calls": tails, "candidate_input_tokens": input_token_count,
                     "greedy": greedy_work}}


class _RowsWithTask(list):
    def __init__(self, rows, task_id):
        super().__init__(rows)
        self.task_id = task_id


def _decode_answer(token_ids, tokenizer, eos_token_id):
    ids = list(token_ids)
    if int(eos_token_id) in ids:
        ids = ids[:ids.index(int(eos_token_id))]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _greedy_all_contexts(model, contexts, rows, prefixes, tokenizer, classes, tail_runtime,
                         torch, device, max_new_tokens, batch_size, budget):
    by_length = defaultdict(list)
    for i, (row, prefix) in enumerate(zip(rows, prefixes)):
        by_length[len(prefix)].append((i, row, prefix))
    outcomes = {name: [None] * len(rows) for name in contexts}
    fw = tails = generated = 0
    for name, context in contexts.items():
        for _width, group in sorted(by_length.items()):
            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                ids = torch.tensor([x[2] for x in chunk], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                finished = torch.zeros(len(chunk), dtype=torch.bool, device=device)
                answer = [[] for _ in chunk]
                for _ in range(max_new_tokens):
                    _append_count(budget, mask)
                    with torch.no_grad():
                        cache = tail_runtime.capture(model, ids, mask, device=device)
                    fw += 1
                    with torch.no_grad():
                        logits = tail_runtime.tail_logits(model, context, cache)[:, -1].float()
                    tails += 1
                    next_ids = logits.argmax(-1)
                    for j, token in enumerate(next_ids.tolist()):
                        if not bool(finished[j]):
                            answer[j].append(int(token)); generated += 1
                            if int(token) == int(tokenizer.eos_token_id): finished[j] = True
                    if bool(finished.all()): break
                    ids = torch.cat((ids, next_ids[:, None]), dim=1)
                    mask = torch.cat((mask, (~finished).long()[:, None]), dim=1)
                for (i, _row, _prefix), ids_out in zip(chunk, answer):
                    text = _decode_answer(ids_out, tokenizer, tokenizer.eos_token_id)
                    outcomes[name][i] = {_normalized_label(label): cid for cid, label in enumerate(classes)}.get(
                        _normalized_label(text))
    metrics = {}
    for name, pred in outcomes.items():
        correct = sum(x == int(row["intent_id"]) for x, row in zip(pred, rows))
        invalid = sum(x is None for x in pred)
        metrics[name] = {"count": len(rows), "correct": correct,
                         "accuracy": correct / len(rows) if rows else None,
                         "invalid_predictions": invalid, "candidate_constrained": False}
    return metrics, {"backbone_forwards": fw, "tail_calls": tails, "generated_tokens": generated}


def run_sleep_event(model, tx, selected, tokenizer, classes, common, cfg, torch, device,
                    budget, tail_runtime, *, sample_tokens, lti_reward,
                    causal_token_kl=None, causal_token_ce=None, route_audit=None):
    """Run selected GKD/LTI objectives on a caller-created Sleep transaction.

    The driver owns sender reset, receiver activation, prospective application,
    transaction snapshot/commit, and wake-clock accounting. This function only
    consumes an already-open transaction with frozen teacher/student contexts.
    """
    classes = _validate_classes(classes)
    max_label_with_eos = max(len(tokenizer.encode(label, add_special_tokens=False)) + 1 for label in classes)
    if int(cfg["max_new_tokens"]) < max_label_with_eos:
        raise ValueError("Sleep rollout cap is shorter than the longest canonical label plus EOS")
    causal_token_kl = causal_token_kl or tail_runtime.causal_token_kl
    causal_token_ce = causal_token_ce or tail_runtime.causal_token_ce
    outputs = []
    gkd_index = 0
    label_set = {_normalized_label(x) for x in classes}
    for kind, row, role in selected:
        if kind not in ("gkd", "lti"):
            raise ValueError("Sleep event selection kind must be gkd or lti")
        _append_count(budget, None)
        prompt_item = common.tokenize_prompt(row["text"], tokenizer, classes,
                                             tokenizer.eos_token_id,
                                             int(cfg["max_sequence_length"]))
        prompt = [int(x) for x in prompt_item["input_ids"][:prompt_item["prompt_length"]]]
        owner = "teacher" if kind == "lti" or gkd_index % 2 == 0 else "student"
        if kind == "gkd": gkd_index += 1
        owner_context = tx.teacher_context() if owner == "teacher" else tx.student_context()
        sampled_result, sampled_answer = sample_tokens(model, owner_context, prompt,
            int(cfg["max_new_tokens"]), float(cfg["temperature"]), tx.generator,
            tokenizer.eos_token_id, torch, device, budget)
        reward = None
        teacher_answer_empty = None
        student_full_answer_empty = None
        if kind == "gkd":
            sequence = sampled_result.token_ids
            answer_mask = sampled_result.answer_mask()
            cache = tail_runtime.capture(model, sequence, torch.ones_like(sequence), device=device)
            with torch.no_grad():
                teacher_logits = tail_runtime.tail_logits(model, tx.teacher_context(), cache)
            if route_audit is None:
                student_logits = tail_runtime.tail_logits(model, tx.student_context(), cache)
                route_stats = None
            else:
                student_logits, route_stats = route_audit(model, tx.student_context(), cache,
                                                          tx.receiver, tx.slot, answer_mask, tail_runtime)
            loss, token_count = causal_token_kl(student_logits, teacher_logits, answer_mask, temperature=1.0)
        else:
            teacher_ids = list(sampled_answer)
            if teacher_ids and int(teacher_ids[-1]) == int(tokenizer.eos_token_id):
                teacher_ids = teacher_ids[:-1]
            prefix_len = (int(torch.randint(0, len(teacher_ids), (1,), generator=tx.generator,
                             device=tx.generator.device).item()) if teacher_ids else 0)
            student_prefix = prompt + teacher_ids[:prefix_len]
            eos_emitted = bool(sampled_answer and int(sampled_answer[-1]) == int(tokenizer.eos_token_id))
            suffix_budget = max(1, len(teacher_ids) - prefix_len + int(eos_emitted))
            student_result, student_answer = sample_tokens(model, tx.student_context(), student_prefix,
                max(1, min(int(cfg["max_new_tokens"]), suffix_budget)), float(cfg["temperature"]),
                tx.generator, tokenizer.eos_token_id, torch, device, budget)
            sequence = student_result.token_ids
            answer_mask = student_result.answer_mask()
            cache = tail_runtime.capture(model, sequence, torch.ones_like(sequence), device=device)
            if route_audit is None:
                student_logits = tail_runtime.tail_logits(model, tx.student_context(), cache)
                route_stats = None
            else:
                student_logits, route_stats = route_audit(model, tx.student_context(), cache,
                                                          tx.receiver, tx.slot, answer_mask, tail_runtime)
            with torch.no_grad():
                teacher_logits = tail_runtime.tail_logits(model, tx.teacher_context(), cache)
            kl, kl_tokens = causal_token_kl(student_logits, teacher_logits, answer_mask, temperature=1.0)
            student_suffix = list(student_answer)
            if student_suffix and int(student_suffix[-1]) == int(tokenizer.eos_token_id):
                student_suffix = student_suffix[:-1]
            teacher_full = list(teacher_ids[:prefix_len]) + student_suffix
            teacher_text = _decode_answer(teacher_ids, tokenizer, tokenizer.eos_token_id)
            student_text = _decode_answer(teacher_full, tokenizer, tokenizer.eos_token_id)
            teacher_answer_empty = not bool(teacher_text.strip())
            student_full_answer_empty = not bool(student_text.strip())
            reward = float(lti_reward(teacher_text, student_text, teacher_tokens=teacher_ids,
                                      student_tokens=teacher_full, z0=int(cfg["lti_edit_distance_z0"]),
                                      canonical_labels=classes))
            ce, ce_tokens = causal_token_ce(student_logits, sequence, answer_mask)
            loss, token_count = 0.5 * kl + 0.5 * reward * ce, int(kl_tokens) + int(ce_tokens)
        loss_value = float(loss.detach().float().cpu())
        if not bool(torch.isfinite(loss)) or not bool(loss.requires_grad):
            raise RuntimeError("Banking Sleep yielded nonfinite or nondifferentiable loss")
        answer_text = _decode_answer(sampled_answer, tokenizer, tokenizer.eos_token_id)
        parsed = _normalized_label(answer_text)
        gold = _normalized_label(classes[int(row["intent_id"])])
        skipped_zero_objective = bool(kind == "lti" and reward == 0.0 and loss_value == 0.0)
        before_zero = int(tx.counters.get("zero_gradient_steps", 0)) if isinstance(tx.counters, Mapping) else 0
        if not skipped_zero_objective and budget is not None:
            backward = getattr(budget, "backward", None)
            update = getattr(budget, "update", None)
            if callable(backward): backward(torch.ones((1, int(token_count)), dtype=torch.long, device=device))
            if callable(update): update()
        if not skipped_zero_objective:
            tx.optimize(loss, kind, max(1, int(token_count)))
        after_zero = int(tx.counters.get("zero_gradient_steps", 0)) if isinstance(tx.counters, Mapping) else before_zero
        if not skipped_zero_objective and after_zero > before_zero:
            raise RuntimeError("Banking Sleep observed a zero-gradient optimizer step")
        row_id = str(row.get("id", ""))
        row_hash = hashlib.sha256(row_id.encode("utf-8")).hexdigest() if row_id else None
        outputs.append({"kind": kind, "sample_owner": owner, "loss": loss_value,
                        "token_count": int(token_count), "lti_reward": reward,
                        "sampled_answer_tokens_including_eos": len(sampled_answer),
                        "sampled_first_token_eos": bool(sampled_answer and
                            int(sampled_answer[0]) == int(tokenizer.eos_token_id)),
                        "decoded_sample_empty": not bool(answer_text.strip()),
                        "teacher_answer_empty": teacher_answer_empty,
                        "student_full_answer_empty": student_full_answer_empty,
                        "sampled_label_canonical": parsed in label_set,
                        "sampled_label_matches_gold": parsed == gold,
                        "skipped_zero_objective": skipped_zero_objective,
                        "optimizer_updated": not skipped_zero_objective,
                        "example_id_sha256": row_hash, "role": str(role),
                        "class_id": int(row["intent_id"]),
                        "zero_reward": bool(kind == "lti" and reward == 0.0),
                        "zero_gradient_steps": after_zero - before_zero,
                        "route_summary": _safe_route_summary(route_stats)})
    by_owner = {}
    for owner in ("teacher", "student"):
        owner_rows = [x for x in outputs if x["sample_owner"] == owner]
        by_owner[owner] = {"count": len(owner_rows),
                           "canonical_valid": sum(x["sampled_label_canonical"] for x in owner_rows),
                           "gold_agreement": sum(x["sampled_label_matches_gold"] for x in owner_rows),
                           "decoded_empty_count": sum(x["decoded_sample_empty"] for x in owner_rows),
                           "first_token_eos_count": sum(x["sampled_first_token_eos"] for x in owner_rows),
                           "sampled_answer_tokens_including_eos": sum(
                               x["sampled_answer_tokens_including_eos"] for x in owner_rows)}
    teacher_rows = [x for x in outputs if x["sample_owner"] == "teacher"]
    lti_rows = [x for x in outputs if x["kind"] == "lti"]
    return {"steps": len(outputs), "gkd_steps": sum(x["kind"] == "gkd" for x in outputs),
            "lti_steps": sum(x["kind"] == "lti" for x in outputs),
            "loss_mean": sum(x["loss"] for x in outputs) / len(outputs) if outputs else None,
            "teacher_canonical_valid": sum(x["sampled_label_canonical"] for x in teacher_rows),
            "teacher_gold_agreement": sum(x["sampled_label_matches_gold"] for x in teacher_rows),
            "sampled_rollouts_by_owner": by_owner,
            "lti_answer_diagnostics": {
                "count": len(lti_rows),
                "teacher_answer_empty_count": sum(bool(x["teacher_answer_empty"]) for x in lti_rows),
                "student_full_answer_empty_count": sum(bool(x["student_full_answer_empty"]) for x in lti_rows)},
            "zero_reward_steps": sum(x["zero_reward"] for x in outputs),
            "skipped_zero_objective_steps": sum(x["skipped_zero_objective"] for x in outputs),
            "zero_reward_gradient_failures": 0,
            "per_step": outputs}


def _safe_route_summary(value):
    if value is None:
        return None
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value if value in {"natural", "controlled", "additive", "receiver"} else None
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if any(token in str(key).casefold() for token in ("text", "prompt", "token_id", "example")):
                continue
            safe = _safe_route_summary(item)
            if safe is not None:
                result[str(key)] = safe
        return result
    # Never serialize per-token route arrays or arbitrary text.
    return None


def gold_ce_loss(logits, input_ids, answer_mask, tail_runtime):
    """Full-vocabulary causal CE over gold answer positions including EOS."""
    return tail_runtime.causal_token_ce(logits, input_ids, answer_mask)


def run_gold_replay_step(model, context, row, tokenizer, classes, common, cfg, torch,
                         device, budget, tail_runtime, *, apply_loss):
    """Compute one selected-example CE update through a caller-owned gold optimizer.

    The caller supplies exactly the IDs/counts used by its matched Sleep schedule;
    this helper does not sample, choose examples, or alter the optimizer scope.
    """
    classes = _validate_classes(classes)
    item = common.tokenize_row(row, tokenizer, classes, tokenizer.eos_token_id,
                               int(cfg["max_sequence_length"]))
    ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    answer_mask = torch.tensor([[int(x != -100) for x in item["labels"]]], dtype=torch.bool, device=device)
    _append_count(budget, mask)
    cache = tail_runtime.capture(model, ids, mask, device=device)
    logits = tail_runtime.tail_logits(model, context, cache)
    loss, tokens = gold_ce_loss(logits, ids, answer_mask, tail_runtime)
    if not bool(torch.isfinite(loss)) or not bool(loss.requires_grad):
        raise RuntimeError("gold replay produced nonfinite or nondifferentiable CE")
    backward = getattr(budget, "backward", None) if budget is not None else None
    update = getattr(budget, "update", None) if budget is not None else None
    if callable(backward): backward(answer_mask.long())
    if callable(update): update()
    apply_loss(loss, int(tokens))
    row_id = str(row.get("id", ""))
    return {"loss": float(loss.detach().float().cpu()), "token_count": int(tokens),
            "example_id_sha256": hashlib.sha256(row_id.encode("utf-8")).hexdigest() if row_id else None,
            "class_id": int(row["intent_id"])}
