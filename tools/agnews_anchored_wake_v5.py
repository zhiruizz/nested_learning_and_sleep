"""Exploratory anchored old-task wake update for AGNews v5.

This is an engineering control, not the paper's Sleep algorithm. The primary
new-task CE remains the v3/v4 objective. A fixed-weight old-task term is
backpropagated before a single global clip and a single CMS collect.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import torch
import torch.nn.functional as F

import agnews_v3_common as c
from nested_sleep.lm import execution_state, immutable_versions


def _collate(rows, device, pad_id, *, use_labels):
    """Collate rows while making gold fields unreachable in soft-KD mode."""
    prepared = []
    for row in rows:
        item = {"tokens": row["tokens"]}
        if use_labels:
            item["marker"] = int(row["marker"])
            item["label"] = int(row["label"])
        else:
            # The KL branch consumes only raw prefix tokens and teacher outputs.
            item["marker"] = 0
            item["label"] = 0
        prepared.append(item)
    return c.engine.shared.collate(prepared, device, pad_id)


def _adapter_fingerprint(model):
    # Production Llama path hashes only adapters; Native fallback is CPU-fixture-sized.
    state = model.adapter_state_dict() if hasattr(model, "adapter_state_dict") else model.state_dict()
    return c.engine.shared.tensor_tree_hash(state)


def _kl_from_logp(teacher_logp, student_logp):
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)


def anchored_step(
    model,
    controller,
    new_rows,
    old_rows,
    pad_id,
    meter,
    stage,
    teacher,
    mode="soft_kd",
    defer_due=False,
):
    """Apply one combined new-CE + old-task anchored CMS wake update.

    ``old_rows`` must be selected by the caller using a fixed, external schedule
    (the intended v5 design uses two examples per old class). In ``soft_kd``
    mode their gold marker/class fields are ignored: the frozen teacher greedily
    supplies the first token, then its full-vocabulary distributions at the
    first and second answer positions supervise the student at that same
    teacher-generated context. In ``gold`` mode old labels are used directly.

    The new and old losses are backpropagated separately to bound graph memory,
    then all mutable wake gradients are globally clipped once and collected
    once by the controller. If ``defer_due`` is true, due proposals are returned
    unapplied for a caller-owned boundary transaction.
    """
    if mode not in ("soft_kd", "gold"):
        raise ValueError("mode must be 'soft_kd' or 'gold'")
    if not new_rows:
        raise ValueError("new_rows must be nonempty")
    if len(old_rows) != 4 or [sum(int(r.get("class_id", -1)) == cls for r in old_rows) for cls in (0, 1)] != [2, 2]:
        raise ValueError("old_rows must contain the externally fixed two examples per old class")
    if model.training:
        raise ValueError("model must be in eval mode for deterministic wake updates")
    if teacher is None or teacher.training:
        raise ValueError("teacher must be present and in eval mode")
    if teacher is model:
        raise ValueError("teacher must be a separate frozen snapshot")
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("teacher parameters must be frozen")

    teacher_policy = execution_state(teacher)
    teacher_adapter = _adapter_fingerprint(teacher)
    teacher_immutable = immutable_versions(teacher)
    rng_before = c.engine.capture_rng()
    rng_hash_before = c.engine.shared.tensor_tree_hash(rng_before)

    device = next(model.parameters()).device
    new_ids, new_mask, new_lengths, new_markers, new_labels = _collate(
        new_rows, device, pad_id, use_labels=True
    )
    old_ids, old_mask, old_lengths, old_markers, old_labels = _collate(
        old_rows, device, pad_id, use_labels=(mode == "gold")
    )

    controller.prepare_wake(model)
    meter.stage = stage

    # Primary new-task CE matches supervised_update exactly.
    new_marked_ids, new_marked_mask = c.engine.shared.append_markers(
        new_ids, new_mask, new_lengths, new_markers, pad_id
    )
    new_logits = model.logits(new_marked_ids, new_marked_mask)
    row_idx_new = torch.arange(len(new_rows), device=device)
    new_marker_logits = new_logits[row_idx_new, new_lengths - 1].float()
    new_class_logits = new_logits[row_idx_new, new_lengths].float()
    new_loss = 0.5 * (
        F.cross_entropy(new_marker_logits, new_markers)
        + F.cross_entropy(new_class_logits, new_labels)
    )
    if not torch.isfinite(new_loss):
        raise RuntimeError("nonfinite primary new-task CE")
    new_loss.backward()
    del new_logits, new_marker_logits, new_class_logits, new_marked_ids, new_marked_mask

    if mode == "gold":
        old_marked_ids, old_marked_mask = c.engine.shared.append_markers(
            old_ids, old_mask, old_lengths, old_markers, pad_id
        )
        old_logits = model.logits(old_marked_ids, old_marked_mask)
        row_idx_old = torch.arange(len(old_rows), device=device)
        old_loss = 0.5 * (
            F.cross_entropy(old_logits[row_idx_old, old_lengths - 1].float(), old_markers)
            + F.cross_entropy(old_logits[row_idx_old, old_lengths].float(), old_labels)
        )
        del old_logits, old_marked_ids, old_marked_mask
    else:
        # Teacher first-token greedy generation is deterministic and uses raw
        # old prompt tokens only; dummy labels from _collate cannot affect it.
        with torch.no_grad():
            teacher_first_logits = teacher.logits(old_ids, old_mask)
            row_idx_old = torch.arange(len(old_rows), device=device)
            first_logp = teacher_first_logits[row_idx_old, old_lengths - 1].float().log_softmax(-1)
            generated_first = first_logp.argmax(-1)
            marked_ids, marked_mask = c.engine.shared.append_markers(
                old_ids, old_mask, old_lengths, generated_first, pad_id
            )
            teacher_second_logits = teacher.logits(marked_ids, marked_mask)
            teacher_first_logp = first_logp.detach()
            teacher_second_logp = teacher_second_logits[row_idx_old, old_lengths].float().log_softmax(-1).detach()
        del teacher_first_logits, teacher_second_logits

        # One student forward at the teacher-generated context gives both
        # next-token distributions, reducing activation/graph overhead.
        student_logits = model.logits(marked_ids, marked_mask)
        student_first_logp = student_logits[row_idx_old, old_lengths - 1].float().log_softmax(-1)
        student_second_logp = student_logits[row_idx_old, old_lengths].float().log_softmax(-1)
        old_loss = torch.stack((
            _kl_from_logp(teacher_first_logp, student_first_logp),
            _kl_from_logp(teacher_second_logp, student_second_logp),
        ), dim=1).mean()
        del student_logits, student_first_logp, student_second_logp, marked_ids, marked_mask

    if not torch.isfinite(old_loss):
        raise RuntimeError("nonfinite old-task anchor loss")
    old_loss_value = float(old_loss.detach())
    old_loss.backward()
    del old_loss

    wake_params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(wake_params, 1.0))
    if not torch.isfinite(torch.tensor(grad_norm)):
        raise RuntimeError("nonfinite combined wake gradient norm")
    proposals = controller.collect(model)
    due_levels = sorted(proposals)
    if not defer_due:
        for level in sorted(proposals, reverse=True):
            controller.apply(model, level, proposals[level])

    teacher_unchanged = (
        _adapter_fingerprint(teacher) == teacher_adapter
        and execution_state(teacher) == teacher_policy
        and immutable_versions(teacher) == teacher_immutable
        and not teacher.training
        and not any(p.requires_grad for p in teacher.parameters())
    )
    rng_hash_after = c.engine.shared.tensor_tree_hash(c.engine.capture_rng())
    if not teacher_unchanged:
        raise RuntimeError("frozen teacher changed during anchored update")
    if rng_hash_after != rng_hash_before:
        raise RuntimeError("anchored update unexpectedly changed global RNG")

    result = {
        "mode": mode,
        "new_ce": float(new_loss.detach()),
        "old_anchor_loss": old_loss_value,
        "teacher_adapter_fingerprint": teacher_adapter,
        "combined_grad_norm_preclip": grad_norm,
        "due_levels": due_levels,
        "controller_counts": list(controller.counts),
        "proposals_applied": not defer_due,
        "teacher_unchanged": True,
        "global_rng_unchanged": True,
        "logical_primary_updates": 1,
    }
    # Proposals are returned as the caller-owned boundary transaction payload;
    # no input token arrays or teacher generations are written into metrics.
    return result, proposals
