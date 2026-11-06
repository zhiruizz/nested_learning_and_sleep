"""Pre-fast-update Sleep transaction for generalized AG News v4 boundaries.

The caller defers the fast CMS proposal at every four-update boundary.  The
slow proposal is applied first only when its own 16-update clock is due.  The
24 row pool is used as token prefixes only; labels and class metadata are
deliberately ignored by the optimizer.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src").resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nested_sleep.lifecycle import SleepSession, fingerprint
from nested_sleep.lm import isolated_clone,execution_state


GKD_UPDATES = 16
LTI_UPDATES = 8
DREAM_TOKENS = 2
SLEEP_LR = 0.003
RECEIVER_MIX = 0.5


def _hash_tree(value) -> str:
    """Stable hash for nested controller state containing tensors and scalars."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda x: str(x)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                visit(child)
        else:
            digest.update(type(item).__name__.encode() + b":" + repr(item).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def _active_slots(model):
    return [sum(bool(slot.active) for slot in bank.slots) for bank in model.banks()]


def _bank_slot_ids(model, level):
    descriptor = model.level_descriptors()[level]
    bank = model.get_submodule(descriptor["bank"])
    return [index for index, slot in enumerate(bank.slots) if bool(slot.active)]


def _pending_slow_hash(controller) -> str:
    return _hash_tree({"count": controller.counts[1], "pending": controller.pending[1]})


def _validate_proposal(model, level, proposal):
    expected = set(model.level_descriptors()[level]["base"])
    if not isinstance(proposal, dict) or set(proposal) != expected:
        raise ValueError(f"level {level} proposal scope mismatch")
    params = dict(model.named_parameters())
    for name, value in proposal.items():
        if not torch.is_tensor(value) or value.shape != params[name].shape:
            raise ValueError(f"level {level} proposal shape mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"level {level} proposal is nonfinite: {name}")


def consolidate(model, controller, proposals, pool, event_seed, meter, stage,
               apply_fast_before_teacher=False):
    """Run one isolated 24-update Sleep event and return committed candidate state.

    Accepted scheduler states are fast-only at slow counts 4, 8, or 12, and
    simultaneous fast+slow at slow count 16.  Fast level 0 remains prospective
    inside SleepSession; a due slow proposal is applied to the candidate first
    so the teacher sees the just-committed slower state.  The optional
    ``apply_fast_before_teacher`` branch is an order control: it commits the
    deferred fast proposal on the private candidate before constructing the
    teacher, then passes the identity fast proposal to SleepSession.
    """
    if len(model.level_descriptors()) != 2:
        raise ValueError("v4 Sleep requires exactly two CMS levels")
    if tuple(controller.periods)!=(4,16):
        raise ValueError('v4 controller periods must be4/16')
    counts = list(controller.counts)
    if counts[0] != 4 or counts[1] not in (4, 8, 12, 16):
        raise ValueError(f"unsupported pre-update clock state: {counts}")
    expected = {0, 1} if counts[1] == 16 else {0}
    if set(proposals) != expected:
        raise ValueError(f"proposals {sorted(proposals)} do not match due levels {sorted(expected)}")
    for level in sorted(expected):
        _validate_proposal(model, level, proposals[level])
    if len(pool) != GKD_UPDATES + LTI_UPDATES:
        raise ValueError("Sleep requires exactly 24 prefix rows")
    if any(not isinstance(row, dict) or "tokens" not in row for row in pool):
        raise ValueError("each Sleep prefix row must be a mapping containing tokens")

    original_model_hash = fingerprint(model)
    original_execution=execution_state(model)
    original_controller_hash = _hash_tree(controller.state_dict())
    pending_before = _pending_slow_hash(controller)
    starting_counts = list(counts)
    candidate = isolated_clone(model, share_backbone=True)
    candidate_controller = type(controller).restore(candidate, controller.state_dict())

    slow_applied = 1 in expected
    if slow_applied:
        candidate_controller.apply(candidate, 1, proposals[1])
    prospective_fast = proposals[0]
    if apply_fast_before_teacher:
        candidate_controller.apply(candidate, 0, proposals[0])
        params = dict(candidate.named_parameters())
        prospective_fast = {
            name: params[name].detach().clone()
            for name in candidate.level_descriptors()[0]["base"]
        }
    session = SleepSession(
        candidate,
        0,
        prospective_fast,
        seed=int(event_seed),
        lr=SLEEP_LR,
        dream_tokens=DREAM_TOKENS,
        share_backbone=True,
        reject_zero_gradient=True,
        receiver_mix_fraction=RECEIVER_MIX,
    )
    receiver_bank = session.student.banks()[1]
    initial_student_fingerprint=fingerprint(session.student)
    teacher_fingerprint=fingerprint(session.teacher)
    if receiver_bank.route_override != {"mix": session.slot, "fraction": RECEIVER_MIX}:
        raise RuntimeError("Sleep student receiver mix policy was not installed")

    device = next(model.parameters()).device
    meter.stage = stage
    for row in pool[:GKD_UPDATES]:
        prefix = torch.as_tensor(row["tokens"], dtype=torch.long, device=device).reshape(1, -1)
        session.seed_step(prefix, owner=None)
    for row in pool[GKD_UPDATES:]:
        prefix = torch.as_tensor(row["tokens"], dtype=torch.long, device=device).reshape(1, -1)
        session.lti_step(prefix)

    kinds = [record["kind"] for record in session.records]
    teacher_updates = sum(kind == "gkd_teacher" for kind in kinds)
    student_updates = sum(kind == "gkd_student_on_policy" for kind in kinds)
    if len(session.records) != 24 or (teacher_updates, student_updates) != (8, 8):
        raise RuntimeError("GKD owner=None schedule did not produce 8/8 alternating owners")
    positive = sum(float(record["grad_norm"]) > 0.0 for record in session.records)
    zero_objectives = sum(abs(float(record["loss"])) <= 1e-6 for record in session.records)

    session.finish_consolidation()
    trained_fingerprint=fingerprint(session.student)
    committed = session.commit()
    if fingerprint(committed)!=trained_fingerprint:raise RuntimeError('commit changed trained student')
    candidate_controller.consume(0)
    expected_counts = [0, 0] if slow_applied else [0, starting_counts[1]]
    if list(candidate_controller.counts) != expected_counts:
        raise RuntimeError(f"Sleep consumed unexpected CMS clocks: {candidate_controller.counts}")
    pending_after = _pending_slow_hash(candidate_controller)
    if not slow_applied and pending_after != pending_before:
        raise RuntimeError("Sleep changed pending slow accumulator before its boundary")
    if fingerprint(model) != original_model_hash or _hash_tree(controller.state_dict()) != original_controller_hash:
        raise RuntimeError("isolated Sleep changed the original model or controller")
    if execution_state(model)!=original_execution:raise RuntimeError('isolated Sleep changed live execution policy')

    summary = {
        "scope": "pre_fast_base_update_sleep_v4",
        "teacher_order": "post_fast" if apply_fast_before_teacher else "pre_fast",
        "teacher_fingerprint":teacher_fingerprint,
        "initial_student_fingerprint":initial_student_fingerprint,
        "committed_student_fingerprint":trained_fingerprint,
        "gkd_updates": GKD_UPDATES,
        "lti_updates": LTI_UPDATES,
        "total_adam_updates": GKD_UPDATES + LTI_UPDATES,
        "dream_updates": 0,
        "dream_tokens": DREAM_TOKENS,
        "lr": SLEEP_LR,
        "receiver_mix_fraction": RECEIVER_MIX,
        "slow_level1_applied_before_teacher": slow_applied,
        "fast_level0_applied_before_teacher": bool(apply_fast_before_teacher),
        "fast_level0_applied_prospectively": not bool(apply_fast_before_teacher),
        "gkd_teacher_updates": teacher_updates,
        "gkd_student_on_policy_updates": student_updates,
        "positive_gradient_updates": positive,
        "zero_objective_updates": zero_objectives,
        "zero_objective_with_zero_gradient_updates": sum(
            abs(float(record["loss"])) <= 1e-6 and float(record["grad_norm"]) == 0.0
            for record in session.records
        ),
        "gradient_norm_min": min(float(record["grad_norm"]) for record in session.records),
        "gradient_norm_max": max(float(record["grad_norm"]) for record in session.records),
        "sender_level": 0,
        "sender_reset_slot_ids": list(session.reset_slots),
        "receiver_level": 1,
        "receiver_slot": session.slot,
        "receiver_active_slots": _bank_slot_ids(committed, 1),
        "receiver_route_policy": dict(committed.banks()[1].route_override or {}),
        "input_fast_count": starting_counts[0],
        "input_slow_count": starting_counts[1],
        "output_controller_counts": list(candidate_controller.counts),
        "pending_slow_hash_before": pending_before,
        "pending_slow_hash_after": pending_after,
        "active_slots_by_bank": _active_slots(committed),
        "candidate_committed": True,
    }
    return committed, candidate_controller, summary
