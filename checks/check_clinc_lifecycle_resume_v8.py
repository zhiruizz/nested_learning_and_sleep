"""Explicit-budget CPU resume proof for CLINC V8 lifecycle transactions.

Budget: 8 AdamW.step calls total (two before checkpoint, two three-step branches). Not collected
as part of pytest's routine per-session update cap.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.clinc_lifecycle_v8 import SleepTransaction, WakeController

MAX_ADAMW_STEPS = 8


def _toy_context_type():
    path = Path(__file__).resolve().parents[1] / "tests" / "test_clinc_lifecycle_v8_cpu.py"
    spec = importlib.util.spec_from_file_location("clinc_lifecycle_test_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load CPU lifecycle fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Context


def _snapshot(context):
    return {name: param.detach().clone() for name, param in context.named_parameters()}


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _rng_draws(tx):
    return (random.random(), float(np.random.random()), torch.rand(1),
            torch.rand(1, generator=tx.generator))


def _sleep_branch(tx, name):
    draws = []
    for _ in range(3):
        py, npy, torchy, gen = _rng_draws(tx)
        draws.append((py, npy, torchy.clone(), gen.clone()))
        p = dict(tx.student.named_parameters())[name]
        target = py + npy + float(torchy) + float(gen)
        tx.optimize((p - target).square().sum(), "lti", tokens=1)
    return draws


def main():
    random.seed(20261107); np.random.seed(20261107); torch.manual_seed(20261107)
    context = _toy_context_type()()
    tx = SleepTransaction(context, sender=1,
                          prospective={"memory.1.a": torch.tensor([2.0]),
                                       "memory.1.b": torch.tensor([3.0])},
                          lr=0.005, weight_decay=0.0)
    name = "memory.2.experts.slots.1.b"
    for target in (0.75, 0.25):
        p = dict(tx.student.named_parameters())[name]
        tx.optimize((p - target).square().sum(), "lti", tokens=1)
    resume = tx.snapshot()
    saved_optimizer = _owned(resume.optimizer_state)
    saved_teacher = _owned(resume.teacher_state)
    saved_student = _owned(resume.student_state)
    first_draws = _sleep_branch(tx, name)
    first_branch = _snapshot(tx.student)
    first_adam = _owned(tx.optimizer.state_dict())
    first_counters = dict(tx.counters)
    first_banks = _owned(resume.bank_states)

    replay = SleepTransaction.from_snapshot(context, resume)
    second_draws = _sleep_branch(replay, name)
    second_branch = _snapshot(replay.student)
    if first_branch.keys() != second_branch.keys() or any(
            not torch.equal(first_branch[key], second_branch[key]) for key in first_branch):
        raise AssertionError("restored branch diverged from uninterrupted branch")
    if not _equal(first_draws, second_draws):
        raise AssertionError("Python/NumPy/Torch/generator RNG replay diverged")
    if not _equal(first_adam, replay.optimizer.state_dict()):
        raise AssertionError("restored Adam moments/step counters diverged")
    if first_counters != replay.counters or replay.counters["adamw_steps"] != 5:
        raise AssertionError(f"unexpected restored counters: {replay.counters}")
    if not _equal(saved_optimizer, resume.optimizer_state):
        raise AssertionError("snapshot Adam state aliased future optimizer steps")
    if not _equal(saved_teacher, resume.teacher_state) or not _equal(saved_student, resume.student_state):
        raise AssertionError("snapshot context tensors aliased subsequent steps")
    if not _equal(first_banks, resume.bank_states):
        raise AssertionError("snapshot routing/generator metadata aliased later changes")
    if not _equal(tx.teacher.state_dict(), replay.teacher.state_dict()):
        raise AssertionError("teacher changed across restored branches")
    if not _equal(context.state_dict(), replay.live.state_dict()):
        raise AssertionError("live context changed across sleep branches")

    # No AdamW calls: snapshot after one of two wake ticks, then replay the due
    # boundary. Proposals must be the same absolute values and remain pending.
    wake = WakeController(context, periods=(2, 4, 6, 8, None), base_lr=0.05, fast_lr=0.02)
    wake.prepare_wake()
    params = dict(context.named_parameters())
    grads = {"memory.0.a": torch.ones_like(params["memory.0.a"]),
             "memory.0.b": torch.full_like(params["memory.0.b"], 2.0),
             "memory.0.experts.slots.0.a": torch.ones_like(params["memory.0.experts.slots.0.a"])}
    wake.record_tick(grads, token_count=1)
    wake_snap = wake.snapshot()
    pending_copy = _owned(wake_snap.pending)
    wake.record_tick(grads, token_count=9)
    proposal_a = wake.proposals()[0]
    if not _equal(pending_copy, wake_snap.pending):
        raise AssertionError("wake snapshot pending gradients aliased future ticks")
    wake_replay = WakeController(context, periods=(2, 4, 6, 8, None), base_lr=0.05, fast_lr=0.02)
    wake_replay.restore(wake_snap)
    wake_replay.record_tick(grads, token_count=9)
    proposal_b = wake_replay.proposals()[0]
    if not _equal(proposal_a, proposal_b):
        raise AssertionError("partial-period wake snapshot produced a different absolute proposal")
    if not _equal(proposal_b, wake_replay.proposals()[0]):
        raise AssertionError("requesting a proposal consumed pending gradients")

    guarded = SleepTransaction(context, sender=1,
                               prospective={"memory.1.a": torch.tensor([2.0]),
                                            "memory.1.b": torch.tensor([3.0])})
    frozen = dict(guarded.student.named_parameters())["memory.0.a"]
    with torch.no_grad():
        frozen.add_(1.0)
    receiver_b = dict(guarded.student.named_parameters())["memory.2.experts.slots.1.b"]
    try:
        guarded.optimize(receiver_b.square().sum(), "gkd", tokens=1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("frozen-domain mutation escaped isolation guard")
    if guarded.counters["adamw_steps"] != 0:
        raise AssertionError("invalid frozen-domain state reached AdamW")

    if MAX_ADAMW_STEPS != 8:
        raise AssertionError("declared proof budget changed")
    print("resume proof passed: 8 AdamW.step calls; exact Adam/RNG replay and pending wake restore")


def _owned(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _owned(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_owned(v) for v in value)
    return value


if __name__ == "__main__":
    main()
