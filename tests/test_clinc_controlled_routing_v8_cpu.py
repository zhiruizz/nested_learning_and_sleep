"""CPU-only checks for the explicitly controlled 0.5 mix intervention.

This policy is a routing ablation, not natural Sleep or a paper-default policy.
There is one AdamW.step in this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from src.nested_sleep.experts import SparseExperts
from tools.clinc_controlled_routing_v8 import (
    ControlledSleepSnapshot, ControlledSleepTransaction, ControlledWakeController,
)
from tests.test_clinc_lifecycle_v8_cpu import Context, assert_state_equal
from tools.clinc_lifecycle_v8 import SleepTransaction


def test_controlled_mix_adds_fresh_receiver_gradient_when_natural_excludes_it():
    bank = SparseExperts(dim=1, rank=1, capacity=2)
    bank.activate(0); bank.activate(1)
    with torch.no_grad():
        bank.slots[0].a.fill_(1.0); bank.slots[0].b.fill_(1.0)
        bank.slots[1].a.fill_(1.0); bank.slots[1].b.fill_(2.0)
        bank.slots[0].route.fill_(1.0); bank.slots[0].priority.fill_(2.0)
        bank.slots[1].route.fill_(-1.0); bank.slots[1].priority.zero_()
    x = torch.tensor([[2.0]])
    natural = bank(x)
    assert bank.last_routes and all(r["slot"] == 0 for r in bank.last_routes)
    bank.zero_grad(set_to_none=True)
    bank.route_override = {"mix": 1, "fraction": .5}
    mixed = bank(x)
    assert torch.allclose(mixed, .5 * natural + .5 * bank.slots[1](x))
    mixed.sum().backward()
    assert bank.slots[1].b.grad is not None and torch.count_nonzero(bank.slots[1].b.grad) > 0


def test_controlled_wake_accepts_only_declared_half_mixes():
    context = Context()
    context.banks()[1].route_override = {"mix": 0, "fraction": .5}
    wake = ControlledWakeController(context, periods=(2, 3, 5, 7, None), base_lr=.1, fast_lr=.01)
    allowed = wake.prepare_wake()
    assert "memory.0.experts.slots.0.a" in allowed
    with pytest.raises(ValueError):
        context.banks()[2].route_override = {"force": 0}
        wake.prepare_wake()
    context.banks()[2].route_override = {"mix": 0, "fraction": .25}
    with pytest.raises(ValueError):
        wake.prepare_wake()
    context.banks()[0].route_override = {"mix": 0, "fraction": .5}
    with pytest.raises(ValueError):
        wake.prepare_wake()
    context.banks()[0].route_override = {"mix": 0, "fraction": .5}
    context.banks()[1].route_override = {"mix": 0, "fraction": .5}
    context.banks()[2].route_override = None
    with pytest.raises(ValueError):
        ControlledSleepTransaction(context, sender=1,
                                   prospective={"memory.1.a": torch.tensor([2.0]),
                                                "memory.1.b": torch.tensor([3.0])})


def test_controlled_sleep_preserves_teacher_mix_and_resets_only_sender_mix():
    live = Context()
    live.banks()[1].route_override = {"mix": 0, "fraction": .5}
    live.banks()[3].route_override = {"mix": 0, "fraction": .5}
    before = deepcopy(live.state_dict())
    tx = ControlledSleepTransaction(
        live, sender=1,
        prospective={"memory.1.a": torch.tensor([2.0]),
                     "memory.1.b": torch.tensor([3.0])},
        lr=.01, weight_decay=0.)
    assert tx.teacher.banks()[1].route_override == {"mix": 0, "fraction": .5}
    assert tx.student.banks()[1].route_override is None
    assert tx.student.banks()[2].route_override == {"mix": tx.slot, "fraction": .5}
    assert tx.student.banks()[3].route_override == {"mix": 0, "fraction": .5}
    assert tx.student.banks()[1].slots[0].b.count_nonzero() == 0
    assert tx.allowed_names == {f"memory.2.experts.slots.{tx.slot}.a",
                                f"memory.2.experts.slots.{tx.slot}.b"}
    assert not any(p.requires_grad for p in tx.teacher.parameters())
    assert_state_equal(before, live.state_dict())
    ptrs = [{p.untyped_storage().data_ptr() for p in c.parameters()}
            for c in (live, tx.teacher, tx.student)]
    assert not (ptrs[0] & ptrs[1] or ptrs[0] & ptrs[2] or ptrs[1] & ptrs[2])

    fresh_b = dict(tx.student.named_parameters())[f"memory.2.experts.slots.{tx.slot}.b"]
    tx.optimize(fresh_b.square().sum() + fresh_b.sum(), "gkd", tokens=1)
    assert tx.counters["adamw_steps"] == 1
    assert_state_equal(before, live.state_dict())

    snap = tx.snapshot()
    assert isinstance(snap, ControlledSleepSnapshot)
    restored = ControlledSleepTransaction.from_snapshot(live, snap)
    assert restored.student.banks()[2].route_override == {"mix": restored.slot, "fraction": .5}
    assert restored.teacher.banks()[1].route_override == {"mix": 0, "fraction": .5}
    wrong = replace(snap, student_routes=deepcopy(snap.student_routes))
    wrong.student_routes[restored.receiver] = None
    with pytest.raises(ValueError):
        ControlledSleepTransaction.from_snapshot(live, wrong)
    with pytest.raises((TypeError, ValueError)):
        SleepTransaction.from_snapshot(live, snap)
