"""Small CPU lifecycle checks; one AdamW.step total in this module."""
from __future__ import annotations

from copy import deepcopy

import torch
import pytest
from torch import nn

from tools.clinc_lifecycle_v8 import SleepTransaction, WakeController


class Slot(nn.Module):
    def __init__(self, active=False):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([0.0]))
        self.register_buffer("active", torch.tensor(active))
        self.register_buffer("priority", torch.tensor(1.0 if active else 0.0))
        self.register_buffer("generation", torch.tensor(1 if active else 0))


class Bank(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([Slot(True), Slot(), Slot()])
        self.route_override = None

    def reset_active(self):
        with torch.no_grad():
            for slot in self.slots:
                if bool(slot.active):
                    slot.b.zero_()
                    slot.active.fill_(False)
                    slot.priority.zero_()
                    slot.generation.add_(1)

    def activate(self, index=None):
        if index is None:
            index = next(i for i, slot in enumerate(self.slots) if not bool(slot.active))
        slot = self.slots[index]
        if bool(slot.active):
            raise RuntimeError("already active")
        with torch.no_grad():
            slot.active.fill_(True)
            slot.generation.add_(1)
            slot.priority.fill_(max(float(s.priority) for s in self.slots) + 1)
        return index


class Level(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([1.0]))
        self.experts = Bank()


class Context:
    def __init__(self):
        self.memory = nn.ModuleList([Level() for _ in range(5)])
        self.readout_a = nn.Parameter(torch.tensor([1.0]))
        self.readout_b = nn.Parameter(torch.tensor([0.0]))
        self.readout_fp32_accumulation = True

    def parameters(self):
        yield from self.memory.parameters()
        yield self.readout_a
        yield self.readout_b

    def named_parameters(self):
        for name, p in self.memory.named_parameters():
            yield f"memory.{name}", p
        yield "readout_a", self.readout_a
        yield "readout_b", self.readout_b

    def banks(self):
        return [level.experts for level in self.memory]

    def state_dict(self):
        return {"memory": deepcopy(self.memory.state_dict()),
                "readout_a": self.readout_a.detach().clone(),
                "readout_b": self.readout_b.detach().clone(),
                "readout_fp32_accumulation": self.readout_fp32_accumulation,
                "routing": [deepcopy(b.route_override) for b in self.banks()],
                "trainable": {n: p.requires_grad for n, p in self.named_parameters()}}

    def load_state_dict(self, state):
        self.memory.load_state_dict(state["memory"])
        with torch.no_grad():
            self.readout_a.copy_(state["readout_a"])
            self.readout_b.copy_(state["readout_b"])
        self.readout_fp32_accumulation = state["readout_fp32_accumulation"]
        for bank, route in zip(self.banks(), state["routing"]):
            bank.route_override = deepcopy(route)
        for name, p in self.named_parameters():
            p.requires_grad_(state["trainable"][name])


def digest(context):
    return [(name, p.detach().clone()) for name, p in context.named_parameters()]


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_wake_lr_weighted_sum_retains_until_explicit_apply_and_freezes_terminal():
    context = Context()
    base = dict((n, p.detach().clone()) for n, p in context.named_parameters())
    wake = WakeController(context, periods=(2, 3, 5, 7, None), base_lr=0.1, fast_lr=0.2)
    wake_names = wake.prepare_wake()
    assert {"memory.0.a", "memory.0.b", "memory.1.a", "memory.1.b",
            "memory.2.a", "memory.2.b", "memory.3.a", "memory.3.b"}.issubset(wake_names)
    fast_name = "memory.0.experts.slots.0.a"
    terminal = "memory.4.a"
    base_grads = {name: torch.ones_like(p) for name, p in context.named_parameters()
                  if name in {"memory.0.a", "memory.0.b", fast_name}}
    assert wake.record_tick(base_grads, token_count=1) == {}
    assert wake.record_tick(base_grads, token_count=7) == {}
    proposal = wake.proposals()
    assert 0 in proposal
    assert torch.equal(proposal[0]["memory.0.a"], base["memory.0.a"] - 0.2)
    assert torch.equal(proposal[0]["memory.0.b"], base["memory.0.b"] - 0.2)
    assert torch.equal(dict(context.named_parameters())["memory.0.a"], base["memory.0.a"])
    assert "memory.4.a" not in proposal.get(4, {})
    first = {k: v.clone() for k, v in proposal[0].items()}
    assert all(torch.equal(first[k], v) for k, v in wake.proposals()[0].items())
    wake.apply(0, proposal[0])
    assert torch.equal(dict(context.named_parameters())["memory.0.a"], base["memory.0.a"] - 0.2)
    assert wake.proposals().get(0, {}) == {}
    assert torch.equal(dict(context.named_parameters())[fast_name], base[fast_name] - 0.4)
    assert torch.equal(dict(context.named_parameters())[terminal], base[terminal])


def test_sleep_sender_is_layer_and_receiver_is_next_layer_fresh_slot():
    live = Context()
    before = digest(live)
    sender_level = 1
    receiver_level = sender_level + 1
    prospective = {f"memory.{sender_level}.a": torch.tensor([3.0]),
                   f"memory.{sender_level}.b": torch.tensor([4.0])}
    tx = SleepTransaction(live, sender=sender_level, prospective=prospective,
                          lr=0.01, weight_decay=0.0)
    student = tx.student
    params = dict(student.named_parameters())
    assert torch.equal(params[f"memory.{sender_level}.a"], torch.tensor([3.0]))
    assert torch.equal(params[f"memory.{sender_level}.b"], torch.tensor([4.0]))
    assert [i for i, slot in enumerate(student.banks()[sender_level].slots) if bool(slot.active)] == []
    assert not torch.equal(student.banks()[sender_level].slots[0].a,
                           live.banks()[sender_level].slots[0].a)
    assert torch.count_nonzero(student.banks()[sender_level].slots[0].b) == 0
    assert [i for i, slot in enumerate(student.banks()[receiver_level].slots) if bool(slot.active)] == [0, 1]
    # Only sender-layer bank resets; receiver uses its first inactive slot.
    for level in (0, 3, 4):
        assert [i for i, slot in enumerate(student.banks()[level].slots) if bool(slot.active)] == [0]
    assert not any(p.requires_grad for p in tx.teacher.parameters())
    live_ptrs = {p.untyped_storage().data_ptr() for p in live.parameters()}
    teacher_ptrs = {p.untyped_storage().data_ptr() for p in tx.teacher.parameters()}
    student_ptrs = {p.untyped_storage().data_ptr() for p in student.parameters()}
    assert not (live_ptrs & teacher_ptrs or live_ptrs & student_ptrs or teacher_ptrs & student_ptrs)
    teacher_before = deepcopy(tx.teacher.state_dict())
    live_state_before = deepcopy(live.state_dict())
    fresh_name = f"memory.{receiver_level}.experts.slots.1.b"
    allowed = {f"memory.{receiver_level}.experts.slots.1.a", fresh_name}
    assert {n for n, p in student.named_parameters() if p.requires_grad} == allowed
    frozen_before = {n: p.detach().clone() for n, p in student.named_parameters() if n not in allowed}
    loss = (dict(student.named_parameters())[fresh_name] - 1.0).square().sum()
    tx.optimize(loss, "gkd", tokens=1)
    assert tx.counters["adamw_steps"] == 1 and tx.counters["gkd_calls"] == 1
    assert all(torch.equal(dict(student.named_parameters())[n], p)
               for n, p in frozen_before.items())
    assert all(torch.equal(dict(live.named_parameters())[n], p) for n, p in before)
    assert_state_equal(teacher_before, tx.teacher.state_dict())
    assert_state_equal(live_state_before, live.state_dict())
    assert tx.commit() is student


def test_sleep_rejects_prospective_updates_outside_sender_base_pair():
    live = Context()
    with pytest.raises((ValueError, KeyError)):
        SleepTransaction(live, sender=1,
                         prospective={"memory.1.a": torch.tensor([2.0]),
                                      "memory.1.b": torch.tensor([3.0]),
                                      "memory.0.a": torch.tensor([4.0])})
