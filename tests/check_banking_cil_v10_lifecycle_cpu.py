from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from nested_sleep.experts import LowRankMemoryLevel  # noqa: E402
from banking_cil_v10_lifecycle import AdditiveSleepTransaction, AdditiveWakeController  # noqa: E402
from banking_cil_v10_routing import activate_additive_receiver, snapshot_route_policy  # noqa: E402
from clinc_lifecycle_v8 import _state_hash  # noqa: E402


class TinyContext:
    """Small v8 MutableContext-shaped CPU fixture with two memory levels."""
    def __init__(self):
        self.memory = nn.ModuleList([
            LowRankMemoryLevel(dim=6, rank=3, capacity=4),
            LowRankMemoryLevel(dim=6, rank=3, capacity=4),
        ])
        self.readout_a = nn.Parameter(torch.randn(3, 6), requires_grad=True)
        self.readout_b = nn.Parameter(torch.zeros(9, 3), requires_grad=True)
        self.readout_fp32_accumulation = True
        self.banks()[0].activate(0)
        activate_additive_receiver(self, 1, slot=0)
        with torch.no_grad():
            self.memory[0].a.fill_(0.05)
            self.memory[0].b.fill_(0.12)
            self.memory[1].a.fill_(-0.04)
            self.memory[1].b.fill_(0.08)
            self.banks()[0].slots[0].a.fill_(0.1)
            self.banks()[0].slots[0].b.fill_(0.07)
            self.banks()[1].slots[0].a.fill_(0.09)
            self.banks()[1].slots[0].b.fill_(0.06)

    def banks(self):
        return [level.experts for level in self.memory]

    def parameters(self):
        yield from self.memory.parameters()
        yield self.readout_a
        yield self.readout_b

    def named_parameters(self):
        yield from self.memory.named_parameters(prefix="memory")
        yield "readout_a", self.readout_a
        yield "readout_b", self.readout_b

    def state_dict(self):
        params = dict(self.named_parameters())
        return {
            "memory": {k: v.detach().cpu().clone() for k, v in self.memory.state_dict().items()},
            "readout_a": self.readout_a.detach().cpu().clone(),
            "readout_b": self.readout_b.detach().cpu().clone(),
            "readout_fp32_accumulation": self.readout_fp32_accumulation,
            "routing": [deepcopy(bank.route_override) for bank in self.banks()],
            "trainable": {k: bool(v.requires_grad) for k, v in params.items()},
        }

    def load_state_dict(self, state, *, strict=True):
        expected = {"memory", "readout_a", "readout_b", "readout_fp32_accumulation", "routing", "trainable"}
        if set(state) != expected:
            raise ValueError("fixture context state schema mismatch")
        result = self.memory.load_state_dict(state["memory"], strict=strict)
        with torch.no_grad():
            self.readout_a.copy_(state["readout_a"])
            self.readout_b.copy_(state["readout_b"])
        if len(state["routing"]) != len(self.banks()):
            raise ValueError("fixture routing count mismatch")
        for bank, policy in zip(self.banks(), state["routing"]):
            bank.route_override = deepcopy(policy)
        self.readout_fp32_accumulation = bool(state["readout_fp32_accumulation"])
        for name, param in self.named_parameters():
            param.requires_grad_(bool(state["trainable"][name]))
        return result


def run_context(context, x):
    for level in context.memory:
        x = level(x)
    return x


def make_tx(live, seed=91):
    prospective = {
        "memory.0.a": live.memory[0].a.detach().clone() + 0.03,
        "memory.0.b": live.memory[0].b.detach().clone() - 0.02,
    }
    return AdditiveSleepTransaction(live, 0, prospective, lr=0.01,
                                    weight_decay=0.0, clip_norm=1.0, seed=seed)


def do_gkd_step(tx, step):
    x = torch.linspace(-0.7, 0.9, 24, dtype=torch.float32).reshape(4, 6) + step * 0.01
    with torch.no_grad():
        target = run_context(tx.teacher_context(), x).clone()
    prediction = run_context(tx.student_context(), x)
    loss = (prediction - target).square().mean()
    tx.optimize(loss, "gkd", tokens=4)


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_additive_wake_preserves_pending_clocks_and_policy_on_restore(tmp_path):
    torch.manual_seed(8)
    context = TinyContext()
    ctrl = AdditiveWakeController(context, periods=(2, None), base_lr=0.01, fast_lr=0.02)
    allowed = ctrl.prepare_wake()
    grads = {name: torch.full_like(param, 0.01) for name, param in context.named_parameters()
             if name in allowed}
    ctrl.record_tick(grads, token_count=11)
    assert ctrl.clocks == {0: 1} and set(ctrl.pending) == {"memory.0.a", "memory.0.b"}
    snap = ctrl.snapshot()
    wake_file = tmp_path / "additive-wake-snapshot.pt"
    torch.save(snap, wake_file)
    snap = torch.load(wake_file, map_location="cpu", weights_only=False)
    clone = deepcopy(context)
    resumed = AdditiveWakeController(clone, periods=(2, None), base_lr=0.01, fast_lr=0.02)
    resumed.restore(snap)
    assert snapshot_route_policy(clone) == snapshot_route_policy(context)
    grads_a = {name: torch.full_like(param, 0.02) for name, param in context.named_parameters()
               if name in allowed}
    grads_b = {name: torch.full_like(param, 0.02) for name, param in clone.named_parameters()
               if name in allowed}
    ctrl.record_tick(grads_a, token_count=13)
    resumed.record_tick(grads_b, token_count=13)
    proposals_a, proposals_b = ctrl.proposals(), resumed.proposals()
    assert proposals_a.keys() == proposals_b.keys() == {0}
    for name in proposals_a[0]:
        torch.testing.assert_close(proposals_a[0][name], proposals_b[0][name], rtol=0, atol=0)
    assert ctrl.clocks == resumed.clocks and ctrl.token_counts == resumed.token_counts
    assert ctrl.pending.keys() == resumed.pending.keys()


def test_wake_restore_at_empty_fast_boundary_does_not_activate_or_rewrite_scope():
    torch.manual_seed(18)
    source = TinyContext()
    source.banks()[0].deactivate(0)
    source_hash = _state_hash(source)
    controller = AdditiveWakeController(source, periods=(2, None), base_lr=0.01, fast_lr=0.02)
    snapshot = controller.snapshot()
    target = TinyContext()
    restored = AdditiveWakeController(target, periods=(2, None), base_lr=0.01, fast_lr=0.02)
    restored.restore(snapshot)
    assert _state_hash(target) == source_hash
    assert not any(bool(slot.active) for slot in target.banks()[0].slots)
    assert restored._fast_names == set()
    # Fast-slot allocation remains an explicit next-wake action.
    restored.prepare_wake()
    assert sum(bool(slot.active) for slot in target.banks()[0].slots) == 1


def test_sleep_updates_only_fresh_receiver_and_keeps_teacher_live_immutable():
    torch.manual_seed(15)
    live = TinyContext()
    live_hash = _state_hash(live)
    tx = make_tx(live)
    assert tx.slot == 1
    assert tx.reset_slots == [0]
    assert tx.student.banks()[0].route_override is None
    assert tx.teacher.banks()[1].route_override == live.banks()[1].route_override
    assert tx.student.banks()[1].route_override["schema"] == "banking-cil-v10-additive-receivers-v1"
    assert tx.student.banks()[1].route_override["slots"] == [
        {"slot": 0, "weight": 1.0}, {"slot": tx.slot, "weight": 1.0}]
    old_names = {f"memory.1.experts.slots.0.{s}" for s in ("a", "b")}
    assert not any(dict(tx.student.named_parameters())[name].requires_grad for name in old_names)
    old_active_params = {name: p.detach().clone() for name, p in tx.student.named_parameters()
                         if name.startswith("memory.1.experts.slots.0.")}
    do_gkd_step(tx, 0)
    new_b = dict(tx.student.named_parameters())[f"memory.1.experts.slots.{tx.slot}.b"]
    assert torch.count_nonzero(new_b.detach()) > 0
    assert all(torch.equal(dict(tx.student.named_parameters())[n], p) for n, p in old_active_params.items())
    assert _state_hash(live) == live_hash
    assert tx.counters["adamw_steps"] == 1 and tx.counters["gkd_calls"] == 1


def test_sleep_disk_snapshot_restores_adam_rng_routes_and_replays_exactly(tmp_path):
    torch.manual_seed(29)
    live = TinyContext()
    tx = make_tx(live, seed=212)
    do_gkd_step(tx, 0)
    do_gkd_step(tx, 1)
    snapshot = tx.snapshot()
    path = tmp_path / "additive-sleep-snapshot.pt"
    torch.save(snapshot, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded.schema == tx.SNAPSHOT_SCHEMA
    ref = AdditiveSleepTransaction.from_snapshot(live, loaded)
    resumed = AdditiveSleepTransaction.from_snapshot(live, loaded)
    saved_payload_digest = _state_hash_payload(loaded.payload)
    for i in range(2, 5):
        do_gkd_step(ref, i)
    assert _state_hash_payload(loaded.payload) == saved_payload_digest
    for i in range(2, 5):
        do_gkd_step(resumed, i)
    assert _same(ref.student.state_dict(), resumed.student.state_dict())
    assert _same(ref.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert ref.counters == resumed.counters
    assert torch.equal(ref.generator.get_state(), resumed.generator.get_state())
    assert snapshot_route_policy(ref.teacher) == snapshot_route_policy(resumed.teacher)
    assert snapshot_route_policy(ref.student) == snapshot_route_policy(resumed.student)


def _state_hash_payload(payload):
    from clinc_lifecycle_v8 import _hash
    return _hash({"teacher": payload.teacher_state, "student": payload.student_state,
                  "optimizer": payload.optimizer_state, "counters": payload.counters,
                  "generator": payload.generator_state, "cpu_rng": payload.cpu_rng,
                  "python_rng": payload.python_rng, "numpy_rng": payload.numpy_rng})


def test_gold_uses_same_receiver_reset_and_counts_separately():
    torch.manual_seed(31)
    live_a = TinyContext()
    torch.manual_seed(31)
    live_b = TinyContext()
    tx_gkd, tx_gold = make_tx(live_a, seed=73), make_tx(live_b, seed=73)
    assert _same(tx_gkd.student.state_dict(), tx_gold.student.state_dict())
    x = torch.linspace(-0.4, 0.8, 24).reshape(4, 6)
    logits = run_context(tx_gold.student_context(), x)
    targets = torch.tensor([1, 2, 0, 1])
    loss = torch.nn.functional.cross_entropy(logits, targets)
    tx_gold.optimize(loss, "gold", tokens=4)
    assert tx_gold.counters["gold_calls"] == 1
    assert tx_gold.counters["gkd_calls"] == 0
    assert torch.count_nonzero(dict(tx_gold.student.named_parameters())[
        f"memory.1.experts.slots.{tx_gold.slot}.b"]) > 0


def test_receiver_capacity_exhaustion_fails_without_changing_live():
    torch.manual_seed(37)
    live = TinyContext()
    bank = live.banks()[1]
    # Fill every remaining slot; the two already active/additive receivers are retained.
    for index, slot in enumerate(bank.slots):
        if not bool(slot.active):
            bank.activate(index)
            with torch.no_grad():
                slot.b.fill_(0.01)
            # Keep these legacy active slots out of the additive policy only
            # for the fixture's capacity failure; route validation still sees
            # the original additive receiver and natural old experts.
    original = _state_hash(live)
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        make_tx(live)
    assert _state_hash(live) == original


def test_zero_gradient_is_rejected_before_physical_optimizer_step():
    torch.manual_seed(41)
    tx = make_tx(TinyContext())
    # Sum of squares around a detached identical target has nonzero output but
    # its derivative is exactly zero at the current student state.
    x = torch.randn(3, 6)
    prediction = run_context(tx.student_context(), x)
    target = prediction.detach()
    loss = (prediction - target).square().mean()
    before = tx.counters["adamw_steps"]
    with pytest.raises(RuntimeError, match="gradient norm must be positive"):
        tx.optimize(loss, "gkd", tokens=3)
    assert tx.counters["adamw_steps"] == before
    assert tx.counters["zero_gradient_steps"] == 1
