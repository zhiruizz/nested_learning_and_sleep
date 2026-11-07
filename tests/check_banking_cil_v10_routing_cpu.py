from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nested_sleep.experts import SparseExperts  # noqa: E402

import importlib.util
spec = importlib.util.spec_from_file_location(
    "banking_cil_v10_routing_test", ROOT / "tools" / "banking_cil_v10_routing.py")
routing = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(routing)


class Context:
    def __init__(self, *banks):
        self._banks = list(banks)

    def banks(self):
        return self._banks


def make_bank():
    bank = SparseExperts(dim=5, rank=3, capacity=5)
    bank.activate(0)
    with torch.no_grad():
        bank.slots[0].a.copy_(torch.arange(15, dtype=torch.float32).reshape(3, 5) / 17)
        bank.slots[0].b.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 13)
        bank.slots[0].route.copy_(torch.tensor([0.3, -0.4, 0.2, 0.1, -0.2]))
    return bank


def test_zero_b_receiver_route_expansion_is_function_preserving():
    torch.manual_seed(2)
    bank = make_bank()
    context = Context(bank)
    x = torch.randn(7, 5)
    before = bank(x).detach().clone()
    slot = routing.activate_additive_receiver(context, 0, slot=3)
    after = bank(x).detach()
    assert slot == 3
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert torch.count_nonzero(bank.slots[slot].b) == 0


def test_fresh_receiver_b_gets_gradient_and_old_slots_are_frozen():
    bank = make_bank()
    context = Context(bank)
    slot = routing.activate_additive_receiver(context, 0, slot=2)
    assert all(not p.requires_grad for p in bank.slots[0].parameters())
    assert all(p.requires_grad for p in bank.slots[slot].parameters())
    x = torch.randn(6, 5)
    bank(x).square().sum().backward()
    assert bank.slots[slot].b.grad is not None
    assert float(bank.slots[slot].b.grad.norm()) > 0
    assert all(p.grad is None for p in bank.slots[0].parameters())


def test_multiple_additive_slots_do_not_dilute_old_softmax_gate():
    bank = make_bank()
    context = Context(bank)
    x = torch.randn(4, 5)
    expected = bank(x).detach()
    first = routing.activate_additive_receiver(context, 0, slot=2)
    second = routing.activate_additive_receiver(context, 0, slot=4)
    with torch.no_grad():
        bank.slots[first].b.fill_(0.15)
        bank.slots[second].b.fill_(-0.07)
    old_only = bank(x).detach()
    with torch.no_grad():
        expected = expected + bank.slots[first](x) + bank.slots[second](x)
    torch.testing.assert_close(old_only, expected, rtol=1e-6, atol=1e-7)
    assert [r["slot"] for r in bank.last_routes if r["kind"] == "v10_additive_receiver"] == [first, second]
    assert all(r["weight"] == 1.0 for r in bank.last_routes if r["kind"] == "v10_additive_receiver")


def test_route_policy_roundtrip_and_deepcopy_reinstallation():
    context = Context(make_bank())
    routing.activate_additive_receiver(context, 0, slot=2)
    saved = routing.snapshot_route_policy(context)
    cloned = copy.deepcopy(context)
    routing.restore_route_policy(cloned, saved)
    assert routing.snapshot_route_policy(cloned) == saved
    x = torch.randn(3, 5)
    torch.testing.assert_close(context.banks()[0](x), cloned.banks()[0](x), rtol=0, atol=0)
    bad = copy.deepcopy(saved)
    bad[0]["slots"][0]["weight"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        routing.restore_route_policy(cloned, bad)


def test_sender_reset_is_a_separate_reproducible_state_change():
    bank = make_bank()
    context = Context(bank)
    routing.activate_additive_receiver(context, 0, slot=2)
    with torch.no_grad():
        bank.slots[0].b.fill_(0.2)
        bank.slots[2].b.fill_(0.1)
    x = torch.randn(5, 5)
    before = bank(x).detach().clone()
    record = routing.reset_sender(context, 0, torch.Generator().manual_seed(41))
    after = bank(x).detach()
    assert record == {"sender": 0, "reset_slots": [0, 2], "reset_policy": routing.RESET_POLICY}
    assert bank.route_override is None
    assert not bool(bank.slots[0].active) and not bool(bank.slots[2].active)
    assert torch.count_nonzero(bank.slots[0].b) == 0 and torch.count_nonzero(bank.slots[2].b) == 0
    assert not torch.equal(before, after)
    clone = make_bank()
    clone.activate(2)
    clone_ctx = Context(clone)
    routing.reset_sender(clone_ctx, 0, torch.Generator().manual_seed(41))
    torch.testing.assert_close(clone.slots[0].a, bank.slots[0].a, rtol=0, atol=0)


def test_legacy_mix_and_dream_are_rejected():
    bank = make_bank()
    bank.route_override = {"mix": 1, "fraction": 0.5}
    with pytest.raises(ValueError, match="natural or the v10 additive schema"):
        routing.install_additive_routes(Context(bank))
    bank.route_override = None
    bank.dream = True
    with pytest.raises(ValueError, match="dream"):
        routing.activate_additive_receiver(Context(bank), 0, slot=2)
