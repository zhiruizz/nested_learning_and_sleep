"""Bounded scalar clock/capacity transitions; no Adam calls or task metrics."""
import torch
from torch import nn

from tests.test_clinc_lifecycle_v8_cpu import Context, Slot
from tools.clinc_controlled_routing_v8 import ControlledSleepTransaction, ControlledWakeController
from tools.run_clinc_controlled_probe_v8 import _simulate_due_slot_peaks


def test_actual_nested_transactions_match_capacity_plan_across_warm_task_boundary():
    context = Context()
    for level, bank in enumerate(context.banks()):
        bank.slots = nn.ModuleList([Slot(level == 0 and i == 0) for i in range(8)])
    wake = ControlledWakeController(context, periods=(16, 64, 256, 1024, None),
                                    base_lr=.001, fast_lr=.1)

    def tick():
        wake.prepare_wake()
        grads = {name: torch.ones_like(p) for name, p in wake.context.named_parameters()
                 if p.requires_grad}
        wake.record_tick(grads, token_count=2)

    # The shared foundation uses ordinary prospective base updates, no Sleep.
    for _ in range(188):
        tick()
        for level in sorted(wake.proposals()):
            wake.apply(level)
    assert wake.clocks == {0: 12, 1: 60, 2: 188, 3: 188}
    counts = lambda: [sum(bool(s.active) for s in b.slots) for b in wake.context.banks()]
    assert counts() == [1, 0, 0, 0, 0]
    plan = _simulate_due_slot_peaks(counts(), wake.clocks, wake.periods, 188, 8)
    actual = []
    for task_tick in range(1, 189):
        tick()
        for sender in sorted(wake.proposals()):
            tx = ControlledSleepTransaction(wake.context, sender, wake.proposals()[sender], seed=task_tick)
            # This test is solely a structural state transition, with no claim
            # of a trained/distilled receiver. Numerical optimization is tested
            # separately, including the real-3B disk continuation qualification.
            assert tx.counters["adamw_steps"] == 0
            wake.bind_context(tx.commit())
            wake.consume(sender)
            actual.append({"task_tick": task_tick, "sender_level": sender,
                           "active_slots_after_transaction": counts()})
    assert actual == plan["events"]
    assert len(actual) == 16
    assert wake.clocks == {0: 8, 1: 56, 2: 120, 3: 376}
    assert max(max(row["active_slots_after_transaction"]) for row in actual) == 4
