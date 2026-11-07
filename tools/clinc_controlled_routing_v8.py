"""Explicit controlled-mix routing ablation for the CLINC V8 lifecycle.

This is a routing intervention, not natural Sleep and not a claim about a
paper-prescribed policy. At overridden levels the output is
``(1-fraction)*natural_output + fraction*selected_expert_output``. The selected
expert may also be picked by natural routing, in which case that branch is
evaluated twice. Active-slot parameter counts therefore do not describe the
extra branch computation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch

# Standalone tools and existing checkpoint pickles use this canonical module
# name. Keep it identical when this file is imported as a package by pytest.
_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
import clinc_lifecycle_v8 as _lifecycle
if Path(_lifecycle.__file__).resolve() != _TOOLS / "clinc_lifecycle_v8.py":
    raise RuntimeError("controlled lifecycle source binding mismatch")
from clinc_lifecycle_v8 import (
    SleepTransaction, SleepSnapshot, WakeController,
    _bank_state, _base_names, _freeze, _params, _slot_names, _state_hash, _versions,
)


def _validate_policy(context, *, allow_bank0_override: bool):
    for index, bank in enumerate(context.banks()):
        if bool(getattr(bank, "dream", False)):
            raise ValueError(f"dream routing is unsupported in controlled mix (bank {index})")
        policy = getattr(bank, "route_override", None)
        if policy is None:
            continue
        if not allow_bank0_override and index == 0:
            raise ValueError("bank zero must retain natural routing")
        if not isinstance(policy, dict) or set(policy) != {"mix", "fraction"}:
            raise ValueError(f"bank {index} needs exactly a controlled mix override")
        slot = policy["mix"]
        fraction = policy["fraction"]
        active = [i for i, item in enumerate(bank.slots) if bool(item.active)]
        if type(slot) is not int or slot not in active:
            raise ValueError(f"bank {index} mix must name an active slot")
        if not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction)) or float(fraction) != .5:
            raise ValueError("controlled routing fraction is fixed at 0.5")


class ControlledWakeController(WakeController):
    """Wake controller accepting inherited per-bank 0.5 mix overrides."""

    def prepare_wake(self):
        _validate_policy(self.context, allow_bank0_override=False)
        bank = self.context.banks()[0]
        active = [i for i, slot in enumerate(bank.slots) if bool(slot.active)]
        if not active:
            bank.activate()
            active = [i for i, slot in enumerate(bank.slots) if bool(slot.active)]
        if len(active) != 1:
            raise ValueError("controlled wake requires one fast expert in bank zero")
        self._fast_names = _slot_names(self.context, 0, active[0])
        allowed = self._fast_names | set().union(*(_base_names(i) for i in self.clocks))
        _freeze(self.context, allowed)
        return allowed

    def restore(self, snapshot):
        super().restore(snapshot)
        _validate_policy(self.context, allow_bank0_override=False)
        self.prepare_wake()


@dataclass
class ControlledSleepSnapshot:
    payload: SleepSnapshot
    controlled_mix_schema: str
    sender: int
    receiver: int
    receiver_slot: int
    teacher_routes: list
    student_routes: list


class ControlledSleepTransaction(SleepTransaction):
    """Sleep transaction with a fixed 0.5 fresh-slot route at the receiver."""

    MIX_SCHEMA = "clinc-v8-controlled-mix-half-v1"

    def __init__(self, live_context, sender: int, prospective, *, lr=1e-3,
                 weight_decay=0., clip_norm=1., seed=17):
        banks = live_context.banks()
        if type(sender) is not int or not 0 <= sender < len(banks) - 1:
            raise ValueError("sender must be a nonterminal memory level")
        if set(prospective) != _base_names(sender):
            raise ValueError("prospective must be exactly sender base A/B absolute values")
        if any(not math.isfinite(x) or x < 0 for x in (lr, weight_decay)) or not math.isfinite(clip_norm) or clip_norm <= 0:
            raise ValueError("invalid sleep optimization configuration")
        # Existing controlled mix policies may target any currently active slot;
        # the sender policy is cleared in the student after its active slots reset.
        _validate_policy(live_context, allow_bank0_override=False)
        self.live = live_context
        self._live_hash = _state_hash(live_context)
        self.sender, self.receiver = sender, sender + 1
        self.lr, self.weight_decay, self.clip_norm = float(lr), float(weight_decay), float(clip_norm)
        device = next(iter(live_context.parameters())).device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.teacher, self.student = deepcopy(live_context), deepcopy(live_context)
        _freeze(self.teacher)
        for module in self.teacher.memory:
            module.eval()
        params = _params(self.student)
        with torch.no_grad():
            for name, value in prospective.items():
                if value.shape != params[name].shape or not bool(torch.isfinite(value).all()):
                    raise ValueError("invalid absolute prospective base value")
                params[name].copy_(value.to(params[name]))

        sender_bank = self.student.banks()[sender]
        reset_ids = [i for i, slot in enumerate(sender_bank.slots) if bool(slot.active)]
        sender_bank.reset_active()
        with torch.no_grad():
            for index in reset_ids:
                sender_bank.slots[index].a.normal_(mean=0., std=.02, generator=self.generator)
                sender_bank.slots[index].b.zero_()
        sender_bank.route_override = None

        receiver_bank = self.student.banks()[self.receiver]
        inactive = [i for i, slot in enumerate(receiver_bank.slots) if not bool(slot.active)]
        if not inactive:
            raise RuntimeError("controlled receiver bank has no fresh slot")
        self.slot = inactive[0]
        receiver_bank.activate(self.slot)
        receiver_bank.route_override = {"mix": self.slot, "fraction": .5}
        _validate_policy(self.teacher, allow_bank0_override=False)
        _validate_policy(self.student, allow_bank0_override=False)

        self.allowed_names = _slot_names(self.student, self.receiver, self.slot)
        _freeze(self.student, self.allowed_names)
        self.optimizer = self._new_optimizer()
        self.counters = {"adamw_steps": 0, "gkd_calls": 0, "lti_calls": 0,
                         "tokens": 0, "backward_attempts": 0, "zero_gradient_steps": 0,
                         "closed": False}
        self._set_guards()

    def snapshot(self):
        payload = super().snapshot()
        return ControlledSleepSnapshot(payload, self.MIX_SCHEMA, self.sender, self.receiver,
                                       self.slot,
                                       [deepcopy(b.route_override) for b in self.teacher.banks()],
                                       [deepcopy(b.route_override) for b in self.student.banks()])

    def restore(self, snapshot):
        if not isinstance(snapshot, ControlledSleepSnapshot) or snapshot.controlled_mix_schema != self.MIX_SCHEMA:
            raise ValueError("controlled snapshot schema mismatch")
        if (snapshot.sender, snapshot.receiver) != (snapshot.payload.metadata.get("sender"),
                                                     snapshot.payload.metadata.get("receiver")):
            raise ValueError("controlled snapshot layer identity mismatch")
        super().restore(snapshot.payload)
        if (self.sender, self.receiver, self.slot) != (snapshot.sender, snapshot.receiver,
                                                        snapshot.receiver_slot):
            raise ValueError("controlled receiver slot identity mismatch")
        _validate_policy(self.teacher, allow_bank0_override=False)
        _validate_policy(self.student, allow_bank0_override=False)
        if [b.route_override for b in self.teacher.banks()] != snapshot.teacher_routes:
            raise ValueError("teacher routing policy mismatch in controlled snapshot")
        if [b.route_override for b in self.student.banks()] != snapshot.student_routes:
            raise ValueError("student routing policy mismatch in controlled snapshot")
        if self.student.banks()[self.sender].route_override is not None:
            raise ValueError("controlled sender bank must be natural after reset")
        expected = {"mix": self.slot, "fraction": .5}
        if self.student.banks()[self.receiver].route_override != expected:
            raise ValueError("controlled receiver route mismatch")

    @classmethod
    def from_snapshot(cls, live_context, snapshot):
        if not isinstance(snapshot, ControlledSleepSnapshot) or snapshot.controlled_mix_schema != cls.MIX_SCHEMA:
            raise ValueError("not a controlled mix snapshot")
        tx = cls.__new__(cls)
        tx.live = live_context
        tx.teacher, tx.student = deepcopy(live_context), deepcopy(live_context)
        tx.restore(snapshot)
        return tx
