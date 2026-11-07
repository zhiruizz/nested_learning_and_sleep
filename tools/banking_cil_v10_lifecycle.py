"""Wake/Sleep controllers for the local additive-receiver v10 intervention.

This module layers the v10 context-local route policy over the existing v8
absolute-base proposal and restore machinery. It is an explicitly local
ablation: receiver expansion is function-preserving at B=0, while prospective
base application and sender reset remain separate potentially disruptive
operations.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch
from torch import Tensor

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
import clinc_lifecycle_v8 as _v8
if Path(_v8.__file__).resolve() != (_TOOLS / "clinc_lifecycle_v8.py").resolve():
    raise RuntimeError("v10 lifecycle must bind to the local v8 lifecycle source")
import banking_cil_v10_routing as _routes
if Path(_routes.__file__).resolve() != (_TOOLS / "banking_cil_v10_routing.py").resolve():
    raise RuntimeError("v10 lifecycle must bind to the local additive routing source")

from clinc_lifecycle_v8 import (  # noqa: E402
    WakeController as _WakeController,
    SleepSnapshot as _SleepSnapshot,
    _bank_state,
    _base_names,
    _freeze,
    _params,
    _slot_names,
    _state_hash,
    _versions,
)


class AdditiveWakeController(_WakeController):
    """v8 clocks/proposals with validated additive routing during wake."""

    def bind_context(self, context):
        super().bind_context(context)
        _routes.install_additive_routes(context)
        _routes.validate_additive_routes(context)

    def prepare_wake(self):
        _routes.install_additive_routes(self.context)
        _routes.validate_additive_routes(self.context)
        if any(bool(getattr(bank, "dream", False)) for bank in self.context.banks()):
            raise ValueError("additive wake requires dream disabled")
        bank = self.context.banks()[0]
        active = [i for i, slot in enumerate(bank.slots) if bool(slot.active)]
        if not active:
            bank.activate()
            active = [i for i, slot in enumerate(bank.slots) if bool(slot.active)]
        if len(active) != 1:
            raise ValueError("wake requires exactly one fast expert in level zero")
        self._fast_names = _slot_names(self.context, 0, active[0])
        allowed = self._fast_names | set().union(*(_base_names(i) for i in self.clocks))
        _freeze(self.context, allowed)
        return allowed

    def restore(self, snapshot):
        super().restore(snapshot)
        _routes.install_additive_routes(self.context)
        _routes.validate_additive_routes(self.context)
        # Do not create a fast slot or rewrite trainable flags at restore time.
        # A safe checkpoint may be at a boundary where bank zero is empty;
        # activation belongs to the next explicit prepare_wake/record_tick.


@dataclass
class AdditiveSleepSnapshot:
    """Disk-safe wrapper binding route policies to the v8 owned snapshot."""

    payload: _SleepSnapshot
    schema: str
    sender: int
    receiver: int
    receiver_slot: int
    reset_slots: list[int]
    teacher_routes: list
    student_routes: list


class AdditiveSleepTransaction(_v8.SleepTransaction):
    """Pre-update teacher and prospective student with a zero-B additive slot.

    The old and new slot outputs are summed; the old gate denominator excludes
    all additive slots. The student optimizes only the fresh receiver A/B pair.
    ``gold`` is a supervised-control counter using the same transaction shape.
    """

    SNAPSHOT_SCHEMA = "banking-cil-v10-additive-sleep-snapshot-v1"

    def __init__(self, live_context, sender: int, prospective, *, lr=1e-3,
                 weight_decay=0.0, clip_norm=1.0, seed=17):
        banks = live_context.banks()
        if type(sender) is not int or not 0 <= sender < len(banks) - 1:
            raise ValueError("sender must be a nonterminal memory level")
        if set(prospective) != _base_names(sender):
            raise ValueError("prospective must be exactly sender base A/B absolute values")
        if (not math.isfinite(float(lr)) or lr < 0 or
                not math.isfinite(float(weight_decay)) or weight_decay < 0 or
                not math.isfinite(float(clip_norm)) or clip_norm <= 0):
            raise ValueError("invalid additive sleep optimization configuration")
        _routes.install_additive_routes(live_context)
        _routes.validate_additive_routes(live_context)
        if any(bool(getattr(bank, "dream", False)) for bank in banks):
            raise ValueError("additive sleep requires dream disabled")

        self.live = live_context
        self._live_hash = _state_hash(live_context)
        self.sender, self.receiver = sender, sender + 1
        self.lr, self.weight_decay, self.clip_norm = float(lr), float(weight_decay), float(clip_norm)
        device = next(iter(self.live.parameters())).device
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.teacher, self.student = deepcopy(live_context), deepcopy(live_context)
        _routes.install_additive_routes(self.teacher)
        _routes.install_additive_routes(self.student)
        _freeze(self.teacher)
        for module in self.teacher.memory:
            module.eval()

        params = _params(self.student)
        with torch.no_grad():
            for name, value in prospective.items():
                if (not isinstance(value, Tensor) or value.shape != params[name].shape
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError("invalid absolute prospective base value")
                params[name].copy_(value.to(params[name]))

        reset = _routes.reset_sender(self.student, self.sender, self.generator)
        self.reset_slots = list(reset["reset_slots"])
        receiver_bank = self.student.banks()[self.receiver]
        inactive = [i for i, slot in enumerate(receiver_bank.slots) if not bool(slot.active)]
        if not inactive:
            raise RuntimeError("additive receiver capacity exhausted; no overwrite permitted")
        self.slot = _routes.activate_additive_receiver(
            self.student, self.receiver, slot=inactive[0], weight=1.0)
        self.allowed_names = _slot_names(self.student, self.receiver, self.slot)
        _freeze(self.student, self.allowed_names)
        self.optimizer = self._new_optimizer()
        self.counters = {"adamw_steps": 0, "gkd_calls": 0, "lti_calls": 0,
                         "gold_calls": 0, "tokens": 0, "backward_attempts": 0,
                         "zero_gradient_steps": 0, "closed": False}
        self._set_guards()

    def _new_optimizer(self):
        params = _params(self.student)
        return torch.optim.AdamW([params[name] for name in sorted(self.allowed_names)],
                                 lr=self.lr, weight_decay=self.weight_decay)

    def optimize(self, loss, kind, tokens):
        if self.counters["closed"]:
            raise RuntimeError("additive sleep transaction is closed")
        if kind not in ("gkd", "lti", "gold") or type(tokens) is not int or tokens < 1:
            raise ValueError("kind must be gkd/lti/gold with a positive token count")
        if (not isinstance(loss, Tensor) or loss.ndim != 0 or
                not bool(torch.isfinite(loss)) or not loss.requires_grad):
            raise ValueError("additive sleep requires a finite differentiable scalar loss")
        self._check()
        self.optimizer.zero_grad(set_to_none=True)
        self.counters["backward_attempts"] += 1
        self.counters["tokens"] += tokens
        loss.backward()
        self._check()
        params = [_params(self.student)[name] for name in sorted(self.allowed_names)]
        grads = [parameter.grad for parameter in params if parameter.grad is not None]
        if not grads or any(not bool(torch.isfinite(grad).all()) for grad in grads):
            raise RuntimeError("missing or nonfinite additive receiver gradients")
        norm = torch.nn.utils.clip_grad_norm_(params, self.clip_norm, error_if_nonfinite=True)
        if not math.isfinite(float(norm)) or float(norm) <= 0.0:
            self.counters["zero_gradient_steps"] += 1
            raise RuntimeError("additive receiver gradient norm must be positive before AdamW")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.counters["adamw_steps"] += 1
        self.counters[f"{kind}_calls"] += 1
        self._check()

    def snapshot(self) -> AdditiveSleepSnapshot:
        payload = _v8.SleepTransaction.snapshot(self)
        payload.metadata["v10_reset_slots"] = list(self.reset_slots)
        teacher_routes = _routes.snapshot_route_policy(self.teacher)
        student_routes = _routes.snapshot_route_policy(self.student)
        return AdditiveSleepSnapshot(payload, self.SNAPSHOT_SCHEMA, self.sender,
                                     self.receiver, self.slot, list(self.reset_slots),
                                     teacher_routes, student_routes)

    def restore(self, snapshot: AdditiveSleepSnapshot):
        if not isinstance(snapshot, AdditiveSleepSnapshot) or snapshot.schema != self.SNAPSHOT_SCHEMA:
            raise ValueError("additive sleep snapshot schema mismatch")
        if (snapshot.sender, snapshot.receiver) != (snapshot.payload.metadata.get("sender"),
                                                     snapshot.payload.metadata.get("receiver")):
            raise ValueError("additive sleep snapshot sender/receiver mismatch")
        if snapshot.payload.metadata.get("v10_reset_slots") != snapshot.reset_slots:
            raise ValueError("additive sleep snapshot reset-slot receipt mismatch")
        _v8.SleepTransaction.restore(self, snapshot.payload)
        self.reset_slots = list(snapshot.reset_slots)
        if (self.sender, self.receiver, self.slot) != (snapshot.sender, snapshot.receiver,
                                                        snapshot.receiver_slot):
            raise ValueError("additive sleep snapshot receiver slot mismatch")
        for context in (self.live, self.teacher, self.student):
            _routes.install_additive_routes(context)
            _routes.validate_additive_routes(context)
        if _routes.snapshot_route_policy(self.teacher) != snapshot.teacher_routes:
            raise ValueError("teacher additive route policy changed in snapshot")
        if _routes.snapshot_route_policy(self.student) != snapshot.student_routes:
            raise ValueError("student additive route policy changed in snapshot")
        if self.student.banks()[self.sender].route_override is not None:
            raise ValueError("sender bank must have natural routing after reset")
        receiver_routes = snapshot.student_routes[self.receiver]
        if not isinstance(receiver_routes, dict) or receiver_routes.get("schema") != _routes.SCHEMA:
            raise ValueError("student receiver additive route policy missing")
        if not any(row == {"slot": self.slot, "weight": 1.0} for row in receiver_routes.get("slots", [])):
            raise ValueError("student receiver slot/weight mismatch")
        if self.reset_slots != snapshot.reset_slots:
            raise ValueError("sender reset slot list mismatch")
        return self

    @classmethod
    def from_snapshot(cls, live_context, snapshot: AdditiveSleepSnapshot):
        if not isinstance(snapshot, AdditiveSleepSnapshot) or snapshot.schema != cls.SNAPSHOT_SCHEMA:
            raise ValueError("not a v10 additive sleep snapshot")
        tx = cls.__new__(cls)
        tx.live = live_context
        tx.teacher, tx.student = deepcopy(live_context), deepcopy(live_context)
        _routes.install_additive_routes(tx.teacher)
        _routes.install_additive_routes(tx.student)
        tx.restore(snapshot)
        return tx

    def commit(self):
        result = _v8.SleepTransaction.commit(self)
        _routes.install_additive_routes(result)
        _routes.validate_additive_routes(result)
        return result
