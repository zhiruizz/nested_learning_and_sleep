"""Optional function-preserving additive receiver routing for CLINC v10.

This is a local routing intervention, not native HOPE/Sleep. It keeps the
existing SparseExperts router/gates over the old active set, then adds each
declared receiver output with a positive coefficient. A freshly activated
receiver has zero B, so adding the route initially preserves the prior
function while still allowing a nonzero gradient into B.
"""
from __future__ import annotations

from copy import deepcopy
import math
from types import MethodType

SCHEMA = "banking-cil-v10-additive-receivers-v1"
RESET_POLICY = "redraw_A_normal_0_0.02_zero_B"


def _banks(context):
    banks = context.banks() if callable(getattr(context, "banks", None)) else None
    if not isinstance(banks, (list, tuple)) or not banks:
        raise TypeError("context must expose a nonempty banks() sequence")
    return list(banks)


def _validate_bank_policy(bank) -> list[dict]:
    policy = getattr(bank, "route_override", None)
    if policy is None:
        return []
    if not isinstance(policy, dict) or set(policy) != {"schema", "slots"} or policy.get("schema") != SCHEMA:
        raise ValueError("bank route_override must be natural or the v10 additive schema")
    rows = policy.get("slots")
    if not isinstance(rows, list) or not rows:
        raise ValueError("additive route policy requires at least one slot")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"slot", "weight"}:
            raise ValueError("malformed additive slot entry")
        slot = row["slot"]
        weight = row["weight"]
        if type(slot) is not int or not 0 <= slot < len(bank.slots) or slot in seen:
            raise ValueError("invalid or duplicate additive slot index")
        if type(weight) not in (int, float) or not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError("additive slot weight must be finite and positive")
        if not bool(bank.slots[slot].active):
            raise ValueError("every additive slot must be active")
        seen.add(slot)
    return [{"slot": int(row["slot"]), "weight": float(row["weight"])} for row in rows]


def validate_additive_routes(context) -> list[dict | None]:
    """Validate serializable policy, active-slot membership, and no legacy mix."""
    result = []
    for bank in _banks(context):
        if bool(getattr(bank, "dream", False)):
            raise ValueError("v10 additive route policy cannot be combined with dream")
        rows = _validate_bank_policy(bank)
        result.append(None if not rows else {"schema": SCHEMA, "slots": rows})
    return result


def install_additive_routes(context) -> None:
    """Install a context-local wrapper; safe to call again after deepcopy/load."""
    for bank in _banks(context):
        _validate_bank_policy(bank)
        if bool(getattr(bank, "dream", False)):
            raise ValueError("v10 additive routes cannot be installed with dream enabled")
        bound = getattr(bank, "forward")
        if (getattr(bank, "_v10_additive_forward_installed", False)
                and getattr(bound, "__self__", None) is bank):
            continue
        original_forward = type(bank).forward
        if not callable(original_forward):
            raise TypeError("bank class must provide a callable forward")

        def additive_forward(self, x, _original=original_forward):
            rows = _validate_bank_policy(self)
            if not rows:
                return _original(self, x)
            if self.dream:
                raise ValueError("v10 additive routes cannot execute with dream enabled")
            original_policy = self.route_override
            additive_ids = [row["slot"] for row in rows]
            try:
                # SparseExperts computes the exact old routing logits/gates with
                # additive slots excluded from its softmax denominator.
                self.route_override = {"disable": additive_ids}
                result = _original(self, x)
            finally:
                self.route_override = original_policy
            flat_count = x.reshape(-1, x.shape[-1]).shape[0]
            for row in rows:
                slot = row["slot"]
                result = result + self.slots[slot](x) * row["weight"]
                self.last_routes.append({"slot": slot, "tokens": flat_count,
                                         "kind": "v10_additive_receiver",
                                         "weight": row["weight"]})
            if self.audit_routes and isinstance(self.last_route_audit, dict):
                old_active = list(self.last_route_audit.get("active", []))
                self.last_route_audit["v10_additive_slots"] = additive_ids
                self.last_route_audit["effective_active_slots"] = len(old_active) + len(additive_ids)
                self.last_route_audit["additive_weights"] = [r["weight"] for r in rows]
            return result

        bank.forward = MethodType(additive_forward, bank)
        bank._v10_additive_forward_installed = True


def activate_additive_receiver(context, level: int, slot: int | None = None,
                               weight: float = 1.0) -> int:
    """Activate a zero-output receiver and append it to that level's route.

    Previously active slots are frozen. The receiver A basis is retained from
    its stored slot state, but B is zeroed; thus route expansion alone does not
    change the output and B can receive a gradient on the first update.
    """
    banks = _banks(context)
    if type(level) is not int or not 0 <= level < len(banks):
        raise ValueError("level index out of range")
    if type(weight) not in (int, float) or not math.isfinite(float(weight)) or float(weight) <= 0:
        raise ValueError("weight must be finite and positive")
    bank = banks[level]
    if bool(bank.dream):
        raise ValueError("cannot add an additive receiver while dream is enabled")
    rows = _validate_bank_policy(bank)
    if slot is not None and type(slot) is not int:
        raise ValueError("slot index must be an integer or None")
    # Validate legacy policies before touching activity or parameter flags.
    policy = getattr(bank, "route_override", None)
    if policy is not None and (not isinstance(policy, dict) or policy.get("schema") != SCHEMA):
        raise ValueError("cannot combine additive receivers with a legacy route override")
    before_active = [i for i, expert in enumerate(bank.slots) if bool(expert.active)]
    new_slot = bank.activate(slot)
    for idx in before_active:
        expert = bank.slots[idx]
        expert.requires_grad_(False)
        for parameter in expert.parameters():
            parameter.grad = None
    receiver = bank.slots[new_slot]
    with __import__("torch").no_grad():
        receiver.b.zero_()
    receiver.requires_grad_(True)
    rows.append({"slot": new_slot, "weight": float(weight)})
    bank.route_override = {"schema": SCHEMA, "slots": rows}
    install_additive_routes(context)
    validate_additive_routes(context)
    return new_slot


def reset_sender(context, sender: int, generator) -> dict:
    """Apply the v8 sender reset to one bank as a separate state operation.

    This deliberately does not perform a wake/prospective update. It clears all
    active slots and the additive policy, then redraws each formerly active A
    and zeros B. This isolates reset effects from receiver-route expansion.
    """
    import torch
    banks = _banks(context)
    if type(sender) is not int or not 0 <= sender < len(banks):
        raise ValueError("sender index out of range")
    if not isinstance(generator, torch.Generator):
        raise TypeError("reset requires an explicit torch.Generator")
    bank = banks[sender]
    if bool(bank.dream):
        raise ValueError("cannot reset a bank while dream is enabled")
    _validate_bank_policy(bank)
    reset_ids = [i for i, expert in enumerate(bank.slots) if bool(expert.active)]
    bank.reset_active()
    bank.route_override = None
    with torch.no_grad():
        for i in reset_ids:
            expert = bank.slots[i]
            expert.a.normal_(mean=0.0, std=0.02, generator=generator)
            expert.b.zero_()
    return {"sender": sender, "reset_slots": reset_ids, "reset_policy": RESET_POLICY}


def snapshot_route_policy(context) -> list[dict | None]:
    """Return a deep-copied, JSON-safe routing payload for context checkpoints."""
    return deepcopy(validate_additive_routes(context))


def restore_route_policy(context, policies) -> None:
    """Restore route_override payload before installing wrappers on a clone."""
    banks = _banks(context)
    if not isinstance(policies, list) or len(policies) != len(banks):
        raise ValueError("routing policy list length mismatch")
    for bank, policy in zip(banks, policies):
        bank.route_override = deepcopy(policy)
    install_additive_routes(context)
    validate_additive_routes(context)
