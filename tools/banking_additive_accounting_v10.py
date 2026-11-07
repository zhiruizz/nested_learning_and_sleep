"""Offline parameter and tail-matmul accounting for Banking v10 contexts.

This module intentionally does not import torch or load a model/checkpoint.
It accepts the ``MutableContext`` object shape or its CPU ``state_dict`` and
counts tensor payloads from ``numel``/shape metadata.  In additive routing,
unique executed parameters and executed branch evaluations are reported
separately: an expert can be evaluated twice in one token path while its
parameters occupy only one unique storage scope.

The analytic matmul estimate is only 2 FLOPs per evaluated matrix parameter
(one multiply and add).  It excludes routing, nonlinearities, norms, attention,
optimizer/backward work, and the frozen backbone; it is not whole-model FLOPs.
"""
from __future__ import annotations

from collections.abc import Mapping
import math


ADDITIVE_SCHEMA = "banking-cil-v10-additive-receivers-v1"


def _numel(value) -> int:
    if hasattr(value, "numel"):
        result = int(value.numel())
    elif hasattr(value, "shape"):
        result = math.prod(int(x) for x in value.shape)
    else:
        raise TypeError("accounting values must expose numel() or shape")
    if result < 0:
        raise ValueError("negative tensor size")
    return result


def _truth(value) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def _sizes_from_state(state: Mapping):
    if not isinstance(state, Mapping) or "memory" not in state:
        raise TypeError("expected MutableContext state_dict with a memory mapping")
    memory = state["memory"]
    if not isinstance(memory, Mapping):
        raise TypeError("state['memory'] must be a mapping")
    if "routing" not in state or "trainable" not in state:
        raise ValueError("state_dict must include routing and trainable metadata")
    routing, trainable = state["routing"], state["trainable"]
    if not isinstance(routing, (list, tuple)) or not isinstance(trainable, Mapping):
        raise TypeError("malformed routing/trainable metadata")

    # The Banking MutableContext wraps the memory ModuleList under `memory`.
    levels = {}
    slots = {}
    active = {}
    for key, value in memory.items():
        parts = key.split(".")
        if len(parts) == 2 and parts[1] in {"a", "b"} and parts[0].isdigit():
            levels.setdefault(int(parts[0]), {})[parts[1]] = _numel(value)
        elif len(parts) == 5 and parts[1:3] == ["experts", "slots"] and parts[0].isdigit():
            level, slot = int(parts[0]), int(parts[3])
            leaf = parts[4]
            if leaf in {"a", "b"}:
                slots.setdefault((level, slot), {})[leaf] = _numel(value)
            elif leaf == "active":
                active[(level, slot)] = _truth(value)
    if not levels:
        raise ValueError("no dense memory levels found in state['memory']")
    level_ids = sorted(levels)
    if level_ids != list(range(len(level_ids))):
        raise ValueError("memory level indices must be contiguous from zero")
    for level, matrices in levels.items():
        if set(matrices) != {"a", "b"}:
            raise ValueError(f"level {level} lacks an A or B matrix")
    bank_levels = sorted({level for level, _ in slots} | {level for level, _ in active})
    if bank_levels != list(range(len(routing))):
        raise ValueError("bank routing metadata does not match expert-bank levels")
    for key in slots:
        if set(slots[key]) != {"a", "b"}:
            raise ValueError(f"expert slot {key} lacks an A or B matrix")
    if set(slots) != set(active):
        raise ValueError("expert parameter slots and active buffers differ")
    for name in ("readout_a", "readout_b"):
        if name not in state:
            raise ValueError(f"state_dict lacks {name}")
    return levels, slots, active, routing, trainable, _numel(state["readout_a"]), _numel(state["readout_b"])


def _policy_counts(policy, active_ids, sizes, dream=False):
    """Return (unique expert params, executed branch params, additive_multi)."""
    active_ids = set(active_ids)
    if not active_ids:
        if policy is not None and policy != {}:
            raise ValueError("route policy refers to an empty bank")
        if dream:
            raise ValueError("dream routing has no active expert")
        return 0, 0, False

    if policy is None or policy == {}:
        if dream:
            # Dream adds a distinct route if available, else repeats an active one.
            natural = max(sizes[i] for i in active_ids)
            # Upper bound assumes the extra branch may be a second distinct slot.
            return min(len(active_ids), 2) * natural, natural + natural, False
        natural = max(sizes[i] for i in active_ids)
        return natural, natural, False

    if not isinstance(policy, Mapping):
        raise TypeError("route policy must be None or a mapping")

    if policy.get("schema") == ADDITIVE_SCHEMA:
        if dream:
            raise ValueError("additive routing cannot be combined with dream")
        if set(policy) != {"schema", "slots"} or not isinstance(policy["slots"], list) or not policy["slots"]:
            raise ValueError("malformed additive receiver policy")
        ids, seen = [], set()
        for row in policy["slots"]:
            if not isinstance(row, Mapping) or set(row) != {"slot", "weight"}:
                raise ValueError("malformed additive slot row")
            slot, weight = row["slot"], row["weight"]
            if type(slot) is not int or slot not in active_ids or slot in seen:
                raise ValueError("additive route slot must be a distinct active slot")
            if type(weight) not in (int, float) or not math.isfinite(float(weight)) or float(weight) <= 0:
                raise ValueError("additive route weight must be finite and positive")
            ids.append(slot)
            seen.add(slot)
        natural_ids = active_ids - seen
        natural = max((sizes[i] for i in natural_ids), default=0)
        explicit = sum(sizes[i] for i in ids)
        branches = explicit + natural
        return explicit + natural, branches, (len(ids) + bool(natural_ids)) > 1

    # Legacy controlled mix computes the normal route and the explicit slot,
    # even when that is the same slot. Count unique storage once, work twice.
    allowed = {"mix", "fraction"}
    if set(policy).issubset(allowed) and "mix" in policy:
        slot = policy["mix"]
        fraction = policy.get("fraction", 0.5)
        if type(slot) is not int or slot not in active_ids:
            raise ValueError("mix slot must be active")
        if type(fraction) not in (int, float) or not math.isfinite(float(fraction)) or not 0 < float(fraction) < 1:
            raise ValueError("mix fraction must lie in (0,1)")
        natural_size = max(sizes[i] for i in active_ids)
        unique_ids_upper = min(len(active_ids), 2)
        unique = unique_ids_upper * natural_size
        return unique, natural_size + sizes[slot], False

    # Include computes the natural route excluding that slot, plus the explicit
    # receiver. Dream may similarly repeat a branch when only one slot is active.
    if set(policy) == {"include"}:
        slot = policy["include"]
        if type(slot) is not int or slot not in active_ids:
            raise ValueError("included slot must be active")
        natural = max((sizes[i] for i in active_ids - {slot}), default=0)
        return natural + sizes[slot], natural + sizes[slot], False
    if set(policy) == {"force"}:
        slot = policy["force"]
        if type(slot) is not int or slot not in active_ids:
            raise ValueError("forced slot must be active")
        return sizes[slot], sizes[slot], False
    raise ValueError("unsupported serialized route policy")


def account_mutable_state(state_or_context, *, include_matmul_estimate: bool = True) -> dict:
    """Count a Banking v10 mutable context or its ``MutableContext.state_dict``.

    No model construction or tensor values beyond active flags are needed for
    state-dict mode. Counts are integer parameters, not bytes.
    """
    if isinstance(state_or_context, Mapping) and "memory" in state_or_context:
        levels, slots, active, routing, trainable_map, readout_a, readout_b = _sizes_from_state(state_or_context)
        dense_by_level = {i: sum(v.values()) for i, v in levels.items()}
        slot_sizes = {key: sum(v.values()) for key, v in slots.items()}
        active_by_level = {i: [slot for (level, slot), on in active.items() if level == i and on]
                           for i in range(len(routing))}
        stored = sum(dense_by_level.values()) + sum(slot_sizes.values()) + readout_a + readout_b
        trainable = 0
        for key, value in state_or_context["memory"].items():
            parts = key.split(".")
            if parts[-1] in {"a", "b"}:
                full = f"memory.{key}"
                if full not in trainable_map:
                    raise ValueError(f"trainable map lacks {full}")
                if bool(trainable_map[full]):
                    trainable += _numel(value)
        for name, size in (("readout_a", readout_a), ("readout_b", readout_b)):
            if name not in trainable_map:
                raise ValueError(f"trainable map lacks {name}")
            if bool(trainable_map[name]):
                trainable += size
        bank_sizes = [{slot: slot_sizes[(level, slot)] for slot in range(max((s for l, s in slots if l == level), default=-1) + 1)}
                      for level in range(len(routing))]
        dream = [False] * len(routing)  # state_dict contract disables dream routing.
    else:
        context = state_or_context
        memory = getattr(context, "memory", None)
        if memory is None or not hasattr(context, "banks"):
            raise TypeError("expected MutableContext-like object or its state_dict")
        banks = list(context.banks())
        if len(memory) != len(banks):
            raise ValueError("context memory and expert-bank counts differ")
        levels, bank_sizes, active_by_level, routing, dream = {}, [], {}, [], []
        stored = trainable = 0
        for name, parameter in context.named_parameters():
            n = _numel(parameter)
            stored += n
            if bool(parameter.requires_grad):
                trainable += n
        for i, level in enumerate(memory):
            levels[i] = {"a": _numel(level.a), "b": _numel(level.b)}
            bank = banks[i]
            row_sizes, active_ids = {}, []
            for slot_id, slot in enumerate(bank.slots):
                size = sum(_numel(p) for p in slot.parameters())
                row_sizes[slot_id] = size
                if _truth(slot.active):
                    active_ids.append(slot_id)
            bank_sizes.append(row_sizes)
            active_by_level[i] = active_ids
            routing.append(getattr(bank, "route_override", None))
            dream.append(bool(getattr(bank, "dream", False)))
        # `context.named_parameters()` includes every dense, readout, and expert
        # parameter, including inactive reserve slots.
        readout_a, readout_b = _numel(context.readout_a), _numel(context.readout_b)

    dense = sum(sum(v.values()) for v in levels.values()) + readout_a + readout_b
    resident_expert = sum(bank_sizes[i][slot] for i, ids in active_by_level.items() for slot in ids)
    unique_expert = branch_eval_expert = 0
    additive_multi = False
    details = []
    for i, policy in enumerate(routing):
        active_ids = active_by_level.get(i, [])
        sizes = bank_sizes[i]
        unique, evaluated, multi = _policy_counts(policy, active_ids, sizes, dream=dream[i])
        unique_expert += unique
        branch_eval_expert += evaluated
        additive_multi |= multi
        details.append({"level": i, "active_slot_ids": list(active_ids),
                        "active_expert_parameters": sum(sizes[j] for j in active_ids),
                        "route_policy": "natural" if policy is None else (policy.get("schema") if isinstance(policy, Mapping) else "invalid"),
                        "unique_expert_parameters_upper": unique,
                        "expert_branch_parameter_evaluations_upper": evaluated})

    dense_matrix_params = dense
    corrected = dense_matrix_params + unique_expert
    branch_evaluations = dense_matrix_params + branch_eval_expert
    result = {
        "stored_mutable_parameters": int(stored),
        "activated_resident_mutable_parameters": int(dense_matrix_params + resident_expert),
        "trainable_mutable_parameters": int(trainable),
        "corrected_per_token_unique_executed_upper": int(corrected),
        "per_token_expert_branch_parameter_evaluations_upper": int(branch_eval_expert),
        "per_token_executed_matrix_parameter_evaluations_upper": int(branch_evaluations),
        "legacy_field_invalid_for_additive": bool(additive_multi),
        "banks": details,
        "accounting_scope": "mutable adapter only; excludes frozen backbone",
    }
    if include_matmul_estimate:
        result["analytic_mutable_tail_forward_matmul_flops_upper"] = int(2 * branch_evaluations)
        result["analytic_estimate_scope"] = (
            "2 FLOPs per evaluated matrix parameter; excludes router, nonlinearities, norms, "
            "attention, optimizer/backward, and frozen-backbone work"
        )
    return result


def account_context(context, *, include_matmul_estimate: bool = True) -> dict:
    """Explicit alias for context-object callers."""
    return account_mutable_state(context, include_matmul_estimate=include_matmul_estimate)

