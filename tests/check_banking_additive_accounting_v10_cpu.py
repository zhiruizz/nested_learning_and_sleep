"""Small metadata-only regressions for the Banking v10 cost accountant."""
from __future__ import annotations

import copy
import unittest

from tools.banking_additive_accounting_v10 import (
    ADDITIVE_SCHEMA,
    account_context,
    account_mutable_state,
)


class TensorStub:
    def __init__(self, *shape, value=None, trainable=False):
        self.shape = shape
        self.value = value
        self.requires_grad = trainable

    def numel(self):
        n = 1
        for x in self.shape:
            n *= x
        return n

    def item(self):
        return self.value


class ParamStub(TensorStub):
    pass


class SlotStub:
    def __init__(self, active, trainable=False):
        self.active = TensorStub(value=active)
        self.a = ParamStub(2, 3, trainable=trainable)
        self.b = ParamStub(3, 2, trainable=trainable)

    def parameters(self):
        return iter((self.a, self.b))


class BankStub:
    def __init__(self, active=(True, False, False), policy=None):
        self.slots = [SlotStub(on, trainable=(policy is not None and i in _policy_ids(policy)))
                      for i, on in enumerate(active)]
        self.route_override = copy.deepcopy(policy)
        self.dream = False


def _policy_ids(policy):
    if isinstance(policy, dict) and policy.get("schema") == ADDITIVE_SCHEMA:
        return [row["slot"] for row in policy["slots"]]
    if isinstance(policy, dict) and "mix" in policy:
        return [policy["mix"]]
    return []


class LevelStub:
    def __init__(self):
        self.a = ParamStub(2, 3)
        self.b = ParamStub(4, 2)
        self.experts = None


class ContextStub:
    def __init__(self, active=(True, False, False), policy=None):
        level = LevelStub()
        bank = BankStub(active, policy)
        level.experts = bank
        self.memory = [level]
        self.readout_a = ParamStub(2, 3)
        self.readout_b = ParamStub(3, 2)

    def banks(self):
        return [level.experts for level in self.memory]

    def named_parameters(self):
        yield "memory.0.a", self.memory[0].a
        yield "memory.0.b", self.memory[0].b
        for i, slot in enumerate(self.memory[0].experts.slots):
            yield f"memory.0.experts.slots.{i}.a", slot.a
            yield f"memory.0.experts.slots.{i}.b", slot.b
        yield "readout_a", self.readout_a
        yield "readout_b", self.readout_b


def state_for(context):
    bank = context.memory[0].experts
    memory = {
        "0.a": context.memory[0].a,
        "0.b": context.memory[0].b,
    }
    trainable = {
        "memory.0.a": False,
        "memory.0.b": False,
        "readout_a": False,
        "readout_b": False,
    }
    for i, slot in enumerate(bank.slots):
        prefix = f"0.experts.slots.{i}"
        memory[prefix + ".a"] = slot.a
        memory[prefix + ".b"] = slot.b
        memory[prefix + ".active"] = slot.active
        trainable[f"memory.{prefix}.a"] = slot.a.requires_grad
        trainable[f"memory.{prefix}.b"] = slot.b.requires_grad
    return {
        "memory": memory,
        "readout_a": context.readout_a,
        "readout_b": context.readout_b,
        "readout_fp32_accumulation": False,
        "routing": [copy.deepcopy(bank.route_override)],
        "trainable": trainable,
    }


class BankingAdditiveAccountingTests(unittest.TestCase):
    def test_natural_matches_legacy_top_one_count_and_state_restore(self):
        context = ContextStub(active=(True, True, False))
        got = account_context(context)
        # Dense A/B=14 plus readout=12, and one active expert A/B=12.
        self.assertEqual(got["stored_mutable_parameters"], 14 + 12 + 36)
        self.assertEqual(got["activated_resident_mutable_parameters"], 14 + 12 + 24)
        self.assertEqual(got["corrected_per_token_unique_executed_upper"], 14 + 12 + 12)
        self.assertEqual(got["per_token_expert_branch_parameter_evaluations_upper"], 12)
        self.assertFalse(got["legacy_field_invalid_for_additive"])
        restored = account_mutable_state(state_for(context))
        self.assertEqual(restored, got)

    def test_two_additive_receivers_plus_natural_branch(self):
        policy = {"schema": ADDITIVE_SCHEMA,
                  "slots": [{"slot": 1, "weight": 1.0}, {"slot": 2, "weight": 0.5}]}
        context = ContextStub(active=(True, True, True), policy=policy)
        got = account_context(context)
        # Dense matrices/readout have 26 params; two explicit receivers plus one natural branch.
        self.assertEqual(got["corrected_per_token_unique_executed_upper"], 26 + 36)
        self.assertEqual(got["per_token_expert_branch_parameter_evaluations_upper"], 36)
        self.assertEqual(got["per_token_executed_matrix_parameter_evaluations_upper"], 26 + 36)
        self.assertEqual(got["analytic_mutable_tail_forward_matmul_flops_upper"], 2 * (26 + 36))
        self.assertTrue(got["legacy_field_invalid_for_additive"])
        self.assertEqual(account_mutable_state(state_for(context)), got)

    def test_mix_with_single_active_slot_counts_duplicate_work_not_storage(self):
        context = ContextStub(active=(True, False, False), policy={"mix": 0, "fraction": 0.5})
        got = account_context(context)
        self.assertEqual(got["banks"][0]["unique_expert_parameters_upper"], 12)
        self.assertEqual(got["banks"][0]["expert_branch_parameter_evaluations_upper"], 24)
        self.assertEqual(got["corrected_per_token_unique_executed_upper"], 38)
        self.assertEqual(got["per_token_executed_matrix_parameter_evaluations_upper"], 50)

    def test_rejects_duplicate_or_inactive_additive_routes(self):
        duplicate = {"schema": ADDITIVE_SCHEMA,
                     "slots": [{"slot": 1, "weight": 1.0}, {"slot": 1, "weight": 0.5}]}
        with self.assertRaisesRegex(ValueError, "distinct active"):
            account_context(ContextStub(active=(True, True, False), policy=duplicate))
        inactive = {"schema": ADDITIVE_SCHEMA, "slots": [{"slot": 2, "weight": 1.0}]}
        with self.assertRaisesRegex(ValueError, "distinct active"):
            account_context(ContextStub(active=(True, True, False), policy=inactive))


if __name__ == "__main__":
    unittest.main()
