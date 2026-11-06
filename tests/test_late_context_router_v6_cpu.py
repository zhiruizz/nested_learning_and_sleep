import unittest

import numpy as np
import torch
from torch import nn

from tools.late_context_router_v6 import FrozenRidgeGate, LateContextRouter, make_gate_mask
from nested_sleep.lm import AugmentedMLP, LlamaCMSLM, ReadoutLoRA, execution_state


class ToyMemory(nn.Module):
    def __init__(self, scale, shift, route_value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[scale]], dtype=torch.float32))
        self.register_buffer("shift", torch.tensor([shift], dtype=torch.float32))
        self.register_buffer("route_state", torch.tensor([route_value], dtype=torch.int64))

    def forward(self, x):
        return x + x * self.weight + self.shift


class CountingLinear(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.as_tensor(weight, dtype=torch.float32).clone(), requires_grad=False)
        self.bias = nn.Parameter(torch.as_tensor(bias, dtype=torch.float32).clone(), requires_grad=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.nn.functional.linear(x, self.weight, self.bias)


class TinyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = AugmentedMLP(nn.Linear(dim, dim), dim, rank=2, capacity=2, levels=2)


class TinyBody(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(dim)])


class TinyBackbone(nn.Module):
    def __init__(self, dim=4, vocab=11):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": dim})()
        self.embedding = nn.Embedding(vocab, dim)
        self.model = TinyBody(dim)
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = ReadoutLoRA(nn.Linear(dim, vocab))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embedding(input_ids)
        residual = hidden
        hidden = self.model.layers[-1].mlp(hidden)
        hidden = self.final_norm(hidden + residual)
        return type("Output", (), {"logits": self.lm_head(hidden)})()


class TinyLlamaCMSLM(LlamaCMSLM):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = TinyBackbone()
        self.rank, self.capacity, self.levels = 2, 2, 2
        self.memory_prefix = "backbone.model.layers.0.mlp.memory"

    def level_descriptors(self):
        return [
            {"base": [], "bank": f"{self.memory_prefix}.{i}.experts"}
            for i in range(self.levels)
        ]

    def adapter_state_dict(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
            if name.startswith(self.memory_prefix)
            or name in ("backbone.lm_head.a", "backbone.lm_head.b")
        }


def context(x, memory):
    y = x
    for module in memory:
        y = module(y)
    return y


def readout(base_head, hidden, a, b, fp32):
    base = base_head(hidden)
    delta = torch.nn.functional.linear(
        torch.nn.functional.linear(hidden, a), b
    )
    return base.float() + delta.float() if fp32 else base + delta


class LateContextRouterCpuTests(unittest.TestCase):
    def make_router_fixture(self):
        router = object.__new__(LateContextRouter)
        router.original_mlp = CountingLinear([[1.1, -0.2], [0.3, 0.8]], [0.2, -0.1])
        router.original_head = CountingLinear(
            [[0.4, 0.2], [-0.1, 0.7], [0.3, -0.5]], [0.01, 0.02, -0.03]
        )
        router.old_memory = nn.ModuleList([
            ToyMemory(0.2, 0.05, 1), ToyMemory(-0.1, 0.03, 2)
        ])
        router.new_memory = nn.ModuleList([
            ToyMemory(-0.25, -0.04, 3), ToyMemory(0.35, 0.07, 4)
        ])
        router.old_a = torch.tensor([[0.2, -0.1]], dtype=torch.float32)
        router.old_b = torch.tensor([[0.2], [-0.3], [0.1]], dtype=torch.float32)
        router.new_a = torch.tensor([[-0.3, 0.25]], dtype=torch.float32)
        router.new_b = torch.tensor([[0.1], [0.4], [-0.2]], dtype=torch.float32)
        router.old_fp32 = True
        router.new_fp32 = False
        router.operation_counts = {
            "original_final_mlp_calls": 0,
            "old_memory_branch_calls": 0,
            "new_memory_branch_calls": 0,
            "original_lm_head_calls": 0,
            "old_readout_adapter_calls": 0,
            "new_readout_adapter_calls": 0,
        }
        router._model_immutable_versions = ()
        router._immutable_model_versions_now = lambda: ()
        router._binding_ids = ()
        router._current_binding_ids = lambda: ()
        router._context_mode = "hard_route"
        router._active_mask = None
        router._active_selector = None
        router._active_prefix_lengths = None
        router._memory_versions = {
            "old_memory": router._module_tensor_signature(router.old_memory),
            "new_memory": router._module_tensor_signature(router.new_memory),
            "heads": router._head_tensor_signature(),
        }
        return router

    def test_late_context_selection_matches_two_full_contexts(self):
        router = self.make_router_fixture()
        x = torch.tensor([
            [[0.2, 0.7], [0.5, -0.4]],
            [[-0.1, 0.3], [0.9, 0.2]],
            [[0.8, -0.6], [0.1, 0.4]],
        ], dtype=torch.float32)
        gate = torch.tensor([False, True, False])

        shared_base = router.original_mlp(x)
        old_hidden = context(shared_base, router.old_memory)
        new_hidden = context(shared_base, router.new_memory)
        expected_hidden = torch.where(gate[:, None, None], new_hidden, old_hidden)
        expected_old = readout(router.original_head, old_hidden, router.old_a, router.old_b, True)
        expected_new = readout(router.original_head, new_hidden, router.new_a, router.new_b, False)
        expected_logits = torch.where(gate[:, None, None], expected_new, expected_old)

        # Reset fixture counters so only the routed execution is measured.
        router.original_mlp.calls = 0
        router.original_head.calls = 0
        actual_hidden = router.route_hidden(x, gate)
        actual_logits = router.route_readout(actual_hidden, gate)

        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        self.assertEqual(router.original_mlp.calls, 1)
        self.assertEqual(router.original_head.calls, 1)
        self.assertEqual(router.operation_counts["old_memory_branch_calls"], 1)
        self.assertEqual(router.operation_counts["new_memory_branch_calls"], 1)
        self.assertEqual(router.operation_counts["old_readout_adapter_calls"], 1)
        self.assertEqual(router.operation_counts["new_readout_adapter_calls"], 1)

    def test_exact_tensor_dedup_aliases_only_equal_new_context_tensors(self):
        router = object.__new__(LateContextRouter)
        router.old_memory = nn.ModuleList([ToyMemory(0.2, 0.05, 1)])
        router.new_memory = nn.ModuleList([ToyMemory(0.2, -0.07, 9)])
        router.old_a = nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32))
        router.old_b = nn.Parameter(torch.tensor([[0.1]], dtype=torch.float32))
        router.new_a = nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32))
        router.new_b = nn.Parameter(torch.tensor([[0.8]], dtype=torch.float32))
        x = torch.tensor([[0.3], [-0.2]])
        old_output_before = context(x, router.old_memory)
        new_output_before = context(x, router.new_memory)
        old_state_before = {k: v.clone() for k, v in router.old_memory.state_dict().items()}
        new_state_before = {k: v.clone() for k, v in router.new_memory.state_dict().items()}

        aliases = router._deduplicate_equal_tensors()

        self.assertIn("memory.0.weight", aliases)
        self.assertIn("head.a", aliases)
        self.assertNotIn("memory.0.shift", aliases)
        self.assertNotIn("memory.0.route_state", aliases)
        self.assertNotIn("head.b", aliases)
        self.assertIs(router.old_memory[0].weight, router.new_memory[0].weight)
        self.assertIs(router.old_a, router.new_a)
        self.assertIsNot(router.old_b, router.new_b)
        self.assertTrue(torch.equal(router.new_memory[0].shift, torch.tensor([-0.07])))
        torch.testing.assert_close(context(x, router.old_memory), old_output_before, rtol=0, atol=0)
        torch.testing.assert_close(context(x, router.new_memory), new_output_before, rtol=0, atol=0)
        for key, value in old_state_before.items():
            self.assertTrue(torch.equal(value, router.old_memory.state_dict()[key]), key)
        for key, value in new_state_before.items():
            self.assertTrue(torch.equal(value, router.new_memory.state_dict()[key]), key)

    def test_gate_modes_are_fixed_and_random_does_not_touch_global_rng(self):
        correct = torch.tensor([True, False, True, False])
        shuffled = torch.tensor([False, True, False, True])
        self.assertTrue(torch.equal(
            make_gate_mask(4, "fixed_correct_ridge", correct_mask=correct), correct
        ))
        self.assertTrue(torch.equal(
            make_gate_mask(4, "fixed_shuffled_ridge", shuffled_mask=shuffled), shuffled
        ))
        self.assertTrue(torch.equal(make_gate_mask(4, "constant_old"), torch.zeros(4, dtype=torch.bool)))
        self.assertTrue(torch.equal(make_gate_mask(4, "constant_new"), torch.ones(4, dtype=torch.bool)))

        torch.manual_seed(12345)
        expected_next = torch.rand(1)
        torch.manual_seed(12345)
        a = make_gate_mask(128, "deterministic_random", seed=19)
        observed_next = torch.rand(1)
        b = make_gate_mask(128, "deterministic_random", seed=19)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(expected_next, observed_next))

    def test_frozen_ridge_matches_numpy_and_ties_select_old(self):
        mean = np.array([0.2, -0.5, 0.1], dtype=np.float64)
        std = np.array([0.7, 1.2, 0.4], dtype=np.float64)
        weights = np.array([0.3, -0.8, 0.5], dtype=np.float64)
        bias = -0.02
        features = np.array([
            [0.8, -0.2, 0.4],
            [0.1, -0.6, 0.0],
            [-0.2, 0.3, 0.5],
        ], dtype=np.float64)
        gate = FrozenRidgeGate(mean, std, weights, bias)
        observed = gate(
            torch.tensor(features[:, None, :], dtype=torch.float32),
            torch.ones(3, dtype=torch.long),
        ).numpy()
        expected_scores = ((features - mean) / std / np.sqrt(features.shape[1])) @ weights + bias
        self.assertTrue(np.array_equal(observed, expected_scores > 0))
        tie_gate = FrozenRidgeGate(mean, std, weights, 0.0)
        tie = tie_gate(
            torch.tensor(mean[None, None, :], dtype=torch.float64),
            torch.ones(1, dtype=torch.long),
        )
        self.assertFalse(bool(tie.item()))
        with torch.no_grad():
            tie_gate.weights.add_(0.01)
        with self.assertRaisesRegex(RuntimeError, "frozen ridge parameters changed"):
            tie_gate(torch.tensor(mean[None, None, :], dtype=torch.float64),
                     torch.ones(1, dtype=torch.long))

    def test_context_version_guard_rejects_in_place_mutation(self):
        _, router = self.make_tiny_router(83)
        router._assert_unchanged()
        with torch.no_grad():
            router.old_memory[0].a.add_(0.01)
        with self.assertRaisesRegex(RuntimeError, "old_memory tensor identity/version changed"):
            router._assert_unchanged()

    def test_context_version_guard_rejects_parameter_replacement(self):
        _, router = self.make_tiny_router(89)
        old = router.new_memory[0].a
        router.new_memory[0].a = nn.Parameter(old.detach().clone(), requires_grad=False)
        with self.assertRaisesRegex(RuntimeError, "new_memory tensor identity/version changed"):
            router._assert_unchanged()

    def test_context_version_guard_rejects_shared_backbone_parameter_replacement(self):
        model, router = self.make_tiny_router(97)
        old = model.backbone.embedding.weight
        model.backbone.embedding.weight = nn.Parameter(old.detach().clone(), requires_grad=False)
        with self.assertRaisesRegex(RuntimeError, "shared frozen backbone"):
            router._assert_unchanged()

    def test_context_version_guard_rejects_original_readout_adapter_replacement(self):
        model, router = self.make_tiny_router(101)
        old = model.backbone.lm_head.a
        model.backbone.lm_head.a = nn.Parameter(old.detach().clone())
        with self.assertRaisesRegex(RuntimeError, "shared model module or adapter binding"):
            router._assert_unchanged()

    def test_gate_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            make_gate_mask(3, "fixed_correct_ridge", correct_mask=torch.ones(2))
        with self.assertRaises(ValueError):
            make_gate_mask(3, "fixed_shuffled_ridge")
        with self.assertRaises(ValueError):
            make_gate_mask(3, "not_a_gate")

    def test_full_tiny_llama_forward_is_exact_and_restores_methods(self):
        torch.manual_seed(31)
        model = TinyLlamaCMSLM().eval()
        model.backbone.model.layers[-1].mlp.original = CountingLinear(
            torch.eye(4), torch.zeros(4)
        )
        model.backbone.lm_head.original = CountingLinear(
            torch.randn(11, 4), torch.randn(11)
        )
        mutable = set(model.policy_names())
        mutable.update(
            name for name, _ in model.named_parameters()
            if name.startswith(model.memory_prefix + ".")
        )
        for name, parameter in model.named_parameters():
            if name not in mutable:
                parameter.requires_grad_(False)
        with torch.no_grad():
            for level in model.backbone.model.layers[-1].mlp.memory:
                level.a.normal_(0, 0.2)
                level.b.normal_(0, 0.15)
            model.backbone.lm_head.a.normal_(0, 0.2)
            model.backbone.lm_head.b.normal_(0, 0.15)
        new_adapter = model.adapter_state_dict()
        with torch.no_grad():
            new_adapter[f"{model.memory_prefix}.0.b"].add_(0.2)
            new_adapter[f"{model.memory_prefix}.1.a"].add_(0.1)
            new_adapter["backbone.lm_head.b"].mul_(-0.5)
        old_state = execution_state(model)
        new_state = {
            "routing": [None, None],
            "readout_fp32_accumulation": False,
        }
        router = LateContextRouter(model, new_adapter, new_state, old_execution_state=old_state)
        self.assertIs(router.old_memory, model.backbone.model.layers[-1].mlp.memory)
        self.assertEqual(router.old_a.data_ptr(), model.backbone.lm_head.a.data_ptr())
        self.assertEqual(router.old_b.data_ptr(), model.backbone.lm_head.b.data_ptr())
        self.assertEqual(router.model_plus_context_unique_payload_bytes,
                         router.unique_context_tensor_bytes)
        original_adapter = model.adapter_state_dict()
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 2, 1]])
        gate = make_gate_mask(3, "fixed_correct_ridge",
                              correct_mask=torch.tensor([False, True, False]))

        mlp = model.backbone.model.layers[-1].mlp
        inputs = model.backbone.embedding(tokens)
        shared_base = mlp.original(inputs)
        old_hidden = context(shared_base, router.old_memory)
        new_hidden = context(shared_base, router.new_memory)
        old_hidden = model.backbone.final_norm(old_hidden + inputs)
        new_hidden = model.backbone.final_norm(new_hidden + inputs)
        old_logits = router.route_readout(old_hidden, torch.zeros(3, dtype=torch.bool))
        new_logits = router.route_readout(new_hidden, torch.ones(3, dtype=torch.bool))
        expected = torch.where(gate[:, None, None], new_logits, old_logits)

        router.operation_counts = {name: 0 for name in router.operation_counts}
        mlp.original.calls = 0
        model.backbone.lm_head.original.calls = 0
        banks = [level.experts for level in router.old_memory]
        old_runtime = [
            (bank.last_routes, bank.last_route_audit, bank.audit_history, bank.route_override)
            for bank in banks
        ]
        logits = router.logits(tokens, gate)

        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        self.assertNotIn("forward", mlp.__dict__)
        self.assertNotIn("forward", model.backbone.lm_head.__dict__)
        self.assertEqual(router.operation_counts["original_final_mlp_calls"], 1)
        self.assertEqual(router.operation_counts["old_memory_branch_calls"], 1)
        self.assertEqual(router.operation_counts["new_memory_branch_calls"], 1)
        self.assertEqual(router.operation_counts["original_lm_head_calls"], 1)
        self.assertEqual(router.operation_counts["old_readout_adapter_calls"], 1)
        self.assertEqual(router.operation_counts["new_readout_adapter_calls"], 1)
        self.assertEqual(mlp.original.calls, 1)
        self.assertEqual(model.backbone.lm_head.original.calls, 1)
        for bank, before in zip(banks, old_runtime):
            self.assertIs(bank.last_routes, before[0])
            self.assertIs(bank.last_route_audit, before[1])
            self.assertIs(bank.audit_history, before[2])
            self.assertIs(bank.route_override, before[3])
        for key, value in original_adapter.items():
            self.assertTrue(torch.equal(value, model.adapter_state_dict()[key]), key)
        self.assertEqual(len(router.context_fingerprints["old"]), 64)
        self.assertEqual(len(router.context_fingerprints["new"]), 64)
        self.assertGreater(router.unique_context_tensor_bytes, 0)

    def make_tiny_router(self, seed):
        torch.manual_seed(seed)
        model = TinyLlamaCMSLM().eval()
        model.backbone.model.layers[-1].mlp.original = CountingLinear(
            torch.eye(4), torch.zeros(4)
        )
        model.backbone.lm_head.original = CountingLinear(
            torch.randn(11, 4), torch.randn(11)
        )
        mutable = set(model.policy_names())
        mutable.update(
            name for name, _ in model.named_parameters()
            if name.startswith(model.memory_prefix + ".")
        )
        for name, parameter in model.named_parameters():
            if name not in mutable:
                parameter.requires_grad_(False)
        with torch.no_grad():
            for level in model.backbone.model.layers[-1].mlp.memory:
                level.a.normal_(0, 0.2)
                level.b.normal_(0, 0.15)
            model.backbone.lm_head.a.normal_(0, 0.2)
            model.backbone.lm_head.b.normal_(0, 0.15)
        new_adapter = model.adapter_state_dict()
        with torch.no_grad():
            new_adapter[f"{model.memory_prefix}.0.b"].add_(0.2)
            new_adapter[f"{model.memory_prefix}.1.a"].add_(0.1)
            new_adapter["backbone.lm_head.b"].mul_(-0.5)
        new_state = {"routing": [None, None], "readout_fp32_accumulation": False}
        return model, LateContextRouter(
            model, new_adapter, new_state, old_execution_state=execution_state(model)
        )

    def test_frozen_ridge_callback_uses_shared_prefix_feature_once(self):
        model, router = self.make_tiny_router(41)
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        lengths = torch.tensor([1, 2, 3])
        mlp = model.backbone.model.layers[-1].mlp
        original_features = mlp.original(model.backbone.embedding(tokens))
        rows = torch.arange(tokens.shape[0])
        expected_features = original_features[rows, lengths - 1].double()
        selector = FrozenRidgeGate(
            mean=torch.zeros(4), std=torch.ones(4), weights=torch.tensor([1.0, -0.5, 0.25, 0.75]),
            bias=-0.1,
        )
        expected_mask = selector(original_features, lengths)
        self.assertTrue(torch.equal(selector.last_features, expected_features))
        selector.calls = 0

        mlp.original.calls = 0
        model.backbone.lm_head.original.calls = 0
        router.operation_counts = {name: 0 for name in router.operation_counts}
        logits = router.logits(tokens, selector=selector, prefix_lengths=lengths)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(router.operation_counts["selector_calls"], 1)
        self.assertEqual(router.operation_counts["original_final_mlp_calls"], 1)
        self.assertEqual(router.operation_counts["original_lm_head_calls"], 1)
        self.assertEqual(mlp.original.calls, 1)
        self.assertEqual(model.backbone.lm_head.original.calls, 1)
        self.assertEqual(router._active_mask, None)

        base = mlp.original(model.backbone.embedding(tokens))
        inputs = model.backbone.embedding(tokens)
        old_hidden, new_hidden = context(base, router.old_memory), context(base, router.new_memory)
        old_hidden = model.backbone.final_norm(old_hidden + inputs)
        new_hidden = model.backbone.final_norm(new_hidden + inputs)
        old_logits = router.route_readout(old_hidden, torch.zeros(3, dtype=torch.bool))
        new_logits = router.route_readout(new_hidden, torch.ones(3, dtype=torch.bool))
        expected = torch.where(expected_mask[:, None, None], new_logits, old_logits)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_uniform_context_mix_is_exact_half_average_with_one_head_call(self):
        model, router = self.make_tiny_router(57)
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mlp = model.backbone.model.layers[-1].mlp
        base = mlp.original(model.backbone.embedding(tokens))
        inputs = model.backbone.embedding(tokens)
        old_hidden, new_hidden = context(base, router.old_memory), context(base, router.new_memory)
        expected_hidden = model.backbone.final_norm((old_hidden + new_hidden) * 0.5 + inputs)
        old_logits = router.route_readout(expected_hidden, torch.zeros(2, dtype=torch.bool))
        new_logits = router.route_readout(expected_hidden, torch.ones(2, dtype=torch.bool))
        expected_logits = (old_logits.float() + new_logits.float()) * 0.5
        mlp.original.calls = 0
        model.backbone.lm_head.original.calls = 0
        router.operation_counts = {name: 0 for name in router.operation_counts}

        logits = router.logits(tokens, mode="uniform_context_mix")

        torch.testing.assert_close(logits, expected_logits, rtol=1e-6, atol=1e-6)
        self.assertEqual(router.operation_counts["original_final_mlp_calls"], 1)
        self.assertEqual(router.operation_counts["old_memory_branch_calls"], 1)
        self.assertEqual(router.operation_counts["new_memory_branch_calls"], 1)
        self.assertEqual(router.operation_counts["original_lm_head_calls"], 1)
        self.assertEqual(router.operation_counts["uniform_mix_calls"], 2)
        self.assertEqual(mlp.original.calls, 1)
        self.assertEqual(model.backbone.lm_head.original.calls, 1)

    def test_deduplication_opt_out_preserves_context_outputs(self):
        model, deduped = self.make_tiny_router(73)
        adapter = {
            model.memory_prefix + "." + key: value.detach().clone()
            for key, value in deduped.new_memory.state_dict().items()
        }
        adapter["backbone.lm_head.a"] = deduped.new_a.detach().clone()
        adapter["backbone.lm_head.b"] = deduped.new_b.detach().clone()
        state = {"routing": [None, None], "readout_fp32_accumulation": False}
        separate = LateContextRouter(
            model, adapter, state, old_execution_state=execution_state(model),
            deduplicate_equal=False,
        )
        self.assertEqual(separate.alias_map, ())
        self.assertTrue(deduped.alias_map)
        self.assertGreater(separate.unique_context_tensor_bytes, deduped.unique_context_tensor_bytes)
        tokens = torch.tensor([[1, 4, 2], [3, 5, 7]])
        gate = torch.tensor([False, True])
        expected = deduped.logits(tokens, gate)
        actual = separate.logits(tokens, gate)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
