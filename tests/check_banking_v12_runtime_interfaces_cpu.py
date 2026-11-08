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

import banking_cil_v10_runtime as v10_runtime  # noqa: E402
import banking_cil_v12_batching as batching  # noqa: E402
import clinc_tail_runtime_v8 as tail_runtime  # noqa: E402
import run_banking_optimizer_boundary_v12 as runner  # noqa: E402
from banking_cil_v10_routing import activate_additive_receiver  # noqa: E402
from nested_sleep.experts import LowRankMemoryLevel  # noqa: E402


HIDDEN, VOCAB, EOS = 6, 9, 2


class TinyOriginalMLP(nn.Module):
    def __init__(self, *, identity=False):
        super().__init__()
        self.linear = nn.Identity() if identity else nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, hidden):
        return self.linear(hidden)


class TinyMLPWrapper(nn.Module):
    def __init__(self, *, identity=False):
        super().__init__()
        self.original = TinyOriginalMLP(identity=identity)
        self.memory = nn.ModuleList()

    def forward(self, hidden):
        result = self.original(hidden)
        for memory in self.memory:
            result = memory(result)
        return result


class ScriptedProjection(nn.Module):
    """Small deterministic EOS policy used only to exercise generation APIs."""
    def forward(self, hidden):
        token = hidden.argmax(dim=-1)
        logits = torch.full((*token.shape, VOCAB), -10.0, device=hidden.device)
        logits[..., EOS] = 0.0
        logits[..., 3] = torch.where(token == 4, 10.0, -10.0)
        logits[..., EOS] = torch.where((token == 5) | (token == 3), 10.0, logits[..., EOS])
        return logits


class TinyHead(nn.Module):
    def __init__(self, *, scripted=False):
        super().__init__()
        self.original = ScriptedProjection() if scripted else nn.Linear(HIDDEN, VOCAB)
        self.a = nn.Parameter(torch.zeros(2, HIDDEN), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(VOCAB, 2), requires_grad=False)
        self.fp32_accumulation = True

    def forward(self, hidden):
        base = self.original(hidden)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(hidden.to(self.a.dtype), self.a), self.b)
        return base.float() + delta.float()


class TinyDecoderLayer(nn.Module):
    def __init__(self, *, scripted=False):
        super().__init__()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = TinyMLPWrapper(identity=scripted)


class TinyTransformer(nn.Module):
    def __init__(self, *, scripted=False):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([TinyDecoderLayer(scripted=scripted)])
        self.norm = nn.Identity()


class TinyBackbone(nn.Module):
    def __init__(self, *, scripted=False):
        super().__init__()
        self.model = TinyTransformer(scripted=scripted)
        self.lm_head = TinyHead(scripted=scripted)


class TinyModel(nn.Module):
    def __init__(self, *, scripted=False):
        super().__init__()
        self.backbone = TinyBackbone(scripted=scripted)
        if scripted:
            with torch.no_grad():
                self.backbone.model.embedding.weight.zero_()
                self.backbone.model.embedding.weight.copy_(torch.eye(VOCAB, HIDDEN))
        self.backbone.model.layers[-1].mlp.memory = nn.ModuleList()

    def banks(self):
        return []

    def logits(self, input_ids, attention_mask=None):
        layer = self.backbone.model.layers[-1]
        residual = self.backbone.model.embedding(input_ids)
        normalized = layer.post_attention_layernorm(residual)
        mlp_output = layer.mlp(normalized)
        hidden = self.backbone.model.norm(residual + mlp_output)
        return self.backbone.lm_head(hidden)


class TinyContext:
    def __init__(self):
        self.memory = nn.ModuleList([
            LowRankMemoryLevel(dim=HIDDEN, rank=3, capacity=4),
            LowRankMemoryLevel(dim=HIDDEN, rank=3, capacity=4),
        ])
        self.readout_a = nn.Parameter(torch.randn(3, HIDDEN))
        self.readout_b = nn.Parameter(torch.zeros(VOCAB, 3))
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
            "memory": deepcopy(self.memory.state_dict()),
            "readout_a": self.readout_a.detach().clone(),
            "readout_b": self.readout_b.detach().clone(),
            "readout_fp32_accumulation": True,
            "routing": [deepcopy(bank.route_override) for bank in self.banks()],
            "trainable": {name: bool(parameter.requires_grad) for name, parameter in params.items()},
        }

    def load_state_dict(self, state, *, strict=True):
        self.memory.load_state_dict(state["memory"], strict=strict)
        with torch.no_grad():
            self.readout_a.copy_(state["readout_a"])
            self.readout_b.copy_(state["readout_b"])
        for bank, policy in zip(self.banks(), state["routing"]):
            bank.route_override = deepcopy(policy)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(bool(state["trainable"][name]))
        return self


def _make_tx(seed=101):
    fixture = TinyContext()
    live = tail_runtime.MutableContext(
        memory=fixture.memory,
        readout_a=fixture.readout_a,
        readout_b=fixture.readout_b,
        readout_fp32_accumulation=True,
    )
    proposal = {
        "memory.0.a": live.memory[0].a.detach().clone() + 0.03,
        "memory.0.b": live.memory[0].b.detach().clone() - 0.02,
    }
    tx = batching.BatchableAdditiveSleepTransaction(
        live, 0, proposal, lr=0.01, weight_decay=0.0, clip_norm=1.0, seed=seed)
    return live, tx


class TinyBudget:
    def __init__(self):
        self.backward_masks = []
        self.update_calls = []
        self.adamw_updates = 0

    def backward(self, mask):
        self.backward_masks.append(mask.detach().clone())

    def update(self, count):
        self.update_calls.append(count)


def test_real_sample_with_context_stops_at_first_eos_and_masks_forced_suffix():
    model = TinyModel(scripted=True)
    context = tail_runtime.MutableContext(
        memory=nn.ModuleList(),
        readout_a=nn.Parameter(torch.zeros(2, HIDDEN), requires_grad=False),
        readout_b=nn.Parameter(torch.zeros(VOCAB, 2), requires_grad=False),
        readout_fp32_accumulation=True,
    )
    prefix = torch.tensor([[5], [4]], dtype=torch.long)
    result = tail_runtime.sample_with_context(
        model, context, prefix, 5, generator=torch.Generator().manual_seed(4),
        temperature=0.0, attention_mask=torch.ones_like(prefix),
        eos_token_id=EOS, return_info=True)

    assert result.token_ids.tolist() == [[5, EOS, EOS], [4, 3, EOS]]
    assert result.valid_lengths.tolist() == [2, 3]
    assert result.answer_mask().tolist() == [[False, True, False], [False, True, True]]
    assert result.actual_generated_tokens == 3
    assert result.executed_extra_tokens == 1
    assert result.executed_steps == 2


def test_runner_rollout_builder_trims_saved_eos_padding_and_counts_requests(tmp_path):
    class Tokenizer:
        eos_token_id = EOS

    class Common:
        @staticmethod
        def tokenize_prompt(text, tokenizer, classes, eos_token_id, max_length):
            value = 4 if text == "synthetic-a" else 5
            return {"input_ids": [1, value], "prompt_length": 2}

    class Runtime:
        @staticmethod
        def sample_with_context(model, teacher, prefix, count, **kwargs):
            assert prefix.tolist() == [[1, 4], [1, 5]]
            assert count == 4
            return tail_runtime.GenerationResult(
                token_ids=torch.tensor([[1, 4, EOS, EOS], [1, 5, 3, EOS]]),
                prefix_length=2,
                valid_lengths=torch.tensor([3, 4]),
                actual_generated_tokens=3,
                executed_extra_tokens=1,
                executed_steps=2,
            )

    class Budget:
        generated_tokens = 0

        @staticmethod
        def as_dict():
            return {"generated_tokens": 3}

    selected = [
        ("gkd", {"text": "synthetic-a", "id": "fake-a"}, "old"),
        ("gkd", {"text": "synthetic-b", "id": "fake-b"}, "current"),
    ]
    cfg = {"rollout_seed_offset": 3, "task_id": 1, "rollout_microbatch_size": 8,
           "max_new_tokens": 4, "rollout_temperature": 0.0}
    rows, stats = runner._make_rollouts(
        object(), object(), selected, Tokenizer(), ["a", "b"], Common(), cfg,
        torch, torch.device("cpu"), Runtime(), Budget(), 17,
        tmp_path / "progress.json", max_sequence_length=16)

    assert rows == [
        {"input_ids": [1, 4, EOS], "prompt_length": 2, "answer_tokens": 1},
        {"input_ids": [1, 5, 3, EOS], "prompt_length": 2, "answer_tokens": 2},
    ]
    assert stats["generated_answer_tokens"] == 3
    assert stats["actual_generated_tokens"] == 3
    assert stats["forced_extra_eos_positions"] == 1


def test_runner_train_batch_invokes_real_request_closures_and_accounts_masks():
    torch.manual_seed(27)
    model = TinyModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    live, tx = _make_tx(seed=103)
    assert live is tx.live
    rollouts = [
        {"input_ids": [1, 4, 5, EOS], "prompt_length": 2, "answer_tokens": 2},
        {"input_ids": [2, 3, 4, 5, EOS], "prompt_length": 3, "answer_tokens": 2},
    ]
    budget = TinyBudget()
    result = runner._train_request_batch(
        model, tx, rollouts, 0, 2, batching.optimize_request_batch,
        torch, torch.device("cpu"), tail_runtime, budget)

    assert result["request_count"] == 2
    assert result["token_count"] == 4
    assert result["backward_calls"] == 2
    assert result["physical_adamw_steps"] == 1
    assert tx.counters["v12_committed_requests"] == 2
    assert tx.counters["adamw_steps"] == 1
    assert [int(mask.sum().item()) for mask in budget.backward_masks] == [2, 2]
    assert budget.backward_masks[0].tolist() == [[0, 0, 1, 1]]
    assert budget.backward_masks[1].tolist() == [[0, 0, 0, 1, 1]]
    assert budget.update_calls == [1]
    assert budget.adamw_updates == 1
    assert all(parameter.grad is None for parameter in tx.student.parameters())


def test_runner_snapshot_comparison_handles_numpy_rng_inside_dataclass():
    _, tx = _make_tx(seed=107)
    snapshot = tx.snapshot()
    same = deepcopy(snapshot)
    changed = deepcopy(snapshot)
    state = list(changed.payload.numpy_rng)
    state_array = state[1].copy()
    state_array[0] ^= np.uint32(1)
    state[1] = state_array
    changed.payload.numpy_rng = tuple(state)

    assert runner._same_nested(snapshot, same, torch)
    assert not runner._same_nested(snapshot, changed, torch)
