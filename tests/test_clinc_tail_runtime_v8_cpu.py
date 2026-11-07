from __future__ import annotations

import pathlib
import sys
import types

import pytest
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import clinc_tail_runtime_v8 as runtime


def test_parameter_counts_distinguish_resident_slots_from_top_one_execution():
    model = FakeModel()
    context = runtime.clone_mutable_context(model)
    bank = context.banks()[0]
    bank.slots[0].active.fill_(True)
    one = runtime.parameter_counts(context)
    bank.slots[1].active.fill_(True)
    two = runtime.parameter_counts(context)
    assert one["stored_mutable_parameters"] == two["stored_mutable_parameters"]
    assert one["per_token_executed_mutable_parameters_upper_bound"] == two["per_token_executed_mutable_parameters_upper_bound"]
    assert one["activated_resident_mutable_parameters"] < two["activated_resident_mutable_parameters"]
    assert two["active_slots_by_level"] == [2, 0]


class ExpertSlot(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Parameter(torch.randn(2, dim) * .1)
        self.b = nn.Parameter(torch.randn(dim, 2) * .03)
        self.route = nn.Parameter(torch.randn(dim) * .1)
        self.priority = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("active", torch.tensor(False))

    def forward(self, x):
        return torch.nn.functional.linear(torch.nn.functional.silu(
            torch.nn.functional.linear(x, self.a)), self.b)


class Bank(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.slots = nn.ModuleList([ExpertSlot(dim), ExpertSlot(dim)])
        self.route_override = None
        self.dream = False
        self.generator = None
        self.last_routes = []
        self.last_route_audit = None
        self.audit_history = []

    def forward(self, x):
        active = [i for i, slot in enumerate(self.slots) if bool(slot.active)]
        if not active:
            return torch.zeros_like(x)
        if self.route_override and "force" in self.route_override:
            return self.slots[self.route_override["force"]](x)
        # Keep the CPU fixture deterministic and differentiable.
        return self.slots[active[0]](x)


class MemoryLevel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Parameter(torch.randn(3, dim) * .1)
        self.b = nn.Parameter(torch.randn(dim, 3) * .04)
        self.experts = Bank(dim)

    def forward(self, x):
        update = torch.nn.functional.linear(torch.nn.functional.silu(
            torch.nn.functional.linear(x, self.a)), self.b)
        return x + update + self.experts(x)


class OriginalMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Linear(dim, dim * 2)
        self.down = nn.Linear(dim * 2, dim)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class WrappedMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.original = OriginalMLP(dim)
        self.memory = nn.ModuleList([MemoryLevel(dim), MemoryLevel(dim)])

    def forward(self, x):
        y = self.original(x)
        for level in self.memory:
            y = level(y)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pre = nn.Linear(dim, dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = WrappedMLP(dim)

    def forward(self, x):
        residual = self.pre(x) + x
        return residual + self.mlp(self.post_attention_layernorm(residual))


class Readout(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.original = nn.Linear(dim, vocab, bias=False)
        self.a = nn.Parameter(torch.randn(2, dim) * .05)
        self.b = nn.Parameter(torch.randn(vocab, 2) * .02)
        self.fp32_accumulation = True

    def forward(self, x):
        base = self.original(x)
        update = torch.nn.functional.linear(torch.nn.functional.linear(x, self.a), self.b)
        return base.float() + update.float() if self.fp32_accumulation else base + update


class FakeModel(nn.Module):
    def __init__(self, dim=8, vocab=13):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.backbone = nn.Module()
        self.backbone.model = nn.Module()
        self.backbone.model.layers = nn.ModuleList([DecoderLayer(dim)])
        self.backbone.model.norm = nn.LayerNorm(dim)
        self.backbone.lm_head = Readout(dim, vocab)
        self.vocab = vocab

    def banks(self):
        return [level.experts for level in self.backbone.model.layers[-1].mlp.memory]

    def logits(self, ids, mask=None):
        x = self.embedding(ids)
        for layer in self.backbone.model.layers:
            x = layer(x)
        return self.backbone.lm_head(self.backbone.model.norm(x))


def model_with_active_memory():
    torch.manual_seed(1907)
    model = FakeModel()
    for bank in model.banks():
        bank.slots[0].active.fill_(True)
    return model


def named_mutables(model):
    layer = model.backbone.model.layers[-1].mlp
    return {**{f"memory.{k}": v for k, v in layer.memory.state_dict().items()},
            "readout_a": model.backbone.lm_head.a,
            "readout_b": model.backbone.lm_head.b}


def test_full_sequence_tail_logits_and_gradients_match_full_model():
    model = model_with_active_memory()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    cache = runtime.capture(model, ids)
    assert cache.y.shape == cache.residual.shape == (2, 4, 8)
    assert cache.reference_logits.shape == (2, 4, model.vocab)

    context = runtime.clone_mutable_context(model)
    tail = runtime.tail_logits(model, context, cache)
    assert torch.allclose(tail, cache.reference_logits, atol=1e-6, rtol=1e-6)

    weights = torch.linspace(-.7, .8, tail.numel()).reshape_as(tail)
    (model.logits(ids) * weights).sum().backward()
    (runtime.tail_logits(model, context, cache) * weights).sum().backward()
    source = named_mutables(model)
    target = {**{f"memory.{k}": v for k, v in context.memory.state_dict().items()},
              "readout_a": context.readout_a, "readout_b": context.readout_b}
    assert source.keys() == target.keys()
    for name in source:
        sg, tg = source[name].grad, target[name].grad
        assert (sg is None) == (tg is None), name
        if sg is not None:
            assert torch.allclose(sg, tg, atol=2e-6, rtol=2e-5), name


def test_contexts_are_independent_and_state_roundtrips():
    model = model_with_active_memory()
    a = runtime.clone_mutable_context(model)
    b = runtime.clone_mutable_context(model)
    p = next(a.parameters())
    original = p.detach().clone()
    with torch.no_grad():
        p.add_(1)
    assert not torch.equal(p, next(b.parameters()))
    assert torch.equal(next(model.backbone.model.layers[-1].mlp.memory.parameters()), original)
    first_name, first_param = next(iter(a.named_parameters()))
    first_param.requires_grad_(False)
    state = a.state_dict()
    restored = runtime.clone_mutable_context(model)
    restored.load_state_dict(state)
    assert torch.equal(next(a.parameters()), next(restored.parameters()))
    assert dict(restored.named_parameters())[first_name].requires_grad is False
    assert runtime.parameter_counts(restored)["stored_mutable_parameters"] > 0


def test_new_sampled_prefix_is_recaptured_each_generation_step(monkeypatch):
    model = model_with_active_memory()
    context = runtime.clone_mutable_context(model)
    ids_a = torch.tensor([[1, 2]])
    ids_b = torch.tensor([[1, 2, 3]])
    cache_a = runtime.capture(model, ids_a)
    cache_b = runtime.capture(model, ids_b)
    assert cache_a.input_length == 2 and cache_b.input_length == 3
    assert cache_b.y.shape[1] == ids_b.shape[1]
    assert not torch.equal(cache_a.y[:, -1], cache_b.y[:, -1])

    lengths = []
    real_capture = runtime.capture

    def counted_capture(m, ids, mask=None, **kwargs):
        lengths.append(ids.shape[1])
        return real_capture(m, ids, mask, **kwargs)

    monkeypatch.setattr(runtime, "capture", counted_capture)
    gen = torch.Generator().manual_seed(42)
    generated = runtime.sample_with_context(model, context, ids_a, 2, generator=gen, temperature=0,
                                            attention_mask=torch.ones_like(ids_a, dtype=torch.bool))
    assert generated.shape == (1, 4)
    assert lengths == [2, 3]


def test_capture_restores_global_rng_state():
    model = model_with_active_memory()
    ids = torch.tensor([[1, 2, 3]])
    torch.manual_seed(7331)
    state = torch.random.get_rng_state().clone()
    runtime.capture(model, ids)
    assert torch.equal(state, torch.random.get_rng_state())

    bank = model.banks()[0]
    bank.generator = torch.Generator().manual_seed(91)
    generator_state = bank.generator.get_state().clone()
    original_logits = model.logits

    def stochastic_logits(token_ids, mask=None):
        torch.rand((), generator=bank.generator)
        return original_logits(token_ids, mask)

    model.logits = stochastic_logits
    runtime.capture(model, ids)
    assert torch.equal(generator_state, bank.generator.get_state())


def test_eos_stops_per_row_and_masks_forced_batch_placeholders(monkeypatch):
    model = model_with_active_memory()
    context = runtime.clone_mutable_context(model)
    prefixes = torch.tensor([[1, 2], [5, 6]])
    captures = []
    real_capture = runtime.capture

    def counted_capture(m, ids, mask=None, **kwargs):
        captures.append((ids.shape[1], None if mask is None else mask.detach().clone()))
        return real_capture(m, ids, mask, **kwargs)

    logits_calls = 0

    def scripted_tail(_model, _context, cache):
        nonlocal logits_calls
        result = torch.zeros((2, cache.input_length, model.vocab))
        if logits_calls == 0:
            result[0, -1, 0] = 9.0  # row 0 emits EOS immediately
            result[1, -1, 3] = 9.0
        elif logits_calls == 1:
            result[1, -1, 4] = 9.0  # row 0 is a forced placeholder
        else:
            result[1, -1, 0] = 9.0  # row 1 emits EOS; row 0 remains placeholder
        logits_calls += 1
        return result

    monkeypatch.setattr(runtime, "capture", counted_capture)
    monkeypatch.setattr(runtime, "tail_logits", scripted_tail)
    result = runtime.sample_with_context(
        model, context, prefixes, 5, generator=torch.Generator().manual_seed(1),
        temperature=0, attention_mask=torch.ones_like(prefixes), eos_token_id=0,
        return_info=True)
    assert isinstance(result, runtime.GenerationResult)
    assert result.token_ids.tolist() == [[1, 2, 0, 0, 0], [5, 6, 3, 4, 0]]
    assert result.valid_lengths.tolist() == [3, 5]
    assert result.answer_mask().tolist() == [[False, False, True, False, False],
                                              [False, False, True, True, True]]
    assert result.actual_generated_tokens == 4
    assert result.executed_extra_tokens == 2
    assert result.executed_steps == 3
    assert [length for length, _ in captures] == [2, 3, 4]
    assert captures[2][1].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]


def test_capture_removes_hooks_and_restores_audit_state_even_on_error():
    model = model_with_active_memory()
    banks = model.banks()
    marker = [{"preserved": True}]
    banks[0].audit_history = marker
    ids = torch.tensor([[1, 2]])
    runtime.capture(model, ids)
    assert banks[0].audit_history is marker
    wrapped = model.backbone.model.layers[-1].mlp
    assert len(wrapped.original._forward_hooks) == 0
    assert len(model.backbone.model.layers[-1].post_attention_layernorm._forward_pre_hooks) == 0

    original_logits = model.logits

    def fail(*_args, **_kwargs):
        raise RuntimeError("fixture failure")

    model.logits = fail
    with pytest.raises(RuntimeError, match="fixture failure"):
        runtime.capture(model, ids)
    assert len(wrapped.original._forward_hooks) == 0
    assert len(model.backbone.model.layers[-1].post_attention_layernorm._forward_pre_hooks) == 0
    model.logits = original_logits


def test_causal_ce_and_full_vocabulary_kl_cover_all_masked_label_tokens():
    torch.manual_seed(23)
    logits = torch.randn(2, 6, 17, requires_grad=True)
    ids = torch.randint(0, 17, (2, 6))
    answer_mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    ce, ce_tokens = runtime.causal_token_ce(logits, ids, answer_mask)
    ref = torch.nn.functional.cross_entropy(
        logits[:, :-1][answer_mask[:, 1:]], ids[:, 1:][answer_mask[:, 1:]])
    assert ce_tokens == 7
    assert torch.allclose(ce, ref)
    ce.backward()
    assert logits.grad is not None and bool(logits.grad.abs().sum() > 0)

    teacher = torch.randn(2, 6, 17)
    student = torch.randn(2, 6, 17, requires_grad=True)
    kl, kl_tokens = runtime.causal_token_kl(student, teacher, answer_mask, temperature=1.7)
    assert kl_tokens == 7
    kl.backward()
    assert student.grad is not None and bool(student.grad.abs().sum() > 0)
    assert teacher.grad is None


def test_causal_losses_reject_empty_or_unshiftable_answer_masks():
    logits = torch.randn(1, 3, 5)
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="selects no"):
        runtime.causal_token_ce(logits, ids, torch.zeros_like(ids, dtype=torch.bool))
    with pytest.raises(ValueError, match="positive"):
        runtime.causal_token_kl(logits, logits, torch.ones_like(ids, dtype=torch.bool), temperature=0)
    with pytest.raises(ValueError, match="nonnegative integer"):
        runtime.sample_with_context(FakeModel(), runtime.clone_mutable_context(FakeModel()),
                                    torch.tensor([[1]]), True,
                                    generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("bad_ids,bad_mask", [
    (torch.tensor([1, 2]), None),
    (torch.tensor([[1, 2]]), torch.ones(1, 3)),
    (torch.empty((1, 0), dtype=torch.long), None),
])
def test_capture_rejects_malformed_inputs(bad_ids, bad_mask):
    model = model_with_active_memory()
    with pytest.raises(ValueError):
        runtime.capture(model, bad_ids, bad_mask)
