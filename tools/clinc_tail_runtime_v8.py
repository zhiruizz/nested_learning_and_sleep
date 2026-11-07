"""Frozen-backbone tail runtime for bounded CLINC continual-learning pilots.

This is a local last-MLP augmentation runtime, not a native HOPE/Titans model.
For every input sequence (including every newly sampled teacher/student prefix),
``capture`` runs the complete frozen Llama backbone and records the last block's
pre-memory MLP output and residual at the original ``[B, S, H]`` shape. Mutable
memory and readout adapters can then be evaluated in isolated small contexts.

The implementation intentionally has no KV cache yet: correctness and exact
full-sequence matrix shapes take priority over generation speed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn


def _parts(model):
    try:
        layer = model.backbone.model.layers[-1]
        wrapper = layer.mlp
        memory = wrapper.memory
        original_mlp = wrapper.original
        residual_norm = layer.post_attention_layernorm
        final_norm = model.backbone.model.norm
        head = model.backbone.lm_head
    except (AttributeError, IndexError, TypeError) as exc:
        raise TypeError("expected a LlamaCMSLM with wrapped final-layer MLP") from exc
    if not isinstance(memory, (nn.ModuleList, list, tuple)):
        raise TypeError("final MLP memory must be a sequence of modules")
    if not all(callable(x) for x in (original_mlp, residual_norm, final_norm, head)):
        raise TypeError("backbone tail components must be callable modules")
    return layer, wrapper, memory, original_mlp, residual_norm, final_norm, head


@dataclass
class TailCache:
    """Detached full-sequence tensors from one exact input sequence."""
    y: Tensor
    residual: Tensor
    reference_logits: Tensor
    input_length: int

    def to(self, device) -> "TailCache":
        return TailCache(*(x.to(device).clone() for x in
                           (self.y, self.residual, self.reference_logits)), self.input_length)


@dataclass
class GenerationResult:
    """Bounded batch generation with first-EOS lengths and work accounting."""
    token_ids: Tensor
    prefix_length: int
    valid_lengths: Tensor
    actual_generated_tokens: int
    executed_extra_tokens: int
    executed_steps: int

    def answer_mask(self) -> Tensor:
        """Mark emitted answer tokens through and including each row's first EOS."""
        positions = torch.arange(self.token_ids.shape[1], device=self.token_ids.device)[None, :]
        return (positions >= self.prefix_length) & (positions < self.valid_lengths[:, None])


@dataclass
class MutableContext:
    """Independent mutable tail state; the multi-billion-parameter backbone is shared."""
    memory: nn.ModuleList
    readout_a: nn.Parameter
    readout_b: nn.Parameter
    readout_fp32_accumulation: bool

    def parameters(self):
        yield from self.memory.parameters()
        yield self.readout_a
        yield self.readout_b

    def named_parameters(self):
        for name, parameter in self.memory.named_parameters():
            yield f"memory.{name}", parameter
        yield "readout_a", self.readout_a
        yield "readout_b", self.readout_b

    def banks(self):
        return [level.experts for level in self.memory if hasattr(level, "experts")]

    def state_dict(self):
        trainable = {name: bool(parameter.requires_grad)
                     for name, parameter in self.named_parameters()}
        return {
            "memory": {k: v.detach().cpu().clone() for k, v in self.memory.state_dict().items()},
            "readout_a": self.readout_a.detach().cpu().clone(),
            "readout_b": self.readout_b.detach().cpu().clone(),
            "readout_fp32_accumulation": bool(self.readout_fp32_accumulation),
            "routing": [deepcopy(bank.route_override) for bank in self.banks()],
            "trainable": trainable,
        }

    def load_state_dict(self, state, *, strict: bool = True):
        if set(state) != {"memory", "readout_a", "readout_b", "readout_fp32_accumulation", "routing", "trainable"}:
            raise ValueError("mutable context state schema mismatch")
        result = self.memory.load_state_dict(state["memory"], strict=strict)
        with torch.no_grad():
            self.readout_a.copy_(state["readout_a"].to(self.readout_a))
            self.readout_b.copy_(state["readout_b"].to(self.readout_b))
        banks = self.banks()
        if len(state["routing"]) != len(banks):
            raise ValueError("mutable context routing count mismatch")
        for bank, policy in zip(banks, state["routing"]):
            bank.route_override = deepcopy(policy)
            bank.dream = False
            bank.generator = None
        self.readout_fp32_accumulation = bool(state["readout_fp32_accumulation"])
        params = dict(self.named_parameters())
        if set(state["trainable"]) != set(params):
            raise ValueError("mutable context trainable-scope mismatch")
        for name, parameter in params.items():
            parameter.requires_grad_(bool(state["trainable"][name]))
        return result


def capture(model, ids: Tensor, mask: Optional[Tensor] = None, *, device=None) -> TailCache:
    """Recompute exact frozen-backbone tail features for ``ids``.

    This performs a fresh complete model forward on every call. It never keys or
    reuses a prior cache, so sampled continuations must call it again after their
    token sequence changes. Captures are detached; adapter gradients are computed
    later by :func:`tail_logits`.
    """
    if not isinstance(ids, Tensor) or ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
        raise ValueError("ids must have shape [batch>=1, sequence>=1]")
    if mask is not None and (not isinstance(mask, Tensor) or mask.shape != ids.shape):
        raise ValueError("mask must match ids shape")
    layer, _wrapper, _memory, original_mlp, residual_norm, _fn, _head = _parts(model)
    captured = {"residual": [], "y": []}

    def pre_residual(_module, inputs):
        if not inputs or not isinstance(inputs[0], Tensor):
            raise TypeError("residual hook did not receive a tensor")
        captured["residual"].append(inputs[0])

    def post_original(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, Tensor):
            raise TypeError("original MLP hook did not receive a tensor")
        captured["y"].append(value)

    handles = []
    bank_attrs = []
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        for bank in model.banks():
            attrs = {}
            for name in ("last_routes", "last_route_audit", "audit_history", "dream", "generator"):
                if hasattr(bank, name):
                    attrs[name] = getattr(bank, name)
            generator = attrs.get("generator")
            if generator is not None and hasattr(generator, "get_state"):
                attrs["generator_state"] = generator.get_state().clone()
            bank_attrs.append((bank, attrs))
        handles.append(residual_norm.register_forward_pre_hook(pre_residual))
        handles.append(original_mlp.register_forward_hook(post_original))
        # Evaluation-only capture; dropout remains governed by the caller's mode.
        with torch.no_grad():
            logits = model.logits(ids, mask)
        if len(captured["residual"]) != 1 or len(captured["y"]) != 1:
            raise RuntimeError("expected one complete final-layer capture")
        residual, y = captured["residual"][0], captured["y"][0]
        if y.ndim != 3 or residual.shape != y.shape or y.shape[:2] != ids.shape:
            raise RuntimeError("captured tail tensors do not match original [B,S,H] sequence shape")
        if logits.ndim != 3 or logits.shape[:2] != ids.shape:
            raise RuntimeError("full model logits do not match input sequence shape")
        target = torch.device(device) if device is not None else ids.device
        return TailCache(y.detach().to(target).clone(), residual.detach().to(target).clone(),
                         logits.detach().to(target).clone(), ids.shape[1])
    finally:
        for handle in reversed(handles):
            handle.remove()
        for bank, attrs in bank_attrs:
            for name, value in attrs.items():
                if name != "generator_state":
                    setattr(bank, name, value)
            generator = attrs.get("generator")
            if generator is not None and "generator_state" in attrs:
                generator.set_state(attrs["generator_state"])
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def clone_mutable_context(model) -> MutableContext:
    """Clone only last-MLP memories and readout LoRA; share all frozen weights."""
    _layer, wrapper, memory, _original, _post, _final, head = _parts(model)
    independent = deepcopy(nn.ModuleList(list(memory)))
    # Dream samplers/generators are runtime controls, not clonable learned state.
    for level in independent:
        if hasattr(level, "experts"):
            level.experts.dream = False
            level.experts.generator = None
            level.experts.last_routes = []
            level.experts.last_route_audit = None
            level.experts.audit_history = []
    return MutableContext(
        memory=independent,
        readout_a=nn.Parameter(head.a.detach().clone(), requires_grad=head.a.requires_grad),
        readout_b=nn.Parameter(head.b.detach().clone(), requires_grad=head.b.requires_grad),
        readout_fp32_accumulation=bool(head.fp32_accumulation),
    )


def tail_logits(model, context: MutableContext, cache: TailCache) -> Tensor:
    """Compute sequence logits from fresh cache and one isolated mutable context."""
    if not isinstance(context, MutableContext) or not isinstance(cache, TailCache):
        raise TypeError("context and cache must be MutableContext and TailCache")
    if cache.y.ndim != 3 or cache.residual.shape != cache.y.shape:
        raise ValueError("tail cache tensors must match [batch, sequence, hidden]")
    if cache.y.shape[1] != cache.input_length:
        raise ValueError("tail cache length metadata mismatch")
    _layer, _wrapper, _memory, _original, _post, final_norm, head = _parts(model)
    if cache.y.device != cache.residual.device:
        raise ValueError("cached y and residual must share a device")
    hidden = cache.y
    for level in context.memory:
        hidden = level(hidden)
    hidden = final_norm(cache.residual + hidden)
    base = head.original(hidden)
    update = torch.nn.functional.linear(
        torch.nn.functional.linear(hidden.to(context.readout_a.dtype), context.readout_a),
        context.readout_b)
    if context.readout_fp32_accumulation:
        return base.float() + update.float()
    return base.to(update.dtype) + update


@torch.no_grad()
def sample_with_context(model, context: MutableContext, prefix: Tensor, count: int, *,
                        generator: torch.Generator, temperature: float = 0.8,
                        attention_mask: Optional[Tensor] = None,
                        eos_token_id: Optional[int] = None,
                        return_info: bool = False) -> Tensor | GenerationResult:
    """Autoregress with a mutable context, recapturing the full sequence per step.

    For each emitted token this recomputes frozen-backbone features from the
    current generated sequence. It is deliberately straightforward and slow,
    providing a correct reference path before any KV-cache optimization.
    """
    if not isinstance(prefix, Tensor) or prefix.ndim != 2 or prefix.shape[1] < 1:
        raise ValueError("prefix must have shape [batch, nonempty sequence]")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if attention_mask is not None:
        if attention_mask.shape != prefix.shape:
            raise ValueError("attention_mask must match prefix")
        if not bool(attention_mask.bool().all()):
            raise ValueError("batched generation requires unpadded prefixes")
    if eos_token_id is not None and (not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool)
                                     or eos_token_id < 0):
        raise ValueError("eos_token_id must be a nonnegative integer or None")
    generated = prefix.clone()
    batch = prefix.shape[0]
    valid_generated = torch.zeros(batch, dtype=torch.long, device=prefix.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=prefix.device)
    actual_tokens = extra_tokens = executed_steps = 0
    for _ in range(count):
        active = ~finished
        active_count = int(active.sum())
        if active_count == 0:
            break
        cache = capture(model, generated, attention_mask)
        logits = tail_logits(model, context, cache)[:, -1].float()
        if eos_token_id is not None and eos_token_id >= logits.shape[-1]:
            raise ValueError("eos_token_id is outside the model vocabulary")
        active_logits = logits[active]
        if temperature == 0:
            sampled = active_logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(active_logits / temperature, dim=-1)
            sampled = torch.multinomial(probs, 1, generator=generator)
        next_token = torch.full((batch, 1), eos_token_id if eos_token_id is not None else 0,
                                device=generated.device, dtype=generated.dtype)
        next_token[active] = sampled.to(next_token.dtype)
        if eos_token_id is not None:
            newly_finished = active & next_token[:, 0].eq(eos_token_id)
            valid_generated[active] += 1
            finished = finished | newly_finished
        else:
            valid_generated += 1
        actual_tokens += active_count
        extra_tokens += batch - active_count
        executed_steps += 1
        generated = torch.cat((generated, next_token), dim=-1)
        if attention_mask is not None:
            # First EOS is a real target; later forced EOS placeholders are masked.
            next_mask = active[:, None].to(device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, next_mask), dim=-1)
    if not return_info:
        return generated
    return GenerationResult(
        token_ids=generated,
        prefix_length=prefix.shape[1],
        valid_lengths=prefix.shape[1] + valid_generated,
        actual_generated_tokens=actual_tokens,
        executed_extra_tokens=extra_tokens,
        executed_steps=executed_steps,
    )


def causal_token_ce(logits: Tensor, token_ids: Tensor, answer_mask: Tensor):
    """Mean full-vocabulary CE over answer tokens in a prompt+label sequence.

    ``answer_mask`` has the same shape as ``token_ids`` and marks target token
    positions (prompt positions are false). The standard causal shift scores
    ``logits[:, t]`` against ``token_ids[:, t+1]``. Returns ``(loss, token_count)``.
    """
    if logits.ndim != 3 or token_ids.ndim != 2 or logits.shape[:2] != token_ids.shape:
        raise ValueError("logits/token_ids must have shapes [B,S,V] and [B,S]")
    if answer_mask.shape != token_ids.shape:
        raise ValueError("answer_mask must match token_ids")
    if token_ids.shape[1] < 2:
        raise ValueError("causal CE requires at least two sequence positions")
    mask = answer_mask[:, 1:].bool()
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("answer_mask selects no causally predictable answer tokens")
    targets = token_ids[:, 1:].to(device=logits.device, dtype=torch.long)
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    selected = mask.to(logits.device).reshape(-1)
    return losses[selected].mean(), count


def causal_token_kl(student_logits: Tensor, teacher_logits: Tensor, answer_mask: Tensor,
                    *, temperature: float = 1.0):
    """Full-vocabulary teacher-to-student KL over sampled answer suffix tokens.

    Both logits must be computed for the exact same token sequence. In the
    sampled GKD path, call :func:`capture` after the sampled sequence changes,
    then evaluate both mutable contexts from that fresh cache. Teacher logits are
    detached and only answer positions selected by ``answer_mask`` contribute.
    Returns ``(loss, token_count)`` with the conventional ``T**2`` multiplier.
    """
    if student_logits.ndim != 3 or teacher_logits.shape != student_logits.shape:
        raise ValueError("student and teacher logits must match [B,S,V]")
    if answer_mask.shape != student_logits.shape[:2]:
        raise ValueError("answer_mask must match logits [B,S]")
    if student_logits.shape[1] < 2:
        raise ValueError("causal KL requires at least two sequence positions")
    if not (temperature > 0):
        raise ValueError("temperature must be positive")
    mask = answer_mask[:, 1:].bool().to(student_logits.device)
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("answer_mask selects no causally predictable answer tokens")
    s = student_logits[:, :-1].float() / temperature
    t = teacher_logits[:, :-1].detach().float() / temperature
    per_position = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(s, dim=-1),
        torch.nn.functional.softmax(t, dim=-1), reduction="none").sum(-1)
    loss = per_position[mask].mean() * (temperature ** 2)
    return loss, count


def parameter_counts(context: MutableContext) -> dict:
    """Separate allocated storage from natural top-one execution per token.

    Activated expert weights are resident knowledge, but each bank normally
    executes at most one expert for any particular token. Whole-batch execution
    can visit multiple experts. These counts are not FLOP measurements.
    """
    stored = resident_experts = executed_upper = trainable = 0
    for parameter in context.parameters():
        n = parameter.numel()
        stored += n
        if parameter.requires_grad:
            trainable += n
    for bank in context.banks():
        active_ids = {i for i, slot in enumerate(bank.slots) if bool(slot.active)}
        sizes = [sum(p.numel() for p in bank.slots[i].parameters()) for i in active_ids]
        resident_experts += sum(sizes)
        if sizes:
            multiplicity = 1
            policy = bank.route_override or {}
            if policy.get("mix") is not None or policy.get("include") is not None or getattr(bank, "dream", False):
                multiplicity += 1
            executed_upper += min(len(sizes), multiplicity) * max(sizes)
    # Dense level matrices and readout adapters are always active computations.
    for level in context.memory:
        executed_upper += level.a.numel() + level.b.numel()
    dense = sum(level.a.numel() + level.b.numel() for level in context.memory)
    dense += context.readout_a.numel() + context.readout_b.numel()
    executed_upper += context.readout_a.numel() + context.readout_b.numel()
    return {"stored_mutable_parameters": stored,
            "activated_resident_mutable_parameters": dense + resident_experts,
            "per_token_executed_mutable_parameters_upper_bound": executed_upper,
            "trainable_mutable_parameters": trainable,
            "active_slots_by_level": [sum(bool(s.active) for s in b.slots) for b in context.banks()]}
