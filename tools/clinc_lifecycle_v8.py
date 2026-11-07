"""Explicit wake clocks and pre-update Sleep transactions for local CLINC v8.

Periods count minibatches. This is a local last-MLP augmentation protocol, not
an assertion that the paper specified these clocks or this parameterization.
The caller clips wake gradients once; this module buffers LR-weighted sums.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import random
from typing import Mapping

import numpy as np
import torch
from torch import Tensor


def _params(ctx):
    entries = list(ctx.named_parameters())
    if len(entries) != len(dict(entries)):
        raise ValueError("duplicate context parameter names")
    return dict(entries)


def _owned(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _owned(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_owned(x) for x in value)
    return deepcopy(value)


def _hash(value):
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, Tensor):
            t = x.detach().cpu().contiguous()
            digest.update(str((t.dtype, tuple(t.shape))).encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, Mapping):
            for k in sorted(x):
                digest.update(str(k).encode())
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)
        else:
            digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


def _state_hash(ctx, exclude=()):
    state = ctx.state_dict()
    for name in exclude:
        if name.startswith("memory."):
            state["memory"].pop(name[len("memory."):], None)
        else:
            state.pop(name, None)
    return _hash(state)


def _freeze(ctx, allowed=()):
    allowed = set(allowed)
    for n, p in ctx.named_parameters():
        p.requires_grad_(n in allowed)
        p.grad = None


def _versions(ctx):
    # No GPU-to-host copies in the per-optimizer-step guard. Deep hashes are
    # checked at construction, checkpoint and commit boundaries as well.
    items = list(ctx.named_parameters()) + [("buffer." + n, b) for n, b in ctx.memory.named_buffers()]
    return {n: (id(t), t._version) for n, t in items}


def _bank_state(ctx):
    return [{"dream": bool(getattr(b, "dream", False)),
             "generator": None if getattr(b, "generator", None) is None else {
                 "device": str(b.generator.device), "state": b.generator.get_state().clone()}}
            for b in ctx.banks()]


def _restore_banks(ctx, states):
    if len(states) != len(ctx.banks()):
        raise ValueError("bank state topology mismatch")
    for bank, state in zip(ctx.banks(), states):
        if hasattr(bank, "dream"):
            bank.dream = state["dream"]
        g = state["generator"]
        bank.generator = None if g is None else torch.Generator(device=g["device"])
        if g is not None:
            bank.generator.set_state(g["state"].cpu().clone())


def _base_names(level):
    return {f"memory.{level}.a", f"memory.{level}.b"}


def _slot_names(ctx, level, slot):
    ids = {id(p) for p in ctx.banks()[level].slots[slot].parameters()}
    names = {n for n, p in ctx.named_parameters() if id(p) in ids}
    if len(names) != 2 or {n.rsplit(".", 1)[1] for n in names} != {"a", "b"}:
        raise ValueError("expected exactly receiver slot A/B parameters")
    return names


@dataclass
class WakeSnapshot:
    context_state: dict
    pending: dict
    clocks: dict
    token_counts: dict
    ticks: int
    periods: tuple
    base_lr: float
    fast_lr: float
    bank_states: list


class WakeController:
    def __init__(self, context, periods=(16, 64, 256, 1024, None), *, base_lr, fast_lr):
        if len(periods) != len(context.memory) or len(context.banks()) != len(periods):
            raise ValueError("one expert bank and one period required per level")
        if any(p is not None and (type(p) is not int or p < 1) for p in periods):
            raise ValueError("periods must be positive integers or None")
        if periods[-1] is not None:
            raise ValueError("terminal level must be held fixed in this local protocol")
        if any(not math.isfinite(x) or x < 0 for x in (base_lr, fast_lr)):
            raise ValueError("finite nonnegative learning rates required")
        self.periods = tuple(periods)
        self.base_lr, self.fast_lr = float(base_lr), float(fast_lr)
        self.clocks = {i: 0 for i, p in enumerate(periods) if p is not None}
        self.token_counts = {i: 0 for i in self.clocks}
        self.pending = {}
        self.ticks = 0
        self.bind_context(context)

    def bind_context(self, context):
        if len(context.banks()) != len(self.periods):
            raise ValueError("context topology changed")
        expected = set().union(*(_base_names(i) for i in self.clocks))
        if not expected.issubset(_params(context)):
            raise ValueError("context lacks base A/B pairs")
        self.context = context
        self._fast_names = set()

    def prepare_wake(self):
        if any(getattr(b, "route_override", None) is not None or getattr(b, "dream", False)
               for b in self.context.banks()):
            raise ValueError("wake protocol requires natural routing without dreaming")
        bank = self.context.banks()[0]
        active = [i for i, s in enumerate(bank.slots) if bool(s.active)]
        if not active:
            bank.activate()
            active = [i for i, s in enumerate(bank.slots) if bool(s.active)]
        if len(active) != 1:
            raise ValueError("wake requires exactly one fast expert")
        self._fast_names = _slot_names(self.context, 0, active[0])
        allowed = self._fast_names | set().union(*(_base_names(i) for i in self.clocks))
        _freeze(self.context, allowed)
        return allowed

    def record_tick(self, gradients, *, token_count=1):
        if type(token_count) is not int or token_count < 1:
            raise ValueError("positive token count required")
        if self.proposals():
            raise RuntimeError("consume all due proposals before another wake tick")
        self.prepare_wake()
        params = _params(self.context)
        allowed = {n for n, p in params.items() if p.requires_grad}
        grads = {}
        for n, g in gradients.items():
            if n not in params:
                raise KeyError(n)
            if g is None:
                continue
            if n not in allowed or g.shape != params[n].shape or not bool(torch.isfinite(g).all()):
                raise ValueError(f"gradient outside scope or invalid: {n}")
            grads[n] = g.detach().to(params[n]).clone()
        with torch.no_grad():
            for n in self._fast_names:
                if n in grads:
                    params[n].add_(grads[n], alpha=-self.fast_lr)
        for level in self.clocks:
            self.clocks[level] += 1
            self.token_counts[level] += token_count
            for n in _base_names(level):
                delta = self.base_lr * grads.get(n, torch.zeros_like(params[n]))
                self.pending[n] = self.pending.get(n, torch.zeros_like(params[n])) + delta
        self.ticks += 1
        # Proposals are requested separately and retained until explicit consume.
        return {}

    def proposals(self):
        params = _params(self.context)
        return {i: {n: params[n].detach().clone() - self.pending[n].to(params[n]) for n in _base_names(i)}
                for i, clock in self.clocks.items() if clock == self.periods[i]}

    def consume(self, level):
        if level not in self.clocks or self.clocks[level] != self.periods[level]:
            raise ValueError("level does not have a due proposal")
        for n in _base_names(level):
            del self.pending[n]
        self.clocks[level] = 0
        self.token_counts[level] = 0

    def apply(self, level, proposal=None):
        due = self.proposals()
        if level not in due:
            raise ValueError("level is not due")
        proposal = due[level] if proposal is None else proposal
        if set(proposal) != _base_names(level) or any(not torch.equal(v.to(due[level][n]), due[level][n]) for n, v in proposal.items()):
            raise ValueError("proposal does not match pending absolute base values")
        params = _params(self.context)
        with torch.no_grad():
            for n, v in proposal.items():
                params[n].copy_(v.to(params[n]))
        self.consume(level)

    def snapshot(self):
        return WakeSnapshot(_owned(self.context.state_dict()), _owned(self.pending), dict(self.clocks),
                            dict(self.token_counts), self.ticks, self.periods,
                            self.base_lr, self.fast_lr, _bank_state(self.context))

    def restore(self, snap):
        if not isinstance(snap, WakeSnapshot) or (snap.periods, snap.base_lr, snap.fast_lr) != (self.periods, self.base_lr, self.fast_lr):
            raise ValueError("wake snapshot config mismatch")
        if set(snap.clocks) != set(self.clocks) or set(snap.token_counts) != set(self.clocks):
            raise ValueError("wake snapshot clock mismatch")
        if any(type(v) is not int or not 0 <= v <= self.periods[i] for i, v in snap.clocks.items()):
            raise ValueError("invalid restored clocks")
        expected = set().union(*(_base_names(i) for i, clock in snap.clocks.items() if clock))
        if set(snap.pending) != expected:
            raise ValueError("pending gradient scope mismatch")
        self.context.load_state_dict(_owned(snap.context_state))
        _restore_banks(self.context, snap.bank_states)
        params = _params(self.context)
        self.pending = {n: v.detach().to(params[n]).clone() for n, v in snap.pending.items()}
        self.clocks, self.token_counts, self.ticks = dict(snap.clocks), dict(snap.token_counts), int(snap.ticks)
        self._fast_names = set()


@dataclass
class SleepSnapshot:
    teacher_state: dict
    student_state: dict
    optimizer_state: dict
    metadata: dict
    counters: dict
    generator_state: Tensor
    cpu_rng: Tensor
    cuda_rng: object
    python_rng: object
    numpy_rng: object
    bank_states: dict


class SleepTransaction:
    """Separate frozen pre-update teacher and prospective student.

    The sender is a memory LEVEL. Only that level's expert bank is reset, and
    exactly one fresh slot in the next level is added and trained. The live
    context remains untouched until the caller accepts the returned student.
    """
    def __init__(self, live_context, sender, prospective, *, lr=1e-3,
                 weight_decay=0., clip_norm=1., seed=17):
        if any(getattr(b, "route_override", None) is not None or getattr(b, "dream", False)
               for b in live_context.banks()):
            raise ValueError("sleep protocol requires natural routing without dreaming")
        if type(sender) is not int or not 0 <= sender < len(live_context.banks())-1:
            raise ValueError("sender must be a nonterminal level")
        if set(prospective) != _base_names(sender):
            raise ValueError("prospective state must be exactly sender base A/B")
        if not all(math.isfinite(x) and x >= 0 for x in (lr, weight_decay)) or not math.isfinite(clip_norm) or clip_norm <= 0:
            raise ValueError("invalid sleep optimization config")
        self.live = live_context
        self._live_hash = _state_hash(self.live)
        self.sender, self.receiver = sender, sender+1
        self.lr, self.weight_decay, self.clip_norm = float(lr), float(weight_decay), float(clip_norm)
        device = next(iter(self.live.parameters())).device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.teacher, self.student = deepcopy(live_context), deepcopy(live_context)
        _freeze(self.teacher)
        for module in self.teacher.memory:
            module.eval()
        params = _params(self.student)
        with torch.no_grad():
            for n, value in prospective.items():
                if value.shape != params[n].shape or not bool(torch.isfinite(value).all()):
                    raise ValueError("invalid absolute prospective base value")
                params[n].copy_(value.to(params[n]))
        sender_bank = self.student.banks()[sender]
        reset_ids = [i for i, slot in enumerate(sender_bank.slots) if bool(slot.active)]
        sender_bank.reset_active()
        # The legacy bank reset retained a learned A basis. For this protocol
        # reset every sender low-rank pair: fresh N(0,.02) A and zero-output B.
        # The initialization distribution is our explicit implementation choice.
        with torch.no_grad():
            for i in reset_ids:
                sender_bank.slots[i].a.normal_(mean=0., std=.02, generator=self.generator)
                sender_bank.slots[i].b.zero_()
        receiver = self.student.banks()[self.receiver]
        inactive = [i for i, slot in enumerate(receiver.slots) if not bool(slot.active)]
        if not inactive:
            raise RuntimeError("fresh receiver capacity exhausted")
        self.slot = inactive[0]
        receiver.activate(self.slot)
        self.allowed_names = _slot_names(self.student, self.receiver, self.slot)
        _freeze(self.student, self.allowed_names)
        self.optimizer = self._new_optimizer()
        self.counters = {"adamw_steps": 0, "gkd_calls": 0, "lti_calls": 0,
                         "tokens": 0, "backward_attempts": 0, "zero_gradient_steps": 0, "closed": False}
        self._set_guards()

    def _new_optimizer(self):
        p = _params(self.student)
        return torch.optim.AdamW([p[n] for n in sorted(self.allowed_names)],
                                 lr=self.lr, weight_decay=self.weight_decay)

    def _set_guards(self):
        self._teacher_hash = _state_hash(self.teacher)
        self._student_frozen_hash = _state_hash(self.student, self.allowed_names)
        self._live_versions, self._teacher_versions = _versions(self.live), _versions(self.teacher)
        self._student_versions = {n: v for n, v in _versions(self.student).items() if n not in self.allowed_names}
        ptrs = [{p.untyped_storage().data_ptr() for p in c.parameters() if p.numel()}
                for c in (self.live, self.teacher, self.student)]
        if any(ptrs[i] & ptrs[j] for i, j in ((0,1), (0,2), (1,2))):
            raise RuntimeError("mutable storage alias across sleep contexts")

    def _check(self, *, deep=False):
        if _versions(self.live) != self._live_versions or _versions(self.teacher) != self._teacher_versions:
            raise RuntimeError("live/teacher mutated during sleep")
        if {n: v for n, v in _versions(self.student).items() if n not in self.allowed_names} != self._student_versions:
            raise RuntimeError("frozen student tensor changed")
        for n, p in self.student.named_parameters():
            if p.requires_grad != (n in self.allowed_names) or (n not in self.allowed_names and p.grad is not None):
                raise RuntimeError("sleep receiver gradient scope violation")
        if any(p.requires_grad or p.grad is not None for p in self.teacher.parameters()):
            raise RuntimeError("teacher is not frozen")
        if deep and (_state_hash(self.live) != self._live_hash or
                     _state_hash(self.teacher) != self._teacher_hash or
                     _state_hash(self.student, self.allowed_names) != self._student_frozen_hash):
            raise RuntimeError("sleep isolation hash mismatch")

    def teacher_context(self):
        return self.teacher

    def student_context(self):
        return self.student

    def optimize(self, loss, kind, tokens):
        if self.counters["closed"]:
            raise RuntimeError("sleep transaction is closed")
        if kind not in ("gkd", "lti") or type(tokens) is not int or tokens < 1:
            raise ValueError("sleep kind and positive token count required")
        if not isinstance(loss, Tensor) or loss.ndim or not bool(torch.isfinite(loss)) or not loss.requires_grad:
            raise ValueError("sleep requires finite differentiable scalar loss")
        self._check()
        self.optimizer.zero_grad(set_to_none=True)
        self.counters["backward_attempts"] += 1
        self.counters["tokens"] += tokens
        loss.backward()
        self._check()
        params = [_params(self.student)[n] for n in sorted(self.allowed_names)]
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError("missing/nonfinite receiver gradients")
        norm = torch.nn.utils.clip_grad_norm_(params, self.clip_norm, error_if_nonfinite=True)
        self.counters["zero_gradient_steps"] += int(float(norm) == 0.)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.counters["adamw_steps"] += 1
        self.counters[f"{kind}_calls"] += 1
        self._check()

    def snapshot(self):
        self._check(deep=True)
        meta = {"sender": self.sender, "receiver": self.receiver, "slot": self.slot,
                "allowed_names": sorted(self.allowed_names), "lr": self.lr,
                "weight_decay": self.weight_decay, "clip_norm": self.clip_norm,
                "live_hash": self._live_hash, "teacher_hash": self._teacher_hash,
                "student_frozen_hash": self._student_frozen_hash,
                "generator_device": str(self.generator.device),
                "sender_reset_policy": "redraw_A_normal_0_0.02_zero_B"}
        return SleepSnapshot(_owned(self.teacher.state_dict()), _owned(self.student.state_dict()),
                             _owned(self.optimizer.state_dict()), meta, dict(self.counters),
                             self.generator.get_state().clone(), torch.get_rng_state().clone(),
                             [x.cpu().clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None,
                             random.getstate(), deepcopy(np.random.get_state()),
                             {"teacher": _bank_state(self.teacher), "student": _bank_state(self.student)})

    def restore(self, snap):
        if not isinstance(snap, SleepSnapshot) or snap.metadata["live_hash"] != _state_hash(self.live):
            raise ValueError("sleep snapshot live origin mismatch")
        meta = snap.metadata
        if meta.get("sender_reset_policy") != "redraw_A_normal_0_0.02_zero_B":
            raise ValueError("sleep reset policy mismatch")
        self.sender, self.receiver, self.slot = meta["sender"], meta["receiver"], meta["slot"]
        self.allowed_names = set(meta["allowed_names"])
        self.lr, self.weight_decay, self.clip_norm = meta["lr"], meta["weight_decay"], meta["clip_norm"]
        self.teacher.load_state_dict(_owned(snap.teacher_state))
        self.student.load_state_dict(_owned(snap.student_state))
        _restore_banks(self.teacher, snap.bank_states["teacher"])
        _restore_banks(self.student, snap.bank_states["student"])
        if self.allowed_names != _slot_names(self.student, self.receiver, self.slot):
            raise ValueError("snapshot receiver identity mismatch")
        self.optimizer = self._new_optimizer()
        self.optimizer.load_state_dict(deepcopy(snap.optimizer_state))
        # load_state_dict may reuse CPU optimizer tensors. Own every tensor so
        # replay cannot advance Adam's step/moments inside the saved checkpoint.
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.clone()
        self.generator = torch.Generator(device=meta["generator_device"])
        self.generator.set_state(snap.generator_state.cpu().clone())
        self.counters = dict(snap.counters)
        self._live_hash = meta["live_hash"]
        self._set_guards()
        if self._teacher_hash != meta["teacher_hash"] or self._student_frozen_hash != meta["student_frozen_hash"]:
            raise ValueError("snapshot frozen tensor hash mismatch")
        torch.set_rng_state(snap.cpu_rng.cpu().clone())
        if snap.cuda_rng is not None:
            torch.cuda.set_rng_state_all([x.cpu().clone() for x in snap.cuda_rng])
        random.setstate(snap.python_rng)
        np.random.set_state(deepcopy(snap.numpy_rng))
        self._check(deep=True)

    @classmethod
    def from_snapshot(cls, live_context, snap):
        tx = cls.__new__(cls)
        tx.live = live_context
        tx.teacher, tx.student = deepcopy(live_context), deepcopy(live_context)
        tx.restore(snap)
        return tx

    def commit(self):
        if self.counters["closed"]:
            raise RuntimeError("sleep transaction already committed")
        self._check(deep=True)
        self.counters["closed"] = True
        _freeze(self.student)
        return self.student
