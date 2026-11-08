"""Mixed-objective gradient accumulation for the isolated Banking V13 study.

Requests remain individually sampled and individually backpropagated. Their
scalar objectives are averaged over requests, followed by one global clip and
one AdamW step per microbatch. This is not numerically equivalent to V10's
per-request Adam updates when microbatch_size > 1.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor
from banking_cil_v10_lifecycle import AdditiveSleepTransaction

KINDS = ("gkd", "lti")
_COUNTER_DEFAULTS = {
    "v13_objective_opportunities": 0,
    "v13_gkd_opportunities": 0,
    "v13_lti_opportunities": 0,
    "v13_zero_objective_opportunities": 0,
    "v13_backward_calls": 0,
    "v13_microbatch_attempts": 0,
    "v13_microbatch_commits": 0,
    "v13_physical_adamw_steps": 0,
    "v13_skipped_zero_gradient_batches": 0,
    "v13_failed_batches": 0,
    "v13_failed": False,
}


def validate_request_schedule(kinds: Sequence[str], *, complete_cycles: bool = True) -> dict[str, int]:
    """Validate the locked GKD,GKD,LTI request order and return opportunities.

    A short terminal suffix is allowed only when ``complete_cycles`` is false;
    V13's 960 and 3200 request budgets are both complete cycles.
    """
    if not isinstance(kinds, (list, tuple)) or not kinds:
        raise ValueError("request schedule must be a nonempty list or tuple")
    values = list(kinds)
    if any(kind not in KINDS for kind in values):
        raise ValueError("request schedule contains a non-GKD/LTI objective")
    expected = ("gkd", "gkd", "lti")
    if any(kind != expected[i % 3] for i, kind in enumerate(values)):
        raise ValueError("request order must repeat GKD,GKD,LTI")
    if complete_cycles and len(values) % 3:
        raise ValueError("request schedule must contain complete GKD,GKD,LTI cycles")
    return {"opportunities": len(values), "gkd": values.count("gkd"), "lti": values.count("lti")}


def group_request_indices(kinds: Sequence[str], microbatch_size: int) -> list[list[int]]:
    """Partition in original order; the last microbatch may be short."""
    validate_request_schedule(kinds, complete_cycles=False)
    if type(microbatch_size) is not int or not 1 <= microbatch_size <= 32:
        raise ValueError("microbatch_size must be an integer in [1,32]")
    return [list(range(i, min(i + microbatch_size, len(kinds))))
            for i in range(0, len(kinds), microbatch_size)]


def _parameters(tx):
    mapping = dict(tx.student.named_parameters())
    names = sorted(tx.allowed_names)
    if not names or any(name not in mapping for name in names):
        raise RuntimeError("invalid trainable receiver scope")
    return names, [mapping[name] for name in names]


def _restore_optimizer(optimizer, state):
    optimizer.load_state_dict(state)
    for item in optimizer.state.values():
        for key, value in item.items():
            if isinstance(value, Tensor):
                item[key] = value.clone()


class BatchableAdditiveSleepTransactionV13(AdditiveSleepTransaction):
    """V10 transaction which permits snapshots only between V13 batches."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in _COUNTER_DEFAULTS.items():
            self.counters.setdefault(key, value)
        self._v13_batch_active = False

    def snapshot(self):
        if self._v13_batch_active:
            raise RuntimeError("snapshot is allowed only between V13 microbatches")
        if any(p.grad is not None for p in self.student.parameters()):
            raise RuntimeError("snapshot requires no partial microbatch gradients")
        return super().snapshot()

    def restore(self, snapshot):
        if getattr(self, "_v13_batch_active", False):
            raise RuntimeError("cannot restore during a V13 microbatch")
        result = super().restore(snapshot)
        for key, value in _COUNTER_DEFAULTS.items():
            self.counters.setdefault(key, value)
        self._v13_batch_active = False
        if any(p.grad is not None for p in self.student.parameters()):
            raise ValueError("restored V13 state contains partial gradients")
        return result

    def commit(self):
        if getattr(self, "_v13_batch_active", False):
            raise RuntimeError("cannot commit during a V13 microbatch")
        if self.counters.get("v13_failed", False):
            raise RuntimeError("cannot commit a failed V13 transaction")
        return super().commit()


def optimize_mixed_request_batch(
    tx,
    requests: Sequence[Callable[[], Mapping[str, Any]]],
    *,
    budget=None,
) -> dict[str, Any]:
    """Evaluate ordered GKD/LTI request callbacks, then commit one AdamW step.

    Each callback returns ``{kind, loss, token_count, zero_objective?, metadata?}``.
    The callback owns sampling/forward/objective construction and must not call
    backward, clip, or optimizer.step. ``loss`` is already token-normalized by
    the objective helper; this function averages those scalar losses by request
    count, including exact-zero objectives. Only a batch made entirely of
    explicitly flagged exact-zero LTI objectives skips AdamW without a fake
    update. Any other zero accumulated gradient is an unexpected cancellation
    and fails closed.
    """
    if not isinstance(requests, (list, tuple)) or not 1 <= len(requests) <= 32:
        raise ValueError("requests must be a list/tuple containing 1..32 callbacks")
    if any(not callable(x) for x in requests):
        raise TypeError("request callbacks must be callable")
    # Validate objective order before any model forward or RNG consumption.
    kinds = getattr(requests, "kinds", None)
    if kinds is not None:
        validate_request_schedule(kinds, complete_cycles=False)
    if tx.counters.get("closed", False) or tx.counters.get("v13_failed", False):
        raise RuntimeError("transaction is closed")
    if getattr(tx, "_v13_batch_active", False):
        raise RuntimeError("nested V13 microbatches are forbidden")
    names, params = _parameters(tx)
    if any(p.grad is not None for p in tx.student.parameters()):
        raise RuntimeError("microbatch must begin with cleared gradients")
    tx._check()
    for key, value in _COUNTER_DEFAULTS.items():
        tx.counters.setdefault(key, value)
    before = {n: p.detach().clone() for n, p in zip(names, params)}
    optimizer_before = deepcopy(tx.optimizer.state_dict())
    tx.counters["v13_microbatch_attempts"] += 1
    losses, token_total, counts = [], 0, {"gkd": 0, "lti": 0}
    zero_count = 0
    backward_calls = 0
    tx._v13_batch_active = True
    try:
        for request in requests:
            result = request()
            # Check protected state immediately after user-supplied sampling /
            # objective code, including the exact-zero path.
            tx._check()
            if not isinstance(result, Mapping):
                raise TypeError("request callback must return a mapping")
            kind = result.get("kind")
            if kind not in KINDS:
                raise ValueError("request result kind must be gkd or lti")
            loss = result.get("loss")
            tokens = result.get("token_count")
            zero = result.get("zero_objective", False)
            if type(zero) is not bool:
                raise TypeError("zero_objective must be bool")
            if type(tokens) is not int or tokens < 1:
                raise ValueError("token_count must be a positive integer")
            if not isinstance(loss, Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("loss must be a finite scalar tensor")
            if not zero and not loss.requires_grad:
                raise ValueError("unflagged objective must be differentiable")
            loss_value = float(loss.detach().float().cpu())
            if zero and (kind != "lti" or loss_value != 0.0):
                raise ValueError("zero_objective is only an exact-zero LTI objective")
            counts[kind] += 1
            tx.counters["v13_objective_opportunities"] += 1
            tx.counters[f"v13_{kind}_opportunities"] += 1
            tx.counters["v13_tokens_observed"] = tx.counters.get("v13_tokens_observed", 0) + tokens
            token_total += tokens
            losses.append(loss_value)
            if zero:
                tx.counters["v13_zero_objective_opportunities"] += 1
                zero_count += 1
            else:
                (loss / len(requests)).backward()
                backward_calls += 1
                tx.counters["v13_backward_calls"] += 1
                tx.counters["backward_attempts"] += 1
                tx.counters["tokens"] += tokens
            tx._check()
            if budget is not None and callable(getattr(budget, "backward", None)) and not zero:
                budget.backward(torch.ones((1, tokens), dtype=torch.long, device=loss.device))

        grads = [p.grad for p in params if p.grad is not None]
        grad_norm = 0.0
        if grads:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, tx.clip_norm,
                                                             error_if_nonfinite=True).detach().cpu())
        if not math.isfinite(grad_norm):
            raise RuntimeError("nonfinite accumulated gradient norm")
        if backward_calls == 0 and zero_count == len(requests):
            tx.optimizer.zero_grad(set_to_none=True)
            tx._check()
            tx.counters["v13_skipped_zero_gradient_batches"] += 1
            return {"request_count": len(requests), "gkd_opportunities": counts["gkd"],
                    "lti_opportunities": counts["lti"], "backward_calls": backward_calls,
                    "token_count": token_total, "physical_adamw_steps": 0,
                    "skipped_zero_gradient": True, "loss_mean": 0.0, "preclip_grad_norm": 0.0,
                    "parameter_update_l2": 0.0}
        if grad_norm == 0.0:
            raise RuntimeError("nonzero-objective batch accumulated an exactly zero gradient")
        tx.counters["v13_optimizer_step_attempts"] = tx.counters.get("v13_optimizer_step_attempts", 0) + 1
        # Reserve the hard update budget before executing AdamW. ``update``
        # counts attempts; successful physical optimizer calls are recorded
        # separately below.
        if budget is not None and callable(getattr(budget, "update", None)):
            budget.update()
        tx.optimizer.step()
        tx.counters["adamw_steps"] += 1
        tx.counters["v13_physical_adamw_steps"] += 1
        tx.optimizer.zero_grad(set_to_none=True)
        tx._check()
        if budget is not None and hasattr(budget, "adamw_updates"):
            budget.adamw_updates += 1
        delta = math.sqrt(sum(float(torch.sum((p.detach().double() - before[n].double()) ** 2).cpu())
                              for n, p in zip(names, params)))
        if not math.isfinite(delta):
            raise RuntimeError("nonfinite receiver update norm")
        tx.counters["gkd_calls"] += counts["gkd"]
        tx.counters["lti_calls"] += counts["lti"]
        tx.counters["v13_microbatch_commits"] += 1
        return {"request_count": len(requests), "gkd_opportunities": counts["gkd"],
                "lti_opportunities": counts["lti"], "backward_calls": backward_calls,
                "token_count": token_total, "physical_adamw_steps": 1,
                "skipped_zero_gradient": False,
                "loss_mean": sum(losses) / len(requests),
                "preclip_grad_norm": grad_norm, "parameter_update_l2": delta}
    except BaseException:
        with torch.no_grad():
            for name, parameter in zip(names, params):
                parameter.copy_(before[name])
        _restore_optimizer(tx.optimizer, optimizer_before)
        tx.optimizer.zero_grad(set_to_none=True)
        tx.counters["closed"] = True
        tx.counters["v13_failed"] = True
        tx.counters["v13_failed_batches"] += 1
        raise
    finally:
        tx._v13_batch_active = False
