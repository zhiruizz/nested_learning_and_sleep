"""Atomic gradient-accumulation helpers for additive Banking sleep transactions.

Each closure computes one request's scalar objective and returns ``(loss,
token_count)``. The helper backpropagates each mean-scaled loss immediately, so
it never retains the request computation graphs. A microbatch commits once via
one global clip and one AdamW step. This is an engineering extension; it does
not change the frozen v10/v11 transaction implementation.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import sys
from typing import Callable, Sequence

import torch
from torch import Tensor

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from banking_cil_v10_lifecycle import AdditiveSleepTransaction  # noqa: E402


_COUNTER_DEFAULTS = {
    # Requests and tokens successfully committed in microbatches.
    "v12_batch_commits": 0,
    "v12_committed_requests": 0,
    "v12_committed_tokens": 0,
    "v12_physical_adamw_steps": 0,
    # Attempt counters include failed batches and are never interpreted as updates.
    "v12_request_attempts": 0,
    "v12_tokens_observed": 0,
    "v12_backward_calls": 0,
    # Counts optimizer.step invocations even when a post-step guard rolls back.
    "v12_adamw_step_attempts": 0,
    "v12_failed_batches": 0,
    "v12_failed": False,
}


class BatchableAdditiveSleepTransaction(AdditiveSleepTransaction):
    """V10 transaction with a batch-boundary snapshot guard.

    The inherited snapshot format is retained. V12 counters live in the
    snapshot's existing counters dictionary, so restore preserves both the
    physical optimizer-step count and the failed-transaction marker.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counters.update(_COUNTER_DEFAULTS)
        self._v12_batch_active = False

    def _v12_ensure_usable(self):
        if self.counters.get("v12_failed", False):
            raise RuntimeError("microbatch transaction is failed and closed")
        if self.counters.get("closed", False):
            raise RuntimeError("additive sleep transaction is closed")

    def optimize(self, loss, kind, tokens):
        if self._v12_batch_active:
            raise RuntimeError("cannot call single-request optimize inside a microbatch")
        self._v12_ensure_usable()
        return super().optimize(loss, kind, tokens)

    def snapshot(self):
        if self._v12_batch_active:
            raise RuntimeError("snapshot is only allowed at a completed microbatch boundary")
        if any(parameter.grad is not None for parameter in self.student.parameters()):
            raise RuntimeError("snapshot requires cleared gradients at a microbatch boundary")
        if not all(name in self.counters for name in _COUNTER_DEFAULTS):
            raise RuntimeError("v12 counters are missing; refusing an incompatible snapshot")
        return super().snapshot()

    def restore(self, snapshot):
        if getattr(self, "_v12_batch_active", False):
            raise RuntimeError("cannot restore during a microbatch")
        result = super().restore(snapshot)
        if not all(name in self.counters for name in _COUNTER_DEFAULTS):
            raise ValueError("snapshot does not contain v12 microbatch counters")
        self._v12_batch_active = False
        if any(parameter.grad is not None for parameter in self.student.parameters()):
            raise ValueError("restored snapshot contains partial gradients")
        return result

    @classmethod
    def from_snapshot(cls, live_context, snapshot):
        tx = super().from_snapshot(live_context, snapshot)
        tx._v12_batch_active = False
        return tx

    def commit(self):
        if self._v12_batch_active:
            raise RuntimeError("cannot commit during a microbatch")
        if self.counters.get("v12_failed", False):
            raise RuntimeError("cannot commit a failed microbatch transaction")
        return super().commit()


def _receiver_parameters(tx: BatchableAdditiveSleepTransaction):
    parameters = dict(tx.student.named_parameters())
    names = sorted(tx.allowed_names)
    if not names or any(name not in parameters for name in names):
        raise RuntimeError("invalid additive receiver parameter scope")
    return names, [parameters[name] for name in names]


def _parameter_delta_l2(names, parameters, before) -> float:
    squared = 0.0
    with torch.no_grad():
        for name, parameter in zip(names, parameters):
            difference = parameter.detach().to(torch.float64) - before[name].to(torch.float64)
            squared += float(torch.sum(difference * difference).cpu())
    return math.sqrt(squared)


def optimize_request_batch(
    tx: BatchableAdditiveSleepTransaction,
    closures: Sequence[Callable[[], tuple[Tensor, int]]],
    kind: str = "gkd",
) -> dict[str, int | float]:
    """Accumulate a mean of request losses and perform one guarded AdamW step.

    ``closures`` must be a materialized list/tuple of 1..32 no-argument
    callables. Each callable performs a fresh forward pass and returns a finite
    scalar loss plus a positive integer token count. A failure closes the
    transaction, discards all partial gradients, rolls back receiver weights
    and optimizer state, and re-raises the original exception.
    """
    if not isinstance(tx, BatchableAdditiveSleepTransaction):
        raise TypeError("tx must be a BatchableAdditiveSleepTransaction")
    if kind != "gkd":
        raise ValueError("v12 request batching currently supports kind='gkd' only")
    if not isinstance(closures, (list, tuple)) or not 1 <= len(closures) <= 32:
        raise ValueError("closures must be a list/tuple containing 1..32 requests")
    if any(not callable(closure) for closure in closures):
        raise TypeError("every request closure must be callable")
    tx._v12_ensure_usable()
    names, parameters = _receiver_parameters(tx)
    if any(parameter.grad is not None for parameter in tx.student.parameters()):
        raise RuntimeError("microbatch must start with cleared gradients")
    # Match the transaction's ordinary per-step version guard. Deep hashing is
    # reserved for the inherited snapshot/commit boundaries; doing it for each
    # request would copy/hash the entire model repeatedly on device.
    tx._check()

    count = len(closures)
    before_parameters = {name: parameter.detach().clone() for name, parameter in zip(names, parameters)}
    before_optimizer = deepcopy(tx.optimizer.state_dict())
    tx._v12_batch_active = True
    losses: list[float] = []
    token_total = 0
    successful_backwards = 0
    try:
        for closure in closures:
            tx.counters["v12_request_attempts"] += 1
            returned = closure()
            if not isinstance(returned, (tuple, list)) or len(returned) != 2:
                raise TypeError("request closure must return (scalar_loss, token_count)")
            loss, tokens = returned
            if (not isinstance(loss, Tensor) or loss.ndim != 0 or not loss.requires_grad
                    or not bool(torch.isfinite(loss))):
                raise ValueError("request closure returned a nonfinite or nondifferentiable scalar loss")
            if type(tokens) is not int or tokens < 1:
                raise ValueError("request closure token_count must be a positive integer")
            token_total += tokens
            tx.counters["v12_tokens_observed"] += tokens
            loss_value = float(loss.detach().cpu())
            # Division before backward weights each request equally, regardless
            # of the request's target-token count.
            scaled_loss = loss / count
            tx.counters["backward_attempts"] += 1
            tx.counters["tokens"] += tokens
            scaled_loss.backward()
            successful_backwards += 1
            tx.counters["v12_backward_calls"] += 1
            tx._check()
            losses.append(loss_value)
            del scaled_loss, loss, returned

        grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not grads or any(not bool(torch.isfinite(grad).all()) for grad in grads):
            raise RuntimeError("missing or nonfinite accumulated receiver gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, tx.clip_norm, error_if_nonfinite=True)
        grad_norm = float(norm.detach().cpu()) if isinstance(norm, Tensor) else float(norm)
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            tx.counters["zero_gradient_steps"] += 1
            raise RuntimeError("accumulated receiver gradient norm must be positive before AdamW")

        tx.counters["v12_adamw_step_attempts"] += 1
        tx.optimizer.step()
        tx.optimizer.zero_grad(set_to_none=True)
        tx._check()
        update_norm = _parameter_delta_l2(names, parameters, before_parameters)
        if not math.isfinite(update_norm):
            raise RuntimeError("receiver parameter update norm is nonfinite")
        result = {
            "loss_mean": sum(losses) / count,
            "request_count": count,
            "token_count": token_total,
            "backward_calls": successful_backwards,
            "physical_adamw_steps": 1,
            "preclip_grad_norm": grad_norm,
            "parameter_update_l2": update_norm,
        }
        # Publish committed-work counters only after the post-step protected
        # state check and update summary both succeed.
        tx.counters["adamw_steps"] += 1
        tx.counters["gkd_calls"] += count
        tx.counters["v12_batch_commits"] += 1
        tx.counters["v12_committed_requests"] += count
        tx.counters["v12_committed_tokens"] += token_total
        tx.counters["v12_physical_adamw_steps"] += 1
        return result
    except BaseException:
        # No partial batch may be committed or resumed. Restore all receiver
        # values and Adam moments before closing the failed transaction.
        with torch.no_grad():
            for name, parameter in zip(names, parameters):
                parameter.copy_(before_parameters[name])
        tx.optimizer.load_state_dict(before_optimizer)
        for state in tx.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.clone()
        tx.optimizer.zero_grad(set_to_none=True)
        tx.counters["closed"] = True
        tx.counters["v12_failed"] = True
        tx.counters["v12_failed_batches"] += 1
        raise
    finally:
        tx._v12_batch_active = False
