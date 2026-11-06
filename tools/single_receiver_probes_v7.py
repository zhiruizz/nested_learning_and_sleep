"""Small, local probes for receiver lesions and gradient alignment.

Gradient cosine here is a local comparison for the supplied losses and
parameters. It is not universal evidence or a causal proof. A slow-output
lesion preserves routing/gating values but changes module output, so it cannot
prove that knowledge is stored uniquely in any one module.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Callable

import torch


@contextmanager
def slow_output_lesion(model, disabled_slots):
    """Zero selected outputs from ``model.banks()[1].slots`` until exit.

    Only forward outputs are masked. Parameters and routing decisions are
    untouched, and all hooks are removed even if the body raises.
    """
    slots = model.banks()[1].slots
    handles = []
    try:
        for index in disabled_slots:
            slot = slots[index]

            def zero_output(_module, _inputs, output):
                return torch.zeros_like(output)

            handles.append(slot.register_forward_hook(zero_output))
        yield
    finally:
        for handle in handles:
            handle.remove()


def gradient_conflict(model, loss_old_fn: Callable[[], torch.Tensor],
                      loss_new_fn: Callable[[], torch.Tensor],
                      groups: dict[str, list[tuple[str, torch.nn.Parameter]]]):
    """Compare old/new loss gradients over named parameter groups.

    Uses autograd.grad only; it does not write ``.grad``, update parameters,
    sample randomness, or depend on a model-specific cache or API.
    """
    del model  # The interface keeps the call site explicit; losses own closures.
    flat = [(group, name, param) for group, entries in groups.items()
            for name, param in entries]
    params = [entry[2] for entry in flat]
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(True)
        old_loss = loss_old_fn()
        old_grads = torch.autograd.grad(old_loss, params, allow_unused=True, retain_graph=False)
        # Build the independent new objective only once, after the old pass.
        new_loss = loss_new_fn()
        new_grads = torch.autograd.grad(new_loss, params, allow_unused=True, retain_graph=False)

        result = {}
        for group, entries in groups.items():
            idxs = [i for i, item in enumerate(flat) if item[0] == group]
            pairs = [(old_grads[i], new_grads[i]) for i in idxs]
            old_sq = sum(float(g.detach().double().pow(2).sum()) for g, _ in pairs if g is not None)
            new_sq = sum(float(g.detach().double().pow(2).sum()) for _, g in pairs if g is not None)
            dot = sum(float((g.detach().double() * h.detach().double()).sum())
                      for g, h in pairs if g is not None and h is not None)
            old_norm, new_norm = math.sqrt(old_sq), math.sqrt(new_sq)
            cosine = dot / (old_norm * new_norm) if old_norm and new_norm else None
            result[group] = {"old_norm": old_norm, "new_norm": new_norm,
                             "dot": dot, "cosine": cosine,
                             "numel": sum(params[i].numel() for i in idxs)}
        return result
    finally:
        for param, flag in zip(params, flags):
            param.requires_grad_(flag)


__all__ = ["slow_output_lesion", "gradient_conflict"]
