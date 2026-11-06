"""Capture exact final-block prefix tails and evaluate cached tails.

Only the final decoder layer is instrumented. The two cached positions are the
prompt-end and following answer position (`lengths-1`, `lengths`), matching the
two-token AG News objective. This helper never registers modules or changes
model weights; temporary hooks and CMS audit buffers are restored on all paths.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _tail_modules(model):
    try:
        layer = model.backbone.model.layers[-1]
        norm = layer.post_attention_layernorm
        original_mlp = layer.mlp.original
        final_norm = model.backbone.model.norm
        head = model.backbone.lm_head
    except (AttributeError, IndexError, TypeError) as exc:
        raise TypeError("model must expose a LlamaCMSLM final decoder layer and wrapped MLP") from exc
    if not callable(norm) or not callable(original_mlp) or not callable(final_norm) or not callable(head):
        raise TypeError("tail model components must be callable modules")
    if not hasattr(layer.mlp, "memory"):
        raise TypeError("final MLP wrapper has no mutable memory levels")
    return layer, norm, original_mlp, final_norm, head


def _two_positions(value: Tensor, lengths: Tensor, name: str) -> Tensor:
    if value.ndim != 3 or value.shape[0] != lengths.numel():
        raise ValueError(f"{name} must have shape [batch, sequence, feature]")
    rows = torch.arange(lengths.numel(), device=value.device)
    positions = lengths.to(value.device)
    return torch.stack((value[rows, positions - 1], value[rows, positions]), dim=1)


def capture(model, ids: Tensor, mask: Optional[Tensor], lengths: Tensor, *, device=None):
    """Run one ordinary full forward and detach-copy the two-position tail.

    `device=None` keeps copies on the model/input device; pass `"cpu"` to move
    the cache off-device. The returned mapping contains `y` (original final
    MLP output), `residual` (input to post-attention layer norm), and full-model
    `logits` at the two selected positions.
    """
    if not isinstance(ids, Tensor) or ids.ndim != 2:
        raise ValueError("ids must be a rank-2 [batch, sequence] tensor")
    if mask is not None and (not isinstance(mask, Tensor) or mask.shape != ids.shape):
        raise ValueError("mask must match ids shape")
    raw_lengths = torch.as_tensor(lengths)
    if raw_lengths.ndim != 1 or raw_lengths.numel() != ids.shape[0]:
        raise ValueError("lengths must have one entry per batch row")
    if raw_lengths.is_floating_point() and not bool(torch.equal(raw_lengths, raw_lengths.round())):
        raise ValueError("lengths must contain integer positions")
    lens = raw_lengths.to(dtype=torch.long)
    if bool((lens < 1).any()) or bool((lens >= ids.shape[1]).any()):
        raise ValueError("each length must satisfy 1 <= length < sequence length")
    if mask is not None and not bool(mask[torch.arange(ids.shape[0], device=mask.device), lens.to(mask.device)-1].bool().all()):
        raise ValueError("the final prompt position must be unmasked")

    layer, post_norm, original_mlp, _, _ = _tail_modules(model)
    captured = {"residual": [], "y": []}

    def before_post_norm(_module, inputs):
        if not inputs or not isinstance(inputs[0], Tensor):
            raise TypeError("post_attention_layernorm pre-hook did not receive a tensor")
        captured["residual"].append(inputs[0])

    def after_original_mlp(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, Tensor):
            raise TypeError("original MLP hook did not receive a tensor output")
        captured["y"].append(value)

    handles = []
    audit = []
    for bank in model.banks():
        had_history=hasattr(bank,"audit_history")
        audit.append((bank,had_history,getattr(bank,"audit_history",None)))
    try:
        handles.append(post_norm.register_forward_pre_hook(before_post_norm))
        handles.append(original_mlp.register_forward_hook(after_original_mlp))
        with torch.no_grad():
            logits = model.logits(ids, mask)
        if len(captured["residual"]) != 1 or len(captured["y"]) != 1:
            raise RuntimeError("expected exactly one final-layer residual and original-MLP capture")
        selected = {
            "residual": _two_positions(captured["residual"][0], lens, "residual"),
            "y": _two_positions(captured["y"][0], lens, "y"),
            "logits": _two_positions(logits, lens, "logits"),
        }
        target_device = torch.device(device) if device is not None else ids.device
        return {key: value.detach().to(device=target_device).clone() for key, value in selected.items()}
    finally:
        for handle in reversed(handles):
            handle.remove()
        for bank, had_history, history in audit:
            if had_history:
                bank.audit_history = history
            elif hasattr(bank,"audit_history"):
                delattr(bank,"audit_history")


def tail_logits(model, y: Tensor, residual: Tensor) -> Tensor:
    """Recompute the final two-position logits from frozen-prefix caches."""
    if not isinstance(y, Tensor) or not isinstance(residual, Tensor):
        raise TypeError("y and residual must be tensors")
    if y.ndim != 3 or residual.shape != y.shape:
        raise ValueError("y and residual must have matching [batch, 2, hidden] shapes")
    if y.shape[1] != 2:
        raise ValueError("tail cache must contain exactly two sequence positions")
    layer, _, _, final_norm, head = _tail_modules(model)
    hidden = y
    for memory in layer.mlp.memory:
        hidden = memory(hidden)
    hidden = final_norm(residual + hidden)
    return head(hidden)


def tail_logits_padded(model, y: Tensor, residual: Tensor, lengths: Tensor, total_length: int) -> Tensor:
    """Run cached positions in the original full sequence matrix shape.

    The two cached vectors are scattered into zero-filled ``[B,total_length,H]``
    tensors at positions ``lengths-1`` and ``lengths``. The final memory stack,
    residual add, final norm and readout therefore see the same flattened M
    dimension as the original full-sequence forward. Only those two positions
    are gathered for the caller's loss.
    """
    if not isinstance(y, Tensor) or not isinstance(residual, Tensor):
        raise TypeError("y and residual must be tensors")
    if y.ndim != 3 or residual.shape != y.shape or y.shape[1] != 2:
        raise ValueError("y and residual must have matching [batch, 2, hidden] shapes")
    if not isinstance(total_length, int) or total_length < 2:
        raise ValueError("total_length must be an integer >= 2")
    raw_lengths = torch.as_tensor(lengths)
    if raw_lengths.ndim != 1 or raw_lengths.numel() != y.shape[0]:
        raise ValueError("lengths must have one entry per cached batch row")
    if raw_lengths.is_floating_point() and not bool(torch.equal(raw_lengths, raw_lengths.round())):
        raise ValueError("lengths must contain integer positions")
    lens = raw_lengths.to(device=y.device, dtype=torch.long)
    if bool((lens < 1).any()) or bool((lens >= total_length).any()):
        raise ValueError("each length must satisfy 1 <= length < total_length")
    layer, _, _, final_norm, head = _tail_modules(model)
    batch, _, hidden_dim = y.shape
    rows = torch.arange(batch, device=y.device)
    positions = torch.stack((lens - 1, lens), dim=1)
    padded_y = y.new_zeros((batch, total_length, hidden_dim))
    padded_residual = residual.new_zeros((batch, total_length, hidden_dim))
    padded_y[rows[:, None], positions] = y
    padded_residual[rows[:, None], positions] = residual
    hidden = padded_y
    for memory in layer.mlp.memory:
        hidden = memory(hidden)
    hidden = final_norm(padded_residual + hidden)
    logits = head(hidden)
    return logits[rows[:, None], positions]
