"""Compare explicitly supplied parameter snapshots without running a model.

The caller must take an independent baseline clone before the update and select
the same named parameters in both snapshots. This is a numerical diagnostic,
not a measure of learned knowledge or a test of continual learning.
"""
from collections.abc import Mapping
import math

import torch


@torch.no_grad()
def parameter_drift(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]):
    """Return per-parameter L2 changes; reject incomparable or nonfinite inputs.

    Norms use CPU float64 arithmetic, one parameter at a time. Relative change
    is None when the baseline norm is zero, including zero-to-zero transitions.
    This function does not mutate snapshots, model parameters, or gradients.
    """
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise TypeError('snapshots must be mappings')
    if not before or set(before) != set(after):
        raise ValueError('snapshots must have the same nonempty set of names')
    if any(not isinstance(name, str) for name in before):
        raise TypeError('parameter names must be strings')
    rows = []
    for name in sorted(before):
        a, b = before[name], after[name]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError(f'{name}: values must be tensors')
        if a.layout != torch.strided or b.layout != torch.strided:
            raise ValueError(f'{name}: only dense strided tensors are supported')
        if not a.is_floating_point() or not b.is_floating_point():
            raise ValueError(f'{name}: only real floating-point tensors are supported')
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f'{name}: shapes and dtypes must match')
        if a.numel() == 0:
            raise ValueError(f'{name}: empty tensors are unsupported')
        x = a.detach().to(device='cpu', dtype=torch.float64)
        y = b.detach().to(device='cpu', dtype=torch.float64)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f'{name}: tensors must be finite')
        baseline = torch.linalg.vector_norm(x).item()
        current = torch.linalg.vector_norm(y).item()
        delta = torch.linalg.vector_norm(y - x).item()
        if not all(math.isfinite(v) for v in (baseline, current, delta)):
            raise ValueError(f'{name}: norm overflow')
        relative = delta / baseline if baseline else None
        if relative is not None and not math.isfinite(relative):
            raise ValueError(f'{name}: relative norm overflow')
        rows.append({'name': name, 'numel': a.numel(), 'baseline_l2': baseline,
                     'current_l2': current, 'delta_l2': delta,
                     'relative_delta_l2': relative,
                     'changed': not torch.equal(x, y)})
    return rows
