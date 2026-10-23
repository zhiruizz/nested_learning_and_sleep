"""NL Eq 93 matrix special case; NOT the deep Eq 89/91 experimental model.

Choices: column vectors, 0.5*sum squared loss, fixed query per prose, scalar
gates from mean+sigmoid, read before write, all targets generated from old state.
No momentum, convolution, chunk-parallel optimization, or CMS meta-learning.
"""
import torch
from torch import nn
from torch.nn import functional as F


def dgd_linear_step(matrix, key, target, eta, alpha, *, anchor=None):
    """Eq 93 with fixed data for the internal derivative, differentiable outside.

    matrix [out,in], key [in], target [out]; anchor is a chunk-start state.
    eta/alpha are scalars. Neither the caller state nor its gradient is mutated.
    """
    if matrix.ndim != 2 or key.ndim != 1 or target.ndim != 1:
        raise ValueError('Expected matrix, key vector, target vector')
    if matrix.shape != (target.numel(), key.numel()):
        raise ValueError('incompatible key/target/matrix dimensions')
    anchor = matrix if anchor is None else anchor
    if anchor.shape != matrix.shape:
        raise ValueError('anchor shape must equal matrix shape')
    eta = torch.as_tensor(eta, dtype=matrix.dtype, device=matrix.device)
    alpha = torch.as_tensor(alpha, dtype=matrix.dtype, device=matrix.device)
    if eta.numel() != 1 or alpha.numel() != 1:
        raise ValueError('This reference requires scalar eta/alpha')
    retention = alpha * torch.eye(key.numel(), dtype=matrix.dtype, device=matrix.device) - eta * torch.outer(key, key)
    gradient = torch.outer(anchor @ key - target, key)
    return matrix @ retention - eta * gradient


class LinearSelfModifyingReference(nn.Module):
    """A serial matrix-only mechanism reference, using externally owned state."""
    boxes = ('k', 'v', 'eta', 'alpha', 'memory')

    def __init__(self, dim, *, eta_scale=0.05, stop_target_gradient=True):
        super().__init__()
        if dim < 1 or eta_scale <= 0:
            raise ValueError('dim and eta_scale must be positive')
        self.dim = dim
        self.eta_scale = eta_scale
        self.stop_target_gradient = stop_target_gradient
        self.initial = nn.ParameterDict({
            name: nn.Parameter(torch.eye(dim) + 0.02 * torch.randn(dim, dim)) for name in self.boxes
        })
        self.query = nn.Parameter(torch.eye(dim))

    def initial_state(self):
        # clone keeps the outer gradient path, while avoiding storage aliases.
        return {name: p.clone() for name, p in self.initial.items()}

    def forward(self, x, state=None):
        if x.shape != (self.dim,):
            raise ValueError('one token vector per explicit stream is required')
        state = self.initial_state() if state is None else state
        if set(state) != set(self.boxes):
            raise ValueError('state must contain exactly the reference memory boxes')
        k = F.normalize(state['k'] @ x, dim=0)
        v = state['v'] @ x
        q = F.normalize(self.query @ x, dim=0)
        eta = self.eta_scale * torch.sigmoid((state['eta'] @ x).mean())
        alpha = torch.sigmoid((state['alpha'] @ x).mean())
        output = state['memory'] @ q
        targets = {name: matrix @ v for name, matrix in state.items()}
        if self.stop_target_gradient:
            targets = {name: value.detach() for name, value in targets.items()}
        new_state = {name: dgd_linear_step(matrix, k, targets[name], eta, alpha)
                     for name, matrix in state.items()}
        return output, new_state, {'key': k, 'value': v, 'targets': targets, 'eta': eta, 'alpha': alpha}
