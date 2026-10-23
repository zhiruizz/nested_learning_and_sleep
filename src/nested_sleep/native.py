"""Attention-free deep HOPE research variant; see docs/paper_to_code.md.

Deep parameter-wise DGD is an explicit interpretation of NL88/89, NOT a
claim that the paper uniquely specifies our feature-wise retention rule.
All fast weights, anchors, counters and convolution caches are external state.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


def tree_map(fn, value):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {k: tree_map(fn, v) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_map(fn, v) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_map(fn, v) for v in value)
    return value


def deep_map(x, weights):
    wi, wo = weights
    hidden = torch.einsum('bhd,bd->bh', wi, x)
    return x + torch.einsum('bdh,bh->bd', wo, F.silu(hidden))


def deep_partials(weights, key, target):
    """Exact partial derivatives of half-squared L2, treating k/target as data.

    Algebra remains differentiable for outer gradients; no finite differences
    or in-place .data writes in the production implementation.
    """
    wi, wo = weights
    u = torch.einsum('bhd,bd->bh', wi, key)
    sig = torch.sigmoid(u)
    hidden = u * sig
    error = key + torch.einsum('bdh,bh->bd', wo, hidden) - target
    go = torch.einsum('bd,bh->bdh', error, hidden)
    delta = torch.einsum('bdh,bd->bh', wo, error) * (sig + u * sig * (1 - sig))
    gi = torch.einsum('bh,bd->bhd', delta, key)
    return (gi, go), (key, hidden)


def feature_dgd_step(running, anchor, key, target, eta, alpha):
    """A3: per-layer feature retention; gradients at chunk-start anchor."""
    gradients, features = deep_partials(anchor, key, target)
    result = []
    for w, g, feature in zip(running, gradients, features):
        # w @ (alpha I - eta phi phi^T), without materializing dense I.
        remembered = torch.einsum('boi,bi->bo', w, feature)
        decay = torch.einsum('bo,bi->boi', remembered, feature)
        result.append(alpha[:, None, None] * w - eta[:, None, None] * (decay + g))
    return tuple(result)


class DeepMap(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.wi = nn.Parameter(torch.randn(hidden, dim) * (0.1 / dim ** 0.5))
        self.wo = nn.Parameter(torch.randn(dim, hidden) * (0.1 / hidden ** 0.5))

    def state(self, batch):
        return tuple(p.unsqueeze(0).expand(batch, *p.shape).clone() for p in (self.wi, self.wo))


@dataclass
class NativeConfig:
    vocab_size: int = 260
    dim: int = 16
    hidden: int = 24
    layers: int = 1
    projection_chunk: int = 4
    memory_chunk: int = 1
    cms_periods: tuple = (1, 4)
    cms_lr: float = 0.01
    eta_max: float = 0.002
    stop_target_gradient: bool = False
    architecture: str = 'native_hope'
    reproduction: str = 'assumption_explicit_deep_feature_dgd'


class DeepSelfModifyingTitans(nn.Module):
    boxes = ('k', 'v', 'eta', 'alpha', 'memory')

    def __init__(self, cfg):
        super().__init__()
        if min(cfg.projection_chunk, cfg.memory_chunk) < 1:
            raise ValueError('positive chunk lengths required')
        self.cfg = cfg
        self.maps = nn.ModuleDict({name: DeepMap(cfg.dim, cfg.hidden) for name in self.boxes})
        self.query = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.conv = nn.Parameter(torch.randn(cfg.dim, 4) * 0.02)

    def initial_state(self, batch):
        w = {name: module.state(batch) for name, module in self.maps.items()}
        return {'weights': w, 'anchors': {k: tuple(t.clone() for t in v) for k, v in w.items()},
                'count': torch.zeros(batch, dtype=torch.long, device=self.conv.device),
                'conv': self.conv.new_zeros(batch, 3, self.cfg.dim)}

    def step(self, x, state, valid=None):
        batch = x.shape[0]
        valid = torch.ones(batch, dtype=torch.bool, device=x.device) if valid is None else valid.bool()
        history = torch.cat([state['conv'], x[:, None]], dim=1)
        mixed = x + (history.transpose(1, 2) * self.conv[None]).sum(-1)
        anchors = state['anchors']
        k = F.normalize(deep_map(mixed, anchors['k']), dim=-1)
        v = deep_map(mixed, anchors['v'])
        q = F.normalize(self.query(mixed), dim=-1)
        eta = self.cfg.eta_max * torch.sigmoid(deep_map(mixed, anchors['eta']).mean(-1))
        alpha = torch.exp(-0.001 * F.softplus(deep_map(mixed, anchors['alpha']).mean(-1)))
        output = deep_map(q, state['weights']['memory'])
        targets = {name: deep_map(v, anchors[name]) for name in self.boxes}
        if self.cfg.stop_target_gradient:
            targets = {k: v.detach() for k, v in targets.items()}
        count = state['count'] + valid.long()
        new_w, new_anchors = {}, {}
        for name in self.boxes:
            update = feature_dgd_step(state['weights'][name], anchors[name], k, targets[name], eta, alpha)
            new_w[name] = tuple(torch.where(valid[:, None, None], new, old)
                                for new, old in zip(update, state['weights'][name]))
            period = self.cfg.memory_chunk if name == 'memory' else self.cfg.projection_chunk
            due = valid & (count.remainder(period) == 0)
            new_anchors[name] = tuple(torch.where(due[:, None, None], new, old)
                                      for new, old in zip(new_w[name], anchors[name]))
        return output, {'weights': new_w, 'anchors': new_anchors, 'count': count,
                        'conv': torch.where(valid[:, None, None], history[:, -3:], state['conv'])}


class FunctionalCMS(nn.Module):
    """NL70/71/73 episodic sequential CMS with meta-learned initial states."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.levels = nn.ModuleList([DeepMap(cfg.dim, cfg.hidden) for _ in cfg.cms_periods])

    def initial_state(self, batch):
        weights = [module.state(batch) for module in self.levels]
        return {'weights': weights, 'pending': [tuple(torch.zeros_like(p) for p in w) for w in weights],
                'events': [0] * len(weights)}

    def forward(self, x, state):
        for index, weights in enumerate(state['weights']):
            old = x
            x = deep_map(x, weights)
            if hasattr(self, 'expert_banks'):
                x = x + self.expert_banks[index](old)
        return x

    def adapt(self, loss, state, *, higher_order=True):
        params = [p for w in state['weights'] for p in w]
        if any(not p.requires_grad for p in params):
            raise RuntimeError('CMS adaptation requires a differentiable fast state')
        grads = torch.autograd.grad(loss, params, create_graph=higher_order, retain_graph=True)
        new = {'weights': [], 'pending': [], 'events': []}
        for i, (weights, pending, count, period) in enumerate(zip(
                state['weights'], state['pending'], state['events'], self.cfg.cms_periods)):
            acc = tuple(a + self.cfg.cms_lr * g for a, g in zip(pending, grads[2*i:2*i+2]))
            due = count + 1 == period
            new['weights'].append(tuple(w - a for w, a in zip(weights, acc)) if due else weights)
            new['pending'].append(tuple(torch.zeros_like(a) for a in acc) if due else acc)
            new['events'].append(0 if due else count + 1)
        return new


class NativeBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.smt = DeepSelfModifyingTitans(cfg)
        self.cms = FunctionalCMS(cfg)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)


class NativeHOPELM(nn.Module):
    """Full causal LM interface for the disclosed native architecture variant."""
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or NativeConfig()
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.dim)
        self.blocks = nn.ModuleList([NativeBlock(self.cfg) for _ in range(self.cfg.layers)])
        self.norm = nn.LayerNorm(self.cfg.dim)
        self.head = nn.Linear(self.cfg.dim, self.cfg.vocab_size, bias=False)

    def initial_state(self, batch, stream_ids=None):
        return {'blocks': [{'smt': b.smt.initial_state(batch), 'cms': b.cms.initial_state(batch)} for b in self.blocks],
                'stream_ids': tuple(range(batch)) if stream_ids is None else tuple(stream_ids)}

    def forward(self, tokens, state=None, *, valid=None, stream_ids=None):
        batch, length = tokens.shape
        state = self.initial_state(batch, stream_ids) if state is None else state
        if stream_ids is not None and tuple(stream_ids) != state['stream_ids']:
            raise ValueError('stream identity changed; explicitly reindex/reset the state')
        if state['blocks'][0]['smt']['count'].shape[0] != batch:
            raise ValueError('state batch does not match inputs')
        blocks = state['blocks']
        logits = []
        for t in range(length):
            x = self.embedding(tokens[:, t])
            updated = []
            for module, old in zip(self.blocks, blocks):
                o, smt = module.smt.step(module.norm1(x), old['smt'], None if valid is None else valid[:, t])
                x = x + o
                x = module.cms(module.norm2(x), old['cms'])
                updated.append({'smt': smt, 'cms': old['cms']})
            blocks = updated
            logits.append(self.head(self.norm(x)))
        return torch.stack(logits, dim=1), {'blocks': blocks, 'stream_ids': state['stream_ids']}

    def adapt_cms(self, loss, state, *, higher_order=True):
        # Independent blocks must share one autograd graph; retain_graph in adapt
        # preserves it until the caller's outer loss backward.
        blocks = [{'smt': s['smt'], 'cms': b.cms.adapt(loss, s['cms'], higher_order=higher_order)}
                  for b, s in zip(self.blocks, state['blocks'])]
        return {'blocks': blocks, 'stream_ids': state['stream_ids']}

    def config_dict(self):
        return asdict(self.cfg)
