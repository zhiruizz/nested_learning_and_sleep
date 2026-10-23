"""Sleep compute/consolidate/update transaction on a tiny linear chain.

The chain is an additive low-rank reference, NOT the paper's sparse MoE router.
No automatic training, real language model, or complete KS/Dreaming loop here.
"""
from copy import deepcopy
import torch
from torch import nn


class LowRankSlot(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.a = nn.Parameter(torch.randn(dim, rank) * 0.02, requires_grad=False)
        self.b = nn.Parameter(torch.zeros(rank, dim), requires_grad=False)
        self.register_buffer('active', torch.tensor(False))

    def forward(self, x):
        if not bool(self.active):
            return torch.zeros_like(x)  # dormant parameters never enter the graph
        return x @ self.a @ self.b

    def reset(self):
        # Deterministic reset for this reference: reuse A, clear B and deactivate.
        # Reinitialization of A/router is a later configurable sparse-expert policy.
        with torch.no_grad():
            self.b.zero_()
            self.active.fill_(False)
        self.requires_grad_(False)
        for p in self.parameters():
            p.grad = None


class LinearExpertChain(nn.Module):
    def __init__(self, dim=4, levels=2):
        super().__init__()
        self.dim = dim
        self.base = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(levels)])
        self.slots = nn.ModuleList([nn.ModuleList() for _ in range(levels)])

    def add_slot(self, level, rank, *, active=False):
        if rank < 1 or rank > self.dim:
            raise ValueError('rank outside reference capacity')
        like = self.base[level].weight
        slot = LowRankSlot(self.dim, rank).to(device=like.device, dtype=like.dtype)
        slot.active.fill_(active)
        self.slots[level].append(slot)
        return slot

    def forward(self, x):
        for base, slots in zip(self.base, self.slots):
            old = x
            x = base(old)
            for slot in slots:
                x = x + slot(old)
        return x

    def topology(self):
        return {'dim': self.dim, 'levels': len(self.base),
                'ranks': [[s.a.shape[1] for s in slots] for slots in self.slots]}

    @classmethod
    def from_topology(cls, topo):
        model = cls(topo['dim'], topo['levels'])
        for level, ranks in enumerate(topo['ranks']):
            for rank in ranks:
                model.add_slot(level, rank)
        return model


def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def same_snapshot(model, expected):
    actual = model.state_dict()
    return actual.keys() == expected.keys() and all(torch.equal(actual[k], v) for k, v in expected.items())


class ConsolidationTransaction:
    """Candidate is private; commit returns a new model, never mutates live.

    The caller must consume sender accumulators only after successful commit.
    This reference is synchronous and has no durable in-flight journal yet.
    """
    def __init__(self, live, sender, prospective_weight, *, rank=2):
        if not 0 <= sender < len(live.base) - 1:
            raise ValueError('requires an adjacent slower receiver')
        if prospective_weight.shape != live.base[sender].weight.shape:
            raise ValueError('prospective sender shape mismatch')
        if not torch.isfinite(prospective_weight).all():
            raise ValueError('prospective sender contains nonfinite values')
        self.live = live
        self.before = snapshot(live)
        self.teacher = deepcopy(live).eval().requires_grad_(False)
        self.student = deepcopy(live).eval().requires_grad_(False)
        with torch.no_grad():
            self.student.base[sender].weight.copy_(prospective_weight)
        for slot in self.student.slots[sender]:
            slot.reset()
        self.new_slot = self.student.add_slot(sender + 1, rank, active=True)
        self.new_slot.requires_grad_(True)
        self.expected_frozen = snapshot(self.student)
        self.mutable = {f'slots.{sender + 1}.{len(self.student.slots[sender + 1]) - 1}.{n}'
                        for n, _ in self.new_slot.named_parameters()}
        self.closed = False

    def trainable_parameters(self):
        return list(self.new_slot.parameters())

    def commit(self):
        if self.closed:
            raise RuntimeError('transaction already closed')
        if not same_snapshot(self.live, self.before) or not same_snapshot(self.teacher, self.before):
            raise RuntimeError('live or teacher changed during consolidation')
        for name, value in self.student.state_dict().items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise RuntimeError('nonfinite candidate')
            if name not in self.mutable and not torch.equal(value, self.expected_frozen[name]):
                raise RuntimeError('frozen candidate state changed: ' + name)
        # Return an independent value so later edits cannot mutate transaction evidence.
        result = deepcopy(self.student).requires_grad_(False)
        for p in result.parameters():
            p.grad = None
        self.closed = True
        return result

    def abort(self):
        if self.closed:
            raise RuntimeError('transaction already closed')
        self.closed = True
