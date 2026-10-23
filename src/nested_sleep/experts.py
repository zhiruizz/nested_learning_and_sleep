"""Independent sparse low-rank slots with serializable lifecycle and routing.

Priority bias for new slots and active-expert random exploration are explicit
implementation choices, not uniquely specified Sleep paper settings.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ExpertSlot(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.a = nn.Parameter(torch.randn(rank, dim) * .02)
        self.b = nn.Parameter(torch.zeros(dim, rank))
        self.register_buffer('route', torch.randn(dim) * .02)
        self.register_buffer('priority', torch.tensor(0.))
        self.register_buffer('active', torch.tensor(False))
        self.register_buffer('generation', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Optional FP32 trainable/master scope with a BF16 frozen backbone.
        return F.linear(F.silu(F.linear(x.to(self.a.dtype), self.a)), self.b).to(x.dtype)


class SparseExperts(nn.Module):
    def __init__(self, dim, rank=4, capacity=8):
        super().__init__()
        self.slots = nn.ModuleList([ExpertSlot(dim, rank) for _ in range(capacity)])
        self.dream = False
        self.generator = None
        self.last_routes = []
        self.route_override = None
        self.audit_routes = False
        self.last_route_audit = None
        self.audit_history = []

    def activate(self, index=None):
        if index is None:
            index = next((i for i, s in enumerate(self.slots) if not bool(s.active)), None)
        if index is None:
            raise RuntimeError('expert reserve exhausted; no silent overwrite')
        slot = self.slots[index]
        if bool(slot.active):
            raise RuntimeError('slot already active')
        with torch.no_grad():
            slot.active.fill_(True)
            slot.generation.add_(1)
            slot.priority.fill_(max([float(s.priority) for s in self.slots if bool(s.active)] + [0.]) + 1.)
        return index

    def reset_active(self):
        ids = []
        with torch.no_grad():
            for i, slot in enumerate(self.slots):
                if bool(slot.active):
                    # Zero functional output and routing lifecycle. A can have
                    # learned during wake and is deliberately retained: this is
                    # basis reuse, not full parameter reinitialization.
                    slot.b.zero_()
                    slot.active.fill_(False)
                    slot.priority.zero_()
                    slot.generation.add_(1)
                    slot.requires_grad_(False)
                    slot.a.grad = slot.b.grad = None
                    ids.append(i)
        return ids

    def deactivate(self, index):
        slot = self.slots[index]
        slot.active.fill_(False)
        slot.requires_grad_(False)
        for p in slot.parameters():
            p.grad = None

    def save_expert(self, index, path):
        torch.save({'schema': 1, 'state': {k: v.detach().cpu().clone()
                   for k, v in self.slots[index].state_dict().items()}}, path)

    def load_expert(self, index, path, *, activate=False):
        data = torch.load(path, weights_only=True, map_location=self.slots[index].a.device)
        if data['schema'] != 1:
            raise ValueError('expert schema mismatch')
        self.slots[index].load_state_dict(data['state'])
        self.slots[index].active.fill_(activate)
        self.slots[index].requires_grad_(False)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        active = [i for i, s in enumerate(self.slots) if bool(s.active)]
        override = self.route_override or {}
        disabled = set(override.get('disable', []))
        active = [i for i in active if i not in disabled]
        forced = override.get('force')
        mixed = override.get('mix')
        fraction = float(override.get('fraction', .5))
        included = override.get('include')
        if included is not None and (included not in active or forced is not None or mixed is not None or self.dream):
            raise ValueError('explicit inclusion requires an active receiver and no force/mix/dream')
        if mixed is not None and (mixed not in active or not 0 < fraction < 1 or forced is not None):
            raise ValueError('mixed routing requires an active expert, fraction in (0,1), and no force')
        if forced is not None and forced not in active:
            raise ValueError('forced expert must be active and not disabled')
        self.last_routes = []
        self.last_route_audit = None
        result = torch.zeros_like(flat)
        if active:
            logits = torch.stack([flat.to(self.slots[i].route.dtype) @ self.slots[i].route
                                  + self.slots[i].priority for i in active], -1)
            winner = logits.argmax(-1)
            gates = torch.softmax(torch.cat([logits, torch.zeros_like(logits[:, :1])], -1), -1)
            if forced is not None:
                winner = torch.full_like(winner, active.index(forced))
                gates = torch.zeros_like(gates)
                gates[:, active.index(forced)] = 1.
            if self.audit_routes:
                ids = torch.tensor(active, device=x.device)
                self.last_route_audit = {'active': active, 'selected': ids[winner].reshape(shape[:-1]).detach(),
                    'probabilities': gates.reshape(*shape[:-1], len(active)+1).detach(),
                    'score_abs_max': float(logits.detach().abs().max())}
                self.audit_history.append(self.last_route_audit)
            executed_winner = winner
            if included is not None:
                # Diagnostic union: keep one OLD expert with its unchanged gate,
                # add the receiver at coefficient1. Router scores stay untouched.
                old_local = [j for j,i in enumerate(active) if i != included]
                executed_winner = (torch.tensor(old_local,device=x.device)[logits[:,old_local].argmax(-1)]
                                   if old_local else torch.full_like(winner,-1))
                if self.audit_routes:
                    self.last_route_audit['included_slot'] = included
                    self.last_route_audit['old_selected'] = torch.where(executed_winner>=0,
                        ids[executed_winner.clamp_min(0)],-1).reshape(shape[:-1]).detach()
            for local, i in enumerate(active):
                rows = executed_winner == local
                if bool(rows.any()):
                    result[rows] = self.slots[i](flat[rows]) * gates[rows, local:local+1]
                    self.last_routes.append({'slot': i, 'tokens': int(rows.sum()), 'kind': 'normal'})
            if included is not None:
                result = result + self.slots[included](flat)
                self.last_routes.append({'slot':included,'tokens':len(flat),'kind':'explicit_receiver_diagnostic'})
            if mixed is not None:
                # Convex residual mixture, same policy in training and deployment.
                # Natural routing is still measured separately from executed coverage.
                result = (1. - fraction) * result + fraction * self.slots[mixed](flat)
                self.last_routes.append({'slot': mixed, 'tokens': len(flat), 'kind': 'controlled_mix'})
                if self.audit_routes:
                    self.last_route_audit['mixed_slot'] = mixed
                    self.last_route_audit['mixed_fraction'] = fraction
        if self.dream:
            # Distinct active experts first; with a singleton active pool the
            # same expert is sampled again. Dormant slots remain masked always.
            normally_used = {r['slot'] for r in self.last_routes}
            pool = [i for i in active if i not in normally_used] or active
            if pool:
                index = int(torch.randint(len(pool), (), device=flat.device, generator=self.generator))
                i = pool[index]
                result = result + .5 * self.slots[i](flat)
                self.last_routes.append({'slot': i, 'tokens': len(flat), 'kind': 'random_extra',
                                         'active': bool(self.slots[i].active)})
        return result.reshape(shape)


class LowRankMemoryLevel(nn.Module):
    """Track B CMS-inspired residual adapter; not a native HOPE FFN."""
    def __init__(self, dim, rank=4, capacity=8):
        super().__init__()
        self.a = nn.Parameter(torch.randn(rank, dim) * .02)
        self.b = nn.Parameter(torch.zeros(dim, rank))
        self.experts = SparseExperts(dim, rank, capacity)

    def forward(self, x):
        update = F.linear(F.silu(F.linear(x.to(self.a.dtype), self.a)), self.b).to(x.dtype)
        return x + update + self.experts(x)
