"""Persistent CMS context flow with checkpointed gradient sums and SGD state.

Time unit is one supervised minibatch, explicit local choice. No Adam moments are
silently lost: this optimizer is plain SGD and its state is counters + pending sums.
Fast active experts update online; base writes are staged until their own boundary.
"""
import torch


class CMSController:
    def __init__(self, model, periods=(4, 32), base_lr=.01, expert_lr=.05):
        levels = model.level_descriptors()
        if len(levels) != len(periods) or any(p < 1 for p in periods):
            raise ValueError('one positive period per level')
        self.periods, self.base_lr, self.expert_lr = tuple(periods), base_lr, expert_lr
        self.counts, self.events = [0] * len(levels), 0
        params = dict(model.named_parameters())
        self.pending = [{n: torch.zeros_like(params[n]) for n in d['base']} for d in levels]

    def prepare_wake(self, model):
        model.requires_grad_(False)
        params = dict(model.named_parameters())
        for descriptor in model.level_descriptors():
            for name in descriptor['base']:
                params[name].requires_grad_(True)
        bank = model.banks()[0]
        if not any(bool(s.active) for s in bank.slots):
            bank.activate()
        for slot in bank.slots:
            if bool(slot.active):
                slot.requires_grad_(True)
        model.zero_grad(set_to_none=True)

    def collect(self, model, *, sequential=False):
        if any(count >= period for count, period in zip(self.counts,self.periods)):
            raise RuntimeError('unconsumed CMS proposal; commit/resolve or stop before collecting more gradients')
        params = dict(model.named_parameters())
        self.events += 1
        proposals = {}
        with torch.no_grad():
            # Fast expert learner: explicit online parameter state.
            for slot in model.banks()[0].slots:
                if bool(slot.active):
                    for p in (slot.a, slot.b):
                        if p.grad is not None:
                            p.add_(p.grad, alpha=-self.expert_lr)
            for level, descriptor in enumerate(model.level_descriptors()):
                self.counts[level] += 1
                for name in descriptor['base']:
                    p = params[name]
                    if p.grad is not None:
                        self.pending[level][name].add_(p.grad, alpha=self.base_lr)
                if sequential or self.counts[level] == self.periods[level]:
                    proposals[level] = {n: params[n].detach() - acc for n, acc in self.pending[level].items()}
        model.zero_grad(set_to_none=True)
        return proposals

    def consume(self, level):
        self.pending[level] = {n: torch.zeros_like(v) for n, v in self.pending[level].items()}
        self.counts[level] = 0

    def apply(self, model, level, proposal):
        params = dict(model.named_parameters())
        with torch.no_grad():
            for n, value in proposal.items():
                params[n].copy_(value)
        self.consume(level)

    def state_dict(self):
        return {'periods': self.periods, 'base_lr': self.base_lr, 'expert_lr': self.expert_lr,
                'counts': self.counts.copy(), 'events': self.events,
                'pending': [{n: x.clone() for n, x in p.items()} for p in self.pending]}

    @classmethod
    def restore(cls, model, state):
        obj = cls(model, state['periods'], state['base_lr'], state['expert_lr'])
        if len(state['counts']) != len(obj.periods):
            raise ValueError('clock topology mismatch')
        obj.counts, obj.events = list(state['counts']), state['events']
        for actual, saved in zip(obj.pending, state['pending']):
            if actual.keys() != saved.keys():
                raise ValueError('optimizer parameter identity mismatch')
            for n in actual:
                actual[n].copy_(saved[n])
        return obj
