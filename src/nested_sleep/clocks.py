"""Exactly C events per non-overlapping accumulation window (explicit choice).

This is the scheduling part of NL Eq 71/Sleep Eq 2, not a complete CMS network.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class GradientWindow:
    period: int
    events: int
    pending: torch.Tensor

    @classmethod
    def create(cls, parameter, period):
        if not isinstance(period, int) or period < 1:
            raise ValueError('period must be a positive number of semantic events')
        return cls(period, 0, torch.zeros_like(parameter))

    def accumulate(self, gradient, learning_rate):
        if gradient.shape != self.pending.shape:
            raise ValueError('gradient shape mismatch')
        if self.events >= self.period:
            raise RuntimeError('a due window must be committed before more accumulation')
        return GradientWindow(self.period, self.events + 1, self.pending + learning_rate * gradient)

    @property
    def due(self):
        return self.events == self.period

    def prospective(self, parameter):
        if not self.due:
            raise RuntimeError('window is not due')
        return parameter - self.pending

    def consume(self):
        if not self.due:
            raise RuntimeError('only committed due windows may be consumed')
        return GradientWindow.create(self.pending, self.period)

    def state_dict(self):
        return {'period': self.period, 'events': self.events, 'pending': self.pending.detach().clone()}

    @classmethod
    def from_state_dict(cls, state):
        period, events, pending = state['period'], state['events'], state['pending']
        if period < 1 or not 0 <= events <= period or not torch.isfinite(pending).all():
            raise ValueError('invalid clock checkpoint')
        return cls(period, events, pending.clone())
