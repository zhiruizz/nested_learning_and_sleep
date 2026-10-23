"""Forward-token accounting includes prompts, teacher, trials and rewards.

Counts are compute proxies, not FLOPs or GPU busy time. Context-local state is
external to model snapshots, so deepcopy cannot duplicate/lose counters.
"""
from contextvars import ContextVar
import torch

ACTIVE_METER = ContextVar('nested_sleep_work_meter', default=None)


class WorkMeter:
    def __init__(self):
        self.stage = 'wake'
        self.counts = {}

    def charge(self, tokens):
        row = self.counts.setdefault(self.stage, {'forward_input_tokens': 0, 'gradient_forward_input_tokens': 0,
                                                 'forward_calls': 0})
        row['forward_input_tokens'] += tokens.numel()
        row['gradient_forward_input_tokens'] += tokens.numel() if torch.is_grad_enabled() else 0
        row['forward_calls'] += 1


def charge(tokens):
    meter = ACTIVE_METER.get()
    if meter is not None:
        meter.charge(tokens)
