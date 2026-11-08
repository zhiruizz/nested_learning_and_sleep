from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from nested_sleep.experts import LowRankMemoryLevel
from banking_cil_v10_lifecycle import AdditiveSleepTransaction
from banking_cil_v10_runtime import run_sleep_event, local_lti_reward
from banking_cil_v13_batching import BatchableAdditiveSleepTransactionV13
from banking_sleep_v13_runtime import run_batched_sleep_event
from run_banking_sleep_lifecycle_v13 import build_v10_equivalent_objective
import clinc_tail_runtime_v8 as tail_runtime


class TinyMutableContext:
    """Real low-rank memory modules behind the V10 mutable-context contract."""
    def __init__(self):
        self.memory = nn.ModuleList([
            LowRankMemoryLevel(dim=80, rank=2, capacity=4),
            LowRankMemoryLevel(dim=80, rank=2, capacity=4),
        ])
        self.readout_a = nn.Parameter(torch.zeros(2, 80))
        self.readout_b = nn.Parameter(torch.zeros(80, 2))
        self.readout_fp32_accumulation = True
        self.banks()[0].activate(0)
        self.banks()[1].activate(0)
        with torch.no_grad():
            for i, level in enumerate(self.memory):
                level.a.fill_(0.02 * (i + 1))
                level.b.fill_(0.01 * (i + 1))
                level.experts.slots[0].a.fill_(0.03 * (i + 1))
                level.experts.slots[0].b.fill_(0.02 * (i + 1))

    def banks(self):
        return [level.experts for level in self.memory]

    def parameters(self):
        yield from self.memory.parameters()
        yield self.readout_a
        yield self.readout_b

    def named_parameters(self):
        yield from self.memory.named_parameters(prefix="memory")
        yield "readout_a", self.readout_a
        yield "readout_b", self.readout_b

    def state_dict(self):
        return {
            "memory": {k: v.detach().cpu().clone() for k, v in self.memory.state_dict().items()},
            "readout_a": self.readout_a.detach().cpu().clone(),
            "readout_b": self.readout_b.detach().cpu().clone(),
            "readout_fp32_accumulation": self.readout_fp32_accumulation,
            "routing": [copy.deepcopy(bank.route_override) for bank in self.banks()],
            "trainable": {name: bool(p.requires_grad) for name, p in self.named_parameters()},
        }

    def load_state_dict(self, state, *, strict=True):
        self.memory.load_state_dict(state["memory"], strict=strict)
        with torch.no_grad():
            self.readout_a.copy_(state["readout_a"])
            self.readout_b.copy_(state["readout_b"])
        for bank, policy in zip(self.banks(), state["routing"]):
            bank.route_override = copy.deepcopy(policy)
        self.readout_fp32_accumulation = bool(state["readout_fp32_accumulation"])
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(bool(state["trainable"][name]))


class TinyTokenizer:
    eos_token_id = 78

    def encode(self, text, add_special_tokens=False):
        if str(text).startswith("class_"):
            return [int(str(text).split("_", 1)[1]) + 1]
        return [1]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"class_{int(x)-1}" for x in ids
                         if 1 <= int(x) <= 77 and int(x) != self.eos_token_id).strip()


class TinyCommon:
    @staticmethod
    def tokenize_prompt(text, tokenizer, classes, eos_token_id, max_sequence_length):
        return {"input_ids": [1, 2], "prompt_length": 2}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("projection", torch.eye(80))


class TinyTailRuntime:
    capture_count = 0

    @staticmethod
    def capture(model, sequence, mask, *, device):
        TinyTailRuntime.capture_count += 1
        return sequence

    @staticmethod
    def tail_logits(model, context, cache):
        x = torch.nn.functional.one_hot(cache.long() % 80, num_classes=80).float()
        for level in context.memory:
            x = level(x)
        return x @ model.projection.T

    causal_token_kl = staticmethod(tail_runtime.causal_token_kl)
    causal_token_ce = staticmethod(tail_runtime.causal_token_ce)


class SampleResult:
    def __init__(self, ids, prefix_length):
        self.token_ids = ids
        self.prefix_length = prefix_length

    def answer_mask(self):
        mask = torch.zeros_like(self.token_ids, dtype=torch.bool)
        mask[:, self.prefix_length:] = True
        return mask


def _context_pair():
    torch.manual_seed(318)
    live_a = TinyMutableContext()
    live_b = copy.deepcopy(live_a)
    prospective = {
        "memory.0.a": live_a.memory[0].a.detach().clone() + 0.005,
        "memory.0.b": live_a.memory[0].b.detach().clone() - 0.003,
    }
    return live_a, live_b, prospective


def _sampler(tx, tokenizer, seen):
    def sample_tokens(model, context, prefix_ids, count, temperature, generator,
                      eos, torch_module, device, budget):
        owner = "teacher" if context is tx.teacher_context() else "student"
        sampled = int(torch.randint(1, 78, (1,), generator=generator, device=generator.device).item())
        answer = [sampled, int(eos)]
        ids = torch.tensor([[*map(int, prefix_ids), *answer]], dtype=torch.long, device=device)
        seen.append((owner, tuple(prefix_ids), int(count), sampled))
        return SampleResult(ids, len(prefix_ids)), answer
    return sample_tokens


def _same_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _same_tree(a, b) for a, b in zip(left, right))
    return left == right


def test_v13_batch_one_matches_v10_sleep_event_objectives_sampling_and_adam():
    classes = [f"class_{i}" for i in range(77)]
    tokenizer = TinyTokenizer()
    cfg = {"max_new_tokens": 10, "max_sequence_length": 32,
           "temperature": 0.8, "lti_edit_distance_z0": 8}
    rows = [
        ("gkd", {"id": "r0", "text": "first", "intent_id": 0}, "current"),
        ("gkd", {"id": "r1", "text": "second", "intent_id": 1}, "current"),
        ("lti", {"id": "r2", "text": "third", "intent_id": 2}, "current"),
    ]

    live10, live13, prospective = _context_pair()
    tx10 = AdditiveSleepTransaction(live10, 0, prospective, lr=0.001,
                                    weight_decay=0.0, clip_norm=1.0, seed=73)
    tx13 = BatchableAdditiveSleepTransactionV13(live13, 0, prospective, lr=0.001,
                                                weight_decay=0.0, clip_norm=1.0, seed=73)
    model = TinyModel()
    seen10, seen13 = [], []
    sampler10, sampler13 = _sampler(tx10, tokenizer, seen10), _sampler(tx13, tokenizer, seen13)

    reference = run_sleep_event(
        model, tx10, rows, tokenizer, classes, TinyCommon, cfg, torch, "cpu", None,
        TinyTailRuntime, sample_tokens=sampler10, lti_reward=local_lti_reward)

    objective = build_v10_equivalent_objective(
        model=model, tx=tx13, tokenizer=tokenizer, classes=classes, cfg=cfg,
        torch=torch, device="cpu", budget=None, tail_runtime=TinyTailRuntime,
        common=TinyCommon, sample_tokens=sampler13, lti_reward=local_lti_reward)
    TinyTailRuntime.capture_count = 0
    actual = run_batched_sleep_event(rows, tx13, microbatch_size=1,
                                     objective_builder=objective)

    assert seen13 == seen10
    assert [x["sample_owner"] for x in reference["per_step"]] == [
        "teacher", "student", "teacher"]
    assert [x["owner"] for x in actual["request_summaries"]] == [
        "teacher", "student", "teacher"]
    for old, new in zip(reference["per_step"], actual["batches"]):
        assert old["kind"] in ("gkd", "lti")
        assert new["request_count"] == 1
        assert new["physical_adamw_steps"] == 1
        assert old["loss"] == pytest.approx(new["loss_mean"], abs=1e-7)
    for old, new in zip(reference["per_step"], actual["request_summaries"]):
        assert old["kind"] == new["kind"]
        assert old["sample_owner"] == new["owner"]
    assert torch.equal(tx13.generator.get_state(), tx10.generator.get_state())
    assert _same_tree(tx13.student.state_dict(), tx10.student.state_dict())
    assert _same_tree(tx13.optimizer.state_dict(), tx10.optimizer.state_dict())
    for counter in ("adamw_steps", "backward_attempts", "tokens", "gkd_calls", "lti_calls"):
        assert tx13.counters[counter] == tx10.counters[counter]
    assert actual["physical_adamw_steps"] == reference["steps"] == 3
    assert actual["backward_calls"] == reference["steps"] == 3
