from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from nested_sleep.experts import LowRankMemoryLevel  # noqa: E402
from banking_cil_v10_routing import activate_additive_receiver  # noqa: E402
from banking_cil_v10_lifecycle import AdditiveSleepTransaction  # noqa: E402
from banking_cil_v12_batching import (  # noqa: E402
    BatchableAdditiveSleepTransaction,
    optimize_request_batch,
)
from clinc_lifecycle_v8 import _state_hash  # noqa: E402


class TinyContext:
    def __init__(self):
        self.memory = nn.ModuleList([
            LowRankMemoryLevel(dim=6, rank=3, capacity=4),
            LowRankMemoryLevel(dim=6, rank=3, capacity=4),
        ])
        self.readout_a = nn.Parameter(torch.randn(3, 6))
        self.readout_b = nn.Parameter(torch.zeros(9, 3))
        self.readout_fp32_accumulation = True
        self.banks()[0].activate(0)
        activate_additive_receiver(self, 1, slot=0)
        with torch.no_grad():
            self.memory[0].a.fill_(0.05)
            self.memory[0].b.fill_(0.12)
            self.memory[1].a.fill_(-0.04)
            self.memory[1].b.fill_(0.08)
            self.banks()[0].slots[0].a.fill_(0.1)
            self.banks()[0].slots[0].b.fill_(0.07)
            self.banks()[1].slots[0].a.fill_(0.09)
            self.banks()[1].slots[0].b.fill_(0.06)

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
            "routing": [deepcopy(bank.route_override) for bank in self.banks()],
            "trainable": {k: bool(v.requires_grad) for k, v in self.named_parameters()},
        }

    def load_state_dict(self, state, *, strict=True):
        expected = {"memory", "readout_a", "readout_b", "readout_fp32_accumulation", "routing", "trainable"}
        if set(state) != expected:
            raise ValueError("fixture context state schema mismatch")
        result = self.memory.load_state_dict(state["memory"], strict=strict)
        with torch.no_grad():
            self.readout_a.copy_(state["readout_a"])
            self.readout_b.copy_(state["readout_b"])
        for bank, policy in zip(self.banks(), state["routing"]):
            bank.route_override = deepcopy(policy)
        self.readout_fp32_accumulation = bool(state["readout_fp32_accumulation"])
        for name, param in self.named_parameters():
            param.requires_grad_(bool(state["trainable"][name]))
        return result


def run_context(context, x):
    for level in context.memory:
        x = level(x)
    return x


def make_tx(tx_type=BatchableAdditiveSleepTransaction, seed=123):
    live = TinyContext()
    prospective = {
        "memory.0.a": live.memory[0].a.detach().clone() + 0.03,
        "memory.0.b": live.memory[0].b.detach().clone() - 0.02,
    }
    tx = tx_type(live, 0, prospective, lr=0.01, weight_decay=0.0,
                 clip_norm=1.0, seed=seed)
    return live, tx


def make_closure(tx, x, target):
    def closure():
        prediction = run_context(tx.student_context(), x)
        return (prediction - target).square().mean(), int(x.shape[0])
    return closure


def _nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_nested_equal(x, y) for x, y in zip(a, b))
    return a == b


def _allowed_state(tx):
    params = dict(tx.student.named_parameters())
    return {name: params[name].detach().clone() for name in sorted(tx.allowed_names)}


def _assert_state_equal(left, right):
    assert _nested_equal(left.student.state_dict(), right.student.state_dict())
    assert _nested_equal(left.optimizer.state_dict(), right.optimizer.state_dict())
    assert left.counters == right.counters
    assert torch.equal(left.generator.get_state(), right.generator.get_state())


def test_batch_size_one_matches_legacy_single_optimize():
    torch.manual_seed(7)
    live_a, legacy = make_tx(AdditiveSleepTransaction)
    torch.manual_seed(7)
    live_b, batched = make_tx(BatchableAdditiveSleepTransaction)
    assert _state_hash(live_a) == _state_hash(live_b)
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10
    target = torch.full((4, 6), 0.13)

    legacy_loss = (run_context(legacy.student_context(), x) - target).square().mean()
    legacy.optimize(legacy_loss, "gkd", tokens=4)
    result = optimize_request_batch(batched, [make_closure(batched, x, target)])

    assert result["request_count"] == 1
    assert result["token_count"] == 4
    assert result["backward_calls"] == 1
    assert result["physical_adamw_steps"] == 1
    assert _nested_equal(legacy.student.state_dict(), batched.student.state_dict())
    assert _nested_equal(legacy.optimizer.state_dict(), batched.optimizer.state_dict())
    assert batched.counters["gkd_calls"] == 1
    assert batched.counters["adamw_steps"] == 1
    assert batched.counters["v12_physical_adamw_steps"] == 1


def test_batch_size_32_matches_one_mean_loss_clip_and_adamw_step():
    torch.manual_seed(11)
    _, reference = make_tx(AdditiveSleepTransaction, seed=81)
    torch.manual_seed(11)
    _, batched = make_tx(BatchableAdditiveSleepTransaction, seed=81)
    generator = torch.Generator().manual_seed(55)
    examples = []
    for _ in range(32):
        x = torch.randn(3, 6, generator=generator)
        target = torch.randn(3, 6, generator=generator)
        examples.append((x, target))

    losses = [
        (run_context(reference.student_context(), x) - target).square().mean()
        for x, target in examples
    ]
    mean_loss = torch.stack(losses).mean()
    reference.optimize(mean_loss, "gkd", tokens=sum(x.shape[0] for x, _ in examples))

    zero_calls = 0
    original_zero_grad = batched.optimizer.zero_grad

    def counted_zero_grad(*args, **kwargs):
        nonlocal zero_calls
        zero_calls += 1
        return original_zero_grad(*args, **kwargs)

    batched.optimizer.zero_grad = counted_zero_grad
    clip_calls = 0
    original_clip = torch.nn.utils.clip_grad_norm_

    def counted_clip(*args, **kwargs):
        nonlocal clip_calls
        clip_calls += 1
        return original_clip(*args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = counted_clip
    try:
        result = optimize_request_batch(
            batched, [make_closure(batched, x, target) for x, target in examples]
        )
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
    assert result["request_count"] == 32
    assert result["token_count"] == 96
    assert result["backward_calls"] == 32
    assert result["physical_adamw_steps"] == 1
    assert batched.counters["gkd_calls"] == 32
    assert batched.counters["adamw_steps"] == 1
    assert batched.counters["backward_attempts"] == 32
    assert batched.counters["tokens"] == 96
    assert zero_calls == 1
    assert clip_calls == 1
    ref_params = dict(reference.student.named_parameters())
    batch_params = dict(batched.student.named_parameters())
    for name in batched.allowed_names:
        assert torch.allclose(ref_params[name], batch_params[name], atol=1e-7, rtol=1e-6)
    assert torch.allclose(
        torch.tensor(result["loss_mean"]), mean_loss.detach(), atol=1e-7, rtol=1e-6
    )
    assert result["preclip_grad_norm"] > 0
    assert result["parameter_update_l2"] > 0


@pytest.mark.parametrize("failure_kind", ["nonfinite", "exception"])
def test_failed_batch_rolls_back_parameters_clears_gradients_and_closes(failure_kind):
    torch.manual_seed(13)
    live, tx = make_tx(seed=83)
    warm_x = torch.randn(3, 6)
    warm_target = torch.zeros_like(warm_x)
    optimize_request_batch(tx, [make_closure(tx, warm_x, warm_target)])
    params_before = _allowed_state(tx)
    student_hash = _state_hash(tx.student)
    live_hash = _state_hash(live)
    teacher_hash = _state_hash(tx.teacher)
    optimizer_before = deepcopy(tx.optimizer.state_dict())

    good_x = torch.randn(2, 6)
    good = make_closure(tx, good_x, torch.ones_like(good_x))

    def bad():
        if failure_kind == "exception":
            raise RuntimeError("synthetic closure failure")
        bad_loss = run_context(tx.student_context(), torch.ones(2, 6)).sum() * torch.tensor(float("nan"))
        return bad_loss, 2

    expected = RuntimeError if failure_kind == "exception" else ValueError
    with pytest.raises(expected):
        optimize_request_batch(tx, [good, bad])

    assert _state_hash(tx.student) == student_hash
    assert _state_hash(live) == live_hash
    assert _state_hash(tx.teacher) == teacher_hash
    assert _nested_equal(tx.optimizer.state_dict(), optimizer_before)
    params_after = _allowed_state(tx)
    assert _nested_equal(params_before, params_after)
    assert all(parameter.grad is None for parameter in tx.student.parameters())
    assert tx.counters["v12_failed"] is True
    assert tx.counters["closed"] is True
    assert tx.counters["adamw_steps"] == 1
    assert tx.counters["v12_physical_adamw_steps"] == 1
    with pytest.raises(RuntimeError, match="failed"):
        tx.commit()

    failed_snapshot = tx.snapshot()
    restored = BatchableAdditiveSleepTransaction.from_snapshot(live, failed_snapshot)
    assert restored.counters["v12_failed"] is True
    assert restored.counters["closed"] is True
    assert all(parameter.grad is None for parameter in restored.student.parameters())
    with pytest.raises(RuntimeError, match="failed"):
        optimize_request_batch(restored, [good])


def test_snapshot_is_rejected_midbatch_and_allowed_at_boundary():
    torch.manual_seed(17)
    live, tx = make_tx(seed=85)
    x = torch.randn(3, 6)
    target = torch.zeros_like(x)

    def closure():
        with pytest.raises(RuntimeError, match="completed microbatch boundary"):
            tx.snapshot()
        return (run_context(tx.student_context(), x) - target).square().mean(), 3

    optimize_request_batch(tx, [closure])
    snapshot = tx.snapshot()
    assert all(parameter.grad is None for parameter in tx.student.parameters())
    restored = BatchableAdditiveSleepTransaction.from_snapshot(live, snapshot)
    assert not restored.counters["v12_failed"]
    assert all(parameter.grad is None for parameter in restored.student.parameters())


def test_post_step_guard_failure_rolls_back_adam_and_does_not_count_commit():
    torch.manual_seed(18)
    live, tx = make_tx(seed=86)
    params_before = _allowed_state(tx)
    optimizer_before = deepcopy(tx.optimizer.state_dict())
    x = torch.randn(3, 6)
    target = torch.zeros_like(x)
    original_check = tx._check
    calls = 0

    def fail_after_step(*, deep=False):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic post-step protected-state failure")
        return original_check(deep=deep)

    tx._check = fail_after_step
    with pytest.raises(RuntimeError, match="post-step"):
        optimize_request_batch(tx, [make_closure(tx, x, target)])
    assert _nested_equal(params_before, _allowed_state(tx))
    assert _nested_equal(optimizer_before, tx.optimizer.state_dict())
    assert tx.counters["adamw_steps"] == 0
    assert tx.counters["gkd_calls"] == 0
    assert tx.counters["v12_physical_adamw_steps"] == 0
    assert tx.counters["v12_adamw_step_attempts"] == 1
    assert tx.counters["v12_batch_commits"] == 0
    assert tx.counters["v12_failed"] is True
    assert all(parameter.grad is None for parameter in tx.student.parameters())
    restored = BatchableAdditiveSleepTransaction.from_snapshot(live, tx.snapshot())
    assert restored.counters["v12_failed"] is True
    assert restored.counters["v12_adamw_step_attempts"] == 1
    assert restored.counters["adamw_steps"] == 0
    assert all(parameter.grad is None for parameter in restored.student.parameters())


def test_disk_restore_matches_continuing_original_transaction(tmp_path):
    torch.manual_seed(19)
    live, tx = make_tx(seed=87)
    first = [make_closure(tx, torch.randn(3, 6), torch.zeros(3, 6)) for _ in range(2)]
    optimize_request_batch(tx, first)
    snapshot = tx.snapshot()
    snapshot_path = tmp_path / "batchable-sleep-snapshot.pt"
    torch.save(snapshot, snapshot_path)
    loaded = torch.load(snapshot_path, map_location="cpu", weights_only=False)

    def random_batch(target_tx):
        closures = []
        for _ in range(3):
            def closure():
                x = torch.randn(3, 6)
                target = torch.randn(3, 6)
                return (run_context(target_tx.student_context(), x) - target).square().mean(), 3
            closures.append(closure)
        return closures

    # Continue the original object, then restore a disk-loaded branch from the
    # saved boundary and replay the same next microbatch from restored RNG.
    optimize_request_batch(tx, random_batch(tx))
    continued_student = deepcopy(tx.student.state_dict())
    continued_optimizer = deepcopy(tx.optimizer.state_dict())
    continued_counters = deepcopy(tx.counters)
    continued_rng = torch.get_rng_state().clone()

    restored = BatchableAdditiveSleepTransaction.from_snapshot(live, loaded)
    optimize_request_batch(restored, random_batch(restored))
    assert _nested_equal(continued_student, restored.student.state_dict())
    assert _nested_equal(continued_optimizer, restored.optimizer.state_dict())
    assert continued_counters == restored.counters
    assert torch.equal(continued_rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in restored.student.parameters())
