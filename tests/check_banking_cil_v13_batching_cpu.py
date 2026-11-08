from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import banking_cil_v13_batching as batching
from banking_sleep_v13_runtime import run_batched_sleep_event, sample_owner


class ToyTransaction:
    def __init__(self, value=0.25):
        self.student = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.student.weight.fill_(value)
        self.allowed_names = {"weight"}
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=0.05, weight_decay=0.0)
        self.clip_norm = 100.0
        self.counters = {"closed": False, "adamw_steps": 0, "backward_attempts": 0,
                         "tokens": 0, "gkd_calls": 0, "lti_calls": 0}

    def _check(self):
        assert torch.isfinite(self.student.weight).all()


def _req(tx, kind, coefficient, *, zero=False):
    p = tx.student.weight.sum()
    loss = p * 0.0 if zero else (p - coefficient) ** 2
    return {"kind": kind, "loss": loss, "token_count": 2, "zero_objective": zero}


def test_locked_schedule_counts_and_960_request_batches():
    kinds = ["gkd", "gkd", "lti"] * 320
    assert batching.validate_request_schedule(kinds) == {"opportunities": 960, "gkd": 640, "lti": 320}
    groups = batching.group_request_indices(kinds, 32)
    assert len(groups) == 30 and all(len(x) == 32 for x in groups)
    assert groups[0][0] == 0 and groups[-1][-1] == 959
    assert [sample_owner("gkd", 0), sample_owner("gkd", 1), sample_owner("lti", 1)] == [
        "teacher", "student", "teacher"]
    extended = ["gkd", "gkd", "lti"] * 1066 + ["gkd", "gkd"]
    assert batching.validate_request_schedule(extended, complete_cycles=False) == {
        "opportunities": 3200, "gkd": 2134, "lti": 1066}
    ext_groups = batching.group_request_indices(extended, 32)
    assert len(ext_groups) == 100 and all(len(x) == 32 for x in ext_groups)


def test_b1_matches_sequential_request_mean_objective_updates():
    v13, reference = ToyTransaction(), ToyTransaction()
    coefficients = [0.1, 0.7, -0.2]
    kinds = ["gkd", "gkd", "lti"]
    v13_results = []
    for kind, c in zip(kinds, coefficients):
        v13_results.append(batching.optimize_mixed_request_batch(v13, [lambda k=kind, c=c: _req(v13, k, c)]))
    for kind, c in zip(kinds, coefficients):
        loss = (reference.student.weight.sum() - c) ** 2
        reference.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.student.parameters(), reference.clip_norm)
        reference.optimizer.step()
    assert torch.equal(v13.student.weight, reference.student.weight)
    assert v13.counters["adamw_steps"] == 3
    assert [x["physical_adamw_steps"] for x in v13_results] == [1, 1, 1]


def test_mixed_batch_averages_all_requests_and_steps_once():
    tx = ToyTransaction()
    coeffs = [0.1, 0.4, -0.3]
    result = batching.optimize_mixed_request_batch(
        tx, [lambda k=k, c=c: _req(tx, k, c) for k, c in zip(("gkd", "gkd", "lti"), coeffs)])
    ref = ToyTransaction()
    losses = [(ref.student.weight.sum() - c) ** 2 for c in coeffs]
    (sum(losses) / 3).backward()
    torch.nn.utils.clip_grad_norm_(ref.student.parameters(), ref.clip_norm)
    ref.optimizer.step()
    assert torch.equal(tx.student.weight, ref.student.weight)
    assert result["request_count"] == 3
    assert result["gkd_opportunities"] == 2 and result["lti_opportunities"] == 1
    assert result["physical_adamw_steps"] == 1 and result["backward_calls"] == 3


def test_exact_zero_flagged_lti_skips_optimizer_but_counts_opportunity():
    tx = ToyTransaction(); before = tx.student.weight.detach().clone()
    result = batching.optimize_mixed_request_batch(
        tx, [lambda: _req(tx, "lti", 0, zero=True)])
    assert result["skipped_zero_gradient"] is True and result["physical_adamw_steps"] == 0
    assert torch.equal(before, tx.student.weight)
    assert tx.counters["v13_objective_opportunities"] == 1
    assert tx.counters["v13_lti_opportunities"] == 1
    assert tx.counters["v13_zero_objective_opportunities"] == 1
    assert tx.counters["adamw_steps"] == 0


def test_unflagged_zero_scalar_with_nonzero_gradient_backprops_and_commits():
    tx = ToyTransaction(); before = tx.student.weight.detach().clone()
    def zero_scalar_nonzero_grad():
        p = tx.student.weight.sum()
        return {"kind": "gkd", "loss": p - p.detach(),
                "token_count": 2, "zero_objective": False}
    result = batching.optimize_mixed_request_batch(tx, [zero_scalar_nonzero_grad])
    assert result["backward_calls"] == 1
    assert result["physical_adamw_steps"] == 1
    assert tx.counters["adamw_steps"] == 1
    assert tx.counters["backward_attempts"] == 1
    assert not torch.equal(before, tx.student.weight)


def test_unflagged_zero_objectives_with_cancelling_gradients_fail_closed():
    tx = ToyTransaction(); before = tx.student.weight.detach().clone()
    def zero_scalar_with_sign(sign):
        p = tx.student.weight.sum()
        return {"kind": "gkd", "loss": sign * (p - p.detach()),
                "token_count": 2, "zero_objective": False}
    with pytest.raises(RuntimeError, match="zero|gradient|objective"):
        batching.optimize_mixed_request_batch(
            tx, [lambda: zero_scalar_with_sign(1.0), lambda: zero_scalar_with_sign(-1.0)])
    assert torch.equal(before, tx.student.weight)
    assert tx.counters["backward_attempts"] == 2
    assert tx.counters["adamw_steps"] == 0
    assert tx.counters["v13_failed"] is True


def test_zero_lti_callback_still_checks_protected_state():
    class ProtectedToyTransaction(ToyTransaction):
        def __init__(self):
            super().__init__()
            self.protected = torch.tensor([1.0])
            self._protected_before = self.protected.clone()

        def _check(self):
            super()._check()
            if not torch.equal(self.protected, self._protected_before):
                raise RuntimeError("protected state changed")

    tx = ProtectedToyTransaction()
    def mutate_then_return_zero():
        tx.protected.add_(1.0)
        return _req(tx, "lti", 0, zero=True)

    with pytest.raises(RuntimeError, match="protected state changed"):
        batching.optimize_mixed_request_batch(tx, [mutate_then_return_zero])
    assert tx.counters["v13_failed"] is True
    assert tx.counters["adamw_steps"] == 0


def test_event_wrapper_preserves_request_order_owner_and_batched_updates():
    tx = ToyTransaction(); selected = [(k, i, "role") for i, k in enumerate(("gkd", "gkd", "lti") * 2)]
    seen = []
    def builder(kind, row, role, owner, current_tx, index):
        seen.append((kind, row, owner, index))
        return {**_req(current_tx, kind, 0.2 + index / 10), "metadata": {"ok": True}}
    result = run_batched_sleep_event(selected, tx, microbatch_size=3, objective_builder=builder)
    assert [(x[0], x[1], x[2]) for x in seen] == [
        ("gkd", 0, "teacher"), ("gkd", 1, "student"), ("lti", 2, "teacher"),
        ("gkd", 3, "teacher"), ("gkd", 4, "student"), ("lti", 5, "teacher")]
    assert result["opportunities"] == {"opportunities": 6, "gkd": 4, "lti": 2}
    assert result["physical_adamw_steps"] == 2 and len(result["request_summaries"]) == 6


def test_schedule_rejects_bad_order_before_builder_runs():
    touched = []
    with pytest.raises(ValueError):
        run_batched_sleep_event([("gkd", 0, "r"), ("lti", 1, "r"), ("gkd", 2, "r")],
                                ToyTransaction(), microbatch_size=3,
                                objective_builder=lambda *a: touched.append(1))
    assert touched == []


def test_failed_batch_rolls_back_and_closes_transaction():
    tx = ToyTransaction(); before = tx.student.weight.detach().clone()
    def bad():
        raise RuntimeError("fixture failure")
    with pytest.raises(RuntimeError):
        batching.optimize_mixed_request_batch(tx, [lambda: _req(tx, "gkd", 0.2), bad])
    assert torch.equal(before, tx.student.weight)
    assert tx.counters["closed"] and tx.counters["v13_failed"]
    assert tx.counters["v13_failed_batches"] == 1


class GuardBudget:
    def __init__(self, reject=False):
        self.reject = reject
        self.update_attempts = 0
        self.adamw_updates = 0

    def update(self):
        self.update_attempts += 1
        if self.reject:
            raise RuntimeError("hard update-attempt budget exceeded")


def test_update_budget_rejection_happens_before_optimizer_step():
    tx = ToyTransaction(); before = tx.student.weight.detach().clone()
    budget = GuardBudget(reject=True)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        batching.optimize_mixed_request_batch(
            tx, [lambda: _req(tx, "gkd", 0.1)], budget=budget)
    assert torch.equal(before, tx.student.weight)
    assert budget.update_attempts == 1 and budget.adamw_updates == 0
    assert tx.counters["adamw_steps"] == 0
    assert tx.counters["v13_optimizer_step_attempts"] == 1
    assert tx.counters["v13_physical_adamw_steps"] == 0
    assert tx.counters["v13_failed"] is True


def test_successful_optimizer_step_updates_budget_physical_count_once():
    tx = ToyTransaction(); budget = GuardBudget()
    result = batching.optimize_mixed_request_batch(
        tx, [lambda: _req(tx, "gkd", 0.1)], budget=budget)
    assert result["physical_adamw_steps"] == 1
    assert budget.update_attempts == 1 and budget.adamw_updates == 1
    assert tx.counters["v13_physical_adamw_steps"] == 1

