from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from banking_sleep_v13_runtime import run_batched_sleep_event


class ToyTransaction:
    def __init__(self):
        self.student = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.student.weight.fill_(0.25)
        self.allowed_names = {"weight"}
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=0.01, weight_decay=0.0)
        self.clip_norm = 100.0
        self.counters = {"closed": False, "adamw_steps": 0, "backward_attempts": 0,
                         "tokens": 0, "gkd_calls": 0, "lti_calls": 0}
        self._v13_batch_active = False

    def _check(self):
        assert torch.isfinite(self.student.weight).all()

    def snapshot(self):
        if self._v13_batch_active:
            raise RuntimeError("snapshot during active microbatch")
        if any(p.grad is not None for p in self.student.parameters()):
            raise RuntimeError("snapshot with pending gradient")
        return self.student.weight.detach().clone()


def _builder(seen):
    def build(kind, row, role, owner, tx, index):
        seen.append((index, kind, owner))
        p = tx.student.weight.sum()
        loss = (p - (0.1 + index / 100)) ** 2
        return {"kind": kind, "loss": loss, "token_count": 2}
    return build


def test_resume_at_committed_boundary_preserves_owner_cursor_and_snapshot_safety():
    selected = [(kind, i, "role") for i, kind in enumerate(("gkd", "gkd", "lti") * 2)]
    tx = ToyTransaction()
    seen_first, callbacks = [], []

    def checkpoint(batch_index, next_offset, next_gkd_ordinal, result):
        # This is the driver's safe snapshot point: after Adam/zero_grad and
        # after the transaction's active-microbatch guard has been cleared.
        callbacks.append((batch_index, next_offset, next_gkd_ordinal,
                          tx.snapshot(), result["physical_adamw_steps"]))

    first = run_batched_sleep_event(
        selected, tx, microbatch_size=3, objective_builder=_builder(seen_first),
        on_batch_committed=checkpoint)
    assert first["physical_adamw_steps"] == 2
    assert [(x[0], x[1], x[2]) for x in seen_first] == [
        (0, "gkd", "teacher"), (1, "gkd", "student"), (2, "lti", "teacher"),
        (3, "gkd", "teacher"), (4, "gkd", "student"), (5, "lti", "teacher")]
    assert callbacks[0][0:3] == (0, 3, 2)
    assert callbacks[1][0:3] == (1, 6, 4)

    resumed, seen_second = ToyTransaction(), []
    resumed.student.load_state_dict(tx.student.state_dict())
    resumed.optimizer.load_state_dict(tx.optimizer.state_dict())
    suffix = run_batched_sleep_event(
        selected, resumed, microbatch_size=3, objective_builder=_builder(seen_second),
        request_offset=3, gkd_ordinal_start=2)
    assert [(x[0], x[1], x[2]) for x in seen_second] == [
        (3, "gkd", "teacher"), (4, "gkd", "student"), (5, "lti", "teacher")]
    assert suffix["opportunities"] == {"opportunities": 6, "gkd": 4, "lti": 2}
    assert suffix["microbatch_count"] == 1 and suffix["physical_adamw_steps"] == 1


def test_resume_rejects_wrong_gkd_ordinal_before_builder():
    selected = [(kind, i, "role") for i, kind in enumerate(("gkd", "gkd", "lti") * 2)]
    seen = []
    with pytest.raises(ValueError, match="gkd_ordinal_start"):
        run_batched_sleep_event(selected, ToyTransaction(), microbatch_size=3,
                                objective_builder=_builder(seen), request_offset=3,
                                gkd_ordinal_start=3)
    assert not seen


def test_resume_rejects_nonboundary_offset_before_builder():
    selected = [(kind, i, "role") for i, kind in enumerate(("gkd", "gkd", "lti") * 2)]
    seen = []
    with pytest.raises(ValueError, match="microbatch boundary"):
        run_batched_sleep_event(selected, ToyTransaction(), microbatch_size=2,
                                objective_builder=_builder(seen), request_offset=3)
    assert not seen


def test_resume_at_offset_32_preserves_partial_cycle_phase():
    selected = [(kind, i, "role") for i, kind in enumerate(("gkd", "gkd", "lti") * 21 + ("gkd",))]
    seen = []
    result = run_batched_sleep_event(
        selected, ToyTransaction(), microbatch_size=32, objective_builder=_builder(seen),
        request_offset=32)
    assert result["processed_opportunities"] == 32
    assert result["request_offset_start"] == 32
    assert result["gkd_ordinal_start"] == 22
    assert result["microbatch_count"] == 1
    assert seen[0] == (32, "lti", "teacher")
    # Global GKD ordinals 22 and 23 retain teacher/student alternation.
    assert seen[1:3] == [(33, "gkd", "teacher"), (34, "gkd", "student")]
    assert seen[-1] == (63, "gkd", "teacher")


def test_checkpoint_callback_failure_closes_committed_transaction():
    selected = [(kind, i, "role") for i, kind in enumerate(("gkd", "gkd", "lti"))]
    tx = ToyTransaction()
    def fail_checkpoint(*_):
        raise OSError("fixture checkpoint failure")
    with pytest.raises(OSError, match="checkpoint failure"):
        run_batched_sleep_event(selected, tx, microbatch_size=3,
                                objective_builder=_builder([]),
                                on_batch_committed=fail_checkpoint)
    assert tx.counters["adamw_steps"] == 1
    assert tx.counters["closed"] and tx.counters["v13_failed"]
