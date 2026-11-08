from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import banking_cil_v13_batching as batching
from banking_sleep_v13_runtime import run_batched_sleep_event
from banking_v13_restore_qualification import make_b32_restore_qualification_callback


@dataclass
class FakePayload:
    cpu_rng: torch.Tensor
    cuda_rng: object
    python_rng: object
    numpy_rng: object


@dataclass
class FakeSnapshot:
    payload: FakePayload
    model: dict
    optimizer: dict
    generator_state: torch.Tensor
    counters: dict


class FakeTransaction:
    def __init__(self, live):
        self.live = live
        self.student = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.student.weight.fill_(0.25)
        self.allowed_names = {"weight"}
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=0.01, weight_decay=0.0)
        self.generator = torch.Generator(device="cpu").manual_seed(1234)
        self.clip_norm = 10.0
        self.counters = {"closed": False, "adamw_steps": 0, "backward_attempts": 0,
                         "tokens": 0, "gkd_calls": 0, "lti_calls": 0}
        self._v13_batch_active = False

    @classmethod
    def from_snapshot(cls, live, snapshot):
        tx = cls(live)
        tx.student.load_state_dict(snapshot.model)
        tx.optimizer.load_state_dict(snapshot.optimizer)
        tx.generator.set_state(snapshot.generator_state.clone())
        tx.counters = deepcopy(snapshot.counters)
        tx._restore_globals(snapshot.payload)
        return tx

    @staticmethod
    def _restore_globals(payload):
        torch.set_rng_state(payload.cpu_rng.clone())
        random.setstate(deepcopy(payload.python_rng))
        np.random.set_state(deepcopy(payload.numpy_rng))

    def snapshot(self):
        if self._v13_batch_active or any(p.grad is not None for p in self.student.parameters()):
            raise RuntimeError("unsafe fake snapshot")
        return FakeSnapshot(
            FakePayload(torch.get_rng_state().clone(), None, random.getstate(), deepcopy(np.random.get_state())),
            {k: v.detach().clone() for k, v in self.student.state_dict().items()},
            deepcopy(self.optimizer.state_dict()), self.generator.get_state().clone(),
            deepcopy(self.counters))

    def _check(self):
        assert torch.isfinite(self.student.weight).all()


class FakeBudget:
    def __init__(self):
        self.update_attempts = self.adamw_updates = self.backward_calls = 0

    def update(self):
        self.update_attempts += 1

    def backward(self, _mask):
        self.backward_calls += 1

    def as_dict(self):
        return {"update_attempts": self.update_attempts,
                "physical_adamw_updates": self.adamw_updates,
                "tail_backward_calls": self.backward_calls}


def _objective_factory(tx):
    def objective(kind, row, role, owner, transaction, index):
        assert transaction is tx
        assert owner in ("teacher", "student")
        # Private generator plus global RNGs make checkpoint replay verify
        # sampler state as well as model/Adam state without storing samples.
        draw = (torch.rand((), generator=tx.generator) + torch.rand(()) +
                torch.tensor(random.random()) + torch.tensor(float(np.random.random())))
        p = tx.student.weight.sum()
        return {"kind": kind, "loss": (p - draw) ** 2, "token_count": 1}
    return objective


def test_b32_restore_qualification_replays_serialized_32_to_64_and_restores_global_rng(monkeypatch):
    monkeypatch.setattr(batching, "BatchableAdditiveSleepTransactionV13", FakeTransaction)
    live = object()
    tx = FakeTransaction(live)
    budget = FakeBudget()
    selected = [(kind, index, "role")
                for index, kind in enumerate(("gkd", "gkd", "lti") * 21 + ("gkd",))]
    with tempfile.TemporaryDirectory(prefix=".v13-restore-test-", dir=ROOT) as tmp:
        qual_dir = Path(tmp) / "qualification"
        qualification = make_b32_restore_qualification_callback(
            selected=selected, tx=tx, objective_builder_factory=_objective_factory,
            budget=budget, output_dir=qual_dir, torch=torch, seed=17, arm="B32_HIGH")

        run_batched_sleep_event(selected, tx, microbatch_size=32,
                                objective_builder=_objective_factory(tx), budget=budget,
                                on_batch_committed=qualification)
        assert qualification.done
        receipt = __import__("json").loads((qual_dir / "restore-qualification.json").read_text())
        assert receipt["status"] == "pass"
        assert len(receipt["snapshot_file_sha256"]) == 64
        assert len(receipt["snapshot_content_sha256"]) == 64
        assert receipt["full_snapshot_content_equal"] is True
        assert receipt["batch_metrics_content_equal"] is True
        assert receipt["original_request_range"] == receipt["restored_request_range"] == [32, 64]
        assert receipt["extra_request_evaluations"] == 32
        assert receipt["extra_physical_adamw_steps"] == 1
        assert receipt["extra_work_delta"]["physical_adamw_updates"] == 1
        assert budget.update_attempts == budget.adamw_updates == 3
        assert tx.counters["adamw_steps"] == 2
        assert not (qual_dir / "failure.json").exists()


def test_qualification_is_restricted_to_target_seed_arm_and_fresh_output():
    selected = [(kind, index, "role")
                for index, kind in enumerate(("gkd", "gkd", "lti") * 21 + ("gkd",))]
    tx = FakeTransaction(object())
    with tempfile.TemporaryDirectory(prefix=".v13-restore-test-", dir=ROOT) as tmp:
        with pytest.raises(ValueError, match="seed 17 / B32_HIGH"):
            make_b32_restore_qualification_callback(
                selected=selected, tx=tx, objective_builder_factory=_objective_factory,
                budget=FakeBudget(), output_dir=Path(tmp) / "wrong", torch=torch,
                seed=29, arm="B32_HIGH")

