"""CPU-only tests for opt-in Sleep receiver routing and checkpoint policy state."""
from __future__ import annotations

import hashlib
import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nested_sleep.experts import SparseExperts
from nested_sleep.lifecycle import SleepSession, fingerprint
from nested_sleep.lm import NativeResearchLM, execution_state
from nested_sleep.native import NativeConfig


@pytest.fixture(autouse=True)
def force_cpu_rng_path(monkeypatch):
    """The repository conftest forbids any CUDA lazy initialization in CPU tests."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def tree_hash(value) -> str:
    h = hashlib.sha256()
    def visit(obj, key=""):
        h.update(key.encode())
        if torch.is_tensor(obj):
            tensor = obj.detach().cpu().contiguous()
            h.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            for name in sorted(obj, key=str):
                visit(obj[name], f"{key}/{name}")
        elif isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj):
                visit(item, f"{key}/{index}")
        else:
            h.update(repr(obj).encode())
    visit(value)
    return h.hexdigest()


def test_natural_top1_can_miss_loss_positions_but_opt_in_mix_routes_gradient():
    router = SparseExperts(dim=2, rank=1, capacity=2)
    old_slot = router.activate()
    new_slot = router.activate()
    with torch.no_grad():
        router.slots[old_slot].route.copy_(torch.tensor([1.0, 0.0]))
        router.slots[new_slot].route.copy_(torch.tensor([-1.0, 0.0]))
        for slot in router.slots:
            slot.a.fill_(0.5)
            slot.b.fill_(0.2)
    # Nine prompt positions route to the new slot. The two answer/loss
    # positions route only to the previous slot, so receiver gradients exist
    # as tensors but are exactly zero at the loss positions.
    hidden = torch.tensor([[[-10.0, 0.0]] * 9 + [[10.0, 0.0], [10.0, 0.0]]])
    router.audit_routes = True
    natural = router(hidden)
    selected = router.last_route_audit["selected"][0].tolist()
    assert selected.count(new_slot) == 9 and selected[-2:] == [old_slot, old_slot]
    natural[:, -2:].sum().backward()
    assert router.slots[new_slot].a.grad is not None
    assert torch.count_nonzero(router.slots[new_slot].a.grad).item() == 0
    assert router.slots[new_slot].b.grad is not None
    assert torch.count_nonzero(router.slots[new_slot].b.grad).item() == 0

    router.zero_grad(set_to_none=True)
    router.route_override = {"mix": new_slot, "fraction": 0.5}
    mixed = router(hidden)
    mixed[:, -2:].sum().backward()
    assert float(router.slots[new_slot].a.grad.norm()) > 0.0
    assert float(router.slots[new_slot].b.grad.norm()) > 0.0


def _tiny_model():
    return NativeResearchLM(
        NativeConfig(vocab_size=32, dim=16, hidden=24, layers=1, cms_periods=(4, 16)),
        rank=2, capacity=4,
    )


def _prospective_fast_update(model):
    parameters = dict(model.named_parameters())
    names = model.level_descriptors()[0]["base"]
    return {name: parameters[name].detach().clone() + 0.25 for name in names}


def _run_one_sleep_event(live, event, *, branch_restore_check):
    old_fingerprint = fingerprint(live)
    old_execution = execution_state(live)
    prospective = _prospective_fast_update(live)
    session = SleepSession(live, 0, prospective, seed=9000 + event, lr=.003,
                           dream_tokens=2, reject_zero_gradient=True,
                           receiver_mix_fraction=.5)
    expected_slot = event - 1
    assert session.slot == expected_slot
    assert session.teacher.banks()[1].route_override == old_execution["routing"][1]
    assert session.student.banks()[1].route_override == {"mix": expected_slot, "fraction": .5}
    prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    session.seed_step(prefix, owner=None)
    assert session.records[-1]["grad_norm"] > 0.0

    if branch_restore_check:
        snapshot = session.state_dict()
        before_hash = tree_hash(snapshot)
        tampered = deepcopy(snapshot)
        tampered["student_execution_state"]["routing"][1] = {"mix": expected_slot, "fraction": .75}
        with pytest.raises(ValueError, match="does not match the saved receiver mix"):
            SleepSession.restore(live, tampered)
        tampered_live = deepcopy(snapshot)
        tampered_live["live_execution_state"]["routing"][0] = {"force": 0}
        with pytest.raises(ValueError, match="policy snapshot"):
            SleepSession.restore(live, tampered_live)
        branch_outputs = []
        for _ in range(2):
            branch = SleepSession.restore(live, snapshot)
            assert branch.receiver_mix_fraction == .5
            assert branch.teacher.banks()[1].route_override == old_execution["routing"][1]
            assert branch.student.banks()[1].route_override == {"mix": expected_slot, "fraction": .5}
            branch.seed_step(prefix, owner=None)
            branch.lti_step(prefix)
            branch_outputs.append({"state": tree_hash(branch.state_dict()),
                                   "teacher": fingerprint(branch.teacher),
                                   "student": fingerprint(branch.student),
                                   "teacher_policy": execution_state(branch.teacher),
                                   "student_policy": execution_state(branch.student)})
        assert branch_outputs[0] == branch_outputs[1]
        assert tree_hash(snapshot) == before_hash

    # Complete the pre-registered 16 GKD + 8 LTI local transaction.
    for _ in range(15):
        session.seed_step(prefix, owner=None)
    for _ in range(8):
        session.lti_step(prefix)
    assert len(session.records) == 24
    assert all(record["grad_norm"] > 0 for record in session.records)
    assert all(record["kind"] != "dreaming" for record in session.records)
    session.finish_consolidation()
    committed = session.commit()
    assert fingerprint(live) == old_fingerprint
    assert execution_state(live) == old_execution
    assert committed.banks()[1].route_override == {"mix": expected_slot, "fraction": .5}
    assert sum(bool(slot.active) for slot in committed.banks()[1].slots) == event
    assert sum(bool(slot.active) for slot in live.banks()[1].slots) == event - 1
    return committed


def test_two_sleep_events_keep_old_slots_and_commit_receiver_mix_with_dual_restore():
    torch.manual_seed(20261105)
    live = _tiny_model()
    first = _run_one_sleep_event(live, 1, branch_restore_check=False)
    assert first.banks()[1].route_override == {"mix": 0, "fraction": .5}

    # Simulate a prior natural router policy that selects the retained slot.
    # The next teacher must preserve this policy; only the student mixes in its
    # newly allocated receiver slot 1.
    first.banks()[1].route_override = {"force": 0}
    second = _run_one_sleep_event(first, 2, branch_restore_check=True)
    assert [bool(slot.active) for slot in second.banks()[1].slots] == [True, True, False, False]
    assert second.banks()[1].route_override == {"mix": 1, "fraction": .5}


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_receiver_mix_fraction_is_rejected(fraction):
    live = _tiny_model()
    prospective = _prospective_fast_update(live)
    with pytest.raises(ValueError, match="receiver_mix_fraction"):
        SleepSession(live, 0, prospective, receiver_mix_fraction=fraction)


def test_legacy_schema_one_natural_route_checkpoint_still_restores():
    live = _tiny_model()
    session = SleepSession(live, 0, _prospective_fast_update(live), seed=51,
                           dream_tokens=2, reject_zero_gradient=True)
    legacy = deepcopy(session.state_dict())
    legacy.pop("teacher_execution_state")
    legacy.pop("student_execution_state")
    legacy.pop("live_execution_state")
    legacy.pop("receiver_mix_fraction")
    restored = SleepSession.restore(live, legacy)
    assert restored.receiver_mix_fraction is None
    assert execution_state(restored.teacher) == execution_state(live)
    assert execution_state(restored.student) == execution_state(live)


@pytest.mark.parametrize("owner", ["live", "teacher", "student"])
def test_commit_rejects_untracked_execution_policy_mutation(owner):
    live = _tiny_model()
    session = SleepSession(live, 0, _prospective_fast_update(live), seed=61,
                           dream_tokens=2, receiver_mix_fraction=.5)
    target = {"live": session.live, "teacher": session.teacher, "student": session.student}[owner]
    target.banks()[1].route_override = {"force": 0}
    session.finish_consolidation()
    with pytest.raises(RuntimeError, match="execution policy changed"):
        session.commit()
