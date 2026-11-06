"""Explicit CPU-only fixtures for the v5 anchored-wake hypothesis helper."""
from copy import deepcopy
from pathlib import Path
import pytest
pytest.importorskip('transformers')
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import agnews_anchored_wake_v5 as anchored
from nested_sleep.lm import NativeResearchLM, execution_state
from nested_sleep.native import NativeConfig
from nested_sleep.optim import CMSController


@pytest.fixture(autouse=True)
def cpu_rng_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _state: None)


def _model(seed=7):
    torch.manual_seed(seed)
    m = NativeResearchLM(
        NativeConfig(vocab_size=32, dim=8, hidden=12, layers=1, cms_periods=(4, 16)),
        rank=2,
        capacity=6,
    ).eval()
    m.requires_grad_(False)
    m.banks()[0].activate(0)
    m.banks()[1].activate(0)
    return m


def _rows(classes, *, label_shift=0, token_shift=0):
    return [
        {
            "tokens": [1, 3 + token_shift, 5 + i],
            "class_id": int(cls),
            "marker": 10,
            "label": 20 + ((int(cls) + label_shift) % 4),
        }
        for i, cls in enumerate(classes)
    ]


def _step(m, ctl, teacher, new_rows, old_rows, *, defer=True, mode="soft_kd"):
    return anchored.anchored_step(
        m, ctl, new_rows, old_rows, 0, anchored.c.engine.WorkMeter(), "cpu-fixture",
        teacher, mode=mode, defer_due=defer,
    )


def test_soft_kd_ignores_old_gold_labels_and_preserves_teacher():
    m1 = _model(); m2 = deepcopy(m1)
    t1 = deepcopy(m1).eval().requires_grad_(False)
    t2 = deepcopy(m2).eval().requires_grad_(False)
    c1 = CMSController(m1, (4, 16), .02, .04)
    c2 = CMSController(m2, (4, 16), .02, .04)
    new = _rows([2, 3])
    old1 = _rows([0, 0, 1, 1], label_shift=0)
    old2 = _rows([0, 0, 1, 1], label_shift=2)
    state1 = anchored.c.engine.shared.tensor_tree_hash({"m": m1.state_dict(), "c": c1.state_dict()})
    state2 = anchored.c.engine.shared.tensor_tree_hash({"m": m2.state_dict(), "c": c2.state_dict()})
    assert state1 == state2

    r1, p1 = _step(m1, c1, t1, new, old1)
    r2, p2 = _step(m2, c2, t2, new, old2)
    assert r1["old_anchor_loss"] == pytest.approx(r2["old_anchor_loss"], abs=0., rel=0.)
    assert r1["new_ce"] == pytest.approx(r2["new_ce"], abs=0., rel=0.)
    assert anchored.c.engine.shared.tensor_tree_hash(m1.state_dict()) == anchored.c.engine.shared.tensor_tree_hash(m2.state_dict())
    assert anchored.c.engine.shared.tensor_tree_hash(c1.state_dict()) == anchored.c.engine.shared.tensor_tree_hash(c2.state_dict())
    assert anchored.c.engine.shared.tensor_tree_hash(p1) == anchored.c.engine.shared.tensor_tree_hash(p2)
    assert r1["teacher_unchanged"] and r2["teacher_unchanged"]
    assert r1["global_rng_unchanged"] and r2["global_rng_unchanged"]
    assert not any(p.requires_grad for p in t1.parameters())
    assert execution_state(t1) == execution_state(t2)


def test_soft_kd_zero_when_teacher_equals_student_before_collect():
    m = _model(); teacher = deepcopy(m).eval().requires_grad_(False)
    ctl = CMSController(m, (4, 16), .02, .04)
    result, _ = _step(m, ctl, teacher, _rows([2, 3]), _rows([0, 0, 1, 1]))
    assert abs(result["old_anchor_loss"]) < 1e-7
    assert result["logical_primary_updates"] == 1


def test_zero_output_fast_activation_preserves_soft_kd_target():
    teacher = _model().eval().requires_grad_(False)
    student = deepcopy(teacher).eval()
    student.banks()[0].deactivate(0)  # Its zero-initialized B makes this functionally inert.
    ctl = CMSController(student, (4, 16), .02, .04)
    result, _ = _step(student, ctl, teacher, _rows([2, 3]), _rows([0, 0, 1, 1]))
    assert abs(result["old_anchor_loss"]) < 1e-7
    assert int(student.banks()[0].slots[0].active) == 1  # prepare_wake activation


def test_gold_mode_uses_old_labels_and_soft_mode_rejects_unknown_modes():
    m = _model(); teacher = deepcopy(m).eval().requires_grad_(False)
    ctl = CMSController(m, (4, 16), .02, .04)
    result, _ = _step(m, ctl, teacher, _rows([2, 3]), _rows([0, 0, 1, 1]), mode="gold")
    assert result["mode"] == "gold"
    assert result["old_anchor_loss"] > 0
    with pytest.raises(ValueError, match="mode must"):
        _step(m, ctl, teacher, _rows([2]), _rows([0, 0, 1, 1]), mode="teacher_logits")


def test_deferred_due_proposals_preserve_slow_pending_until_step16():
    m = _model(); teacher = deepcopy(m).eval().requires_grad_(False)
    ctl = CMSController(m, (4, 16), .02, .04)
    new, old = _rows([2, 3]), _rows([0, 0, 1, 1])
    for step in range(1, 17):
        result, proposals = _step(m, ctl, teacher, new, old)
        if step in (4, 8, 12):
            assert result["due_levels"] == [0]
            assert ctl.counts == [4, step]
            slow_pending = {k: v.clone() for k, v in ctl.pending[1].items()}
            ctl.apply(m, 0, proposals[0])
            assert ctl.counts == [0, step]
            assert all(torch.equal(ctl.pending[1][k], slow_pending[k]) for k in slow_pending)
        elif step == 16:
            assert result["due_levels"] == [0, 1]
            assert ctl.counts == [4, 16]
            ctl.apply(m, 1, proposals[1])
            ctl.apply(m, 0, proposals[0])
            assert ctl.counts == [0, 0]
        else:
            assert result["due_levels"] == []
            assert ctl.counts == [step % 4, step]
