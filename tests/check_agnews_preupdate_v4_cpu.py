"""Run explicitly in a separate bounded pytest process; see v4 protocol.

The 233-update boundary fixture intentionally stays outside default test_*.py
collection so the ordinary suite retains its 256-update per-process guard.
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import agnews_preupdate_sleep_v4 as sleep_v4
from agnews_preupdate_sleep_v4 import consolidate
from nested_sleep.lifecycle import fingerprint
from nested_sleep.lm import NativeResearchLM
from nested_sleep.native import NativeConfig
from nested_sleep.optim import CMSController


def _toy():
    torch.manual_seed(81)
    model = NativeResearchLM(
        NativeConfig(vocab_size=9, dim=4, hidden=5, layers=1,
                     projection_chunk=1, memory_chunk=1,
                     cms_periods=(4, 16)),
        rank=1,
        capacity=3,
    )
    model.eval().requires_grad_(False)
    model.banks()[0].activate(0)
    model.banks()[1].activate(0)
    controller = CMSController(model, periods=(4, 16), base_lr=0.02, expert_lr=0.04)
    return model, controller


def _boundary(model, controller, slow_count):
    controller.counts = [4, slow_count]
    for level, descriptor in enumerate(model.level_descriptors()):
        for name in descriptor["base"]:
            controller.pending[level][name].fill_(0.01 * (level + 1))
    params = dict(model.named_parameters())
    proposals = {0: {name: params[name].detach().clone() + 0.002
                     for name in model.level_descriptors()[0]["base"]}}
    if slow_count == 16:
        proposals[1] = {name: params[name].detach().clone() - 0.003
                         for name in model.level_descriptors()[1]["base"]}
    return proposals


@pytest.mark.parametrize("slow_count, expected_counts", [(4, [0, 4]), (8, [0, 8]),
                                                           (12, [0, 12]), (16, [0, 0])])
def test_generalized_preupdate_boundaries_are_isolated_and_preserve_reserve(
        slow_count, expected_counts, monkeypatch):
    # CPU-only RNG capture shape, matching the production helper's CUDA state API.
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    model, controller = _toy()
    proposals = _boundary(model, controller, slow_count)
    pool = [{"tokens": [1, 2], "class_id": i % 4, "label": 3} for i in range(24)]
    original_model = fingerprint(model)
    original_controller = copy.deepcopy(controller.state_dict())
    slow_pending_before = {k: v.clone() for k, v in controller.pending[1].items()}

    committed, new_controller, summary = consolidate(
        model, controller, proposals, pool, event_seed=901, meter=type("Meter", (), {})(),
        stage="cpu-test",
    )

    assert fingerprint(model) == original_model
    assert controller.counts == original_controller["counts"]
    assert all(torch.equal(controller.pending[1][k], v) for k, v in slow_pending_before.items())
    assert new_controller.counts == expected_counts
    assert summary["scope"] == "pre_fast_base_update_sleep_v4"
    assert summary["total_adam_updates"] == 24
    assert summary["gkd_teacher_updates"] == summary["gkd_student_on_policy_updates"] == 8
    assert summary["input_slow_count"] == slow_count
    assert summary["slow_level1_applied_before_teacher"] is (slow_count == 16)
    assert (summary["pending_slow_hash_before"] == summary["pending_slow_hash_after"]
            if slow_count != 16 else summary["pending_slow_hash_before"] != summary["pending_slow_hash_after"])
    assert summary["receiver_slot"] == 1  # Existing slot 0 remains reserved and active.
    assert summary["receiver_active_slots"] == [0, 1]
    assert summary["receiver_route_policy"] == {"mix": 1, "fraction": 0.5}
    assert summary["active_slots_by_bank"] == [0, 2]
    assert summary["sender_reset_slot_ids"] == [0]
    assert len(list(committed.parameters())) == len(list(model.parameters()))


def test_teacher_order_switch_changes_only_candidate_seen_by_teacher(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    real_session = sleep_v4.SleepSession
    teacher_inputs = []
    initial_students=[]

    def recording_session(candidate, *args, **kwargs):
        teacher_inputs.append(fingerprint(candidate))
        session=real_session(candidate, *args, **kwargs)
        initial_students.append(fingerprint(session.student))
        return session

    monkeypatch.setattr(sleep_v4, "SleepSession", recording_session)
    pool = [{"tokens": [1, 2], "class_id": i % 4, "label": 3} for i in range(24)]
    fingerprints = []
    summaries = []
    for apply_fast_before_teacher in (False, True):
        model, controller = _toy()
        proposals = _boundary(model, controller, 8)
        before = fingerprint(model)
        committed, _, summary = consolidate(
            model, controller, proposals, pool, event_seed=902,
            meter=type("Meter", (), {})(), stage="order-test",
            apply_fast_before_teacher=apply_fast_before_teacher,
        )
        assert fingerprint(model) == before
        fingerprints.append(fingerprint(committed))
        summaries.append(summary)

    assert teacher_inputs[0] != teacher_inputs[1]
    assert initial_students[0]==initial_students[1]
    assert fingerprints[0] != fingerprints[1]
    assert [row["teacher_order"] for row in summaries] == ["pre_fast", "post_fast"]
    assert [row["fast_level0_applied_before_teacher"] for row in summaries] == [False, True]
    assert [row["output_controller_counts"] for row in summaries] == [[0, 8], [0, 8]]


def test_rejects_wrong_clock_or_pool_without_touching_live_state():
    model, controller = _toy()
    controller.counts = [3, 8]
    model_before = fingerprint(model)
    with pytest.raises(ValueError, match="unsupported pre-update clock"):
        consolidate(model, controller, {0: {}}, [], 1, type("Meter", (), {})(), "invalid")
    assert fingerprint(model) == model_before

def test_sequential_fast_events_preserve_slow_window_and_resume_nonboundary(monkeypatch):
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    model=NativeResearchLM(NativeConfig(vocab_size=9,dim=4,hidden=5,layers=1,projection_chunk=1,memory_chunk=1,cms_periods=(4,16)),rank=1,capacity=6)
    model.eval().requires_grad_(False);ctl=CMSController(model,(4,16),.02,.04)
    pool=[{'tokens':[1,2]}for _ in range(24)]
    def wake(m,c):
        c.prepare_wake(m);logits=m.logits(torch.tensor([[1,2,3]]));loss=torch.nn.functional.cross_entropy(logits[:,-1],torch.tensor([4]))
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters()if p.requires_grad],1.)
        return c.collect(m)
    checkpoint=None
    for step in range(1,17):
        proposals=wake(model,ctl)
        if step==9:
            branch=copy.deepcopy(checkpoint['model']);branch_ctl=CMSController.restore(branch,checkpoint['controller'])
            branch_proposals=wake(branch,branch_ctl)
            assert fingerprint(branch)==fingerprint(model)
            assert sleep_v4._hash_tree(branch_ctl.state_dict())==sleep_v4._hash_tree(ctl.state_dict())
            assert sleep_v4._hash_tree(branch_proposals)==sleep_v4._hash_tree(proposals)
        if step in(4,8,12):
            before=sleep_v4._pending_slow_hash(ctl)
            model,ctl,row=consolidate(model,ctl,proposals,pool,910+step,type('Meter',(),{})(),'sequence')
            assert ctl.counts==[0,step]and sleep_v4._pending_slow_hash(ctl)==before
        if step==8:checkpoint={'model':copy.deepcopy(model),'controller':copy.deepcopy(ctl.state_dict())}
        if step==16:
            assert set(proposals)=={0,1}and ctl.counts==[4,16]
            expected={k:v.clone()for k,v in proposals[1].items()}
            ctl.apply(model,1,proposals[1]);ctl.apply(model,0,proposals[0])
            assert ctl.counts==[0,0]and all(torch.count_nonzero(v)==0 for v in ctl.pending[1].values())
            assert all(torch.equal(dict(model.named_parameters())[k],v)for k,v in expected.items())
