"""Small CPU-only checks for the controlled CLINC probe's fixed contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools import run_clinc_controlled_probe_v8 as probe
from tools import run_clinc_cil_v8_r2 as r2


def test_task1_due_order_reuses_receiver_capacity_before_next_hierarchy_boundary():
    result = probe._simulate_due_slot_peaks(
        [1, 1, 1, 1, 0], {0: 12, 1: 60, 2: 188, 3: 188},
        (16, 64, 256, 1024, None), 188, 8)
    assert result["event_counts_by_sender"] == {"0": 12, "1": 3, "2": 1, "3": 0}
    assert len(result["events"]) == 16
    assert result["active_slot_peaks_by_level"] == [1, 4, 3, 2, 0]
    same_tick = [event["sender_level"] for event in result["events"] if event["task_tick"] == 4]
    assert same_tick == [0, 1]
    assert max(result["active_slot_peaks_by_level"]) <= 8


def test_causal_route_coverage_uses_logits_t_to_target_t_plus_one():
    selected = torch.tensor([[1, 0, 1, 0]])
    answer_mask = torch.tensor([[0, 1, 0, 0]])
    stats = probe._aligned_route_coverage(selected, answer_mask, receiver_slot=1)
    assert stats["causal_answer_tokens"] == 1
    assert stats["natural_top1_receiver_tokens"] == 1
    assert stats["natural_top1_receiver_coverage"] == 1.0


def test_pre_and_post_sleep_checkpoint_names_are_unique_for_same_tick_events(tmp_path):
    pre0 = probe._phase_checkpoint_path(tmp_path, "SLEEP_MIX", 4, "pre-sleep-level0-event000")
    post0 = probe._phase_checkpoint_path(tmp_path, "SLEEP_MIX", 4, "post-sleep-level0-event000")
    pre1 = probe._phase_checkpoint_path(tmp_path, "SLEEP_MIX", 4, "pre-sleep-level1-event001")
    post1 = probe._phase_checkpoint_path(tmp_path, "SLEEP_MIX", 4, "post-sleep-level1-event001")
    assert len({pre0, post0, pre1, post1}) == 4
    for index, path in enumerate((pre0, post0, pre1, post1)):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"state-{index}".encode()
        path.write_bytes(payload)
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(payload).digest()


def test_paired_selector_balances_roles_covers_classes_and_matches_arms():
    cfg = {"gkd_calls_per_event": 2, "lti_calls_per_event": 1}
    old = [{"id": f"old-{cls}-{ex}", "intent_id": cls}
           for cls in range(15) for ex in range(4)]
    current = [{"id": f"new-{cls}-{ex}", "intent_id": cls + 15}
               for cls in range(15) for ex in range(4)]
    row_pool = {"old": old, "current": current}
    sleep = [item for event in range(16) for item in probe._select_event_prompts(
        row_pool, event=event, task_id=1, cfg=cfg, r2=r2)]
    gold = [item for event in range(16) for item in probe._select_event_prompts(
        row_pool, event=event, task_id=1, cfg=cfg, r2=r2)]
    assert [kind for kind, _ in sleep[:3]] == ["gkd", "gkd", "lti"]
    assert [row["id"] for _, row in sleep] == [row["id"] for _, row in gold]
    assert len({row["id"] for _, row in sleep}) == 48
    for role in ("old", "current"):
        selected = []
        for global_call, (_kind, row) in enumerate(sleep):
            meta = probe._prompt_selection_meta(event=global_call // 3,
                call=global_call % 3, task_id=1)
            if meta["role"] == role:
                selected.append((row, meta))
        assert len(selected) == 24
        assert {meta["class_index"] for _, meta in selected} == set(range(15))
        assert all(row["intent_id"] == (meta["class_index"] if role == "old"
                    else meta["class_index"] + 15) for row, meta in selected)


def test_gold_control_executes_three_optimizer_calls_on_paired_prompts(monkeypatch):
    class FakeBudget:
        def __init__(self):
            self.backward_calls = 0
            self.update_calls = 0

        def backward(self, _tokens):
            self.backward_calls += 1

        def update(self):
            self.update_calls += 1

    class FakeContext:
        def __init__(self):
            self.weight = nn.Parameter(torch.tensor(1.0))

    class FakeTx:
        def __init__(self):
            self.context = FakeContext()
            self.receiver = 1
            self.slot = 2
            self.allowed_names = {"weight"}
            self.counters = {"zero_gradient_steps": 0, "adamw_steps": 0}

        def student_context(self):
            return self.context

        def optimize(self, loss, _kind, _tokens):
            loss.backward()
            with torch.no_grad():
                self.context.weight.add_(self.context.weight.grad, alpha=-0.1)
                self.context.weight.grad = None
            self.counters["adamw_steps"] += 1

    old_pool = [{"id": f"old-{cls}-{ex}", "intent_id": cls, "text": f"old-{cls}-{ex}"}
               for cls in range(15) for ex in range(4)]
    current_pool = [{"id": f"new-{cls}-{ex}", "intent_id": cls + 15, "text": f"new-{cls}-{ex}"}
                   for cls in range(15) for ex in range(4)]
    cfg = {"gkd_calls_per_event": 2, "lti_calls_per_event": 1}
    tx, budget, last_event = FakeTx(), FakeBudget(), {}

    def fake_batch(_rows, _tokenizer, _view, _cfg, _torch, _device):
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        mask = torch.ones_like(ids)
        labels = torch.tensor([[-100, 2, 3]], dtype=torch.long)
        return {}, ids, mask, labels

    def fake_capture(_model, ids, _mask, device=None):
        return SimpleNamespace(ids=ids)

    fake_runtime = SimpleNamespace(capture=fake_capture, causal_token_ce=r2.__dict__.get("_unused", None))
    from tools import clinc_tail_runtime_v8 as real_runtime
    fake_runtime.causal_token_ce = real_runtime.causal_token_ce

    def fake_audited(_model, context, cache, _level, _slot, _answer_mask, _runtime):
        template = torch.stack((context.weight, -context.weight, context.weight * 0.5,
                                context.weight * 0.0))
        logits = template.view(1, 1, 4).expand(1, cache.ids.shape[1], 4)
        return logits, {"natural_top1_receiver_tokens": 0,
                        "causal_answer_tokens": 2,
                        "natural_top1_receiver_coverage": 0.0,
                        "controlled_mix_fraction": 0.5,
                        "fresh_branch_executed_answer_coverage": 1.0}

    monkeypatch.setitem(sys.modules, "clinc_tail_runtime_v8", fake_runtime)
    monkeypatch.setattr(r2, "_batch", fake_batch)
    monkeypatch.setattr(probe, "_audited_student_logits", fake_audited)
    result = probe._run_gold_ce_event(None, tx, old_pool, current_pool, None, None, cfg, torch, "cpu", budget,
                                      task_id=1, event=4, r2=r2, last_event=last_event)
    assert len(result) == 3
    assert tx.counters["adamw_steps"] == 3
    assert budget.backward_calls == budget.update_calls == 3
    assert [item["paired_sleep_request_kind"] for item in result] == ["gkd", "gkd", "lti"]
    assert all(item["uses_gold_intent"] for item in result)
    assert tx.context.weight.item() < 1.0
    assert all(item["same_sample_fit_after"] < item["same_sample_fit_before"] for item in result)


def test_local_warmup_preflight_binds_source_paths_and_all_resume_loader_kwargs(tmp_path, monkeypatch):
    tools_dir = Path(probe.__file__).resolve().parent
    project_root = tools_dir.parent
    cfg = json.loads((project_root / "configs" / "clinc_cil_v8_hope_task1_resume_r2.json").read_text(encoding="utf-8"))
    cfg = r2.validate_config(cfg)
    parent_dir = tmp_path / "r1"
    parent_dir.mkdir()
    q_path = tmp_path / "rescore.json"
    captured = {}

    def fake_source_hashes(_project, task_root):
        assert Path(task_root).resolve() == project_root.resolve()
        return {"fixture-source": "a" * 64}

    def fake_load(parent, qualification, **kwargs):
        captured.update(kwargs)
        assert parent == parent_dir and qualification == q_path
        return {"arms": {"HOPE": {"fixture": True}}}

    fake_r2 = SimpleNamespace(__file__=str(tools_dir / "run_clinc_cil_v8_r2.py"),
                              _source_hashes=fake_source_hashes,
                              _load_resume_inputs=fake_load)
    q_path.write_text("{}", encoding="utf-8")
    info, aggregate_sha, file_sha = probe._load_warmup_reference(parent_dir, q_path,
        cfg=cfg, project_root=project_root, tools_dir=tools_dir,
        manifest_source_sha256="f" * 64, training_manifest_sha256=cfg["training_manifest_sha256"],
        r2=fake_r2)
    assert info["arms"]["HOPE"]["fixture"] is True
    assert aggregate_sha == probe._canonical_sha({"fixture-source": "a" * 64})
    assert file_sha == probe._sha(tools_dir / "run_clinc_cil_v8_r2.py")
    assert set(captured) == {"cfg", "current_source_lock_sha256", "current_r2_source_sha256",
                             "training_manifest_sha256", "manifest_source_sha256"}
    assert captured["current_r2_source_sha256"] == file_sha
    assert captured["manifest_source_sha256"] == "f" * 64
