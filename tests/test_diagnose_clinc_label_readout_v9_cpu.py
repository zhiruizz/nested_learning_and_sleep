"""Small CPU-only regressions for the v9 readout diagnostic helpers."""
from __future__ import annotations

import sys
import hashlib
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import diagnose_clinc_label_readout_v9 as diag


def test_candidate_sequences_are_gold_independent_prefix_label_eos():
    candidates = diag.build_candidate_sequences([4, 5, 6], [[8], [9, 10]], eos_token_id=2)
    assert candidates == [[4, 5, 6, 8, 2], [4, 5, 6, 9, 10, 2]]
    with pytest.raises(ValueError, match="cannot be empty"):
        diag.build_candidate_sequences([1], [[]], eos_token_id=2)


def test_teacher_forced_scores_mask_prefix_and_include_eos():
    # Input targets at sequence positions 1..4. Prompt length=2 means only
    # positions 2,3,4 (label,label,EOS) contribute, not target position 1.
    ids = torch.tensor([[0, 1, 2, 3, 4]])
    logits = torch.zeros((1, 4, 6), dtype=torch.float64)
    logits[0, 0, 1] = 1.0  # prompt target: should not affect score
    logits[0, 1, 2] = 1.0
    logits[0, 2, 3] = 2.0
    logits[0, 3, 4] = 3.0  # EOS position is included
    sums, means = diag.teacher_forced_candidate_scores(logits, ids, [2], [3])
    lp = torch.log_softmax(logits.float(), dim=-1)
    expected = lp[0, 1, 2] + lp[0, 2, 3] + lp[0, 3, 4]
    assert torch.allclose(sums[0], expected)
    assert torch.allclose(means[0], expected / 3)
    assert not torch.allclose(sums[0], lp[0, 0, 1] + expected)


def test_per_candidate_length_mean_uses_its_own_answer_count():
    ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    logits = torch.zeros((2, 3, 5))
    logits[0, 1, 2] = 2.0
    logits[0, 2, 3] = 2.0
    logits[1, 1, 2] = 2.0
    logits[1, 2, 3] = 2.0
    sums, means = diag.teacher_forced_candidate_scores(logits, ids, [2, 2], [2, 1])
    assert torch.allclose(means[0], sums[0] / 2)
    assert torch.allclose(means[1], sums[1])


def test_argmax_ties_follow_canonical_label_order():
    assert diag.ordered_candidate_predictions(torch.tensor([[1.0, 1.0, 0.0]]),
                                              ["alpha", "beta", "gamma"]) == ["alpha"]


def test_frozen_base_logits_use_unaugmented_final_mlp_cache():
    class Identity(torch.nn.Module):
        def forward(self, value):
            return value

    head = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.]]))
    fake = SimpleNamespace(backbone=SimpleNamespace(
        model=SimpleNamespace(layers=[object()], norm=Identity()), lm_head=head))
    cache = SimpleNamespace(residual=torch.tensor([[[1., 2.]]]), y=torch.tensor([[[3., 4.]]]))
    actual = diag._base_logits_from_cache(fake, cache)
    assert torch.equal(actual, torch.tensor([[[4., 6., 10.]]]))


def test_fixed_development_cohort_uses_two_stable_clean_ids_per_class():
    order = list(range(150))
    train = [{"intent_id": 0, "text": "Same text"}]
    dev = []
    for cid in range(30):
        for suffix, text in (("z", f"clean {cid} z"), ("a", f"clean {cid} a"),
                             ("d", "Same text")):
            dev.append({"id": f"{cid:03d}{suffix}", "intent_id": cid, "text": text})
    rows = diag.fixed_clean_development_rows(train, dev, order)
    assert len(rows) == 60
    assert len({row["id"] for row in rows}) == 60
    assert all(row["id"].endswith(("a", "z")) for row in rows)
    assert {row["intent_id"] for row in rows} == set(range(30))


def _context_document():
    h = "a" * 64
    foundation = {"name": "TASK0_FOUNDATION", "kind": "checkpoint", "checkpoint_path": "f.pt",
                  "checkpoint_sha256": "b" * 64, "receipt_path": "f.json", "receipt_sha256": "c" * 64,
                  "expected_mode": "foundation", "expected_arm": None, "expected_tasks_completed": 1,
                  "expected_child_config_sha256": h, "expected_foundation_checkpoint_sha256": None}
    rows = [{"name": "FROZEN_BASE", "kind": "zero_adapter"}, foundation]
    for name, arm, child in (("HOPE_TASK1", "HOPE", "d" * 64),
                             ("SLEEP_TASK1", "HOPE_SLEEP_FIXEDMIX", "e" * 64)):
        rows.append({"name": name, "kind": "checkpoint", "checkpoint_path": f"{name}.pt",
                     "checkpoint_sha256": "f" * 64, "receipt_path": f"{name}.json", "receipt_sha256": "1" * 64,
                     "metrics_path": f"{name}.jsonl", "metrics_sha256": "2" * 64,
                     "expected_mode": "continuation", "expected_arm": arm, "expected_tasks_completed": 2,
                     "expected_child_config_sha256": child, "expected_foundation_checkpoint_sha256": "b" * 64})
    return {"schema": "clinc-v9-readout-contexts-v1", "seed": 17,
            "master_config_sha256": h, "training_manifest_sha256": h,
            "source_lock_sha256": h, "contexts": rows}


def test_context_map_binds_exact_four_roles_and_shared_foundation():
    doc = _context_document()
    result = diag.validate_context_map(doc, master_sha="a" * 64, manifest_sha="a" * 64,
                                       source_lock_sha="a" * 64)
    assert set(result) == set(diag.CONTEXT_NAMES)
    doc["contexts"][2]["expected_foundation_checkpoint_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="paired"):
        diag.validate_context_map(doc, master_sha="a" * 64, manifest_sha="a" * 64,
                                  source_lock_sha="a" * 64)


def _verified_checkpoint_record_fixture(tmp_path, monkeypatch):
    import run_clinc_cil_v8_full as full
    master_sha, manifest_sha, source_sha = "a" * 64, "b" * 64, "c" * 64
    foundation_child = full._child_config_sha(master_sha, "foundation", 17, None)
    hope_child = full._child_config_sha(master_sha, "continuation", 17, "HOPE")
    sleep_child = full._child_config_sha(master_sha, "continuation", 17, "HOPE_SLEEP_FIXEDMIX")
    foundation_bytes = b"foundation-safe-checkpoint"
    foundation_sha = hashlib.sha256(foundation_bytes).hexdigest()
    contexts = [{"name": "FROZEN_BASE", "kind": "zero_adapter"}]

    def add_checkpoint(name, mode, arm, child, payload, receipt_extra, *, metrics=False):
        cp = tmp_path / f"{name}.pt"
        cp.write_bytes(payload)
        cp_sha = hashlib.sha256(payload).hexdigest()
        receipt = {"seed": 17, "master_config_sha256": master_sha,
                   "training_manifest_sha256": manifest_sha, "source_lock_sha256": source_sha,
                   "child_config_sha256": child, "status": "complete", **receipt_extra}
        receipt_path = tmp_path / f"{name}.receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        row = {"name": name, "kind": "checkpoint", "checkpoint_path": str(cp),
               "checkpoint_sha256": cp_sha, "receipt_path": str(receipt_path),
               "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
               "expected_mode": mode, "expected_arm": arm,
               "expected_tasks_completed": 1 if mode == "foundation" else 2,
               "expected_child_config_sha256": child,
               "expected_foundation_checkpoint_sha256": None if mode == "foundation" else foundation_sha}
        if metrics:
            metrics_path = tmp_path / f"{name}.metrics.jsonl"
            metrics_path.write_text(json.dumps({"event": "safe_checkpoint", "task_id": 1, "step": 188,
                                                "path": "checkpoints\\task01-next0189-safe.pt",
                                                "sha256": cp_sha}) + "\n", encoding="utf-8")
            row.update(metrics_path=str(metrics_path),
                       metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest())
        contexts.append(row)
        return cp_sha

    add_checkpoint("TASK0_FOUNDATION", "foundation", None, foundation_child, foundation_bytes,
                   {"schema": "clinc-cil-v8-common-foundation-receipt", "acquisition_qualified": True,
                    "test_data_included": False, "tasks_completed": 1, "final_checkpoint_sha256": foundation_sha})
    for name, arm, child in (("HOPE_TASK1", "HOPE", hope_child),
                             ("SLEEP_TASK1", "HOPE_SLEEP_FIXEDMIX", sleep_child)):
        add_checkpoint(name, "continuation", arm, child, name.encode(),
                       {"schema": "clinc-cil-v8-full-run-receipt", "test_data_included": False,
                        "arm": arm, "tasks_completed": 10, "foundation_checkpoint_sha256": foundation_sha,
                        "task_metrics": [{"task_id": 1}]}, metrics=True)
    document = {"schema": "clinc-v9-readout-contexts-v1", "seed": 17,
                "master_config_sha256": master_sha, "training_manifest_sha256": manifest_sha,
                "source_lock_sha256": source_sha, "contexts": contexts}
    checked_map = diag.validate_context_map(document, master_sha=master_sha,
                                             manifest_sha=manifest_sha, source_lock_sha=source_sha)
    validated = diag._read_context_records(checked_map, project_root=tmp_path, master={},
                                           master_sha=master_sha, manifest_sha=manifest_sha,
                                           source_lock_sha=source_sha)

    class FakeContext:
        def __init__(self): self.p = torch.nn.Parameter(torch.ones(1))
        def parameters(self): return iter((self.p,))
        def banks(self): return []
        def load_state_dict(self, state): self.p.data.copy_(state["value"])

    import clinc_tail_runtime_v8
    clone_calls = []
    monkeypatch.setattr(clinc_tail_runtime_v8, "clone_mutable_context",
                        lambda _model: (clone_calls.append(True), FakeContext())[1])
    seen = {}
    def fake_safe_load(path, *, torch, expected):
        seen[Path(path).name] = expected["checkpoint_sha256"]
        task_id = 0 if expected["mode"] == "foundation" else 1
        task_count = 1 if task_id == 0 else 2
        cursor = full.checkpoint_cursor(mode=expected["mode"], task_id=task_id, next_step=189,
                                        task_count=task_count, schedule_sha256="3" * 64,
                                        replay_ids_sha256="4" * 64)
        return {"mode": expected["mode"], "cursor": cursor, "context": {"value": torch.tensor([0.])}}
    monkeypatch.setattr(full, "_load_safe_checkpoint", fake_safe_load)
    preflight = diag._preflight_checkpoint_payloads(validated, torch=torch)
    assert len(preflight) == 3 and clone_calls == []
    for name, payload in preflight.items():
        validated[name]["preflight_payload"] = payload
    loaded = diag._load_context_states(object(), validated, torch=torch)
    assert set(loaded) == set(diag.CONTEXT_NAMES)
    assert seen == {f"{name}.pt": row["expected"]["checkpoint_sha256"] for name, row in validated.items()
                    if row is not None}


def test_verified_checkpoint_record_passes_hash_binding_to_deserializer(monkeypatch):
    # Keep temp artifacts inside the writable workspace instead of relying on
    # machine-global pytest temp policy.
    with tempfile.TemporaryDirectory(prefix="clinc-v9-context-fixture-", dir=ROOT) as tmp:
        _verified_checkpoint_record_fixture(Path(tmp), monkeypatch)

