"""CPU-only gates/cohort tests for the v11 development comparison."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_banking_lti_ablation_v11 as ev

H = "a" * 64


def _reference():
    contexts = []
    foundation = {"receipt_sha256": "b" * 64, "checkpoint_sha256": "c" * 64}
    for seed in ev.SEEDS:
        for arm in ev.ARMS:
            family = "v10" if arm in ev.V10_ARMS else "v11"
            contexts.append({
                "seed": seed, "family": family, "arm": arm,
                "run_dir": f"{family}/seed{seed}/{arm}",
                "protocol_path": f"{family}/seed{seed}/{arm}/protocol.json",
                "protocol_sha256": "d" * 64,
                "receipt_path": f"{family}/seed{seed}/{arm}/run-receipt.json",
                "receipt_sha256": "e" * 64,
                "checkpoint_path": f"{family}/seed{seed}/{arm}/checkpoints/task01-final.pt",
                "checkpoint_sha256": "f" * 64, "task_id": 1, "step": 600 + seed,
                "foundation_receipt_sha256": foundation["receipt_sha256"],
                "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
                "baseline_training_seal_sha256": "1" * 64,
                "comparison_v10_sleep_task1_checkpoint_sha256": None if family == "v10" else "2" * 64,
            })
    return {
        "schema": ev.REFERENCE_SCHEMA, "status": "frozen",
        "seeds": list(ev.SEEDS), "arms": list(ev.ARMS),
        "model_revision": "model-rev", "model_assets_sha256": {"weights.safetensors": H},
        "training_manifest_sha256": H,
        "evaluation_policy": {
            "cohort": "all_10_development_rows_per_class_in_seed_task0_and_task1_blocks",
            "rows_per_seed": 220, "classes_per_seed": 22,
            "primary": "global_77_sum_log_likelihood",
            "secondary": ["global_77_length_mean_log_likelihood", "unconstrained_greedy_generation"],
            "candidate_label_mask": None, "max_sequence_length": 128,
            "candidate_batch_size": 32, "max_new_tokens": 10,
            "max_backbone_forwards": 50000, "max_wall_seconds": 7200.0,
            "optimizer_updates": 0},
        "v10_config_file_sha256": "3" * 64, "v10_config_sha256": "4" * 64,
        "v10_source_lock_sha256": "5" * 64,
        "v10_source_hashes": {"src/nested_sleep/lm.py": H},
        "v10_grid_seal_sha256": "1" * 64, "v10_release_sha256": "6" * 64,
        "foundation_refs": {str(seed): dict(foundation) for seed in ev.SEEDS},
        "v11_config_file_sha256": "7" * 64, "v11_config_sha256": "8" * 64,
        "v11_source_lock_sha256": "9" * 64,
        "v11_source_hashes": {"tools/run_banking_lti_ablation_v11.py": H},
        "evaluator_sha256": "a" * 64, "contexts": contexts,
    }


def test_reference_requires_exact_fifteen_contexts_and_separate_pairing():
    ref = _reference()
    rows = ev.validate_reference(ref)
    assert len(rows) == 15
    assert ev.ARMS == ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE",
                       "LTI_NO_REWARD", "LTI_CANONICAL_TEACHER")
    assert not any("EOS" in arm for arm in ev.ARMS)
    assert [(r["seed"], r["arm"]) for r in rows[:5]] == [(17, x) for x in ev.ARMS]
    assert rows[0]["comparison_v10_sleep_task1_checkpoint_sha256"] is None
    bad = _reference()
    bad["contexts"].pop()
    with pytest.raises(ValueError, match="fifteen"):
        ev.validate_reference(bad)
    bad = _reference()
    bad["contexts"][-1]["comparison_v10_sleep_task1_checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="duplicate context"):
        # duplicate the arm by modifying the last reference row
        bad["contexts"][-1]["arm"] = "LTI_NO_REWARD"
        ev.validate_reference(bad)


def test_task01_development_selection_is_exact_220_rows_from_22_classes():
    order = list(range(77))
    random.Random(17).shuffle(order)
    rows = [{"id": f"{cid:02d}-{i:02d}", "intent_id": cid, "text": "development"}
            for cid in range(77) for i in range(10)]
    cohort = ev.select_task01_dev_rows(rows, order)
    assert len(cohort) == 220
    assert {r["intent_id"] for r in cohort} == set(order[:22])
    assert [r["id"] for r in cohort[:10]] == sorted(r["id"] for r in cohort[:10])
    with pytest.raises(ValueError, match="exactly ten"):
        ev.select_task01_dev_rows(rows[:-1], order)


def _payload(family, seed=17, arm="LTI_NO_REWARD", step=600):
    context_state = {"layer": {"a": 1}}
    wake = SimpleNamespace(context_state=context_state, pending={}, clocks={},
                           token_counts={}, ticks=step,
                           periods=(16, 64, 256, 1024, None),
                           base_lr=0.1, fast_lr=0.2, bank_states=[])
    if family == "v10":
        return {"schema": "banking-cil-v10-checkpoint", "seed": seed,
                "arm": arm, "task_id": 1, "step": step,
                "context": context_state, "wake": wake,
                "rng": {"python": None, "numpy": None, "torch_cpu": None, "torch_cuda": None},
                "config_sha256": "4" * 64, "source_lock_sha256": "5" * 64,
                "training_manifest_sha256": H}
    return {"schema": "banking-cil-v11-checkpoint-v1", "seed": seed,
            "arm": arm, "task_id": 1, "step": step,
            "context": context_state, "wake": wake,
            "rng": {"python": None, "numpy": None, "torch_cpu": None, "torch_cuda": None},
            "config_sha256": "8" * 64, "v11_source_lock_sha256": "9" * 64,
            "v10_source_lock_sha256": "5" * 64, "training_manifest_sha256": H,
            "foundation_checkpoint_sha256": "c" * 64,
            "baseline_training_seal_sha256": "1" * 64,
            "test_data_included": False}


def test_task_checkpoint_schema_cursor_wake_and_lineage_are_checked():
    ref = _reference()
    row = next(r for r in ref["contexts"] if r["seed"] == 17 and r["arm"] == "CMS_ONLY")
    ev.validate_task_checkpoint(_payload("v10", arm="CMS_ONLY", step=row["step"]), row,
                                config_sha256="4" * 64, source_lock_sha256="5" * 64,
                                training_manifest_sha256=H)
    bad = _payload("v10", arm="CMS_ONLY", step=row["step"])
    bad["wake"].ticks += 1
    with pytest.raises(ValueError, match="wake cursor"):
        ev.validate_task_checkpoint(bad, row, config_sha256="4" * 64,
                                    source_lock_sha256="5" * 64,
                                    training_manifest_sha256=H)
    for arm in ("LTI_NO_REWARD", "LTI_CANONICAL_TEACHER"):
        row11 = next(r for r in ref["contexts"] if r["seed"] == 17 and r["arm"] == arm)
        ev.validate_task_checkpoint(_payload("v11", arm=arm, step=row11["step"]), row11,
                                    config_sha256="8" * 64, source_lock_sha256="9" * 64,
                                    v10_source_lock_sha256="5" * 64,
                                    training_manifest_sha256=H)


def test_protocol_receipt_and_checkpoint_lineage_crosscheck_five_contexts_per_seed(tmp_path):
    import json
    import torch

    ref = _reference()
    v10_root, v11_root = tmp_path / "v10", tmp_path / "v11"
    baseline_runs = {}
    # Build the three V10 task-1 endpoint contexts first; the V11 paired metadata
    # points to the same-seed SLEEP_ADDITIVE context, never to its initialization.
    for row in ref["contexts"]:
        if row["family"] != "v10":
            continue
        run_dir = v10_root / f"seed{row['seed']}" / row["arm"]
        cp_path = run_dir / "checkpoints" / "task01-final.pt"
        cp_path.parent.mkdir(parents=True)
        torch.save(_payload("v10", seed=row["seed"], arm=row["arm"], step=row["step"]), cp_path)
        cp_sha = ev.sha256_file(cp_path)
        foundation = ref["foundation_refs"][str(row["seed"])]
        protocol = {"schema": "banking-cil-v10-protocol", "phase": "main",
                    "seed": row["seed"], "arm": row["arm"],
                    "config_sha256": ref["v10_config_sha256"],
                    "source_lock_sha256": ref["v10_source_lock_sha256"],
                    "training_manifest_sha256": H, "source_hashes": ref["v10_source_hashes"],
                    "test_data_included": False, "test_reader_available": False,
                    "parent": {"release_artifact_sha256": ref["v10_release_sha256"],
                               "foundation_receipt_sha256": foundation["receipt_sha256"],
                               "foundation_checkpoint_sha256": foundation["checkpoint_sha256"]}}
        protocol_path = run_dir / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        receipt = {"schema": "banking-cil-v10-run-receipt", "status": "complete",
                   "tasks_completed": 7, "seed": row["seed"], "arm": row["arm"],
                   "config_sha256": ref["v10_config_sha256"],
                   "source_lock_sha256": ref["v10_source_lock_sha256"],
                   "training_manifest_sha256": H,
                   "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
                   "test_data_included": False,
                   "model_immutable_versions_unchanged": True,
                   "task_metrics": [{"task_id": 1, "global_wake_tick": row["step"],
                                     "checkpoint_sha256": cp_sha}]}
        receipt_path = run_dir / "run-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        row.update(run_dir=str(run_dir), protocol_path=str(protocol_path),
                   protocol_sha256=ev.sha256_file(protocol_path),
                   receipt_path=str(receipt_path), receipt_sha256=ev.sha256_file(receipt_path),
                   checkpoint_path=str(cp_path), checkpoint_sha256=cp_sha)
        baseline_runs[(row["seed"], row["arm"])] = {"task1_checkpoint_sha256": cp_sha}
    # The V11 arms fork from the qualified task-0 foundation but identify the
    # same-seed SLEEP task-1 CP only as the paired evaluation comparison.
    for row in ref["contexts"]:
        if row["family"] != "v11":
            continue
        run_dir = v11_root / f"seed{row['seed']}" / row["arm"]
        cp_path = run_dir / "checkpoints" / "task01-final.pt"
        cp_path.parent.mkdir(parents=True)
        torch.save(_payload("v11", seed=row["seed"], arm=row["arm"], step=row["step"]), cp_path)
        cp_sha = ev.sha256_file(cp_path)
        foundation = ref["foundation_refs"][str(row["seed"])]
        comparison = baseline_runs[(row["seed"], "SLEEP_ADDITIVE")]["task1_checkpoint_sha256"]
        row["comparison_v10_sleep_task1_checkpoint_sha256"] = comparison
        protocol = {"schema": "banking-cil-v11-protocol-v1", "status": "complete",
                    "config_sha256": ref["v11_config_sha256"],
                    "source_hashes": ref["v11_source_hashes"],
                    "source_lock_sha256": ref["v11_source_lock_sha256"],
                    "v10_source_lock_sha256": ref["v10_source_lock_sha256"],
                    "v10_config_sha256": ref["v10_config_sha256"],
                    "training_manifest_sha256": H,
                    "baseline_training_seal_sha256": ref["v10_grid_seal_sha256"],
                    "foundation_receipt_sha256": foundation["receipt_sha256"],
                    "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
                    "comparison_v10_sleep_task1_checkpoint_sha256": comparison,
                    "seed": row["seed"], "arm": row["arm"], "task_id": 1,
                    "epochs": 2, "test_data_included": False, "test_data_read": False}
        protocol_path = run_dir / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        receipt = {"schema": "banking-cil-v11-run-receipt-v1", "status": "complete",
                   "seed": row["seed"], "arm": row["arm"], "task_id": 1, "epochs": 2,
                   "task1_wake_ticks": 100,
                   "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
                   "foundation_receipt_sha256": foundation["receipt_sha256"],
                   "baseline_training_seal_sha256": ref["v10_grid_seal_sha256"],
                   "comparison_v10_sleep_task1_checkpoint_sha256": comparison,
                   "checkpoint": "checkpoints/task01-final.pt", "checkpoint_sha256": cp_sha,
                   "config_sha256": ref["v11_config_sha256"],
                   "source_lock_sha256": ref["v11_source_lock_sha256"],
                   "v10_source_lock_sha256": ref["v10_source_lock_sha256"],
                   "training_manifest_sha256": H,
                   "model_immutable_versions_unchanged": True,
                   "test_data_included": False, "test_data_read": False,
                   "task_metrics": [{"task_id": 1, "task_step": 100,
                                     "global_wake_tick": row["step"],
                                     "evaluation_rows": 220, "checkpoint_sha256": cp_sha}]}
        receipt_path = run_dir / "run-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        row.update(run_dir=str(run_dir), protocol_path=str(protocol_path),
                   protocol_sha256=ev.sha256_file(protocol_path),
                   receipt_path=str(receipt_path), receipt_sha256=ev.sha256_file(receipt_path),
                   checkpoint_path=str(cp_path), checkpoint_sha256=cp_sha)
    baseline = {"runs": baseline_runs}
    verified = ev.verify_context_artifacts(
        ev.validate_reference(ref), v10_root=v10_root, v11_root=v11_root,
        reference=ref, v10_baseline=baseline)
    assert len(verified) == 15
    assert all(row["receipt"]["status"] == "complete" for row in verified)
    # Terminal metadata must match the real producer: `running` is rejected.
    target = next(row for row in ref["contexts"] if row["family"] == "v11")
    protocol_path = Path(target["protocol_path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["status"] = "running"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    target["protocol_sha256"] = ev.sha256_file(protocol_path)
    with pytest.raises(ValueError, match="V11 task-1 protocol/source/foundation/baseline binding mismatch"):
        ev.verify_context_artifacts(ev.validate_reference(ref), v10_root=v10_root,
                                    v11_root=v11_root, reference=ref,
                                    v10_baseline=baseline)
    protocol["status"] = "complete"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    target["protocol_sha256"] = ev.sha256_file(protocol_path)
    # The receipt's actual comparison field must equal paired same-seed Sleep.
    receipt_path = Path(target["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["comparison_v10_sleep_task1_checkpoint_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    target["receipt_sha256"] = ev.sha256_file(receipt_path)
    with pytest.raises(ValueError, match="V11 task-1 receipt/metrics endpoint mismatch"):
        ev.verify_context_artifacts(ev.validate_reference(ref), v10_root=v10_root,
                                    v11_root=v11_root, reference=ref,
                                    v10_baseline=baseline)
    receipt["comparison_v10_sleep_task1_checkpoint_sha256"] = baseline["runs"][(
        target["seed"], "SLEEP_ADDITIVE")]["task1_checkpoint_sha256"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    target["receipt_sha256"] = ev.sha256_file(receipt_path)
    # A V11 record cannot silently rebrand its baseline comparison as its parent.
    bad = next(row for row in ref["contexts"] if row["family"] == "v11")
    bad["comparison_v10_sleep_task1_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="comparison target"):
        ev.verify_context_artifacts(ev.validate_reference(ref), v10_root=v10_root,
                                    v11_root=v11_root, reference=ref,
                                    v10_baseline=baseline)
