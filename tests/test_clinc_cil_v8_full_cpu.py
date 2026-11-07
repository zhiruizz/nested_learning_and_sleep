"""CPU-only contract tests for the full CLINC candidate; no model/data rows."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("clinc_full_candidate", ROOT / "tools" / "run_clinc_cil_v8_full.py")
FULL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FULL)


def fake_pools(old_classes: int):
    old, current = [], []
    for role, nclasses, rows in (("old", old_classes, old), ("current", 15, current)):
        base = 0 if role == "old" else 10000
        for cls in range(nclasses):
            for example in range(4):
                rid = f"{base + cls * 4 + example:064x}"
                rows.append({"id": rid, "intent_id": (cls if role == "old" else 135 + cls),
                             "text": "not used"})
    return {"old": old, "current": current}


def hash64(ch: str) -> str:
    return hashlib.sha256(ch.encode()).hexdigest()


class FullCandidateCpuTests(unittest.TestCase):
    def test_k1_selector_matches_frozen_15_old_class_round_robin(self):
        pools = fake_pools(15)
        selected = FULL.select_full_sleep_prompts(pools, event=4, task_id=1, gkd_calls=2, lti_calls=1)
        expected = []
        classes = {role: [[r for r in pools[role] if r["intent_id"] == cls]
                          for cls in sorted({r["intent_id"] for r in pools[role]})]
                   for role in ("old", "current")}
        for call, kind in enumerate(("gkd", "gkd", "lti")):
            g = 3 * 4 + call
            role = "old" if (g + 1) % 2 == 0 else "current"
            e = g // 2
            expected.append((kind, classes[role][e % 15][(e // 15) % 4]["id"], role))
        self.assertEqual([(k, r["id"], role) for k, r, role in selected], expected)

    def test_k10_covers_135_old_classes_in_nine_events_and_reports_unique_ids(self):
        pools = fake_pools(135)
        selected = []
        for event in range(9):
            selected.extend(FULL.select_full_sleep_prompts(pools, event=event, task_id=9,
                gkd_calls=20, lti_calls=10))
        exposure = FULL.summarize_sleep_exposure(pools, selected)
        self.assertEqual(exposure["old"]["calls"], 135)
        self.assertEqual(exposure["old"]["classes_covered_count"], 135)
        self.assertEqual(exposure["old"]["unique_id_count"], 135)
        self.assertEqual(exposure["current"]["calls"], 135)
        self.assertEqual(exposure["current"]["classes_covered_count"], 15)
        self.assertEqual(exposure["current"]["unique_id_count"], 60)

    def test_k1_exposure_does_not_claim_full_135_class_coverage(self):
        pools = fake_pools(135)
        selected = []
        for event in range(15):
            selected.extend(FULL.select_full_sleep_prompts(pools, event=event, task_id=9))
        exposure = FULL.summarize_sleep_exposure(pools, selected)
        self.assertLess(exposure["old"]["classes_covered_count"], 135)
        self.assertEqual(exposure["current"]["classes_covered_count"], 15)

    def test_thresholded_edit_component_matches_hand_calculation(self):
        self.assertAlmostEqual(FULL.paper_edit_similarity([1, 2, 3], [1, 3], z0=2), 2 / 3)
        self.assertEqual(FULL.paper_edit_similarity([1, 2, 3], [4, 5, 6], z0=2), 0.0)
        self.assertEqual(FULL.paper_edit_similarity([], [], z0=2), 1.0)
        self.assertEqual(FULL.paper_edit_similarity([], [1, 2], z0=2), 0.0)

    def test_foundation_mode_trains_only_task_zero_and_complete_cursor_is_empty(self):
        self.assertEqual(list(FULL.mode_task_range("foundation", 0)), [0])
        self.assertEqual(list(FULL.mode_task_range("foundation", 1)), [])
        self.assertEqual(list(FULL.mode_task_range("continuation", 1)), list(range(1, 10)))
        self.assertEqual(list(FULL.mode_task_range("continuation", 10)), [])
        FULL.assert_foundation_terminal_state(wake_ticks=188,
            pending_clocks={0: 12, 1: 60, 2: 188, 3: 188}, task_metrics=[{"task_id": 0}],
            cursor={"task_id": 1, "tasks_completed": 1})
        with self.assertRaises(ValueError):
            FULL.assert_foundation_terminal_state(wake_ticks=376,
                pending_clocks={0: 8, 1: 56, 2: 376, 3: 376},
                task_metrics=[{"task_id": 0}, {"task_id": 1}],
                cursor={"task_id": 2, "tasks_completed": 2})

    def test_terminal_receipt_is_immutable_and_blocks_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            run_dir = Path(tmp)
            receipt = run_dir / FULL.terminal_receipt_name("foundation")
            FULL._atomic_json(receipt, {"status": "complete"}, refuse_existing=True)
            original = receipt.read_bytes()
            with self.assertRaises(ValueError):
                FULL.refuse_terminal_run(run_dir, "foundation")
            with self.assertRaises(FileExistsError):
                FULL._atomic_json(receipt, {"status": "overwritten"}, refuse_existing=True)
            self.assertEqual(receipt.read_bytes(), original)

    def test_frozen_config_rejects_placeholder_revision(self):
        cfg = json.loads((ROOT / "configs" / "clinc_cil_v8_full_master_frozen_k10.json").read_text(encoding="utf-8"))
        cfg["status"] = "frozen"
        cfg["source_revision"] = "TBD_AFTER_SOURCE_REVIEW"
        with self.assertRaisesRegex(ValueError, "placeholder"):
            FULL.validate_master_config(cfg)

    def test_foundation_seal_checks_receipt_and_actual_checkpoint_bytes(self):
        master, manifest, source = hash64("master"), hash64("manifest"), hash64("source")
        assets = {"asset": hash64("asset")}
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            for seed in FULL.SEEDS:
                seed_dir = root / f"seed{seed}"
                seed_dir.mkdir()
                cp_bytes = f"checkpoint-{seed}".encode()
                (seed_dir / "foundation.pt").write_bytes(cp_bytes)
                receipt = {"schema": FULL.FOUNDATION_SCHEMA, "status": "complete", "seed": seed,
                    "acquisition_qualified": True, "test_data_included": False,
                    "master_config_sha256": master,
                    "child_config_sha256": FULL._child_config_sha(master, "foundation", seed, None),
                    "training_manifest_sha256": manifest, "source_lock_sha256": source,
                    "model_revision": "model@rev", "model_assets_sha256": assets,
                    "final_checkpoint": "foundation.pt",
                    "final_checkpoint_sha256": hashlib.sha256(cp_bytes).hexdigest(),
                    "tasks_completed": 1, "task0_clean_current_accuracy": 0.7,
                    "task_metrics": [{"task_id": 0}], "wake_ticks": 188,
                    "pending_clocks": {"0": 12, "1": 60, "2": 188, "3": 188}, "work": {"updates": 188}}
                (seed_dir / "foundation-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            _receipts, verified = FULL.load_and_verify_foundation_grid(root, master_sha=master,
                manifest_sha=manifest, source_lock_sha=source, model_revision="model@rev",
                model_assets_sha256=assets)
            self.assertEqual(len(verified), 3)
            self.assertTrue(all(row["work_counted_once"] for row in verified))
            receipt_path = root / "seed17" / "foundation-receipt.json"
            changed = json.loads(receipt_path.read_text(encoding="utf-8"))
            changed["tasks_completed"] = 2
            receipt_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                FULL.load_and_verify_foundation_grid(root, master_sha=master,
                    manifest_sha=manifest, source_lock_sha=source, model_revision="model@rev",
                    model_assets_sha256=assets)
            changed["tasks_completed"] = 1
            receipt_path.write_text(json.dumps(changed), encoding="utf-8")
            (root / "seed17" / "foundation.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes mismatch"):
                FULL.load_and_verify_foundation_grid(root, master_sha=master,
                    manifest_sha=manifest, source_lock_sha=source, model_revision="model@rev",
                    model_assets_sha256=assets)
            (root / "seed29" / "foundation-receipt.json").unlink()
            with self.assertRaises(FileNotFoundError):
                FULL.load_and_verify_foundation_grid(root, master_sha=master,
                    manifest_sha=manifest, source_lock_sha=source, model_revision="model@rev",
                    model_assets_sha256=assets)

    def test_safe_checkpoint_roundtrip_contract_at_mid_and_task_boundary(self):
        bindings = {"master_config_sha256": hash64("master"), "child_config_sha256": hash64("child"),
                    "training_manifest_sha256": hash64("manifest"), "source_lock_sha256": hash64("source"),
                    "seed": 29, "arm": "HOPE_SLEEP_FIXEDMIX"}
        for next_step, tasks_completed in ((65, 1), (189, 2)):
            cursor = FULL.checkpoint_cursor(mode="continuation", task_id=1, next_step=next_step,
                task_count=tasks_completed, schedule_sha256=hash64("schedule"), replay_ids_sha256=hash64("pool"))
            payload = {"schema": FULL.CHECKPOINT_SCHEMA, **bindings, "mode": "continuation",
                "foundation_checkpoint_sha256": hash64("foundation"), "cursor": cursor,
                "context": {"weights": "toy"}, "wake": {"pending": {"memory.0.a": [1, 2]}, "clocks": {0: 3}},
                "rng": {"python": (1, 2)}, "work_cumulative": {"updates": 7},
                "physical_sleep_adamw_steps": 3, "physical_wake_updates_run": 100,
                "sleep_event_index": 12, "sleep_exposure": {"by_task": {"1": {"old": 3}}}}
            FULL.validate_checkpoint_payload(payload, master_config_sha256=bindings["master_config_sha256"],
                child_config_sha256=bindings["child_config_sha256"],
                manifest_sha256=bindings["training_manifest_sha256"],
                source_lock_sha256=bindings["source_lock_sha256"], seed=29,
                arm="HOPE_SLEEP_FIXEDMIX")
            self.assertEqual(payload["wake"]["pending"], {"memory.0.a": [1, 2]})
            self.assertEqual(payload["cursor"]["next_step"], next_step)
            broken = dict(payload)
            broken["cursor"] = dict(cursor, safe_point="mid_transaction")
            with self.assertRaises(ValueError):
                FULL.validate_checkpoint_payload(broken, master_config_sha256=bindings["master_config_sha256"],
                    child_config_sha256=bindings["child_config_sha256"],
                    manifest_sha256=bindings["training_manifest_sha256"],
                    source_lock_sha256=bindings["source_lock_sha256"], seed=29,
                    arm="HOPE_SLEEP_FIXEDMIX")

    def test_exact_six_grid_gate_rejects_missing_or_mismatched_run(self):
        rows = []
        for seed in FULL.SEEDS:
            for arm in FULL.ARMS:
                rows.append({"schema": FULL.RECEIPT_SCHEMA, "status": "complete",
                    "seed": seed, "arm": arm, "tasks_completed": 10,
                    "master_config_sha256": hash64("master"),
                    "training_manifest_sha256": hash64("manifest"),
                    "source_lock_sha256": hash64("source"),
                    "acquisition_qualified": True, "test_data_included": False,
                    "foundation_checkpoint_sha256": hash64("foundation"),
                    "foundation_receipt_sha256": hash64(f"foundation-receipt-{seed}"),
                    "inner_cycles_per_event": 1,
                    "objective_version": "canonical_agreement_plus_thresholded_length_normalized_edit_v1",
                    "task_metrics": [{"task_id": task_id} for task_id in range(10)],
                    "final_checkpoint_sha256": hash64(f"{seed}-{arm}")})
        FULL.validate_main_grid_receipts(rows, master_sha=hash64("master"),
            manifest_sha=hash64("manifest"), source_lock_sha=hash64("source"),
            inner_cycles_per_event=1,
            objective_version="canonical_agreement_plus_thresholded_length_normalized_edit_v1")
        with self.assertRaises(ValueError):
            FULL.validate_main_grid_receipts(rows[:-1], master_sha=hash64("master"),
                manifest_sha=hash64("manifest"), source_lock_sha=hash64("source"),
                inner_cycles_per_event=1,
                objective_version="canonical_agreement_plus_thresholded_length_normalized_edit_v1")
        wrong = [dict(row) for row in rows]
        wrong[0]["training_manifest_sha256"] = hash64("other")
        with self.assertRaises(ValueError):
            FULL.validate_main_grid_receipts(wrong, master_sha=hash64("master"),
                manifest_sha=hash64("manifest"), source_lock_sha=hash64("source"),
                inner_cycles_per_event=1,
                objective_version="canonical_agreement_plus_thresholded_length_normalized_edit_v1")
        wrong_k = [dict(row) for row in rows]
        wrong_k[0]["inner_cycles_per_event"] = 10
        with self.assertRaises(ValueError):
            FULL.validate_main_grid_receipts(wrong_k, master_sha=hash64("master"),
                manifest_sha=hash64("manifest"), source_lock_sha=hash64("source"),
                inner_cycles_per_event=1,
                objective_version="canonical_agreement_plus_thresholded_length_normalized_edit_v1")
        missing_boundary = [dict(row) for row in rows]
        missing_boundary[0]["task_metrics"] = missing_boundary[0]["task_metrics"][:-1]
        with self.assertRaises(ValueError):
            FULL.validate_main_grid_receipts(missing_boundary, master_sha=hash64("master"),
                manifest_sha=hash64("manifest"), source_lock_sha=hash64("source"),
                inner_cycles_per_event=1,
                objective_version="canonical_agreement_plus_thresholded_edit_v1")
        paired_mismatch = [dict(row) for row in rows]
        paired_mismatch[1]["foundation_checkpoint_sha256"] = hash64("different-foundation")
        with self.assertRaisesRegex(ValueError, "do not share"):
            FULL.validate_main_grid_receipts(paired_mismatch, master_sha=hash64("master"),
                manifest_sha=hash64("manifest"), source_lock_sha=hash64("source"),
                inner_cycles_per_event=1,
                objective_version="canonical_agreement_plus_thresholded_length_normalized_edit_v1")
        receipt_mismatch = [dict(row) for row in rows]
        receipt_mismatch[1]["foundation_receipt_sha256"] = hash64("other-receipt")
        with self.assertRaisesRegex(ValueError, "do not share"):
            FULL.validate_main_grid_receipts(receipt_mismatch, master_sha=hash64("master"),
                manifest_sha=hash64("manifest"), source_lock_sha=hash64("source"),
                inner_cycles_per_event=1,
                objective_version="canonical_agreement_plus_thresholded_length_normalized_edit_v1")


if __name__ == "__main__":
    unittest.main()
