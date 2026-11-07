import hashlib
import io
import json
import os
import random
import subprocess
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import banking_cil_common_v10 as common
import banking_cil_v10_runtime as banking_runtime
import probe_banking_transfer_v10 as probe


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_order(seed=17):
    values = list(range(77))
    random.Random(f"banking77-cil-v10:{seed}").shuffle(values)
    return values


def train_rows(order):
    rows = []
    for class_id in common.task_class_ids(order, 0):
        for index in range(4):
            rows.append({"id": h(f"id-{class_id}-{index}"), "intent_id": class_id,
                         "text": f"class {class_id} request {index}"})
    return rows


class BankingTransferProbeCpuTests(unittest.TestCase):
    def test_tree_hash_handles_real_adamw_scalars_and_roundtrips_exactly(self):
        import torch
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        state = {"optimizer": optimizer.state_dict(),
                 "zero_long": torch.tensor(0, dtype=torch.long),
                 "zero_bool": torch.tensor(False, dtype=torch.bool),
                 "zero_float": torch.tensor(0.0)}
        before = probe._tree_hash(state, torch)
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        restored = torch.load(stream, map_location="cpu", weights_only=False)
        self.assertEqual(before, probe._tree_hash(restored, torch))
        restored["zero_long"].add_(1)
        self.assertNotEqual(before, probe._tree_hash(restored, torch))

    def test_work_snapshot_includes_tail_calls_for_failure_receipts(self):
        class Budget:
            def as_dict(self): return {"forward_calls": 921, "adamw_updates": 6}
            probe_triplet_chunks = 2
        value = probe._probe_work_snapshot(Budget(), {
            "tail_calls": 17, "tail_calls_with_grad": 9, "tail_calls_without_grad": 8})
        self.assertEqual(value["tail_calls"], 17)
        self.assertEqual(value["tail_calls_with_grad"] + value["tail_calls_without_grad"], 17)
        self.assertEqual(value["triplet_chunks"], 2)
        self.assertEqual(value["forward_calls"], 921)

    def test_standalone_probe_bootstraps_package_imports_without_pythonpath(self):
        script = (
            "import runpy; "
            "from pathlib import Path; "
            f"ns=runpy.run_path({str(ROOT / 'tools' / 'probe_banking_transfer_v10.py')!r}); "
            "import tools.run_banking_cil_v10; "
            f"assert ns['_bootstrap_task_paths']()[0].resolve() == Path({str(ROOT)!r}).resolve()"
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, env=env,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_standalone_v11_driver_import_does_not_depend_on_pythonpath(self):
        script = (
            "import runpy; "
            f"ns=runpy.run_path({str(ROOT / 'tools' / 'run_banking_lti_ablation_v11.py')!r}); "
            "assert ns['_load_driver']().__name__ == 'run_banking_cil_v10'"
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, env=env,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_frozen_parent_config_uses_20_10_parent_schedule_and_foundation_caps(self):
        from tools import run_banking_cil_v10 as driver
        raw = json.loads((ROOT / "configs" / "banking_v10_frozen_k10.json").read_text(encoding="utf-8"))
        cfg = driver.validate_config(raw, phase="foundation")
        schedule = probe._validate_parent_schedule(cfg, driver)
        self.assertEqual((cfg["gkd_calls_per_event"], cfg["lti_calls_per_event"]), (20, 10))
        self.assertEqual(len(schedule), 30)
        self.assertEqual(probe._canonical_sha({k: v for k, v in cfg.items() if k != "phase"}),
                         driver._config_sha(cfg))
        limits = driver._budget_config(cfg, "foundation")
        self.assertEqual((limits["max_wall_seconds"], limits["max_forward_calls"],
                          limits["max_update_steps"]), (1800, 20000, 1000))

        bad = dict(cfg, gkd_calls_per_event=2)
        with self.assertRaisesRegex(ValueError, "20 GKD and 10 LTI"):
            probe._validate_parent_schedule(bad, driver)

    def test_fixed_30_call_old_class_schedule_covers_all_11_and_repeats_methods(self):
        order = task_order()
        schedule = probe.select_old_prompt_schedule(train_rows(order), order)
        audit = probe.safe_schedule_receipt(schedule)
        self.assertEqual(len(schedule), 30)
        self.assertEqual([row["method"] for row in schedule], list(probe.METHODS) * 10)
        self.assertEqual(audit["unique_classes"], 11)
        self.assertTrue(all(2 <= calls <= 3 for calls in audit["calls_by_class"].values()))
        self.assertEqual(len({row["example_id_sha256"] for row in schedule}), 30)

    def test_prompt_selector_requires_all_four_stable_rows_per_old_class(self):
        order = task_order()
        rows = train_rows(order)
        with self.assertRaisesRegex(ValueError, "four task-0 training prompts"):
            probe.select_old_prompt_schedule(rows[:-1], order)
        with self.assertRaisesRegex(ValueError, "fixed to 30"):
            probe.select_old_prompt_schedule(rows, order, calls=29)

    def test_foundation_receipt_and_checkpoint_must_bind_to_qualified_seed17(self):
        cfg, lock, manifest, cp = (h("cfg"), h("lock"), h("manifest"), h("cp"))
        receipt = {"schema": "banking-cil-v10-foundation-receipt", "status": "qualified", "seed": 17,
                   "arm": "FOUNDATION_CMS_ONLY", "task_id": 0, "config_sha256": cfg,
                   "source_lock_sha256": lock, "training_manifest_sha256": manifest,
                   "checkpoint_sha256": cp, "test_data_included": False,
                   "model_immutable_versions_unchanged": True, "gate": {"passed": True},
                   "metric": {"accuracy": 0.8}, "steps": 188}
        checkpoint = {"schema": "banking-cil-v10-checkpoint", "seed": 17,
                      "arm": "FOUNDATION_CMS_ONLY", "task_id": 0, "step": 188,
                      "config_sha256": cfg, "source_lock_sha256": lock,
                      "training_manifest_sha256": manifest, "context": {"state": 1},
                      "wake": {"snapshot": 1}, "rng": {"torch_cpu": "state"}, "extra": {}}
        probe.validate_foundation_payload(receipt, checkpoint, seed=17, config_sha256=cfg,
            source_lock_sha256=lock, training_manifest_sha256=manifest, checkpoint_sha256=cp)
        bad = dict(receipt, status="failed_gate")
        with self.assertRaisesRegex(ValueError, "not qualified"):
            probe.validate_foundation_payload(bad, checkpoint, seed=17, config_sha256=cfg,
                source_lock_sha256=lock, training_manifest_sha256=manifest, checkpoint_sha256=cp)
        bad = dict(checkpoint, config_sha256=h("other"))
        with self.assertRaisesRegex(ValueError, "checkpoint binding mismatch"):
            probe.validate_foundation_payload(receipt, bad, seed=17, config_sha256=cfg,
                source_lock_sha256=lock, training_manifest_sha256=manifest, checkpoint_sha256=cp)

    def test_sleep_uses_tail_kl_and_gold_updates_are_not_counted_as_gkd(self):
        kl = lambda *args, **kwargs: None
        ce = lambda *args, **kwargs: None
        self.assertEqual(probe._causal_objectives(SimpleNamespace(causal_token_kl=kl, causal_token_ce=ce)), (kl, ce))
        with self.assertRaisesRegex(TypeError, "causal_token_kl"):
            probe._causal_objectives(SimpleNamespace())

        class AdditiveLike:
            def __init__(self): self.counters = {"gkd_calls": 0, "gold_calls": 0}
            def optimize(self, loss, kind, tokens): self.counters[kind + "_calls"] += 1

        class LegacyMixLike:
            def __init__(self): self.counters = {"gkd_calls": 0, "lti_calls": 0}
            def optimize(self, loss, kind, tokens):
                if kind not in ("gkd", "lti"): raise ValueError("legacy kind mismatch")
                self.counters[kind + "_calls"] += 1

        additive = AdditiveLike()
        probe._apply_gold_update(additive, object(), 4, legacy_fixedmix=False)
        self.assertEqual(additive.counters, {"gkd_calls": 0, "gold_calls": 1})
        legacy = LegacyMixLike()
        probe._apply_gold_update(legacy, object(), 4, legacy_fixedmix=True)
        self.assertEqual(legacy.counters["gkd_calls"], 0)
        self.assertEqual(legacy.counters["gold_ce_calls"], 1)

    def test_probe_lti_callback_uses_full_v8_length_denominator_for_all_arms(self):
        cfg = {"lti_semantic_weight": 0.5, "lti_edit_weight": 0.5}
        callbacks = {arm: probe._lti_reward_callback(banking_runtime, cfg) for arm in probe.ARMS}
        arguments = {"teacher_tokens": [1, 2], "student_tokens": [1, 3],
                     "z0": 8, "canonical_labels": tuple(f"label {i}" for i in range(77))}
        values = {arm: callback("label 0", "label 1", **arguments)
                  for arm, callback in callbacks.items()}
        self.assertEqual(set(values.values()), {0.25})
        self.assertEqual(len(set(values.values())), 1)
        # Distance 1 over a maximum sequence length of 2 gives similarity .5;
        # the obsolete 1-distance/z0 form would have produced .875 (reward .4375).
        self.assertEqual(banking_runtime.paper_edit_similarity([1, 2], [1, 3], z0=8), 0.5)
        with self.assertRaisesRegex(TypeError, "local_lti_reward"):
            probe._lti_reward_callback(object(), cfg)


if __name__ == "__main__":
    unittest.main()
