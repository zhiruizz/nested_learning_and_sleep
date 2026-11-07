import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import banking_cil_v10_runtime as banking_runtime
import run_banking_lti_ablation_v11 as v11
import run_banking_cil_v10 as v10


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def make_grid(root, config_sha, source_hashes, manifest_sha):
    source_sha = v11.canonical_sha(source_hashes)
    release_sha = sha("release")
    runs_root = root / "runs"
    rows = []
    for seed in v11.SEEDS:
        foundation_sha = sha(f"foundation-{seed}")
        for arm in v11.BASE_ARMS:
            run_dir = runs_root / f"seed{seed}" / arm
            task_cp = run_dir / "checkpoints" / "task01-final.pt"
            final_cp = run_dir / "checkpoints" / "final-task06.pt"
            task_cp.parent.mkdir(parents=True, exist_ok=True)
            task_cp.write_bytes(f"task1-{seed}-{arm}".encode())
            final_cp.write_bytes(f"final-{seed}-{arm}".encode())
            task_sha = hashlib.sha256(task_cp.read_bytes()).hexdigest()
            final_sha = hashlib.sha256(final_cp.read_bytes()).hexdigest()
            receipt = {"schema": "banking-cil-v10-run-receipt", "status": "complete",
                "seed": seed, "arm": arm, "tasks_completed": 7,
                "config_sha256": config_sha, "source_lock_sha256": source_sha,
                "training_manifest_sha256": manifest_sha,
                "foundation_checkpoint_sha256": foundation_sha,
                "final_checkpoint": "checkpoints/final-task06.pt",
                "final_checkpoint_sha256": final_sha,
                "model_immutable_versions_unchanged": True, "test_data_included": False,
                "task_metrics": [{"task_id": task_id, "checkpoint_sha256": task_sha if task_id == 1 else sha(f"task-{task_id}")}
                                 for task_id in range(1, 7)]}
            receipt_path = run_dir / "run-receipt.json"
            write_json(receipt_path, receipt)
            protocol = {"schema": "banking-cil-v10-protocol", "phase": "main",
                "seed": seed, "arm": arm, "config_sha256": config_sha,
                "source_lock_sha256": source_sha, "training_manifest_sha256": manifest_sha,
                "source_hashes": source_hashes,
                "parent": {"release_artifact_sha256": release_sha,
                    "foundation_receipt_sha256": sha(f"foundation-receipt-{seed}"),
                    "foundation_checkpoint_sha256": foundation_sha}}
            write_json(run_dir / "protocol.json", protocol)
            rows.append({"seed": seed, "arm": arm, "status": "complete",
                "tasks_completed": 7, "run_dir": str(run_dir.resolve()),
                "run_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "final_checkpoint": "checkpoints/final-task06.pt",
                "final_checkpoint_sha256": final_sha})
    seal = {"schema": v11.BASE_SEAL_SCHEMA, "complete": True,
        "config_sha256": config_sha, "source_lock_sha256": source_sha,
        "training_manifest_sha256": manifest_sha, "test_data_included": False,
        "official_test_read": False, "release_artifact_sha256": release_sha, "runs": rows}
    seal_path = root / "training-seal.json"
    write_json(seal_path, seal)
    return runs_root, seal_path


class BankingLtiAblationV11Tests(unittest.TestCase):
    def test_no_reward_and_nonempty_teacher_callbacks_preserve_policy(self):
        cfg = {"lti_semantic_weight": 0.5, "lti_edit_weight": 0.5}
        no_reward = v11.make_reward_callback("LTI_NO_REWARD", banking_runtime, cfg)
        filtered = v11.make_reward_callback("LTI_CANONICAL_TEACHER", banking_runtime, cfg)
        kwargs = {"teacher_tokens": [1, 2], "student_tokens": [1, 3], "z0": 8,
                  "canonical_labels": tuple(f"label {i}" for i in range(77))}
        self.assertEqual(no_reward("label 0", "label 1", **kwargs), 0.0)
        self.assertEqual(filtered("label 0", "label 1", **kwargs), 0.25)
        self.assertEqual(filtered("", "label 1", **{**kwargs, "teacher_tokens": []}), 0.0)

    def test_canonical_gate_uses_runtime_normalizer_and_never_checks_row_gold(self):
        seen = []
        def old_reward(teacher_text, student_text, **kwargs):
            seen.append((teacher_text, student_text, kwargs))
            return 0.375
        runtime = SimpleNamespace(local_lti_reward=old_reward,
                                  _normalized_label=banking_runtime._normalized_label)
        callback = v11.make_reward_callback("LTI_CANONICAL_TEACHER", runtime,
            {"lti_semantic_weight": 0.5, "lti_edit_weight": 0.5})
        kwargs = {"teacher_tokens": [5], "student_tokens": [7], "z0": 8,
                  "canonical_labels": ("cash_withdrawal", "cash deposit")}
        # Canonical-but-wrong-for-this-row label remains eligible; callback
        # has no row/gold input and forwards unchanged Eq.4 reward.
        self.assertEqual(callback(" CASH withdrawal ", "cash deposit", **kwargs), 0.375)
        self.assertEqual(len(seen), 1)
        self.assertEqual(callback("unrecognized answer", "cash deposit", **kwargs), 0.0)
        self.assertEqual(callback("   ", "cash deposit", **kwargs), 0.0)
        self.assertEqual(len(seen), 1)
        self.assertEqual(banking_runtime.paper_edit_similarity([1, 2], [1, 3], z0=8), 0.5)

    def test_runtime_facade_injects_callback_without_mutating_runtime(self):
        calls = []
        def runtime_sleep(*args, **kwargs):
            calls.append(kwargs["lti_reward"]("", "label 0", teacher_tokens=[],
                student_tokens=[], z0=8, canonical_labels=tuple(f"label {i}" for i in range(77))))
            return {"ok": True}
        runtime = SimpleNamespace(local_lti_reward=banking_runtime.local_lti_reward,
                                  _normalized_label=banking_runtime._normalized_label,
                                  run_sleep_event=runtime_sleep)
        original = runtime.run_sleep_event
        facade = v11.RuntimeFacade(runtime, "LTI_CANONICAL_TEACHER",
            {"lti_semantic_weight": 0.5, "lti_edit_weight": 0.5})
        self.assertEqual(facade.run_sleep_event(object(), lti_reward=lambda *_a, **_k: 99), {"ok": True})
        self.assertEqual(calls, [0.0])
        self.assertIs(runtime.run_sleep_event, original)

    def test_both_arms_reuse_the_same_frozen_v10_prompt_schedule(self):
        old = {cid: [{"id": f"old-{cid}", "intent_id": cid, "text": f"old prompt {cid}"}]
               for cid in range(11)}
        current = {cid: [{"id": f"new-{cid}", "intent_id": cid, "text": f"new prompt {cid}"}]
                   for cid in range(11, 22)}
        kinds = ["gkd", "gkd", "lti"] * 10
        first = v10.make_sleep_request_schedule(old, current, event_index=7, task_id=1,
                                                method_kinds=kinds)
        second = v10.make_sleep_request_schedule(old, current, event_index=7, task_id=1,
                                                 method_kinds=kinds)
        self.assertEqual([(k, r["id"], role) for k, r, role in first],
                         [(k, r["id"], role) for k, r, role in second])
        self.assertEqual([k for k, _, _ in first], kinds)

    def test_full_baseline_grid_validates_actual_receipt_and_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="v11-fixture-") as temp:
            root = Path(temp)
            c, m = sha("cfg"), sha("manifest")
            source_hashes = {"tools/fake.py": sha("src")}; s = v11.canonical_sha(source_hashes)
            runs_root, seal_path = make_grid(root, c, source_hashes, m)
            actual = v11.validate_baseline_seal(seal_path, runs_root,
                expected_seal_sha256=v11.sha256_file(seal_path), config_sha256=c,
                source_lock_sha256=s, manifest_sha256=m, source_hashes=source_hashes)
            self.assertEqual(len(actual["runs"]), 9)
            self.assertEqual(set(actual["foundation_checkpoint_sha256_by_seed"]), set(v11.SEEDS))

    def test_baseline_seal_rejects_arm_foundation_mismatch_missing_row_and_cp_tamper(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="v11-fixture-") as temp:
            root = Path(temp); c, m = sha("cfg"), sha("manifest")
            source_hashes = {"tools/fake.py": sha("src")}; s = v11.canonical_sha(source_hashes)
            runs_root, seal_path = make_grid(root, c, source_hashes, m)
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            target = next(row for row in seal["runs"] if row["seed"] == 17 and row["arm"] == "GOLD_ADDITIVE")
            receipt_path = Path(target["run_dir"]) / "run-receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["foundation_checkpoint_sha256"] = sha("wrong-foundation")
            write_json(receipt_path, receipt)
            target["run_receipt_sha256"] = v11.sha256_file(receipt_path)
            protocol_path = Path(target["run_dir"]) / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["parent"]["foundation_checkpoint_sha256"] = sha("wrong-foundation")
            write_json(protocol_path, protocol)
            write_json(seal_path, seal)
            with self.assertRaisesRegex(ValueError, "paired V10 baseline arms"):
                v11.validate_baseline_seal(seal_path, runs_root,
                    expected_seal_sha256=v11.sha256_file(seal_path), config_sha256=c,
                    source_lock_sha256=s, manifest_sha256=m, source_hashes=source_hashes)

            # Rebuild a clean seal, then tamper the real task-1 CP bytes.
            runs_root, seal_path = make_grid(root / "fresh", c, source_hashes, m)
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            seal["runs"].pop()
            write_json(seal_path, seal)
            with self.assertRaisesRegex(ValueError, "exactly nine"):
                v11.validate_baseline_seal(seal_path, runs_root,
                    expected_seal_sha256=v11.sha256_file(seal_path), config_sha256=c,
                    source_lock_sha256=s, manifest_sha256=m, source_hashes=source_hashes)
            runs_root, seal_path = make_grid(root / "tamper", c, source_hashes, m)
            changed = next(row for row in json.loads(seal_path.read_text(encoding="utf-8"))["runs"]
                           if row["seed"] == 29 and row["arm"] == "SLEEP_ADDITIVE")
            (Path(changed["run_dir"]) / "checkpoints" / "task01-final.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "task-1 checkpoint bytes"):
                v11.validate_baseline_seal(seal_path, runs_root,
                    expected_seal_sha256=v11.sha256_file(seal_path), config_sha256=c,
                    source_lock_sha256=s, manifest_sha256=m, source_hashes=source_hashes)

    def test_foundation_requires_task0_gate_and_test_exclusion(self):
        receipt = {"schema": "banking-cil-v10-foundation-receipt", "status": "qualified",
            "seed": 17, "task_id": 0, "config_sha256": sha("cfg"),
            "source_lock_sha256": sha("src"), "training_manifest_sha256": sha("manifest"),
            "checkpoint_sha256": sha("cp"), "metric": {"count": 110, "classes_evaluated": 11,
                "accuracy": .6, "task_class_macro_accuracy": .45},
            "gate": {"passed": True, "global_candidate_sum_accuracy": .5,
                "task0_class_macro_accuracy": .4},
            "model_immutable_versions_unchanged": True, "test_data_included": False}
        v11.validate_foundation(receipt, seed=17, config_sha256=sha("cfg"),
            source_lock_sha256=sha("src"), manifest_sha256=sha("manifest"),
            checkpoint_sha256=sha("cp"))
        low = dict(receipt, metric={**receipt["metric"], "task_class_macro_accuracy": .39})
        with self.assertRaisesRegex(ValueError, "gate or test-isolation"):
            v11.validate_foundation(low, seed=17, config_sha256=sha("cfg"),
                source_lock_sha256=sha("src"), manifest_sha256=sha("manifest"),
                checkpoint_sha256=sha("cp"))


if __name__ == "__main__":
    unittest.main()
