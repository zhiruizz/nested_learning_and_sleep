import unittest
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import torch
from tools import evaluate_clinc_test_v8 as evaluator
from tools import run_clinc_cil_v8_r2 as r2


class TestAccessGateCpu(unittest.TestCase):
    def _seal(self):
        grid = [{"seed": seed, "arm": arm}
                for seed in (17, 29, 43)
                for arm in ("HOPE", "HOPE_SLEEP_FIXEDMIX")]
        return ({"schema": "clinc-cil-v8-main-six-seal", "complete": True,
                 "test_data_included": False, "test_release_ready": True,
                 "master_config_sha256": "a" * 64,
                 "training_manifest_sha256": "b" * 64,
                 "source_hashes": {"runner": "c" * 64},
                 "source_lock_sha256": "d" * 64,
                 "main_grid": grid, "verified_foundations": [{}, {}, {}],
                 "verified_run_files": [{}, {}, {}, {}, {}, {}],
                 "runs": [{"seed": s, "arm": a} for s, a in
                          ((17, "HOPE"), (17, "HOPE_SLEEP_FIXEDMIX"),
                           (29, "HOPE"), (29, "HOPE_SLEEP_FIXEDMIX"),
                           (43, "HOPE"), (43, "HOPE_SLEEP_FIXEDMIX"))]},
                "a" * 64, "b" * 64, {"runner": "c" * 64}, "d" * 64, grid)

    def test_complete_exact_grid_seal_passes_pure_gate(self):
        seal, master, manifest, sources, lock, grid = self._seal()
        evaluator.validate_seal_summary(seal, expected_master_sha=master,
            expected_manifest_sha=manifest, expected_source_hashes=sources,
            expected_source_lock_sha=lock, expected_grid=grid)

    def test_incomplete_or_test_contaminated_seal_rejected(self):
        seal, master, manifest, sources, lock, grid = self._seal()
        for changes in ({"complete": False}, {"test_data_included": True},
                        {"test_release_ready": False}, {"verified_run_files": [{}, {}]}):
            with self.subTest(changes=changes):
                changed = dict(seal, **changes)
                with self.assertRaises(ValueError):
                    evaluator.validate_seal_summary(changed, expected_master_sha=master,
                        expected_manifest_sha=manifest, expected_source_hashes=sources,
                        expected_source_lock_sha=lock, expected_grid=grid)

    def test_stale_config_source_and_grid_bindings_are_rejected(self):
        seal, master, manifest, sources, lock, grid = self._seal()
        cases = [
            ({"master_config_sha256": "e" * 64}, master, manifest, sources, lock, grid),
            (None, master, manifest, {"runner": "f" * 64}, lock, grid),
            (None, master, manifest, sources, "0" * 64, grid),
            (None, master, manifest, sources, lock, grid[:-1]),
        ]
        for changed_seal, exp_master, exp_manifest, exp_sources, exp_lock, exp_grid in cases:
            with self.subTest(expected=exp_lock):
                value = dict(seal, **changed_seal) if changed_seal else seal
                with self.assertRaises(ValueError):
                    evaluator.validate_seal_summary(value, expected_master_sha=exp_master,
                        expected_manifest_sha=exp_manifest, expected_source_hashes=exp_sources,
                        expected_source_lock_sha=exp_lock, expected_grid=exp_grid)

    def test_access_plan_is_bound_to_passed_gate_and_exact_source(self):
        with self.assertRaises(ValueError):
            evaluator.access_plan({"passed": False, "test_payload_read": False})
        gate = {"passed": True, "test_payload_read": False,
                "evaluation_contract_sha256": "1" * 64}
        plan = evaluator.access_plan(gate)
        metadata_path = (Path(__file__).resolve().parent / "fixtures" /
                         "clinc_oos_plus_test_pin_v1.json")
        import json
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["metadata_source"],
            "https://huggingface.co/api/datasets/clinc/clinc_oos/revision/"
            "155b9c710419136e17307b80d0a13e68cd46b4ec")
        self.assertEqual(plan["repo_id"], metadata["repo_id"])
        self.assertEqual(plan["source"]["revision"], metadata["revision"])
        self.assertEqual(plan["source"]["path"], metadata["sibling"]["path"])
        self.assertEqual(plan["source"]["bytes"], metadata["sibling"]["size"])
        self.assertEqual(plan["source"]["sha256"], metadata["sibling"]["lfs_sha256"])
        self.assertEqual(plan["source"]["metadata_sha256"], metadata["metadata_sha256"])

    def test_evaluation_contract_binds_source_config_policy_and_evaluator(self):
        contract = {"schema": "clinc-v8-test-evaluation-contract-v1", "status": "frozen",
            "evaluator_sha256": "a" * 64, "training_master_sha256": "b" * 64,
            "source_lock_sha256": "c" * 64, "test_source": evaluator.TEST_SOURCE,
            "evaluation_policy": "score all 4500 in-scope test rows on the six sealed final checkpoints; "
                                 "free generation, no task mask, no updates, no checkpoint selection"}
        evaluator.validate_evaluation_contract(contract, evaluator_sha="a" * 64,
            master_sha="b" * 64, source_lock_sha="c" * 64)
        for field, changed in (("evaluator_sha256", "d" * 64),
                               ("training_master_sha256", "e" * 64),
                               ("source_lock_sha256", "f" * 64),
                               ("evaluation_policy", "choose the best checkpoint")):
            with self.subTest(field=field):
                candidate = dict(contract, **{field: changed})
                with self.assertRaises(ValueError):
                    evaluator.validate_evaluation_contract(candidate, evaluator_sha="a" * 64,
                        master_sha="b" * 64, source_lock_sha="c" * 64)

    def test_huggingface_parquet_metadata_info_features_classlabel_decode(self):
        names = {"0": "alpha", "1": "beta", "2": "oos"}
        raw = ("{\"info\":{\"features\":{\"intent\":{\"names\":" +
               __import__("json").dumps(names) + "}}}}" ).encode()
        table = SimpleNamespace(schema=SimpleNamespace(metadata={b"huggingface": raw}))
        self.assertEqual(evaluator._intent_names_from_metadata(table), ["alpha", "beta", "oos"])

    def test_filter_maps_150_style_labels_and_retains_duplicate_rows(self):
        classes = ("alpha", "beta")
        texts = ["repeat", "repeat", "oos", "other"]
        intents = [0, 1, 2, 2]
        rows, info = evaluator.filter_intent_records(texts, intents, classes,
            intent_names=["alpha", "beta", "oos"], oos_id=2,
            expected_total=2, expected_per_class=1, expected_raw_rows=4)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["text"] for row in rows], ["repeat", "repeat"])
        self.assertEqual(info["oos_excluded_rows"], 2)
        self.assertEqual(info["oos_distinct_labels"], 1)
        self.assertEqual(info["oos_raw_label_id"], 2)

    def test_integer_classlabel_requires_and_uses_embedded_name_map(self):
        with self.assertRaisesRegex(ValueError, "151 names"):
            evaluator.filter_intent_records(["x", "y"], [0, 1], ("alpha", "beta"),
                expected_total=2, expected_per_class=1)
        rows, _ = evaluator.filter_intent_records(["x", "y", "z"], [0, 1, 2],
            ("alpha", "beta"), intent_names=["alpha", "beta", "oos"], oos_id=2,
            expected_total=2, expected_per_class=1)
        self.assertEqual([row["intent_id"] for row in rows], [0, 1])
        with self.assertRaisesRegex(ValueError, "unique declared raw ClassLabel id 2"):
            evaluator.filter_intent_records(["x"], [0], ("alpha", "beta"),
                intent_names=["alpha", "oos", "beta"], oos_id=2, expected_total=1,
                expected_per_class=1)

    def test_checkpoint_context_digest_matches_r2(self):
        from tools.clinc_tail_runtime_v8 import MutableContext
        from src.nested_sleep.experts import LowRankMemoryLevel
        context = MutableContext(torch.nn.ModuleList([LowRankMemoryLevel(2, 1, 2)]),
            torch.nn.Parameter(torch.ones(1, 2)), torch.nn.Parameter(torch.ones(5, 1)), True)
        self.assertEqual(evaluator.mutable_context_digest(context.state_dict(), torch),
                         r2._mutable_context_sha(context))

    def test_exposure_audit_does_not_trust_unconditional_receipt_flag(self):
        hope = {"sleep_event_index": 0, "physical_sleep_adamw_steps": 0,
                "work": {"sleep_events": 0},
                "data_exposure": {"sleep_text_replay": True}}
        audited = evaluator.audit_sleep_exposure(hope, "HOPE", 17)
        self.assertFalse(audited["sleep_text_replay_derived_from_counters"])
        self.assertTrue(audited["receipt_sleep_text_replay_claim"])
        self.assertFalse(audited["receipt_flag_matches_counters"])
        changed = dict(hope, sleep_event_index=1, work={"sleep_events": 1})
        with self.assertRaisesRegex(ValueError, "unexpectedly records"):
            evaluator.audit_sleep_exposure(changed, "HOPE", 17)

    def test_continuation_must_start_from_the_same_foundation_context(self):
        context_sha = "a" * 64
        evaluator.validate_paired_foundation_start(
            {"initial_mutable_sha256": context_sha},
            {"foundation_context_sha256": context_sha})
        with self.assertRaisesRegex(ValueError, "differs from its paired foundation"):
            evaluator.validate_paired_foundation_start(
                {"initial_mutable_sha256": "b" * 64},
                {"foundation_context_sha256": context_sha})

    def test_evaluation_rejects_gate_before_any_test_path_access(self):
        base = Path(__file__).resolve().parent
        with mock.patch.object(evaluator, "verify_release_gate",
                               side_effect=ValueError("seal gate failed")) as gate, \
             mock.patch.object(evaluator, "_read_test_rows",
                               side_effect=AssertionError("test payload reader called")) as reader:
            with self.assertRaisesRegex(ValueError, "seal gate failed"):
                evaluator.evaluate(project_root=base, model_path=base,
                    master_config=base / "missing-config.json",
                    training_pack=base / "missing-pack.json",
                    foundation_dir=base / "foundations", run_root=base / "runs",
                    seal_path=base / "missing-seal.json",
                    evaluation_contract=base / "missing-contract.json", test_path=base / "missing.parquet",
                    output_path=base / "result.json")
            gate.assert_called_once()
            reader.assert_not_called()

    def test_evaluate_uses_frozen_master_runtime_cfg_and_budget_sentinel(self):
        """Exercise the whole evaluator path without loading a model or test payload."""
        import tempfile
        from tools import run_clinc_cil_v8_full as full

        task_root = Path(__file__).resolve().parents[1]
        for entry in (str(task_root / "src"), str(task_root / "tools")):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        import clinc_tail_runtime_v8 as tail_runtime
        master_path = task_root / "configs" / "clinc_cil_v8_full_master_frozen_k10.json"
        master = json.loads(master_path.read_text(encoding="utf-8"))
        source_hashes = {"runner": "7" * 64}
        scores = {"accuracy": 1.0, "invalid_predictions": 0, "n": 4500}
        classes = tuple(f"class_{i}" for i in range(150))
        view = SimpleNamespace(classes=classes,
            class_orders={str(seed): list(range(150)) for seed in (17, 29, 43)})

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def logits(self, tokens, mask=None):
                return torch.zeros((tokens.shape[0], 2), dtype=torch.float32)

        class FakeTokenizer:
            eos_token_id = 1
            def encode(self, text, add_special_tokens=False):
                return [1]

        class FakeContext:
            def __init__(self):
                self.memory = torch.nn.ModuleList()
                self.state = {}
            def load_state_dict(self, state, strict=True):
                self.state = state
            def parameters(self):
                return iter(())

        with tempfile.TemporaryDirectory(dir=task_root) as temporary:
            root = Path(temporary)
            pack = root / "training-pack.json"
            pack.write_text("{}", encoding="utf-8")
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps({"schema": "test"}), encoding="utf-8")
            test_path = root / "locked-test.parquet"
            test_path.write_bytes(b"not opened by this sentinel")
            output = root / "result.json"
            verified = []
            for seed in (17, 29, 43):
                for arm in ("HOPE", "HOPE_SLEEP_FIXEDMIX"):
                    run = root / f"{seed}-{arm}"
                    run.mkdir()
                    receipt_path, checkpoint_path = run / "receipt.json", run / "final.pt"
                    receipt_path.write_bytes(b"receipt")
                    checkpoint_path.write_bytes(b"checkpoint")
                    cp_sha = evaluator.sha256_file(checkpoint_path)
                    verified.append({"seed": seed, "arm": arm,
                        "receipt_path": str(receipt_path),
                        "receipt_sha256": evaluator.sha256_file(receipt_path),
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": cp_sha,
                        "receipt": {"final_checkpoint_sha256": cp_sha,
                            "foundation_checkpoint_sha256": "f" * 64}})
            gate = {"passed": True, "test_payload_read": False,
                "verified_receipts": verified, "training_exposure_audit": [
                    {"seed": seed, "arm": arm, "sleep_events": 0,
                     "physical_sleep_adamw_steps": 0,
                     "sleep_text_replay_derived_from_counters": False}
                    for seed in (17, 29, 43)
                    for arm in ("HOPE", "HOPE_SLEEP_FIXEDMIX")],
                "master_config_sha256": "a" * 64,
                "training_manifest_sha256": "b" * 64,
                "source_hashes": source_hashes, "source_lock_sha256": "c" * 64,
                "seal_sha256": "d" * 64,
                "evaluator_source_sha256": evaluator.sha256_file(Path(evaluator.__file__)),
                "evaluation_contract_sha256": evaluator.sha256_file(contract_path),
                "evaluation_contract_path": str(contract_path)}

            def fake_predictions(model, context, rows, tokenizer, pred_view, cfg,
                                 torch_module, device, budget):
                tokens = torch.ones((1, 2), dtype=torch.long)
                model.logits(tokens, torch.ones_like(tokens))
                return ([classes[0]] * len(rows), [0] * len(rows), {})

            with mock.patch.object(evaluator, "verify_release_gate", return_value=gate), \
                 mock.patch.object(evaluator, "assert_gate_inputs_unchanged") as recheck, \
                 mock.patch.object(evaluator, "_read_test_rows",
                    return_value=([{"text": f"row-{i}", "intent_id": i % 150}
                                   for i in range(4500)],
                                  {"all_rows": 5500, "primary_rows": 4500,
                                   "oos_excluded_rows": 1000})), \
                 mock.patch.object(evaluator, "_load_full_runner", return_value=full), \
                 mock.patch.object(full, "_source_hashes", return_value=source_hashes), \
                 mock.patch.object(full, "_load_safe_checkpoint", return_value={
                    "cursor": {"tasks_completed": 10, "task_id": 9, "next_step": 189,
                               "safe_point": "post_all_due_events"}, "context": {}}), \
                 mock.patch.object(r2, "_load_common", return_value=(
                    torch, FakeTokenizer(), FakeModel(),
                    lambda model: {"frozen": (0, 0)})), \
                 mock.patch.object(r2, "_greedy_predictions", side_effect=fake_predictions), \
                 mock.patch.object(r2, "_mutable_context_sha", return_value="context-hash"), \
                 mock.patch.object(tail_runtime, "clone_mutable_context",
                                   side_effect=lambda model: FakeContext()), \
                 mock.patch("tools.clinc_cil_common_v8.make_training_view", return_value=view), \
                 mock.patch("tools.clinc_cil_common_v8.iter_role", return_value=iter(())), \
                 mock.patch("tools.clinc_cil_common_v8.score_predictions", return_value=scores):
                result = evaluator.evaluate(project_root=task_root, model_path=root,
                    master_config=master_path, training_pack=pack,
                    foundation_dir=root / "foundations", run_root=root / "runs",
                    seal_path=root / "seal.json", evaluation_contract=contract_path,
                    test_path=test_path, output_path=output)

            self.assertEqual(result["runtime_config"]["device"], "cuda:0")
            self.assertEqual(result["runtime_config"]["batch_size"], 8)
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual(result["backward_calls"], 0)
            self.assertEqual(result["work"]["full_backbone_forward_calls"], 6)
            self.assertEqual(len(result["runs"]), 6)
            self.assertEqual(recheck.call_count, 2)
            progress = output.with_name("result.progress.jsonl")
            entries = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(entries), 8)
            self.assertEqual(entries[-1]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
