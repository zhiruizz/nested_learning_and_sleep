import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "tools"))
import launch_banking_memory_policy_v11 as launcher
import banking_cil_v10_runtime as banking_runtime


def sha(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def make_complete_test_fixture(contract):
    grid = [(seed, arm) for seed in (17, 29, 43)
            for arm in ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")]
    seal_rows = []
    verified = []
    primary = {}
    for seed, arm in grid:
        cp_sha, receipt_sha = sha(f"cp-{seed}-{arm}"), sha(f"receipt-{seed}-{arm}")
        seal_rows.append({"seed": seed, "arm": arm, "status": "complete", "tasks_completed": 7,
            "final_checkpoint": "checkpoints/final-task06.pt", "final_checkpoint_sha256": cp_sha,
            "run_receipt_sha256": receipt_sha, "config_sha256": sha("v10-config"),
            "source_lock_sha256": contract["v10_source_lock_sha256"],
            "training_manifest_sha256": contract["training_manifest_sha256"],
            "run_dir": f"seed{seed}/{arm}", "endpoint_metrics": {}, "work": {}})
        verified.append({"seed": seed, "arm": arm, "checkpoint_sha256": cp_sha,
                         "receipt_sha256": receipt_sha})
        primary[f"seed{seed}:{arm}"] = {
            "teacher_forced_sum": {"correct": 1500, "count": 3080},
            "teacher_forced_length_mean": {"correct": 1501, "count": 3080},
            "greedy_generation": {"correct": 1499, "count": 3080, "invalid_predictions": 0},
            "gold_label_token_mean_nll": {"count": 3080, "mean": 1.0}}
    seal = {"schema": "banking-cil-v10-training-seal-v1", "complete": True,
        "config_sha256": sha("v10-config"), "source_lock_sha256": contract["v10_source_lock_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "test_data_included": False, "official_test_read": False, "runs": seal_rows}
    Path(contract["baseline_seal"]).write_text(json.dumps(seal), encoding="utf-8")
    contract["baseline_seal_sha256"] = launcher.sha256_file(contract["baseline_seal"])
    evaluation = {"schema": "banking-cil-v10-test-evaluator-v1", "status": "complete",
        "training": {"config_sha256": sha("v10-config"),
            "source_lock_sha256": contract["v10_source_lock_sha256"],
            "training_manifest_sha256": contract["training_manifest_sha256"],
            "seal_sha256": contract["baseline_seal_sha256"],
            "evaluator_sha256": contract["v10_test_evaluator_sha256"],
            "verified_contexts": verified},
        "test_source": {"source_revision": "9d081458ff52e53cf7e848f414e6e9344e4e6696",
            "source_sha256": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
            "source_bytes": 239961, "raw_rows": 3080, "classes": 77,
            "rows_per_class": 40, "primary_includes_all_official_rows": True},
        "optimizer_updates": 0, "all_contexts_frozen": True, "backbone_unchanged": True,
        "raw_prompts_predictions_and_token_ids_saved": False,
        "primary_raw_official_test": primary}
    Path(contract["v10_test_evaluation"]).write_text(json.dumps(evaluation), encoding="utf-8")
    contract["v10_test_evaluation_sha256"] = launcher.sha256_file(contract["v10_test_evaluation"])
    return evaluation


class FakeV11:
    ARMS = ("LTI_NO_REWARD", "LTI_CANONICAL_TEACHER")
    @staticmethod
    def _load_driver():
        return SimpleNamespace(validate_config=lambda raw, phase: {},
            _config_sha=lambda cfg: sha("v10-config"))
    @staticmethod
    def output_path_guard(path, *, protected_roots):
        assert protected_roots


def make_contract(tmp: Path) -> dict:
    tools = ROOT / "tools"
    paths = {}
    for name in ("project", "model", "pack", "v10_config", "config", "seal",
                 "baseline_runs", "foundation", "test_evaluation"):
        path = tmp / name
        if name in ("project", "model", "baseline_runs", "foundation"):
            path.mkdir()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "v10_config":
                contents = json.dumps({"device": "cuda:0"})
            elif name == "pack":
                contents = json.dumps({"schema": "fixture-pack-manifest-v1", "files": {
                    "train": {"sha256": sha("train-jsonl-bytes"), "rows": 4},
                    "development": {"sha256": sha("development-jsonl-bytes"), "rows": 2}}},
                    sort_keys=True, separators=(",", ":"))
            else:
                contents = "fixture"
            path.write_text(contents, encoding="utf-8")
        paths[name] = str(path)
    evaluator_path = tools / "evaluate_banking_test_v10.py"
    Path(paths["test_evaluation"]).write_text("{}", encoding="utf-8")
    return {
        "schema": launcher.SCHEMA,
        "launcher_sha256": launcher.sha256_file(tools / "launch_banking_memory_policy_v11.py"),
        "policy_launcher_v10_sha256": launcher.sha256_file(tools / "launch_banking_memory_policy_v10.py"),
        "project_root": paths["project"], "task_root": str(ROOT), "model_path": paths["model"],
        "training_pack": paths["pack"], "v10_config_path": paths["v10_config"],
        "config_path": paths["config"], "baseline_seal": paths["seal"],
        "baseline_runs_root": paths["baseline_runs"], "foundation_dir": paths["foundation"],
        "v10_test_evaluation": paths["test_evaluation"],
        "v10_test_evaluation_sha256": launcher.sha256_file(paths["test_evaluation"]),
        "v10_test_evaluator_sha256": launcher.sha256_file(evaluator_path),
        "output_dir": str(tmp / "output"), "seed": 17, "arm": "LTI_NO_REWARD",
        "script_key": launcher.V11_SCRIPT,
        "script_sha256": launcher.sha256_file(tools / launcher.V11_SCRIPT),
        "launch_record_path": str(tmp / "launch-record.json"),
        "allocator_conf": launcher.POLICY, "memory_fraction": launcher.MEMORY_FRACTION,
        "device": "cuda:0", "torch_version": launcher.TORCH_VERSION,
        "config_sha256": sha("config"), "v11_source_lock_sha256": sha("v11-lock"),
        "v10_source_lock_sha256": sha("v10-lock"),
        "training_manifest_sha256": launcher.sha256_file(paths["pack"]),
        "baseline_seal_sha256": sha("seal"), "foundation_checkpoint_sha256": sha("foundation-cp"),
    }


def test_contract_is_exact_and_policy_is_fixed():
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        path = Path(folder) / "contract.json"
        contract = make_contract(Path(folder))
        path.write_text(json.dumps(contract), encoding="utf-8")
        loaded = launcher._load_contract(path, launcher.sha256_file(path))
        assert loaded["arm"] == "LTI_NO_REWARD"
        bad = dict(contract, memory_fraction=0.9)
        path.write_text(json.dumps(bad), encoding="utf-8")
        with __import__("pytest").raises(ValueError, match="fixed memory policy"):
            launcher._load_contract(path, launcher.sha256_file(path))


def test_v11_cpu_preflight_binds_source_config_seal_and_foundation_before_policy():
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        tmp = Path(folder)
        contract = make_contract(tmp)
        fake_eval = make_complete_test_fixture(contract)
        observed = []
        result = {"config_sha256": contract["config_sha256"],
            "source_lock_sha256": contract["v11_source_lock_sha256"],
            "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
            # Frozen V11 producer's legacy output key carries the train JSONL
            # digest, not the pack-manifest file digest pinned by the contract.
            "training_manifest_sha256": sha("train-jsonl-bytes"),
            "baseline_seal_sha256": contract["baseline_seal_sha256"],
            "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
            "seed": 17, "arm": "LTI_NO_REWARD", "test_data_included": False,
            "model_loaded": False, "optimizer_updates": 0}
        class FakeV11ForPreflight(FakeV11):
            @staticmethod
            def _preflight(args):
                observed.append("cpu-preflight")
                assert not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                return result
        checked, _ = launcher._preflight_contract(contract, task_root=ROOT, v11_module=FakeV11ForPreflight)
        assert checked["config_sha256"] == result["config_sha256"]
        assert checked["training_manifest_sha256"] == contract["training_manifest_sha256"]
        assert checked["training_file_sha256"] == result["training_manifest_sha256"]
        assert checked["training_manifest_sha256"] != checked["training_file_sha256"]
        assert checked["v10_test_evaluation_binding"]["official_test_rows"] == 3080
        assert observed == ["cpu-preflight"]
        fake_eval["status"] = "incomplete"
        Path(contract["v10_test_evaluation"]).write_text(json.dumps(fake_eval), encoding="utf-8")
        contract["v10_test_evaluation_sha256"] = launcher.sha256_file(contract["v10_test_evaluation"])
        with __import__("pytest").raises(ValueError, match="completed official-test evaluation"):
            launcher._validate_completed_v10_test(contract, Path(contract["v10_test_evaluation"]),
                task_root=ROOT, v11_module=FakeV11,
                v10_config=Path(contract["v10_config_path"]))
        fake_eval["status"] = "complete"
        for context in fake_eval["primary_raw_official_test"].values():
            context["gold_label_token_mean_nll"]["count"] = 5178
        Path(contract["v10_test_evaluation"]).write_text(json.dumps(fake_eval), encoding="utf-8")
        contract["v10_test_evaluation_sha256"] = launcher.sha256_file(contract["v10_test_evaluation"])
        with __import__("pytest").raises(ValueError, match="3,080 per-example means"):
            launcher._validate_completed_v10_test(contract, Path(contract["v10_test_evaluation"]),
                task_root=ROOT, v11_module=FakeV11,
                v10_config=Path(contract["v10_config_path"]))
        for context in fake_eval["primary_raw_official_test"].values():
            context["gold_label_token_mean_nll"]["count"] = 3080
            context["gold_label_token_mean_nll"]["mean"] = float("inf")
        Path(contract["v10_test_evaluation"]).write_text(json.dumps(fake_eval), encoding="utf-8")
        contract["v10_test_evaluation_sha256"] = launcher.sha256_file(contract["v10_test_evaluation"])
        with __import__("pytest").raises(ValueError, match="finite per-example NLL mean"):
            launcher._validate_completed_v10_test(contract, Path(contract["v10_test_evaluation"]),
                task_root=ROOT, v11_module=FakeV11,
                v10_config=Path(contract["v10_config_path"]))


def test_preflight_rejects_wrong_pack_manifest_or_producer_train_file_hash():
    for bad_binding in ("contract-manifest", "producer-train-file"):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            tmp = Path(folder)
            contract = make_contract(tmp)
            if bad_binding == "contract-manifest":
                contract["training_manifest_sha256"] = sha("wrong-manifest")
            make_complete_test_fixture(contract)

            class FakeProducer:
                ARMS = FakeV11.ARMS
                _load_driver = staticmethod(FakeV11._load_driver)
                output_path_guard = staticmethod(FakeV11.output_path_guard)
                @staticmethod
                def _preflight(_args):
                    return {"config_sha256": contract["config_sha256"],
                        "source_lock_sha256": contract["v11_source_lock_sha256"],
                        "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
                        "training_manifest_sha256": (sha("wrong-train-file") if bad_binding == "producer-train-file"
                                                      else sha("train-jsonl-bytes")),
                        "baseline_seal_sha256": contract["baseline_seal_sha256"],
                        "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
                        "seed": 17, "arm": "LTI_NO_REWARD", "test_data_included": False,
                        "model_loaded": False, "optimizer_updates": 0}

            with __import__("pytest").raises(ValueError, match=(
                    "training pack manifest bytes" if bad_binding == "contract-manifest"
                    else "train-file digest differs")):
                launcher._preflight_contract(contract, task_root=ROOT, v11_module=FakeProducer)

def test_official_test_gate_rejects_missing_context_wrong_denominators_and_hash_crossbinding():
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        root = Path(folder)
        cases = ("missing-context", "wrong-denominator", "wrong-nll-count", "wrong-checkpoint-binding",
                 "wrong-receipt-binding")
        for case in cases:
            case_dir = root / case
            case_dir.mkdir()
            contract = make_contract(case_dir)
            evaluation = make_complete_test_fixture(contract)
            if case == "missing-context":
                evaluation["primary_raw_official_test"].pop("seed43:GOLD_ADDITIVE")
                error = "exactly all nine"
            elif case == "wrong-denominator":
                evaluation["primary_raw_official_test"]["seed29:SLEEP_ADDITIVE"]["greedy_generation"]["count"] = 3079
                error = "must cover 3,080 rows"
            elif case == "wrong-nll-count":
                evaluation["primary_raw_official_test"]["seed29:SLEEP_ADDITIVE"]["gold_label_token_mean_nll"]["count"] = 5178
                error = "3,080 per-example means"
            elif case == "wrong-checkpoint-binding":
                evaluation["training"]["verified_contexts"][0]["checkpoint_sha256"] = sha("other-cp")
                error = "do not match the baseline seal"
            else:
                evaluation["training"]["verified_contexts"][0]["receipt_sha256"] = sha("other-receipt")
                error = "do not match the baseline seal"
            Path(contract["v10_test_evaluation"]).write_text(json.dumps(evaluation), encoding="utf-8")
            contract["v10_test_evaluation_sha256"] = launcher.sha256_file(contract["v10_test_evaluation"])
            with __import__("pytest").raises(ValueError, match=error):
                launcher._validate_completed_v10_test(contract, Path(contract["v10_test_evaluation"]),
                    task_root=ROOT, v11_module=FakeV11,
                    v10_config=Path(contract["v10_config_path"]))


def test_frozen_runtime_nll_counter_counts_per_example_means_not_answer_tokens():
    rows = [{"intent_id": 0, "gold_answer_length": 1},
            {"intent_id": 1, "gold_answer_length": 4},
            {"intent_id": 2, "gold_answer_length": 9}]
    ids = [0, 1, 2]
    _metrics, gold = banking_runtime._score_rows({}, ids, ids, [1.0, 2.0, 3.0], rows,
        tuple(f"class_{index}" for index in range(77)), None)
    assert gold["all77"]["count"] == len(rows) == 3
    assert gold["all77"]["mean_token_nll"] == 2.0


def test_policy_is_queued_after_preflight_and_target_receipt_is_verified():
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        tmp = Path(folder)
        contract = make_contract(tmp)
        contract_path = tmp / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        contract_sha = launcher.sha256_file(contract_path)
        stages = []

        class FakeCuda:
            initialized = False
            queued = []
            def is_initialized(self): return self.initialized
            def _lazy_call(self, callback): self.queued.append(callback)
            def set_per_process_memory_fraction(self, fraction, *, device):
                assert (fraction, device) == (0.8, "cuda:0")
                self.initialized = True
        fake_torch = SimpleNamespace(__version__=launcher.TORCH_VERSION, cuda=FakeCuda())
        preflight = {"comparison_v10_sleep_task1_checkpoint_sha256": sha("v10task1"),
                     "v10_test_evaluation_binding": {"official_test_rows": 3080}}
        def checked_preflight(_contract, *, task_root):
            stages.append("preflight")
            assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") in (None, launcher.POLICY)
            return preflight, None
        class FakePolicyLauncher:
            @staticmethod
            def _install_memory_policy(torch, device):
                assert stages == ["preflight"]
                assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == launcher.POLICY
                state = {"registered": True, "called": False, "error": None}
                def apply():
                    state["called"] = True
                    torch.cuda.set_per_process_memory_fraction(0.8, device=device)
                torch.cuda._lazy_call(apply)
                return state
        def fake_run_path(_script, *, run_name):
            stages.append("target")
            assert run_name == "__main__"
            assert not fake_torch.cuda.initialized
            fake_torch.cuda.queued.pop(0)()
            output = Path(contract["output_dir"])
            (output / "checkpoints").mkdir(parents=True)
            cp = output / "checkpoints" / "task01-final.pt"
            cp.write_bytes(b"checkpoint")
            receipt = {"schema": "banking-cil-v11-run-receipt-v1", "status": "complete",
                "seed": 17, "arm": "LTI_NO_REWARD", "task_id": 1, "task1_wake_ticks": 20,
                "config_sha256": contract["config_sha256"],
                "source_lock_sha256": contract["v11_source_lock_sha256"],
                "v10_source_lock_sha256": contract["v10_source_lock_sha256"],
                "training_manifest_sha256": contract["training_manifest_sha256"],
                "baseline_training_seal_sha256": contract["baseline_seal_sha256"],
                "foundation_checkpoint_sha256": contract["foundation_checkpoint_sha256"],
                "comparison_v10_sleep_task1_checkpoint_sha256": preflight["comparison_v10_sleep_task1_checkpoint_sha256"],
                "epochs": 2,
                "test_data_included": False, "test_data_read": False,
                "model_immutable_versions_unchanged": True,
                "checkpoint": "checkpoints/task01-final.pt",
                "checkpoint_sha256": launcher.sha256_file(cp)}
            (output / "run-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            raise SystemExit(0)

        with mock.patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": ""}, clear=False):
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with mock.patch.object(launcher, "_preflight_contract", checked_preflight), \
                     mock.patch.object(launcher, "_load_v10_policy_launcher", return_value=FakePolicyLauncher), \
                     mock.patch.object(launcher.runpy, "run_path", side_effect=fake_run_path):
                    assert launcher.launch(contract_path, contract_sha) == 0
        assert stages == ["preflight", "target"]
        record = json.loads(Path(contract["launch_record_path"]).read_text(encoding="utf-8"))
        assert record["status"] == "complete"
        assert record["memory_fraction_callback_called"] is True
        assert record["terminal_receipt"]["checkpoint_sha256"] == launcher.sha256_file(
            Path(contract["output_dir"]) / "checkpoints" / "task01-final.pt")
