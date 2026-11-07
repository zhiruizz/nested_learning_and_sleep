"""CPU-only tests for the dedicated official-test memory launcher."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import launch_banking_test_memory_policy_v10 as launcher


H = "a" * 64


def _contract(tmp_path: Path):
    source_hashes = {f"source/{i:02d}.py": f"{i + 1:064x}" for i in range(24)}
    pack = tmp_path / "verified-train-dev-pack" / "manifest.json"
    main_root = tmp_path / "sealed-main-runs"
    raw_train = tmp_path / "datasets" / "banking77-original-9d08145" / "raw" / "train.csv"
    return {
        "schema": launcher.SCHEMA, "launcher_sha256": H,
        "project_root": str(tmp_path / "project"), "task_root": str(tmp_path / "task"),
        "model_dir": str(tmp_path / "model"), "training_pack": str(pack), "training_pack_sha256": H,
        "config_path": str(tmp_path / "config.json"), "config_file_sha256": H,
        "config_sha256": H, "source_lock_sha256": H, "source_hashes": source_hashes,
        "training_manifest_path": str(pack), "training_manifest_sha256": H,
        "training_root": str(main_root), "seal_path": str(main_root / "training-seal.json"),
        "seal_sha256": H, "main_release_path": str(main_root / "main-release.json"),
        "main_release_sha256": H, "foundation_root": str(tmp_path / "foundations"),
        "evaluation_contract_path": str(tmp_path / "evaluation-contract.json"),
        "evaluation_contract_sha256": H, "evaluator_sha256": H,
        "contract_builder_sha256": H, "publisher_train_csv_path": str(raw_train),
        "publisher_train_csv_sha256": launcher.TRAIN_SOURCE_SHA256,
        "publisher_train_csv_bytes": launcher.TRAIN_SOURCE_BYTES,
        "test_csv_path": str(raw_train.parent / "test.csv"),
        "test_csv_expected_sha256": launcher.TEST_SOURCE_SHA256,
        "test_csv_expected_bytes": launcher.TEST_SOURCE_BYTES,
        "test_csv_revision": launcher.TEST_SOURCE_REVISION,
        "preflight_output_dir": str(tmp_path / "outputs" / "preflight"),
        "output_dir": str(tmp_path / "outputs" / "evaluation"),
        "launch_record_path": str(tmp_path / "records" / "launch.json"),
        "device": "cuda:0", "allocator_conf": launcher.ALLOCATOR_CONF,
        "memory_fraction": launcher.MEMORY_FRACTION, "torch_version": launcher.TORCH_VERSION,
    }


def test_evaluator_argv_uses_main_root_not_training_pack_parent(tmp_path):
    contract = _contract(tmp_path)
    assert Path(contract["training_root"]) != Path(contract["training_pack"]).parent
    preflight = launcher.evaluator_argv(contract, mode="preflight")
    assert preflight[preflight.index("--training-root") + 1] == str(Path(contract["training_root"]).resolve())
    assert "--test-csv" not in preflight
    assert preflight[preflight.index("--output-dir") + 1] == str(Path(contract["preflight_output_dir"]).resolve())
    evaluate = launcher.evaluator_argv(contract, mode="evaluate")
    assert evaluate[evaluate.index("--training-root") + 1] == str(Path(contract["training_root"]).resolve())
    assert evaluate[evaluate.index("--test-csv") + 1] == str(Path(contract["test_csv_path"]).resolve())
    launcher.validate_test_csv_location(contract["publisher_train_csv_path"], contract["test_csv_path"])
    with pytest.raises(ValueError, match="sibling"):
        launcher.validate_test_csv_location(contract["publisher_train_csv_path"], tmp_path / "elsewhere.csv")


def test_contract_requires_exact_source_graph_and_fixed_deferred_policy(tmp_path):
    contract = _contract(tmp_path)
    launcher.validate_contract_shape(contract, launcher_sha256=H)
    assert len(contract["source_hashes"]) == 24
    contract["memory_fraction"] = 0.9
    with pytest.raises(ValueError, match="fixed process-only policy"):
        launcher.validate_contract_shape(contract, launcher_sha256=H)


def test_contract_rejects_missing_source_pin(tmp_path):
    contract = _contract(tmp_path)
    del contract["source_hashes"]["source/00.py"]
    with pytest.raises(ValueError, match="exact 24-file"):
        launcher.validate_contract_shape(contract, launcher_sha256=H)


def test_real_evaluator_validator_receives_logical_config_hash(tmp_path):
    from tools import build_banking_test_contract_v10 as builder
    from tools import evaluate_banking_test_v10 as evaluator

    source_hashes = {f"source/{i:02d}.py": f"{i + 1:064x}" for i in range(24)}
    cfg = {"_logical_sha256": H, "source_hashes": source_hashes,
           "model_revision": "locked-revision", "model_assets_sha256": {"weights": H},
           "max_sequence_length": 128}
    refs = {str(seed): {"receipt_sha256": H, "checkpoint_sha256": H} for seed in (17, 29, 43)}
    evaluation_contract = builder.make_contract(
        evaluator_sha256=H, config=cfg, config_file_sha256=H, source_lock_sha256=H,
        training_manifest_sha256=H, training_seal_sha256=H, main_release_sha256=H,
        foundation_refs=refs, evaluator=evaluator)
    launch_contract = _contract(tmp_path)
    # Mirrors driver.validate_config output, which does not include the private
    # _logical_sha256 key expected by the evaluator validator.
    driver_cfg = {key: value for key, value in cfg.items() if key != "_logical_sha256"}
    launcher._validate_eval_contract_binding(evaluator, evaluation_contract, driver_cfg, launch_contract)


def _sealed_grid():
    seal_rows, preflight_rows, terminal_rows = [], [], []
    for seed in (17, 29, 43):
        for arm in ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE"):
            checkpoint = f"{seed:02d}" * 32
            receipt = f"{seed + 1:02d}" * 32
            seal_rows.append({"seed": seed, "arm": arm,
                              "final_checkpoint_sha256": checkpoint,
                              "run_receipt_sha256": receipt})
            preflight_rows.append({"seed": seed, "arm": arm,
                                   "checkpoint_sha256": checkpoint,
                                   "receipt_sha256": receipt, "task_id": 6, "step": 1000})
            terminal_rows.append({"seed": seed, "arm": arm,
                                  "checkpoint_sha256": checkpoint,
                                  "receipt_sha256": receipt})
    return {"runs": seal_rows}, preflight_rows, terminal_rows


def test_preflight_and_terminal_gates_require_exact_nine_and_safe_flags(tmp_path):
    from types import SimpleNamespace

    contract = _contract(tmp_path)
    seal, verified, final_contexts = _sealed_grid()
    foundations = {"release_sha256": H,
        "foundations": {str(s): {"receipt_sha256": H, "checkpoint_sha256": H} for s in (17, 29, 43)},
        "all_qualified": True, "all_checkpoints_cpu_deserialized": True}
    evaluator = SimpleNamespace(SCHEMA="banking-cil-v10-test-evaluator-v1")
    preflight = {"schema": evaluator.SCHEMA, "status": "pass", "mode": "preflight",
        "training_config_sha256": contract["config_sha256"], "source_lock_sha256": contract["source_lock_sha256"],
        "evaluation_contract_sha256": contract["evaluation_contract_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "seal_sha256": contract["seal_sha256"], "main_release_sha256": contract["main_release_sha256"],
        "foundation_audit": foundations, "test_payload_read": False, "model_loaded": False,
        "checkpoint_audit": {"count": 9, "cpu_deserialized": True, "model_loaded": False,
            "cuda_initialized": False, "optimizer_updates": 0, "verified_runs": verified}}
    launcher._validate_preflight_result(preflight, contract, seal, evaluator, foundations)
    preflight["checkpoint_audit"]["cuda_initialized"] = True
    with pytest.raises(RuntimeError, match="nine checkpoints"):
        launcher._validate_preflight_result(preflight, contract, seal, evaluator, foundations)

    preflight["checkpoint_audit"]["cuda_initialized"] = False
    training = {key: contract[key] for key in (
        "config_sha256", "source_lock_sha256", "training_manifest_sha256", "seal_sha256",
        "evaluation_contract_sha256", "evaluator_sha256", "main_release_sha256")}
    training.update(foundation_audit=foundations, verified_contexts=final_contexts)
    result = {"schema": evaluator.SCHEMA, "status": "complete", "training": training,
        "optimizer_updates": 0, "all_contexts_frozen": True, "backbone_unchanged": True,
        "raw_prompts_predictions_and_token_ids_saved": False,
        "test_source": {"source_revision": launcher.TEST_SOURCE_REVISION,
            "source_sha256": launcher.TEST_SOURCE_SHA256, "source_bytes": launcher.TEST_SOURCE_BYTES,
            "raw_rows": 3080, "classes": 77, "rows_per_class": 40}}
    launcher._validate_terminal_result(result, contract, seal, preflight, evaluator)
    result["raw_prompts_predictions_and_token_ids_saved"] = True
    with pytest.raises(RuntimeError, match="safety flags"):
        launcher._validate_terminal_result(result, contract, seal, preflight, evaluator)


def test_memory_fraction_callback_is_deferred_until_cuda_initialization():
    class FakeCuda:
        def __init__(self):
            self.initialized = False
            self.callback = None
            self.fraction_calls = []

        def is_initialized(self):
            return self.initialized

        def _lazy_call(self, callback):
            assert not self.initialized
            self.callback = callback

        def set_per_process_memory_fraction(self, fraction, *, device):
            self.fraction_calls.append((fraction, device))

    class FakeTorch:
        def __init__(self):
            self.cuda = FakeCuda()

    torch = FakeTorch()
    state = launcher._install_memory_policy(torch, "cuda:0")
    assert state == {"registered": True, "called": False, "error": None}
    assert torch.cuda.fraction_calls == []
    torch.cuda.callback()
    assert state["called"] is True
    assert torch.cuda.fraction_calls == [(0.8, "cuda:0")]

