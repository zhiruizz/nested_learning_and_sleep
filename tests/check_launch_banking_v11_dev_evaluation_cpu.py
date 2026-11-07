import sys
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate_banking_lti_ablation_v11 as evaluator
import launch_banking_v11_dev_evaluation as launcher


def _sha(ch):
    return ch * 64


@pytest.fixture
def contract_and_reference(tmp_path):
    source_hashes = {key: _sha(str(i % 10)) for i, key in enumerate(launcher.EXPECTED_SOURCES)}
    contract = {
        "schema": launcher.SCHEMA, "status": "frozen", "task_root": str(ROOT),
        "project_root": str(tmp_path), "model_dir": str(tmp_path),
        "training_pack_root": str(tmp_path), "training_manifest_path": str(tmp_path / "manifest"),
        "v10_config_path": str(tmp_path / "v10cfg"), "v11_config_path": str(tmp_path / "v11cfg"),
        "v10_grid_seal_path": str(tmp_path / "seal"), "v10_release_artifact_path": str(tmp_path / "release"),
        "v10_runs_root": str(tmp_path), "v11_runs_root": str(tmp_path),
        "foundation_root": str(tmp_path), "reference_path": str(tmp_path / "ref"),
        "preflight_output_dir": str(tmp_path / "preflight"),
        "evaluation_output_dir": str(tmp_path / "evaluation"),
        "record_path": str(tmp_path / "record.json"), "device": "cuda:0",
        "reference_sha256": _sha("a"), "training_manifest_sha256": _sha("b"),
        "v10_config_file_sha256": _sha("c"), "v11_config_file_sha256": _sha("d"),
        "v10_grid_seal_sha256": _sha("e"), "v10_release_sha256": _sha("f"),
        "source_hashes": source_hashes, "model_assets_sha256": {"model.safetensors": _sha("0")},
    }
    reference = {
        "evaluator_sha256": source_hashes["tools/evaluate_banking_lti_ablation_v11.py"],
        "reference_sha256": contract["reference_sha256"],
        "v10_config_sha256": _sha("1"), "v11_config_sha256": _sha("2"),
    }
    return contract, reference


def _valid_result(contract, reference):
    return {
        "schema": "banking-lti-ablation-v11-dev-evaluator-v1", "status": "complete",
        "protocol": {"seeds": list(evaluator.SEEDS), "contexts_per_seed": 5,
            "arms": list(evaluator.ARMS), "rows_per_seed": 220,
            "official_test_accessed": False, "raw_examples_predictions_and_token_ids_saved": False},
        "frozen_inputs": {
            "reference_sha256": contract["reference_sha256"],
            "evaluator_sha256": reference["evaluator_sha256"],
            "v10_config_sha256": reference["v10_config_sha256"],
            "v11_config_sha256": reference["v11_config_sha256"],
            "v10_grid_seal_sha256": contract["v10_grid_seal_sha256"],
            "v10_release_sha256": contract["v10_release_sha256"],
            "training_manifest_sha256": contract["training_manifest_sha256"],
        },
        "development": {str(seed): {"cohort_rows": 220, "context_count": 5,
            "candidate_metrics": {arm: {} for arm in evaluator.ARMS},
            "greedy_generation_metrics": {arm: {} for arm in evaluator.ARMS}}
            for seed in evaluator.SEEDS},
        "observed_model_logits_calls": 1234, "optimizer_updates": 0,
        "contexts_unchanged": True, "frozen_backbone_unchanged": True,
    }


def test_preflight_receipt_matches_real_evaluator_api_shape(contract_and_reference):
    _, reference = contract_and_reference
    reference["_actual_reference_sha256"] = reference["reference_sha256"]
    launcher.validate_preflight_receipt({
        "schema": evaluator.SCHEMA, "status": "pass", "mode": "preflight",
        "context_count": 15, "v10_terminal_checkpoint_count": 9,
        "cpu_checkpoint_deserialization": True, "model_loaded": False,
        "cuda_initialized": False, "official_test_accessed": False,
        "optimizer_updates": 0, "reference_sha256": reference["reference_sha256"],
        "evaluator_sha256": reference["evaluator_sha256"],
    }, reference=reference)


def test_accepts_complete_three_seed_five_context_evaluation(contract_and_reference):
    contract, reference = contract_and_reference
    launcher.validate_evaluation_result(_valid_result(contract, reference),
        contract=contract, reference=reference,
        callback={"registered": True, "called": True, "error": None})


@pytest.mark.parametrize("mutation", ["test_read", "missing_seed", "changed_frozen_input", "callback"])
def test_rejects_incomplete_or_unbound_terminal_result(contract_and_reference, mutation):
    contract, reference = contract_and_reference
    result = _valid_result(contract, reference)
    callback = {"registered": True, "called": True, "error": None}
    if mutation == "test_read":
        result["protocol"]["official_test_accessed"] = True
    elif mutation == "missing_seed":
        del result["development"]["29"]
    elif mutation == "changed_frozen_input":
        result["frozen_inputs"]["v10_release_sha256"] = _sha("9")
    else:
        callback["called"] = False
    with pytest.raises(ValueError):
        launcher.validate_evaluation_result(result, contract=contract,
            reference=reference, callback=callback)


def test_preflight_and_evaluation_only_change_mode_and_fresh_output(contract_and_reference, tmp_path):
    contract, _ = contract_and_reference
    before = launcher.evaluator_argv(contract, mode="preflight", output_dir=tmp_path / "pre")
    after = launcher.evaluator_argv(contract, mode="evaluate", output_dir=tmp_path / "eval")
    def normalized(argv):
        result = dict(zip(argv[::2], argv[1::2]))
        result["--mode"] = "<MODE>"
        result["--output-dir"] = "<OUTPUT>"
        return result
    assert normalized(before) == normalized(after)
    assert "--training-pack-root" in before and "--v10-grid-seal" in before


def test_stale_launcher_self_hash_fails_source_gate_before_runtime(tmp_path):
    source_hashes = {}
    for relative in launcher.EXPECTED_SOURCES:
        path = ROOT / relative
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    launcher_key = "tools/launch_banking_v11_dev_evaluation.py"
    source_hashes[launcher_key] = _sha("f")
    with pytest.raises(ValueError, match="pinned source changed: tools/launch_banking_v11_dev_evaluation.py"):
        launcher.verify_source_hashes(ROOT, source_hashes)


def test_unpinned_foundation_test_helper_change_fails_source_gate():
    source_hashes = {}
    for relative in launcher.EXPECTED_SOURCES:
        source_hashes[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    helper = "tools/evaluate_banking_test_v10.py"
    source_hashes[helper] = _sha("f")
    with pytest.raises(ValueError, match="pinned source changed: tools/evaluate_banking_test_v10.py"):
        launcher.verify_source_hashes(ROOT, source_hashes)
