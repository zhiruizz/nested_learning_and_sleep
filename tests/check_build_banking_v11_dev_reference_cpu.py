import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_banking_v11_dev_reference as builder
import evaluate_banking_lti_ablation_v11 as evaluator
import run_banking_lti_ablation_v11 as v11_driver
import run_banking_cil_v10 as v10_driver


def _sha(ch):
    return ch * 64


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_bytes(payload)


@pytest.fixture
def artifact_grid(tmp_path):
    v10_root, v11_root = tmp_path / "v10", tmp_path / "v11"
    foundation = {str(seed): {"receipt_sha256": _sha("a"),
                              "checkpoint_sha256": _sha("b")}
                  for seed in evaluator.SEEDS}
    baseline_runs = {}
    for seed in evaluator.SEEDS:
        for arm in evaluator.V10_ARMS:
            run_dir = v10_root / f"seed{seed}-{arm}"
            cp = run_dir / "checkpoints" / "task01-final.pt"
            _write(run_dir / "protocol.json", {"seed": seed, "arm": arm})
            _write(cp, f"v10-{seed}-{arm}".encode())
            _write(run_dir / "run-receipt.json", {
                "task_metrics": [{"task_id": 1, "global_wake_tick": 123,
                                  "checkpoint_sha256": builder.sha256_file(cp)}]})
            baseline_runs[(seed, arm)] = {
                "run_dir": str(run_dir), "task1_checkpoint": "checkpoints/task01-final.pt",
                "task1_checkpoint_sha256": builder.sha256_file(cp),
                "task1_metrics": {"candidate": 0.5}}
    for seed in evaluator.SEEDS:
        for arm in evaluator.V11_ARMS:
            run_dir = builder._v11_run_dir(v11_root, seed, arm)
            cp = run_dir / "checkpoints" / "task01-final.pt"
            _write(run_dir / "protocol.json", {"seed": seed, "arm": arm})
            _write(cp, f"v11-{seed}-{arm}".encode())
            _write(run_dir / "run-receipt.json", {
                "task_metrics": [{"task_id": 1, "global_wake_tick": 456,
                                  "checkpoint_sha256": builder.sha256_file(cp)}]})
    baseline = {"runs": baseline_runs}
    rows = builder.assemble_context_rows(v10_root=v10_root, v11_root=v11_root,
        baseline=baseline, foundation_refs=foundation, seal_sha256=_sha("c"))
    ref = {
        "schema": evaluator.REFERENCE_SCHEMA, "status": "frozen",
        "seeds": list(evaluator.SEEDS), "arms": list(evaluator.ARMS),
        "model_revision": "pinned-rev", "model_assets_sha256": {"asset": _sha("d")},
        "training_manifest_sha256": _sha("e"),
        "evaluation_policy": {
            "cohort": "all_10_development_rows_per_class_in_seed_task0_and_task1_blocks",
            "rows_per_seed": 220, "classes_per_seed": 22,
            "primary": "global_77_sum_log_likelihood",
            "secondary": ["global_77_length_mean_log_likelihood", "unconstrained_greedy_generation"],
            "candidate_label_mask": None, "max_sequence_length": 128,
            "candidate_batch_size": 32, "max_new_tokens": 10,
            "max_backbone_forwards": 50000, "max_wall_seconds": 7200.0,
            "optimizer_updates": 0,
        },
        "v10_config_file_sha256": _sha("f"), "v10_config_sha256": _sha("1"),
        "v10_source_lock_sha256": _sha("2"), "v10_source_hashes": {"src/a.py": _sha("3")},
        "v10_grid_seal_sha256": _sha("c"), "v10_release_sha256": _sha("4"),
        "foundation_refs": foundation,
        "v11_config_file_sha256": _sha("5"), "v11_config_sha256": _sha("6"),
        "v11_source_lock_sha256": _sha("7"), "v11_source_hashes": {"src/a.py": _sha("8")},
        "evaluator_sha256": _sha("9"), "contexts": rows,
    }
    return ref, baseline_runs


def test_builds_exact_fifteen_contexts_with_fixed_task1_checkpoints(artifact_grid):
    reference, _ = artifact_grid
    rows = evaluator.validate_reference(reference)
    assert len(rows) == 15
    assert {(r["seed"], r["family"], r["arm"]) for r in rows} == {
        (seed, family, arm) for seed in evaluator.SEEDS
        for family, arms in (("v10", evaluator.V10_ARMS), ("v11", evaluator.V11_ARMS))
        for arm in arms}
    assert {r["step"] for r in rows if r["family"] == "v10"} == {123}
    assert {r["step"] for r in rows if r["family"] == "v11"} == {456}


def test_rejects_missing_v10_context(artifact_grid, tmp_path):
    reference, baseline_runs = artifact_grid
    baseline_runs.pop((17, "CMS_ONLY"))
    with pytest.raises(ValueError, match="missing sealed V10 run"):
        builder.assemble_context_rows(v10_root=tmp_path / "v10",
            v11_root=tmp_path / "v11", baseline={"runs": baseline_runs},
            foundation_refs=reference["foundation_refs"],
            seal_sha256=reference["v10_grid_seal_sha256"])


def test_binds_v11_rows_to_same_seed_v10_sleep_endpoint(artifact_grid):
    reference, baseline_runs = artifact_grid
    for row in (r for r in reference["contexts"] if r["family"] == "v11"):
        expected = baseline_runs[(row["seed"], "SLEEP_ADDITIVE")]["task1_checkpoint_sha256"]
        assert row["comparison_v10_sleep_task1_checkpoint_sha256"] == expected
    evaluator.validate_reference(reference)


def test_rejects_shared_foundation_mismatch_within_seed(artifact_grid):
    reference, _ = artifact_grid
    row = next(r for r in reference["contexts"] if r["seed"] == 29 and r["family"] == "v11")
    row["foundation_checkpoint_sha256"] = _sha("0")
    with pytest.raises(ValueError, match="share one qualified foundation"):
        evaluator.validate_reference(reference)


def test_locked_preflight_passes_source_map_to_required_baseline_keyword(tmp_path, monkeypatch):
    class ReachedBaseline(Exception):
        pass

    manifest = tmp_path / "manifest.json"
    v10_config_path = tmp_path / "v10.json"
    v11_config_path = tmp_path / "v11.json"
    seal_path = tmp_path / "seal.json"
    release_path = tmp_path / "release.json"
    for path, value in ((manifest, {}), (v10_config_path, {}),
                        (v11_config_path, {}), (seal_path, {"seal": True}),
                        (release_path, {"release": True})):
        _write(path, value)

    manifest_sha = evaluator.sha256_file(manifest)
    config_sha, source_lock, v11_config_sha, v11_source_lock = (
        _sha("a"), _sha("b"), _sha("c"), _sha("d"))
    v10_sources, v11_sources = {"src/locked.py": _sha("e")}, {"src/locked.py": _sha("f")}
    foundations = {"17": {"receipt_sha256": _sha("1"),
                           "checkpoint_sha256": _sha("2")}}
    assets = {"asset": _sha("3")}
    v10_cfg = {"training_manifest_sha256": manifest_sha,
               "model_revision": "revision", "model_assets_sha256": assets}
    v11_cfg = {**v10_cfg, "v10_config_sha256": config_sha,
               "v10_source_lock_sha256": source_lock}
    monkeypatch.setattr("banking_cil_common_v10.validate_manifest", lambda _: None)
    monkeypatch.setattr(v10_driver, "validate_config", lambda *_a, **_k: v10_cfg)
    monkeypatch.setattr(v10_driver, "_config_sha", lambda _: config_sha)
    monkeypatch.setattr(v10_driver, "_source_hashes", lambda *_a: v10_sources)
    monkeypatch.setattr(v10_driver, "_verify_source_lock", lambda *_a: None)
    monkeypatch.setattr(v10_driver, "_verify_model_assets", lambda *_a: None)
    monkeypatch.setattr(v11_driver, "candidate_source_hashes", lambda *_a: v11_sources)
    monkeypatch.setattr(v11_driver, "validate_config", lambda *_a, **_k: v11_cfg)
    monkeypatch.setattr(v11_driver, "_config_hash", lambda _: v11_config_sha)
    monkeypatch.setattr(evaluator, "verify_source_map", lambda *_a, **_k: None)
    monkeypatch.setattr(evaluator, "canonical_sha",
                        lambda value: source_lock if value == v10_sources else v11_source_lock)
    monkeypatch.setattr(evaluator, "_verify_model_assets", lambda *_a: None, raising=False)

    import evaluate_banking_test_v10 as test_evaluator
    monkeypatch.setattr(test_evaluator, "verify_foundation_grid",
        lambda *_a, **_k: {"foundations": foundations})
    seen = {}

    def strict_baseline(_seal, _runs, *, expected_seal_sha256, config_sha256,
                        source_lock_sha256, manifest_sha256, source_hashes):
        seen.update(expected_seal_sha256=expected_seal_sha256,
                    config_sha256=config_sha256, source_lock_sha256=source_lock_sha256,
                    manifest_sha256=manifest_sha256, source_hashes=source_hashes)
        raise ReachedBaseline

    monkeypatch.setattr(v11_driver, "validate_baseline_seal", strict_baseline)
    reference = {
        "evaluator_sha256": evaluator.sha256_file(evaluator.__file__),
        "training_manifest_sha256": manifest_sha,
        "v10_config_file_sha256": evaluator.sha256_file(v10_config_path),
        "v10_config_sha256": config_sha, "v10_source_hashes": v10_sources,
        "v10_source_lock_sha256": source_lock,
        "v11_config_file_sha256": evaluator.sha256_file(v11_config_path),
        "v11_config_sha256": v11_config_sha, "v11_source_hashes": v11_sources,
        "v11_source_lock_sha256": v11_source_lock,
        "v10_grid_seal_sha256": evaluator.sha256_file(seal_path),
        "v10_release_sha256": evaluator.sha256_file(release_path),
        "foundation_refs": foundations, "model_revision": "revision",
        "model_assets_sha256": assets,
    }
    with pytest.raises(ReachedBaseline):
        evaluator.validate_locked_inputs(reference=reference,
            evaluator_path=evaluator.__file__, v10_config_path=v10_config_path,
            v11_config_path=v11_config_path, manifest_path=manifest,
            v10_seal_path=seal_path, v10_release_path=release_path,
            v10_runs_root=tmp_path / "v10-runs", v11_runs_root=tmp_path / "v11-runs",
            foundation_root=tmp_path / "foundations", project_root=tmp_path / "project",
            model_dir=tmp_path / "model", task_root=ROOT, torch=object())
    assert seen == {"expected_seal_sha256": reference["v10_grid_seal_sha256"],
                    "config_sha256": config_sha, "source_lock_sha256": source_lock,
                    "manifest_sha256": manifest_sha, "source_hashes": v10_sources}


def _real_v11_terminal_context(artifact_grid):
    reference, baseline_runs = artifact_grid
    row = next(r for r in reference["contexts"]
               if r["family"] == "v11" and r["seed"] == 17
               and r["arm"] == "LTI_NO_REWARD")
    v10_root = Path(baseline_runs[(17, "CMS_ONLY")]["run_dir"]).parents[0]
    v11_root = v10_root.parent / "v11"
    cp_path = v11_root / row["checkpoint_path"]
    run_dir = cp_path.parent.parent
    foundation = reference["foundation_refs"]["17"]
    paired_sleep = baseline_runs[(17, "SLEEP_ADDITIVE")]["task1_checkpoint_sha256"]
    protocol = {
        "schema": "banking-cil-v11-protocol-v1", "status": "complete",
        "config_sha256": reference["v11_config_sha256"],
        "source_lock_sha256": reference["v11_source_lock_sha256"],
        "v10_source_lock_sha256": reference["v10_source_lock_sha256"],
        "v10_config_sha256": reference["v10_config_sha256"],
        "training_manifest_sha256": reference["training_manifest_sha256"],
        "baseline_training_seal_sha256": reference["v10_grid_seal_sha256"],
        "foundation_receipt_sha256": foundation["receipt_sha256"],
        "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
        "comparison_v10_sleep_task1_checkpoint_sha256": paired_sleep,
        "seed": 17, "arm": "LTI_NO_REWARD", "task_id": 1, "epochs": 2,
        "test_data_included": False, "test_data_read": False,
        "source_hashes": reference["v11_source_hashes"],
    }
    receipt = {
        "schema": "banking-cil-v11-run-receipt-v1", "status": "complete",
        "seed": 17, "arm": "LTI_NO_REWARD", "task_id": 1, "epochs": 2,
        "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
        "foundation_receipt_sha256": foundation["receipt_sha256"],
        "baseline_training_seal_sha256": reference["v10_grid_seal_sha256"],
        "comparison_v10_sleep_task1_checkpoint_sha256": paired_sleep,
        "checkpoint": "checkpoints/task01-final.pt",
        "checkpoint_sha256": row["checkpoint_sha256"],
        "config_sha256": reference["v11_config_sha256"],
        "source_lock_sha256": reference["v11_source_lock_sha256"],
        "v10_source_lock_sha256": reference["v10_source_lock_sha256"],
        "training_manifest_sha256": reference["training_manifest_sha256"],
        "model_immutable_versions_unchanged": True,
        "test_data_included": False, "test_data_read": False,
        "task1_wake_ticks": row["step"],
        "task_metrics": [{"task_id": 1, "checkpoint_sha256": row["checkpoint_sha256"],
                          "task_step": row["step"], "global_wake_tick": row["step"],
                          "evaluation_rows": evaluator.TASK01_ROWS}],
    }
    _write(run_dir / "protocol.json", protocol)
    _write(run_dir / "run-receipt.json", receipt)
    row["protocol_sha256"] = builder.sha256_file(run_dir / "protocol.json")
    row["receipt_sha256"] = builder.sha256_file(run_dir / "run-receipt.json")
    return reference, row, {"runs": baseline_runs}, run_dir, v10_root, v11_root


def test_verifies_real_complete_v11_protocol_and_receipt_shape(artifact_grid):
    reference, row, baseline, run_dir, v10_root, v11_root = _real_v11_terminal_context(artifact_grid)
    verified = evaluator.verify_context_artifacts([row],
        v10_root=v10_root, v11_root=v11_root,
        reference=reference, v10_baseline=baseline)
    assert len(verified) == 1


def test_rejects_running_v11_protocol_status(artifact_grid):
    reference, row, baseline, run_dir, v10_root, v11_root = _real_v11_terminal_context(artifact_grid)
    protocol_path = run_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["status"] = "running"
    _write(protocol_path, protocol)
    row["protocol_sha256"] = builder.sha256_file(protocol_path)
    with pytest.raises(ValueError, match="V11 task-1 protocol/source/foundation/baseline binding mismatch"):
        evaluator.verify_context_artifacts([row], v10_root=v10_root,
            v11_root=v11_root, reference=reference, v10_baseline=baseline)


def test_rejects_incorrect_v11_receipt_comparison_checkpoint(artifact_grid):
    reference, row, baseline, run_dir, v10_root, v11_root = _real_v11_terminal_context(artifact_grid)
    receipt_path = run_dir / "run-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["comparison_v10_sleep_task1_checkpoint_sha256"] = _sha("0")
    _write(receipt_path, receipt)
    row["receipt_sha256"] = builder.sha256_file(receipt_path)
    with pytest.raises(ValueError, match="V11 task-1 receipt/metrics endpoint mismatch"):
        evaluator.verify_context_artifacts([row], v10_root=v10_root,
            v11_root=v11_root, reference=reference, v10_baseline=baseline)
