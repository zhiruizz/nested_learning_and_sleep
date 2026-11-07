import hashlib
import json
import sys
from pathlib import Path

import pytest

from tools import aggregate_banking_v11_lti_accounting as agg


def _forbid_torch_load_if_preloaded(monkeypatch):
    before_torch = sys.modules.get("torch")
    if before_torch is not None:
        def forbidden_load(*args, **kwargs):
            raise AssertionError("offline accounting must not deserialize with torch.load")
        monkeypatch.setattr(before_torch, "load", forbidden_load)
    return before_torch


def _assert_torch_import_state_unchanged(before_torch):
    assert sys.modules.get("torch") is before_torch


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def _make_fixture(tmp_path):
    project = tmp_path / "public"
    task = tmp_path / "task"
    v11_runs = tmp_path / "v11-runs"
    baseline_runs = tmp_path / "v10-runs"
    for p in (project / "src/nested_sleep", project / "tools", task / "tools", v11_runs, baseline_runs):
        p.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "src/nested_sleep/lm.py": project / "src/nested_sleep/lm.py",
        "tools/run_banking_cil_v10.py": project / "tools/run_banking_cil_v10.py",
        "tools/run_banking_lti_ablation_v11.py": task / "tools/run_banking_lti_ablation_v11.py",
    }
    for i, path in enumerate(source_paths.values()):
        path.write_text(f"frozen source {i}\n", encoding="utf-8")
    source_map = {key: _sha(path) for key, path in source_paths.items()}
    v10_cfg = {
        "status": "frozen", "phase": "main", "training_manifest_sha256": "a" * 64,
        "model_revision": "model@revision", "model_assets_sha256": {"weights": "b" * 64},
        "initialization_seed": 1907008, "epochs_per_task": 2, "batch_size": 8,
        "sleep_lr": 0.001, "sleep_weight_decay": 0.0, "sleep_clip_norm": 1.0,
        "lti_semantic_weight": 0.5, "lti_edit_weight": 0.5,
        "task0_global_candidate_gate": 0.5, "task0_class_macro_gate": 0.4,
        "source_hashes": {k: v for k, v in source_map.items() if k != "tools/run_banking_lti_ablation_v11.py"},
    }
    v10_cfg_path = _write(task / "configs/v10.json", v10_cfg)
    v10_config_sha = agg._v10_config_sha(v10_cfg)
    v10_lock = agg.canonical_sha(v10_cfg["source_hashes"])
    foundation_root = tmp_path / "foundations"
    foundation = {}
    release_foundations = {}
    for seed in agg.SEEDS:
        seed_root = foundation_root / f"seed{seed}"
        cp = seed_root / "checkpoints/foundation-final.pt"
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(f"foundation-checkpoint-{seed}".encode())
        cp_sha = _sha(cp)
        foundation_receipt = {"schema": "banking-cil-v10-foundation-receipt", "status": "qualified",
            "seed": seed, "arm": "FOUNDATION_CMS_ONLY", "task_id": 0,
            "config_sha256": v10_config_sha, "source_lock_sha256": v10_lock,
            "training_manifest_sha256": v10_cfg["training_manifest_sha256"],
            "checkpoint": "checkpoints/foundation-final.pt", "checkpoint_sha256": cp_sha,
            "model_immutable_versions_unchanged": True, "test_data_included": False,
            "metric": {"count": 110, "classes_evaluated": 11, "accuracy": .8,
                       "task_class_macro_accuracy": .8},
            "gate": {"passed": True, "global_candidate_sum_accuracy": .5,
                     "task0_class_macro_accuracy": .4}}
        receipt_path = _write(seed_root / "foundation-receipt.json", foundation_receipt)
        receipt_sha = _sha(receipt_path)
        foundation[seed] = {"checkpoint_sha256": cp_sha, "receipt_sha256": receipt_sha}
        release_foundations[str(seed)] = {"status": "qualified", "config_sha256": v10_config_sha,
            "source_lock_sha256": v10_lock, "training_manifest_sha256": v10_cfg["training_manifest_sha256"],
            "checkpoint_sha256": cp_sha, "receipt_sha256": receipt_sha, "run_dir": str(seed_root)}
    release_path = _write(task / "release/main-release.json", {
        "schema": "banking-cil-v10-main-release-v1", "status": "ready",
        "config_sha256": v10_config_sha, "source_lock_sha256": v10_lock,
        "training_manifest_sha256": v10_cfg["training_manifest_sha256"],
        "foundations": release_foundations})
    release_sha = _sha(release_path)
    seal_rows = []
    for seed in agg.SEEDS:
        for arm in agg.BASE_ARMS:
            run_dir = baseline_runs / f"seed{seed}-{arm}"
            task_metrics = [{"task_id": i, "evaluation_rows": 44 if i == 1 else 0,
                "candidate_metrics": {"count": 44, "correct": 20, "accuracy": 20 / 44,
                    "classes_evaluated": 22, "task_class_macro_accuracy": 100 / 220,
                    "gold_label_token_mean_nll": {"count": 44, "mean_token_nll": 2.0}}}
                for i in range(1, 7)]
            receipt = {"schema": agg.V10_RECEIPT_SCHEMA, "status": "complete", "tasks_completed": 7,
                "seed": seed, "arm": arm, "config_sha256": v10_config_sha,
                "source_lock_sha256": v10_lock, "training_manifest_sha256": v10_cfg["training_manifest_sha256"],
                "test_data_included": False, "model_immutable_versions_unchanged": True,
                "foundation_checkpoint_sha256": foundation[seed]["checkpoint_sha256"],
                "task_metrics": task_metrics}
            receipt_path = _write(run_dir / "run-receipt.json", receipt)
            protocol = {"schema": agg.V10_PROTOCOL_SCHEMA, "phase": "main", "seed": seed, "arm": arm,
                "config_sha256": v10_config_sha, "source_lock_sha256": v10_lock,
                "training_manifest_sha256": v10_cfg["training_manifest_sha256"],
                "source_hashes": v10_cfg["source_hashes"], "parent": {
                    "release_artifact_sha256": release_sha,
                    "foundation_checkpoint_sha256": foundation[seed]["checkpoint_sha256"],
                    "foundation_receipt_sha256": foundation[seed]["receipt_sha256"]}}
            _write(run_dir / "protocol.json", protocol)
            seal_rows.append({"seed": seed, "arm": arm, "run_dir": str(run_dir),
                "run_receipt_sha256": _sha(receipt_path)})
    seal = {"schema": agg.V10_SEAL_SCHEMA, "complete": True, "test_data_included": False,
        "official_test_read": False, "config_sha256": v10_config_sha, "source_lock_sha256": v10_lock,
        "training_manifest_sha256": v10_cfg["training_manifest_sha256"],
        "release_artifact_sha256": release_sha, "runs": seal_rows}
    seal_path = _write(task / "seal/training-seal.json", seal)
    v11_cfg = {
        "schema": agg.V11_CONFIG_SCHEMA, "status": "frozen",
        "v10_config_sha256": v10_config_sha, "v10_source_lock_sha256": v10_lock,
        "v10_source_hashes": v10_cfg["source_hashes"], "source_hashes": source_map,
        "baseline_training_seal_sha256": _sha(seal_path),
        "training_manifest_sha256": v10_cfg["training_manifest_sha256"],
        "model_revision": v10_cfg["model_revision"], "model_assets_sha256": v10_cfg["model_assets_sha256"],
        "initialization_seed": v10_cfg["initialization_seed"], "seeds": list(agg.SEEDS), "arms": list(agg.ARMS),
        "task_id": 1, "epochs": 2, "batch_size": 8, "sleep_lr": .001,
        "sleep_weight_decay": 0.0, "sleep_clip_norm": 1.0,
        "lti_semantic_weight": .5, "lti_edit_weight": .5,
        "lti_teacher_validity_policy": "canonical_teacher_required_for_reward",
        "objective_version": "v10_eq4_reward_term_ablation_with_kl_retained_v1",
        "test_data_included": False, "max_wall_seconds": 1800,
        "max_forward_calls": 30000, "max_update_steps": 3000, "candidate_batch_size": 32,
    }
    v11_cfg_path = _write(task / "configs/v11.json", v11_cfg)
    agg.EXPECTED_V11_CONFIG_FILE_SHA256 = _sha(v11_cfg_path)
    agg.EXPECTED_V11_SOURCE_LOCK_SHA256 = agg.canonical_sha(source_map)
    run_entries = []
    for seed in agg.SEEDS:
        for arm in agg.ARMS:
            run_dir = v11_runs / f"seed{seed}-{arm}"
            cp = run_dir / "checkpoints/task01-final.pt"
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(f"checkpoint:{seed}:{arm}".encode())
            cp_sha = _sha(cp)
            work = {name: 0 for name in agg.INTEGER_WORK_FIELDS}
            work.update({"active_wall_seconds": 12.5, "full_backbone_forward_calls": 2,
                "full_backbone_forward_tokens": 12, "executed_input_tokens_including_padding": 32,
                "padded_input_tokens": 20, "tail_calls": 3, "tail_grad_enabled_calls": 1,
                "tail_no_grad_calls": 2, "tail_backward_calls": 1, "tail_backward_tokens": 5,
                "wake_updates": 4, "physical_adamw_updates": 6, "update_attempts": 6,
                "sleep_events": 2, "sampled_output_tokens": 7,
                "forward_shape_histogram": {"2x8": 2}, "tail_shape_histogram": {"1x4": 3}})
            metric = {"count": 220, "correct": 120 if arm == agg.ARMS[1] else 100,
                "accuracy": (120 if arm == agg.ARMS[1] else 100) / 220,
                "classes_evaluated": 22, "task_class_macro_accuracy": .5,
                "gold_label_token_mean_nll": {"count": 220, "mean_token_nll": 2.3}}
            task_record = {"task_id": 1, "task_step": 4, "global_wake_tick": 12,
                "evaluation_rows": 220, "candidate_metrics": metric, "checkpoint_sha256": cp_sha}
            receipt = {"schema": agg.V11_RECEIPT_SCHEMA, "status": "complete", "seed": seed, "arm": arm,
                "task_id": 1, "epochs": 2, "task1_wake_ticks": 4,
                "foundation_checkpoint_sha256": foundation[seed]["checkpoint_sha256"],
                "foundation_receipt_sha256": foundation[seed]["receipt_sha256"],
                "baseline_training_seal_sha256": _sha(seal_path),
                "checkpoint": "checkpoints/task01-final.pt", "checkpoint_sha256": cp_sha,
                "checkpoint_size_bytes": cp.stat().st_size,
                "config_sha256": agg.canonical_sha(v11_cfg), "source_lock_sha256": agg.canonical_sha(source_map),
                "v10_source_lock_sha256": v10_lock,
                "training_manifest_sha256": v11_cfg["training_manifest_sha256"],
                "task_metrics": [task_record], "event_counts": {"0": 2}, "work": work,
                "sleep_summary": {"events": 2, "calls": 6, "gkd_calls": 4, "lti_calls": 2,
                    "zero_reward_lti_calls": 0, "empty_teacher_lti_calls": 0,
                    "skipped_zero_objective_calls": 0, "physical_adamw_steps": 6,
                    "request_exposures_by_role": {"old": 3, "current": 3}},
                "controller_state": {"ticks": 12}, "model_immutable_versions_unchanged": True,
                "test_data_included": False, "test_data_read": False}
            receipt_path = _write(run_dir / "run-receipt.json", receipt)
            protocol = {"schema": agg.V11_PROTOCOL_SCHEMA, "status": "complete", "seed": seed, "arm": arm,
                "task_id": 1, "config_sha256": agg.canonical_sha(v11_cfg),
                "source_lock_sha256": agg.canonical_sha(source_map), "source_hashes": source_map,
                "v10_source_lock_sha256": v10_lock, "v10_config_sha256": v10_config_sha,
                "training_manifest_sha256": v11_cfg["training_manifest_sha256"],
                "baseline_training_seal_sha256": _sha(seal_path),
                "foundation_checkpoint_sha256": foundation[seed]["checkpoint_sha256"],
                "foundation_receipt_sha256": foundation[seed]["receipt_sha256"],
                "test_data_included": False, "test_data_read": False,
                "run_receipt_sha256": _sha(receipt_path), "checkpoint": "checkpoints/task01-final.pt",
                "checkpoint_sha256": cp_sha}
            _write(run_dir / "protocol.json", protocol)
            _write(run_dir / "progress.json", {"status": "complete", "seed": seed,
                "arm": arm, "task_id": 1, "work": work})
            run_entries.append({"seed": seed, "arm": arm, "run_dir": str(run_dir)})
    manifest = {"schema": agg.INPUT_SCHEMA, "project_root": str(project), "task_root": str(task),
        "v11_config_path": str(v11_cfg_path), "v10_config_path": str(v10_cfg_path),
        "baseline_seal_path": str(seal_path), "main_release_path": str(release_path),
        "foundation_root": str(foundation_root), "baseline_runs_root": str(baseline_runs),
        "v11_runs_root": str(v11_runs), "runs": run_entries}
    manifest_path = _write(task / "input.json", manifest)
    return manifest_path, run_entries


def test_exact_grid_aggregates_v11_and_keeps_v10_reference_separate(tmp_path, monkeypatch):
    before_torch = _forbid_torch_load_if_preloaded(monkeypatch)
    manifest, _ = _make_fixture(tmp_path)
    result = agg.aggregate(manifest)
    assert result["scope"]["main_runs"] == 6
    assert result["scope"]["official_test_rows_read"] is False
    assert result["by_arm"]["LTI_NO_REWARD"]["metrics"]["accuracy"] == {
        "n": 3, "mean": 100 / 220, "sample_sd": 0.0}
    assert result["by_arm"]["LTI_CANONICAL_TEACHER"]["metrics"]["accuracy"]["mean"] == 120 / 220
    assert result["by_arm"]["LTI_NO_REWARD"]["work"]["physical_adamw_updates"]["total"] == 18
    assert result["v10_legacy_task1_monitor_not_comparable"]["rows_per_run"] == 44
    assert result["by_arm"]["LTI_NO_REWARD"]["per_seed"][0]["count"] == 220
    assert "metrics" not in result["v10_legacy_task1_monitor_not_comparable"]
    _assert_torch_import_state_unchanged(before_torch)


def test_unknown_work_stays_null_not_zero(tmp_path):
    manifest, runs = _make_fixture(tmp_path)
    target = next(Path(r["run_dir"]) for r in runs if r["seed"] == 17 and r["arm"] == agg.ARMS[0])
    rp = target / "run-receipt.json"
    receipt = agg.read_json(rp)
    receipt["work"].pop("tail_backward_tokens")
    _write(rp, receipt)
    pp = target / "protocol.json"
    protocol = agg.read_json(pp)
    protocol["run_receipt_sha256"] = _sha(rp)
    _write(pp, protocol)
    result = agg.aggregate(manifest)
    summary = result["by_arm"][agg.ARMS[0]]["work"]["tail_backward_tokens"]
    assert summary["known_sum"] == 10
    assert summary["total"] is None and summary["complete"] is False
    assert summary["unknown_run_count"] == 1


def test_duplicate_or_missing_grid_run_rejected(tmp_path):
    manifest, runs = _make_fixture(tmp_path)
    value = agg.read_json(manifest)
    value["runs"][-1] = dict(value["runs"][0])
    _write(manifest, value)
    with pytest.raises(ValueError, match="duplicate|unexpected"):
        agg.aggregate(manifest)


def test_checkpoint_bytes_are_verified_without_torch_load(tmp_path, monkeypatch):
    before_torch = _forbid_torch_load_if_preloaded(monkeypatch)
    manifest, runs = _make_fixture(tmp_path)
    run = next(r for r in runs if r["seed"] == 29 and r["arm"] == agg.ARMS[1])
    cp = Path(run["run_dir"]) / "checkpoints/task01-final.pt"
    cp.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        agg.aggregate(manifest)
    _assert_torch_import_state_unchanged(before_torch)


def test_fresh_isolated_import_does_not_import_torch():
    import subprocess

    module_path = Path(agg.__file__).resolve()
    script = (
        "import runpy,sys; "
        "runpy.run_path(sys.argv[1], run_name='offline_accounting_isolated'); "
        "assert 'torch' not in sys.modules; "
        "print('PURE_STDLIB_IMPORT_OK')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(module_path)],
        check=True, capture_output=True, text=True,
    )
    assert completed.stdout.strip() == "PURE_STDLIB_IMPORT_OK"


def test_unsafe_checkpoint_relative_path_rejected(tmp_path):
    manifest, runs = _make_fixture(tmp_path)
    run = next(r for r in runs if r["seed"] == 17 and r["arm"] == agg.ARMS[0])
    pp = Path(run["run_dir"]) / "protocol.json"
    protocol = agg.read_json(pp)
    protocol["checkpoint"] = "checkpoints/../escape.pt"
    _write(pp, protocol)
    with pytest.raises(ValueError, match="protocol binding|unsafe|fixed task-1 endpoint"):
        agg.aggregate(manifest)


@pytest.mark.parametrize("bad", ["2", True, -1, "²"])
def test_malformed_event_count_is_rejected_not_filtered(tmp_path, bad):
    manifest, runs = _make_fixture(tmp_path)
    run = next(r for r in runs if r["seed"] == 17 and r["arm"] == agg.ARMS[0])
    rp = Path(run["run_dir"]) / "run-receipt.json"
    receipt = agg.read_json(rp)
    receipt["event_counts"]["0"] = bad
    _write(rp, receipt)
    pp = Path(run["run_dir"]) / "protocol.json"
    protocol = agg.read_json(pp)
    protocol["run_receipt_sha256"] = _sha(rp)
    _write(pp, protocol)
    with pytest.raises(ValueError, match="nonnegative exact integer"):
        agg.aggregate(manifest)


def test_main_release_artifact_bytes_are_verified(tmp_path):
    manifest, _ = _make_fixture(tmp_path)
    value = agg.read_json(manifest)
    release_path = Path(value["main_release_path"])
    with release_path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="main-release artifact bytes"):
        agg.aggregate(manifest)


def test_foundation_checkpoint_bytes_are_verified_against_release(tmp_path):
    manifest, _ = _make_fixture(tmp_path)
    value = agg.read_json(manifest)
    cp = Path(value["foundation_root"]) / "seed17/checkpoints/foundation-final.pt"
    cp.write_bytes(b"tampered foundation")
    with pytest.raises(ValueError, match="foundation checkpoint bytes"):
        agg.aggregate(manifest)


def test_frozen_v11_config_bytes_are_pinned(tmp_path):
    manifest, _ = _make_fixture(tmp_path)
    value = agg.read_json(manifest)
    config_path = Path(value["v11_config_path"])
    cfg = agg.read_json(config_path)
    cfg["sleep_lr"] = 0.25
    _write(config_path, cfg)
    with pytest.raises(ValueError, match="config bytes"):
        agg.aggregate(manifest)


def test_v10_44_row_monitor_is_not_mislabeled_as_v11_220_endpoint(tmp_path):
    manifest, runs = _make_fixture(tmp_path)
    baseline_receipt = next((Path(p) for p in tmp_path.rglob("run-receipt.json")
                             if "v10-runs" in str(p) and "seed17-CMS_ONLY" in str(p)))
    receipt = agg.read_json(baseline_receipt)
    receipt["task_metrics"][0]["evaluation_rows"] = 220
    receipt["task_metrics"][0]["candidate_metrics"]["count"] = 220
    _write(baseline_receipt, receipt)
    seal_path = Path(agg.read_json(manifest)["baseline_seal_path"])
    seal = agg.read_json(seal_path)
    row = next(r for r in seal["runs"] if r["seed"] == 17 and r["arm"] == "CMS_ONLY")
    row["run_receipt_sha256"] = _sha(baseline_receipt)
    _write(seal_path, seal)
    value = agg.read_json(manifest)
    cfg_path = Path(value["v11_config_path"])
    cfg = agg.read_json(cfg_path)
    cfg["baseline_training_seal_sha256"] = _sha(seal_path)
    _write(cfg_path, cfg)
    agg.EXPECTED_V11_CONFIG_FILE_SHA256 = _sha(cfg_path)
    with pytest.raises(ValueError, match="44 rows"):
        agg.aggregate(manifest)
