from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_banking_cil_v10 as driver  # noqa: E402


def sample_config():
    return {
        "schema": driver.SCHEMA,
        "status": "frozen",
        "source_revision": "frozen-source-revision",
        "source_hashes": {"src/nested_sleep/lm.py": "a" * 64},
        "model_revision": "local-model@revision",
        "model_assets_sha256": {"model.safetensors": "b" * 64},
        "training_manifest_sha256": "c" * 64,
        "initialization_seed": 1907008,
        "seeds": [17, 29, 43], "tasks": 7, "classes_per_task": 11,
        "arms": list(driver.ARMS), "rank": 64, "capacity": 8, "levels": 5,
        "periods": [16, 64, 256, 1024, None], "batch_size": 8,
        "foundation_epochs": 2, "epochs_per_task": 2,
        "base_lr": 0.001, "fast_lr": 0.1, "sleep_lr": 0.001,
        "sleep_weight_decay": 0.0, "sleep_clip_norm": 1.0,
        "temperature": 0.8, "max_new_tokens": 10, "lti_edit_distance_z0": 8,
        "lti_semantic_weight": 0.5, "lti_edit_weight": 0.5,
        "gkd_calls_per_event": 20, "lti_calls_per_event": 10,
        "eval_examples_per_class": 2, "max_sequence_length": 128,
        "task0_global_candidate_gate": 0.5, "task0_class_macro_gate": 0.4,
        "foundation_max_wall_seconds": 1800, "foundation_max_forward_calls": 20000,
        "foundation_max_update_steps": 1000, "main_max_wall_seconds": 7200,
        "main_max_forward_calls": 150000, "main_max_update_steps": 15000,
        "device": "cuda:0", "sleep_prompt_policy": "balanced_seen_class_cycle",
        "candidate_batch_size": 32, "checkpoint_every_steps": 64,
    }


def test_config_accepts_locked_foundation_and_main_limits():
    cfg = sample_config()
    assert driver.validate_config(cfg, phase="foundation")["phase"] == "foundation"
    assert driver.validate_config(cfg, phase="main")["phase"] == "main"
    assert driver._budget_config(cfg, "foundation")["max_update_steps"] == 1000
    assert driver._budget_config(cfg, "main")["max_forward_calls"] == 150000


@pytest.mark.parametrize(("field", "value"), [
    ("max_new_tokens", 8), ("candidate_batch_size", 33), ("capacity", 7),
    ("periods", [16, 64, 256, 1024, 2048]), ("gkd_calls_per_event", 19),
])
def test_config_rejects_unlocked_or_inadequate_values(field, value):
    cfg = sample_config(); cfg[field] = value
    with pytest.raises(ValueError):
        driver.validate_config(cfg, phase="foundation")


def test_gpu_phases_reject_unfrozen_candidate_config_but_preflight_can_inspect_it():
    cfg = sample_config(); cfg["status"] = "candidate"
    assert driver.validate_config(cfg, phase="preflight")["status"] == "candidate"
    with pytest.raises(ValueError, match="require a frozen config"):
        driver.validate_config(cfg, phase="foundation")
    cfg.pop("status")
    with pytest.raises(ValueError, match="missing required fields"):
        driver.validate_config(cfg, phase="foundation")


def test_seed_orders_partition_all_77_classes_into_7_by_11():
    order = list(range(77))
    mapping = driver.class_task_map(order)
    assert set(mapping) == set(range(77))
    assert [len(driver.task_class_ids(order, i)) for i in range(7)] == [11] * 7
    assert tuple(driver.task_class_ids(order, 6)) == tuple(range(66, 77))
    with pytest.raises(ValueError):
        driver.class_task_map(order[:-1] + [0])


def test_sleep_and_gold_selector_are_same_triplets_and_cover_both_roles():
    old = {cid: [{"id": f"old-{cid}", "intent_id": cid} for _ in range(4)] for cid in range(11)}
    current = {cid: [{"id": f"new-{cid}", "intent_id": cid} for _ in range(4)] for cid in range(11, 22)}
    cfg = sample_config()
    methods = driver.sleep_method_schedule(cfg)
    assert methods == ["gkd", "gkd", "lti"] * 10
    left = driver.make_sleep_request_schedule(old, current, event_index=0, task_id=1, method_kinds=methods)
    right = driver.make_sleep_request_schedule(old, current, event_index=0, task_id=1, method_kinds=methods)
    assert [(kind, row["id"], role) for kind, row, role in left] == [(kind, row["id"], role) for kind, row, role in right]
    assert sum(role == "old" for _, _, role in left) == 15
    assert sum(role == "current" for _, _, role in left) == 15
    assert {row["intent_id"] for _, row, role in left if role == "old"} == set(old)
    assert {row["intent_id"] for _, row, role in left if role == "current"} == set(current)
    assert all(kind == methods[i] for i, (kind, _, _) in enumerate(left))


def test_candidate_extractor_uses_global_sum_counts_and_only_observed_classes_macro():
    result = {"schema": "banking-cil-v10-runtime-v1", "candidate_metrics": {
        "context": {
            "teacher_forced_sum": {
                "all77": {"count": 22, "correct": 12, "accuracy": 12 / 22},
                "per_class": {str(i): {"count": 2 if i < 11 else 0,
                                      "correct": 1 if i < 11 else 0} for i in range(77)},
            },
            "gold_label_token_mean_nll": {"all77": {"count": 22, "mean_token_nll": 1.25}},
        }
    }, "label_dictionary_sha256": "d" * 64, "classes": 77}
    metric = driver._extract_candidate_metric(result)
    assert metric["accuracy"] == 12 / 22
    assert metric["classes_evaluated"] == 11
    assert metric["task_class_macro_accuracy"] == 0.5
    assert metric["gold_label_token_mean_nll"]["mean_token_nll"] == 1.25
    result["candidate_metrics"]["context"]["teacher_forced_sum"]["all77"]["correct"] = 23
    with pytest.raises(ValueError): driver._extract_candidate_metric(result)


def test_config_hash_excludes_only_cli_phase():
    cfg = sample_config()
    foundation = driver.validate_config(cfg, phase="foundation")
    main = driver.validate_config(cfg, phase="main")
    assert driver._config_sha(foundation) == driver._config_sha(main)
    modified = dict(main); modified["sleep_lr"] = 0.002
    assert driver._config_sha(main) != driver._config_sha(modified)


def test_source_lock_mismatch_fails_closed():
    with pytest.raises(ValueError, match="source lock mismatch"):
        driver._verify_source_lock(sample_config(), {"src/nested_sleep/lm.py": "e" * 64})


def test_output_path_cannot_overlap_model_or_public_project():
    with pytest.raises(ValueError):
        driver._validate_output_path(Path("C:/pub/outputs/run"), Path("C:/pub"), Path("C:/model"))
    with pytest.raises(ValueError):
        driver._validate_output_path(Path("C:/model/run"), Path("C:/pub"), Path("C:/model"))


def test_budget_counts_actual_forward_and_attempted_steps_separately():
    cfg = driver._budget_config(sample_config(), "foundation")
    budget = driver.Budget(cfg)
    budget.update()
    budget.adamw_updates += 1
    budget.check()
    assert budget.update_attempts == 1 and budget.adamw_updates == 1
    budget.wake_updates = 999
    with pytest.raises(RuntimeError, match="update-attempt"):
        budget.update()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_main_release_rechecks_receipt_gate_values_and_artifact_bytes(tmp_path):
    cfg = sample_config()
    cfg_sha = driver._config_sha(driver.validate_config(cfg, phase="main"))
    source_sha = "d" * 64
    manifest_sha = cfg["training_manifest_sha256"]
    foundations = {}
    for seed in driver.SEEDS:
        run_dir = tmp_path / f"foundation-{seed}"
        (run_dir / "checkpoints").mkdir(parents=True)
        cp = run_dir / "checkpoints" / "foundation-final.pt"
        cp.write_bytes(f"checkpoint-{seed}".encode())
        receipt = {
            "schema": "banking-cil-v10-foundation-receipt", "status": "qualified",
            "seed": seed, "arm": "FOUNDATION_CMS_ONLY", "task_id": 0,
            "checkpoint": "checkpoints/foundation-final.pt", "checkpoint_sha256": _sha(cp),
            "config_sha256": cfg_sha, "source_lock_sha256": source_sha,
            "training_manifest_sha256": manifest_sha,
            "metric": {"count": 110, "correct": 65, "accuracy": 65 / 110,
                       "classes_evaluated": 11, "task_class_macro_accuracy": .41},
            "gate": {"global_candidate_sum_accuracy": .5,
                     "task0_class_macro_accuracy": .4, "passed": True},
            "model_immutable_versions_unchanged": True,
        }
        receipt_path = run_dir / "foundation-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        foundations[str(seed)] = {"status": "qualified", "source_lock_sha256": source_sha,
            "config_sha256": cfg_sha, "training_manifest_sha256": manifest_sha,
            "checkpoint_sha256": _sha(cp), "receipt_sha256": _sha(receipt_path),
            "run_dir": str(run_dir)}
    probe_path = tmp_path / "probe.json"
    probe_path.write_text("{}", encoding="utf-8")
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps({
        "schema": "banking-cil-v10-main-release-v1", "status": "ready",
        "config_sha256": cfg_sha, "source_lock_sha256": source_sha,
        "training_manifest_sha256": manifest_sha, "foundations": foundations,
        "transfer_probe": {"status": "complete", "resume_parity": True,
            "source_lock_sha256": source_sha, "artifact_path": str(probe_path),
            "artifact_sha256": _sha(probe_path)}, "probe_source_lock_sha256": source_sha,
    }), encoding="utf-8")
    selected, _ = driver._validate_foundation_release(release_path, cfg, cfg_sha,
        source_sha, manifest_sha, 17, Path(foundations["17"]["run_dir"]))
    assert selected["status"] == "ready"
    receipt_path = Path(foundations["29"]["run_dir"]) / "foundation-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["metric"]["accuracy"] = .49
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    foundations["29"]["receipt_sha256"] = _sha(receipt_path)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["foundations"] = foundations
    release_path.write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ValueError, match="does not meet the frozen task-0 gates"):
        driver._validate_foundation_release(release_path, cfg, cfg_sha,
            source_sha, manifest_sha, 17, Path(foundations["17"]["run_dir"]))


def test_cli_accepts_repeated_run_dirs_for_exact_seal_grid(monkeypatch):
    captured = {}
    def fake_run(args):
        captured["args"] = args
        return {"ok": True}
    monkeypatch.setattr(driver, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_banking_cil_v10.py", "--phase", "seal",
        "--project-root", "P", "--model-path", "M", "--training-pack", "D",
        "--config", "C", "--output-dir", "O", "--run-dir", "one", "--run-dir", "two"])
    driver.main()
    assert [p.name for p in captured["args"].run_dir] == ["one", "two"]


def test_lti_callback_uses_runtime_eq4_signature_and_thresholded_edit():
    import banking_cil_v10_runtime as runtime
    labels = [f"label-{i}" for i in range(77)]
    label = labels[0]
    same = driver._lti_reward_adapter(label, label, teacher_tokens=[1, 2],
        student_tokens=[1, 2], z0=1, canonical_labels=labels)
    different_but_same_tokens = driver._lti_reward_adapter(label, "not a label",
        teacher_tokens=[1, 2], student_tokens=[1, 2], z0=1,
        canonical_labels=labels)
    beyond_threshold = driver._lti_reward_adapter(label, label,
        teacher_tokens=[1, 2, 3], student_tokens=[7, 8, 9], z0=1,
        canonical_labels=labels)
    assert same == 1.0
    assert different_but_same_tokens == 0.5
    assert beyond_threshold == 0.5
    assert driver._lti_reward_adapter(label, label, teacher_tokens=[1, 2],
        student_tokens=[1, 2], z0=1, canonical_labels=labels) == runtime.local_lti_reward(
            label, label, teacher_tokens=[1, 2], student_tokens=[1, 2], z0=1,
            canonical_labels=labels, semantic_weight=.5, edit_weight=.5)


def test_seal_pairs_exact_foundations_and_cpu_checks_terminal_states(tmp_path, monkeypatch):
    import torch
    cfg = driver.validate_config(sample_config(), phase="seal")
    cfg_sha = driver._config_sha(cfg)
    lock_sha = "d" * 64
    manifest_sha = cfg["training_manifest_sha256"]
    source_hashes = {"src/nested_sleep/lm.py": "a" * 64}
    project = tmp_path / "project"; model = tmp_path / "model"; pack = tmp_path / "pack.json"
    project.mkdir(); model.mkdir(); pack.write_text("{}", encoding="utf-8")
    foundations = {}
    for seed in driver.SEEDS:
        run_dir = tmp_path / "foundations" / str(seed)
        cp_path = run_dir / "checkpoints" / "foundation-final.pt"
        cp_path.parent.mkdir(parents=True)
        cp_path.write_bytes(f"foundation-{seed}".encode())
        receipt = {"schema": "banking-cil-v10-foundation-receipt", "status": "qualified",
            "seed": seed, "arm": "FOUNDATION_CMS_ONLY", "task_id": 0,
            "checkpoint": "checkpoints/foundation-final.pt", "checkpoint_sha256": _sha(cp_path),
            "config_sha256": cfg_sha, "source_lock_sha256": lock_sha,
            "training_manifest_sha256": manifest_sha,
            "metric": {"count": 110, "correct": 60, "accuracy": 60 / 110,
                "classes_evaluated": 11, "task_class_macro_accuracy": .41},
            "gate": {"global_candidate_sum_accuracy": .5, "task0_class_macro_accuracy": .4,
                "passed": True}, "model_immutable_versions_unchanged": True}
        receipt_path = run_dir / "foundation-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        foundations[str(seed)] = {"status": "qualified", "source_lock_sha256": lock_sha,
            "config_sha256": cfg_sha, "training_manifest_sha256": manifest_sha,
            "checkpoint_sha256": _sha(cp_path), "receipt_sha256": _sha(receipt_path),
            "run_dir": str(run_dir)}
    probe_path = tmp_path / "probe.json"; probe_path.write_text("{}", encoding="utf-8")
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps({"schema": "banking-cil-v10-main-release-v1", "status": "ready",
        "config_sha256": cfg_sha, "source_lock_sha256": lock_sha,
        "training_manifest_sha256": manifest_sha, "foundations": foundations,
        "transfer_probe": {"status": "complete", "resume_parity": True,
            "source_lock_sha256": lock_sha, "artifact_path": str(probe_path),
            "artifact_sha256": _sha(probe_path)}, "probe_source_lock_sha256": lock_sha}), encoding="utf-8")
    def fake_prepare(*_args):
        return source_hashes, lock_sha, {"class_orders": {}}, [], [{}] * 770
    monkeypatch.setattr(driver, "_source_prepare", fake_prepare)
    run_dirs = []
    for seed in driver.SEEDS:
        for arm in driver.ARMS:
            run_dir = tmp_path / "runs" / f"{seed}-{arm}"
            (run_dir / "checkpoints").mkdir(parents=True)
            task_metrics = []
            for task_id in range(1, 7):
                task_cp = run_dir / "checkpoints" / f"task{task_id:02d}-final.pt"
                task_cp.write_bytes(f"task-{task_id}".encode())
                count = 2 * 11 * (task_id + 1)
                task_metrics.append({"task_id": task_id, "evaluation_rows": count,
                    "candidate_metrics": {"count": count, "classes_evaluated": 11 * (task_id + 1)},
                    "checkpoint_sha256": _sha(task_cp)})
            final_cp = run_dir / "checkpoints" / "final-task06.pt"
            context_state = {"small": torch.tensor([seed], dtype=torch.int64)}
            wake = SimpleNamespace(context_state={"small": context_state["small"].clone()},
                pending={}, clocks={}, token_counts={}, ticks=100, periods=tuple(cfg["periods"]),
                base_lr=.001, fast_lr=.1, bank_states=[])
            torch.save({"schema": "banking-cil-v10-checkpoint", "seed": seed, "arm": arm,
                "task_id": 6, "step": 100, "context": context_state, "wake": wake,
                "rng": {"python": (), "numpy": (), "torch_cpu": torch.tensor([1]), "torch_cuda": None},
                "config_sha256": cfg_sha, "source_lock_sha256": lock_sha,
                "training_manifest_sha256": manifest_sha, "extra": {}}, final_cp)
            final_sha = _sha(final_cp)
            parent = {"foundation_receipt_sha256": foundations[str(seed)]["receipt_sha256"],
                "foundation_checkpoint_sha256": foundations[str(seed)]["checkpoint_sha256"],
                "release_artifact_sha256": _sha(release_path)}
            protocol = {"schema": "banking-cil-v10-protocol", "phase": "main", "seed": seed,
                "arm": arm, "config_sha256": cfg_sha, "training_manifest_sha256": manifest_sha,
                "source_hashes": source_hashes, "source_lock_sha256": lock_sha, "parent": parent}
            (run_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            receipt = {"schema": "banking-cil-v10-run-receipt", "status": "complete",
                "seed": seed, "arm": arm, "tasks_completed": 7,
                "foundation_checkpoint_sha256": foundations[str(seed)]["checkpoint_sha256"],
                "final_checkpoint": "checkpoints/final-task06.pt", "final_checkpoint_sha256": final_sha,
                "config_sha256": cfg_sha, "source_lock_sha256": lock_sha,
                "training_manifest_sha256": manifest_sha, "task_metrics": task_metrics,
                "endpoint_metrics": {"count": 770, "classes_evaluated": 77},
                "model_immutable_versions_unchanged": True, "test_data_included": False,
                "work": {}}
            (run_dir / "run-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            run_dirs.append(run_dir)
    args = SimpleNamespace(project_root=project, model_path=model, training_pack=pack,
        output_dir=tmp_path / "sealed", run_dir=run_dirs, release_artifact=release_path)
    seal = driver._seal(args, cfg)
    assert seal["complete"] is True
    assert len(seal["terminal_checkpoint_cpu_preflight"]) == 9
    assert [(x["seed"], x["arm"]) for x in seal["runs"]] == [
        (seed, arm) for seed in driver.SEEDS for arm in driver.ARMS]
    payload_path = run_dirs[0] / "checkpoints" / "final-task06.pt"
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    payload["wake"].ticks = 99
    torch.save(payload, payload_path)
    with pytest.raises(ValueError, match="step/wake cursor mismatch"):
        driver._validate_terminal_checkpoint_cpu(payload_path, _sha(payload_path), seed=17,
            arm="CMS_ONLY", cfg_sha=cfg_sha, lock_sha=lock_sha, manifest_sha=manifest_sha,
            periods=cfg["periods"], torch=torch)
