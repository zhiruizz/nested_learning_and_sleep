"""Frozen v5 anchored-wake comparison (development only).

Uses the fixed v3 train/validation pack, v4 warm checkpoints and local model
assets only. No confirmation input is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import agnews_v3_common as c
import agnews_preupdate_sleep_v4 as boundary
import agnews_anchored_wake_v5 as wake
import qualify_agnews_boundary_transfer_v4 as transfer
from nested_sleep.lm import execution_state, immutable_versions, isolated_clone
from nested_sleep.optim import CMSController

SEEDS = (17, 29, 43)
ARMS = (
    "warm_CMS_reference",
    "warm_anchorKD",
    "warm_anchorGold",
    "warm_anchorKD_sleepP16_latest",
    "warm_anchorKD_sleepP16_warmmix",
    "warm_self_sleepP16_latest",
    "warm_self_sleepP16_warmmix",
)
STEPS = (0, 16, 64, 112, 256, 448)
SLEEP_ARMS = {
    "warm_anchorKD_sleepP16_latest", "warm_anchorKD_sleepP16_warmmix",
    "warm_self_sleepP16_latest", "warm_self_sleepP16_warmmix",
}
KD_ARMS = {"warm_anchorKD", "warm_anchorKD_sleepP16_latest", "warm_anchorKD_sleepP16_warmmix"}
GOLD_ARMS = {"warm_anchorGold"}
WARM_MIX_ARMS = {"warm_anchorKD_sleepP16_warmmix", "warm_self_sleepP16_warmmix"}
PRIMARY_STEPS = 448
EVENT_STEPS = tuple(range(16, 449, 16))
EVENT_ORDER = [0, 2, 3, 1, 1, 3, 2, 0] * 2 + [0, 2, 1, 3] * 2


def sha_file(path: Path) -> str:
    return c.sha(path)


def hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def expected_active_slots(active_parameters):
    counts = active_parameters["active_slots_by_bank"]
    if len(counts) != 2 or any(not isinstance(x, int) or x < 0 for x in counts):
        raise AssertionError(f"unexpected active slot counts: {counts!r}")
    return counts


def attach_final_active_parameters(row, active_parameters, is_sleep_arm):
    # Attach before any read: regression guard for the former end-of-arm KeyError.
    row["final_active_parameters"] = active_parameters
    counts = expected_active_slots(row["final_active_parameters"])
    if is_sleep_arm:
        assert counts == [0, 29]
    return counts


def update_loss_stats(stats, key, value):
    if value is None:
        return
    value = float(value)
    row = stats.setdefault(key, {"count": 0, "sum": 0.0, "min": None, "max": None})
    row["count"] += 1
    row["sum"] += value
    row["min"] = value if row["min"] is None else min(row["min"], value)
    row["max"] = value if row["max"] is None else max(row["max"], value)


def active_prefix_hash(model, bank_index, count):
    slots = model.banks()[bank_index].slots
    if count > len(slots):
        raise AssertionError("active slot prefix exceeds allocated slots")
    state = {str(i): {k: v.detach().cpu().clone() for k, v in slots[i].state_dict().items()}
             for i in range(count)}
    return c.engine.shared.tensor_tree_hash(state)


def old_receiver_zero_hash(model):
    slots = model.banks()[1].slots
    if not slots:
        raise AssertionError("old receiver slot zero is absent")
    state = {k: v.detach().cpu().clone() for k, v in slots[0].state_dict().items()}
    return c.engine.shared.tensor_tree_hash(state)


def verify_initial_warm_state(model, warm_cp):
    assert execution_state(model) == warm_cp["execution_state"]
    assert c.engine.shared.tensor_tree_hash(c.engine.capture_rng()) == c.engine.shared.tensor_tree_hash(warm_cp["rng"])


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_progress(path: Path, value):
    # Progress is intentionally replaceable; sealed outputs remain create-only.
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def event_pools(pack, seed):
    """Same deterministic balanced pool recipe as v4 at boundaries 16..448."""
    ids = {k: [i for i, r in enumerate(pack["records"])
               if r["role"] == "train" and r["class_id"] == k] for k in range(4)}
    pools = {}
    for step in EVENT_STEPS:
        rng = random.Random(f"nl-sleep-v4-boundary:{seed}:{step}")
        pool = [rng.choice(ids[k]) for k in EVENT_ORDER]
        counts = {k: sum(pack["records"][i]["class_id"] == k for i in pool) for k in range(4)}
        assert counts == {k: 6 for k in range(4)}
        pools[step] = pool
    return pools


def old_aux_schedule(pack, seed):
    """Fixed 2/class old-train reservoir sample, shared by KD and gold arms."""
    ids = {k: [i for i, r in enumerate(pack["records"])
               if r["role"] == "train" and r["class_id"] == k] for k in (0, 1)}
    schedule = {}
    for step in range(1, PRIMARY_STEPS + 1):
        rng = random.Random(f"nl-sleep-v5-old-aux:{seed}:{step}")
        chosen = [i for cls in (0, 1) for i in rng.sample(ids[cls], 2)]
        rng.shuffle(chosen)
        assert len(chosen) == 4 and [sum(pack["records"][i]["class_id"] == cls for i in chosen)
                                     for cls in (0, 1)] == [2, 2]
        schedule[step] = chosen
    return schedule


def clone_frozen_teacher(model):
    teacher = isolated_clone(model, share_backbone=True)
    teacher.eval().requires_grad_(False)
    assert teacher is not model and all(not p.requires_grad for p in teacher.parameters())
    return teacher


def arm_kind(arm):
    if arm in KD_ARMS:
        return "soft_kd"
    if arm in GOLD_ARMS:
        return "gold"
    return "cms"


def task_gate_details(train_metrics, validation_metrics, phase):
    classes = (0, 1) if phase == "old" else (2, 3)
    detail = {
        "train_phase_em": train_metrics[phase][c.EM],
        "train_phase_pass": train_metrics[phase][c.EM] >= 0.80,
        "validation_phase_em": validation_metrics[phase][c.EM],
        "validation_phase_pass": validation_metrics[phase][c.EM] >= 0.75,
        "validation_class_em": {str(k): validation_metrics[f"class{k}"][c.EM] for k in classes},
        "validation_class_pass": {str(k): validation_metrics[f"class{k}"][c.EM] >= 0.70 for k in classes},
    }
    detail["pass"] = detail["train_phase_pass"] and detail["validation_phase_pass"] and all(detail["validation_class_pass"].values())
    return detail


def set_warmmix_route(model, arm):
    if arm in WARM_MIX_ARMS:
        model.banks()[1].route_override = {"mix": 0, "fraction": 0.5}


def save_checkpoint(args, cfg, proto_sha, model, controller, seed, arm, step, row, pack, pool_hashes, meter):
    before_fp = c.fingerprint(model)
    policy = execution_state(model)
    rng = c.engine.capture_rng()
    rng_hash = c.engine.shared.tensor_tree_hash(rng)
    vrows = c.rows(pack, seed, role="validation")
    metrics = c.score(model, vrows, pack, seed, f"v5_validation_seed{seed}_{arm}_step{step}", meter)
    assert metrics["all"]["examples"] == 512
    assert metrics["old"]["examples"] == metrics["new"]["examples"] == 256
    assert c.fingerprint(model) == before_fp
    assert execution_state(model) == policy
    assert c.engine.shared.tensor_tree_hash(c.engine.capture_rng()) == rng_hash
    cp_path = args.out_dir / "checkpoints" / f"seed-{seed}-{arm}-step-{step:03}.pt"
    info = c.checkpoint(cp_path, model, controller, seed, step, {
        "arm": arm, "primary_cursor": step, "sleep_Adam_updates": row["sleep_Adam_updates"],
        "protocol_sha256": proto_sha, "driver_source_sha256": sha_file(Path(__file__)),
        "config_sha256": sha_file(args.config), "new_schedule_sha256": pack["schedules"][str(seed)]["new_sha256"],
        "old_aux_schedule_sha256": row["old_aux_schedule_sha256"],
        "event_pool_hashes_through_step": {str(k): pool_hashes[k] for k in EVENT_STEPS if k <= step},
    })
    info.update(path=str(cp_path.relative_to(args.out_dir)), primary_step=step)
    row["checkpoints"].append(info)
    row["validation"].append({"primary_step": step, "metrics": metrics,
        "adapter_fingerprint": before_fp, "execution_policy": policy, "rng_sha256": rng_hash,
        "controller_counts": list(controller.counts)})
    return metrics


def check_v4_reference(args, seed, step, candidate_cp, candidate_metrics, v4_run):
    old = next(r for r in v4_run["runs"] if r["seed"] == seed and r["arm"] == "warm_CMS_no_sleep")
    old_info = next(x for x in old["checkpoints"] if x["primary_step"] == step)
    v4_cp_path = args.v4_run_dir / old_info["path"]
    assert sha_file(v4_cp_path) == old_info["sha256"]
    v4_cp = torch.load(v4_cp_path, map_location="cpu", weights_only=False)
    new_cp = torch.load(args.out_dir / candidate_cp["path"], map_location="cpu", weights_only=False)
    assert c.engine.shared.tensor_tree_hash(new_cp["model"]) == c.engine.shared.tensor_tree_hash(v4_cp["model"])
    assert c.engine.shared.tensor_tree_hash(new_cp["execution_state"]) == c.engine.shared.tensor_tree_hash(v4_cp["execution_state"])
    assert c.engine.shared.tensor_tree_hash(new_cp["controller"]) == c.engine.shared.tensor_tree_hash(v4_cp["controller"])
    assert c.engine.shared.tensor_tree_hash(new_cp["rng"]) == c.engine.shared.tensor_tree_hash(v4_cp["rng"])
    old_metrics = next(x["metrics"] for x in old["validation"] if x["primary_step"] == step)
    assert candidate_metrics == old_metrics
    return {"step": step, "checkpoint_model_equal": True, "execution_policy_equal": True,
        "controller_equal": True, "rng_equal": True, "validation_metrics_equal": True,
        "v4_checkpoint_sha256": old_info["sha256"]}


def validate_config(cfg):
    assert cfg["schema"] == "agnews-anchored-comparison-config-v5"
    assert cfg["review_status"] == "FROZEN_DEVELOPMENT_AFTER_REVIEW"
    assert cfg["arms"] == list(ARMS)
    assert cfg["primary_steps"] == PRIMARY_STEPS and cfg["evaluation_steps"] == list(STEPS)
    assert cfg["sleep_period"] == 16 and cfg["capacity"] == 32
    assert cfg["base_lr"] == .02 and cfg["expert_lr"] == .04
    assert cfg["elapsed_limit_seconds"] == 7200
    assert cfg["confirmation_input"] is False and cfg["confirmation_scored"] is False
    assert len(cfg["qualification_result_sha256"]) == 64
    assert len(cfg["qualification_plan_sha256"]) == 64
    assert len(cfg["qualification_driver_sha256"]) == 64


def validate_qualification(path, cfg):
    q = c.read(path)
    assert q.get("schema") == "agnews-anchored-resume-qualified-v5"
    assert q.get("passed") is True
    assert q.get("exact_model_controller_policy_RNG") is True
    assert q.get("scoring") is False and q.get("confirmation_scored") is False
    assert q.get("SGD_updates") == 18 and q.get("Sleep_Adam_updates") == 48
    assert q.get("anchor_helper_sha256") == sha_file(Path(wake.__file__))
    assert q.get("source_sha256") == cfg["qualification_driver_sha256"]
    assert q.get("plan_sha256") == cfg["qualification_plan_sha256"]
    assert sha_file(path) == cfg["qualification_result_sha256"]
    branches = q.get("branches")
    assert isinstance(branches, list) and len(branches) == 2
    for key in ("model", "controller", "execution_policy", "RNG"):
        assert branches[0][key] == branches[1][key], f"qualification restore branch mismatch: {key}"
    cp = q["private_checkpoint"]
    cp_path = Path(cp["path"])
    if not cp_path.is_absolute():
        cp_path = path.parent / cp_path
    assert sha_file(cp_path) == cp["sha256"]
    return q, cp_path


def run(args):
    cfg = c.read(args.config)
    validate_config(cfg)
    qualification, qualification_checkpoint = validate_qualification(args.qualification, cfg)
    pack = c.load_pack(args.records)
    warm = c.read(args.warm_dir / "transfer-result.json")
    v4 = c.read(args.v4_run_dir / "development-sealed.json")
    v4_protocol = c.read(args.v4_run_dir / "protocol.json")
    assert warm["all3_qualified"] and warm["phaseB_permitted"] and warm["confirmation_scored"] is False
    assert len(warm["qualified_models"]) == 3 and {x["seed"] for x in warm["qualified_models"]} == set(SEEDS)
    assert v4["complete"] and v4["confirmation_scored"] is False and v4["no_holdout_input"] is True
    assert v4_protocol["confirmation_input"] is False and v4_protocol["confirmation_scored"] is False
    assert v4["protocol_sha256"] == sha_file(args.v4_run_dir / "protocol.json")
    assert v4["warm_result_sha256"] == sha_file(args.warm_dir / "transfer-result.json")
    locked_current_sources = {
        "driver_source_sha256": sha_file(Path(__file__)),
        "wake_helper_sha256": sha_file(Path(wake.__file__)),
        "v4_boundary_sha256": sha_file(Path(boundary.__file__)),
        "common_sha256": sha_file(Path(c.__file__)),
        "engine_sha256": sha_file(Path(c.engine.__file__)),
        "warm_result_sha256": sha_file(args.warm_dir / "transfer-result.json"),
        "v4_development_seal_sha256": sha_file(args.v4_run_dir / "development-sealed.json"),
        "record_pack_sha256": sha_file(args.records),
    }
    assert all(cfg.get(key) == value for key, value in locked_current_sources.items()), "config source/data lock mismatch"
    assert [r["arm"] for r in v4["runs"] if r["seed"] == SEEDS[0]].count("warm_CMS_no_sleep") == 1
    for x in warm["qualified_models"]:
        assert transfer.transfer_passes(warm["teachers"][str(x["seed"])]["validation"], x["train"], x["validation"])
        assert sha_file(args.warm_dir / x["checkpoint"]["path"]) == x["checkpoint"]["sha256"]
    c.preflight(args.model_dir, pack)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "checkpoints").mkdir()
    start_time = time.monotonic()
    meter = c.engine.WorkMeter()
    token = c.engine.ACTIVE_METER.set(meter)
    pools_by_seed = {seed: event_pools(pack, seed) for seed in SEEDS}
    pool_hashes_by_seed = {seed: {step: c.engine.shared.tensor_tree_hash(pool)
        for step, pool in pools_by_seed[seed].items()} for seed in SEEDS}
    aux_by_seed = {seed: old_aux_schedule(pack, seed) for seed in SEEDS}
    aux_hashes = {seed: hash_json({str(k): v for k, v in aux_by_seed[seed].items()}) for seed in SEEDS}
    source_paths = {
        "driver": Path(__file__),
        "common": Path(c.__file__),
        "v4_boundary": Path(boundary.__file__),
        "wake_helper": Path(wake.__file__),
        "v4_engine": Path(c.engine.__file__),
    }
    source_hashes = {k: sha_file(v) for k, v in source_paths.items()}
    config_hash = sha_file(args.config)
    warm_hash = sha_file(args.warm_dir / "transfer-result.json")
    v4_seal_hash = sha_file(args.v4_run_dir / "development-sealed.json")
    v4_protocol_hash = sha_file(args.v4_run_dir / "protocol.json")
    protocol = {
        "schema": "agnews-anchored-comparison-protocol-v5",
        "status": "frozen_development",
        "config_sha256": config_hash, "config": cfg,
        "source_sha256": source_hashes,
        "warm_result_sha256": warm_hash,
        "qualification_result_sha256": sha_file(args.qualification),
        "qualification_plan_sha256": qualification["plan_sha256"],
        "qualification_driver_sha256": qualification["source_sha256"],
        "qualification_private_checkpoint_sha256": sha_file(qualification_checkpoint),
        "v4_development_seal_sha256": v4_seal_hash,
        "v4_protocol_sha256": v4_protocol_hash,
        "record_pack_sha256": sha_file(args.records),
        "warm_common_budget": warm["chosen_common_budget"],
        "arms": list(ARMS), "seeds": list(SEEDS), "primary_steps": PRIMARY_STEPS,
        "evaluation_steps": list(STEPS), "planned_primary_updates": 7 * 3 * 448,
        "planned_sleep_events": 4 * 3 * 28, "planned_sleep_adam_updates": 4 * 3 * 28 * 24,
        "planned_kd_old_aux_steps": 3 * 3 * 448, "planned_gold_old_aux_steps": 3 * 448,
        "planned_kd_old_prefix_draws": 3 * 3 * 448 * 4,
        "planned_gold_old_prefix_draws": 3 * 448 * 4,
        "planned_extra_old_answer_targets": 3 * 448 * 4,
        "cost_note": "Every primary update uses one CMS collect. Soft KD adds two frozen-teacher forwards plus one student old-context forward/backward; Gold adds one student old-context forward/backward and direct old-answer CE targets. These paths are not equal-FLOP.",
        "sleep_pool_hashes_by_seed": {str(s): {str(k): v for k, v in pool_hashes_by_seed[s].items()} for s in SEEDS},
        "old_aux_schedule_sha256_by_seed": {str(s): aux_hashes[s] for s in SEEDS},
        "confirmation_input": False, "confirmation_scored": False,
        "exploratory_only": True, "automatic_retry": False,
        "frozen_before_training_utc": datetime.now(timezone.utc).isoformat(),
    }
    c.write(args.out_dir / "protocol.json", protocol)
    protocol_sha = sha_file(args.out_dir / "protocol.json")
    runs = []
    primary_attempts = 0
    sleep_adam_updates = 0
    current_event = False

    def check_time():
        if time.monotonic() - start_time >= cfg["elapsed_limit_seconds"]:
            raise TimeoutError("v5 development elapsed-time limit")

    try:
        for seed in SEEDS:
            check_time()
            model = c.model(args.model_dir, seed)
            train_rows = c.rows(pack, seed, role="train")
            rows_by_pack_index = c.rows(pack, seed)
            # The v4 warm checkpoint is the start state for all seven arms.
            warm_row = next(x for x in warm["qualified_models"] if x["seed"] == seed)
            warm_cp = torch.load(args.warm_dir / warm_row["checkpoint"]["path"], map_location="cpu", weights_only=False)
            c.restore(model, warm_cp)
            c.engine.restore_rng(warm_cp["rng"])
            verify_initial_warm_state(model, warm_cp)
            assert c.fingerprint(model) == warm_row["checkpoint"]["adapter_fingerprint"]
            warm_active_counts = expected_active_slots(c.engine.shared.active_parameter_counts(model))
            if not warm_active_counts[1]:
                raise AssertionError("warm checkpoint has no active old receiver slot prefix")
            warm_receiver0_hash = old_receiver_zero_hash(model)
            warm_prefix_hash = active_prefix_hash(model, 1, warm_active_counts[1])
            teacher = clone_frozen_teacher(model)
            teacher_fp = c.fingerprint(teacher)
            teacher_policy = execution_state(teacher)
            teacher_immutable = immutable_versions(teacher)
            initial_metrics_reference = None

            for arm in ARMS:
                check_time()
                arm_start_time = time.monotonic()
                c.restore(model, warm_cp)
                c.engine.restore_rng(warm_cp["rng"])
                verify_initial_warm_state(model, warm_cp)
                assert c.fingerprint(model) == warm_row["checkpoint"]["adapter_fingerprint"]
                controller = CMSController(model, (4, 16), cfg["base_lr"], cfg["expert_lr"])
                pools = pools_by_seed[seed]
                row = {"seed": seed, "arm": arm, "status": "running", "primary_updates": 0,
                    "sleep_Adam_updates": 0, "events": [], "checkpoints": [], "validation": [],
                    "train_final": None, "initial_adapter_fingerprint": c.fingerprint(model),
                    "immutable_teacher_fingerprint": teacher_fp,
                    "old_aux_schedule_sha256": aux_hashes[seed], "work": {}}
                row["training_loss_stats"] = {}
                runs.append(row)
                save = lambda step: save_checkpoint(args, cfg, protocol_sha, model, controller, seed, arm, step, row, pack, pool_hashes_by_seed[seed], meter)
                def write_checkpoint_progress():
                    write_progress(args.out_dir / "progress.json", {"schema":"agnews-anchored-progress-v5",
                        "runs":runs,"confirmation_scored":False,"complete":False})
                try:
                    initial = save(0)
                    write_checkpoint_progress()
                    assert initial["old"] == warm_row["validation"]["old"]
                    if initial_metrics_reference is None:
                        initial_metrics_reference = initial
                    else:
                        assert initial == initial_metrics_reference, "all arms must share identical warm-start step-0 metrics"
                    for step in range(1, PRIMARY_STEPS + 1):
                        check_time()
                        new_indices = pack["schedules"][str(seed)]["new"][step - 1]
                        schedule_rows = [rows_by_pack_index[i] for i in new_indices]
                        old_rows = [rows_by_pack_index[i] for i in aux_by_seed[seed][step]]
                        sleep_now = arm in SLEEP_ARMS and step in EVENT_STEPS
                        defer = sleep_now
                        primary_attempts += 1
                        row["attempted_primary_updates"] = step
                        kind = arm_kind(arm)
                        if kind in ("soft_kd", "gold"):
                            wake_result, proposals = wake.anchored_step(
                                model, controller, schedule_rows, old_rows,
                                pack["tokenizer"]["pad_token_id"], meter,
                                f"v5_seed{seed}_{arm}_primary",
                                teacher,
                                mode=kind, defer_due=defer)
                        else:
                            event = c.engine.shared.step(model, controller, schedule_rows,
                                pack["tokenizer"]["pad_token_id"], meter,
                                f"v5_seed{seed}_{arm}_primary", defer_due=defer)
                            wake_result, proposals = {"mode":"cms","due_levels":sorted(event["proposals"]),
                                "controller_counts":list(controller.counts),"new_ce":event["loss"],
                                "combined_grad_norm_preclip":event["grad_norm"]}, event["proposals"]
                        update_loss_stats(row["training_loss_stats"], "new_ce", wake_result.get("new_ce"))
                        update_loss_stats(row["training_loss_stats"], "old_anchor_loss", wake_result.get("old_anchor_loss"))
                        update_loss_stats(row["training_loss_stats"], "combined_grad_norm_preclip", wake_result.get("combined_grad_norm_preclip"))
                        row["primary_updates"] = step
                        if sleep_now:
                            current_event = True
                            pool = [rows_by_pack_index[i] for i in pools[step]]
                            model, controller, event_summary = boundary.consolidate(
                                model, controller, proposals, pool,
                                520000000 + seed * 1000 + step, meter,
                                f"v5_seed{seed}_{arm}_sleep{step}")
                            current_event = False
                            set_warmmix_route(model, arm)
                            event_summary.update({"primary_boundary":step,"arm":arm,
                                "wake_mode":kind,"anchor_old_aux_count":len(old_rows),
                                "planned_pool_class_counts":{str(k):6 for k in range(4)},
                                "receiver_route_after_commit":model.banks()[1].route_override})
                            row["events"].append({"wake":wake_result,"sleep":event_summary})
                            row["sleep_Adam_updates"] += 24
                            sleep_adam_updates += 24
                        if step in STEPS[1:]:
                            metrics = save(step)
                            write_checkpoint_progress()
                            if arm == "warm_CMS_reference" and step in (16, 112):
                                row.setdefault("v4_reference_checks",[]).append(
                                    check_v4_reference(args,seed,step,row["checkpoints"][-1],metrics,v4))
                        if step in (16,64,112,256,448):
                            print(json.dumps({"seed":seed,"arm":arm,"primary_step":step,
                                "sleep_adam_updates":row["sleep_Adam_updates"],"status":"running"}),flush=True)

                    # Fixed final scoring on train and validation; no endpoint selection.
                    pre_train_score_fp = c.fingerprint(model)
                    pre_train_score_policy = execution_state(model)
                    pre_train_score_rng = c.engine.shared.tensor_tree_hash(c.engine.capture_rng())
                    train_metrics = c.score(model, train_rows, pack, seed,
                        f"v5_train_seed{seed}_{arm}_step448", meter)
                    assert c.fingerprint(model) == pre_train_score_fp
                    assert execution_state(model) == pre_train_score_policy
                    assert c.engine.shared.tensor_tree_hash(c.engine.capture_rng()) == pre_train_score_rng
                    check_time()
                    assert set(("old","new","all","class0","class1","class2","class3")).issubset(train_metrics)
                    assert train_metrics["all"]["examples"] == 1024
                    assert train_metrics["old"]["examples"] == train_metrics["new"]["examples"] == 512
                    final_val = row["validation"][-1]["metrics"]
                    gates = {phase: task_gate_details(train_metrics, final_val, phase) for phase in ("old", "new")}
                    assert gates["old"]["pass"] == c.qualifies(train_metrics, final_val, "old")
                    assert gates["new"]["pass"] == c.qualifies(train_metrics, final_val, "new")
                    row["train_final"] = train_metrics
                    row["qualification"] = {"gates":gates,"all_tasks_pass":all(x["pass"] for x in gates.values()),
                        "gate_thresholds":{"train_em":.80,"validation_em":.75,"each_class_validation_em":.70}}
                    assert controller.counts == [0,0]
                    expected_events = 28 if arm in SLEEP_ARMS else 0
                    assert len(row["events"]) == expected_events
                    assert row["sleep_Adam_updates"] == expected_events * 24
                    final_slot_counts = attach_final_active_parameters(
                        row, c.engine.shared.active_parameter_counts(model), arm in SLEEP_ARMS)
                    row["old_receiver_provenance"] = {
                        "receiver_zero_equal_warm": old_receiver_zero_hash(model) == warm_receiver0_hash,
                        "warm_active_slot_prefix_equal": active_prefix_hash(model, 1, warm_active_counts[1]) == warm_prefix_hash,
                        "warm_active_slot_prefix_count": warm_active_counts[1],
                    }
                    assert row["old_receiver_provenance"]["receiver_zero_equal_warm"]
                    assert row["old_receiver_provenance"]["warm_active_slot_prefix_equal"]
                    row.update(status="complete",seconds=time.monotonic()-arm_start_time,
                        final_fingerprint=c.fingerprint(model),
                        work={k:v.copy() for k,v in meter.counts.items() if k.startswith((
                            f"v5_seed{seed}_{arm}_", f"v5_validation_seed{seed}_{arm}_", f"v5_train_seed{seed}_{arm}_"))})
                    assert c.fingerprint(teacher) == teacher_fp
                    assert execution_state(teacher) == teacher_policy
                    assert immutable_versions(teacher) == teacher_immutable
                except BaseException as exc:
                    partial_upper = 24 if current_event else 0
                    row.update(status="failed",error_type=type(exc).__name__,error=str(exc),
                        attempted_primary_updates=row.get("attempted_primary_updates",row["primary_updates"]),
                        sleep_Adam_updates_completed=row["sleep_Adam_updates"],
                        partial_sleep_adam_updates_lower_bound=0,
                        partial_sleep_adam_updates_upper_bound=partial_upper,
                        sleep_adam_updates_completed_plus_partial_upper=row["sleep_Adam_updates"]+partial_upper)
                    try:
                        c.checkpoint(args.out_dir/"failure-state-private.pt",model,controller,seed,
                            row["primary_updates"],{"arm":arm,"incomplete_event":current_event,"protocol_sha256":protocol_sha})
                    except Exception:
                        pass
                    write_progress(args.out_dir/"progress.json",{"schema":"agnews-anchored-progress-v5",
                        "runs":runs,"confirmation_scored":False,"complete":False})
                    raise
                write_progress(args.out_dir/"progress.json",{"schema":"agnews-anchored-progress-v5",
                    "runs":runs,"confirmation_scored":False,"complete":False})
            del model, teacher, controller, warm_cp
            torch.cuda.empty_cache()
        assert len(runs) == 21 and all(r["status"] == "complete" for r in runs)
        assert primary_attempts == 9408 and sleep_adam_updates == 8064
        assert sum(r["primary_updates"] for r in runs) == 9408
        assert sum(len(r["events"]) for r in runs) == 336
        assert all(len(r["validation"]) == 6 for r in runs)
        assert all(r["train_final"] is not None and "qualification" in r for r in runs)
        seal={"schema":"agnews-anchored-development-seal-v5","complete":True,"runs":runs,
            "protocol_sha256":protocol_sha,"config_sha256":config_hash,"source_sha256":source_hashes,
            "warm_result_sha256":warm_hash,"v4_development_seal_sha256":v4_seal_hash,
            "actual_primary_updates":primary_attempts,"actual_sleep_Adam_updates":sleep_adam_updates,
            "actual_kd_old_aux_steps":3*3*448,"actual_gold_old_aux_steps":3*448,
            "work":meter.counts,"seconds":time.monotonic()-start_time,
            "development_only":True,"confirmation_scored":False,"no_holdout_input":True,
            "completed_utc":datetime.now(timezone.utc).isoformat()}
        write_json(args.out_dir/"development-sealed.json",seal)
        write_progress(args.out_dir/"progress.json",{"schema":"agnews-anchored-progress-v5",
            "runs":runs,"confirmation_scored":False,"complete":True})
        print("ALL21_ANCHORED_DEVELOPMENT_ARMS_SEALED",flush=True)
    except BaseException as exc:
        if not (args.out_dir/"failure.json").exists():
            partial_upper = 24 if current_event else 0
            write_json(args.out_dir/"failure.json",{"schema":"agnews-anchored-failure-v5",
                "error_type":type(exc).__name__,"error":str(exc),"runs":runs,
                "attempted_primary_updates":primary_attempts,"completed_sleep_Adam_updates":sleep_adam_updates,
                "partial_sleep_adam_updates_lower_bound":0,
                "partial_sleep_adam_updates_upper_bound":partial_upper,
                "completed_sleep_adam_updates_plus_partial_upper":sleep_adam_updates+partial_upper,
                "work":meter.counts,"confirmation_scored":False})
        raise
    finally:
        c.engine.ACTIVE_METER.reset(token)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("records","warm-dir","v4-run-dir","model-dir","config","qualification","out-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
