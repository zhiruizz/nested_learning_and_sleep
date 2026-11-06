"""Frozen development comparison of two balanced gold-aux continuation controls.

Both arms resume the same v5 CMS-reference checkpoint at primary update 448,
use the same fixed new-task schedule through 1024, and make one CMS collect per
update. Auxiliary CE examples come from either the old or new training classes.
This is a controlled engineering comparison, not a Sleep method or test result.
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
import agnews_fair_continuation_v6 as helper
import agnews_anchored_wake_v5 as frozen_v5_helper
from nested_sleep.lm import execution_state, immutable_versions, isolated_clone
from nested_sleep.optim import CMSController

SEEDS = (17, 29, 43)
ARMS = (("aux_new_CE", (2, 3)), ("aux_old_gold_CE", (0, 1)))
START_STEP, END_STEP = 448, 1024
EVAL_STEPS = (448, 512, 768, 1024)
PRIMARY_COUNT = END_STEP - START_STEP


def sha(path):
    return c.sha(Path(path))


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_progress(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def aux_schedule(pack, seed, rows, classes):
    by_class = {k: [i for i, r in enumerate(pack["records"])
                    if r["role"] == "train" and r["class_id"] == k] for k in classes}
    schedule = {}
    for step in range(START_STEP + 1, END_STEP + 1):
        rng = random.Random(f"nl-sleep-v6-aux:{'new' if classes == (2, 3) else 'old'}:{seed}:{step}")
        chosen = [i for cls in classes for i in rng.sample(by_class[cls], 2)]
        rng.shuffle(chosen)
        assert len(chosen) == 4 and [sum(pack["records"][i]["class_id"] == cls for i in chosen)
                                     for cls in classes] == [2, 2]
        schedule[step] = chosen
    return schedule


def schedule_hash(schedule):
    return hash_json({str(k): v for k, v in schedule.items()})


def build_plan_identity(current, cp_info, qualification_sha256, new_hashes, aux_hashes):
    return {"source_sha256":current,"warm_start_checkpoints":cp_info,
        "qualification_sha256":qualification_sha256,
        "new_schedule_sha256_by_seed":new_hashes,
        "aux_schedule_sha256_by_seed_arm":aux_hashes,
        "steps":{"start":START_STEP,"end":END_STEP,"evaluation":list(EVAL_STEPS)},
        "arms":[x[0] for x in ARMS],"loss_weights":{"primary_new_ce":1.0,"auxiliary_ce":1.0}}


def qualification_shared_sources_match(qplan, current, qualification_driver_sha256):
    shared_keys=set(current)-{"driver_source_sha256"}
    return (all(qplan.get(k)==current[k] for k in shared_keys)
            and qplan.get("qualifier_source_sha256")==qualification_driver_sha256)


def active_prefix_hash(model, bank_index, count):
    slots = model.banks()[bank_index].slots
    if count > len(slots):
        raise AssertionError("active warm prefix exceeds allocated slots")
    state = {str(i): {k:v.detach().cpu().clone() for k,v in slots[i].state_dict().items()}
             for i in range(count)}
    return c.engine.shared.tensor_tree_hash(state)


def receiver0_hash(model):
    slots=model.banks()[1].slots
    if not slots: raise AssertionError("old receiver slot zero is missing")
    return c.engine.shared.tensor_tree_hash({k:v.detach().cpu().clone()for k,v in slots[0].state_dict().items()})


def assert_adapter_nonalias(model, teacher):
    model_state=model.adapter_state_dict(); teacher_state=teacher.adapter_state_dict()
    assert set(model_state)==set(teacher_state)
    model_mutable={id(p) for n,p in model.named_parameters() if n in model_state}
    teacher_mutable={id(p) for n,p in teacher.named_parameters() if n in teacher_state}
    assert model_mutable.isdisjoint(teacher_mutable), "updating adapter tensors alias frozen teacher"


def locate_reference_run(seal, seed):
    rows = [r for r in seal["runs"] if r["seed"] == seed and r["arm"] == "warm_CMS_reference"]
    assert len(rows) == 1 and rows[0]["status"] == "complete"
    assert rows[0]["primary_updates"] == START_STEP
    return rows[0]


def fixed_checkpoint(run_row, step):
    rows = [x for x in run_row["checkpoints"] if x["primary_step"] == step]
    assert len(rows) == 1
    return rows[0]


def fixed_validation(run_row, step):
    rows = [x for x in run_row["validation"] if x["primary_step"] == step]
    assert len(rows) == 1
    return rows[0]


def task_gate_details(train, val, phase):
    classes = (0, 1) if phase == "old" else (2, 3)
    out = {"train_phase_em": train[phase][c.EM], "train_phase_pass": train[phase][c.EM] >= .80,
        "validation_phase_em": val[phase][c.EM], "validation_phase_pass": val[phase][c.EM] >= .75,
        "validation_class_em": {str(k): val[f"class{k}"][c.EM] for k in classes},
        "validation_class_pass": {str(k): val[f"class{k}"][c.EM] >= .70 for k in classes}}
    out["pass"] = out["train_phase_pass"] and out["validation_phase_pass"] and all(out["validation_class_pass"].values())
    return out


def validate_inputs(a, cfg, pack, warm, v5, v5_protocol):
    assert cfg["schema"] == "agnews-fair-continuation-config-v6"
    assert cfg["review_status"] == "FROZEN_DEVELOPMENT_AFTER_REVIEW"
    assert cfg["arms"] == [x[0] for x in ARMS]
    assert cfg["seeds"] == list(SEEDS)
    assert cfg["start_step"] == START_STEP and cfg["end_step"] == END_STEP
    assert cfg["evaluation_steps"] == list(EVAL_STEPS)
    assert cfg["primary_updates_per_arm_seed"] == PRIMARY_COUNT
    assert cfg["primary_schedule_batch_slice"] == [448,1024]
    assert cfg["pack_schedule_batch_count"] == 4096 and cfg["examples_per_primary_batch"] == 4
    assert cfg["primary_batch_size"] == 4 and cfg["auxiliary_batch_size"] == 4
    assert cfg["base_lr"] == .02 and cfg["expert_lr"] == .04
    assert cfg["periods"] == [4, 16] and cfg["gradient_clip"] == 1.0
    assert cfg["capacity"] == 32
    assert cfg["primary_optimizer"] == "CMS" and cfg["auxiliary_mode"] == "gold_ce"
    assert cfg["loss_weights"] == {"primary_new_ce": 1.0, "auxiliary_ce": 1.0}
    assert cfg["auxiliary_examples_per_class"] == 2
    assert cfg["no_sleep"] is True and cfg["early_stopping"] is False
    assert cfg["elapsed_limit_seconds"] == 1800
    assert cfg["elapsed_limit_scope"] == "active model load, training, and scoring; CPU preflight excluded"
    assert cfg["confirmation_input"] is False and cfg["confirmation_scored"] is False
    expected_new_hashes = {str(s):pack["schedules"][str(s)]["new_sha256"] for s in SEEDS}
    assert cfg["new_schedule_sha256_by_seed"] == expected_new_hashes
    expected_aux_hashes = {f"{s}:{name}":schedule_hash(aux_schedule(pack,s,c.rows(pack,s),classes))
                           for s in SEEDS for name,classes in ARMS}
    assert cfg["aux_schedule_sha256_by_seed_arm"] == expected_aux_hashes
    current = {"driver_source_sha256": sha(Path(__file__)),
        "continuation_helper_sha256": sha(Path(helper.__file__)),
        "v5_anchor_helper_sha256": sha(Path(frozen_v5_helper.__file__)),
        "common_sha256": sha(Path(c.__file__)), "engine_sha256": sha(Path(c.engine.__file__)),
        "warm_result_sha256": sha(a.warm_dir / "transfer-result.json"),
        "v5_development_seal_sha256": sha(a.v5_run_dir / "development-sealed.json"),
        "v5_protocol_sha256": sha(a.v5_run_dir / "protocol.json"), "records_sha256": sha(a.records)}
    assert all(cfg.get(k) == v for k, v in current.items()), "frozen config source/input lock mismatch"
    assert warm["all3_qualified"] and warm["confirmation_scored"] is False
    assert v5["complete"] and v5["confirmation_scored"] is False and v5["no_holdout_input"] is True
    assert v5_protocol["confirmation_input"] is False and v5_protocol["confirmation_scored"] is False
    assert v5["protocol_sha256"] == current["v5_protocol_sha256"]
    assert v5["warm_result_sha256"] == current["warm_result_sha256"]
    assert all(len(pack["schedules"][str(s)]["new"]) == 4096 for s in SEEDS)
    assert all(len(batch) == 4 for s in SEEDS for batch in pack["schedules"][str(s)]["new"])
    return current


def validate_qualification(path, cfg, current, cp_hashes):
    q = c.read(path)
    plan_path = Path(path).parent / "plan.json"
    qplan = c.read(plan_path)
    assert q.get("schema") == "agnews-fair-resume-qualified-v6" and q.get("passed") is True
    assert q.get("exact_model_controller_policy_RNG") is True
    assert q.get("scoring") is False and q.get("confirmation_scored") is False and q.get("confirmation_input") is False
    assert q.get("actual_primary_updates") == 12
    assert q.get("actual_student_forward_passes") == q.get("actual_student_backward_passes") == 24
    assert q.get("actual_teacher_forward_passes") == 0
    assert q.get("actual_auxiliary_forward_passes") == q.get("actual_auxiliary_backward_passes") == 12
    assert q.get("actual_primary_target_examples") == q.get("actual_auxiliary_target_examples") == 48
    assert q.get("actual_primary_prefix_draws") == q.get("actual_auxiliary_prefix_draws") == 48
    assert q.get("actual_primary_answer_token_targets") == q.get("actual_auxiliary_answer_token_targets") == 96
    assert q.get("continuation_helper_sha256") == current["continuation_helper_sha256"]
    assert q.get("source_sha256") == cfg["qualification_driver_sha256"]
    assert q.get("config_sha256") == cfg["qualification_config_sha256"]
    assert sha(path) == cfg["qualification_result_sha256"]
    assert q.get("plan_sha256") == cfg["qualification_plan_sha256"] == sha(plan_path)
    assert qplan.get("schema") == "agnews-fair-resume-plan-v6"
    assert qplan.get("v5_start_checkpoint_sha256") == q.get("v5_start_checkpoint_sha256")
    assert qplan.get("aux_schedule_sha256_by_arm") == {x["arm"]:x["aux_schedule_sha256"] for x in q.get("arms",[])}
    assert qplan.get("consumed_aux_schedule_prefix_sha256_by_arm") == {
        x["arm"]:x["consumed_aux_schedule_prefix_sha256"] for x in q.get("arms",[])}
    assert {x["arm"] for x in q.get("arms",[])} == {x[0] for x in ARMS}
    assert qualification_shared_sources_match(qplan,current,cfg["qualification_driver_sha256"])
    assert next(x["checkpoint_sha256"] for x in cp_hashes if x["seed"] == 17) == q["v5_start_checkpoint_sha256"]
    assert len(q.get("arms", [])) == 2 and all(x.get("exact_model_controller_policy_rng") is True for x in q["arms"])
    assert all(x["resume_branch_state_sha256"][0] == x["resume_branch_state_sha256"][1] for x in q["arms"])
    for arm in q["arms"]:
        pending = Path(arm["pending_checkpoint"]["path"])
        if not pending.is_absolute(): pending = Path(path).parent / pending
        assert sha(pending) == arm["pending_checkpoint"]["sha256"]
    return q


def run(a):
    setup_start = time.monotonic()
    cfg = c.read(a.config)
    pack = c.load_pack(a.records)
    warm = c.read(a.warm_dir / "transfer-result.json")
    v5 = c.read(a.v5_run_dir / "development-sealed.json")
    v5_protocol = c.read(a.v5_run_dir / "protocol.json")
    current = validate_inputs(a, cfg, pack, warm, v5, v5_protocol)
    cp_info = []
    for seed in SEEDS:
        start_meta = fixed_checkpoint(locate_reference_run(v5, seed), START_STEP)
        cp_path = a.v5_run_dir / start_meta["path"]
        assert sha(cp_path) == start_meta["sha256"]
        assert start_meta["step"] == START_STEP and start_meta["controller_counts"] == [0,0]
        cp_info.append({"seed": seed, "checkpoint_sha256": start_meta["sha256"], "path": start_meta["path"]})
    qualification = validate_qualification(a.qualification, cfg, current, cp_info)
    qarms = {x["arm"]:x for x in qualification["arms"]}
    for arm, classes in ARMS:
        full_schedule = aux_schedule(pack,17,c.rows(pack,17),classes)
        expected = schedule_hash(full_schedule)
        assert qarms[arm]["aux_schedule_sha256"] == expected
        assert qarms[arm]["consumed_aux_schedule_prefix_sha256"] == schedule_hash(
            {s:full_schedule[s] for s in (449,450,451,452)})
    c.preflight(a.model_dir, pack)
    a.out_dir.mkdir(parents=True, exist_ok=False)
    (a.out_dir / "checkpoints").mkdir()
    plan_identity = build_plan_identity(current,cp_info,sha(a.qualification),
        {str(s):pack["schedules"][str(s)]["new_sha256"] for s in SEEDS},
        cfg["aux_schedule_sha256_by_seed_arm"])
    plan_sha = hash_json(plan_identity)
    assert cfg["protocol_plan_sha256"] == plan_sha
    protocol = {"schema":"agnews-fair-continuation-protocol-v6", "status":"frozen_development",
        "plan_sha256":plan_sha,"plan_identity":plan_identity,
        "config_sha256":sha(a.config), "config":cfg, "source_sha256":current,
        "qualification_sha256":sha(a.qualification), "qualification_plan_sha256":qualification["plan_sha256"],
        "v5_start_checkpoints":cp_info, "seeds":list(SEEDS), "arms":[x[0] for x in ARMS],
        "start_step":START_STEP, "end_step":END_STEP, "evaluation_steps":list(EVAL_STEPS),
        "planned_primary_updates":2*3*PRIMARY_COUNT, "planned_student_forward_passes":2*3*PRIMARY_COUNT*2,
        "planned_student_backward_passes":2*3*PRIMARY_COUNT*2, "planned_teacher_forward_passes":0,
        "planned_auxiliary_forward_passes":2*3*PRIMARY_COUNT,
        "planned_auxiliary_backward_passes":2*3*PRIMARY_COUNT,
        "planned_primary_target_examples":2*3*PRIMARY_COUNT*4,
        "planned_auxiliary_target_examples":2*3*PRIMARY_COUNT*4,
        "planned_primary_answer_token_targets":2*3*PRIMARY_COUNT*4*2,
        "planned_auxiliary_answer_token_targets":2*3*PRIMARY_COUNT*4*2,
        "planned_auxiliary_prefix_draws":2*3*PRIMARY_COUNT*4,
        "planned_old_class_auxiliary_prefix_draws":3*PRIMARY_COUNT*4,
        "planned_new_class_auxiliary_prefix_draws":3*PRIMARY_COUNT*4,
        "planned_primary_prefix_draws":2*3*PRIMARY_COUNT*4,
        "pack_schedule_batch_count":4096,"primary_schedule_batch_slice":[448,1024],
        "primary_schedule_batch_slice_semantics":"zero_based_half_open",
        "examples_per_primary_batch":4,
        "cost_note":"Both arms use one new-task B4 forward/backward and one balanced auxiliary B4 forward/backward before a single global clip and one CMS collect per primary update. Aux class source differs by arm (new classes 2/3 versus old classes 0/1), so label information differs; the comparison is compute-matched, not information-matched. The frozen teacher is passed only for helper integrity checks and incurs zero teacher forwards in gold mode.",
        "new_schedule_sha256_by_seed":{str(s):pack["schedules"][str(s)]["new_sha256"] for s in SEEDS},
        "aux_schedule_sha256_by_seed_arm":{}, "confirmation_input":False,"confirmation_scored":False,
        "exploratory_only":True,"automatic_retry":False,"created_utc":datetime.now(timezone.utc).isoformat()}
    aux_schedules = {}
    for seed in SEEDS:
        for name, classes in ARMS:
            sch = aux_schedule(pack, seed, c.rows(pack, seed), classes)
            aux_schedules[(seed,name)] = sch
            protocol["aux_schedule_sha256_by_seed_arm"][f"{seed}:{name}"] = schedule_hash(sch)
    protocol["elapsed_limit_seconds"] = cfg["elapsed_limit_seconds"]
    protocol["elapsed_limit_scope"] = cfg["elapsed_limit_scope"]
    protocol["cpu_preflight_and_plan_wall_seconds"] = time.monotonic()-setup_start
    c.write(a.out_dir / "protocol.json", protocol)
    protocol_sha = sha(a.out_dir / "protocol.json")
    start_time = time.monotonic()
    meter = c.engine.WorkMeter()
    meter_token = c.engine.ACTIVE_METER.set(meter)
    runs = []
    attempts = 0

    def check_time():
        if time.monotonic() - start_time >= cfg["elapsed_limit_seconds"]:
            raise TimeoutError("v6 development elapsed-time limit")

    def save_checkpoint(model, controller, seed, arm, stepnum, row, pack, meter):
        before_fp = c.fingerprint(model)
        before_policy = execution_state(model)
        before_rng = c.engine.shared.tensor_tree_hash(c.engine.capture_rng())
        before_immutable = immutable_versions(model)
        val_rows = c.rows(pack, seed, role="validation")
        val = c.score(model, val_rows, pack, seed, f"v6_validation_seed{seed}_{arm}_step{stepnum}", meter)
        assert val["all"]["examples"] == 512 and val["old"]["examples"] == val["new"]["examples"] == 256
        assert c.fingerprint(model) == before_fp and execution_state(model) == before_policy
        assert c.engine.shared.tensor_tree_hash(c.engine.capture_rng()) == before_rng
        assert immutable_versions(model) == before_immutable
        check_time()
        cp_path = a.out_dir / "checkpoints" / f"seed-{seed}-{arm}-step-{stepnum:04}.pt"
        info = c.checkpoint(cp_path, model, controller, seed, stepnum, {
            "arm":arm,"primary_cursor":stepnum,"protocol_sha256":protocol_sha,
            "driver_source_sha256":current["driver_source_sha256"],"config_sha256":sha(a.config),
            "new_schedule_sha256":pack["schedules"][str(seed)]["new_sha256"],
            "aux_schedule_sha256":row["aux_schedule_sha256"],"aux_class_ids":row["aux_class_ids"],
        })
        info.update(path=str(cp_path.relative_to(a.out_dir)),primary_step=stepnum)
        row["checkpoints"].append(info)
        row["validation"].append({"primary_step":stepnum,"metrics":val,"adapter_fingerprint":before_fp,
            "execution_policy":before_policy,"rng_sha256":before_rng,"controller_counts":list(controller.counts)})
        write_progress(a.out_dir / "progress.json",{"schema":"agnews-fair-continuation-progress-v6",
            "runs":runs,"attempted_primary_updates":attempts,"complete":False,"confirmation_scored":False})
        return val

    try:
        for seed in SEEDS:
            check_time()
            train_rows = c.rows(pack, seed, role="train")
            rows = c.rows(pack, seed)
            v5_run = locate_reference_run(v5, seed)
            start_meta = fixed_checkpoint(v5_run, START_STEP)
            start_cp_path = a.v5_run_dir / start_meta["path"]
            start_cp = torch.load(start_cp_path, map_location="cpu", weights_only=False)
            assert start_cp["seed"] == seed and start_cp["step"] == START_STEP
            assert start_cp["controller"]["periods"] == (4,16)
            warm_ref = next(x for x in warm["qualified_models"] if x["seed"] == seed)
            warm_cp_path = a.warm_dir / warm_ref["checkpoint"]["path"]
            assert sha(warm_cp_path) == warm_ref["checkpoint"]["sha256"]
            warm_cp = torch.load(warm_cp_path, map_location="cpu", weights_only=False)
            model = c.model(a.model_dir, seed)
            c.restore(model, start_cp)
            c.engine.restore_rng(start_cp["rng"])
            model.eval().requires_grad_(False)
            assert c.fingerprint(model) == start_meta["adapter_fingerprint"]
            warm_policy = execution_state(model)
            warm_immutable = immutable_versions(model)
            warm_capacity = model.capacity
            assert warm_capacity == cfg["capacity"]
            warm_active = c.engine.shared.active_parameter_counts(model)["active_slots_by_bank"]
            warm_prefix_count = int(warm_active[1])
            assert warm_prefix_count > 0
            warm_prefix = active_prefix_hash(model,1,warm_prefix_count)
            warm_receiver0 = receiver0_hash(model)
            # Copy mutable adapters from the qualified warm snapshot while
            # sharing only the already-frozen identical backbone tensors.
            teacher = isolated_clone(model, share_backbone=True)
            c.restore(teacher, warm_cp)
            teacher.eval().requires_grad_(False)
            assert_adapter_nonalias(model,teacher)
            teacher_fp = c.fingerprint(teacher)
            teacher_policy = execution_state(teacher)
            teacher_immutable = immutable_versions(teacher)
            initial_fp = c.fingerprint(model)
            initial_policy = execution_state(model)
            initial_rng = c.engine.shared.tensor_tree_hash(c.engine.capture_rng())
            initial_counts = list(start_cp["controller"]["counts"])
            assert initial_counts == [0, 0]
            for arm, aux_classes in ARMS:
                check_time()
                c.restore(model, start_cp)
                c.engine.restore_rng(start_cp["rng"])
                controller = CMSController.restore(model, start_cp["controller"])
                assert c.fingerprint(model) == initial_fp
                assert execution_state(model) == initial_policy
                assert c.engine.shared.tensor_tree_hash(c.engine.capture_rng()) == initial_rng
                assert controller.counts == [0,0]
                sch = aux_schedules[(seed,arm)]
                row = {"seed":seed,"arm":arm,"aux_class_ids":list(aux_classes),"status":"running",
                    "start_step":START_STEP,"primary_updates":0,"checkpoints":[],"validation":[],
                    "train_final":None,"aux_schedule_sha256":protocol["aux_schedule_sha256_by_seed_arm"][f"{seed}:{arm}"],
                    "work":{},"training_loss_stats":{},"teacher_initial_fingerprint":teacher_fp,
                    "student_forward_passes":0,"student_backward_passes":0,"teacher_forward_passes":0,
                    "primary_target_examples":0,"auxiliary_target_examples":0,"auxiliary_prefix_draws":0,
                    "primary_answer_token_targets":0,"auxiliary_answer_token_targets":0,
                    "auxiliary_forward_passes":0,"auxiliary_backward_passes":0}
                runs.append(row)
                # Starting checkpoint is the existing v5 validation point, not rescored here.
                row["validation"].append({"primary_step":START_STEP,"metrics":fixed_validation(v5_run,START_STEP)["metrics"],
                    "adapter_fingerprint":initial_fp,"execution_policy":initial_policy,
                    "rng_sha256":initial_rng,"controller_counts":[0,0],"inherited_from_v5":True})
                write_progress(a.out_dir / "progress.json",{"schema":"agnews-fair-continuation-progress-v6",
                    "runs":runs,"attempted_primary_updates":attempts,"complete":False,"confirmation_scored":False})
                try:
                    for stepnum in range(START_STEP+1,END_STEP+1):
                        check_time()
                        new_idx = pack["schedules"][str(seed)]["new"][stepnum-1]
                        new_rows = [rows[i] for i in new_idx]
                        aux_rows = [rows[i] for i in sch[stepnum]]
                        assert len(new_rows) == len(aux_rows) == 4
                        attempts += 1
                        result, _ = helper.anchored_step(model,controller,new_rows,aux_rows,
                            pack["tokenizer"]["pad_token_id"],meter,
                            f"v6_seed{seed}_{arm}_primary",teacher,mode="gold",aux_classes=aux_classes)
                        row["primary_updates"] = stepnum-START_STEP
                        row["student_forward_passes"] = row["student_backward_passes"] = row["primary_updates"]*2
                        assert result["student_forward_passes"] == result["student_backward_passes"] == 2
                        assert result["auxiliary_forward_passes"] == result["auxiliary_backward_passes"] == 1
                        assert result["auxiliary_class_ids"] == list(aux_classes)
                        row["auxiliary_forward_passes"] += result["auxiliary_forward_passes"]
                        row["auxiliary_backward_passes"] += result["auxiliary_backward_passes"]
                        row["primary_target_examples"] = row["auxiliary_target_examples"] = row["primary_updates"]*4
                        row["auxiliary_prefix_draws"] = row["primary_updates"]*4
                        row["primary_answer_token_targets"] = row["auxiliary_answer_token_targets"] = row["primary_updates"]*8
                        for key in ("new_ce","auxiliary_gold_ce","combined_grad_norm_preclip"):
                            value = float(result[key])
                            stat = row["training_loss_stats"].setdefault(key,{"count":0,"sum":0.,"min":None,"max":None})
                            stat["count"] += 1; stat["sum"] += value
                            stat["min"] = value if stat["min"] is None else min(stat["min"],value)
                            stat["max"] = value if stat["max"] is None else max(stat["max"],value)
                        assert result["teacher_unchanged"] and result["global_rng_unchanged"]
                        assert immutable_versions(model)==warm_immutable and model.capacity==warm_capacity
                        if stepnum in EVAL_STEPS[1:]:
                            save_checkpoint(model,controller,seed,arm,stepnum,row,pack,meter)
                        if stepnum in (512,768,1024):
                            print(json.dumps({"seed":seed,"arm":arm,"primary_step":stepnum,"status":"running"}),flush=True)
                    assert controller.counts == [0,0]
                    primary_work=meter.counts[f"v6_seed{seed}_{arm}_primary"]
                    assert primary_work["forward_calls"] == row["primary_updates"]*2
                    assert primary_work["gradient_forward_input_tokens"] == primary_work["forward_input_tokens"]
                    row["primary_training_work"]={k:primary_work[k] for k in (
                        "forward_calls","forward_input_tokens","gradient_forward_input_tokens")}
                    # Endpoint train scoring is fixed at 1024 and guarded for state/RNG changes.
                    before_fp=c.fingerprint(model); before_policy=execution_state(model)
                    before_rng=c.engine.shared.tensor_tree_hash(c.engine.capture_rng())
                    before_immutable=immutable_versions(model)
                    train_metrics=c.score(model,train_rows,pack,seed,f"v6_train_seed{seed}_{arm}_step1024",meter)
                    assert c.fingerprint(model)==before_fp and execution_state(model)==before_policy
                    assert c.engine.shared.tensor_tree_hash(c.engine.capture_rng())==before_rng
                    assert immutable_versions(model)==before_immutable
                    check_time()
                    assert train_metrics["all"]["examples"]==1024
                    assert train_metrics["old"]["examples"]==train_metrics["new"]["examples"]==512
                    val1024=row["validation"][-1]["metrics"]
                    gates={phase:task_gate_details(train_metrics,val1024,phase)for phase in("old","new")}
                    row["train_final"]=train_metrics
                    row["qualification"]={"gates":gates,"all_tasks_pass":all(x["pass"]for x in gates.values()),
                        "gate_thresholds":{"train_em":.80,"validation_em":.75,"each_class_validation_em":.70}}
                    assert c.fingerprint(teacher)==teacher_fp and execution_state(teacher)==teacher_policy
                    assert immutable_versions(teacher)==teacher_immutable
                    final_prefix_equal=active_prefix_hash(model,1,warm_prefix_count)==warm_prefix
                    final_receiver0_equal=receiver0_hash(model)==warm_receiver0
                    assert final_prefix_equal and final_receiver0_equal
                    assert c.fingerprint(model) != "" and model.capacity==warm_capacity
                    row.update(status="complete",final_fingerprint=c.fingerprint(model),
                        old_receiver_provenance={"receiver_zero_equal_start_v5":final_receiver0_equal,
                            "warm_active_prefix_equal_start_v5":final_prefix_equal,
                            "warm_active_prefix_count":warm_prefix_count},
                        work={k:v.copy()for k,v in meter.counts.items()if k.startswith((
                            f"v6_seed{seed}_{arm}_",f"v6_validation_seed{seed}_{arm}_",f"v6_train_seed{seed}_{arm}_"))})
                except BaseException as exc:
                    row.update(status="failed",error_type=type(exc).__name__,error=str(exc),
                        attempted_primary_updates=row.get("primary_updates",0))
                    c.checkpoint(a.out_dir/"failure-state-private.pt",model,controller,seed,
                        START_STEP+row["primary_updates"],{"arm":arm,"protocol_sha256":protocol_sha})
                    write_progress(a.out_dir/"progress.json",{"schema":"agnews-fair-continuation-progress-v6",
                        "runs":runs,"attempted_primary_updates":attempts,"complete":False,"confirmation_scored":False})
                    raise
                write_progress(a.out_dir/"progress.json",{"schema":"agnews-fair-continuation-progress-v6",
                    "runs":runs,"attempted_primary_updates":attempts,"complete":False,"confirmation_scored":False})
            del model, teacher, controller, start_cp, warm_cp
            torch.cuda.empty_cache()
        assert len(runs)==6 and all(r["status"]=="complete"for r in runs)
        assert attempts==2*3*PRIMARY_COUNT and sum(r["primary_updates"]for r in runs)==attempts
        assert all([x["primary_step"]for x in r["validation"]]==list(EVAL_STEPS)for r in runs)
        assert all(r["train_final"] is not None and "qualification" in r for r in runs)
        seal={"schema":"agnews-fair-continuation-development-seal-v6","complete":True,"runs":runs,
            "protocol_sha256":protocol_sha,"config_sha256":sha(a.config),"source_sha256":current,
            "qualification_sha256":sha(a.qualification),"actual_primary_updates":attempts,
            "actual_student_forward_passes":attempts*2,"actual_student_backward_passes":attempts*2,
            "actual_teacher_forward_passes":0,"actual_primary_target_examples":attempts*4,
            "actual_auxiliary_target_examples":attempts*4,"actual_auxiliary_prefix_draws":attempts*4,
            "actual_primary_answer_token_targets":attempts*8,
            "actual_auxiliary_answer_token_targets":attempts*8,
            "actual_old_class_auxiliary_prefix_draws":3*PRIMARY_COUNT*4,
            "actual_new_class_auxiliary_prefix_draws":3*PRIMARY_COUNT*4,
            "work":meter.counts,"seconds":time.monotonic()-start_time,"development_only":True,
            "cpu_preflight_and_plan_wall_seconds":protocol["cpu_preflight_and_plan_wall_seconds"],
            "confirmation_scored":False,"no_holdout_input":True,"completed_utc":datetime.now(timezone.utc).isoformat()}
        write_json(a.out_dir/"development-sealed.json",seal)
        write_progress(a.out_dir/"progress.json",{"schema":"agnews-fair-continuation-progress-v6",
            "runs":runs,"attempted_primary_updates":attempts,"complete":True,"confirmation_scored":False})
        print("ALL6_FAIR_CONTINUATION_RUNS_SEALED",flush=True)
    except BaseException as exc:
        if not (a.out_dir/"failure.json").exists():
            write_json(a.out_dir/"failure.json",{"schema":"agnews-fair-continuation-failure-v6",
                "error_type":type(exc).__name__,"error":str(exc),"runs":runs,
                "attempted_primary_updates":attempts,"work":meter.counts,"confirmation_scored":False})
        raise
    finally:
        c.engine.ACTIVE_METER.reset(meter_token)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("records","warm-dir","v5-run-dir","model-dir","config","qualification","out-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
