"""GPU qualification for exact v6 fair-continuation checkpoint resume.

This correctness run has no scoring/selection. It resumes the two planned arms
from the same v5 CMS step-448 state, forks at pending fast clock [2, 2], and
checks exact model/controller/policy/RNG equality after steps 451-452.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import random
import time
from pathlib import Path

import torch

import agnews_v3_common as c
import agnews_fair_continuation_v6 as helper
import agnews_anchored_wake_v5 as frozen_v5_helper
from nested_sleep.lm import isolated_clone, execution_state
from nested_sleep.optim import CMSController

ARMS = (("aux_new_CE", (2, 3)), ("aux_old_gold_CE", (0, 1)))


def hash_path(path):
    return c.sha(Path(path))


def state_hashes(model, controller):
    state = {"model": model.adapter_state_dict(), "controller": controller.state_dict(),
             "execution_policy": execution_state(model), "RNG": c.engine.capture_rng()}
    return {k: c.engine.shared.tensor_tree_hash(v) for k, v in state.items()}


def assert_adapter_nonalias(model,teacher):
    ms=model.adapter_state_dict();ts=teacher.adapter_state_dict();assert set(ms)==set(ts)
    a={id(p)for n,p in model.named_parameters()if n in ms};b={id(p)for n,p in teacher.named_parameters()if n in ts}
    assert a.isdisjoint(b),"updating adapter tensors alias frozen teacher"


def schedule_hash(schedule):
    payload = json.dumps({str(k):v for k,v in schedule.items()},sort_keys=True,separators=(",",":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_aux_schedule(rows, seed, classes, end_step=1024):
    by_class = {k: [i for i, r in enumerate(rows) if r["role"] == "train" and r["class_id"] == k]
                for k in classes}
    out = {}
    for step in range(449, end_step + 1):
        rng = random.Random(f"nl-sleep-v6-aux:{'new' if classes == (2, 3) else 'old'}:{seed}:{step}")
        picked = [i for cls in classes for i in rng.sample(by_class[cls], 2)]
        rng.shuffle(picked)
        assert len(picked) == 4 and [sum(rows[i]["class_id"] == cls for i in picked) for cls in classes] == [2, 2]
        out[step] = picked
    return out


def prefix_schedule_hash(schedule, steps=(449,450,451,452)):
    return schedule_hash({s:schedule[s] for s in steps})


def run(a):
    setup_start=time.monotonic()
    pack = c.load_pack(a.records)
    warm = c.read(a.warm_dir / "transfer-result.json")
    v5 = c.read(a.v5_run_dir / "development-sealed.json")
    v5_protocol = c.read(a.v5_run_dir / "protocol.json")
    assert warm["all3_qualified"] and warm["confirmation_scored"] is False
    assert v5["complete"] and v5["confirmation_scored"] is False and v5["no_holdout_input"] is True
    assert v5_protocol["confirmation_input"] is False and v5_protocol["confirmation_scored"] is False
    assert v5["protocol_sha256"] == hash_path(a.v5_run_dir / "protocol.json")
    assert v5["warm_result_sha256"] == hash_path(a.warm_dir / "transfer-result.json")
    current = {"qualifier_source_sha256": hash_path(Path(__file__)),
               "continuation_helper_sha256": hash_path(Path(helper.__file__)),
               "v5_anchor_helper_sha256": hash_path(Path(frozen_v5_helper.__file__)),
               "common_sha256": hash_path(Path(c.__file__)),
               "engine_sha256": hash_path(Path(c.engine.__file__)),
               "warm_result_sha256": hash_path(a.warm_dir / "transfer-result.json"),
               "v5_development_seal_sha256": hash_path(a.v5_run_dir / "development-sealed.json"),
               "v5_protocol_sha256": hash_path(a.v5_run_dir / "protocol.json"),
               "records_sha256": hash_path(a.records)}
    cfg = c.read(a.config)
    assert cfg["schema"] == "agnews-fair-continuation-config-v6"
    assert cfg["review_status"] == "FROZEN_DEVELOPMENT_AFTER_REVIEW"
    assert cfg["arms"] == [x[0] for x in ARMS]
    assert cfg["start_step"] == 448 and cfg["fork_step"] == 450 and cfg["resume_steps"] == [451,452]
    assert cfg["loss_weights"] == {"primary_new_ce":1.0,"auxiliary_ce":1.0}
    assert cfg["primary_batch_size"] == cfg["auxiliary_batch_size"] == 4
    assert cfg["expected_primary_updates"] == 12 and cfg["confirmation_input"] is False
    assert cfg["confirmation_scored"] is False and cfg["scoring"] is False
    assert cfg["elapsed_limit_seconds"] == 600
    assert cfg["elapsed_limit_scope"] == "active model training; CPU preflight excluded"
    assert cfg["pack_schedule_batch_count"] == 4096 and cfg["primary_schedule_batch_slice"] == [448,452]
    assert cfg["primary_schedule_batch_slice_semantics"] == "zero_based_half_open"
    assert all(cfg.get(k) == v for k, v in current.items())
    # The pack stores 4096 B4 batches (not 4096 individual example indices).
    assert all(len(pack["schedules"][str(s)]["new"]) == 4096 for s in (17,29,43))
    assert all(len(batch) == 4 for s in (17,29,43) for batch in pack["schedules"][str(s)]["new"])
    arm_rows = [r for r in v5["runs"] if r["seed"] == 17 and r["arm"] == "warm_CMS_reference"]
    assert len(arm_rows) == 1
    run_row = arm_rows[0]
    start_cp_meta = next(x for x in run_row["checkpoints"] if x["primary_step"] == 448)
    start_cp_path = a.v5_run_dir / start_cp_meta["path"]
    assert hash_path(start_cp_path) == start_cp_meta["sha256"]
    assert cfg["resume_start_checkpoint_sha256"] == hash_path(start_cp_path)
    assert run_row["status"] == "complete" and run_row["primary_updates"] == 448
    assert start_cp_meta["controller_counts"] == [0, 0]
    rows_seed17 = c.rows(pack,17)
    schedules = {arm:build_aux_schedule(rows_seed17,17,classes) for arm,classes in ARMS}
    assert cfg["aux_schedule_sha256_by_arm"] == {arm:schedule_hash(schedules[arm]) for arm,_ in ARMS}
    qualification_plan = {
        "schema": "agnews-fair-resume-plan-v6", "review_status": cfg["review_status"],
        **current, "config_sha256": hash_path(a.config),
        "v5_start_checkpoint_sha256": hash_path(start_cp_path), "seed": 17,
        "arms": [x[0] for x in ARMS], "start_step": 448, "prefix_steps": [449, 450],
        "fork_step": 450, "resume_steps": [451, 452], "expected_controller_at_fork": [2, 2],
        "expected_controller_after_resume": [0, 4], "branch_count": 2,
        "expected_primary_updates": 12, "expected_student_forward_passes": 24,
        "expected_student_backward_passes": 24, "expected_auxiliary_examples": 48,
        "expected_primary_prefix_draws":48,"expected_auxiliary_prefix_draws":48,
        "expected_primary_answer_token_targets":96,"expected_auxiliary_answer_token_targets":96,
        "primary_schedule_batch_slice":[448,452],"primary_schedule_batch_slice_semantics":"zero_based_half_open",
        "aux_schedule_sha256_by_arm":{arm:schedule_hash(schedules[arm]) for arm,_ in ARMS},
        "consumed_aux_schedule_prefix_sha256_by_arm":{arm:prefix_schedule_hash(schedules[arm]) for arm,_ in ARMS},
        "teacher_forward_passes": 0, "new_schedule_batch_count":4096,
        "primary_batch_indices_consumed":[448,452],"examples_per_primary_batch":4,
        "scoring": False, "confirmation_scored": False,
        "confirmation_input": False, "automatic_retry": False,"elapsed_limit_seconds":600,
        "elapsed_limit_scope":"active model training; CPU preflight excluded",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    a.out_dir.mkdir(parents=True, exist_ok=False)
    c.write(a.out_dir / "plan.json", qualification_plan)
    c.preflight(a.model_dir, pack)
    preflight_wall_seconds=time.monotonic()-setup_start
    start_time = time.monotonic()
    meter = c.engine.WorkMeter()
    token = c.engine.ACTIVE_METER.set(meter)
    attempts = 0
    outputs = []

    def step(model, controller, teacher, rows, schedule, classes, stepnum):
        nonlocal attempts
        if time.monotonic()-start_time >= cfg["elapsed_limit_seconds"]:
            raise TimeoutError("v6 resume qualification elapsed limit")
        attempts += 1
        new_idx = pack["schedules"]["17"]["new"][stepnum - 1]
        new_batch = [rows[i] for i in new_idx]
        aux_batch = [rows[i] for i in schedule[stepnum]]
        assert len(new_batch) == len(aux_batch) == 4
        result, _ = helper.anchored_step(model, controller, new_batch, aux_batch,
            pack["tokenizer"]["pad_token_id"], meter, f"v6qual_{classes}_{stepnum}",
            teacher, mode="gold", aux_classes=classes)
        assert result["teacher_unchanged"] and result["global_rng_unchanged"]
        return result

    try:
        model = c.model(a.model_dir, 17)
        rows = rows_seed17
        v5_cp = torch.load(start_cp_path, map_location="cpu", weights_only=False)
        assert v5_cp["seed"] == 17 and v5_cp["step"] == 448
        assert v5_cp["controller"]["periods"] == (4,16)
        c.restore(model, v5_cp)
        c.engine.restore_rng(v5_cp["rng"])
        teacher_ref = next(x for x in warm["qualified_models"] if x["seed"] == 17)
        teacher_cp_path = a.warm_dir / teacher_ref["checkpoint"]["path"]
        assert hash_path(teacher_cp_path) == teacher_ref["checkpoint"]["sha256"]
        teacher_cp = torch.load(teacher_cp_path, map_location="cpu", weights_only=False)
        teacher = isolated_clone(model, share_backbone=True)
        c.restore(teacher, teacher_cp)
        teacher.eval().requires_grad_(False)
        assert_adapter_nonalias(model,teacher)
        teacher_hash = c.engine.shared.tensor_tree_hash(teacher.adapter_state_dict())
        for arm, classes in ARMS:
            c.restore(model, v5_cp)
            c.engine.restore_rng(v5_cp["rng"])
            controller = CMSController.restore(model, v5_cp["controller"])
            assert controller.counts == [0, 0]
            schedule = schedules[arm]
            for stepnum in (449, 450):
                step(model, controller, teacher, rows, schedule, classes, stepnum)
            assert controller.counts == [2, 2]
            fork_state = c.engine.shared.compact_cpu({"model": model.adapter_state_dict(),
                "execution_state": execution_state(model), "controller": controller.state_dict(),
                "rng": c.engine.capture_rng()})
            fork_hashes = state_hashes(model, controller)
            pending_path = a.out_dir / f"pending-{arm}-seed17-step450-private.pt"
            pending_meta = c.checkpoint(pending_path, model, controller, 17, 450,
                {"arm":arm,"aux_classes":list(classes),"qualification_plan_sha256":hash_path(a.out_dir/"plan.json")})
            assert hash_path(pending_path) == pending_meta["sha256"]
            fork_loaded = torch.load(pending_path, map_location="cpu", weights_only=False)
            assert state_hashes(model, controller) == {
                "model":c.engine.shared.tensor_tree_hash(fork_loaded["model"]),
                "controller":c.engine.shared.tensor_tree_hash(fork_loaded["controller"]),
                "execution_policy":c.engine.shared.tensor_tree_hash(fork_loaded["execution_state"]),
                "RNG":c.engine.shared.tensor_tree_hash(fork_loaded["rng"])}
            branch_hashes = []
            for branch_index in range(2):
                c.engine.shared.load_snapshot(model, fork_loaded["model"])
                from nested_sleep.lm import load_execution_state
                load_execution_state(model, fork_loaded["execution_state"])
                model.eval().requires_grad_(False)
                controller = CMSController.restore(model, fork_loaded["controller"])
                c.engine.restore_rng(fork_loaded["rng"])
                for stepnum in (451, 452):
                    step(model, controller, teacher, rows, schedule, classes, stepnum)
                assert controller.counts == [0, 4]
                branch_hashes.append(state_hashes(model, controller))
            assert branch_hashes[0] == branch_hashes[1]
            assert c.engine.shared.tensor_tree_hash(teacher.adapter_state_dict()) == teacher_hash
            outputs.append({"arm": arm, "classes": list(classes), "fork_controller_counts": [2, 2],
                "fork_state_sha256": fork_hashes, "resume_branch_state_sha256": branch_hashes,
                "exact_model_controller_policy_rng": True, "primary_updates": 6,
                "student_forward_passes": 12, "student_backward_passes": 12,
                "auxiliary_forward_passes":6,"auxiliary_backward_passes":6,
                "primary_target_examples":24,"auxiliary_target_examples":24,
                "primary_prefix_draws":24,"auxiliary_prefix_draws":24,
                "primary_answer_token_targets":48,"auxiliary_answer_token_targets":48,
                "auxiliary_examples": 24, "teacher_forward_passes": 0,
                "aux_schedule_sha256":schedule_hash(schedule),
                "consumed_aux_schedule_prefix_sha256":prefix_schedule_hash(schedule),
                "pending_checkpoint":{"path":pending_path.name,"sha256":pending_meta["sha256"]}})
        qual_stages={k:v for k,v in meter.counts.items() if k.startswith("v6qual_")}
        assert sum(v["forward_calls"] for v in qual_stages.values()) == attempts*2
        assert sum(v["gradient_forward_input_tokens"] for v in qual_stages.values()) == sum(
            v["forward_input_tokens"] for v in qual_stages.values())
        result = {"schema": "agnews-fair-resume-qualified-v6", "passed": True,
            "exact_model_controller_policy_RNG": True, "arms": outputs,
            "source_sha256": current["qualifier_source_sha256"],
            "continuation_helper_sha256": current["continuation_helper_sha256"],
            "config_sha256": hash_path(a.config), "plan_sha256": hash_path(a.out_dir / "plan.json"),
            "v5_start_checkpoint_sha256": hash_path(start_cp_path),
            "actual_primary_updates": attempts, "actual_student_forward_passes": attempts * 2,
            "actual_student_backward_passes": attempts * 2, "actual_teacher_forward_passes": 0,
            "actual_auxiliary_forward_passes":attempts,"actual_auxiliary_backward_passes":attempts,
            "actual_primary_target_examples": attempts * 4,
            "actual_auxiliary_target_examples": attempts * 4,
            "actual_primary_prefix_draws":attempts*4,"actual_auxiliary_prefix_draws":attempts*4,
            "actual_primary_answer_token_targets":attempts*8,
            "actual_auxiliary_answer_token_targets":attempts*8,
            "scoring": False, "confirmation_scored": False, "confirmation_input": False,
            "cpu_preflight_wall_seconds":preflight_wall_seconds,
            "work": meter.counts, "seconds": time.monotonic() - start_time,
            "completed_utc": datetime.now(timezone.utc).isoformat()}
        assert attempts == 12
        c.write(a.out_dir / "qualification.json", result)
    except BaseException as exc:
        c.write(a.out_dir / "failure.json", {"schema":"agnews-fair-resume-failure-v6",
            "error_type":type(exc).__name__,"error":str(exc),"attempted_primary_updates":attempts,
            "work":meter.counts,"scoring":False,"confirmation_scored":False})
        raise
    finally:
        c.engine.ACTIVE_METER.reset(token)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("records", "warm-dir", "v5-run-dir", "model-dir", "config", "out-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
