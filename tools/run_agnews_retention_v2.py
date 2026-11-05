"""AG News train-role main v2 with opt-in controlled Sleep receiver mixing.

This file deliberately has no official-test path, evaluator, or auto-retry path.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = (ROOT / "src").resolve()
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(TOOLS))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import run_x870e_synthetic_dev_v2 as shared
import agnews_records_v1 as records_source
from agnews_records_v1 import (
    collate_extra_replay_batch,
    collate_scheduled_batch,
    load_agnews_records,
    sleep_prefix_records,
)
from nested_sleep.accounting import ACTIVE_METER, WorkMeter
from nested_sleep.lifecycle import SleepSession, fingerprint
from nested_sleep.lm import (
    LlamaCMSLM,
    execution_state,
    isolated_clone,
    load_execution_state,
    promote_mutable_to_fp32,
)
from nested_sleep.optim import CMSController

SEEDS = (17, 29, 43)
ARMS = ("frozen", "sft", "cms", "cms_sleep", "cms_extra_replay")
LR_PAIR = (0.005, 0.01)
FOUNDATION_UPDATES = 512
PRIMARY_UPDATES = 256
SLEEP_BOUNDARIES = (64, 128, 192, 256)
EXTRA_REPLAY_BATCHES_PER_EVENT = 12
SLEEP_GKD_STEPS = 16
SLEEP_LTI_STEPS = 8
SLEEP_DREAM_TOKENS = 2
SLEEP_RECEIVER_MIX_FRACTION = 0.5
HARD_SECONDS = 3600
DATA_SHA256 = "7b5169daadcf7d4ccaefc9d0bb79f1a4eef92fc275fae0c3a7d593e06b7e9e59"
SHARED_HELPER_SHA256 = "6c7bf0a079a052eb87458010c1b55e52db1d9daa6dce99e55708b2da98385948"
RECORD_HELPER_SHA256 = "aFA35BA68366BD268B7C1A9EC4C1460FCF8F64DA942D6CDBA38FC4ECD8B9EE2E".lower()
MODEL_REVISION = "48d84dd14886f01964f72e7f742a404dd86f9fb6"
WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors": "584d8d3e3f82f7964955174dfe5e3b1cf117a9d859f022cfdf7fcb884856e002",
    "model-00002-of-00002.safetensors": "4719a04514ec2f060240711b7c33ab21187cac730ecaba3040b7a0fd95a9cefb",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_atomic(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def capture_rng() -> dict:
    return {"torch": torch.get_rng_state().clone(), "cuda": [x.clone() for x in torch.cuda.get_rng_state_all()],
            "python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_sources() -> dict:
    import nested_sleep.accounting, nested_sleep.experts, nested_sleep.lifecycle
    import nested_sleep.lm, nested_sleep.memory, nested_sleep.native, nested_sleep.optim
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        if name == "nested_sleep" or name.startswith("nested_sleep."):
            module_file = getattr(module, "__file__", None)
            if module_file:
                resolved = Path(module_file).resolve()
                if not resolved.is_relative_to(SRC):
                    raise RuntimeError(f"external nested_sleep module refused: {name}={resolved}")
                loaded[name] = {"path": str(resolved), "sha256": sha256(resolved)}
    required = {"nested_sleep.accounting", "nested_sleep.experts", "nested_sleep.lifecycle",
                "nested_sleep.lm", "nested_sleep.memory", "nested_sleep.native", "nested_sleep.optim"}
    if required - set(loaded):
        raise RuntimeError(f"source binding incomplete: {sorted(required-set(loaded))}")
    return loaded


SOURCE_HASHES = ensure_sources()


class HardClock:
    def __init__(self, deadline: float):
        self.deadline = deadline

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("fixed 3600-second main cap reached; no automatic retry")


def validate_config(config: dict) -> dict:
    """Reject config edits that are not represented by this reviewed runner."""
    if not isinstance(config, dict) or config.get("schema") != "agnews-retention-main-config-v2":
        raise ValueError("unexpected main v2 config schema")
    expected_data = {"schema": "nl-sleep-agnews-train-roles-v1", "train_roles_json_sha256": DATA_SHA256,
                     "records_helper_sha256": RECORD_HELPER_SHA256,
                     "official_test_in_runner": False, "old_class_ids": [0, 1], "new_class_ids": [2, 3]}
    for key, expected in expected_data.items():
        if config.get("data", {}).get(key) != expected:
            raise ValueError(f"main config data.{key} differs from frozen protocol")
    expected_model = {"revision": MODEL_REVISION, "rank": 16, "capacity_per_bank": 8, "levels": 2,
                      "backbone_dtype": "BF16", "mutable_dtype": "FP32"}
    for key, expected in expected_model.items():
        if config.get("model", {}).get(key) != expected:
            raise ValueError(f"main config model.{key} differs from frozen protocol")
    expected_training = {
        "seeds": list(SEEDS), "arms": list(ARMS), "foundation_updates_per_seed": FOUNDATION_UPDATES,
        "primary_new_updates_per_adaptive_arm": PRIMARY_UPDATES, "base_lr": LR_PAIR[0], "expert_lr": LR_PAIR[1],
        "foundation_cms_periods": [4, 16], "sft_periods": [1, 1], "cms_periods": [4, 16],
        "sleep_boundaries_by_primary_step": list(SLEEP_BOUNDARIES),
        "sleep_gkd_updates_per_event": SLEEP_GKD_STEPS, "sleep_lti_updates_per_event": SLEEP_LTI_STEPS,
        "sleep_dream_updates_per_event": 0, "sleep_dream_tokens": SLEEP_DREAM_TOKENS,
        "sleep_receiver_mix_fraction": SLEEP_RECEIVER_MIX_FRACTION,
        "extra_replay_batches_per_event": EXTRA_REPLAY_BATCHES_PER_EVENT,
        "hard_wall_seconds": HARD_SECONDS, "automatic_retry": False,
        "total_planned_wake_updates": 4752, "total_planned_sleep_adam_updates": 288,
    }
    for key, expected in expected_training.items():
        if config.get("training", {}).get(key) != expected:
            raise ValueError(f"main config training.{key} differs from frozen protocol")
    if config.get("outputs", {}).get("seal_contains_test_scores") is not False:
        raise ValueError("main config may not enable test scoring in the training driver")
    return {"schema": config["schema"], "validated_fields": sorted(expected_data.keys() | expected_model.keys() | expected_training.keys())}


def build_checkpoint(model, controller, rng, *, seed, arm, base_step, records_sha, config_sha,
                     source_sha, schedule_sha, primary_cursor, replay_events):
    return shared.compact_cpu({
        "model": model.adapter_state_dict(),
        "execution_state": execution_state(model),
        "controller": None if controller is None else controller.state_dict(),
        "rng": rng,
        "seed": seed, "arm": arm, "base_step": base_step,
        "records_sha256": records_sha, "config_sha256": config_sha,
        "source_sha256": source_sha, "schedule_sha256": schedule_sha,
        "schedule_cursor": {"primary_new_step": primary_cursor, "extra_replay_events": replay_events},
        "execution_state_metadata": {"complete": True, "official_test_opened": False},
    })


def save_checkpoint(out_dir: Path, model, controller, seed, arm, base_step, records_sha,
                    config_sha, source_sha, schedule_sha, *, primary_cursor, replay_events):
    name = f"seed-{seed}-arm-{arm}-step-{base_step:03d}.pt"
    path = out_dir / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    state = build_checkpoint(model, controller, capture_rng(), seed=seed, arm=arm, base_step=base_step,
        records_sha=records_sha, config_sha=config_sha, source_sha=source_sha,
        schedule_sha=schedule_sha, primary_cursor=primary_cursor, replay_events=replay_events)
    torch.save(state, path)
    return {"base_step": base_step, "path": str(path.relative_to(out_dir)), "sha256": sha256(path),
            "adapter_fingerprint": fingerprint(model), "primary_cursor": primary_cursor,
            "replay_events": replay_events,
            "active_parameters": shared.active_parameter_counts(model),
            "controller_counts": None if controller is None else list(controller.counts)}


def make_labeled_rows(doc: dict, seed: int) -> list[dict]:
    mapping = doc["code_maps"][str(seed)]
    return [{"tokens": row["tokens"], "class_id": row["class_id"],
             "marker": mapping["marker_token_id"], "label": mapping["class_to_token_ids"][row["class_id"]]}
            for row in doc["records"]]


def supervised_update(model, controller, batch, pad_id, meter, stage) -> dict:
    controller.prepare_wake(model)
    meter.stage = stage
    ids, mask, lengths, markers, labels = batch
    logits = model.logits(ids, mask)
    row_idx = torch.arange(len(labels), device=ids.device)
    marker_logits = logits[row_idx, lengths - 1].float()
    class_logits = logits[row_idx, lengths].float()
    loss = 0.5 * (F.cross_entropy(marker_logits, markers) + F.cross_entropy(class_logits, labels))
    if not torch.isfinite(loss):
        raise RuntimeError(f"nonfinite supervised loss in {stage}")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
    proposals = controller.collect(model)
    for level in sorted(proposals, reverse=True):
        controller.apply(model, level, proposals[level])
    return {"loss": float(loss.detach()), "grad_norm": grad_norm,
            "due_levels": sorted(proposals), "nonpad_input_tokens": int(mask.sum())}


def sleep_event(model, controller, proposals, pool, event_index, seed, meter, clock, marker_id):
    if controller.counts != [4, 16] or set(proposals) != {0, 1}:
        raise RuntimeError(f"Sleep boundary must defer simultaneous due [0,1]; got counts={controller.counts}")
    original_model = fingerprint(model)
    original_controller = shared.tensor_tree_hash(controller.state_dict())
    candidate = isolated_clone(model, share_backbone=True)
    candidate_controller = CMSController.restore(candidate, controller.state_dict())
    candidate_controller.apply(candidate, 1, proposals[1])
    if candidate_controller.counts != [4, 0]:
        raise RuntimeError("candidate slow proposal did not consume level 1 first")
    session = SleepSession(candidate, 0, proposals[0], seed=seed * 10000 + event_index,
        lr=0.003, dream_tokens=SLEEP_DREAM_TOKENS, share_backbone=True, reject_zero_gradient=True,
        receiver_mix_fraction=SLEEP_RECEIVER_MIX_FRACTION)
    if fingerprint(session.teacher) != fingerprint(candidate):
        raise RuntimeError("Sleep teacher was not constructed from the slow-updated candidate")
    receiver_bank = session.student.banks()[1]
    receiver_bank.audit_routes = True
    if receiver_bank.route_override != {"mix": session.slot, "fraction": SLEEP_RECEIVER_MIX_FRACTION}:
        raise RuntimeError("Sleep student receiver mix policy was not installed")
    device = next(model.parameters()).device
    gkd_marker_hits = 0
    lti_marker_hits = 0
    natural_gkd_hits = natural_gkd_positions = 0
    natural_lti_hits = natural_lti_positions = 0
    positive = []
    def count_natural_loss_routes(session, prefix, kind):
        audit = receiver_bank.last_route_audit
        if not isinstance(audit, dict) or not torch.is_tensor(audit.get("selected")):
            raise RuntimeError("natural top-1 route audit missing from existing Sleep student logits")
        if audit.get("mixed_slot") != session.slot or audit.get("mixed_fraction") != SLEEP_RECEIVER_MIX_FRACTION:
            raise RuntimeError("actual student loss forward did not execute the fixed receiver mixture")
        sequence = session.loss_context["sequence"]
        prefix_length = int(session.loss_context["prefix_length"])
        target_tokens = int(sequence.shape[1]) - prefix_length
        if target_tokens <= 0:
            raise RuntimeError("Sleep loss sequence has no scored target positions")
        start = prefix_length - 1
        selected = audit["selected"][:, start:start + target_tokens]
        if selected.numel() != target_tokens:
            raise RuntimeError("natural route audit does not cover every answer loss position")
        hits = int((selected == session.slot).sum().item())
        return hits, selected.numel()
    meter.stage = f"seed{seed}_cms_sleep_event{event_index+1}"
    for row in pool[:SLEEP_GKD_STEPS]:
        clock.check()
        prefix = torch.tensor([row["tokens"]], dtype=torch.long, device=device)
        session.seed_step(prefix, owner=None)
        positive.append(float(session.records[-1]["grad_norm"]) > 0.0)
        gkd_marker_hits += int(session.loss_context["sequence"][0, prefix.shape[1]].item() == marker_id)
        hits, count = count_natural_loss_routes(session, prefix, "gkd")
        natural_gkd_hits += hits
        natural_gkd_positions += count
    teacher_owners = [r["kind"] for r in session.records]
    for row in pool[SLEEP_GKD_STEPS:SLEEP_GKD_STEPS+SLEEP_LTI_STEPS]:
        clock.check()
        prefix = torch.tensor([row["tokens"]], dtype=torch.long, device=device)
        session.lti_step(prefix)
        positive.append(float(session.records[-1]["grad_norm"]) > 0.0)
        lti_marker_hits += int(session.loss_context["sequence"][0, prefix.shape[1]].item() == marker_id)
        hits, count = count_natural_loss_routes(session, prefix, "lti")
        natural_lti_hits += hits
        natural_lti_positions += count
    if len(positive) != 24 or not all(positive):
        raise RuntimeError("Sleep receiver gradient gate failed")
    if sum(k == "gkd_teacher" for k in teacher_owners) != 8 or sum(k == "gkd_student_on_policy" for k in teacher_owners) != 8:
        raise RuntimeError("owner=None GKD did not alternate to eight teacher/eight student updates")
    session.finish_consolidation()
    committed = session.commit()  # finite/scope/frozen/immutable guards are enforced by SleepSession.
    candidate_controller.consume(0)
    if candidate_controller.counts != [0, 0]:
        raise RuntimeError("Sleep commit did not consume the fast prospective proposal")
    if fingerprint(model) != original_model or shared.tensor_tree_hash(controller.state_dict()) != original_controller:
        raise RuntimeError("deferred Sleep candidate mutated original live model/controller")
    summary = {"event": event_index + 1, "gkd_updates": SLEEP_GKD_STEPS, "lti_updates": SLEEP_LTI_STEPS,
        "dream_updates": 0, "dream_tokens": SLEEP_DREAM_TOKENS,
        "slow_level1_applied_to_candidate_before_teacher": True,
        "fast_level0_passed_prospectively_without_preapply": True,
        "candidate_committed": True, "receiver_gradients_all_positive": True,
        "gkd_teacher_updates": 8, "gkd_student_on_policy_updates": 8,
        "gkd_marker_hits": gkd_marker_hits, "gkd_marker_total": 16,
        "lti_teacher_marker_hits": lti_marker_hits, "lti_teacher_marker_total": 8,
        "receiver_mix_fraction": SLEEP_RECEIVER_MIX_FRACTION,
        "receiver_gradient_norm_min": min(r["grad_norm"] for r in session.records),
        "receiver_gradient_norm_max": max(r["grad_norm"] for r in session.records),
        "receiver_gradient_norm_mean": sum(r["grad_norm"] for r in session.records) / len(session.records),
        "executed_mixed_receiver_loss_position_coverage": 1.0,
        "natural_receiver_gkd_loss_position_hits": natural_gkd_hits,
        "natural_receiver_gkd_loss_positions": natural_gkd_positions,
        "natural_receiver_lti_loss_position_hits": natural_lti_hits,
        "natural_receiver_lti_loss_positions": natural_lti_positions,
        "natural_receiver_loss_position_hit_rate": ((natural_gkd_hits + natural_lti_hits) /
            max(1, natural_gkd_positions + natural_lti_positions)),
        "semantic_judge": "exact_output_proxy_NOT_paper_semantic_RM",
        "postcommit_active_slots_by_bank": shared.active_parameter_counts(committed)["active_slots_by_bank"]}
    del session
    return committed, candidate_controller, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True,
                        help="frozen train-derived AG News roles JSON; no official-test input is accepted")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "agnews-retention-v2.json")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"output directory must be new: {args.out_dir}")
    if sha256(args.records) != DATA_SHA256:
        raise RuntimeError("frozen train-role records SHA mismatch")
    doc, record_summary = load_agnews_records(args.records)
    config_doc = json.loads(args.config.read_text(encoding="utf-8"))
    config_validation = validate_config(config_doc)
    if doc.get("test_opened") is not False:
        raise RuntimeError("training record pack does not certify test_opened=false")
    if doc["phase_class_ids"] != {"old": [0, 1], "new": [2, 3]}:
        raise RuntimeError("frozen class phase mapping mismatch")
    config_sha = sha256(args.config)
    source_sha = sha256(Path(__file__).resolve())
    if sha256(Path(shared.__file__).resolve()) != SHARED_HELPER_SHA256:
        raise RuntimeError("shared helper source SHA differs from the reviewed DEV fixture source")
    record_helper_sha = sha256(Path(records_source.__file__).resolve())
    if record_helper_sha != RECORD_HELPER_SHA256:
        raise RuntimeError("AG News frozen records helper source SHA mismatch")
    if config_doc["data"]["records_helper_sha256"] != record_helper_sha:
        raise RuntimeError("config does not pin the loaded AG News record helper bytes")
    if SOURCE_HASHES != ensure_sources():
        raise RuntimeError("nested_sleep source changed during preflight")
    if "receiver_mix_fraction" not in inspect.signature(SleepSession.__init__).parameters:
        raise RuntimeError("loaded SleepSession lacks the reviewed receiver_mix_fraction API")
    for filename, expected in WEIGHT_SHA256.items():
        if sha256(args.model_dir / filename) != expected:
            raise RuntimeError(f"pinned local model weight SHA mismatch: {filename}")
    for entry in doc["tokenizer"]["file_inventory"]:
        if sha256(args.model_dir / entry["name"]) != entry["sha256"]:
            raise RuntimeError(f"tokenizer asset SHA mismatch: {entry['name']}")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    clock = HardClock(started + HARD_SECONDS)
    protocol = {"schema": "agnews-retention-main-v2", "status": "training_in_progress",
        "seeds": list(SEEDS), "arms": list(ARMS), "lr_pair": list(LR_PAIR),
        "foundation_updates_per_seed": FOUNDATION_UPDATES, "primary_updates_per_adaptive_arm": PRIMARY_UPDATES,
        "sleep_boundaries": list(SLEEP_BOUNDARIES), "extra_replay_batches_per_event": EXTRA_REPLAY_BATCHES_PER_EVENT,
        "sleep_gkd_updates_per_event": SLEEP_GKD_STEPS, "sleep_lti_updates_per_event": SLEEP_LTI_STEPS,
        "sleep_dream_updates_per_event": 0, "sleep_dream_tokens": SLEEP_DREAM_TOKENS,
        "sleep_receiver_mix_fraction": SLEEP_RECEIVER_MIX_FRACTION,
        "sleep_natural_route_audit": "natural top-1 receiver hits at answer-loss positions from existing student logits; no extra forward",
        "cms_periods": [4, 16], "rank": 16, "capacity": 8, "hard_seconds": HARD_SECONDS,
        "records_sha256": DATA_SHA256, "records_summary": record_summary,
        "schedule_hashes_sha256_by_seed": record_summary["schedules_sha256_by_seed"],
        "config_sha256": config_sha, "runner_sha256": source_sha,
        "shared_helper_sha256": SHARED_HELPER_SHA256, "nested_source_hashes": SOURCE_HASHES,
        "records_helper_sha256": record_helper_sha, "config_validation": config_validation,
        "model_source": "ModelScope LLM-Research/Llama-3.2-3B",
        "model_revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256,
        "backbone_dtype": "BF16", "mutable_dtype": "FP32", "attention_backend": "eager",
        "controller_policy": "fresh empty clocks for every arm; foundation weights shared within seed",
        "primary_schedule": "frozen JSON new_primary256 indices; same per seed across arms; replay/Sleep RNG does not shift it",
        "extra_replay_interpretation": "near-compute/prompt-exposure proxy; not exact FLOPs or isolated Sleep mechanism",
        "total_planned_wake_updates": 4752,
        "total_planned_sleep_adam_updates": 288,
        "official_test_opened": False, "automatic_retry": False,
        "started_utc": started_utc, "sealed": False}
    write_new(args.out_dir / "training-protocol.json", protocol)
    progress_path = args.out_dir / "training-progress.json"
    progress = {"status": "in_progress", "foundations": {}, "runs": {}, "official_test_opened": False}
    write_new(progress_path, progress)
    print(json.dumps({"status": "started", "pid": os.getpid(), "output_dir": str(args.out_dir),
                      "hard_seconds": HARD_SECONDS, "runner_sha256": source_sha,
                      "config_sha256": config_sha}), flush=True)

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        write_new(args.out_dir / "failure.json", {"status": "FAILED_UNSEALED",
            "error_type": "RuntimeError", "error": "CUDA device unavailable; no training started",
            "completed_foundations": 0, "completed_runs": 0, "automatic_retry": False,
            "official_test_opened": False})
        raise RuntimeError("CUDA device unavailable; no training started")
    props = torch.cuda.get_device_properties(0)
    runtime = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_name": props.name, "gpu_total_memory_bytes": props.total_memory,
        "nested_source_hashes": SOURCE_HASHES, "weights_sha256": WEIGHT_SHA256,
        "tokenizer_sha256": record_summary["tokenizer_files_sha256"],
        "work_meter_fields": ["forward_input_tokens", "gradient_forward_input_tokens", "forward_calls",
                              "gradient_forward_calls"], "test_opened": False}
    write_new(args.out_dir / "runtime.json", runtime)
    rows_by_seed = {seed: make_labeled_rows(doc, seed) for seed in SEEDS}
    pad_id = int(doc["tokenizer"]["pad_token_id"])
    meter = WorkMeter()
    meter_token = ACTIVE_METER.set(meter)
    completed_foundations = 0
    completed_runs = 0
    current_context = {"seed": None, "arm": None, "primary_cursor": 0, "foundation_cursor": 0}
    try:
        for seed in SEEDS:
            clock.check()
            current_context.update(seed=seed, arm="foundation", foundation_cursor=0, primary_cursor=0)
            seed_all(seed)
            backbone = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                trust_remote_code=False, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
            model = LlamaCMSLM(backbone, rank=16, capacity=8, levels=2)
            promote_mutable_to_fp32(model)
            model.eval().requires_grad_(False)
            foundation_controller = CMSController(model, periods=(4, 16), base_lr=LR_PAIR[0], expert_lr=LR_PAIR[1])
            controller = foundation_controller
            seed_rows = rows_by_seed[seed]
            foundation_start = time.perf_counter()
            for step_index in range(FOUNDATION_UPDATES):
                clock.check()
                current_context["foundation_cursor"] = step_index
                batch = collate_scheduled_batch(doc, seed, "foundation512", step_index, "cuda", pad_id)
                supervised_update(model, foundation_controller, batch, pad_id, meter, f"seed{seed}_foundation")
                if (step_index + 1) % 128 == 0:
                    print(json.dumps({"seed": seed, "stage": "foundation", "step": step_index + 1}), flush=True)
            if foundation_controller.counts != [0, 0]:
                raise RuntimeError(f"foundation controller has unconsumed proposals: {foundation_controller.counts}")
            foundation_rng = capture_rng()
            foundation_model = model.adapter_state_dict()
            foundation_fp = fingerprint(model)
            foundation_path = args.out_dir / "checkpoints" / f"seed-{seed}-foundation.pt"
            foundation_path.parent.mkdir(parents=True, exist_ok=True)
            foundation_state = shared.compact_cpu({"model": foundation_model,
                "execution_state": execution_state(model), "controller": foundation_controller.state_dict(),
                "rng": foundation_rng, "seed": seed, "arm": "foundation", "base_step": FOUNDATION_UPDATES,
                "records_sha256": DATA_SHA256, "config_sha256": config_sha, "source_sha256": source_sha,
                "schedule_sha256": record_summary["schedules_sha256_by_seed"][str(seed)],
                "schedule_cursor": {"foundation_step": FOUNDATION_UPDATES},
                "execution_state_metadata": {"complete": True, "official_test_opened": False}})
            torch.save(foundation_state, foundation_path)
            progress["foundations"][str(seed)] = {"status": "complete", "updates": FOUNDATION_UPDATES,
                "seconds": time.perf_counter() - foundation_start, "checkpoint": str(foundation_path.relative_to(args.out_dir)),
                "checkpoint_sha256": sha256(foundation_path), "adapter_fingerprint": foundation_fp}
            write_atomic(progress_path, progress)

            for arm in ARMS:
                clock.check()
                current_context.update(seed=seed, arm=arm, foundation_cursor=FOUNDATION_UPDATES, primary_cursor=0)
                model.load_state_dict(foundation_model, strict=False)
                load_execution_state(model, foundation_state["execution_state"])
                model.eval().requires_grad_(False)
                restore_rng(foundation_rng)
                if fingerprint(model) != foundation_fp:
                    raise RuntimeError(f"arm {arm} did not restore the common foundation model")
                if arm == "frozen":
                    controller = None
                else:
                    periods = (1, 1) if arm == "sft" else (4, 16)
                    controller = CMSController(model, periods=periods, base_lr=LR_PAIR[0], expert_lr=LR_PAIR[1])
                run_key = f"{seed}/{arm}"
                schedule_sha = record_summary["schedules_sha256_by_seed"][str(seed)]
                run_info = {"status": "in_progress", "checkpoints": [], "primary_updates": 0,
                            "extra_replay_updates": 0, "sleep_adam_updates": 0,
                            "sleep_events": [], "seconds": 0.0}
                progress["runs"][run_key] = run_info
                write_atomic(progress_path, progress)
                run_started = time.perf_counter()
                run_info["checkpoints"].append(save_checkpoint(args.out_dir, model, controller, seed, arm, 0,
                    DATA_SHA256, config_sha, source_sha, schedule_sha, primary_cursor=0, replay_events=0))
                replay_events_done = 0
                if arm != "frozen":
                    for primary_step in range(1, PRIMARY_UPDATES + 1):
                        clock.check()
                        current_context["primary_cursor"] = primary_step - 1
                        batch = collate_scheduled_batch(doc, seed, "new_primary256", primary_step - 1, "cuda", pad_id)
                        defer = arm == "cms_sleep" and primary_step in SLEEP_BOUNDARIES
                        ids, mask, lengths, markers, labels = batch
                        controller.prepare_wake(model)
                        meter.stage = f"seed{seed}_{arm}_primary"
                        logits = model.logits(ids, mask)
                        row_idx = torch.arange(len(labels), device=ids.device)
                        marker_logits = logits[row_idx, lengths - 1].float()
                        class_logits = logits[row_idx, lengths].float()
                        loss = 0.5 * (F.cross_entropy(marker_logits, markers) + F.cross_entropy(class_logits, labels))
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"nonfinite loss at {run_key} primary {primary_step}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                        proposals = controller.collect(model)
                        if defer:
                            if sorted(proposals) != [0, 1] or controller.counts != [4, 16]:
                                raise RuntimeError("Sleep boundary did not defer both due proposals")
                        else:
                            for level in sorted(proposals, reverse=True):
                                controller.apply(model, level, proposals[level])
                        run_info["primary_updates"] = primary_step
                        if primary_step in (64, 128, 192, 256):
                            if arm == "cms_sleep":
                                pool = sleep_prefix_records(doc, seed, SLEEP_BOUNDARIES.index(primary_step))
                                mapping = doc["code_maps"][str(seed)]
                                marker_id = int(mapping["marker_token_id"])
                                model, controller, summary = sleep_event(model, controller, proposals, pool,
                                    SLEEP_BOUNDARIES.index(primary_step), seed, meter, clock, marker_id)
                                run_info["sleep_events"].append(summary)
                                run_info["sleep_adam_updates"] += SLEEP_GKD_STEPS + SLEEP_LTI_STEPS
                            elif arm == "cms_extra_replay":
                                event = SLEEP_BOUNDARIES.index(primary_step)
                                for replay_batch in range(EXTRA_REPLAY_BATCHES_PER_EVENT):
                                    clock.check()
                                    replay = collate_extra_replay_batch(doc, seed, event, replay_batch, "cuda", pad_id)
                                    supervised_update(model, controller, replay, pad_id, meter,
                                        f"seed{seed}_cms_extra_replay_event{event+1}")
                                    run_info["extra_replay_updates"] += 1
                                replay_events_done += 1
                            run_info["checkpoints"].append(save_checkpoint(args.out_dir, model, controller,
                                seed, arm, primary_step, DATA_SHA256, config_sha, source_sha,
                                schedule_sha,
                                primary_cursor=primary_step, replay_events=replay_events_done))
                            write_atomic(progress_path, progress)
                        if primary_step % 64 == 0:
                            print(json.dumps({"seed": seed, "arm": arm, "primary_step": primary_step,
                                              "extra_replay_updates": run_info["extra_replay_updates"]}), flush=True)
                if arm == "frozen":
                    run_info["primary_updates"] = 0
                    run_info["checkpoints"].append(save_checkpoint(args.out_dir, model, None, seed, arm, 256,
                        DATA_SHA256, config_sha, source_sha, schedule_sha, primary_cursor=0, replay_events=0))
                if arm != "frozen" and controller.counts != [0, 0]:
                    raise RuntimeError(f"run ended with unresolved CMS clocks {run_key}: {controller.counts}")
                run_info["seconds"] = time.perf_counter() - run_started
                run_info["work"] = {k: deepcopy(v) for k, v in meter.counts.items() if k.startswith(f"seed{seed}_{arm}")}
                run_info["active_parameters"] = shared.active_parameter_counts(model)
                run_info["final_adapter_fingerprint"] = fingerprint(model)
                run_info["status"] = "complete"
                current_context["primary_cursor"] = 256
                completed_runs += 1
                write_atomic(progress_path, progress)
                print(json.dumps({"seed": seed, "arm": arm, "status": "complete",
                                  "elapsed_seconds": run_info["seconds"]}), flush=True)
            completed_foundations += 1
            del foundation_controller, foundation_state, model, backbone
            torch.cuda.empty_cache()
            progress["completed_foundations"] = completed_foundations
            progress["completed_runs"] = completed_runs
            progress["elapsed_seconds"] = time.monotonic() - started
            write_atomic(progress_path, progress)

        if completed_foundations != 3 or completed_runs != 15:
            raise RuntimeError(f"refusing to seal incomplete run: foundations={completed_foundations} arms={completed_runs}")
        if time.monotonic() >= clock.deadline:
            raise TimeoutError("main hard cap reached before sealing; results remain unsealed")
        protocol["status"] = "training_sealed"
        protocol["sealed"] = True
        protocol["sealed_utc"] = datetime.now(timezone.utc).isoformat()
        protocol["elapsed_seconds"] = time.monotonic() - started
        protocol["completed_foundations"] = completed_foundations
        protocol["completed_runs"] = completed_runs
        protocol["work_total"] = deepcopy(meter.counts)
        protocol["official_test_opened"] = False
        write_atomic(args.out_dir / "training-protocol.json", protocol)
        run_rows = [{"seed": int(key.split("/", 1)[0]), "arm": key.split("/", 1)[1], **value}
                    for key, value in sorted(progress["runs"].items())]
        foundation_rows = [{"seed": int(seed), **value}
                           for seed, value in sorted(progress["foundations"].items(), key=lambda item: int(item[0]))]
        seal = {"schema": "agnews-retention-training-seal-v2", "complete": True,
            "seeds": list(SEEDS), "arms": list(ARMS), "records_sha256": DATA_SHA256,
            "config_sha256": config_sha, "source_sha256": source_sha,
            "sleep_receiver_mix_fraction": SLEEP_RECEIVER_MIX_FRACTION,
            "records_helper_sha256": record_helper_sha, "nested_source_hashes": SOURCE_HASHES,
            "schedule_hashes_sha256_by_seed": record_summary["schedules_sha256_by_seed"],
            "protocol_sha256": sha256(args.out_dir / "training-protocol.json"),
            "foundation_checkpoints": foundation_rows, "runs": run_rows,
            "total_planned_wake_updates": protocol["total_planned_wake_updates"],
            "total_planned_sleep_adam_updates": protocol["total_planned_sleep_adam_updates"],
            "elapsed_seconds": protocol["elapsed_seconds"], "official_test_opened": False,
            "contains_evaluation_scores": False, "contains_raw_text_or_token_ids": False,
            "contains_model_weights": False}
        write_new(args.out_dir / "training-sealed.json", seal)
        progress["status"] = "training_sealed"
        progress["completed_foundations"] = completed_foundations
        progress["completed_runs"] = completed_runs
        write_atomic(progress_path, progress)
    except BaseException as exc:
        failure_state_capture_error = None
        if "model" in locals():
            try:
                failure_state = shared.compact_cpu({
                    "model": model.adapter_state_dict(), "execution_state": execution_state(model),
                    "controller": None if "controller" not in locals() or controller is None else controller.state_dict(),
                    "rng": capture_rng(), "seed": current_context["seed"], "arm": current_context["arm"],
                    "base_step": current_context["primary_cursor"],
                    "records_sha256": DATA_SHA256, "config_sha256": config_sha,
                    "source_sha256": source_sha, "failure_state_not_resumable": True,
                })
                torch.save(failure_state, args.out_dir / "failure-state.pt")
            except BaseException as state_exc:
                failure_state_capture_error = f"{type(state_exc).__name__}: {state_exc}"
        failure = {"status": "FAILED_UNSEALED", "error_type": type(exc).__name__, "error": str(exc),
            "completed_foundations": completed_foundations, "completed_runs": completed_runs,
            "elapsed_seconds": time.monotonic() - started, "work_total": deepcopy(meter.counts),
            "automatic_retry": False, "official_test_opened": False}
        if failure_state_capture_error:
            failure["failure_state_capture_error"] = failure_state_capture_error
        try:
            write_new(args.out_dir / "failure.json", failure)
        except FileExistsError:
            write_atomic(args.out_dir / "failure.json", failure)
        raise
    finally:
        ACTIVE_METER.reset(meter_token)


if __name__ == "__main__":
    main()
