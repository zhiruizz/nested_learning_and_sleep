"""Controlled Banking77 batched Sleep lifecycle experiment (development only).

This is a new V13 runner. It preserves the locked V10 foundation restore
contract while varying only the post-foundation wake batch and Sleep optimizer
microbatch/LR plan described by the separate V13 config.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "banking-v13-batched-lifecycle-config-v1"
V10_CONFIG_SHA256 = "785ded5cc9d205dc3cbc6d25179abf6fd0d589ce6943a42626aa1692b54dfd9d"
V10_SOURCE_LOCK_SHA256 = "3fcc2e2cb930149e9ee5204dcf8fa8910888cd7b05cd31f66d10ceb852d6844c"
TRAINING_MANIFEST_SHA256 = "ed9f4b047768021cbf16a3fa042db12baf212038f89f33a00099f2e32b7300f3"
SEEDS = (17, 29, 43)
ARMS = (
    {"name": "CMS_ONLY", "sleep": False, "effective_batch_size": None,
     "learning_rate": None, "requests_per_event": 0},
    {"name": "B1_HIGH", "sleep": True, "effective_batch_size": 1,
     "learning_rate": 0.001, "requests_per_event": 960},
    {"name": "B1_LOW", "sleep": True, "effective_batch_size": 1,
     "learning_rate": 0.000005, "requests_per_event": 960},
    {"name": "B32_HIGH", "sleep": True, "effective_batch_size": 32,
     "learning_rate": 0.001, "requests_per_event": 960},
    {"name": "B32_LOW", "sleep": True, "effective_batch_size": 32,
     "learning_rate": 0.000005, "requests_per_event": 960},
    {"name": "B32_LOW_DOSE", "sleep": True, "effective_batch_size": 32,
     "learning_rate": 0.000005, "requests_per_event": 3200},
)
EXECUTION_ORDER = ["B32_HIGH", "CMS_ONLY", "B1_HIGH", "B1_LOW", "B32_LOW", "B32_LOW_DOSE"]
EVALUATION_STAGES = ["event1_before", "event1_after", "event2_before", "event2_after", "postwake15"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(cfg: dict, *, require_frozen: bool) -> dict:
    """Reject recipe drift before loading model state or initializing CUDA."""
    if not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA:
        raise ValueError("V13 config schema mismatch")
    allowed_status = ("frozen",) if require_frozen else ("draft_not_for_gpu_launch", "frozen")
    if cfg.get("status") not in allowed_status:
        raise ValueError("V13 config status is not permitted for this phase")
    fixed = {
        "v10_config_file_sha256": V10_CONFIG_SHA256,
        "v10_source_lock_sha256": V10_SOURCE_LOCK_SHA256,
        "training_manifest_sha256": TRAINING_MANIFEST_SHA256,
        "seeds": list(SEEDS), "task_id": 1, "sender": 0,
        "wake_batch_size": 32,
        "wake_batch_policy": "full_32_requests_from_concatenated_deterministic_epochs_0_and_1_no_drop_or_partial",
        "inherited_foundation_wake_batch_size": 8,
        "periods": [16, 64, 256, 1024, None], "clock_unit": "wake_optimizer_minibatches_local_assumption_not_paper_token_units",
        "sleep_events_per_run": 2, "wake_ticks_between_sleep_events": 16,
        "wake_ticks_after_last_sleep": 15,
        "sleep_method_cycle": ["gkd", "gkd", "lti"],
        "gkd_sampling_owner_policy": "alternate_teacher_student_start_teacher_each_event",
        "sleep_selector_event_stride": 3200,
        "sleep_selector_policy": "v10_balanced_old_current_seen_class_cycle_first_prefix_of_common_3200_request_schedule_each_event",
        "temperature": 0.8, "max_new_tokens": 10, "max_sequence_length": 128,
        "weight_decay": 0.0, "clip_norm": 1.0,
        "objective": "v10_local_GKD_and_LTI_assembly_unchanged_request_mean_gradient_accumulation",
        "arms": list(ARMS), "execution_order_per_seed": EXECUTION_ORDER,
        "test_data_read": False, "no_test_based_selection": True,
        "automatic_retries": 0,
    }
    for key, value in fixed.items():
        if cfg.get(key) != value:
            raise ValueError(f"V13 locked field mismatch: {key}")
    due = cfg.get("prefix_ticks_by_seed")
    expected_prefix = {"17": 4, "29": 10, "43": 14}
    if due != expected_prefix:
        raise ValueError("V13 first-due prefix ticks differ from the qualified foundations")
    total = cfg.get("expected_total_wake_ticks_by_seed")
    if total != {"17": 35, "29": 41, "43": 45}:
        raise ValueError("V13 wake horizon differs from the two-event design")
    ev = cfg.get("evaluation")
    if ev != {"split": "development_only", "old_rows": 110, "current_rows": 110,
              "candidate_labels": 77, "candidate_batch_size": 32,
              "primary": "teacher_forced_sum", "include_greedy": True,
              "stages": EVALUATION_STAGES}:
        raise ValueError("V13 evaluation design mismatch")
    objective = cfg.get("objective_numerics")
    if objective != {"KL_temperature": 1.0, "LTI_KL_weight": 0.5,
                     "LTI_reward_CE_weight": 0.5, "semantic_proxy_weight": 0.5,
                     "edit_weight": 0.5, "edit_distance_z0": 8}:
        raise ValueError("V13 objective numerics differ from the inherited local recipe")
    for key in ("max_wall_seconds", "max_forward_calls", "max_update_steps"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"V13 budget {key} must be a positive integer")
    source_hashes = cfg.get("source_hashes")
    if require_frozen and (not isinstance(source_hashes, dict) or not source_hashes):
        raise ValueError("frozen V13 config must pin all new runner/helper sources")
    return cfg


def predicted_budget_per_seed(cfg: dict) -> dict:
    """Exact prompt request/update budget, excluding wake/evaluation work."""
    validate_config(cfg, require_frozen=False)
    req = updates = 0
    per_arm = {}
    for arm in cfg["arms"]:
        n = int(arm["requests_per_event"]) * cfg["sleep_events_per_run"]
        b = arm["effective_batch_size"]
        if n and (type(b) is not int or n % b):
            raise ValueError("Sleep arm exposure must divide evenly into full microbatches")
        u = n // b if n else 0
        req += n
        updates += u
        per_arm[arm["name"]] = {"requests": n, "committed_updates": u}
    return {"per_arm": per_arm, "total_requests": req, "total_updates": updates}


def validate_clock_horizon(clocks: dict[int, int], seed: int, periods: list,
                           prefix: int, between: int, after: int) -> dict:
    """Verify actual inherited clocks produce exactly two level-0 dues."""
    if seed not in SEEDS or clocks.get(0) is None or periods[0] != 16:
        raise ValueError("invalid actual clock state")
    first = 16 - int(clocks[0])
    second = first + between
    end = second + after
    if first != prefix:
        raise ValueError("first due differs from locked prefix")
    other_due = []
    level0 = int(clocks[0])
    observed_level0 = []
    for offset in range(1, end + 1):
        level0 += 1
        if level0 == 16:
            observed_level0.append(offset)
            level0 = 0
    if observed_level0 != [first, second]:
        raise ValueError("level-0 clock yields unexpected due events within horizon")
    for level, clock in clocks.items():
        period = periods[level]
        if period is None:
            continue
        for offset in range(1, end + 1):
            if level != 0 and (int(clock) + offset) % int(period) == 0:
                other_due.append({"level": level, "offset": offset})
    if other_due:
        raise ValueError("another memory level becomes due within the frozen horizon")
    return {"first_due": first, "second_due": second, "endpoint_offset": end,
            "other_level_due": other_due}


def _preflight(args) -> dict:
    """CPU-only source/model/pack/foundation lock checks; never loads a model."""
    import run_banking_cil_v10 as v10
    config_path = Path(args.config).resolve(strict=True)
    v10_config_path = Path(args.v10_config).resolve(strict=True)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(cfg, require_frozen=True)
    if args.seed not in SEEDS or args.arm not in {x['name'] for x in ARMS}:
        raise ValueError('seed or arm is outside the frozen grid')
    task_root=Path(__file__).resolve().parents[1]
    required={
        'tools/run_banking_sleep_lifecycle_v13.py','tools/banking_v13_execution.py',
        'tools/launch_banking_sleep_lifecycle_v13.py','tools/banking_cil_v13_batching.py',
        'tools/banking_sleep_v13_runtime.py','tools/banking_v13_restore_qualification.py',
        'tools/run_banking_v13_cpu_acceptance.py','tools/run_banking_optimizer_boundary_v12.py'}
    if not required.issubset(cfg['source_hashes']):
        raise ValueError('frozen source graph omits an execution dependency')
    new_hashes={}
    for relative,expected in cfg['source_hashes'].items():
        path=(task_root/relative).resolve()
        if not path.is_relative_to(task_root) or not path.is_file() or sha256_file(path)!=expected:
            raise ValueError('V13 source hash/path mismatch: '+relative)
        new_hashes[relative]=expected
    if sha256_file(v10_config_path) != V10_CONFIG_SHA256:
        raise ValueError("V10 config bytes differ from the qualified foundation recipe")
    vcfg = json.loads(v10_config_path.read_text(encoding="utf-8"))
    project = Path(args.project_root).resolve(strict=True)
    model = Path(args.model_path).resolve(strict=True)
    pack = Path(args.training_pack).resolve(strict=True)
    hashes, lock_sha, manifest, train, dev = v10._source_prepare(
        project, Path(__file__).resolve().parents[1], model, pack, vcfg)
    if lock_sha != V10_SOURCE_LOCK_SHA256:
        raise ValueError("V10 source-lock digest differs from V13's frozen foundation")
    foundation_root = (Path(args.foundation_root).resolve(strict=True)/f'seed{args.seed}').resolve(strict=True)
    release_path = Path(args.release_artifact).resolve(strict=True)
    manifest_sha = cfg["training_manifest_sha256"]
    release, receipt = v10._validate_foundation_release(release_path, vcfg,
        v10._config_sha(vcfg), lock_sha, manifest_sha, args.seed, foundation_root)
    cp = (foundation_root / receipt["checkpoint"]).resolve(strict=True)
    cp_result = v10._validate_foundation_checkpoint_cpu(cp, receipt["checkpoint_sha256"],
        seed=args.seed, cfg=vcfg, source_lock=lock_sha,
        manifest_sha=manifest_sha)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("V13 CPU foundation audit must not initialize CUDA")
    foundation_state = torch.load(cp, map_location="cpu", weights_only=False)
    wake_state = foundation_state["wake"]
    clocks = {int(k): int(v) for k, v in wake_state.clocks.items()}
    level0_clock = clocks.get(0)
    first_due = 16 - level0_clock if level0_clock is not None else -1
    clock_horizon = validate_clock_horizon(clocks, args.seed, cfg["periods"],
        cfg["prefix_ticks_by_seed"][str(args.seed)],
        cfg["wake_ticks_between_sleep_events"], cfg["wake_ticks_after_last_sleep"])
    tick1 = int(wake_state.ticks) + first_due
    tick2 = tick1 + int(cfg["wake_ticks_between_sleep_events"])
    tick_end = tick2 + int(cfg["wake_ticks_after_last_sleep"])
    if [tick1, tick2, tick_end] != {
            17: [352, 368, 383], 29: [336, 352, 367], 43: [336, 352, 367]}[args.seed]:
        raise ValueError("derived actual-clock horizon differs from the frozen lifecycle")
    foundation_record = release["foundations"][str(args.seed)]
    seed_receipt_path = Path(foundation_record["run_dir"]).resolve() / "foundation-receipt.json"
    return {"status": "cpu_preflight_pass", "seed": args.seed,
            "v13_config_sha256": sha256_file(config_path),
            "v10_config_sha256": sha256_file(v10_config_path),
            "source_lock_sha256": lock_sha, "source_hashes": hashes,
            "v13_source_hashes":new_hashes,
            "training_manifest_sha256": cfg["training_manifest_sha256"],
            "train_rows": len(train), "development_rows": len(dev),
            "class_count": len(manifest["classes"]),
            "main_release_artifact_sha256": sha256_file(release_path),
            "foundation_receipt_sha256": sha256_file(seed_receipt_path),
            "foundation_receipt_path": str(seed_receipt_path),
            "foundation_run_dir": str(seed_receipt_path.parent),
            "foundation_checkpoint_sha256": receipt["checkpoint_sha256"],
            "foundation_checkpoint_cpu_check": cp_result,
            "inherited_wake_ticks": int(wake_state.ticks), "inherited_clocks": clocks,
            "first_due_after_wake_ticks": first_due,
            "clock_horizon": clock_horizon,
            "event_ticks_and_endpoint": [tick1, tick2, tick_end],
            "test_data_read": False, "model_loaded": False, "optimizer_updates": 0}


def build_v10_equivalent_objective(*, model, tx, tokenizer, classes, cfg, torch,
                                  device, budget, tail_runtime, common,
                                  sample_tokens, lti_reward):
    """Return a batched-runtime callback preserving the local V10 objective.

    This deliberately constructs one request at a time but leaves backward,
    clipping, and AdamW to the V13 batch helper. Returned metadata is aggregate-
    safe and contains no row ids, text, tokens, or labels.
    """
    import banking_cil_v10_runtime as runtime
    from banking_cil_v10_runtime import _decode_answer, _normalized_label

    max_label_with_eos = max(len(tokenizer.encode(label, add_special_tokens=False)) + 1
                             for label in classes)
    if int(cfg["max_new_tokens"]) < max_label_with_eos:
        raise ValueError("Sleep rollout cap is shorter than the longest label plus EOS")
    label_set = {_normalized_label(x) for x in classes}

    def objective(kind, row, role, owner, transaction, request_index):
        if transaction is not tx or kind not in ("gkd", "lti") or owner not in ("teacher", "student"):
            raise ValueError("invalid V13 objective callback arguments")
        prompt_item = common.tokenize_prompt(row["text"], tokenizer, classes,
            tokenizer.eos_token_id, int(cfg["max_sequence_length"]))
        prompt = [int(x) for x in prompt_item["input_ids"][:prompt_item["prompt_length"]]]
        owner_context = tx.teacher_context() if owner == "teacher" else tx.student_context()
        sampled_result, sampled_answer = sample_tokens(model, owner_context, prompt,
            int(cfg["max_new_tokens"]), float(cfg["temperature"]), tx.generator,
            tokenizer.eos_token_id, torch, device, budget)
        if kind == "gkd":
            sequence = sampled_result.token_ids
            answer_mask = sampled_result.answer_mask()
            cache = tail_runtime.capture(model, sequence, torch.ones_like(sequence), device=device)
            with torch.no_grad():
                teacher_logits = tail_runtime.tail_logits(model, tx.teacher_context(), cache)
            student_logits = tail_runtime.tail_logits(model, tx.student_context(), cache)
            loss, token_count = tail_runtime.causal_token_kl(
                student_logits, teacher_logits, answer_mask, temperature=1.0)
            reward = None
        else:
            teacher_ids = list(sampled_answer)
            if teacher_ids and int(teacher_ids[-1]) == int(tokenizer.eos_token_id):
                teacher_ids = teacher_ids[:-1]
            prefix_len = (int(torch.randint(0, len(teacher_ids), (1,), generator=tx.generator,
                device=tx.generator.device).item()) if teacher_ids else 0)
            student_prefix = prompt + teacher_ids[:prefix_len]
            eos_emitted = bool(sampled_answer and int(sampled_answer[-1]) == int(tokenizer.eos_token_id))
            suffix_budget = max(1, len(teacher_ids) - prefix_len + int(eos_emitted))
            student_result, student_answer = sample_tokens(model, tx.student_context(), student_prefix,
                max(1, min(int(cfg["max_new_tokens"]), suffix_budget)), float(cfg["temperature"]),
                tx.generator, tokenizer.eos_token_id, torch, device, budget)
            sequence = student_result.token_ids
            answer_mask = student_result.answer_mask()
            cache = tail_runtime.capture(model, sequence, torch.ones_like(sequence), device=device)
            student_logits = tail_runtime.tail_logits(model, tx.student_context(), cache)
            with torch.no_grad():
                teacher_logits = tail_runtime.tail_logits(model, tx.teacher_context(), cache)
            kl, kl_tokens = tail_runtime.causal_token_kl(
                student_logits, teacher_logits, answer_mask, temperature=1.0)
            student_suffix = list(student_answer)
            if student_suffix and int(student_suffix[-1]) == int(tokenizer.eos_token_id):
                student_suffix = student_suffix[:-1]
            teacher_full = list(teacher_ids[:prefix_len]) + student_suffix
            teacher_text = _decode_answer(teacher_ids, tokenizer, tokenizer.eos_token_id)
            student_text = _decode_answer(teacher_full, tokenizer, tokenizer.eos_token_id)
            reward = float(lti_reward(teacher_text, student_text,
                teacher_tokens=teacher_ids, student_tokens=teacher_full,
                z0=int(cfg["lti_edit_distance_z0"]), canonical_labels=classes))
            ce, ce_tokens = tail_runtime.causal_token_ce(student_logits, sequence, answer_mask)
            loss, token_count = 0.5 * kl + 0.5 * reward * ce, int(kl_tokens) + int(ce_tokens)
        if not bool(torch.isfinite(loss)) or not bool(loss.requires_grad):
            raise RuntimeError("V13 local Sleep objective is nonfinite or nondifferentiable")
        loss_value = float(loss.detach().float().cpu())
        zero_objective = bool(kind == "lti" and reward == 0.0 and loss_value == 0.0)
        return {"kind": kind, "loss": loss, "token_count": max(1, int(token_count)),
                "zero_objective": zero_objective,
                "metadata": {"request_index": int(request_index), "role": str(role),
                             "owner": owner, "lti_reward": reward}}
    return objective


def _run(args, preflight):
    """Run the reviewed finite lifecycle through its separately pinned executor."""
    from banking_v13_execution import run_lifecycle
    return run_lifecycle(args, preflight)
