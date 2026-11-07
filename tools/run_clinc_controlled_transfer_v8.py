"""Development-only transfer-budget probe from a qualified no-Sleep HOPE warmup.

The two target arms are SLEEP_MIX (sampled GKD/LTI) and GOLD_CE_MIX (gold-label
receiver control). Both use the same task-0 HOPE checkpoint and a fixed 0.5
receiver mix. This is a local controlled-routing ablation, not paper-faithful
Sleep. Gold CE exposes extra labels; equal update counts do not imply equal
FLOPs or information. No test data is accepted or read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))


TARGET_ARMS = ("SLEEP_MIX", "GOLD_CE_MIX")
SCHEMA = "clinc-v8-controlled-transfer-probe"


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _simulate_due_slot_peaks(initial_counts, start_clocks, periods, ticks, capacity):
    """Simulate actual sorted due-event resets/activations without model tensors."""
    counts = [int(x) for x in initial_counts]
    clocks = {int(k): int(v) for k, v in start_clocks.items()}
    if len(counts) != len(periods) or any(x < 0 or x > capacity for x in counts):
        raise ValueError("invalid initial active-slot counts")
    if set(clocks) != {i for i, p in enumerate(periods) if p is not None}:
        raise ValueError("clock levels do not match nonterminal periods")
    peaks = list(counts)
    events = []
    for tick in range(1, ticks + 1):
        due = []
        for level, period in enumerate(periods):
            if period is None:
                continue
            clocks[level] += 1
            if clocks[level] == period:
                due.append(level)
        for level in sorted(due):
            if level < len(periods) - 1:
                counts[level] = 0  # transaction resets sender before commit
                counts[level + 1] += 1  # fresh slot is then activated in receiver
                peaks[level + 1] = max(peaks[level + 1], counts[level + 1])
                if counts[level + 1] > capacity:
                    raise ValueError(f"receiver bank {level+1} exceeds capacity at tick {tick}")
            clocks[level] = 0
            events.append({"task_tick": tick, "sender_level": level,
                           "active_slots_after_transaction": list(counts)})
    return {"ticks": ticks, "events": events, "event_counts_by_sender": {
                str(level): sum(e["sender_level"] == level for e in events)
                for level in range(len(periods) - 1)},
            "active_slot_peaks_by_level": peaks, "ending_active_slots_by_level": counts,
            "ending_clocks": clocks}


def _aligned_route_coverage(selected, answer_mask, receiver_slot):
    """Measure natural top-1 at causal target positions (logits t -> token t+1)."""
    target_mask = answer_mask[:, 1:].bool()
    target_selected = selected[:, :-1]
    if target_selected.shape != target_mask.shape:
        raise ValueError("route selection and causal target mask are misaligned")
    total = int(target_mask.sum().item())
    covered = int(((target_selected == int(receiver_slot)) & target_mask).sum().item())
    return {"natural_top1_receiver_tokens": covered, "causal_answer_tokens": total,
            "natural_top1_receiver_coverage": covered / total if total else None}


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                         allow_nan=False).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.partial")
    with tmp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(payload).digest():
        raise IOError("atomic JSON readback mismatch")


def _phase_checkpoint_path(root: Path, arm: str, step: int, phase: str) -> Path:
    return root / "checkpoints" / arm / f"task01-step{step:05d}-{phase}.pt"


def _sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_parameters(context):
    from clinc_tail_runtime_v8 import parameter_counts
    return parameter_counts(context)


def _sleep_method_schedule(cfg):
    k = int(cfg["inner_cycles_per_event"])
    return [kind for _ in range(k) for kind in ("gkd", "gkd", "lti")]


def _prompt_selection_meta(*, event, call, task_id, method_count=3):
    global_call = int(event) * int(method_count) + int(call)
    role = (global_call + int(task_id)) % 2  # 0=old, 1=current
    exposure = global_call // 2
    return {"global_call": global_call, "role": "old" if role == 0 else "current",
            "role_index": role, "role_exposure_index": exposure,
            "class_index": exposure % 15, "example_index": (exposure // 15) % 4}


def _select_event_prompts(row_pool, *, event, task_id, cfg, r2):
    """Select a paired, role-balanced round-robin over all 15 classes/role.

    Each pool must contain exactly four task-training prompts per class. Across
    16 events x 3 calls this yields 24 old and 24 current examples, and touches
    every class in each role. It intentionally does not use the legacy
    `_sleep_prompt_batch` offset (which sampled only a few leading classes).
    """
    methods = _sleep_method_schedule(cfg)
    if not isinstance(row_pool, dict) or set(row_pool) != {"old", "current"}:
        raise ValueError("paired selector expects explicit {'old': ..., 'current': ...} pools")
    old_pool, current_pool = row_pool["old"], row_pool["current"]
    if len(old_pool) != 60 or len(current_pool) != 60:
        raise ValueError("paired selector requires 15 classes x four prompts for each role")
    def indexed(pool):
        grouped = {}
        for row in pool:
            grouped.setdefault(int(row["intent_id"]), []).append(row)
        if len(grouped) != 15 or any(len(rows) != 4 for rows in grouped.values()):
            raise ValueError("each role must have exactly four prompts in each of 15 classes")
        return [sorted(grouped[key], key=lambda row: str(row["id"])) for key in sorted(grouped)]
    role_pools = (indexed(old_pool), indexed(current_pool))
    chosen = []
    for call, kind in enumerate(methods):
        meta = _prompt_selection_meta(event=event, call=call, task_id=task_id,
                                      method_count=len(methods))
        row = role_pools[meta["role_index"]][meta["class_index"]][meta["example_index"]]
        chosen.append((kind, row))
    return chosen


def validate_probe_config(raw):
    required = {"schema", "source_arm", "target_arms", "probe_max_wall_seconds",
                "probe_max_forward_calls", "probe_max_update_steps", "probe_expected_sleep_events",
                "sleep_prompt_selector", "inner_cycles_per_event", "objective_version"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"controlled probe config missing fields: {missing}")
    if raw["schema"] != SCHEMA or raw["source_arm"] != "HOPE":
        raise ValueError("probe must be schema-bound to the R1 HOPE source arm")
    if raw["target_arms"] != list(TARGET_ARMS):
        raise ValueError(f"target_arms must be exactly {list(TARGET_ARMS)}")
    if raw["sleep_prompt_selector"] != "role-balanced-rr-v2-budgeted":
        raise ValueError("sleep_prompt_selector must bind role-balanced-rr-v2-budgeted")
    k = raw["inner_cycles_per_event"]
    if k not in (1, 10) or raw["gkd_calls_per_event"] != 2*k or raw["lti_calls_per_event"] != k:
        raise ValueError("only fixed k=1/10 triplet budgets are supported")
    if raw["objective_version"] != "paper-bounded-token-edit-v1":
        raise ValueError("LTI objective version must be explicitly pinned")
    cfg = dict(raw)
    cfg["arms"] = ["HOPE"]  # only for validating/reading the parent R1 asset
    cfg["schema"] = "clinc-cil-v8-pilot"
    cfg["max_wall_seconds"] = int(raw["probe_max_wall_seconds"])
    cfg["max_forward_calls"] = int(raw["probe_max_forward_calls"])
    cfg["max_update_steps"] = int(raw["probe_max_update_steps"])
    cfg["eval_full_development"] = True
    cfg["full_development_scope"] = "seen_classes"
    cfg["endpoint_full_clean"] = True
    from tools import run_clinc_cil_v8_r2 as r2
    checked = r2.validate_config(cfg)
    if checked["max_steps_per_task"] != 188 or checked["periods"] != (16, 64, 256, 1024, None):
        raise ValueError("probe is fixed to the task-1 188-update R2 schedule")
    if checked["capacity"] != 8 or checked["batch_size"] != 8 or checked["seed"] != 17:
        raise ValueError("probe is fixed to seed17, batch8, capacity8")
    if checked["probe_expected_sleep_events"] != 16:
        raise ValueError("probe expects 16 controlled receiver cycles")
    expected_budget = {1:(1800,20000,600),10:(3600,40000,1600)}[k]
    if (int(raw["probe_max_wall_seconds"]),int(raw["probe_max_forward_calls"]),
        int(raw["probe_max_update_steps"])) != expected_budget:
        raise ValueError("k-specific wall/FW/update caps do not match frozen budget")
    if _sleep_method_schedule(raw) != [x for _ in range(k) for x in ("gkd","gkd","lti")]:
        raise ValueError("sleep method order must be repeated GKD,GKD,LTI triplets")
    return checked


def bounded_token_edit_similarity(left, right, z0):
    """Paper-shaped bounded token-ID edit term: cutoff z0, normalized by maxlen."""
    a,b=list(map(int,left)),list(map(int,right))
    if not a and not b:return 1.0
    if type(z0) is not int or z0<1:raise ValueError("z0 must be positive integer")
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    distance=previous[-1]
    if distance>z0:return 0.0
    return 1.0-distance/max(len(a),len(b),1)


def transfer_lti_reward(teacher_text,student_text,*,teacher_tokens,student_tokens,z0,canonical_labels,semantic_weight=.5,edit_weight=.5):
    import unicodedata
    norm=lambda x:" ".join(unicodedata.normalize("NFKC",str(x)).casefold().replace("_", " ").split())
    allowed={norm(x) for x in canonical_labels}
    t,s=norm(teacher_text),norm(student_text)
    semantic=float(bool(t) and t==s and t in allowed)
    return semantic_weight*semantic+edit_weight*bounded_token_edit_similarity(teacher_tokens,student_tokens,z0)


def _source_lock(project_root: Path, tools_dir: Path, r2):
    base = r2._source_hashes(project_root, tools_dir.parent)
    controlled = tools_dir / "clinc_controlled_routing_v8.py"
    own = Path(__file__).resolve()
    for path in (controlled, own):
        if not path.is_file():
            raise FileNotFoundError(path)
        base[str(path.resolve())] = _sha(path)
    return base, _canonical_sha(base)


def _load_warmup_reference(parent_dir, qualification_path, *, cfg, project_root,
                           tools_dir, manifest_source_sha256, training_manifest_sha256, r2):
    r2_hashes = r2._source_hashes(project_root, tools_dir.parent)
    expected_r2_file = (tools_dir / "run_clinc_cil_v8_r2.py").resolve()
    if Path(r2.__file__).resolve() != expected_r2_file:
        raise RuntimeError("R2 checkpoint validator imported from an unexpected source path")
    r2_file_sha = _sha(expected_r2_file)
    info = r2._load_resume_inputs(parent_dir, qualification_path, cfg=cfg,
        current_source_lock_sha256=_canonical_sha(r2_hashes),
        current_r2_source_sha256=r2_file_sha,
        training_manifest_sha256=training_manifest_sha256,
        manifest_source_sha256=manifest_source_sha256)
    return info, _canonical_sha(r2_hashes), r2_file_sha


def _source_asset_config(parent_dir, target_cfg, r2):
    """Build the legacy-R1 loader config, allowing only dormant GKD/LTI metadata."""
    protocol=json.loads((Path(parent_dir)/"protocol.json").read_text(encoding="utf-8"))
    parent_cfg=protocol.get("config")
    if not isinstance(parent_cfg,dict):raise ValueError("parent protocol lacks its exact config")
    source=dict(target_cfg,schema="clinc-cil-v8-pilot",arms=["HOPE"])
    source["periods"]=list(source["periods"])
    source["gkd_calls_per_event"],source["lti_calls_per_event"]=2,1
    source["source_revision"]=parent_cfg.get("source_revision")
    allowed={"gkd_calls_per_event","lti_calls_per_event"}
    fields=tuple(dict.fromkeys((*r2._RESUME_SCIENCE_FIELDS,
        "training_manifest_sha256","model_revision","model_assets_sha256")))
    differences=[]
    def comparable(value):
        if isinstance(value,tuple):return [comparable(x) for x in value]
        if isinstance(value,list):return [comparable(x) for x in value]
        if isinstance(value,dict):return {k:comparable(v) for k,v in value.items()}
        return value
    for key in fields:
        if key not in parent_cfg or key not in source:raise ValueError(f"parent/source asset config missing {key}")
        if comparable(parent_cfg[key])!=comparable(source[key]):differences.append(key)
    if differences:
        raise ValueError(f"parent/source config differs outside dormant budget allowlist: {differences}")
    resume_differences=r2._resume_config_differences(parent_cfg,source)
    resume_keys=set(resume_differences)
    if resume_keys-allowed:raise ValueError(f"parent/source config differs outside dormant budget allowlist: {resume_differences}")
    if parent_cfg.get("gkd_calls_per_event")!=2 or parent_cfg.get("lti_calls_per_event")!=1:
        raise ValueError("source parent is not the original dormant 2-GKD/1-LTI asset")
    source_checked=r2.validate_config(source)
    return source_checked,{"allowlisted_dormant_budget_fields":sorted(allowed),
        "parent_dormant_gkd_lti":{"gkd_calls_per_event":2,"lti_calls_per_event":1},
        "parent_source_revision_metadata":parent_cfg.get("source_revision"),
        "target_gkd_lti":{"gkd_calls_per_event":target_cfg["gkd_calls_per_event"],"lti_calls_per_event":target_cfg["lti_calls_per_event"]},
        "compared_science_fields":list(fields),
        "allowed_nontraining_metadata_differences":{key:{"parent":parent_cfg.get(key),"loader":source.get(key)}
            for key in ("eval_examples_per_class","eval_full_development","full_development_scope",
                        "endpoint_full_clean","checkpoint_every_steps")
            if comparable(parent_cfg.get(key))!=comparable(source.get(key))},
        "r2_resume_science_fields":list(r2._RESUME_SCIENCE_FIELDS),
        "r2_resume_config_differences":resume_differences}


def _assert_no_prior_sleep(parent_receipt):
    work=parent_receipt.get("work_arm_delta",{})
    # R1's terminal receipt names this field sleep_events, while the newer
    # controlled runner uses sleep_event_count. Do not invent a missing zero.
    event_fields=[parent_receipt[key] for key in ("sleep_events","sleep_event_count")
                  if key in parent_receipt]
    observed=event_fields+[parent_receipt.get("physical_sleep_adamw_steps"),work.get("sleep_events")]
    if not event_fields or any(type(value) is not int or value != 0 for value in observed):
        raise ValueError(f"source HOPE checkpoint must have zero prior Sleep event/Adam counters, got {observed}")


def _restore_parent(context, controller, cp_info, *, r2, cfg, torch):
    item = cp_info["arms"]["HOPE"]
    return r2._restore_parent_task0_checkpoint(context, controller, Path(item["checkpoint_path"]),
        receipt=item["receipt"], qualification=item["qualification"], cfg=cfg, torch=torch)


def _save_phase(root, arm, step, phase, context, wake, *, r2, torch, cfg,
                source_lock, parent_cp_sha, config_sha, manifest_sha,
                event=None, transaction=None):
    checkpoint = {"schema": "clinc-v8-controlled-transfer-state-v1", "arm": arm,
        "source_arm": "HOPE", "seed": cfg["seed"], "task_id": 1, "wake_step": step,
        "phase": phase, "context": context.state_dict(), "wake": wake.snapshot(),
        "rng": r2._rng_state(torch), "source_lock_sha256": source_lock,
        "config_sha256": config_sha, "training_manifest_sha256": manifest_sha,
        "parent_task0_checkpoint_sha256": parent_cp_sha,
        "event": event}
    if transaction is not None:
        checkpoint["transaction_snapshot"] = transaction.snapshot()
    path = _phase_checkpoint_path(root, arm, step, phase)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint phase: {path}")
    digest = r2._atomic_torch(path, checkpoint, torch)
    return {"path": str(path.relative_to(root)), "sha256": digest,
            "task_id": 1, "wake_step": step, "phase": phase,
            "parent_task0_checkpoint_sha256": parent_cp_sha}


def _audited_student_logits(model, context, cache, receiver_level, receiver_slot, answer_mask,
                            tail_runtime):
    banks = context.banks()
    saved = [(bool(bank.audit_routes), bank.last_route_audit, list(bank.audit_history),
              deepcopy(bank.last_routes)) for bank in banks]
    try:
        for index, bank in enumerate(banks):
            bank.audit_routes = index == receiver_level
        logits = tail_runtime.tail_logits(model, context, cache)
        audit = banks[receiver_level].last_route_audit
        if audit is None or audit.get("mixed_slot") != receiver_slot or audit.get("mixed_fraction") != 0.5:
            raise RuntimeError("fresh receiver mix audit did not match the fixed 0.5 policy")
        coverage = _aligned_route_coverage(audit["selected"], answer_mask, receiver_slot)
        coverage.update({"controlled_mix_slot": receiver_slot,
                         "controlled_mix_fraction": 0.5,
                         "fresh_branch_executed_answer_coverage": 1.0,
                         "natural_router_score_abs_max": audit.get("score_abs_max")})
        return logits, coverage
    finally:
        for bank, (audit_routes, last_audit, history, routes) in zip(banks, saved):
            bank.audit_routes = audit_routes
            bank.last_route_audit = last_audit
            bank.audit_history = history
            bank.last_routes = routes


def _run_controlled_sleep_event(model, tx, old_pool, current_pool, tokenizer, view, cfg, torch,
                                device, budget, task_id, event, r2, last_event):
    import clinc_tail_runtime_v8 as tail_runtime
    from clinc_tail_runtime_v8 import capture, causal_token_ce, causal_token_kl
    selections = _select_event_prompts({"old": old_pool, "current": current_pool},
        event=event, task_id=task_id, cfg=cfg, r2=r2)
    outputs = []
    gkd_index = 0
    valid_labels = {r2.normalize_text(x) for x in view.classes}
    for call, (kind, row) in enumerate(selections):
        selection_meta = _prompt_selection_meta(event=event, call=call, task_id=task_id,
                                                method_count=len(selections))
        budget.check()
        prompt = r2._prefix(row, tokenizer, view, cfg)
        owner = "teacher" if kind == "lti" or gkd_index % 2 == 0 else "student"
        if kind == "gkd":
            gkd_index += 1
        owner_context = tx.teacher_context() if owner == "teacher" else tx.student_context()
        sampled_result, sampled_answer = r2._sample_tokens(model, owner_context, prompt,
            cfg["max_new_tokens"], cfg["temperature"], tx.generator, tokenizer.eos_token_id,
            torch, device, budget)
        if kind == "gkd":
            sequence = sampled_result.token_ids
            answer_mask = sampled_result.answer_mask()
            cache = capture(model, sequence, torch.ones_like(sequence), device=device)
            with torch.no_grad():
                teacher_logits = tail_runtime.tail_logits(model, tx.teacher_context(), cache)
            student_logits, route_stats = _audited_student_logits(model, tx.student_context(),
                cache, tx.receiver, tx.slot, answer_mask, tail_runtime)
            loss, token_count = causal_token_kl(student_logits, teacher_logits, answer_mask,
                                                temperature=1.0)
            pre_kl = float(loss.detach().float().cpu())
            reward = None
            student_ce_value = None
        else:
            teacher_ids = sampled_answer[:-1] if sampled_answer and sampled_answer[-1] == tokenizer.eos_token_id else sampled_answer
            prefix_len = (int(torch.randint(0, len(teacher_ids), (1,), generator=tx.generator,
                             device=tx.generator.device).item()) if teacher_ids else 0)
            student_prefix = prompt + teacher_ids[:prefix_len]
            eos_emitted = bool(sampled_answer and sampled_answer[-1] == tokenizer.eos_token_id)
            suffix_budget = max(1, len(teacher_ids) - prefix_len + int(eos_emitted))
            student_result, student_answer = r2._sample_tokens(model, tx.student_context(),
                student_prefix, max(1, min(cfg["max_new_tokens"], suffix_budget)),
                cfg["temperature"], tx.generator, tokenizer.eos_token_id, torch, device, budget)
            sequence = student_result.token_ids
            answer_mask = student_result.answer_mask()
            cache = capture(model, sequence, torch.ones_like(sequence), device=device)
            student_logits, route_stats = _audited_student_logits(model, tx.student_context(),
                cache, tx.receiver, tx.slot, answer_mask, tail_runtime)
            with torch.no_grad():
                teacher_logits = tail_runtime.tail_logits(model, tx.teacher_context(), cache)
            kl, token_count = causal_token_kl(student_logits, teacher_logits, answer_mask, temperature=1.0)
            pre_kl = float(kl.detach().float().cpu())
            teacher_text = r2._decode_answer(teacher_ids, tokenizer, tokenizer.eos_token_id)
            student_suffix = student_answer[:-1] if student_answer and student_answer[-1] == tokenizer.eos_token_id else student_answer
            student_full = teacher_ids[:prefix_len] + student_suffix
            student_text = r2._decode_answer(student_full, tokenizer, tokenizer.eos_token_id)
            reward = transfer_lti_reward(teacher_text, student_text, teacher_tokens=teacher_ids,
                student_tokens=student_full, z0=cfg["lti_edit_distance_z0"],
                canonical_labels=view.classes, semantic_weight=cfg["lti_semantic_weight"],
                edit_weight=cfg["lti_edit_weight"])
            student_ce, ce_tokens = causal_token_ce(student_logits, sequence, answer_mask)
            student_ce_value = float(student_ce.detach().float().cpu())
            loss, token_count = 0.5 * kl + 0.5 * reward * student_ce, token_count + ce_tokens

        loss_value = float(loss.detach().float().cpu())
        if not bool(torch.isfinite(loss)) or not loss.requires_grad:
            last_event.update({"loss": loss_value, "loss_requires_grad": bool(loss.requires_grad),
                               "route": route_stats, "kind": kind, "row_id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest()})
            raise RuntimeError("controlled Sleep produced a nonfinite or nondifferentiable loss")
        budget.backward(torch.ones((1, token_count), dtype=torch.long, device=device))
        budget.update()
        previous_zero = tx.counters["zero_gradient_steps"]
        tx.optimize(loss, kind, max(1, int(token_count)))
        if tx.counters["zero_gradient_steps"] != previous_zero:
            last_event.update({"loss": loss_value, "route": route_stats, "kind": kind,
                               "row_id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest(),
                               "zero_gradient_step": True})
            raise RuntimeError("controlled receiver optimizer observed zero gradient")
        with torch.no_grad():
            after_logits, after_route_stats = _audited_student_logits(model, tx.student_context(),
                cache, tx.receiver, tx.slot, answer_mask, tail_runtime)
            post_kl, _ = causal_token_kl(after_logits, teacher_logits, answer_mask, temperature=1.0)
            post_kl_value = float(post_kl.detach().float().cpu())
            post_ce_value = None
            post_loss_value = post_kl_value
            if kind == "lti":
                post_ce, _ = causal_token_ce(after_logits, sequence, answer_mask)
                post_ce_value = float(post_ce.detach().float().cpu())
                post_loss_value = 0.5 * post_kl_value + 0.5 * reward * post_ce_value
        teacher_text = r2._decode_answer(sampled_answer, tokenizer, tokenizer.eos_token_id)
        outputs.append({"kind": kind, "sample_owner": owner, "loss": loss_value,
            "paired_prompt_selection": selection_meta,
            "token_count": int(token_count), "kl_weight": 1.0 if kind == "gkd" else 0.5,
            "same_sample_fit_before": {"kl": pre_kl, "student_ce": student_ce_value,
                                        "objective": loss_value},
            "same_sample_fit_after": {"kl": post_kl_value, "student_ce": post_ce_value,
                                       "objective": post_loss_value},
            "lti_reward_proxy": reward, "sampled_label_valid": r2.normalize_text(teacher_text) in valid_labels,
            "sampled_label_matches_prompt_gold_diagnostic": r2.normalize_text(teacher_text) ==
                r2.normalize_text(view.classes[int(row["intent_id"])]),
            "prompt_id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest(),
            "prompt_task_id": next(i for i in range(10) if int(row["intent_id"]) in
                set(view.class_orders[str(cfg["seed"])][i * 15:(i + 1) * 15])),
            "natural_vs_controlled_route_before": route_stats,
            "natural_vs_controlled_route_after": after_route_stats})
        last_event.update({"loss": loss_value, "kind": kind, "route": route_stats,
                           "row_id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest()})
    return outputs


def _run_gold_ce_event(model, tx, old_pool, current_pool, tokenizer, view, cfg, torch, device, budget,
                       task_id, event, r2, last_event):
    import clinc_tail_runtime_v8 as tail_runtime
    from clinc_tail_runtime_v8 import capture, causal_token_ce
    outputs = []
    selections = _select_event_prompts({"old": old_pool, "current": current_pool},
        event=event, task_id=task_id, cfg=cfg, r2=r2)
    for call, (kind, row) in enumerate(selections):
        selection_meta = _prompt_selection_meta(event=event, call=call, task_id=task_id,
                                                method_count=len(selections))
        _data, ids, mask, labels = r2._batch([row], tokenizer, view, cfg, torch, device)
        cache = capture(model, ids, mask, device=device)
        answer_mask = labels.ne(-100)
        logits, route_stats = _audited_student_logits(model, tx.student_context(), cache,
            tx.receiver, tx.slot, answer_mask, tail_runtime)
        loss, token_count = causal_token_ce(logits, ids, answer_mask)
        target_mask = labels[:, 1:].ne(-100)
        loss_value = float(loss.detach().float().cpu())
        last_event.update({"kind": "gold_ce", "loss": loss_value,
                           "route": route_stats,
                           "row_id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest()})
        if not bool(torch.isfinite(loss)) or not loss.requires_grad:
            raise RuntimeError("gold receiver control produced nonfinite/nondifferentiable CE")
        budget.backward(target_mask.long())
        budget.update()
        previous_zero = tx.counters["zero_gradient_steps"]
        tx.optimize(loss, "gkd", max(1, int(token_count)))
        if tx.counters["zero_gradient_steps"] != previous_zero:
            raise RuntimeError("gold receiver control observed zero gradient")
        with torch.no_grad():
            after_logits, after_route_stats = _audited_student_logits(model, tx.student_context(), cache,
                tx.receiver, tx.slot, answer_mask, tail_runtime)
            post_ce, _ = causal_token_ce(after_logits, ids, answer_mask)
            post_loss = float(post_ce.detach().float().cpu())
        outputs.append({"kind": "gold_ce", "uses_gold_intent": True,
            "paired_sleep_request_kind": kind,
            "paired_prompt_selection": selection_meta,
            "loss": loss_value, "token_count": int(token_count),
            "same_sample_fit_before": loss_value,
            "same_sample_fit_after": post_loss,
            "prompt_id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest(),
            "natural_vs_controlled_route_before": route_stats,
            "natural_vs_controlled_route_after": after_route_stats})
    return outputs


def _delta_stats(before, after, allowed):
    left, right = dict(before.named_parameters()), dict(after.named_parameters())
    total_sq = 0.0
    changed = 0
    for name in allowed:
        delta = (right[name].detach().float() - left[name].detach().float()).cpu()
        total_sq += float(delta.square().sum())
        changed += int(bool(torch.count_nonzero(delta)))
    return {"receiver_parameter_names": sorted(allowed), "changed_parameter_tensors": changed,
            "receiver_delta_l2": math.sqrt(total_sq)}


def _budget_delta(before, after):
    out = {}
    for key, value in after.items():
        old = before.get(key)
        if isinstance(value, dict):
            out[key] = {k: v - (old or {}).get(k, 0) for k, v in value.items()
                        if isinstance(v, (int, float))}
        elif isinstance(value, (int, float)):
            out[key] = value - (old if isinstance(old, (int, float)) else 0)
    return out


def run(args):
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    pack_path = Path(args.training_pack).resolve()
    output_dir = Path(args.output_dir).resolve()
    cfg_path = Path(args.config).resolve()
    parent_dir = Path(args.warmup_run_dir).resolve()
    qualification_path = Path(args.qualification).resolve()
    for path, label in ((model_path, "model directory"), (pack_path, "training pack"),
                        (cfg_path, "config"), (parent_dir, "warmup run directory"),
                        (qualification_path, "corrected qualification")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not model_path.is_dir() or not pack_path.is_file() or not cfg_path.is_file() or not parent_dir.is_dir() or not qualification_path.is_file():
        raise ValueError("invalid model/pack/config/warmup/qualification path type")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    if output_dir == project_root or project_root in output_dir.parents or output_dir == model_path or model_path in output_dir.parents:
        raise ValueError("output must be outside public source and model assets")
    cfg = validate_probe_config(json.loads(cfg_path.read_text(encoding="utf-8")))
    from tools import run_clinc_cil_v8_r2 as r2
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from tools import run_clinc_cil_v8_r2 as r2
    r2_cfg, source_budget_note = _source_asset_config(parent_dir, cfg, r2)
    source_hashes, source_lock = _source_lock(project_root, tools_dir, r2)
    if _sha(pack_path) != cfg["training_manifest_sha256"]:
        raise ValueError("training manifest SHA does not match config")
    for relative, expected in cfg["model_assets_sha256"].items():
        asset = (model_path / relative).resolve()
        if model_path not in asset.parents or not asset.is_file() or _sha(asset) != expected:
            raise ValueError(f"locked model asset mismatch: {relative}")

    manifest = json.loads(pack_path.read_text(encoding="utf-8"))
    from clinc_cil_common_v8 import (make_training_view, iter_role, validate_role_partition,
                                     task_class_ids, shuffled_ids)
    view = make_training_view(manifest)
    pack_root = pack_path.parent
    train_rows = list(iter_role(view, "train", pack_root))
    dev_rows = list(iter_role(view, "development", pack_root))
    partition = validate_role_partition(train_rows, dev_rows)
    train_norm = {r2.normalize_text(row["text"]) for row in train_rows}
    clean_dev = [row for row in dev_rows if r2.normalize_text(row["text"]) not in train_norm]

    # This validates the corrected task-0 rescore and the authentic R1 HOPE
    # terminal receipt without converting the parent run into a completed probe.
    cp_info, r2_lock, r2_file_sha = _load_warmup_reference(parent_dir, qualification_path,
        cfg=r2_cfg, project_root=project_root, tools_dir=tools_dir,
        manifest_source_sha256=view.source_sha256,
        training_manifest_sha256=cfg["training_manifest_sha256"], r2=r2)
    if set(cp_info["arms"]) != {"HOPE"}:
        raise ValueError("controlled probe source asset must be exactly the HOPE arm")
    parent_receipt = cp_info["arms"]["HOPE"]["receipt"]
    _assert_no_prior_sleep(parent_receipt)
    parent_cp_sha = parent_receipt["final_checkpoint_sha256"]

    order = view.class_orders[str(cfg["seed"])]
    task_ids = set(task_class_ids(order, 1))
    task_train = [row for row in train_rows if int(row["intent_id"]) in task_ids]
    if len(task_train) < cfg["examples_per_task"] or len(order) != 150:
        raise ValueError("training-only pack does not contain the locked task-1 schedule")
    if len(dev_rows) != 3000 or len(train_rows) != 15000:
        raise ValueError("expected locked 15000 train/3000 development rows")

    output_dir.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    budget = r2.Budget(cfg)
    budget.started = started
    torch.manual_seed(cfg["initialization_seed"])
    random.seed(cfg["initialization_seed"])
    np.random.seed(cfg["initialization_seed"])
    try:
        loaded_torch, tokenizer, model, immutable_versions = r2._load_common(project_root, model_path,
            cfg["device"], cfg)
        if loaded_torch is not torch:
            raise RuntimeError("model loader returned a different torch module")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        frozen_versions = immutable_versions(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        from clinc_tail_runtime_v8 import clone_mutable_context
        from clinc_controlled_routing_v8 import ControlledSleepTransaction, ControlledWakeController

        original_logits = model.logits
        def counted_logits(tokens, mask=None):
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_logits(tokens, mask)
        model.logits = counted_logits
        import clinc_tail_runtime_v8 as tail_runtime
        original_tail_logits = tail_runtime.tail_logits
        def counted_tail_logits(tail_model, context, cache):
            active = torch.is_grad_enabled() and any(p.requires_grad for p in context.parameters())
            budget.tail(cache, active)
            return original_tail_logits(tail_model, context, cache)
        tail_runtime.tail_logits = counted_tail_logits

        _atomic_json(output_dir / "protocol.json", {"schema": SCHEMA,
            "config": cfg, "source_arm": "HOPE", "target_arms": list(TARGET_ARMS),
            "parent_r1_protocol_sha256": cp_info["parent_protocol_sha256"],
            "parent_r1_source_lock_sha256": cp_info["parent_source_lock_sha256"],
            "parent_corrected_rescore_sha256": cp_info["qualification_sha256"],
            "parent_task0_checkpoint_sha256": parent_cp_sha,
            "source_asset_config_binding": source_budget_note,
            "target_budget_config_sha256": _canonical_sha(cfg),
            "source_asset_config_sha256": _canonical_sha(r2_cfg),
            "r2_source_lock_sha256_used_for_rescore_validation": r2_lock,
            "r2_source_file_sha256_used_for_rescore_validation": r2_file_sha,
            "source_hashes": source_hashes, "source_lock_sha256": source_lock,
            "training_manifest_sha256": cfg["training_manifest_sha256"],
            "model_revision": cfg["model_revision"], "model_assets_sha256": cfg["model_assets_sha256"],
            "source_task0_clean_accuracy": cp_info["arms"]["HOPE"]["task0_clean_accuracy"],
            "task1_schedule": "same seed17 shuffled_ids(train_task1, task_id=1), 1500 rows, batch8, 188 updates",
            "sleep_policy": "48*k paired calls across 16 events: 24*k old + 24*k current prompts; deterministic round-robin covers all 15 classes per role; prior class metadata selects text, labels excluded from Sleep loss",
            "sleep_prompt_selector": {"schema": "role-balanced-rr-v2-budgeted",
                "formula": "global_call=event*(3*k)+call; role=(global_call+task_id)%2 (0=old,1=current); role_exposure=global_call//2; class_index=role_exposure%15; example_index=(role_exposure//15)%4",
                "expected_task1_calls_by_role": {"old": 24*cfg["inner_cycles_per_event"], "current": 24*cfg["inner_cycles_per_event"]},
                "expected_task1_class_indices_by_role": list(range(15))},
            "objective_version": cfg["objective_version"],
            "inner_cycles_per_event": cfg["inner_cycles_per_event"],
            "gkd_lti_policy": "repeat [GKD,GKD,LTI] k times/event, alternating GKD sample owner; local LTI proxy with bounded token-edit similarity",
            "controlled_route": "fresh receiver residual mix fraction 0.5 at same sender/receiver lifecycle; persistent after commit",
            "routing_cost_note": "natural winner branch plus explicit fresh branch execute; overlap may execute the fresh expert twice, so active parameter count is not FLOPs or per-token work",
            "gold_control": "same receiver scope and update count, three gold-CE optimizer steps/event; extra labels and different sampling/FW cost, not FLOP matched",
            "checkpoint_schedule": "task1 start, pre-sleep boundary, every 16 wake steps, task1 endpoint; failure preserves known work and transaction forensic snapshot",
            "update_counter_semantics": "budget optimizer_or_lifecycle_update_steps counts updates reserved before optimization; physical_sleep_adamw_steps uses completed transaction AdamW counters",
            "evaluation_scope": "seen 30 classes only, full-clean-development primary, fixed 2/class intermediate, no task ID or candidate restriction",
            "data_partition_audit": partition,
            "test_data_read": False,
            "scientific_limitations": ["last-MLP CMS-inspired Llama adapter, not native HOPE", "controlled route intervention is not natural Sleep", "GOLD_CE_MIX has extra labeled targets", "k=1 uses the corrected bounded token-edit objective and is not numerically interchangeable with the historical k=1 probe that used a different edit proxy", "full-backbone forward counts are not FLOPs"]})

        metrics_path = output_dir / "metrics.jsonl"
        receipts = []
        config_sha = _canonical_sha(cfg)
        manifest_sha = _sha(pack_path)
        common_source_context_sha = None
        probe_plan = None
        task_train_by_id = {str(row["id"]): row for row in task_train}
        schedule_ids = shuffled_ids(task_train, cfg["seed"], 1)[:cfg["examples_per_task"]]
        schedule_rows = [task_train_by_id[row_id] for row_id in schedule_ids]
        old_ids = set(task_class_ids(order, 0))
        old_rows = [row for row in train_rows if int(row["intent_id"]) in old_ids]
        current_pool = []
        for class_id in sorted(task_ids):
            current_pool.extend(sorted((row for row in task_train if int(row["intent_id"]) == class_id), key=lambda x: x["id"])[:4])
        old_pool = []
        for class_id in sorted(old_ids):
            old_pool.extend(sorted((row for row in old_rows if int(row["intent_id"]) == class_id), key=lambda x: x["id"])[:4])
        if cfg["sleep_prompt_policy"] != "seen_replay" or not old_pool:
            raise ValueError("controlled Sleep probe requires fixed seen_replay prompts")
        if len(old_pool) != 60 or len(current_pool) != 60:
            raise ValueError("fixed replay pools must contain four task-train prompts per class")
        selector_ids = []
        selector_counts = {role: {"calls": 0, "classes": set(), "ids": []}
                           for role in ("old", "current")}
        for event_index in range(int(cfg["probe_expected_sleep_events"])):
            selected = _select_event_prompts({"old": old_pool, "current": current_pool},
                event=event_index, task_id=1, cfg=cfg, r2=r2)
            for call, (_kind, row) in enumerate(selected):
                meta = _prompt_selection_meta(event=event_index, call=call,
                    task_id=1, method_count=len(selected))
                entry = selector_counts[meta["role"]]
                entry["calls"] += 1
                entry["classes"].add(meta["class_index"])
                entry["ids"].append(str(row["id"]))
                selector_ids.append(str(row["id"]))
        expected_per_role = 24 * int(cfg["inner_cycles_per_event"])
        if any(entry["calls"] != expected_per_role or len(entry["classes"]) != 15
               for entry in selector_counts.values()) or len(selector_ids) != 48 * int(cfg["inner_cycles_per_event"]):
            raise RuntimeError("paired seen-replay selector failed role/class coverage preflight")
        selector_audit = {role: {"calls": entry["calls"],
            "class_indices_covered": sorted(entry["classes"]),
            "prompt_ids_sha256": _canonical_sha(entry["ids"])}
            for role, entry in selector_counts.items()}

        with metrics_path.open("w", encoding="utf-8", newline="\n", buffering=1) as metrics:
            for arm in TARGET_ARMS:
                context = clone_mutable_context(model)
                wake = ControlledWakeController(context, periods=cfg["periods"],
                    base_lr=cfg["base_lr"], fast_lr=cfg["fast_lr"])
                wake.prepare_wake()
                restored = _restore_parent(context, wake, cp_info, r2=r2, cfg=r2_cfg, torch=torch)
                wake.prepare_wake()
                context_sha = r2._mutable_context_sha(context)
                if common_source_context_sha is None:
                    common_source_context_sha = context_sha
                elif context_sha != common_source_context_sha:
                    raise RuntimeError("controlled arms did not start from byte-identical HOPE warmup context")
                initial_active = [sum(bool(slot.active) for slot in bank.slots) for bank in context.banks()]
                slot_plan = _simulate_due_slot_peaks(initial_active, wake.clocks, wake.periods,
                    cfg["max_steps_per_task"], cfg["capacity"])
                if len(slot_plan["events"]) != cfg["probe_expected_sleep_events"]:
                    raise ValueError("actual warmup pending clocks imply unexpected Sleep cycle count")
                if probe_plan is None:
                    probe_plan = {"source_context_sha256": context_sha,
                        "warmup_ticks": wake.ticks, "warmup_pending_clocks": dict(wake.clocks),
                        "initial_active_slots_by_level": initial_active,
                        "task1_due_slot_simulation": slot_plan,
                        "task1_schedule_ids_sha256": _canonical_sha(schedule_ids),
                        "sleep_prompt_old_ids_sha256": _canonical_sha([row["id"] for row in old_pool]),
                        "sleep_prompt_current_ids_sha256": _canonical_sha([row["id"] for row in current_pool]),
                        "sleep_prompt_exposures": selector_audit}
                elif slot_plan != probe_plan["task1_due_slot_simulation"]:
                    raise RuntimeError("controlled arms have different due-event/slot trajectories")

                arm_start = budget.as_dict()
                arm_started = time.monotonic()
                latest_event = {}
                arm_checkpoints = []
                sleep_adam_steps = 0
                metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "event": "run_start",
                    "source_arm": "HOPE", "parent_checkpoint_sha256": parent_cp_sha,
                    "warm_context_sha256": context_sha, "restored_ticks": restored["restored_ticks"],
                    "restored_clocks": restored["restored_clocks"], "active_slots_by_level": initial_active,
                    "slot_plan": slot_plan, "parameter_counts": _count_parameters(context)}, sort_keys=True) + "\n")

                def save_phase(step, phase, event=None, transaction=None):
                    item = _save_phase(output_dir, arm, step, phase, context, wake,
                        r2=r2, torch=torch, cfg=cfg, source_lock=source_lock,
                        parent_cp_sha=parent_cp_sha, config_sha=config_sha,
                        manifest_sha=manifest_sha, event=event, transaction=transaction)
                    arm_checkpoints.append(item)
                    return item

                try:
                    tx = None
                    # Full clean seen-only and a fixed 2/class progress subset; no score-based selection.
                    seen_ids = set(order[:30])
                    clean_seen = [row for row in clean_dev if int(row["intent_id"]) in seen_ids]
                    progress_rows = r2.fixed_eval_rows(clean_seen, per_class=2)
                    for label, rows in (("clean_full", clean_seen), ("clean_fixed_2_per_class", progress_rows),
                                        ("development_full_seen", dev_rows)):
                        result = r2._evaluate(model, context, rows, order, 1, tokenizer, view, cfg,
                            torch, cfg["device"], budget, include_unseen=False)
                        metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": 1,
                            "event": f"task1_start_eval_{label}", "evaluation_scope": "seen_classes",
                            "result": result}, sort_keys=True) + "\n")
                    save_phase(0, "task1-start")
                    sleep_events = 0
                    task_updates = 0
                    for step, start in enumerate(range(0, len(schedule_rows), cfg["batch_size"]), 1):
                        budget.check()
                        wake.prepare_wake()
                        rows = schedule_rows[start:start + cfg["batch_size"]]
                        _data, ids, mask, labels = r2._batch(rows, tokenizer, view, cfg, torch, cfg["device"])
                        cache = tail_runtime.capture(model, ids, mask, device=cfg["device"])
                        for parameter in context.parameters():
                            parameter.grad = None
                        logits = tail_runtime.tail_logits(model, context, cache)
                        answer_mask = labels.ne(-100)
                        targets = labels[:, 1:].reshape(-1)
                        valid = targets.ne(-100)
                        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float()[valid], targets[valid])
                        if not torch.isfinite(loss):
                            raise RuntimeError("nonfinite wake CE")
                        loss.backward()
                        budget.backward(mask * answer_mask.long())
                        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in context.parameters() if p.requires_grad],
                            cfg["sleep_clip_norm"], error_if_nonfinite=True)
                        allowed = {name: p.grad.detach().clone() for name, p in context.named_parameters() if p.grad is not None}
                        budget.update()
                        wake.record_tick(allowed, token_count=int(answer_mask.sum().item()))
                        for parameter in context.parameters():
                            parameter.grad = None
                        task_updates += 1
                        metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": 1,
                            "step": step, "global_wake_tick": wake.ticks, "event": "wake_update",
                            "ce": float(loss.detach().float().cpu()), "grad_norm_before_clip": float(grad_norm.detach().float().cpu()),
                            "active_slots_by_level": [sum(bool(s.active) for s in b.slots) for b in context.banks()],
                            "last_forward_routes_prompt_plus_answer": [[dict(route) for route in b.last_routes] for b in context.banks()]}, sort_keys=True) + "\n")

                        for level in sorted(wake.proposals()):
                            tx = None
                            proposal = wake.proposals()[level]
                            if level < len(cfg["periods"]) - 1:
                                pre_cp = save_phase(step, f"pre-sleep-level{level}-event{sleep_events:03d}", {"level": level,
                                    "global_tick": wake.ticks, "event_index": sleep_events})
                                tx = ControlledSleepTransaction(context, level, proposal,
                                    lr=cfg["sleep_lr"], weight_decay=cfg["sleep_weight_decay"],
                                    clip_norm=cfg["sleep_clip_norm"], seed=cfg["seed"] + step * 100 + level)
                                outputs = []
                                if arm == "SLEEP_MIX":
                                    outputs = _run_controlled_sleep_event(model, tx, old_pool, current_pool, tokenizer, view,
                                        cfg, torch, cfg["device"], budget, 1, sleep_events, r2, latest_event)
                                else:
                                    outputs = _run_gold_ce_event(model, tx, old_pool, current_pool, tokenizer, view, cfg,
                                        torch, cfg["device"], budget, 1, sleep_events, r2, latest_event)
                                sleep_adam_steps += tx.counters["adamw_steps"]
                                new_context = tx.commit()
                                delta = _delta_stats(context, new_context, tx.allowed_names)
                                wake.bind_context(new_context)
                                wake.consume(level)
                                context = new_context
                                sleep_events += 1
                                budget.sleep_events += 1
                                metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": 1,
                                    "step": step, "global_wake_tick": wake.ticks, "event": "controlled_sleep_commit",
                                    "sender_level": level, "receiver_level": tx.receiver, "receiver_slot": tx.slot,
                                    "sleep_optimizer_updates": tx.counters["adamw_steps"],
                                    "receiver_parameter_delta": delta, "losses": outputs,
                                    "teacher_label_use": arm == "GOLD_CE_MIX",
                                    "natural_router_coverage_semantics": "top1 selection at causal target positions only",
                                    "controlled_mix_fraction": 0.5,
                                    "checkpoint_before_event": pre_cp,
                                    "active_slots_by_level": [sum(bool(s.active) for s in b.slots) for b in context.banks()]}, sort_keys=True) + "\n")
                                remaining_due = sorted(wake.proposals())
                                save_phase(step, f"post-sleep-level{level}-event{sleep_events-1:03d}",
                                    {"sender_level": level, "completed_event_index": sleep_events - 1,
                                     "other_due_levels_pending": remaining_due,
                                     "safe_to_resume_at_next_wake": not remaining_due})
                                tx = None
                            else:
                                wake.apply(level, proposal)
                                context = wake.context
                        if step % 16 == 0:
                            save_phase(step, "post-step-checkpoint")
                        if step >= cfg["max_steps_per_task"]:
                            break

                    expected_adam=48*int(cfg["inner_cycles_per_event"])
                    if task_updates != 188 or sleep_events != 16 or sleep_adam_steps != expected_adam:
                        raise RuntimeError(f"task1 schedule mismatch: updates={task_updates}, sleep_events={sleep_events}, physical_sleep_adamw={sleep_adam_steps}, expected_adam={expected_adam}")
                    for label, rows in (("clean_full", clean_seen), ("clean_fixed_2_per_class", progress_rows),
                                        ("development_full_seen", dev_rows)):
                        result = r2._evaluate(model, context, rows, order, 1, tokenizer, view, cfg,
                            torch, cfg["device"], budget, include_unseen=False)
                        metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "task_id": 1,
                            "event": f"task1_end_eval_{label}", "evaluation_scope": "seen_classes",
                            "result": result}, sort_keys=True) + "\n")
                    final_cp = save_phase(188, "task1-endpoint")
                    receipt = {"arm": arm, "seed": cfg["seed"], "source_arm": "HOPE",
                        "parent_task0_checkpoint_sha256": parent_cp_sha,
                        "parent_corrected_rescore_sha256": cp_info["qualification_sha256"],
                        "warm_context_sha256": context_sha, "ticks": wake.ticks,
                        "task1_physical_wake_updates": task_updates,
                        "task1_logical_cumulative_ticks": wake.ticks,
                        "sleep_event_count": sleep_events,
                        "physical_sleep_adamw_steps": sleep_adam_steps,
                        "checkpoint_endpoint": final_cp,
                        "work_arm_delta": _budget_delta(arm_start, budget.as_dict()),
                        "active_seconds": time.monotonic() - arm_started,
                        "parameter_counts": _count_parameters(context),
                        "active_slots_by_level": [sum(bool(s.active) for s in b.slots) for b in context.banks()],
                        "all_events_committed": True}
                    receipts.append(receipt)
                    metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "event": "run_complete",
                                              "receipt": receipt}, sort_keys=True) + "\n")
                    metrics.flush()
                except BaseException as exc:
                    failure = {"schema": "clinc-v8-controlled-transfer-failure", "status": "failed",
                        "arm": arm, "seed": cfg["seed"], "error_type": type(exc).__name__,
                        "error": str(exc), "active_seconds": time.monotonic() - arm_started,
                        "work_global": budget.as_dict(), "work_arm_delta": _budget_delta(arm_start, budget.as_dict()),
                        "latest_sleep_loss_and_route": latest_event,
                        "checkpoints": arm_checkpoints, "last_pending_levels": wake.clocks,
                        "transaction_snapshot_stored": False,
                        "automatic_retry": False}
                    if tx is not None and not tx.counters.get("closed", False):
                        try:
                            forensic = {"schema": "clinc-v8-controlled-transfer-failed-transaction-v1",
                                "arm": arm, "task_id": 1, "wake_tick": wake.ticks,
                                "event_index": sleep_events, "transaction": tx.snapshot(),
                                "last_loss_route": latest_event, "source_lock_sha256": source_lock,
                                "parent_task0_checkpoint_sha256": parent_cp_sha}
                            forensic_path = output_dir / "failure" / f"{arm}-event{sleep_events:03d}.pt"
                            failure["transaction_snapshot_path"] = str(forensic_path.relative_to(output_dir))
                            failure["transaction_snapshot_sha256"] = r2._atomic_torch(forensic_path, forensic, torch)
                            failure["transaction_snapshot_stored"] = True
                        except BaseException as snapshot_exc:
                            failure["transaction_snapshot_error"] = f"{type(snapshot_exc).__name__}: {snapshot_exc}"
                    _atomic_json(output_dir / "failure.json", failure)
                    metrics.write(json.dumps({"arm": arm, "seed": cfg["seed"], "event": "run_failure",
                                              "failure": failure}, sort_keys=True) + "\n")
                    raise
                finally:
                    del context, wake
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        sealed = {"schema": "clinc-v8-controlled-transfer-seal", "complete": len(receipts) == 2,
            "runs": receipts, "arms_expected": list(TARGET_ARMS), "arms_completed": len(receipts),
            "source_arm": "HOPE", "source_context_sha256": common_source_context_sha,
            "schedule_and_capacity_plan": probe_plan,
            "training_manifest_sha256": cfg["training_manifest_sha256"],
            "source_lock_sha256": source_lock, "parent_task0_checkpoint_sha256": parent_cp_sha,
            "test_data_read": False, "work": budget.as_dict(),
            "elapsed_seconds": time.monotonic() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "immutable_base_unchanged": immutable_versions(model) == frozen_versions,
            "claim_scope": f"development-only seed17 task1 transfer-budget probe k={cfg['inner_cycles_per_event']}; no final test access"}
        if not sealed["immutable_base_unchanged"]:
            raise RuntimeError("frozen Llama backbone changed during controlled probe")
        _atomic_json(output_dir / "development-sealed.json", sealed)
        return sealed
    except BaseException as exc:
        if not (output_dir / "failure.json").exists():
            _atomic_json(output_dir / "failure.json", {"schema": "clinc-v8-controlled-transfer-failure",
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                "elapsed_seconds": time.monotonic() - started, "work": budget.as_dict(),
                "test_data_read": False, "automatic_retry": False})
        raise


def _read_run_event_objects(path, event_name, arm):
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if item.get("event") == event_name and item.get("arm") == arm:
                    rows.append(item)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-pack", required=True)
    parser.add_argument("--warmup-run-dir", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"status": "complete", "arms_completed": result["arms_completed"],
                      "output_dir": str(Path(args.output_dir).resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
