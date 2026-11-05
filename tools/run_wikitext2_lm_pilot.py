"""Fixed-budget Llama 3.2-3B LM pilot on article-disjoint WikiText train windows."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from nested_sleep.accounting import ACTIVE_METER, WorkMeter
from nested_sleep.lifecycle import SleepSession, fingerprint
from nested_sleep.lm import LlamaCMSLM, promote_mutable_to_fp32
from nested_sleep.optim import CMSController


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_local_implementation_sources() -> dict:
    source_root = (ROOT.parent / "src").resolve()
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        if name == "nested_sleep" or name.startswith("nested_sleep."):
            module_file = getattr(module, "__file__", None)
            if not module_file:
                continue
            path = Path(module_file).resolve()
            if not path.is_relative_to(source_root):
                raise RuntimeError(f"refusing to run with external {name} implementation: {path}")
            loaded[name] = {"path": str(path), "sha256": sha256(path)}
    required = {"nested_sleep.accounting", "nested_sleep.lifecycle", "nested_sleep.lm", "nested_sleep.optim"}
    missing = sorted(required - set(loaded))
    if missing:
        raise RuntimeError(f"required public source modules were not loaded: {missing}")
    return loaded


IMPLEMENTATION_SOURCE_HASHES = verify_local_implementation_sources()


def json_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def read_examples(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["examples"]


def encode_examples(tokenizer, examples: list[dict], context_tokens: int) -> list[torch.Tensor]:
    rows = []
    for row in examples:
        ids = tokenizer.encode(row["text"], add_special_tokens=True)
        if len(ids) < context_tokens:
            raise ValueError(f"article window {row['article_id_sha256']} has {len(ids)} tokens; need {context_tokens}")
        rows.append(torch.tensor(ids[:context_tokens], dtype=torch.long))
    return rows


def load_model(model_dir: Path, seed: int, config: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    backbone = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=config["attention_backend"],
    ).eval()
    model = LlamaCMSLM(backbone, rank=config["rank"], capacity=config["capacity"], levels=config["levels"])
    promote_mutable_to_fp32(model)
    model.eval().requires_grad_(False)
    return model, tokenizer


def adapter_snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.adapter_state_dict().items()}


def restore_adapter(model, state):
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise RuntimeError("unexpected adapter keys: " + ",".join(unexpected))
    model.eval().requires_grad_(False)


def save_adapter(model, path: Path, seed: int, method: str):
    torch.save({"adapter": adapter_snapshot(model), "seed": seed, "method": method,
                "model_config": model.export_config()}, path)


def load_adapter(model, path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    restore_adapter(model, obj["adapter"])
    return obj


def next_token_loss(model, sequence: torch.Tensor) -> tuple[torch.Tensor, int]:
    device = next(model.parameters()).device
    sequence = sequence.to(device).unsqueeze(0)
    logits = model.logits(sequence[:, :-1])
    target = sequence[:, 1:]
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))
    return loss, int(target.numel())


def apply_proposals(model, controller, proposals, *, levels=None):
    selected = sorted(proposals) if levels is None else [x for x in levels if x in proposals]
    for level in selected:
        controller.apply(model, level, proposals[level])


def wake(model, sequences, config, method: str, meter: WorkMeter):
    periods = [1, 1] if method == "sequential_sft" else config["cms_periods"]
    controller = CMSController(model, periods=periods, base_lr=config["base_lr"], expert_lr=config["expert_lr"])
    events = []
    session_info = None
    live = model
    for step, sequence in enumerate(sequences, start=1):
        controller.prepare_wake(model)
        meter.stage = "wake"
        loss, supervised = next_token_loss(model, sequence)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite wake loss at step {step}")
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        proposals = controller.collect(model, sequential=(method == "sequential_sft"))
        if method == "sequential_sft":
            apply_proposals(model, controller, proposals)
        elif step == config["boundary_steps"][0]:
            if set(proposals) != {0}:
                raise RuntimeError(f"expected only level-0 proposal at step {step}; got {sorted(proposals)}")
            apply_proposals(model, controller, proposals)
        elif step == config["wake_steps"]:
            if set(proposals) != {0}:
                raise RuntimeError(f"expected only level-0 terminal proposal at step {step}; got {sorted(proposals)}")
            if method == "cms_wake_ks_gkd_lti":
                # The proposal is prospective: pass it to SleepSession before applying it.
                prefix = sequence[:config["sleep_prefix_tokens"]].to(next(model.parameters()).device).unsqueeze(0)
                meter.stage = "sleep_gkd"
                session = SleepSession(
                    model, 0, proposals[0], seed=config["sleep_seed_base"] + step,
                    lr=config["sleep_lr"], dream_tokens=config["dream_tokens"],
                    share_backbone=True, reject_zero_gradient=True,
                )
                before_base = fingerprint(model)
                for _ in range(config["gkd_steps"]):
                    session.seed_step(prefix)
                meter.stage = "sleep_lti"
                for _ in range(config["lti_steps"]):
                    session.lti_step(prefix)
                session.finish_consolidation()
                restored = session.commit()
                if fingerprint(live) != before_base:
                    raise RuntimeError("live sender changed before candidate commit")
                model = restored
                controller.consume(0)
                session_info = {"status": "committed", "sender": 0, "receiver": 1,
                                "records": session.records, "work": session.work,
                                "reset_slots": session.reset_slots, "receiver_slot": session.slot,
                                "frozen_backbone_shared": True,
                                "terminal_level_pending": controller.counts[1],
                                "terminal_level_period": controller.periods[1]}
                del session
            else:
                apply_proposals(model, controller, proposals)
        else:
            if proposals:
                raise RuntimeError(f"unexpected CMS due boundary at step {step}: {sorted(proposals)}")
        events.append({"step": step, "loss": float(loss.detach()), "supervised_tokens": supervised,
                       "due_levels": sorted(proposals), "pending_counts": list(controller.counts),
                       "method": method})
    if method != "sequential_sft" and controller.counts != [0, config["wake_steps"]]:
        raise RuntimeError(f"unexpected final clock state: {controller.counts}")
    model.eval().requires_grad_(False)
    return model, controller, events, session_info


def compact_sleep_state(session):
    state = session.state_dict()
    state["teacher"] = session.teacher.adapter_state_dict()
    state["student"] = session.student.adapter_state_dict()
    # state_dict() exposes optimizer tensors by reference. Isolate every tensor
    # on CPU before either restore branch can advance Adam's moments in-place.
    def cpu_clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu").clone()
        if isinstance(value, dict):
            return {key: cpu_clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_clone(item) for item in value)
        return value
    return cpu_clone(state)


def tensor_tree_hash(value):
    h = hashlib.sha256()
    def visit(obj, prefix=""):
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu().contiguous()
            h.update(prefix.encode()); h.update(str((tuple(t.shape), t.dtype)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str): visit(obj[key], prefix + "/" + str(key))
        elif isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj): visit(item, prefix + f"/{index}")
        else:
            h.update((prefix + "=" + repr(obj)).encode())
    visit(value)
    return h.hexdigest()


def instrument_optimizer(session):
    trace = {}
    original_step = session.optimizer.step
    def step(*args, **kwargs):
        trace["optimizer_state_before"] = tensor_tree_hash(session.optimizer.state_dict())
        trace["gradients_before"] = tensor_tree_hash([p.grad for p in (session.new_slot.a, session.new_slot.b)])
        original_step(*args, **kwargs)
    session.optimizer.step = step
    return trace


def checkpoint_step_aliases(session, checkpoint):
    aliases = []
    group = checkpoint["optimizer"]["param_groups"][0]
    local = session.optimizer.param_groups[0]["params"]
    for saved_id, parameter in zip(group["params"], local):
        saved_step = checkpoint["optimizer"]["state"][saved_id].get("step")
        live_step = session.optimizer.state[parameter].get("step")
        aliases.append({
            "saved_step": None if saved_step is None else float(saved_step),
            "restored_step": None if live_step is None else float(live_step),
            "same_object": saved_step is live_step,
            "same_storage": bool(saved_step is not None and live_step is not None and saved_step.device == live_step.device
                                  and saved_step.data_ptr() == live_step.data_ptr()),
            "saved_device": None if saved_step is None else str(saved_step.device),
            "restored_device": None if live_step is None else str(live_step.device),
        })
    return aliases


def recovery_fixture(model, tokenizer, adaptation_examples, sequences, config, out_dir: Path, seed: int):
    """Verify a compact SleepSession checkpoint resumes with identical next update."""
    model.eval().requires_grad_(False)
    controller = CMSController(model, periods=config["cms_periods"], base_lr=config["base_lr"],
                               expert_lr=config["expert_lr"])
    meter = WorkMeter()
    token = ACTIVE_METER.set(meter)
    started = time.perf_counter()
    proposal = None
    for sequence in sequences[:config["boundary_steps"][0]]:
        controller.prepare_wake(model)
        meter.stage = "fixture_wake"
        loss, _ = next_token_loss(model, sequence)
        loss.backward()
        proposals = controller.collect(model)
        if proposals:
            if set(proposals) != {0}:
                raise RuntimeError("unexpected simultaneous proposal in resume fixture")
            proposal = proposals[0]
    if proposal is None:
        raise RuntimeError("recovery fixture did not produce sender-0 proposal")
    prefix = sequences[0][:config["sleep_prefix_tokens"]].to(next(model.parameters()).device).unsqueeze(0)
    session = SleepSession(model, 0, proposal, seed=seed, lr=config["sleep_lr"],
                           dream_tokens=config["dream_tokens"], share_backbone=True, reject_zero_gradient=True)
    meter.stage = "fixture_sleep_first"
    session.seed_step(prefix)
    checkpoint_fingerprint = fingerprint(session.student)
    state_path = out_dir / f"correctness-resume-s{seed}.pt"
    compact = compact_sleep_state(session)
    torch.save(compact, state_path)
    saved = torch.load(state_path, map_location="cpu", weights_only=False)
    saved_optimizer_hash_before_restore = tensor_tree_hash(saved["optimizer"])
    reference = SleepSession.restore(model, saved, mutable_only=True)
    aliases_after_restore = checkpoint_step_aliases(reference, saved)
    restore_fingerprint = fingerprint(reference.student)
    restore_state_exact = restore_fingerprint == checkpoint_fingerprint
    saved_optimizer_hash = tensor_tree_hash(saved["optimizer"])
    reference_trace = instrument_optimizer(reference)
    reference.seed_step(prefix)
    saved_optimizer_hash_after_reference = tensor_tree_hash(saved["optimizer"])
    if saved_optimizer_hash_after_reference != saved_optimizer_hash:
        failure = {"status": "failed", "seed": seed,
                   "reason": "optimizer checkpoint snapshot mutated after restored branch update",
                   "saved_optimizer_hash_before_restore": saved_optimizer_hash_before_restore,
                   "saved_optimizer_hash_after_restore": saved_optimizer_hash,
                   "saved_optimizer_hash_after_reference": saved_optimizer_hash_after_reference,
                   "step_tensor_aliases_after_restore": aliases_after_restore,
                   "reference_optimizer_and_gradient": reference_trace,
                   "seconds": time.perf_counter() - started, "work": meter.counts,
                   "cuda_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        failure_path = out_dir / f"correctness-failure-s{seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError("recovery fixture detected optimizer checkpoint alias mutation")
    expected = fingerprint(reference.student)
    resumed = SleepSession.restore(model, saved, mutable_only=True)
    resumed_trace = instrument_optimizer(resumed)
    meter.stage = "fixture_sleep_resume"
    resumed.seed_step(prefix)
    actual = fingerprint(resumed.student)
    if expected != actual:
        expected_state = reference.student.adapter_state_dict()
        actual_state = resumed.student.adapter_state_dict()
        differences = []
        for name in expected_state:
            left, right = expected_state[name], actual_state[name]
            if left.is_floating_point():
                delta = float((left.float() - right.float()).abs().max())
                if delta:
                    differences.append((delta, name))
        largest = sorted(differences, reverse=True)[:5]
        failure = {
            "status": "failed",
            "seed": seed,
            "reason": "restored next GKD update did not match byte-identically",
            "restored_pre_step_exact": restore_state_exact,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "restored_pre_step_fingerprint": restore_fingerprint,
            "expected_next_fingerprint": expected,
            "actual_next_fingerprint": actual,
            "largest_float_deltas": largest,
            "expected_update": reference.records[-1],
            "actual_update": resumed.records[-1],
            "reference_optimizer_and_gradient": reference_trace,
            "resumed_optimizer_and_gradient": resumed_trace,
            "generated_teacher_sample_sha256": hashlib.sha256(reference.teacher_data[-1].cpu().numpy().tobytes()).hexdigest(),
            "resumed_teacher_sample_sha256": hashlib.sha256(resumed.teacher_data[-1].cpu().numpy().tobytes()).hexdigest(),
            "saved_optimizer_hash_before_reference": saved_optimizer_hash,
            "saved_optimizer_hash_after_reference": saved_optimizer_hash_after_reference,
            "seconds": time.perf_counter() - started,
            "work": meter.counts,
            "cuda_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "implementation_source_hashes": IMPLEMENTATION_SOURCE_HASHES,
        }
        failure_path = out_dir / f"correctness-failure-s{seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError("SleepSession checkpoint resume diverged on next GKD step; "
                           f"restored_pre_step_exact={restore_state_exact}; largest_float_deltas={largest}; "
                           f"failure_artifact={failure_path.name}")
    if fingerprint(model) != session.before:
        raise RuntimeError("recovery fixture mutated the live model")
    report = {"status": "passed", "seed": seed, "gkd_steps_before_checkpoint": 1,
              "sleep_session_source": inspect.getfile(SleepSession),
              "implementation_source_hashes": IMPLEMENTATION_SOURCE_HASHES,
              "resumed_next_gkd_step_fingerprint": actual, "byte_identical_resume": True,
              "restored_pre_step_exact": restore_state_exact,
              "checkpoint_bytes": state_path.stat().st_size,
              "seconds": time.perf_counter() - started, "work": meter.counts,
              "heldout_roles_read": False}
    ACTIVE_METER.reset(token)
    model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    return report


def run_train(config_path: Path, model_dir: Path, data_dir: Path, out_dir: Path, model_manifest_path: Path | None):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg_hash = json_hash(config)
    data_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if data_manifest["source_file_sha256"] != config["dataset"]["source_file_sha256"]:
        raise RuntimeError("prepared data source SHA differs from protocol")
    selected_ids = [v for ids in data_manifest["selected_article_ids"].values() for v in ids]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("article IDs overlap between prepared roles")
    expected_roles = {"adaptation": config["data_split"]["adaptation_examples"],
                      "calibration": config["data_split"]["calibration_examples"],
                      "evaluation": config["data_split"]["evaluation_examples"]}
    if {k: len(v) for k, v in data_manifest["selected_article_ids"].items()} != expected_roles:
        raise RuntimeError("prepared role counts differ from protocol")
    if sha256(data_dir / "adaptation.json") != data_manifest["cache_sha256"]["adaptation"]:
        raise RuntimeError("adaptation cache hash mismatch")
    if (out_dir / "training-sealed.json").exists():
        raise RuntimeError("refusing to overwrite a sealed run directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    adaptation = read_examples(data_dir / "adaptation.json")
    if len(adaptation) != config["data_split"]["adaptation_examples"]:
        raise RuntimeError("adaptation window count differs from protocol")
    manifest_path = model_manifest_path
    if manifest_path is None:
        candidate = config_path.parent / "model-asset-manifest.json"
        manifest_path = candidate if candidate.exists() else None
    if manifest_path is None:
        manifest_path = ROOT.parent / "Nested_Learning_and_Sleep" / "evidence" / "phase2" / "llama32-3b-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("pass or provide the local Llama asset manifest")
    manifest_digest = sha256(manifest_path)
    asset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if asset_manifest.get("provider", "").casefold() != config["model"]["source"].casefold() or \
            asset_manifest.get("id") != config["model"]["model_id"]:
        raise RuntimeError("model asset manifest does not match the frozen model source/id")
    asset_checks = []
    for item in asset_manifest["files"]:
        asset_path = model_dir / item["path"]
        if not asset_path.is_file():
            raise FileNotFoundError(f"model asset missing: {item['path']}")
        actual = sha256(asset_path)
        if actual != item["sha256"]:
            raise RuntimeError(f"model asset SHA mismatch: {item['path']}")
        asset_checks.append({"path": item["path"], "sha256": actual, "bytes": asset_path.stat().st_size})
    header = {"schema": "nl-sleep-wikitext2-lm-run-v1", "config_sha256": cfg_hash,
              "config": config, "data_manifest_sha256": sha256(data_dir / "manifest.json"),
              "implementation_source_hashes": IMPLEMENTATION_SOURCE_HASHES,
              "runner_sha256": sha256(Path(__file__).resolve()),
              "model_asset_manifest_sha256": manifest_digest,
              "model_asset_checks": asset_checks,
              "source_text_is_not_committed": True,
              "evaluation_file_read_during_training": False,
              "started_unix": time.time()}
    (out_dir / "manifest.json").write_text(json.dumps(header, indent=2), encoding="utf-8")
    all_results = []
    correctness = None
    for seed in config["training"]["seeds"]:
        model, tokenizer = load_model(model_dir, seed, config["model"])
        initial = adapter_snapshot(model)
        if correctness is None:
            sample_sequences = encode_examples(tokenizer, adaptation[:config["training"]["wake_steps"]],
                                               config["training"]["context_tokens"])
            correctness = recovery_fixture(model, tokenizer, adaptation, sample_sequences,
                                           config["training"], out_dir, seed)
            restore_adapter(model, initial)
        init_path = out_dir / f"initial-s{seed}.pt"
        torch.save({"adapter": initial, "seed": seed, "method": "frozen"}, init_path)
        sequences = encode_examples(tokenizer, adaptation, config["training"]["context_tokens"])
        if len(sequences) != config["training"]["wake_steps"]:
            raise RuntimeError("adaptation samples do not match fixed wake steps")
        order = list(range(len(sequences)))
        random.Random(seed).shuffle(order)
        sequences = [sequences[i] for i in order]
        for method in config["training"]["methods"]:
            run = {"seed": seed, "method": method, "status": "started"}
            started = time.perf_counter()
            meter = WorkMeter()
            meter_token = ACTIVE_METER.set(meter)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                event_start.record()
            try:
                restore_adapter(model, initial)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                if method == "frozen":
                    events, session_info, controller = [], None, None
                else:
                    model, controller, events, session_info = wake(model, sequences, config["training"], method, meter)
                path = out_dir / f"{method}-s{seed}.pt"
                save_adapter(model, path, seed, method)
                if torch.cuda.is_available():
                    event_end.record()
                    torch.cuda.synchronize()
                run.update({"status": "completed", "seconds": time.perf_counter() - started,
                            "wake_events": events, "sleep_session": session_info,
                            "training_sample_order": order if method != "frozen" else [],
                            "cms_state": None if controller is None else {
                                "periods": list(controller.periods), "counts": list(controller.counts),
                                "events": controller.events,
                                "pending_scaled_gradient_l2": [
                                    math.sqrt(sum(float(v.float().square().sum()) for v in row.values()))
                                    for row in controller.pending]},
                            "work": meter.counts, "cuda_event_interval_ms": event_start.elapsed_time(event_end)
                                if torch.cuda.is_available() else None,
                            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                                if torch.cuda.is_available() else None,
                            "checkpoint": path.name})
            except Exception as exc:
                run.update({"status": "failed", "seconds": time.perf_counter() - started,
                            "error_type": type(exc).__name__, "error": str(exc), "work": meter.counts})
                # Keep each failure and proceed to the next fixed arm.
                if method == "cms_wake_ks_gkd_lti":
                    run["sleep_session"] = {"status": "failed_before_commit"}
            finally:
                model.zero_grad(set_to_none=True)
                model.eval().requires_grad_(False)
                ACTIVE_METER.reset(meter_token)
            all_results.append(run)
            print(json.dumps({"seed": seed, "method": method, "status": run["status"],
                              "seconds": round(run["seconds"], 2)}), flush=True)
        del tokenizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (out_dir / "correctness-resume.json").write_text(json.dumps(correctness, indent=2), encoding="utf-8")
    (out_dir / "training-results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    sealed = {"schema": "nl-sleep-wikitext2-lm-training-sealed-v1", "config_sha256": cfg_hash,
              "sealed_unix": time.time(), "correctness_fixture": correctness["status"],
              "all_seed_method_statuses": [{k: r[k] for k in ("seed", "method", "status")} for r in all_results],
              "evaluation_file_read_during_training": False,
              "all_adapters_written": all(r["status"] == "completed" for r in all_results)}
    (out_dir / "training-sealed.json").write_text(json.dumps(sealed, indent=2), encoding="utf-8")


def run_evaluate(config_path: Path, model_dir: Path, data_dir: Path, out_dir: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg_hash = json_hash(config)
    sealed_path = out_dir / "training-sealed.json"
    if not sealed_path.exists():
        raise RuntimeError("evaluation requires a sealed training run")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    if sealed["config_sha256"] != cfg_hash:
        raise RuntimeError("config differs from training protocol")
    run_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256(data_dir / "evaluation.json") != data_manifest["cache_sha256"]["evaluation"]:
        raise RuntimeError("evaluation cache hash mismatch")
    if sha256(data_dir / "manifest.json") != run_manifest["data_manifest_sha256"]:
        raise RuntimeError("data manifest differs from the frozen training run")
    # This is the first point where the evaluation-role text file is opened.
    evaluation = read_examples(data_dir / "evaluation.json")
    results = []
    for seed in config["training"]["seeds"]:
        model, tokenizer = load_model(model_dir, seed, config["model"])
        eval_sequences = encode_examples(tokenizer, evaluation, config["training"]["context_tokens"])
        for method in config["training"]["methods"]:
            checkpoint = out_dir / (f"initial-s{seed}.pt" if method == "frozen" else f"{method}-s{seed}.pt")
            status = next((r["status"] for r in json.loads((out_dir / "training-results.json").read_text(encoding="utf-8"))
                           if r["seed"] == seed and r["method"] == method), "missing")
            if status != "completed" or not checkpoint.exists():
                results.append({"seed": seed, "method": method, "status": "unavailable",
                                "reason": f"training status {status}; checkpoint missing or failed"})
                continue
            obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
            restore_adapter(model, obj["adapter"])
            meter = WorkMeter()
            meter.stage = "heldout_evaluation"
            meter_token = ACTIVE_METER.set(meter)
            started = time.perf_counter()
            losses, token_count = [], 0
            try:
                with torch.no_grad():
                    for sequence in eval_sequences:
                        loss, n = next_token_loss(model, sequence)
                        losses.append(float(loss))
                        token_count += n
            finally:
                ACTIVE_METER.reset(meter_token)
            nll = sum(loss * (config["training"]["context_tokens"] - 1) for loss in losses) / token_count
            results.append({"seed": seed, "method": method, "status": "completed",
                            "examples": len(eval_sequences), "scored_tokens": token_count,
                            "nll_nats_per_token": nll, "perplexity": math.exp(min(nll, 50)),
                            "work": meter.counts, "seconds": time.perf_counter() - started})
            print(json.dumps({"seed": seed, "method": method, "nll": round(nll, 5)}, flush=True))
        del tokenizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (out_dir / "evaluation-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path)
    args = parser.parse_args()
    # Make the compact resume fixture meaningful on CUDA: SDPA/gradient reduction
    # nondeterminism can otherwise be amplified by Adam despite exact restored state.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    if args.stage == "train":
        run_train(args.config, args.model_dir, args.data_dir, args.out_dir, args.model_manifest)
    else:
        run_evaluate(args.config, args.model_dir, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
