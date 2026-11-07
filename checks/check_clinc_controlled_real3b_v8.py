#!/usr/bin/env python3
"""Real 3B qualification of controlled-route Sleep snapshot replay.

This is an engineering ablation fixture with two hard-coded manual prompts,
not a CLINC/data run and not a paper algorithm claim. It checks the controlled
0.5 routing intervention and exactly eight GKD AdamW steps: two before one
snapshot, three uninterrupted reference steps, then three restored replays.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


MAX_ADAMW_CALLS = 8
MAX_BACKBONE_FORWARDS = 64
RANK, LEVELS, CAPACITY = 64, 5, 3
SLEEP_SEED = 8122026
PERIODS = (2, 3, 5, 7, None)
PROMPTS = ("Please check the balance in my savings account.",
           "I would like to book a table for two tonight.")
LEDGER: dict[str, Any] = {"phase": "startup", "adamw_calls": 0,
                          "backbone_forwards": 0, "reference_steps": 0,
                          "replay_steps": 0}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load qualification helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_value(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(obj):
        import numpy as np
        import torch
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, np.ndarray):
            digest.update(str((obj.dtype, obj.shape)).encode())
            digest.update(np.ascontiguousarray(obj).tobytes())
        elif is_dataclass(obj):
            visit(asdict(obj))
        elif isinstance(obj, dict):
            for key in sorted(obj, key=lambda item: repr(item)):
                digest.update(repr(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            digest.update(type(obj).__name__.encode())
            for child in obj:
                visit(child)
        elif obj is None or isinstance(obj, (str, int, float, bool, bytes)):
            digest.update(repr(obj).encode())
        else:
            raise TypeError(f"unsupported audit value: {type(obj)!r}")

    visit(value)
    return digest.hexdigest()


def _equal(left: Any, right: Any) -> bool:
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if is_dataclass(left):
        return is_dataclass(right) and _equal(asdict(left), asdict(right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right))
    return left == right


class ForwardMeter:
    def __init__(self, backbone):
        self.count = 0
        self.handle = backbone.register_forward_pre_hook(self._hook)

    def _hook(self, _module, _inputs):
        if self.count >= MAX_BACKBONE_FORWARDS:
            raise RuntimeError(f"backbone-forward cap {MAX_BACKBONE_FORWARDS} reached before call")
        self.count += 1
        LEDGER["backbone_forwards"] = self.count

    def close(self):
        self.handle.remove()


def _load_sources(project_root: Path, task_root: Path):
    src, tools = (project_root / "src").resolve(), (task_root / "tools").resolve()
    if not src.is_dir() or not tools.is_dir():
        raise FileNotFoundError("expected public src/ and task tools/ roots")
    for item in (str(task_root.resolve()), str(tools), str(src)):
        if item in sys.path:
            sys.path.remove(item)
        sys.path.insert(0, item)
    from nested_sleep.lm import LlamaCMSLM, immutable_versions, promote_mutable_to_fp32
    from clinc_lifecycle_v8 import SleepTransaction, WakeController
    import clinc_tail_runtime_v8 as tail

    life_path = Path(sys.modules["clinc_lifecycle_v8"].__file__).resolve()
    lm_path = Path(sys.modules["nested_sleep.lm"].__file__).resolve()
    tail_path = Path(tail.__file__).resolve()
    if src not in lm_path.parents or tools not in life_path.parents or tools not in tail_path.parents:
        raise RuntimeError("base runtime modules did not load from the pinned source roots")

    # Keep any package-qualified lookup aliased to the same canonical flat
    # lifecycle module used by checkpoint pickles and the controlled helper.
    sys.modules["tools.clinc_lifecycle_v8"] = sys.modules["clinc_lifecycle_v8"]
    controlled = importlib.import_module("tools.clinc_controlled_routing_v8")
    controlled_path = Path(controlled.__file__).resolve()
    if tools not in controlled_path.parents:
        raise RuntimeError("controlled routing helper loaded outside task tools")
    if controlled.SleepTransaction is not SleepTransaction or controlled.WakeController is not WakeController:
        raise RuntimeError("controlled lifecycle does not share the pinned natural lifecycle classes")
    source_paths = {"lm": lm_path, "experts": lm_path.with_name("experts.py"),
                    "natural_lifecycle": life_path, "tail_runtime": tail_path,
                    "controlled_routing": controlled_path,
                    "resume_reference": Path(__file__).resolve().with_name("check_clinc_real3b_resume_v8.py")}
    if not all(path.is_file() for path in source_paths.values()):
        raise FileNotFoundError("one or more pinned source files are missing")
    return {"LlamaCMSLM": LlamaCMSLM, "immutable_versions": immutable_versions,
            "promote_mutable_to_fp32": promote_mutable_to_fp32,
            "SleepTransaction": SleepTransaction, "WakeController": WakeController,
            "ControlledSleepTransaction": controlled.ControlledSleepTransaction,
            "ControlledWakeController": controlled.ControlledWakeController,
            "tail": tail, "src": src, "tools": tools, "source_paths": source_paths}


def _rng_bundle(torch, np):
    import random as random_module
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
            "python": deepcopy(random_module.getstate()),
            "numpy": deepcopy(np.random.get_state())}


def _controlled_signature(tx, torch) -> dict:
    snapshot = tx.snapshot()
    payload = snapshot.payload
    return {"controlled_snapshot": _hash_value(snapshot),
            "student": _hash_value(tx.student.state_dict()),
            "teacher": _hash_value(tx.teacher.state_dict()),
            "optimizer": _hash_value(tx.optimizer.state_dict()),
            "private_generator": _hash_value(tx.generator.get_state()),
            "counters": _hash_value(tx.counters),
            "teacher_bank_state": _hash_value(payload.bank_states["teacher"]),
            "student_bank_state": _hash_value(payload.bank_states["student"]),
            "teacher_routes": _hash_value(snapshot.teacher_routes),
            "student_routes": _hash_value(snapshot.student_routes),
            "rng": _hash_value(_rng_bundle(torch, __import__("numpy")))}


def _route_policies(context):
    return [deepcopy(bank.route_override) for bank in context.banks()]


def _check_bank_mix_semantics(bank, selected_slot: int, hidden_size: int, device, torch):
    """Check ExpertBank's real-device output is the configured 0.5 convex mix."""
    original_policy = deepcopy(bank.route_override)
    original_routes = deepcopy(bank.last_routes)
    original_audit = bank.last_route_audit
    original_history = list(bank.audit_history)
    values = torch.linspace(-0.1, 0.1, steps=hidden_size, device=device, dtype=torch.float32)
    probe = values.reshape(1, hidden_size)
    try:
        bank.route_override = None
        natural = bank(probe)
        selected = bank.slots[selected_slot](probe)
        bank.route_override = {"mix": selected_slot, "fraction": .5}
        observed = bank(probe)
        expected = .5 * natural + .5 * selected
        difference = float((observed - expected).abs().max().item())
        torch.testing.assert_close(observed, expected, rtol=0., atol=1e-6)
        return {"selected_slot": selected_slot, "fraction": .5,
                "max_abs_error": difference, "passed": True}
    finally:
        bank.route_override = original_policy
        bank.last_routes = original_routes
        bank.last_route_audit = original_audit
        bank.audit_history = original_history


def run(args) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    task_root = Path(__file__).resolve().parents[1]
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output_dir}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model assets missing: {model_path}")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("this qualification requires an available CUDA device")
    output_dir.mkdir(parents=True, exist_ok=True)
    LEDGER.update({"phase": "source-preflight", "adamw_calls": 0,
                   "backbone_forwards": 0, "reference_steps": 0, "replay_steps": 0})
    source = _load_sources(project_root, task_root)
    source_hashes = {name: _sha(path) for name, path in source["source_paths"].items()}
    source_hashes["qualification_script"] = _sha(Path(__file__).resolve())
    source_hashes["reference_script"] = _sha(Path(__file__).resolve().with_name("check_clinc_real3b_resume_v8.py"))
    LEDGER["source_hashes"] = source_hashes
    LEDGER["model_path"] = str(model_path)

    torch.manual_seed(20261107)
    torch.cuda.manual_seed_all(20261107)
    random.seed(20261107)
    np.random.seed(20261107)
    LEDGER["phase"] = "model-load"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map=args.device, attn_implementation="eager").eval()
    backbone.requires_grad_(False)
    model = source["LlamaCMSLM"](backbone, rank=RANK, capacity=CAPACITY, levels=LEVELS)
    source["promote_mutable_to_fp32"](model)
    model.eval().requires_grad_(False)
    if (model.rank, model.capacity, model.levels) != (RANK, CAPACITY, LEVELS):
        raise AssertionError("loaded model topology differs from the qualification lock")
    frozen_versions = source["immutable_versions"](model)
    if not frozen_versions:
        raise AssertionError("immutable backbone guard has empty scope")

    # Two short manual user requests are the only input strings in this check.
    prompt_ids = []
    for text in PROMPTS:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=False)
        ids = encoded["input_ids"].to(args.device)
        if not 1 <= ids.shape[1] <= 96:
            raise RuntimeError("manual prompt exceeds the short sequence bound")
        prompt_ids.append(ids)

    LEDGER["phase"] = "controlled-context-setup"
    live = source["tail"].clone_mutable_context(model)
    live.banks()[0].activate(0)
    live.banks()[1].activate(0)
    live.banks()[1].route_override = {"mix": 0, "fraction": .5}
    live_policy_before = _route_policies(live)
    controller = source["ControlledWakeController"](
        live, periods=PERIODS, base_lr=.007, fast_lr=.003)
    controller.prepare_wake()
    if _route_policies(live) != live_policy_before:
        raise AssertionError("controlled wake changed inherited route policy")

    sender = 0
    prospective = {f"memory.{sender}.a": live.memory[sender].a.detach().clone(),
                   f"memory.{sender}.b": live.memory[sender].b.detach().clone()}
    proposal_gen = torch.Generator(device="cpu").manual_seed(SLEEP_SEED + 1)
    delta = torch.randn(prospective[f"memory.{sender}.b"].shape, generator=proposal_gen)
    delta = (delta / delta.abs().max().clamp_min(1e-12) * 1e-3).to(prospective[f"memory.{sender}.b"])
    prospective[f"memory.{sender}.b"] += delta

    tx = source["ControlledSleepTransaction"](live, sender, prospective,
        lr=.001, weight_decay=0., clip_norm=1., seed=SLEEP_SEED)
    if tx.teacher.banks()[1].route_override != {"mix": 0, "fraction": .5}:
        raise AssertionError("teacher did not preserve prior bank-1 mixed route")
    if tx.student.banks()[1].route_override != {"mix": 1, "fraction": .5}:
        raise AssertionError("student did not route a 0.5 mixture to fresh receiver slot 1")
    if tx.student.banks()[0].route_override is not None:
        raise AssertionError("controlled sender bank was not returned to natural routing")
    if _route_policies(live) != live_policy_before:
        raise AssertionError("Sleep transaction modified its live source route policy")
    if tx.slot != 1 or tx.sender != 0 or tx.receiver != 1:
        raise AssertionError("controlled sender/receiver identity mismatch")
    pre_snapshot = tx.snapshot()
    if pre_snapshot.payload.metadata.get("sender_reset_policy") != "redraw_A_normal_0_0.02_zero_B":
        raise AssertionError("controlled snapshot omitted the locked sender reset policy")
    route_semantics = {
        "teacher_bank1": _check_bank_mix_semantics(
            tx.teacher.banks()[1], 0, int(model.backbone.config.hidden_size), args.device, torch),
        "student_bank1": _check_bank_mix_semantics(
            tx.student.banks()[1], 1, int(model.backbone.config.hidden_size), args.device, torch),
    }

    meter = ForwardMeter(model.backbone)
    helper = load_module(Path(__file__).resolve().with_name("check_clinc_real3b_resume_v8.py"),
                         "natural_resume_helpers_for_controlled_check")
    ledger = {"adamw_calls": 0}

    def apply_adam_counter(transaction):
        step = transaction.optimizer.step
        def counted(*step_args, **step_kwargs):
            if LEDGER["adamw_calls"] >= MAX_ADAMW_CALLS:
                raise RuntimeError("controlled qualification exceeded the eight-AdamW cap")
            result = step(*step_args, **step_kwargs)
            LEDGER["adamw_calls"] += 1
            ledger["adamw_calls"] += 1
            return result
        transaction.optimizer.step = counted

    # The helper's GKD path performs a new full-model capture for every actual
    # sampled suffix and checks that the fresh receiver B gradient/update is >0.
    if helper.MAX_ADAMW_CALLS != MAX_ADAMW_CALLS:
        raise AssertionError("shared engineering fixture has a different AdamW budget")
    apply_adam_counter(tx)
    live_hash = _hash_value(live.state_dict())
    teacher_hash = _hash_value(tx.teacher.state_dict())
    live_policy_hash = _hash_value(_route_policies(live))
    reference_rows = []
    try:
        LEDGER["phase"] = "controlled-pre-snapshot"
        for index in range(2):
            owner = "teacher" if index == 0 else "student"
            reference_rows.append(helper._one_gkd(tx, model, source["tail"], prompt_ids[index],
                                                   owner=owner, ledger=ledger))
        if tx.counters["adamw_steps"] != 2 or not tx.optimizer.state:
            raise AssertionError("controlled pre-snapshot did not populate Adam moments")
        if any(not {"step", "exp_avg", "exp_avg_sq"}.issubset(values)
               for values in tx.optimizer.state.values()):
            raise AssertionError("controlled optimizer state lacks Adam step/moments")

        snapshot = tx.snapshot()
        controlled_snapshot_type = type(snapshot)
        if not hasattr(snapshot, "payload") or not hasattr(snapshot, "controlled_mix_schema"):
            raise AssertionError("controlled snapshot wrapper was not returned")
        if snapshot.controlled_mix_schema != source["ControlledSleepTransaction"].MIX_SCHEMA:
            raise AssertionError("controlled snapshot schema mismatch")
        if snapshot.payload.metadata.get("live_hash") != tx._live_hash:
            raise AssertionError("controlled wrapper payload changed live origin")
        snap_digest = _hash_value(snapshot)
        optimizer_digest = _hash_value(snapshot.payload.optimizer_state)
        disk_path = output_dir / "controlled-step02-snapshot.pt"
        partial = output_dir / "controlled-step02-snapshot.pt.partial"
        torch.save(snapshot, partial)
        with partial.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        disk_sha = _sha(partial)
        os.replace(partial, disk_path)
        if _sha(disk_path) != disk_sha:
            raise AssertionError("controlled snapshot file hash changed after write")
        LEDGER["snapshot_path"] = str(disk_path)
        LEDGER["snapshot_sha256"] = disk_sha
        disk_snapshot = torch.load(disk_path, map_location="cpu", weights_only=False)
        if type(disk_snapshot) is not controlled_snapshot_type or _hash_value(disk_snapshot) != snap_digest:
            raise AssertionError("controlled wrapper snapshot failed exact disk roundtrip")

        owners = ("teacher", "student", "teacher")
        prompt_schedule = tuple(prompt_ids[(i + 1) % len(prompt_ids)] for i in range(3))
        ref_results = []
        LEDGER["phase"] = "controlled-reference-three"
        for index, (owner, prompt) in enumerate(zip(owners, prompt_schedule)):
            result = helper._one_gkd(tx, model, source["tail"], prompt,
                                     owner=owner, ledger=ledger)
            ref_results.append(result)
            reference_rows.append(result)
            LEDGER["reference_steps"] += 1
        ref_signature = _controlled_signature(tx, torch)
        ref_rng = helper._rng_bundle(torch, __import__("numpy"), random)
        ref_live_hash = _hash_value(tx.live.state_dict())
        ref_teacher_hash = _hash_value(tx.teacher.state_dict())
        ref_student_hash = _hash_value(tx.student.state_dict())
        ref_policies = {"live": _route_policies(tx.live), "teacher": _route_policies(tx.teacher),
                        "student": _route_policies(tx.student)}
        if _hash_value(ref_policies["live"]) != live_policy_hash:
            raise AssertionError("live routing policy changed during uninterrupted branch")
        if _hash_value(snapshot) != snap_digest or _hash_value(snapshot.payload.optimizer_state) != optimizer_digest:
            raise AssertionError("reference updates aliased the saved controlled snapshot")

        random.random(); np.random.random(); torch.rand((), device="cpu"); torch.rand((), device="cuda")
        LEDGER["phase"] = "controlled-snapshot-restore"
        replay = source["ControlledSleepTransaction"].from_snapshot(live, disk_snapshot)
        apply_adam_counter(replay)
        if not _equal(helper._rng_bundle(torch, np, random), helper.snap_rng_bundle(snapshot.payload)):
            raise AssertionError("controlled restore failed to restore process RNG streams")
        if not torch.equal(replay.generator.get_state(), snapshot.payload.generator_state):
            raise AssertionError("controlled restore failed to restore its private generator")
        if replay.student.banks()[1].route_override != {"mix": 1, "fraction": .5}:
            raise AssertionError("controlled restore lost receiver mix route")
        if replay.teacher.banks()[1].route_override != {"mix": 0, "fraction": .5}:
            raise AssertionError("controlled restore lost teacher's inherited mix route")
        replay_rows = []
        LEDGER["phase"] = "controlled-replay-three"
        for index, (owner, prompt) in enumerate(zip(owners, prompt_schedule)):
            result = helper._one_gkd(replay, model, source["tail"], prompt,
                                     owner=owner, ledger=ledger)
            replay_rows.append(result)
            LEDGER["replay_steps"] += 1
            for field in ("sequence_sha256", "loss", "receiver_b_grad_norm", "receiver_b_delta_norm"):
                if result[field] != ref_results[index][field]:
                    raise AssertionError(f"controlled replay {index} differs in {field}")

        replay_signature = _controlled_signature(replay, torch)
        if replay_signature != ref_signature:
            raise AssertionError("controlled snapshot replay diverged in weights/Adam/RNG/policy")
        if _hash_value(replay.live.state_dict()) != ref_live_hash or ref_live_hash != live_hash:
            raise AssertionError("controlled replay mutated its shared live context")
        if _hash_value(replay.teacher.state_dict()) != ref_teacher_hash or ref_teacher_hash != teacher_hash:
            raise AssertionError("controlled replay mutated the frozen teacher")
        if _hash_value(replay.student.state_dict()) != ref_student_hash:
            raise AssertionError("controlled replay student differs from uninterrupted branch")
        if {"live": _route_policies(replay.live), "teacher": _route_policies(replay.teacher),
                "student": _route_policies(replay.student)} != ref_policies:
            raise AssertionError("controlled routing policies changed during replay")
        if not _equal(helper._rng_bundle(torch, np, random), ref_rng):
            raise AssertionError("global RNG streams diverged after controlled replay")
        if _hash_value(snapshot) != snap_digest or _hash_value(snapshot.payload.optimizer_state) != optimizer_digest:
            raise AssertionError("restored branch mutated the saved controlled snapshot")
        if source["immutable_versions"](model) != frozen_versions:
            raise AssertionError("frozen Llama backbone changed during controlled Sleep")
        if LEDGER["adamw_calls"] != MAX_ADAMW_CALLS or tx.counters["adamw_steps"] != 5 or replay.counters["adamw_steps"] != 5:
            raise AssertionError("controlled path did not execute exactly 2+3+3 actual AdamW updates")
        if meter.count > MAX_BACKBONE_FORWARDS:
            raise AssertionError("controlled path exceeded full-backbone forward budget")
    finally:
        meter.close()

    active_slots = [[sum(bool(slot.active) for slot in bank.slots) for bank in context.banks()]
                    for context in (live, tx.teacher, tx.student, replay.teacher, replay.student)]
    result = {"schema": "clinc-controlled-routing-real3b-v8-qualification-v1", "complete": True,
              "claim_scope": "controlled 0.5 route-mix engineering ablation; not natural Sleep or a paper algorithm",
              "model": {"path": str(model_path), "rank": RANK, "capacity": CAPACITY, "levels": LEVELS,
                        "backbone_dtype": "bfloat16", "mutable_dtype": "float32"},
              "source_hashes": source_hashes,
              "manual_prompt_hashes": [_hash_value(torch.tensor(tokenizer(text, add_special_tokens=True)["input_ids"])) for text in PROMPTS],
              "mix_policy": {"live_bank1": {"mix": 0, "fraction": .5},
                             "teacher_bank1": {"mix": 0, "fraction": .5},
                             "student_bank1": {"mix": 1, "fraction": .5},
                             "student_sender_bank0": None,
                             "teacher_retains_old_mixed_policy": True},
              "mix_semantics_checks": route_semantics,
              "active_compute_note": "The controlled 0.5 mix can evaluate the natural and selected expert branches (at most one additional selected branch per token; selected may equal natural). This is a routing ablation, not a FLOP-matched or paper-algorithm claim.",
              "active_slot_counts_live_teacher_student_replay": active_slots,
              "snapshot": {"path": str(disk_path), "sha256": disk_sha,
                           "wrapper_schema": snapshot.controlled_mix_schema,
                           "exact_wrapper_readback": True,
                           "payload_optimizer_sha256": optimizer_digest},
              "steps": {"pre_snapshot": 2, "reference_after_snapshot": 3,
                        "restored_replay": 3, "actual_adamw_calls": LEDGER["adamw_calls"],
                        "backbone_forwards": meter.count,
                        "forward_cap": MAX_BACKBONE_FORWARDS,
                        "receiver_b_grad_norms_reference": [x["receiver_b_grad_norm"] for x in ref_results],
                        "receiver_b_grad_norms_replay": [x["receiver_b_grad_norm"] for x in replay_rows]},
              "exact_resume": True, "teacher_live_frozen_and_policy_unchanged": True,
              "all_rng_streams_exact": True, "data_accessed": False,
              "elapsed_seconds": time.perf_counter() - started}
    (output_dir / "check-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8", newline="\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, help="public source checkout for nested_sleep.lm")
    parser.add_argument("--model-path", required=True, help="local Llama 3B asset directory")
    parser.add_argument("--output-dir", required=True, help="new private output directory")
    parser.add_argument("--device", required=True, help="CUDA device, e.g. cuda:0")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        result = run(args)
    except BaseException as exc:
        output_dir = Path(args.output_dir).resolve()
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        failure = {"schema": "clinc-controlled-routing-real3b-v8-qualification-failure-v1",
                   "complete": False, "phase": LEDGER.get("phase"),
                   "error_type": type(exc).__name__, "error": str(exc),
                   "adamw_calls": LEDGER.get("adamw_calls", 0),
                   "backbone_forwards": LEDGER.get("backbone_forwards", 0),
                   "ledger": dict(LEDGER), "elapsed_seconds": time.perf_counter() - started,
                   "data_accessed": False}
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        raise
    print(json.dumps({key: result[key] for key in
                      ("complete", "exact_resume", "steps", "elapsed_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
