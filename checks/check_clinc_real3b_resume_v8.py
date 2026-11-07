"""Real Llama 3B parity and resume proof for the local CLINC v8 runtime.

This is a bounded engineering qualification, not a CLINC experiment. It uses
only two hard-coded manual prompts and local model/tokenizer files. The Sleep
path performs exactly eight AdamW updates (two warm-up, three uninterrupted
reference updates, and three replay updates from one shared snapshot). The independent wake
clock fixture uses synthetic gradients and no Adam optimizer.

Run only after review; this script intentionally has no dataset argument and
uses ``local_files_only=True`` for all model assets.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


MAX_ADAMW_CALLS = 8
MAX_BACKBONE_FORWARDS = 64
RANK = 64
LEVELS = 5
CAPACITY = 3
SLEEP_SEED = 8122026
PERIODS = (2, 3, 5, 7, None)
PROMPTS = (
    "Please check the balance in my savings account.",
    "I would like to book a table for two tonight.",
)

LEDGER: dict[str, Any] = {
    "phase": "startup", "adamw_calls": 0, "backbone_forwards": 0,
    "reference_steps": 0, "replay_steps": 0, "wake_ticks": 0,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_value(value: Any) -> str:
    h = hashlib.sha256()

    def visit(obj):
        import numpy as np
        import torch
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu().contiguous()
            h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, np.ndarray):
            h.update(str((obj.dtype, obj.shape)).encode())
            h.update(np.ascontiguousarray(obj).tobytes())
        elif is_dataclass(obj):
            visit(asdict(obj))
        elif isinstance(obj, dict):
            for key in sorted(obj, key=lambda x: repr(x)):
                h.update(repr(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            h.update(type(obj).__name__.encode())
            for child in obj:
                visit(child)
        elif obj is None or isinstance(obj, (str, int, float, bool, bytes)):
            h.update(repr(obj).encode())
        else:
            raise TypeError(f"unsupported value in audit digest: {type(obj)!r}")

    visit(value)
    return h.hexdigest()


def _equal(left: Any, right: Any) -> bool:
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right))
    if is_dataclass(left):
        return is_dataclass(right) and _equal(asdict(left), asdict(right))
    return left == right


def _state_dict_digest(state: dict) -> str:
    return _hash_value(state)


def _all_natural(context) -> bool:
    return all(getattr(bank, "route_override", None) is None and
               not bool(getattr(bank, "dream", False)) for bank in context.banks())


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
    src = (project_root / "src").resolve()
    tools = (task_root / "tools").resolve()
    if not src.is_dir() or not tools.is_dir():
        raise FileNotFoundError("expected public src/ and task tools/ roots")
    for entry in (str(tools), str(src)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    from nested_sleep.lm import LlamaCMSLM, immutable_versions, promote_mutable_to_fp32
    from clinc_lifecycle_v8 import SleepTransaction, WakeController
    import clinc_tail_runtime_v8 as tail

    lm_path = Path(sys.modules["nested_sleep.lm"].__file__).resolve()
    life_path = Path(sys.modules["clinc_lifecycle_v8"].__file__).resolve()
    tail_path = Path(tail.__file__).resolve()
    if src not in lm_path.parents:
        raise RuntimeError(f"nested_sleep.lm loaded outside the requested public source: {lm_path}")
    if tools not in life_path.parents or tools not in tail_path.parents:
        raise RuntimeError("CLINC v8 lifecycle/tail modules did not load from the pinned tools directory")
    tool_hashes = {name: _sha(tools / name) for name in
                   ("clinc_lifecycle_v8.py", "clinc_tail_runtime_v8.py")}
    if "clinc_lifecycle_v8.py" not in tool_hashes or "clinc_tail_runtime_v8.py" not in tool_hashes:
        raise RuntimeError("required CLINC v8 tool source hashes are missing")
    return {
        "LlamaCMSLM": LlamaCMSLM,
        "promote_mutable_to_fp32": promote_mutable_to_fp32,
        "immutable_versions": immutable_versions,
        "SleepTransaction": SleepTransaction,
        "WakeController": WakeController,
        "tail": tail,
        "src": src,
        "tools": tools,
        "lm_path": lm_path,
        "life_path": life_path,
        "tail_path": tail_path,
        "tool_hashes": tool_hashes,
    }


def _gradient_hook(tx, grad_norms):
    names = sorted(tx.allowed_names)
    b_name = next((name for name in names if name.endswith(".b")), None)
    if b_name is None:
        raise RuntimeError("receiver B parameter was not included in allowed scope")
    params = dict(tx.student.named_parameters())
    return params[b_name].register_hook(
        lambda grad: grad_norms.append(float(grad.detach().float().norm().item())))


def _one_gkd(tx, model, runtime, prefix, *, owner: str, ledger: dict) -> dict:
    import torch
    grad_norms = []
    hook = _gradient_hook(tx, grad_norms)
    try:
        context = tx.teacher if owner == "teacher" else tx.student
        sampled = runtime.sample_with_context(
            model, context, prefix, count=2, generator=tx.generator,
            temperature=0.8, attention_mask=torch.ones_like(prefix, dtype=torch.long))
        if sampled.shape != (prefix.shape[0], prefix.shape[1] + 2):
            raise AssertionError("sample path did not produce exactly two continuation tokens")

        # This is a fresh full-backbone capture of the actual sampled sequence.
        # Both contexts then score every vocabulary item at each answer position.
        mask = torch.zeros_like(sampled, dtype=torch.bool)
        mask[:, prefix.shape[1]:] = True
        cache = runtime.capture(model, sampled, torch.ones_like(sampled, dtype=torch.long))
        with torch.no_grad():
            teacher_logits = runtime.tail_logits(model, tx.teacher, cache)
        student_logits = runtime.tail_logits(model, tx.student, cache)
        if teacher_logits.shape != student_logits.shape or teacher_logits.shape[-1] != model.backbone.config.vocab_size:
            raise AssertionError("GKD did not score the full model vocabulary")
        if teacher_logits.shape[1] != sampled.shape[1] or int(mask.sum()) != 2:
            raise AssertionError("GKD answer suffix mask/sequence shape mismatch")
        loss, token_count = runtime.causal_token_kl(student_logits, teacher_logits, mask,
                                                     temperature=1.0)
        if token_count != 2 or not bool(torch.isfinite(loss)):
            raise AssertionError("GKD loss is empty or nonfinite")
        sequence_digest = _hash_value(sampled.detach().cpu())
        teacher_logit_shape = list(teacher_logits.shape)
        receiver_b = dict(tx.student.named_parameters())[next(
            name for name in tx.allowed_names if name.endswith(".b"))]
        before_b = receiver_b.detach().clone()
        tx.optimize(loss, "gkd", tokens=2)
        after_b = receiver_b.detach()
        if not grad_norms or grad_norms[-1] <= 0:
            raise AssertionError("fresh receiver B gradient was not positive")
        delta_b = float((after_b.float() - before_b.float()).norm().item())
        if delta_b <= 0:
            raise AssertionError("AdamW did not change the fresh receiver B")
        if ledger["adamw_calls"] > MAX_ADAMW_CALLS:
            raise RuntimeError("AdamW call budget exceeded")
        return {"owner": owner, "loss": float(loss.detach()), "tokens": token_count,
                "vocab_size": int(teacher_logits.shape[-1]), "logit_shape": teacher_logit_shape,
                "sequence_sha256": sequence_digest, "receiver_b_grad_norm": grad_norms[-1],
                "receiver_b_delta_norm": delta_b,
                "adamw_step": int(tx.counters["adamw_steps"])}
    finally:
        hook.remove()


def _compact_snapshot(tx, np):
    """Take owned adapter-only transaction snapshot; never retain base weights."""
    snap = tx.snapshot()
    if snap.metadata.get("sender_reset_policy") != "redraw_A_normal_0_0.02_zero_B":
        raise RuntimeError("Sleep snapshot omitted the required sender reset policy")
    if snap.numpy_rng is None:
        raise RuntimeError("Sleep snapshot omitted NumPy RNG state")
    if snap.generator_state is None or snap.generator_state.numel() == 0:
        raise RuntimeError("Sleep snapshot omitted private generator state")
    return snap


def _rng_bundle(torch, np, random):
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [x.cpu().clone() for x in torch.cuda.get_rng_state_all()],
        "python": deepcopy(random.getstate()),
        "numpy": deepcopy(np.random.get_state()),
    }


def _sleep_signature(tx, torch) -> dict:
    return {
        "student": _hash_value(tx.student.state_dict()),
        "teacher": _hash_value(tx.teacher.state_dict()),
        "optimizer": _hash_value(tx.optimizer.state_dict()),
        "private_generator": _hash_value(tx.generator.get_state()),
        "counters": _hash_value(tx.counters),
        "teacher_bank_state": _hash_value(tx.snapshot().bank_states["teacher"]),
        "student_bank_state": _hash_value(tx.snapshot().bank_states["student"]),
        "rng": _hash_value(_rng_bundle(torch, __import__("numpy"), random)),
    }


def _check_wake_pending(model, runtime, WakeController) -> dict:
    """Synthetic 2/3/5/7/None clock boundary snapshot/replay; no optimizer steps."""
    import torch
    periods = PERIODS
    live = runtime.clone_mutable_context(model)
    if not _all_natural(live):
        raise AssertionError("wake fixture must start with natural routing")
    live.banks()[0].activate(0)
    controller = WakeController(live, periods=periods, base_lr=.007, fast_lr=.003)
    allowed = controller.prepare_wake()
    params = dict(live.named_parameters())
    grads = {}
    for level, period in enumerate(periods):
        if period is None:
            continue
        for suffix in ("a", "b"):
            name = f"memory.{level}.{suffix}"
            grads[name] = torch.full_like(params[name], .001 * (level + 1))
    for name in controller._fast_names:
        grads[name] = torch.full_like(params[name], .002)
    if set(grads) - allowed:
        raise AssertionError("synthetic wake gradients exceed controller scope")

    replay_context = runtime.clone_mutable_context(model)
    replay = WakeController(replay_context, periods=periods, base_lr=.007, fast_lr=.003)
    checked = []
    for tick in range(1, 8):
        controller.record_tick(grads, token_count=1 + tick % 2)
        LEDGER["wake_ticks"] += 1
        proposals = controller.proposals()
        if not proposals:
            continue
        snap = controller.snapshot()
        pending_hash = _hash_value(snap.pending)
        replay.restore(snap)
        replay_proposals = replay.proposals()
        if set(proposals) != set(replay_proposals):
            raise AssertionError(f"wake proposal levels differ at synthetic tick {tick}")
        for level in proposals:
            if _hash_value(proposals[level]) != _hash_value(replay_proposals[level]):
                raise AssertionError(f"wake proposal mismatch at tick {tick}, level {level}")
        levels = sorted(proposals)
        for level in levels:
            controller.apply(level, proposals[level])
            replay.apply(level, replay_proposals[level])
        if _hash_value(live.state_dict()) != _hash_value(replay_context.state_dict()):
            raise AssertionError(f"wake restored context differs after due tick {tick}")
        if _hash_value(snap.pending) != pending_hash:
            raise AssertionError("wake snapshot pending state aliased later application")
        checked.append({"tick": tick, "due_levels": levels,
                        "pending_sha256": pending_hash,
                        "restored_context_sha256": _hash_value(replay_context.state_dict())})
    if checked == [] or 4 in controller.clocks or any(name.startswith("memory.4.")
                                                       for name in controller.pending):
        raise AssertionError("period=None terminal level was unexpectedly clocked")
    expected_due = [[0], [1], [0], [2], [0, 1], [3]]
    if [row["due_levels"] for row in checked] != expected_due:
        raise AssertionError("synthetic 2/3/5/7 clock boundary sequence changed")
    return {"periods": list(periods), "ticks": LEDGER["wake_ticks"],
            "boundary_restores": checked, "terminal_none_is_unclocked": True,
            "adamw_calls": 0}


def run(args) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    task_root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model assets not found: {model_path}")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("this qualification is specifically for an available CUDA device")

    LEDGER["phase"] = "source-preflight"
    source = _load_sources(project_root, task_root)
    LEDGER["source_hashes"] = {
        "qualification_script": _sha(Path(__file__).resolve()),
        "lm": _sha(source["lm_path"]),
        "experts": _sha(source["lm_path"].with_name("experts.py")),
        "lifecycle": _sha(source["tools"] / "clinc_lifecycle_v8.py"),
        "tail_runtime": _sha(source["tools"] / "clinc_tail_runtime_v8.py"),
        "all_tools": source["tool_hashes"],
    }
    torch.manual_seed(20261107)
    torch.cuda.manual_seed_all(20261107)
    random.seed(20261107)
    np.random.seed(20261107)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    LEDGER["phase"] = "model-load"
    backbone = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager").eval()
    backbone.requires_grad_(False)
    model = source["LlamaCMSLM"](backbone, rank=RANK, capacity=CAPACITY, levels=LEVELS)
    source["promote_mutable_to_fp32"](model)
    model.eval().requires_grad_(False)
    if model.levels != LEVELS or model.rank != RANK or model.capacity != CAPACITY:
        raise AssertionError("real qualification adapter topology does not match v8 bounds")
    if any(not p.is_floating_point() for p in model.parameters()):
        raise AssertionError("unexpected nonfloating model parameter")
    frozen_versions = source["immutable_versions"](model)
    if not frozen_versions:
        raise AssertionError("immutable backbone guard returned an empty scope")

    # Fixed local manual prompts; no CLINC acquisition/records path is opened.
    token_rows = []
    for text in PROMPTS:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=False)
        ids = encoded["input_ids"].to(args.device)
        if ids.shape[1] == 0 or ids.shape[1] > 96:
            raise RuntimeError("manual prompt is outside the short-sequence cap")
        token_rows.append(ids)

    # Wake pending snapshot tests use synthetic gradients, including the explicit
    # None terminal clock. They do not change the independent Sleep context below.
    LEDGER["phase"] = "wake-pending-fixture"
    wake_result = _check_wake_pending(model, source["tail"], source["WakeController"])

    LEDGER["phase"] = "sleep-transaction-setup"
    live = source["tail"].clone_mutable_context(model)
    if not _all_natural(live):
        raise AssertionError("initial Sleep context is not natural-route/zero-dream")
    # A prior fast receiver gives the sender reset path a real A tensor to redraw.
    live.banks()[0].activate(0)
    sender = 0
    prospective = {f"memory.{sender}.a": live.memory[sender].a.detach().clone(),
                   f"memory.{sender}.b": live.memory[sender].b.detach().clone()}
    # A small deterministic pre-update candidate makes the teacher/student KL
    # informative while remaining an explicit synthetic engineering fixture.
    # It is not learned from an example and performs no optimizer step.
    proposal_gen = torch.Generator(device="cpu").manual_seed(SLEEP_SEED + 1)
    delta = torch.randn(prospective[f"memory.{sender}.b"].shape, generator=proposal_gen)
    delta = (delta / delta.abs().max().clamp_min(1e-12) * 1e-3).to(prospective[f"memory.{sender}.b"])
    prospective[f"memory.{sender}.b"] = prospective[f"memory.{sender}.b"] + delta
    tx = source["SleepTransaction"](live, sender, prospective,
                                    lr=0.001, weight_decay=0., clip_norm=1., seed=SLEEP_SEED)
    if not (_all_natural(tx.live) and _all_natural(tx.teacher) and _all_natural(tx.student)):
        raise AssertionError("Sleep branch changed natural routing policy")
    setup_snapshot = tx.snapshot()
    if setup_snapshot.metadata.get("sender_reset_policy") != "redraw_A_normal_0_0.02_zero_B":
        raise RuntimeError("Sleep transaction did not record the required private-generator sender reset")
    if bool(tx.student.banks()[sender].slots[0].active):
        raise AssertionError("sender expert was not reset/deactivated in candidate")
    if not bool(tx.teacher.banks()[sender].slots[0].active):
        raise AssertionError("teacher did not retain its pre-transaction sender expert")
    teacher_a = tx.teacher.banks()[sender].slots[0].a.detach()
    student_a = tx.student.banks()[sender].slots[0].a.detach()
    if torch.equal(teacher_a, student_a):
        raise AssertionError("private generator did not redraw the candidate sender A")
    rng_seed = torch.Generator(device=tx.generator.device).manual_seed(SLEEP_SEED)
    if torch.equal(tx.generator.get_state(), rng_seed.get_state()):
        raise AssertionError("private generator state does not include sender reset consumption")
    private_gen_after_reset_sha = _hash_value(tx.generator.get_state())

    # Count all actual HF backbone invocations across sequence sampling and fresh
    # post-sample capture. Tail-only loss evaluation does not invoke the backbone.
    meter = ForwardMeter(model.backbone)
    def bind_adam_counter(transaction):
        original_step = transaction.optimizer.step

        def counted_step(*step_args, **step_kwargs):
            if LEDGER["adamw_calls"] >= MAX_ADAMW_CALLS:
                raise RuntimeError(f"AdamW cap {MAX_ADAMW_CALLS} exceeded")
            result = original_step(*step_args, **step_kwargs)
            LEDGER["adamw_calls"] += 1
            return result

        transaction.optimizer.step = counted_step

    bind_adam_counter(tx)
    live_hash = _hash_value(live.state_dict())
    teacher_hash = _hash_value(tx.teacher.state_dict())
    reference_steps = []

    # Two pre-snapshot AdamW updates make moments and step tensors nonempty.
    LEDGER["phase"] = "sleep-pre-updates"
    for i in range(2):
        prompt = token_rows[i % len(token_rows)]
        owner = "teacher" if i == 0 else "student"
        reference_steps.append(_one_gkd(tx, model, source["tail"], prompt, owner=owner, ledger=LEDGER))
    if tx.counters["adamw_steps"] != 2 or not tx.optimizer.state:
        raise AssertionError("pre-snapshot phase did not create nonempty AdamW state")
    if any(not ("exp_avg" in values and "exp_avg_sq" in values and "step" in values)
           for values in tx.optimizer.state.values()):
        raise AssertionError("nonempty AdamW moments/step state was not established")

    # One owned checkpoint after two updates, followed by three uninterrupted
    # reference calls and three calls replayed from that exact same checkpoint.
    snap = _compact_snapshot(tx, np)
    snapshot_digest = _hash_value(snap)
    snapshot_adam_digest = _hash_value(snap.optimizer_state)
    checkpoint_path = output_dir / "step02-sleep-checkpoint.pt"
    temporary_path = output_dir / "step02-sleep-checkpoint.pt.partial"
    torch.save(snap, temporary_path)
    disk_sha = _sha(temporary_path)
    with temporary_path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, checkpoint_path)
    if _sha(checkpoint_path) != disk_sha:
        raise AssertionError("serialized checkpoint hash changed after commit")
    # Deserialize only our newly written local mutable-state checkpoint.
    # It contains no frozen 3B backbone weights and is never published.
    disk_snapshot = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if _hash_value(disk_snapshot) != snapshot_digest:
        raise AssertionError("serialized Sleep snapshot did not roundtrip exactly")
    owners = ("teacher", "student", "teacher")
    prompts = tuple(token_rows[(i + 1) % len(token_rows)] for i in range(3))
    ref_results = []
    for i, (owner, prompt) in enumerate(zip(owners, prompts)):
        LEDGER["phase"] = f"reference-{i}"
        result = _one_gkd(tx, model, source["tail"], prompt, owner=owner, ledger=LEDGER)
        ref_results.append(result)
        reference_steps.append(result)
        LEDGER["reference_steps"] += 1
    reference_signature = _sleep_signature(tx, torch)
    reference_rng = _rng_bundle(torch, np, random)
    reference_student = _hash_value(tx.student.state_dict())
    reference_teacher = _hash_value(tx.teacher.state_dict())
    reference_live = _hash_value(live.state_dict())
    if _hash_value(snap) != snapshot_digest or _hash_value(snap.optimizer_state) != snapshot_adam_digest:
        raise AssertionError("checkpoint tensors or Adam state aliased uninterrupted updates")

    # Disturb process-global RNGs so the restore must actively reinstate all
    # captured streams before replaying the same three sampled GKD updates.
    random.random(); np.random.random(); torch.rand((), device="cpu"); torch.rand((), device="cuda")
    LEDGER["phase"] = "restore-one-snapshot"
    replay = source["SleepTransaction"].from_snapshot(live, disk_snapshot)
    bind_adam_counter(replay)
    if not _equal(_rng_bundle(torch, np, random), snap_rng_bundle(snap)):
        raise AssertionError("Sleep restore did not reproduce global RNG state before replay")
    if not torch.equal(replay.generator.get_state(), snap.generator_state):
        raise AssertionError("private generator state was not restored")
    if _hash_value(replay.teacher.state_dict()) != _hash_value(snap.teacher_state):
        raise AssertionError("teacher state differs from the Sleep snapshot")

    replay_results = []
    for i, (owner, prompt) in enumerate(zip(owners, prompts)):
        LEDGER["phase"] = f"snapshot-replay-{i}"
        result = _one_gkd(replay, model, source["tail"], prompt, owner=owner, ledger=LEDGER)
        replay_results.append(result)
        reference_steps.append({"replay_of": i, **result})
        LEDGER["replay_steps"] += 1
        for field in ("sequence_sha256", "loss", "receiver_b_grad_norm", "receiver_b_delta_norm"):
            if result[field] != ref_results[i][field]:
                raise AssertionError(f"snapshot replay {i} differs in {field}")

    if _sleep_signature(replay, torch) != reference_signature:
        raise AssertionError("restored Adam/model/private/global RNG state diverged after three replays")
    if _hash_value(replay.student.state_dict()) != reference_student:
        raise AssertionError("restored student model differs from uninterrupted three-step branch")
    if _hash_value(replay.teacher.state_dict()) != reference_teacher:
        raise AssertionError("teacher changed across restored branch")
    if _hash_value(replay.live.state_dict()) != reference_live or reference_live != live_hash:
        raise AssertionError("live context changed across Sleep branches")
    if not _equal(_rng_bundle(torch, np, random), reference_rng):
        raise AssertionError("global CPU/CUDA/Python/NumPy RNG diverged after replay")
    if _hash_value(snap) != snapshot_digest or _hash_value(snap.optimizer_state) != snapshot_adam_digest:
        raise AssertionError("snapshot aliased the restored optimizer or mutable contexts")
    if not _all_natural(replay.live) or not _all_natural(replay.teacher) or not _all_natural(replay.student):
        raise AssertionError("restored Sleep branch did not preserve natural route policy")

    meter.close()
    if LEDGER["adamw_calls"] != MAX_ADAMW_CALLS:
        raise AssertionError(f"expected exactly {MAX_ADAMW_CALLS} AdamW updates, got {LEDGER['adamw_calls']}")
    if meter.count > MAX_BACKBONE_FORWARDS:
        raise AssertionError("backbone-forward cap exceeded")
    if _hash_value(live.state_dict()) != live_hash or _hash_value(tx.teacher.state_dict()) != teacher_hash:
        raise AssertionError("live context or source teacher mutated during Sleep updates")
    if source["immutable_versions"](model) != frozen_versions:
        raise AssertionError("frozen 3B backbone parameter versions changed")
    if tx.counters["zero_gradient_steps"]:
        raise AssertionError("receiver had a zero-gradient AdamW step")
    current_tool_hashes = {name: _sha(source["tools"] / name) for name in source["tool_hashes"]}
    if current_tool_hashes != source["tool_hashes"]:
        raise AssertionError("v8 tool sources changed during qualification")

    return {
        "schema": "clinc-real3b-sleep-resume-v8-v1",
        "complete": True,
        "device": args.device,
        "model_path": str(model_path),
        "architecture_scope": "LlamaCMSLM last-MLP augmentation; rank64, levels5, expert capacity3; local partial protocol",
        "dtype_backbone": "bfloat16",
        "dtype_mutable": "float32",
        "manual_prompt_count": len(PROMPTS),
        "manual_prompt_lengths": [int(row.shape[1]) for row in token_rows],
        "clinc_data_accessed": False,
        "route_policy": "natural; no overrides; no dream",
        "sleep_sender_reset_policy": snap.metadata["sender_reset_policy"],
        "private_generator_includes_sender_reset": True,
        "prospective_sender_update": "deterministic 1e-3 synthetic level-0 B perturbation; no text-derived update",
        "gkd_steps": reference_steps,
        "reference_adamw_calls": 5,
        "snapshot_replay_adamw_calls": 3,
        "adamw_calls": LEDGER["adamw_calls"],
        "max_adamw_calls": MAX_ADAMW_CALLS,
        "backbone_forwards": meter.count,
        "max_backbone_forwards": MAX_BACKBONE_FORWARDS,
        "receiver_zero_gradient_steps": tx.counters["zero_gradient_steps"],
        "receiver_b_updates_positive": all(x.get("receiver_b_grad_norm", 0) > 0 and
                                             x.get("receiver_b_delta_norm", 0) > 0
                                             for x in reference_steps),
        "rng_state_bundle": ["private_generator", "global_cpu", "global_cuda", "python", "numpy"],
        "sleep_resume_replays_exact": True,
        "checkpoint_disk_roundtrip_exact": True,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": disk_sha,
        "wake_pending_fixture": wake_result,
        "source_hashes": LEDGER["source_hashes"],
        "elapsed_seconds": time.perf_counter() - started,
        "notes": [
            "Teacher/student full-vocabulary KL is evaluated on a freshly recaptured sampled sequence.",
            "One checkpoint after two updates is followed by three continuous reference updates and three continuous replay updates.",
            "Wake clock test uses only synthetic gradients and tests periods 2/3/5/7 plus an unclocked None terminal level.",
            "No model weights, prompt text, CLINC examples, or sampled token IDs are written to the output file.",
        ],
    }


def snap_rng_bundle(snap):
    """Convert the explicit SleepSnapshot RNG fields to the shared comparison form."""
    return {"cpu": snap.cpu_rng.detach().cpu().clone(),
            "cuda": None if snap.cuda_rng is None else [x.cpu().clone() for x in snap.cuda_rng],
            "python": deepcopy(snap.python_rng),
            "numpy": deepcopy(snap.numpy_rng)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, help="public source checkout for nested_sleep.lm")
    parser.add_argument("--model-path", required=True, help="existing local Llama 3B asset directory")
    parser.add_argument("--output-dir", required=True, help="new private output directory")
    parser.add_argument("--device", required=True, help="CUDA device, e.g. cuda:0")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
        (output_dir / "check-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: result[k] for k in (
            "complete", "adamw_calls", "backbone_forwards", "sleep_resume_replays_exact",
            "receiver_b_updates_positive", "elapsed_seconds")}, sort_keys=True))
        return 0
    except BaseException as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {"schema": "clinc-real3b-sleep-resume-v8-failure-v1", "complete": False,
                   "phase": LEDGER.get("phase"), "error_type": type(exc).__name__,
                   "error": str(exc), "adamw_calls": LEDGER.get("adamw_calls", 0),
                   "backbone_forwards": LEDGER.get("backbone_forwards", 0),
                   "ledger": {k: v for k, v in LEDGER.items() if k != "source_hashes"},
                   "source_hashes": LEDGER.get("source_hashes", {}),
                   "elapsed_seconds": time.perf_counter() - started,
                   "model_path": str(Path(args.model_path).resolve()),
                   "clinc_data_accessed": False}
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: failure[k] for k in (
            "complete", "phase", "error_type", "adamw_calls", "backbone_forwards")}, sort_keys=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
