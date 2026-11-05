"""Bounded synthetic-fact DEV on the pinned local Llama 3.2-3B asset.

This writes private progress/checkpoint files only under ignored outputs. It has
no held-out-test code path. The result is an LR feasibility probe, not a main
experiment or a natural-language/continual-learning result.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parent
SRC = (ROOT.parent / "src").resolve()
sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_sha256(value):
    t = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def candidate_qualifies(old_post):
    train = old_post["train"]["full_vocab_two_token_greedy_exact_match"]
    calibration = old_post["calibration"]["full_vocab_two_token_greedy_exact_match"]
    return train >= 0.80 and calibration >= 0.50


def cpu_contract_tests():
    # BF16 and zero-dimensional tensors both pass the exact byte-hash path.
    bf16 = torch.tensor([1.25, -0.5], dtype=torch.bfloat16)
    scalar = torch.tensor(3.0, dtype=torch.float32)
    assert tensor_sha256(bf16) == tensor_sha256(bf16.clone())
    assert tensor_sha256(scalar) == tensor_sha256(scalar.clone())
    assert candidate_qualifies({"train": {"full_vocab_two_token_greedy_exact_match": .80},
                                "calibration": {"full_vocab_two_token_greedy_exact_match": .50}})
    assert not candidate_qualifies({"train": {"full_vocab_two_token_greedy_exact_match": .79},
                                    "calibration": {"full_vocab_two_token_greedy_exact_match": .99}})
    assert not candidate_qualifies({"train": {"full_vocab_two_token_greedy_exact_match": .99},
                                    "calibration": {"full_vocab_two_token_greedy_exact_match": .49}})
    try:
        candidate_qualifies({"train": {}, "calibration": {}})
    except KeyError:
        pass
    else:
        raise AssertionError("missing score schema key must fail, not become a scientific no-qualifier")
    return {"status": "passed", "tensor_hash_dtypes": ["bfloat16", "float32_scalar"],
            "candidate_score_schema_cases": 4}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def ensure_sources():
    import nested_sleep.accounting, nested_sleep.lifecycle, nested_sleep.lm, nested_sleep.optim
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        if name == "nested_sleep" or name.startswith("nested_sleep."):
            module_file = getattr(module, "__file__", None)
            if module_file:
                p = Path(module_file).resolve()
                if not p.is_relative_to(SRC):
                    raise RuntimeError(f"external source module refused: {name}={p}")
                loaded[name] = {"path": str(p), "sha256": sha256(p)}
    needed = {"nested_sleep.accounting", "nested_sleep.lifecycle", "nested_sleep.lm", "nested_sleep.optim"}
    if needed - set(loaded):
        raise RuntimeError(f"source binding incomplete: {sorted(needed-set(loaded))}")
    return loaded


SOURCE_HASHES = ensure_sources()
from nested_sleep.accounting import ACTIVE_METER, WorkMeter
from nested_sleep.lifecycle import SleepSession, fingerprint
from nested_sleep.lm import LlamaCMSLM, isolated_clone, promote_mutable_to_fp32
from nested_sleep.optim import CMSController


TEMPLATES = [
    "What code is assigned to entry {entity}? Answer:",
    "For entry {entity}, return its code. Code:",
    "Recall the code for record {entity}. Answer:",
    "Lookup {entity}; provide only its code:",
]
CAL_TEMPLATE = "Which code belongs to the entry named {entity}? The code is"
HELDOUT_TEST_TEMPLATE_UNUSED = "Give the stored code for the record named {entity}. Code:"
LABEL_TEXTS = [" red", " blue", " green", " gold"]
COMPLETION_MARKER = " tag"


class Clock:
    def __init__(self, deadline):
        self.deadline = deadline
    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("hard 1200-second DEV wall cap reached; no extension")


def fact_rows(tokenizer, phase: str, namespace: str, count=64):
    rows = []
    labels = []
    prefixes = []
    for text in LABEL_TEXTS:
        ids = tokenizer.encode(COMPLETION_MARKER + text, add_special_tokens=False)
        if len(ids) != 2:
            raise RuntimeError(f"answer must be exactly shared-marker + class token: {(COMPLETION_MARKER+text)!r} -> {ids}")
        prefixes.append(ids[0]); labels.append(ids[1])
    if len(set(prefixes)) != 1 or len(set(labels)) != len(labels):
        raise RuntimeError("completion marker is not common or class token IDs collide")
    assignments = [i % len(labels) for i in range(count)]
    random.Random(f"balanced-labels:{namespace}:907").shuffle(assignments)
    for i in range(count):
        # Deterministic opaque synthetic entity; no external text or P2 examples.
        digest = hashlib.sha256(f"{namespace}\0{i:04d}".encode()).hexdigest()[:12].upper()
        entity = f"{namespace}-{digest}"
        label_ix = assignments[i]
        if phase == "train":
            prompts = [t.format(entity=entity) for t in TEMPLATES]
        elif phase == "calibration":
            prompts = [CAL_TEMPLATE.format(entity=entity)]
        else:
            raise ValueError("this DEV process intentionally generates only train/calibration roles")
        for template_index, prompt in enumerate(prompts):
            token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if not token_ids or len(token_ids) > 96:
                raise RuntimeError("synthetic prompt length outside frozen bound")
            rows.append({"entity": entity, "template": template_index, "tokens": token_ids,
                         "marker": prefixes[0], "label": labels[label_ix]})
    return rows


def collate(rows, device, pad_id):
    lengths = torch.tensor([len(row["tokens"]) for row in rows], dtype=torch.long, device=device)
    width = int(lengths.max())
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row["tokens"])] = torch.tensor(row["tokens"], dtype=torch.long, device=device)
        mask[i, :len(row["tokens"])] = 1
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long, device=device)
    markers = torch.tensor([row["marker"] for row in rows], dtype=torch.long, device=device)
    return ids, mask, lengths, markers, labels


def score(model, rows, pad_id, *, batch=8):
    device = next(model.parameters()).device
    losses, marker_losses, class_losses, correct, class_correct, marker_hits, n, fw = 0., 0., 0., 0, 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), batch):
            ids, mask, lengths, markers, labels = collate(rows[offset:offset+batch], device, pad_id)
            full, full_mask = append_markers(ids, mask, lengths, markers, pad_id)
            logits = model.logits(full, full_mask)
            first = logits[torch.arange(len(labels), device=device), lengths - 1].float()
            second = logits[torch.arange(len(labels), device=device), lengths].float()
            m = F.cross_entropy(first, markers, reduction="sum")
            c = F.cross_entropy(second, labels, reduction="sum")
            marker_losses += float(m); class_losses += float(c); losses += float(m+c)
            marker_pred, class_pred = first.argmax(-1), second.argmax(-1)
            marker_hits += int((marker_pred == markers).sum())
            class_correct += int((class_pred == labels).sum())
            correct += int(((marker_pred == markers) & (class_pred == labels)).sum())
            n += len(labels)
            fw += int(full_mask.sum())
    return {"full_vocab_nll_nats_per_answer_token": losses / (2*n),
            "full_vocab_marker_nll_nats": marker_losses / n,
            "full_vocab_class_token_nll_nats": class_losses / n,
            "full_vocab_two_token_greedy_exact_match": correct / n,
            "marker_greedy_accuracy": marker_hits / n,
            "class_token_greedy_accuracy": class_correct / n, "examples": n,
            "nonpad_input_tokens": fw}


def append_markers(ids, mask, lengths, markers, pad_id):
    full = torch.full((ids.shape[0], ids.shape[1] + 1), pad_id, dtype=ids.dtype, device=ids.device)
    full_mask = torch.zeros_like(full)
    full[:, :ids.shape[1]] = ids
    full_mask[:, :mask.shape[1]] = mask
    rows = torch.arange(ids.shape[0], device=ids.device)
    full[rows, lengths] = markers
    full_mask[rows, lengths] = 1
    return full, full_mask


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.adapter_state_dict().items()}


def load_snapshot(model, state):
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected keys while restoring identical init: {result.unexpected_keys}")
    model.eval().requires_grad_(False)


def step(model, controller, batch_rows, pad_id, meter, stage, *, lr_clip=1., defer_due=False):
    device = next(model.parameters()).device
    controller.prepare_wake(model)
    meter.stage = stage
    ids, mask, lengths, markers, labels = collate(batch_rows, device, pad_id)
    full, full_mask = append_markers(ids, mask, lengths, markers, pad_id)
    logits = model.logits(full, full_mask)
    rows = torch.arange(len(labels), device=device)
    first = logits[rows, lengths - 1].float()
    second = logits[rows, lengths].float()
    loss = .5 * (F.cross_entropy(first, markers) + F.cross_entropy(second, labels))
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite supervised loss")
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, lr_clip))
    proposals = controller.collect(model)
    # Fixed local simultaneous-due rule outside the Sleep terminal boundary:
    # commit slow base first, then fast base.
    if not defer_due:
        for level in sorted(proposals, reverse=True):
            controller.apply(model, level, proposals[level])
    return {"loss": float(loss.detach()), "grad_norm": grad_norm,
            "due_levels": sorted(proposals), "proposals": proposals if defer_due else {},
            "nonpad_tokens": int(full_mask.sum())}


def cpu_state_fixture():
    """CPU fixture for restore plus deferred slow-first/Sleep branch at 63/64."""
    torch.manual_seed(904)
    # Use actual model/controller API with a tiny two-level synthetic model to
    # assert due proposal order/accounting without a 3B model allocation.
    from nested_sleep.lm import NativeResearchLM
    from nested_sleep.native import NativeConfig
    model = NativeResearchLM(NativeConfig(vocab_size=32, dim=16, hidden=24, layers=1, cms_periods=(4, 16)),
                             rank=2, capacity=2)
    controller = CMSController(model, periods=(4, 16), base_lr=.01, expert_lr=.02)
    def one_event(m, c, event):
        c.prepare_wake(m)
        logits = m.logits(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        logits.sum().backward()
        proposals = c.collect(m)
        expected = [] if event % 4 else ([0, 1] if event % 16 == 0 else [0])
        if sorted(proposals) != expected:
            raise AssertionError((event, sorted(proposals), expected))
        for level in sorted(proposals, reverse=True):
            c.apply(m, level, proposals[level])
    # Bring the source branch to event15, then restore model and pending clocks.
    for event in range(1, 16):
        one_event(model, controller, event)
    restored = deepcopy(model)
    restored_controller = CMSController.restore(restored, controller.state_dict())
    one_event(model, controller, 16)
    one_event(restored, restored_controller, 16)
    if fingerprint(model) != fingerprint(restored) or tensor_tree_hash(controller.state_dict()) != tensor_tree_hash(restored_controller.state_dict()):
        raise AssertionError("CPU controller/model restore diverged at 15/16")
    for event in range(17, 64):
        one_event(model, controller, event)
    controller.prepare_wake(model)
    logits = model.logits(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    logits.sum().backward()
    proposals = controller.collect(model)
    if sorted(proposals) != [0, 1] or controller.counts != [4, 16]:
        raise AssertionError(f"expected deferred simultaneous due [4,16], got {sorted(proposals)} {controller.counts}")
    controller_deferred = tensor_tree_hash(controller.state_dict())
    live_deferred = fingerprint(model)
    proposals = {level: {n:v.detach().clone() for n,v in row.items()} for level,row in proposals.items()}
    candidate = deepcopy(model)
    candidate_controller = CMSController.restore(candidate, controller.state_dict())
    candidate_controller.apply(candidate, 1, proposals[1])
    if candidate_controller.counts != [4, 0]:
        raise AssertionError("slow proposal was not consumed first on the candidate")
    session = SleepSession(candidate, 0, proposals[0], seed=1904, lr=.003,
                           dream_tokens=2, reject_zero_gradient=True)
    sender_params = dict(session.student.named_parameters())
    if not all(torch.equal(sender_params[n], v) for n,v in proposals[0].items()):
        raise AssertionError("fast prospective proposal was not installed in Sleep student")
    if fingerprint(session.teacher) != fingerprint(candidate):
        raise AssertionError("Sleep teacher is not the slow-updated/pre-fast candidate")
    session.seed_step(torch.tensor([[1, 2, 3]], dtype=torch.long), owner="teacher")
    if session.records[-1]["grad_norm"] <= 0:
        raise AssertionError("CPU receiver gradient gate failed")
    session.finish_consolidation()
    committed = session.commit()
    candidate_controller.consume(0)
    if candidate_controller.counts != [0, 0]:
        raise AssertionError("candidate clocks not consumed after successful Sleep commit")
    if fingerprint(model) != live_deferred or tensor_tree_hash(controller.state_dict()) != controller_deferred:
        raise AssertionError("deferred Sleep candidate mutated the original live branch")
    return {"status": "passed", "events": 64, "restored_next_event_15_to_16": True,
            "simultaneous_due_event":64,"deferred_live_counts_before_transaction":[4,16],
            "candidate_counts_after_slow_then_fast_sleep":[0,0],
            "proposal_order":["slow_level1_apply","fast_level0_prospective_sleep"],
            "live_unchanged_before_replacement":True,"final_counts":candidate_controller.counts}


def compact_cpu(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, dict):
        return {k: compact_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(compact_cpu(v) for v in value)
    return deepcopy(value)


def tensor_tree_hash(value):
    h = hashlib.sha256()
    def visit(obj, key=""):
        if torch.is_tensor(obj):
            t = obj.detach().cpu().contiguous()
            h.update(key.encode()); h.update(str((tuple(t.shape), str(t.dtype))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            for k in sorted(obj, key=str): visit(obj[k], key+"/"+str(k))
        elif isinstance(obj, (tuple, list)):
            for i, item in enumerate(obj): visit(item, key+f"/{i}")
        else:
            h.update((key+"="+repr(obj)).encode())
    visit(value)
    return h.hexdigest()


def active_parameter_counts(model):
    return {"active_slots_by_bank": [sum(bool(s.active) for s in bank.slots) for bank in model.banks()],
            "mutable_adapter_parameters": sum(p.numel() for n,p in model.named_parameters()
                if n.startswith(model.memory_prefix) or n in model.policy_names()),
            "currently_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def alias_restore_fixture(model, prefix, session, private_dir, meter, marker_id):
    """Real Llama SleepSession save/restore: independent restored branches."""
    session.seed_step(prefix)
    expected_pre = fingerprint(session.student)
    state = session.state_dict()
    # Compact adapter-only restore payload while retaining the full optimizer,
    # RNG, generator, event, and scope state required by mutable_only restore.
    state["teacher"] = session.teacher.adapter_state_dict()
    state["student"] = session.student.adapter_state_dict()
    np_state = np.random.get_state()
    state["rng"]["numpy"] = {"kind": np_state[0], "keys": torch.from_numpy(np_state[1].copy()),
                               "position": int(np_state[2]), "has_gauss": int(np_state[3]),
                               "cached_gaussian": float(np_state[4])}
    saved = compact_cpu(state)
    path = private_dir / "dev907-sleep-restore.pt"
    torch.save(saved, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    before_hash = tensor_tree_hash(saved)
    def restore_numpy_rng(rng_state):
        v = rng_state["numpy"]
        np.random.set_state((v["kind"], v["keys"].cpu().numpy().astype(np.uint32),
                             int(v["position"]), int(v["has_gauss"]), float(v["cached_gaussian"])))
    restore_numpy_rng(saved["rng"])
    np_first = float(np.random.random())
    restore_numpy_rng(saved["rng"])
    np_second = float(np.random.random())
    if np_first != np_second:
        raise RuntimeError("NumPy RNG checkpoint did not restore identically")
    restore_numpy_rng(saved["rng"])
    first = SleepSession.restore(model, saved, mutable_only=True)
    meter.stage = "dev_restore_fixture"
    first.seed_step(prefix)
    after_first = tensor_tree_hash(saved)
    restore_numpy_rng(saved["rng"])
    second = SleepSession.restore(model, saved, mutable_only=True)
    second.seed_step(prefix)
    meter.stage = "dev_sleep_transaction"
    if fingerprint(first.student) != fingerprint(second.student):
        raise RuntimeError("two restored real-3B next GKD updates diverged")
    if before_hash != after_first:
        raise RuntimeError("restored real-3B update mutated serialized lifecycle snapshot")
    return {"status": "passed", "precheckpoint_student_fingerprint": expected_pre,
            "postrestore_next_gkd_equal": True, "snapshot_metadata_unchanged": True,
            "private_checkpoint_bytes": path.stat().st_size,
            "heldout_roles_read": False}


def sleep_transaction_fixture(model, controller, old_train, new_train, pad_id, private_dir, meter, seed, clock, marker_id):
    """Advance exactly 64 new wake updates and transact one simultaneous due."""
    rng = random.Random(seed + 64)
    source = new_train
    last = None
    for i in range(64):
        clock.check()
        batch = [source[rng.randrange(len(source))] for _ in range(4)]
        last = step(model, controller, batch, pad_id, meter, "dev_new_wake", defer_due=(i == 63))
    if last is None or last["due_levels"] != [0, 1] or controller.counts != [4, 16]:
        raise RuntimeError(f"expected simultaneous due at 64, got due={last} counts={controller.counts}")
    # At event 64 the online expert changes are already part of this model.
    # Branch controller/model; apply level-1 on the candidate first, then pass
    # fast level-0 proposal prospectively to SleepSession (never pre-apply it).
    proposals = last["proposals"]
    if set(proposals) != {0,1}:
        raise RuntimeError("deferred due step did not retain both CMS proposals")
    proposals = {level:{n:v.detach().clone() for n,v in row.items()} for level,row in proposals.items()}
    candidate = isolated_clone(model, share_backbone=True)
    candidate_controller = CMSController.restore(candidate, controller.state_dict())
    candidate_controller.apply(candidate, 1, proposals[1])
    live_before = fingerprint(model)
    # Balanced prompt-only pool: alternating old/new training prompts, no labels.
    prompts = []
    for i in range(min(32, len(old_train), len(new_train))):
        prompts.extend([old_train[i]["tokens"], new_train[i]["tokens"]])
    prefix_choice = prompts[0]
    prefix = torch.tensor(prefix_choice, dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)
    meter.stage = "dev_sleep_transaction"
    session = SleepSession(candidate, 0, proposals[0], seed=seed + 7000,
                           lr=0.003, dream_tokens=2, share_backbone=True,
                           reject_zero_gradient=True)
    marker_observations = []
    original_teacher_sample = session.teacher.sample
    def observe_teacher_sample(sample_prefix, count, **kwargs):
        generated, trace = original_teacher_sample(sample_prefix, count, **kwargs)
        generated_marker = generated[:, sample_prefix.shape[1]]
        marker_observations.extend((generated_marker == marker_id).detach().cpu().tolist())
        return generated, trace
    session.teacher.sample = observe_teacher_sample
    controller_snapshot_hash = tensor_tree_hash(controller.state_dict())
    # True restore fixture is charged inside this same development budget.
    restore = alias_restore_fixture(candidate, prefix, session, private_dir, meter, marker_id)
    # Continue the original transaction at its current (one-step) state.
    session.step_observer = None
    for i in range(15):
        clock.check()
        session.seed_step(torch.tensor(prompts[(i + 1) % len(prompts)], dtype=torch.long,
                                       device=prefix.device).unsqueeze(0), owner=None)
    for i in range(8):
        clock.check()
        session.lti_step(torch.tensor(prompts[(i + 16) % len(prompts)], dtype=torch.long,
                                      device=prefix.device).unsqueeze(0))
    valid_gradients = all(r.get("grad_norm", 0.) > 0 for r in session.records if r.get("kind") != "dreaming")
    if not valid_gradients:
        raise RuntimeError("Sleep candidate failed the fixed nonzero receiver-gradient engineering gate")
    session.finish_consolidation()
    committed = session.commit()
    candidate_controller.consume(0)
    if fingerprint(model) != live_before:
        raise RuntimeError("Sleep candidate changed the original live model before replacement")
    if tensor_tree_hash(controller.state_dict()) != controller_snapshot_hash:
        raise RuntimeError("Sleep candidate changed the original controller pending state before replacement")
    if candidate_controller.counts != [0, 0]:
        raise RuntimeError(f"unexpected lifecycle clocks after commit: {candidate_controller.counts}")
    sleep_summary = {"status": "committed", "gkd_steps": 16, "lti_steps": 8,
                     "dream_steps": 0, "dream_tokens": 2,
                     "semantic_judge": "exact_output_proxy_NOT_paper_semantic_RM",
                     "slow_applied_before_teacher": True, "fast_proposal_preapplied": False,
                     "atomic_replacement_ready": True, "receiver_slot": session.slot,
                     "original_live_fingerprint_unchanged": True,
                     "original_controller_pending_unchanged": True,
                     "receiver_gradient_norms_nonzero": valid_gradients,
                     "teacher_forced_token_count": session.work,
                     "teacher_gkd_and_lti_marker_hits": sum(marker_observations),
                     "teacher_gkd_and_lti_marker_observations": len(marker_observations),
                     "teacher_gkd_and_lti_marker_hit_rate": (sum(marker_observations)/len(marker_observations)
                                                               if marker_observations else None),
                     "records_with_token_ids_exported": False,
                     "post_commit_model_capacity_counts": active_parameter_counts(committed)}
    return committed, candidate_controller, {"sleep": sleep_summary, "restore": restore}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "x870e_synthetic_dev_v2.json")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--cpu-fixture-only", action="store_true")
    parser.add_argument("--cpu-tests-only", action="store_true")
    parser.add_argument("--cumulative-start-utc", type=str,
                        help="original attempt-1 wall-clock start; preserves the single cumulative budget across retries")
    parser.add_argument("--prior-attempt-foundation-updates", type=int, default=0)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    if args.cpu_fixture_only:
        print(json.dumps({"cpu_fixture": cpu_state_fixture(), "source_hashes": SOURCE_HASHES}, indent=2), flush=True)
        return
    if args.cpu_tests_only:
        print(json.dumps({"cpu_contract_tests": cpu_contract_tests(), "source_hashes": SOURCE_HASHES}, indent=2), flush=True)
        return
    if args.model_dir is None or args.out_dir is None:
        parser.error("--model-dir and --out-dir are required unless --cpu-fixture-only is set")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    global TEMPLATES, CAL_TEMPLATE, HELDOUT_TEST_TEMPLATE_UNUSED
    TEMPLATES = list(cfg["template_roles"]["train"])
    CAL_TEMPLATE = cfg["template_roles"]["calibration"][0]
    HELDOUT_TEST_TEMPLATE_UNUSED = cfg["template_roles"]["heldout_test_unused"][0]
    configured_templates = TEMPLATES + [CAL_TEMPLATE, HELDOUT_TEST_TEMPLATE_UNUSED]
    if any(x != x.rstrip() for x in configured_templates):
        raise RuntimeError("prompt templates must not contain trailing whitespace")
    if len(TEMPLATES) != cfg["train_templates_per_fact"] or len({*TEMPLATES, CAL_TEMPLATE, HELDOUT_TEST_TEMPLATE_UNUSED}) != len(TEMPLATES)+2:
        raise RuntimeError("configured template roles must have fixed counts and disjoint surface forms")
    cfg_sha = json_hash(cfg)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    budget_start_epoch = (datetime.fromisoformat(args.cumulative_start_utc.replace("Z", "+00:00")).timestamp()
                          if args.cumulative_start_utc else time.time())
    remaining = max(0., budget_start_epoch + cfg["budget"]["hard_wall_seconds"] - time.time())
    clock = Clock(time.monotonic() + remaining)
    elapsed_before_launch = cfg["budget"]["hard_wall_seconds"] - remaining
    print(json.dumps({"status": "started", "pid": os.getpid(),
                      "output_dir": str(args.out_dir), "config_sha256": cfg_sha,
                      "cumulative_budget_start_utc": datetime.fromtimestamp(budget_start_epoch,timezone.utc).isoformat(),
                      "cumulative_elapsed_before_launch_seconds": elapsed_before_launch,
                      "remaining_wall_seconds": remaining}), flush=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not torch.cuda.is_available():
        raise RuntimeError("X870E CUDA device required for this DEV stage")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    np.random.seed(cfg["seed"])
    gpu = torch.cuda.get_device_properties(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                               trust_remote_code=False)
    labels = [tokenizer.encode(COMPLETION_MARKER + x, add_special_tokens=False) for x in LABEL_TEXTS]
    if any(len(x) != 2 for x in labels) or len({x[0] for x in labels}) != 1 or len({x[1] for x in labels}) != len(labels):
        raise RuntimeError("completion marker/class tokenizer boundary check failed")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    old_train = fact_rows(tokenizer, "train", "DEV907-OLD")
    old_cal = fact_rows(tokenizer, "calibration", "DEV907-OLD")
    new_train = fact_rows(tokenizer, "train", "DEV907-NEW")
    new_cal = fact_rows(tokenizer, "calibration", "DEV907-NEW")
    if set(r["entity"] for r in old_train) & set(r["entity"] for r in new_train):
        raise RuntimeError("old/new entity collision")
    manifest = {"schema": "x870e-synthetic-dev-v2-run", "config_sha256": cfg_sha,
                "source_hashes": SOURCE_HASHES, "roles": {"old_train_rows": len(old_train),
                "old_calibration_rows": len(old_cal), "new_train_rows": len(new_train),
                "new_calibration_rows": len(new_cal), "heldout_test_rows_generated": 0},
                "role_entity_sha256": {"old": hashlib.sha256("|".join(sorted({r["entity"] for r in old_train})).encode()).hexdigest(),
                                        "new": hashlib.sha256("|".join(sorted({r["entity"] for r in new_train})).encode()).hexdigest()},
                "asset_inventory": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                                    for p in sorted(args.model_dir.glob("*.safetensors"))],
                "runtime": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                            "gpu_name": gpu.name, "gpu_total_memory_bytes": gpu.total_memory,
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                            "attention_backend": cfg["model"]["attention_backend"]},
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cumulative_budget_start_utc": datetime.fromtimestamp(budget_start_epoch,timezone.utc).isoformat(),
                "cumulative_elapsed_before_launch_seconds": elapsed_before_launch,
                "remaining_wall_seconds_at_launch": remaining,
                "main_test_touched": False}
    cpu_fixture = cpu_state_fixture()
    manifest["cpu_fixture"] = cpu_fixture
    manifest["token_boundary_check"] = {"completion_tokens": 2,
        "shared_marker_token_id": int(labels[0][0]),
        "class_token_ids": [int(x[1]) for x in labels], "full_vocab_scoring": True}
    (args.out_dir / "dev-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    from nested_sleep.lm import LlamaCMSLM, promote_mutable_to_fp32
    backbone = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    model = LlamaCMSLM(backbone, rank=cfg["rank"], capacity=cfg["capacity"], levels=cfg["levels"])
    promote_mutable_to_fp32(model)
    model.eval().requires_grad_(False)
    pad_id = tokenizer.pad_token_id
    init = snapshot(model)
    common_init_sha = json_hash({k: tensor_sha256(v) for k, v in sorted(init.items())})
    meter = WorkMeter()
    meter_token = ACTIVE_METER.set(meter)
    started_all = time.perf_counter()
    results = []
    selected = None
    selected_index = None
    selected_state = None
    selected_controller_state = None
    hard_cap_hit = False
    fatal_failure = None
    candidate_qualifications_complete = 0
    try:
        for cidx, (base_lr, expert_lr) in enumerate(cfg["candidate_lr_pairs"]):
            load_snapshot(model, init)
            torch.manual_seed(cfg["seed"])
            torch.cuda.manual_seed_all(cfg["seed"])
            controller = CMSController(model, periods=cfg["cms_periods"], base_lr=base_lr,
                                       expert_lr=expert_lr)
            rng = random.Random(cfg["seed"])
            result = {"candidate": cidx, "base_lr": base_lr, "expert_lr": expert_lr,
                      "status": "started", "initial_adapter_sha256": common_init_sha}
            cstart = time.perf_counter()
            training_nonpad_tokens = 0
            torch.cuda.reset_peak_memory_stats()
            cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            try:
                clock.check()
                meter.stage = f"dev_score_candidate_{cidx}"
                old_pre = {"train": score(model, old_train, pad_id),
                           "calibration": score(model, old_cal, pad_id)}
                for i in range(cfg["foundation_steps"]):
                    clock.check()
                    batch = [old_train[rng.randrange(len(old_train))] for _ in range(cfg["batch_size"])]
                    event = step(model, controller, batch, pad_id, meter, f"dev_foundation_candidate_{cidx}")
                    training_nonpad_tokens += event["nonpad_tokens"]
                clock.check()
                meter.stage = f"dev_score_candidate_{cidx}"
                old_post = {"train": score(model, old_train, pad_id),
                            "calibration": score(model, old_cal, pad_id)}
                cuda_end.record(); torch.cuda.synchronize()
                candidate_stages = {k:v for k,v in meter.counts.items()
                                    if k in (f"dev_score_candidate_{cidx}", f"dev_foundation_candidate_{cidx}")}
                result.update({"status": "completed", "foundation_updates": cfg["foundation_steps"],
                               "old_pre": old_pre, "old_post": old_post,
                               "bank_counts_at_foundation": active_parameter_counts(model),
                               "cms_final_counts": list(controller.counts),
                               "cms_events": controller.events,
                               "supervised_nonpad_forward_tokens": training_nonpad_tokens,
                               "cuda_event_ms": cuda_start.elapsed_time(cuda_end),
                               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "work": candidate_stages,
                               "seconds": time.perf_counter() - cstart,
                               "qualification_status": "not_evaluated_yet"})
                score_path = args.out_dir / f"candidate-{cidx+1:02d}-scores.json"
                write_json(score_path, {"schema":"x870e-candidate-scores-v1",
                    "candidate":cidx,"base_lr":base_lr,"expert_lr":expert_lr,
                    "old_pre":old_pre,"old_post":old_post,
                    "foundation_updates":cfg["foundation_steps"],
                    "cms_final_counts":list(controller.counts),"cms_events":controller.events,
                    "supervised_nonpad_forward_tokens":training_nonpad_tokens,
                    "work":candidate_stages,"cuda_event_ms":result["cuda_event_ms"],
                    "wall_seconds":result["seconds"],"qualification_status":"pending"})
                checkpoint = {"schema":1,"candidate":cidx,"base_lr":base_lr,"expert_lr":expert_lr,
                    "adapter":snapshot(model),"cms":controller.state_dict(),
                    "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
                    "torch_cpu_rng":torch.get_rng_state(),"torch_cuda_rng":torch.cuda.get_rng_state_all(),
                    "config_sha256":cfg_sha,"initial_adapter_sha256":common_init_sha}
                checkpoint_path=args.out_dir / f"candidate-{cidx+1:02d}-foundation-private.pt"
                torch.save(checkpoint,checkpoint_path)
                result["score_artifact"]=score_path.name
                result["private_checkpoint"]={"name":checkpoint_path.name,"bytes":checkpoint_path.stat().st_size,
                                              "sha256":sha256(checkpoint_path)}
                result["qualifies"] = candidate_qualifies(old_post)
                result["qualification_status"] = "qualified" if result["qualifies"] else "did_not_qualify"
                candidate_qualifications_complete += 1
                write_json(args.out_dir / f"candidate-{cidx+1:02d}-qualification.json",
                           {"candidate":cidx,"qualifies":result["qualifies"],
                            "train_threshold":.80,"calibration_threshold":.50,
                            "train_two_token_em":old_post["train"]["full_vocab_two_token_greedy_exact_match"],
                            "calibration_two_token_em":old_post["calibration"]["full_vocab_two_token_greedy_exact_match"]})
                results.append(result)
                if result["qualifies"]:
                    if selected is None:
                        selected = (base_lr, expert_lr)
                        selected_index = cidx
                        selected_state = snapshot(model)
                        selected_controller_state = controller.state_dict()
            except TimeoutError as e:
                hard_cap_hit = True
                result.update({"status": "hard_cap_interrupted", "error": str(e),
                               "work": deepcopy(meter.counts), "seconds": time.perf_counter()-cstart})
                results.append(result)
                break
            except Exception as e:
                failure_class = "numerical_candidate_failure" if "nonfinite" in str(e).lower() else "engineering_failure"
                result.update({"status": failure_class, "error_type": type(e).__name__, "error": str(e),
                               "work": deepcopy(meter.counts), "seconds": time.perf_counter()-cstart})
                results.append(result)
                fatal_failure = {"class":failure_class,"candidate":cidx,"type":type(e).__name__,"message":str(e)}
                break
        all_qualified = candidate_qualifications_complete == len(cfg["candidate_lr_pairs"])
        if selected is not None and not hard_cap_hit and fatal_failure is None and all_qualified:
            # Restore the already sealed lowest-qualifying foundation snapshot;
            # no replayed training or evaluation-based extension is performed.
            load_snapshot(model, selected_state)
            base_lr, expert_lr = selected
            controller = CMSController.restore(model, selected_controller_state)
            # Capture old and new task training windows; one simultaneous due
            # at event64 exercises the local slow-first/prospective-fast rule.
            try:
                final_model, final_controller, fixture = sleep_transaction_fixture(
                    model, controller, old_train, new_train, pad_id, args.out_dir, meter,
                    cfg["seed"], clock, marker_id=int(labels[0][0]))
            except TimeoutError as e:
                hard_cap_hit = True
                result = {"status": "hard_cap_interrupted_during_selected_sleep_fixture",
                          "error": str(e), "selected_candidate": selected_index,
                          "selected_lr_pair": list(selected), "all_candidate_qualification": results,
                          "work_total": deepcopy(meter.counts),
                          "elapsed_seconds": time.perf_counter()-started_all,
                          "fixture_checkpoint_may_exist_private_only": True,
                          "training_test_role_access": False}
                final_model = final_controller = fixture = None
            if hard_cap_hit:
                # Keep the actual completed work/cost ledger. No endpoint-based
                # continuation and no score-driven change to the fixed budget.
                pass
            else:
                # Compute-proxy calibration against a balanced 2-old/2-new
                # supervised labeled batch from train roles only.
                clock.check()
                proxy_sleep = sum(v["forward_input_tokens"] + 2*v["gradient_forward_input_tokens"]
                                  for k,v in meter.counts.items() if k.startswith("dev_sleep_transaction"))
                mixed_batch = [old_train[i] for i in (0, 4)] + [new_train[i] for i in (0, 4)]
                step(final_model, final_controller, mixed_batch, pad_id, meter, "extra_replay_batch_calibration")
                after_proxy = meter.counts["extra_replay_batch_calibration"]
                proxy_batch = after_proxy["forward_input_tokens"] + 2*after_proxy["gradient_forward_input_tokens"]
                extra_batches = max(4, int(round((proxy_sleep / max(proxy_batch, 1)) / 4.0)) * 4)
                fixture["sleep"]["sleep_compute_proxy_fw_plus_2gradfw_tokens"] = proxy_sleep
                fixture["sleep"]["one_mixed_labeled_batch_proxy"] = proxy_batch
                fixture["sleep"]["extra_replay_batches_per_event_calibrated"] = extra_batches
                fixture["sleep"]["extra_replay_proxy_tokens"] = extra_batches * proxy_batch
                fixture["sleep"]["extra_replay_proxy_ratio"] = (extra_batches * proxy_batch / proxy_sleep) if proxy_sleep else None
                fixture["sleep"]["interpretation"] = "proxy-calibrated compute and prompt-exposure comparison; differs in teacher targets vs explicit old/new labels, not a mechanism-isolated comparison or exact FLOP match"
                result = {"status": "completed", "selected_candidate": selected_index,
                          "selected_lr_pair": list(selected), "new_wake_updates": 64,
                          "engineering_fixture": fixture, "all_candidate_qualification": results,
                          "work_total": deepcopy(meter.counts),
                          "hard_cap_seconds": cfg["budget"]["hard_wall_seconds"],
                          "elapsed_seconds": time.perf_counter()-started_all,
                          "training_test_role_access": False}
        elif fatal_failure is not None:
            result = {"status":fatal_failure["class"],"failure":fatal_failure,
                      "all_candidate_qualification":results,"work_total":deepcopy(meter.counts),
                      "training_test_role_access":False,"no_silent_continue":True}
        elif hard_cap_hit:
            result = {"status": "hard_cap_before_selected_sleep_fixture",
                      "selected_candidate": selected_index, "selected_lr_pair": selected,
                      "all_candidate_qualification": results, "work_total": deepcopy(meter.counts),
                      "elapsed_seconds": time.perf_counter()-started_all,
                      "training_test_role_access": False, "no_endpoint_extension": True}
        elif all_qualified:
            result = {"status": "no_lr_pair_met_fixed_train_calibration_gate",
                      "all_candidate_qualification": results,
                      "work_total": deepcopy(meter.counts),
                      "hard_cap_seconds": cfg["budget"]["hard_wall_seconds"],
                      "elapsed_seconds": time.perf_counter()-started_all,
                      "no_grid_expansion": True, "training_test_role_access": False}
        else:
            result = {"status":"qualification_grid_incomplete",
                      "all_candidate_qualification":results,"work_total":deepcopy(meter.counts),
                      "training_test_role_access":False}
    finally:
        ACTIVE_METER.reset(meter_token)
    result["cumulative_wall_elapsed_seconds"] = max(0., time.time() - budget_start_epoch)
    result["cumulative_wall_remaining_seconds"] = max(0., budget_start_epoch + cfg["budget"]["hard_wall_seconds"] - time.time())
    result["cumulative_wall_hard_stop_utc"] = datetime.fromtimestamp(
        budget_start_epoch + cfg["budget"]["hard_wall_seconds"], timezone.utc).isoformat()
    result["prior_attempt_foundation_updates"] = args.prior_attempt_foundation_updates
    result["foundation_updates_this_attempt"] = sum(int(x.get("foundation_updates",0)) for x in results)
    result["total_foundation_updates_including_prior_attempts"] = (
        args.prior_attempt_foundation_updates + result["foundation_updates_this_attempt"])
    (args.out_dir / "dev-results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "candidate_count": len(results), "selected": result.get("selected_lr_pair")}), flush=True)


if __name__ == "__main__":
    main()
