#!/usr/bin/env python3
"""Zero-update boundary scorer for the completed controlled CLINC probe.

Each of the 16 Sleep commits in each arm is scored on the same 60 clean,
train-deduplicated development examples. Four contexts separate the actual
pre-Sleep context, a base-only proposal application, a reset/expanded initial
Sleep student (a reconstruction, not a trained checkpoint), and the actual
post-Sleep checkpoint. No wake controller is resumed and no optimizer step is
performed. This is a read-only development diagnostic, not a test evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


SCHEMA = "clinc-controlled-boundary-evaluation-v3"
PROBE_SCHEMA = "clinc-v8-controlled-routing-probe"
CHECKPOINT_SCHEMA = "clinc-v8-controlled-probe-state-v1"
ARMS = ("SLEEP_MIX", "GOLD_CE_MIX")
EVENTS_PER_ARM = 16
CONTEXTS_PER_EVENT = ("teacher_pre", "base_only_reconstruction",
                      "initial_reset_expanded_reconstruction", "optimized_post")
MAX_FORWARD_CALLS = 16_000
MAX_WALL_SECONDS = 900


def append_progress_record(path: Path, record: dict, *, first=False):
    """Keep every completed pair if a later budget or I/O failure interrupts."""
    if not first and not path.is_file():
        raise FileNotFoundError("progress header must exist before appending")
    with path.open("x" if first else "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _value_sha(value) -> str:
    """Stable hash for the checkpoint's mutable state, without exporting it."""
    import torch
    digest = hashlib.sha256()

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                digest.update(str(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            digest.update(type(obj).__name__.encode())
            for child in obj:
                visit(child)
        elif obj is None or isinstance(obj, (str, int, float, bool, bytes)):
            digest.update(repr(obj).encode())
        else:
            raise TypeError(f"unsupported checkpoint-hash value: {type(obj).__name__}")
    visit(value)
    return digest.hexdigest()


def checkpoint_pairs(items, *, expected_arms=ARMS, events_per_arm=EVENTS_PER_ARM):
    """Validate explicit pre/post phase history and return ordered pairs.

    Items have arm, side, phase, wake_step, wake_tick, sha256, path, and event.
    This function deliberately validates event identity, sender level, and the
    post checkpoint's pending-level metadata; a post state is not assumed safe
    to resume merely because its event committed.
    """
    import re
    pattern = re.compile(r"^(pre|post)-sleep-level(\d+)-event(\d{3})$")
    indexed = {}
    for item in items:
        match = pattern.fullmatch(str(item.get("phase", "")))
        if not match:
            raise ValueError("malformed boundary phase")
        side, level_text, event_text = match.groups()
        if item.get("side") != side:
            raise ValueError("phase side does not match enumerated checkpoint side")
        arm, level, event = str(item.get("arm")), int(level_text), int(event_text)
        key = (arm, event, side)
        if key in indexed:
            raise ValueError(f"duplicate boundary phase: {key}")
        metadata = item.get("event")
        if not isinstance(metadata, dict):
            raise ValueError(f"missing event metadata: {key}")
        event_level = metadata.get("level", metadata.get("sender_level"))
        if event_level is None or int(event_level) != level:
            raise ValueError(f"phase/event level mismatch: {key}")
        event_id = metadata.get("event_index") if side == "pre" else metadata.get("completed_event_index")
        if event_id is None or int(event_id) != event:
            raise ValueError(f"phase/event index mismatch: {key}")
        indexed[key] = item

    pairs = []
    for arm in expected_arms:
        last_tick = -1
        for event in range(events_per_arm):
            pre, post = indexed.get((arm, event, "pre")), indexed.get((arm, event, "post"))
            if pre is None or post is None:
                raise ValueError(f"missing boundary checkpoint for {arm} event {event:03d}")
            if int(pre["wake_step"]) != int(post["wake_step"]):
                raise ValueError("pre/post wake step mismatch")
            if int(pre["wake_tick"]) != int(post["wake_tick"]):
                raise ValueError("pre/post global wake tick mismatch")
            # Multiple sender levels can be due on the same wake tick; event
            # indices disambiguate those transactions, so equal ticks are valid.
            if int(pre["wake_tick"]) < last_tick:
                raise ValueError("boundary ticks moved backwards")
            last_tick = int(pre["wake_tick"])
            pre_level = int(pre["event"].get("level", -1))
            post_level = int(post["event"].get("sender_level", -1))
            if pre_level != post_level:
                raise ValueError("pre/post sender level mismatch")
            pending = post["event"].get("other_due_levels_pending")
            safe = post["event"].get("safe_to_resume_at_next_wake")
            if not isinstance(pending, list):
                raise ValueError("post event lacks other-due-level list")
            if safe is None or bool(safe) != (len(pending) == 0):
                raise ValueError("post pending/safe metadata mismatch")
            if pre.get("parent_sha256") != post.get("parent_sha256"):
                raise ValueError("pre/post parent checkpoint mismatch")
            pairs.append({"arm": arm, "event_index": event, "level": pre_level,
                          "wake_step": int(pre["wake_step"]), "wake_tick": int(pre["wake_tick"]),
                          "pre": pre, "post": post})
    if set(key[0] for key in indexed) != set(expected_arms):
        raise ValueError("unexpected arm in boundary checkpoint set")
    if len(indexed) != 2 * len(expected_arms) * events_per_arm:
        raise ValueError("unexpected extra/incomplete boundary checkpoints")
    return pairs


def summarize_pair(metrics_by_context: dict) -> dict:
    """Compute endpoint and staged deltas using aggregate-only metrics."""
    required = set(CONTEXTS_PER_EVENT)
    if set(metrics_by_context) != required:
        raise ValueError("boundary summary requires all four explicitly named contexts")
    groups = {}
    for group in ("old", "current", "seen"):
        values = {name: metrics_by_context[name][group] for name in CONTEXTS_PER_EVENT}
        out = {name: value.get("accuracy") for name, value in values.items()}
        out["base_only_delta_pp"] = _delta_pp(out["teacher_pre"], out["base_only_reconstruction"])
        out["reset_expand_delta_pp"] = _delta_pp(out["base_only_reconstruction"],
                                                  out["initial_reset_expanded_reconstruction"])
        out["sleep_fit_delta_pp"] = _delta_pp(out["initial_reset_expanded_reconstruction"],
                                               out["optimized_post"])
        out["total_delta_pp"] = _delta_pp(out["teacher_pre"], out["optimized_post"])
        out["invalid_by_context"] = {name: values[name].get("invalid") for name in CONTEXTS_PER_EVENT}
        out["count_by_context"] = {name: values[name].get("count") for name in CONTEXTS_PER_EVENT}
        if group == "old":
            pre, post = out["teacher_pre"], out["optimized_post"]
            out["old_retention_ratio_post_over_pre"] = (
                None if pre in (None, 0) or post is None else post / pre)
        groups[group] = out
    return groups


def _delta_pp(left, right):
    return None if left is None or right is None else 100.0 * (right - left)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_source_lock(protocol: dict, project_root: Path):
    hashes = protocol.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes or canonical_sha(hashes) != protocol.get("source_lock_sha256"):
        raise ValueError("probe source lock is missing or does not match its digest")
    loaded = {}
    src = (project_root / "src").resolve()
    for raw_path, expected in hashes.items():
        path = Path(raw_path).resolve()
        if not path.is_file() or sha256_file(path).lower() != str(expected).lower():
            raise ValueError(f"locked source file changed or missing: {raw_path}")
        loaded[str(path)] = str(expected).lower()
    for suffix in ("nested_sleep/lm.py", "nested_sleep/experts.py"):
        if not any(Path(path).as_posix().endswith(suffix) and src in Path(path).parents for path in loaded):
            raise ValueError(f"source lock does not include public {suffix}")
    return loaded


def _resolve_inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"checkpoint path missing or escapes run directory: {relative}")
    return path


def _phase_checkpoint(root, arm, step, phase):
    return _resolve_inside(root, f"checkpoints/{arm}/task01-step{int(step):05d}-{phase}.pt")


def _phase_items(run_dir, protocol, metrics_rows, torch):
    commits = {}
    for row in metrics_rows:
        if row.get("event") != "controlled_sleep_commit":
            continue
        arm, receipt = row.get("arm"), row.get("checkpoint_before_event")
        if arm not in ARMS or not isinstance(receipt, dict):
            raise ValueError("invalid controlled Sleep commit receipt")
        phase = receipt.get("phase", "")
        try:
            event = int(str(phase).rsplit("event", 1)[1])
        except (IndexError, ValueError):
            raise ValueError("pre-checkpoint phase has no explicit event index")
        key = (arm, event)
        if key in commits:
            raise ValueError(f"duplicate metrics commit for {key}")
        commits[key] = row

    output = []
    expected_parent = protocol["parent_task0_checkpoint_sha256"]
    for arm in ARMS:
        for event in range(EVENTS_PER_ARM):
            row = commits.get((arm, event))
            if row is None:
                raise ValueError(f"missing metrics commit for {arm} event {event:03d}")
            pre_receipt = row["checkpoint_before_event"]
            if pre_receipt.get("parent_task0_checkpoint_sha256") != expected_parent:
                raise ValueError("pre-checkpoint receipt parent task-0 hash mismatch")
            level = int(row["sender_level"])
            pre_phase = pre_receipt["phase"]
            post_phase = f"post-sleep-level{level}-event{event:03d}"
            paths = {
                "pre": _resolve_inside(run_dir, pre_receipt["path"]),
                "post": _phase_checkpoint(run_dir, arm, pre_receipt["wake_step"], post_phase),
            }
            if sha256_file(paths["pre"]).lower() != str(pre_receipt["sha256"]).lower():
                raise ValueError("pre-checkpoint receipt file SHA mismatch")
            for side, path, phase in (("pre", paths["pre"], pre_phase),
                                      ("post", paths["post"], post_phase)):
                digest = sha256_file(path)
                state = torch.load(path, map_location="cpu", weights_only=False)
                if not isinstance(state, dict) or state.get("schema") != CHECKPOINT_SCHEMA:
                    raise ValueError(f"unsupported checkpoint schema: {path.name}")
                if (state.get("arm") != arm or state.get("source_arm") != "HOPE" or
                    state.get("seed") != protocol["config"]["seed"] or state.get("task_id") != 1):
                    raise ValueError(f"checkpoint arm/source/seed/task mismatch: {path.name}")
                if state.get("phase") != phase or int(state.get("wake_step", -1)) != int(pre_receipt["wake_step"]):
                    raise ValueError(f"checkpoint phase/step mismatch: {path.name}")
                if state.get("parent_task0_checkpoint_sha256") != expected_parent:
                    raise ValueError(f"checkpoint parent task-0 hash mismatch: {path.name}")
                if state.get("source_lock_sha256") != protocol["source_lock_sha256"]:
                    raise ValueError(f"checkpoint source lock mismatch: {path.name}")
                if state.get("config_sha256") != canonical_sha(protocol["config"]):
                    raise ValueError(f"checkpoint canonical config hash mismatch: {path.name}")
                if state.get("training_manifest_sha256") != protocol["training_manifest_sha256"]:
                    raise ValueError(f"checkpoint manifest hash mismatch: {path.name}")
                if (not isinstance(state.get("context"), dict) or "memory" not in state["context"] or
                    not isinstance(state.get("rng"), dict)):
                    raise ValueError(f"checkpoint mutable context/RNG missing: {path.name}")
                event_meta = state.get("event")
                if not isinstance(event_meta, dict):
                    raise ValueError(f"checkpoint phase event metadata missing: {path.name}")
                wake = state.get("wake")
                tick = int(getattr(wake, "ticks", -1))
                if tick < 0:
                    raise ValueError(f"checkpoint wake snapshot missing or malformed: {path.name}")
                output.append({"arm": arm, "side": side, "phase": phase,
                    "wake_step": int(pre_receipt["wake_step"]), "wake_tick": tick,
                    "sha256": digest, "path": str(path.relative_to(run_dir)),
                    "event": event_meta, "parent_sha256": expected_parent,
                    "context_sha256": _value_sha(state["context"])})
                if side == "pre":
                    if int(row.get("step", -1)) != int(pre_receipt["wake_step"]):
                        raise ValueError("metrics step differs from pre checkpoint")
                    if int(row.get("global_wake_tick", -1)) != tick:
                        raise ValueError("metrics tick differs from pre checkpoint")
                    if int(row.get("sender_level", -1)) != int(event_meta.get("level", -1)):
                        raise ValueError("metrics sender level differs from pre checkpoint")
                    if int(event_meta.get("event_index", -1)) != event:
                        raise ValueError("pre event index differs from metrics event")
                    level_clock = wake.clocks.get(level)
                    if level_clock != wake.periods[level]:
                        raise ValueError("pre checkpoint does not retain the due sender proposal")
                    pending = wake.pending
                    if not {f"memory.{level}.a", f"memory.{level}.b"}.issubset(pending):
                        raise ValueError("pre checkpoint is missing sender proposal gradients")
                else:
                    if int(event_meta.get("completed_event_index", -1)) != event:
                        raise ValueError("post event index differs from filename")
                    if int(event_meta.get("sender_level", -1)) != level:
                        raise ValueError("post sender level differs from pre/metrics")
                    # The post event metadata does not duplicate a global tick;
                    # exact pre/post tick equality is checked by checkpoint_pairs.
                    pending_levels = sorted(i for i, value in wake.clocks.items()
                                            if value == wake.periods[i])
                    if pending_levels != sorted(event_meta.get("other_due_levels_pending", [])):
                        raise ValueError("post pending due levels differ from checkpoint clocks")
                del state
    return output


def _rng_snapshot(torch, np):
    return (torch.get_rng_state().clone(),
            [x.cpu().clone() for x in torch.cuda.get_rng_state_all()],
            random.getstate(), np.random.get_state())


def _restore_rng(torch, np, state):
    cpu, cuda, py, npr = state
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(cuda)
    random.setstate(py)
    np.random.set_state(npr)


def _make_context(model, state_dict, clone_mutable_context):
    context = clone_mutable_context(model)
    context.load_state_dict(state_dict, strict=True)
    # MutableContext is a dataclass, not an nn.Module.
    context.memory.eval()
    for parameter in context.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return context


def _proposal_from_wake(context, wake, sender):
    names = (f"memory.{sender}.a", f"memory.{sender}.b")
    params = dict(context.named_parameters())
    if set(names) - set(params) or set(names) - set(wake.pending):
        raise ValueError("sender base/proposal scope mismatch")
    return {name: params[name].detach().clone() - wake.pending[name].to(params[name])
            for name in names}


def _apply_base_proposal(context, sender, proposal, torch):
    names = {f"memory.{sender}.a", f"memory.{sender}.b"}
    params = dict(context.named_parameters())
    if set(proposal) != names or names - set(params):
        raise ValueError("base-only proposal must contain exactly sender A/B")
    with torch.no_grad():
        for name, value in proposal.items():
            params[name].copy_(value.to(params[name]))


def _score_context(model, context, fixed_rows, order, tokenizer, view, cfg, torch, device, budget):
    before = _value_sha(context.state_dict())
    result = __import__("tools.run_clinc_cil_v8_r2", fromlist=["_evaluate"])._evaluate(
        model, context, fixed_rows, order, 1, tokenizer, view, cfg, torch, device, budget)
    after = _value_sha(context.state_dict())
    if before != after:
        raise RuntimeError("read-only evaluation changed mutable context")
    return result["metrics"]


def run(args):
    import numpy as np
    import torch

    started = time.monotonic()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    project_root, model_path = Path(args.project_root).resolve(), Path(args.model_path).resolve()
    pack_path = Path(args.training_pack).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if output_dir == run_dir or run_dir in output_dir.parents or output_dir in run_dir.parents:
        raise ValueError("output directory must be separate from the run")
    if output_dir == project_root or project_root in output_dir.parents or model_path in output_dir.parents:
        raise ValueError("output directory must be outside source/model trees")
    if pack_path in output_dir.parents or output_dir == pack_path.parent:
        raise ValueError("output directory must be separate from training data")
    if not model_path.is_dir() or not pack_path.is_file():
        raise FileNotFoundError("local model and training manifest are required")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    budget = None
    phase = "protocol-preflight"
    try:
        protocol_path, seal_path, metrics_path = (run_dir / "protocol.json", run_dir / "development-sealed.json",
                                                   run_dir / "metrics.jsonl")
        if not all(p.is_file() for p in (protocol_path, seal_path, metrics_path)):
            raise FileNotFoundError("complete protocol, seal, and metrics are required")
        protocol, seal = _load_json(protocol_path), _load_json(seal_path)
        if protocol.get("schema") != PROBE_SCHEMA or seal.get("complete") is not True:
            raise ValueError("probe run is incomplete or has an unsupported schema")
        if protocol.get("test_data_read") is not False or seal.get("test_data_read") is not False:
            raise ValueError("probe did not attest that test data remained unread")
        cfg = protocol.get("config")
        if (not isinstance(cfg, dict) or cfg.get("seed") != 17 or
            cfg.get("target_arms") != list(ARMS) or cfg.get("source_arm") != "HOPE"):
            raise ValueError("probe config does not match the two locked target arms")
        if protocol.get("source_arm") != "HOPE" or protocol.get("parent_task0_checkpoint_sha256") != seal.get("parent_task0_checkpoint_sha256"):
            raise ValueError("probe source arm/parent checkpoint binding mismatch")
        if protocol.get("training_manifest_sha256") != cfg.get("training_manifest_sha256"):
            raise ValueError("config and protocol data hashes differ")
        if sha256_file(pack_path).lower() != protocol["training_manifest_sha256"].lower():
            raise ValueError("training manifest SHA differs from sealed probe")
        if protocol.get("model_assets_sha256") != cfg.get("model_assets_sha256"):
            raise ValueError("model asset hash maps differ")
        for relative, expected in protocol["model_assets_sha256"].items():
            path = (model_path / relative).resolve()
            if model_path not in path.parents or not path.is_file() or sha256_file(path).lower() != expected.lower():
                raise ValueError(f"locked model asset mismatch: {relative}")
        source_hashes = _verify_source_lock(protocol, project_root)
        sealed_runs = {row["arm"]: row for row in seal.get("runs", [])}
        if set(sealed_runs) != set(ARMS):
            raise ValueError("seal must contain exactly both target arms")
        for arm in ARMS:
            if sealed_runs[arm].get("all_events_committed") is not True or sealed_runs[arm].get("sleep_event_count") != EVENTS_PER_ARM:
                raise ValueError(f"seal does not certify 16 events for {arm}")
        rows = []
        for line_number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid metrics line {line_number}") from exc
        # The locked flat module makes the pickled WakeSnapshot type resolvable.
        task_root = Path(__file__).resolve().parents[1]
        tools_dir = task_root / "tools"
        for item in (str(task_root), str(tools_dir)):
            if item in sys.path:
                sys.path.remove(item)
            sys.path.insert(0, item)
        import clinc_lifecycle_v8  # noqa: F401
        phase = "checkpoint-preflight"
        phase_items = _phase_items(run_dir, protocol, rows, torch)
        pairs = checkpoint_pairs(phase_items)
        if len(pairs) != 32:
            raise ValueError("exactly 32 event pairs are required")
        own_path = Path(__file__).resolve()
        own_sha = sha256_file(own_path)

        phase = "data-and-model-load"
        from tools import run_clinc_cil_v8_r2 as r2
        import clinc_cil_common_v8 as common
        import clinc_tail_runtime_v8 as tail
        from clinc_tail_runtime_v8 import clone_mutable_context
        if Path(r2.__file__).resolve() != tools_dir / "run_clinc_cil_v8_r2.py":
            raise RuntimeError("R2 evaluator import path mismatch")
        if Path(common.__file__).resolve() != tools_dir / "clinc_cil_common_v8.py" or Path(tail.__file__).resolve() != tools_dir / "clinc_tail_runtime_v8.py":
            raise RuntimeError("common/tail helper import path mismatch")
        manifest = _load_json(pack_path)
        view = common.make_training_view(manifest)
        train_rows = list(common.iter_role(view, "train", pack_path.parent))
        development_rows = list(common.iter_role(view, "development", pack_path.parent))
        train_texts = {r2.normalize_text(row["text"]) for row in train_rows}
        clean = [row for row in development_rows if r2.normalize_text(row["text"]) not in train_texts]
        order = view.class_orders[str(cfg["seed"])]
        def first_two(class_ids):
            selected = []
            for class_id in order:
                if class_id in class_ids:
                    candidates = sorted((row for row in clean if int(row["intent_id"]) == class_id),
                                        key=lambda row: row["id"])
                    if len(candidates) < 2:
                        raise ValueError(f"class {class_id} has fewer than two clean development rows")
                    selected.extend(candidates[:2])
            return selected
        old_ids, current_ids = set(order[:15]), set(order[15:30])
        old_rows, current_rows = first_two(old_ids), first_two(current_ids)
        fixed_rows = old_rows + current_rows
        if len(fixed_rows) != 60 or len({row["id"] for row in fixed_rows}) != 60:
            raise ValueError("fixed scoring subset must contain 60 unique rows")
        scoring_scope = {"old_count": len(old_rows), "current_count": len(current_rows), "seen_count": len(fixed_rows),
            "selection": "first two clean development rows per seen class sorted by stable record ID",
            "old_class_ids_sha256": canonical_sha(sorted(old_ids)),
            "current_class_ids_sha256": canonical_sha(sorted(current_ids)),
            "row_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in fixed_rows).encode()).hexdigest(),
            "label_map_size": len(view.classes), "task_id_or_candidate_mask": False, "test_data_read": False}
        cfg_eval = dict(cfg)
        cfg_eval.update({"max_forward_calls": MAX_FORWARD_CALLS, "max_wall_seconds": MAX_WALL_SECONDS,
                         "max_update_steps": 0, "eval_examples_per_class": 2})
        seed = int(cfg["initialization_seed"])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch, tokenizer, model, immutable_versions = r2._load_common(
            project_root, model_path, cfg["device"], cfg_eval)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        frozen_before = immutable_versions(model)
        budget = r2.Budget(cfg_eval)
        budget.started = started
        original_logits, original_tail = model.logits, tail.tail_logits
        def counted_logits(tokens, mask=None):
            budget.check()
            if budget.forward_calls >= MAX_FORWARD_CALLS:
                raise RuntimeError("full-backbone forward cap reached before next call")
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_logits(tokens, mask)
        def counted_tail(tail_model, context, cache):
            budget.tail(cache, torch.is_grad_enabled() and any(p.requires_grad for p in context.parameters()))
            return original_tail(tail_model, context, cache)
        model.logits = counted_logits
        tail.tail_logits = counted_tail

        phase = "paired-counterfactual-scoring"
        evaluated = []
        def make_rng_guard():
            return _rng_snapshot(torch, np)
        progress_path = output_dir / "progress.jsonl"
        append_progress_record(progress_path, {
            "kind": "header", "schema": SCHEMA, "scorer_sha256": own_sha,
            "probe_protocol_sha256": sha256_file(protocol_path),
            "input_source_lock_sha256": protocol["source_lock_sha256"],
            "training_manifest_sha256": protocol["training_manifest_sha256"],
            "scoring_scope": scoring_scope, "expected_pairs": len(pairs),
            "max_forward_calls": MAX_FORWARD_CALLS, "max_wall_seconds": MAX_WALL_SECONDS,
            "budget_amendment": "v2 stopped at 10000 forwards before all fixed pairs completed; only resource allowance/progress persistence changed",
            "optimizer_steps": 0, "test_data_read": False}, first=True)
        for pair_index, pair in enumerate(pairs):
            if time.monotonic() - started > MAX_WALL_SECONDS:
                raise TimeoutError("900-second wall cap reached between event pairs")
            metrics = {}
            context_hashes = {}
            actual_states = {}
            for side in ("pre", "post"):
                item = pair[side]
                cp_path = run_dir / item["path"]
                state = torch.load(cp_path, map_location="cpu", weights_only=False)
                if sha256_file(cp_path) != item["sha256"]:
                    raise ValueError("checkpoint changed after preflight hash verification")
                actual_states[side] = state
            pre_state, post_state = actual_states["pre"], actual_states["post"]
            pre_context = _make_context(model, pre_state["context"], clone_mutable_context)
            # Base-only applies the absolute proposal to an isolated clone; no route,
            # slot reset, transaction, optimizer, or wake-controller state changes.
            proposal = _proposal_from_wake(pre_context, pre_state["wake"], pair["level"])
            base_only = _make_context(model, pre_state["context"], clone_mutable_context)
            _apply_base_proposal(base_only, pair["level"], proposal, torch)
            # Initial reset/expanded is reconstructed via the same controlled
            # transaction constructor and exact event seed, with no optimizer step.
            saved_rng = make_rng_guard()
            from clinc_controlled_routing_v8 import ControlledSleepTransaction
            transaction = ControlledSleepTransaction(pre_context, pair["level"], proposal,
                lr=cfg["sleep_lr"], weight_decay=cfg["sleep_weight_decay"],
                clip_norm=cfg["sleep_clip_norm"],
                seed=int(cfg["seed"]) + int(pair["wake_step"]) * 100 + int(pair["level"]))
            initial = transaction.student
            _restore_rng(torch, np, saved_rng)
            if transaction.counters["adamw_steps"] != 0 or transaction.optimizer.state:
                raise RuntimeError("initial counterfactual unexpectedly used optimizer state")
            post_context = _make_context(model, post_state["context"], clone_mutable_context)
            contexts = {"teacher_pre": pre_context,
                "base_only_reconstruction": base_only,
                "initial_reset_expanded_reconstruction": initial,
                "optimized_post": post_context}
            context_hashes = {name: _value_sha(context.state_dict()) for name, context in contexts.items()}
            for name in CONTEXTS_PER_EVENT:
                if time.monotonic() - started > MAX_WALL_SECONDS:
                    raise TimeoutError("900-second wall cap reached before next context score")
                metrics[name] = _score_context(model, contexts[name], fixed_rows, order,
                                               tokenizer, view, cfg_eval, torch, cfg["device"], budget)
                budget.check()
            if transaction.counters["adamw_steps"] != 0 or transaction.optimizer.state:
                raise RuntimeError("counterfactual transaction performed an Adam update")
            evaluated.append({"arm": pair["arm"], "event_index": pair["event_index"],
                "sender_level": pair["level"], "wake_step": pair["wake_step"],
                "global_wake_tick": pair["wake_tick"],
                "pre_checkpoint_sha256": pair["pre"]["sha256"],
                "post_checkpoint_sha256": pair["post"]["sha256"],
                "pre_mutable_state_sha256": pair["pre"]["context_sha256"],
                "post_mutable_state_sha256": pair["post"]["context_sha256"],
                "reconstruction_mutable_state_sha256": {
                    "base_only_reconstruction": context_hashes["base_only_reconstruction"],
                    "initial_reset_expanded_reconstruction": context_hashes["initial_reset_expanded_reconstruction"]},
                "metrics": metrics, "staged_changes": summarize_pair(metrics),
                "post_other_due_levels_pending": pair["post"]["event"].get("other_due_levels_pending", []),
                "post_safe_to_resume_next_wake": pair["post"]["event"].get("safe_to_resume_at_next_wake")})
            append_progress_record(progress_path, {"kind": "completed_event_pair",
                "pair": evaluated[-1], "cumulative_work": budget.as_dict()})
            del contexts, transaction, pre_context, base_only, post_context, initial, actual_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (pair_index + 1) % 4 == 0:
                print(json.dumps({"completed_pairs": pair_index + 1, "of": len(pairs),
                                  "forward_calls": budget.forward_calls,
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        if immutable_versions(model) != frozen_before:
            raise RuntimeError("frozen model changed during read-only scoring")
        if budget.update_steps or budget.backward_calls:
            raise RuntimeError("read-only evaluator performed a gradient/update step")
        result = {"schema": SCHEMA, "status": "complete", "test_data_read": False,
            "zero_optimizer_steps": True, "zero_wake_updates_or_resume": True,
            "counterfactuals_are_reconstructions_not_training_checkpoints": True,
            "four_context_order": list(CONTEXTS_PER_EVENT),
            "probe_protocol_sha256": sha256_file(protocol_path), "probe_seal_sha256": sha256_file(seal_path),
            "probe_metrics_sha256": sha256_file(metrics_path), "probe_source_lock_sha256": protocol["source_lock_sha256"],
            "probe_source_hashes_verified": source_hashes,
            "evaluator_source_sha256": own_sha, "training_manifest_sha256": protocol["training_manifest_sha256"],
            "model_revision": protocol["model_revision"], "model_assets_sha256": protocol["model_assets_sha256"],
            "config_sha256": canonical_sha(cfg), "parent_task0_checkpoint_sha256": protocol["parent_task0_checkpoint_sha256"],
            "scoring_scope": scoring_scope, "paired_events": evaluated,
            "work": budget.as_dict(), "forward_cap": MAX_FORWARD_CALLS,
            "elapsed_seconds": time.monotonic() - started, "wall_cap_seconds": MAX_WALL_SECONDS,
            "immutable_backbone_unchanged": True,
            "interpretation": "teacher_pre→base-only reconstructs only the due sender proposal; initial-reset-expanded reconstructs the controlled transaction's reset/fresh slot/policy before optimization; optimized_post is the actual committed post checkpoint. Counterfactuals are not training checkpoints and are not paper Sleep claims."}
        (output_dir / "evaluation.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                                                     encoding="utf-8", newline="\n")
        return result
    except BaseException as exc:
        failure = {"schema": SCHEMA, "status": "failed", "phase": phase,
            "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
            "work": budget.as_dict() if budget is not None else None,
            "optimizer_steps": 0, "wake_updates": 0, "test_data_read": False,
            "automatic_retry": False}
        (output_dir / "failure.json").write_text(json.dumps(failure, sort_keys=True, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--training-pack", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"status": result["status"], "paired_events": len(result["paired_events"]),
        "full_backbone_forward_calls": result["work"]["full_backbone_forward_calls"],
        "elapsed_seconds": result["elapsed_seconds"], "output_dir": str(Path(args.output_dir).resolve())},
        sort_keys=True))


if __name__ == "__main__":
    main()
