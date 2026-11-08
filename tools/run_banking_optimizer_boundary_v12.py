"""Single-boundary Banking77 receiver-optimizer diagnostic (development only).

This runner is deliberately separate from the frozen v10/v11 training drivers.
It replays the first actually-due level-0 boundary, builds fixed teacher-policy
rollouts, and compares request-batch aggregation/receiver LR. It is not a full
Sleep lifecycle or paper-faithful reproduction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time


SCHEMA = "banking-optimizer-boundary-v12-config-v1"
CONFIG_V10_SHA = "785ded5cc9d205dc3cbc6d25179abf6fd0d589ce6943a42626aa1692b54dfd9d"
MANIFEST_SHA = "ed9f4b047768021cbf16a3fa042db12baf212038f89f33a00099f2e32b7300f3"
SEEDS = (17, 29, 43)
ARMS = (
    {"name": "B1_HIGH", "effective_batch_size": 1, "learning_rate": 1e-3,
     "max_requests": 960, "checkpoint_requests": [30, 960]},
    {"name": "B1_LOW", "effective_batch_size": 1, "learning_rate": 5e-6,
     "max_requests": 960, "checkpoint_requests": [30, 960]},
    {"name": "B32_HIGH", "effective_batch_size": 32, "learning_rate": 1e-3,
     "max_requests": 960, "checkpoint_requests": [960]},
    {"name": "B32_LOW", "effective_batch_size": 32, "learning_rate": 5e-6,
     "max_requests": 3200, "checkpoint_requests": [960, 3200]},
)
SOURCE_HASH_MINIMUM = {
    "tools/run_banking_optimizer_boundary_v12.py",
    "tools/banking_cil_v12_batching.py",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_state_sha(value, torch) -> str:
    """Hash nested snapshots by their content, independent of torch container bytes."""
    def plain(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return {"__tensor__": True, "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        if is_dataclass(item) and not isinstance(item, type):
            return {"__dataclass__": type(item).__qualname__,
                    "fields": {field.name: plain(getattr(item, field.name))
                               for field in fields(item)}}
        if isinstance(item, dict):
            return {str(key): plain(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (tuple, list)):
            return [plain(x) for x in item]
        try:
            import numpy as np
            if isinstance(item, np.ndarray):
                return {"__ndarray__": True, "dtype": str(item.dtype),
                        "shape": list(item.shape),
                        "sha256": hashlib.sha256(item.tobytes()).hexdigest()}
            if isinstance(item, np.generic):
                return item.item()
        except ImportError:
            pass
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return {"__type__": type(item).__qualname__, "__repr__": repr(item)}
    return canonical_sha(plain(value))


def arm_request_update_totals(config: dict) -> dict:
    """Pure budget calculation; qualifications are accounted separately."""
    arms = config.get("arms")
    if not isinstance(arms, list) or len(arms) != 4:
        raise ValueError("V12 requires the four frozen factorial arms")
    requests = updates = 0
    names = set()
    for arm in arms:
        if not isinstance(arm, dict) or arm.get("name") in names:
            raise ValueError("malformed or duplicate V12 arm")
        names.add(arm.get("name"))
        batch, amount = arm.get("effective_batch_size"), arm.get("max_requests")
        checkpoints = arm.get("checkpoint_requests")
        if (type(batch) is not int or batch not in (1, 32) or type(amount) is not int
                or amount < 1 or amount % batch or not isinstance(checkpoints, list)
                or not checkpoints or checkpoints != sorted(set(checkpoints))
                or any(type(x) is not int or x < 1 or x > amount for x in checkpoints)
                or checkpoints[-1] != amount):
            raise ValueError("arm request/update/checkpoint budget is malformed")
        requests += amount
        updates += amount // batch
    if names != {x["name"] for x in ARMS}:
        raise ValueError("V12 arm names differ from the frozen comparison")
    return {"request_exposures_per_seed": requests,
            "physical_adamw_steps_per_seed": updates}


def _budget_delta(before: dict, after: dict) -> dict:
    result = {}
    for key, value in after.items():
        old = before.get(key)
        if isinstance(value, (int, float)) and isinstance(old, (int, float)):
            result[key] = value - old
        elif isinstance(value, dict) and isinstance(old, dict):
            result[key] = {name: count - old.get(name, 0)
                           for name, count in value.items()
                           if isinstance(count, (int, float))}
    return result


def validate_config(config: dict, *, require_frozen: bool) -> dict:
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("V12 config schema mismatch")
    if config.get("status") not in (("frozen",) if require_frozen else ("draft", "frozen")):
        raise ValueError("V12 execution requires a frozen config")
    if config.get("v10_config_file_sha256") != CONFIG_V10_SHA:
        raise ValueError("V12 must bind the reviewed v10 config bytes")
    if config.get("training_manifest_sha256") != MANIFEST_SHA:
        raise ValueError("V12 must bind the reviewed train/dev manifest")
    if tuple(config.get("seeds", ())) != SEEDS:
        raise ValueError("V12 seeds must be exactly 17, 29, and 43")
    if (config.get("task_id"), config.get("sender")) != (1, 0):
        raise ValueError("V12 is fixed to task 1 and sender level 0")
    if config.get("boundary_selection") != "first_level0_due_after_foundation_no_performance_selection":
        raise ValueError("V12 boundary must be selected by first actual due event")
    if type(config.get("max_prefix_wake_ticks")) is not int or config["max_prefix_wake_ticks"] < 1:
        raise ValueError("V12 prefix wake bound must be positive")
    if (config.get("fixed_teacher_rollout_requests"), config.get("rollout_microbatch_size"),
            config.get("max_new_tokens")) != (3200, 8, 10):
        raise ValueError("fixed rollout count, microbatch, or token cap changed")
    if config.get("request_selection") != "v10_balanced_seen_class_cycle_gkd_only":
        raise ValueError("V12 request selector changed")
    if config.get("objective") != "fixed_teacher_policy_full_vocab_KL_T1_mean_tokens_then_mean_requests":
        raise ValueError("V12 objective/weighting changed")
    expected_arms = [{**arm, "learning_rate": float(arm["learning_rate"])} for arm in ARMS]
    actual_arms = [{**arm, "learning_rate": float(arm["learning_rate"])}
                   for arm in config.get("arms", [])]
    if actual_arms != expected_arms:
        raise ValueError("V12 factorial arms or checkpoints differ from the reviewed design")
    if config.get("execution_order") != ["B32_LOW", "B1_HIGH", "B1_LOW", "B32_HIGH"]:
        raise ValueError("V12 arm execution order changed")
    if (config.get("weight_decay"), config.get("clip_norm")) != (0.0, 1.0):
        raise ValueError("V12 receiver optimizer regularization changed")
    if config.get("evaluation") != {"split": "development_only", "old_rows": 110,
            "current_rows": 110, "candidate_labels": 77, "include_greedy": True,
            "candidate_batch_size": 32}:
        raise ValueError("V12 evaluation cohort/settings changed")
    qualification = config.get("restore_qualification")
    if qualification != {"seed": 17, "arm": "B32_LOW", "snapshot_after_requests": 32,
                        "compare_after_requests": 64, "require_original_vs_disk_restore": True}:
        raise ValueError("V12 restore qualification contract changed")
    if config.get("post_fit_wake_ticks") != 0 or config.get("test_data_read") is not False:
        raise ValueError("V12 must remain single-boundary and development-only")
    for key in ("max_wall_seconds", "max_forward_calls", "max_update_steps"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"V12 budget {key} must be a positive integer")
    hashes = config.get("source_hashes")
    if not isinstance(hashes, dict) or (require_frozen and not SOURCE_HASH_MINIMUM.issubset(hashes)):
        raise ValueError("V12 frozen config must pin runner and batch helper hashes")
    if any(not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
           or any(ch not in "0123456789abcdef" for ch in digest.lower())
           for name, digest in hashes.items()):
        raise ValueError("malformed V12 source hash map")
    arm_request_update_totals(config)
    return config


def select_first_due_level0(wake, max_ticks: int, tick) -> tuple[int, dict]:
    """Run next task ticks until level zero is actually due, not at a guessed tick."""
    if type(max_ticks) is not int or max_ticks < 1 or not callable(tick):
        raise ValueError("invalid first-due search arguments")
    steps = 0
    while steps < max_ticks:
        tick()
        steps += 1
        proposals = wake.proposals()
        if 0 in proposals:
            return steps, proposals
        for due_level, due_proposal in sorted(proposals.items()):
            wake.apply(due_level, due_proposal)
    raise RuntimeError("level zero did not become due within the frozen prefix bound")


def fixed_teacher_request_rows(driver, train_rows, order, request_count: int):
    if type(request_count) is not int or request_count != 3200:
        raise ValueError("fixed request stream must contain exactly 3,200 requests")
    pools = driver._group_sleep_pools(train_rows, order, 1)
    selected = driver.make_sleep_request_schedule(
        pools["old"], pools["current"], event_index=0, task_id=1,
        method_kinds=["gkd"] * request_count)
    if len(selected) != request_count or any(kind != "gkd" for kind, _row, _role in selected):
        raise ValueError("frozen V10 GKD-only request selector returned an invalid stream")
    return selected


def evaluation_rows(dev_rows, order):
    old_ids = set(map(int, order[:11]))
    current_ids = set(map(int, order[11:22]))
    selected = [row for row in dev_rows if int(row["intent_id"]) in old_ids | current_ids]
    counts = {"old": sum(int(row["intent_id"]) in old_ids for row in selected),
              "current": sum(int(row["intent_id"]) in current_ids for row in selected)}
    if counts != {"old": 110, "current": 110} or len(selected) != 220:
        raise ValueError("V12 development cohort must be 110 old + 110 current examples")
    per_class = {cid: sum(int(row["intent_id"]) == cid for row in selected)
                 for cid in old_ids | current_ids}
    if len(per_class) != 22 or any(count != 10 for count in per_class.values()):
        raise ValueError("V12 development cohort must contain exactly ten rows per seen class")
    return selected, counts


def extract_valid_rollouts(result, prompt_lengths, max_new_tokens: int):
    """Trim common batched-generation padding to each row's first-EOS length."""
    if (type(max_new_tokens) is not int or max_new_tokens < 1
            or len(prompt_lengths) != int(result.token_ids.shape[0])
            or result.token_ids.ndim != 2):
        raise ValueError("generation result shape/cap mismatch")
    output_width = int(result.token_ids.shape[1])
    for prompt_length in prompt_lengths:
        if type(prompt_length) is not int or not 1 <= prompt_length < output_width:
            raise ValueError("prompt length is outside generated sequence")
    if max(prompt_lengths) + max_new_tokens < output_width:
        raise ValueError("generation output exceeds the configured token cap")
    sequences = []
    for row_i, prompt_length in enumerate(prompt_lengths):
        valid_length = int(result.valid_lengths[row_i].item())
        if valid_length <= prompt_length or valid_length > output_width:
            raise ValueError("teacher rollout answer length is invalid")
        sequences.append([int(x) for x in result.token_ids[row_i, :valid_length].tolist()])
    return sequences


def count_rollout_answer_tokens(sequences, prompt_lengths):
    if len(sequences) != len(prompt_lengths):
        raise ValueError("rollout sequence/prompt length counts differ")
    counts = [len(sequence) - int(prompt_length)
              for sequence, prompt_length in zip(sequences, prompt_lengths)]
    if any(count < 1 for count in counts):
        raise ValueError("rollout contains an empty answer")
    return sum(counts)


def freeze_teacher_context(context):
    """Freeze a tail MutableContext without treating its dataclass as a Module.

    The mutable context owns memory modules and standalone readout Parameters;
    only the memory children have training/evaluation modes. Frozen backbone
    and head modules remain owned by the shared model and are left untouched.
    """
    memory = getattr(context, "memory", None)
    parameters = getattr(context, "parameters", None)
    if memory is None or not callable(parameters):
        raise TypeError("teacher context must expose memory modules and parameters()")
    for module in memory:
        eval_method = getattr(module, "eval", None)
        if not callable(eval_method):
            raise TypeError("teacher memory entries must be torch modules")
        eval_method()
    for parameter in parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return context


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _atomic_torch(path: Path, value, torch):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def _source_hashes(task_root: Path, declared: dict) -> dict:
    actual = {}
    for rel in sorted(declared):
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("V12 source lock path must be relative and contained")
        source = (task_root / rel).resolve()
        if task_root.resolve() not in source.parents or not source.is_file():
            raise FileNotFoundError(f"locked V12 source is missing: {rel}")
        actual[rel] = sha256_file(source)
    if actual != declared:
        raise ValueError("V12 runner/helper source lock mismatch")
    return actual


def _preflight(args):
    """CPU-only lock check; no output directory creation or model/CUDA load."""
    task_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve(strict=True)
    v10_config_path = Path(args.v10_config).resolve(strict=True)
    project_root = Path(args.project_root).resolve(strict=True)
    model_path = Path(args.model_path).resolve(strict=True)
    pack_path = Path(args.training_pack).resolve(strict=True)
    foundation_root = Path(args.foundation_root).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    if args.seed not in SEEDS:
        raise ValueError("seed must be in the frozen 17/29/43 set")
    config_bytes_sha = sha256_file(config_path)
    config = validate_config(json.loads(config_path.read_text(encoding="utf-8")), require_frozen=True)
    if config.get("v10_config_file_sha256") != sha256_file(v10_config_path):
        raise ValueError("V12 referenced v10 config bytes changed")
    v10_cfg = json.loads(v10_config_path.read_text(encoding="utf-8"))
    task_tools = task_root / "tools"
    if str(task_tools) not in sys.path:
        sys.path.insert(0, str(task_tools))
    import run_banking_cil_v10 as driver
    driver.validate_config(v10_cfg, phase="main")
    source_hashes = _source_hashes(task_root, config["source_hashes"])
    source_hashes_v10, lock_sha, manifest, train_rows, dev_rows = driver._source_prepare(
        project_root, task_root, model_path, pack_path, v10_cfg)
    if v10_cfg["training_manifest_sha256"] != config["training_manifest_sha256"]:
        raise ValueError("V12 and v10 training manifest locks differ")
    if len(manifest.get("classes", [])) != 77 or len(train_rows) != 9229 or len(dev_rows) != 770:
        raise ValueError("locked Banking train/dev metadata shape mismatch")
    foundation_dir = (foundation_root / f"seed{args.seed}").resolve(strict=True)
    receipt_path = foundation_dir / "foundation-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("qualified foundation receipt is missing")
    receipt_sha = sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint = (foundation_dir / str(receipt.get("checkpoint", ""))).resolve()
    if foundation_dir not in checkpoint.parents:
        raise ValueError("foundation checkpoint escapes its seed directory")
    driver._validate_foundation_checkpoint_cpu(
        checkpoint, receipt.get("checkpoint_sha256"), seed=args.seed, cfg=v10_cfg,
        source_lock=lock_sha, manifest_sha=config["training_manifest_sha256"])
    if (receipt.get("schema"), receipt.get("status"), receipt.get("seed"),
            receipt.get("arm"), receipt.get("task_id"), receipt.get("config_sha256"),
            receipt.get("source_lock_sha256"), receipt.get("training_manifest_sha256"),
            receipt.get("model_immutable_versions_unchanged")) != (
            "banking-cil-v10-foundation-receipt", "qualified", args.seed,
            "FOUNDATION_CMS_ONLY", 0, driver._config_sha(v10_cfg), lock_sha,
            config["training_manifest_sha256"], True):
        raise ValueError("foundation receipt provenance/status mismatch")
    gate = receipt.get("gate")
    metric = receipt.get("metric")
    if (not isinstance(gate, dict) or gate.get("passed") is not True
            or not isinstance(metric, dict) or metric.get("count") != 110
            or metric.get("classes_evaluated") != 11):
        raise ValueError("foundation task-0 qualification gate is incomplete")
    if sha256_file(checkpoint) != receipt.get("checkpoint_sha256"):
        raise ValueError("foundation checkpoint digest mismatch")
    if output_dir.exists():
        raise FileExistsError("V12 scientific output directory must be fresh")
    driver._validate_output_path(output_dir, project_root, model_path)
    for protected in (pack_path.parent, foundation_root):
        if output_dir == protected or protected in output_dir.parents or output_dir in protected.parents:
            raise ValueError("V12 output overlaps training pack or foundation assets")
    if config.get("test_data_read") is not False:
        raise ValueError("V12 must not include official-test data")
    return {
        "status": "preflight_pass", "seed": args.seed,
        "config_sha256": config_bytes_sha, "v10_config_sha256": sha256_file(v10_config_path),
        "v12_source_hashes": source_hashes, "v10_source_lock_sha256": lock_sha,
        "v10_source_hashes": source_hashes_v10,
        "training_manifest_sha256": config["training_manifest_sha256"],
        "train_rows": len(train_rows), "development_rows": len(dev_rows),
        "class_count": len(manifest["classes"]), "foundation_receipt_sha256": receipt_sha,
        "foundation_checkpoint_sha256": receipt["checkpoint_sha256"],
        "foundation_step": int(receipt.get("steps", -1)),
        "model_loaded": False, "cuda_initialized": False,
        "official_test_read": False, "test_data_read": False,
        "optimizer_updates": 0,
    }


def _same_nested(a, b, torch):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same_nested(a[k], b[k], torch) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_same_nested(x, y, torch) for x, y in zip(a, b))
    if is_dataclass(a) and not isinstance(a, type):
        return (is_dataclass(b) and type(a) is type(b)
                and all(_same_nested(getattr(a, field.name), getattr(b, field.name), torch)
                        for field in fields(a)))
    try:
        import numpy as np
        if isinstance(a, np.ndarray):
            return isinstance(b, np.ndarray) and np.array_equal(a, b)
    except ImportError:
        pass
    return a == b


def _make_rollouts(model, teacher, selected, tokenizer, classes, common, cfg, torch,
                   device, tail_runtime, budget, seed, progress_path, max_sequence_length):
    """Generate once from the frozen pre-proposal teacher, preserving selector order."""
    grouped = defaultdict(list)
    for index, (_kind, row, _role) in enumerate(selected):
        prompt = common.tokenize_prompt(row["text"], tokenizer, classes, tokenizer.eos_token_id,
                                        int(max_sequence_length))
        ids = [int(x) for x in prompt["input_ids"][:prompt["prompt_length"]]]
        grouped[len(ids)].append((index, ids))
    sequences = [None] * len(selected)
    prompt_lengths = [0] * len(selected)
    generator = torch.Generator(device=device).manual_seed(
        int(cfg["rollout_seed_offset"]) + int(seed) * 100003 + int(cfg["task_id"]) * 1009)
    generated_tokens = 0
    forced_extra_eos_positions = 0
    actual_generated_tokens = 0
    completed = 0
    for width, group in sorted(grouped.items()):
        for start in range(0, len(group), int(cfg["rollout_microbatch_size"])):
            chunk = group[start:start + int(cfg["rollout_microbatch_size"])]
            prefix = torch.tensor([item[1] for item in chunk], dtype=torch.long, device=device)
            mask = torch.ones_like(prefix)
            with torch.no_grad():
                result = tail_runtime.sample_with_context(
                    model, teacher, prefix, int(cfg["max_new_tokens"]), generator=generator,
                    temperature=float(cfg["rollout_temperature"]), attention_mask=mask,
                    eos_token_id=int(tokenizer.eos_token_id), return_info=True)
            if not width < result.token_ids.shape[1] <= width + int(cfg["max_new_tokens"]):
                raise RuntimeError("teacher rollout width is outside the generation cap")
            actual_generated_tokens += int(result.actual_generated_tokens)
            forced_extra_eos_positions += int(result.executed_extra_tokens)
            trimmed = extract_valid_rollouts(
                result, [len(item[1]) for item in chunk], int(cfg["max_new_tokens"]))
            for row_i, (global_index, prompt_ids) in enumerate(chunk):
                sequence = trimmed[row_i]
                sequences[global_index] = sequence
                prompt_lengths[global_index] = len(prompt_ids)
                generated_tokens += len(sequence) - len(prompt_ids)
                completed += 1
            budget.generated_tokens += int(result.actual_generated_tokens)
            if completed % 128 == 0 or completed == len(selected):
                _atomic_json(progress_path, {
                    "status": "running", "phase": "teacher_rollouts", "seed": seed,
                    "rollouts_completed": completed, "rollouts_total": len(selected),
                    "generated_answer_tokens": generated_tokens,
                    "work": budget.as_dict(),
                })
    if any(sequence is None for sequence in sequences):
        raise RuntimeError("teacher rollout generation omitted request rows")
    rows = [{"input_ids": seq, "prompt_length": prompt_lengths[i],
             "answer_tokens": len(seq) - prompt_lengths[i]}
            for i, seq in enumerate(sequences)]
    if any(row["answer_tokens"] < 1 for row in rows):
        raise RuntimeError("teacher rollout contains an empty answer")
    return rows, {"requests": len(rows), "generated_answer_tokens": generated_tokens,
                  "actual_generated_tokens": actual_generated_tokens,
                  "forced_extra_eos_positions": forced_extra_eos_positions,
                  "microbatch_size": int(cfg["rollout_microbatch_size"]),
                  "prompt_length_groups": {str(width): len(group) for width, group in sorted(grouped.items())}}


def _train_request_batch(model, tx, rollouts, start, count, batcher, torch,
                         device, tail_runtime, budget):
    closures = []
    for row in rollouts[start:start + count]:
        def closure(row=row):
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            cache = tail_runtime.capture(model, ids, mask, device=device)
            with torch.no_grad():
                teacher_logits = tail_runtime.tail_logits(model, tx.teacher_context(), cache)
            student_logits = tail_runtime.tail_logits(model, tx.student_context(), cache)
            answer_mask = torch.zeros_like(ids, dtype=torch.bool)
            answer_mask[:, row["prompt_length"]:] = True
            loss, tokens = tail_runtime.causal_token_kl(
                student_logits, teacher_logits, answer_mask, temperature=1.0)
            return loss, int(tokens)
        closures.append(closure)
    before_backward = int(tx.counters.get("v12_backward_calls", 0))
    before_step_attempts = int(tx.counters.get("v12_adamw_step_attempts", 0))
    try:
        result = batcher(tx, closures, kind="gkd")
    except BaseException:
        backward_count = max(0, int(tx.counters.get("v12_backward_calls", 0)) - before_backward)
        for row in rollouts[start:start + backward_count]:
            answer_mask = torch.zeros((1, len(row["input_ids"])), dtype=torch.long, device=device)
            answer_mask[:, row["prompt_length"]:] = 1
            budget.backward(answer_mask)
        step_attempts = max(0, int(tx.counters.get("v12_adamw_step_attempts", 0)) - before_step_attempts)
        for _ in range(step_attempts):
            budget.update(1)
        raise
    for row in rollouts[start:start + count]:
        answer_mask = torch.zeros((1, len(row["input_ids"])), dtype=torch.long, device=device)
        answer_mask[:, row["prompt_length"]:] = 1
        budget.backward(answer_mask)
    budget.update(1)
    budget.adamw_updates += 1
    return result


def _save_tx_snapshot(path: Path, tx, *, torch, seed, arm, requests, step):
    snapshot = tx.snapshot()
    digest = _atomic_torch(path, {"schema": "banking-v12-receiver-snapshot-v1",
                                  "seed": seed, "arm": arm, "requests": requests,
                                  "adamw_steps": step, "snapshot": snapshot}, torch)
    return {"path": path.name, "sha256": digest, "requests": requests, "adamw_steps": step}


def _apply_proposal_copy(context, proposal):
    params = dict(context.named_parameters())
    with __import__("torch").no_grad():
        for name, value in proposal.items():
            if name not in params or params[name].shape != value.shape:
                raise ValueError("boundary proposal does not match receiver context parameters")
            params[name].copy_(value.to(params[name]))
    return context


def _run(args, preflight):
    """Execute one frozen V12 seed. GPU policy belongs to its parent launcher."""
    task_root = Path(__file__).resolve().parents[1]
    if preflight.get("status") != "preflight_pass" or preflight.get("seed") != args.seed:
        raise ValueError("fresh matching CPU preflight is required")
    config_path = Path(args.config).resolve(strict=True)
    config = validate_config(json.loads(config_path.read_text(encoding="utf-8")), require_frozen=True)
    if sha256_file(config_path) != preflight.get("config_sha256"):
        raise RuntimeError("V12 config bytes changed after preflight")
    v10_config_path = Path(args.v10_config).resolve(strict=True)
    v10cfg = json.loads(v10_config_path.read_text(encoding="utf-8"))
    project_root, model_path = Path(args.project_root).resolve(), Path(args.model_path).resolve()
    pack_path, foundation_root = Path(args.training_pack).resolve(), Path(args.foundation_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("V12 output directory already exists")
    if sha256_file(v10_config_path) != CONFIG_V10_SHA:
        raise RuntimeError("V10 config changed after CPU preflight")
    task_tools = task_root / "tools"
    if str(task_tools) not in sys.path:
        sys.path.insert(0, str(task_tools))
    import run_banking_cil_v10 as driver
    source_hashes, lock_sha, manifest, train_rows, dev_rows = driver._source_prepare(
        project_root, task_root, model_path, pack_path, v10cfg)
    v12_hashes_now = _source_hashes(task_root, config["source_hashes"])
    if v12_hashes_now != preflight.get("v12_source_hashes"):
        raise RuntimeError("V12 sources changed after CPU preflight")
    if lock_sha != preflight.get("v10_source_lock_sha256"):
        raise RuntimeError("V10 source lock changed after CPU preflight")
    foundation_dir = foundation_root / f"seed{args.seed}"
    receipt = json.loads((foundation_dir / "foundation-receipt.json").read_text(encoding="utf-8"))
    if sha256_file(foundation_dir / "foundation-receipt.json") != preflight.get("foundation_receipt_sha256"):
        raise RuntimeError("foundation receipt bytes changed after CPU preflight")
    foundation_cp = (foundation_dir / receipt["checkpoint"]).resolve()
    if sha256_file(foundation_cp) != preflight.get("foundation_checkpoint_sha256"):
        raise RuntimeError("foundation changed after CPU preflight")

    import torch
    from banking_cil_v10_runtime import score_candidates
    from banking_cil_v10_lifecycle import AdditiveWakeController
    from banking_cil_v12_batching import BatchableAdditiveSleepTransaction, optimize_request_batch
    import banking_cil_common_v10 as common
    import clinc_tail_runtime_v8 as tail_runtime

    driver._init_rng(int(v10cfg["initialization_seed"]), torch)
    torch, tokenizer, model, immutable_versions = driver._load_model(project_root, model_path, v10cfg)
    driver._verify_loaded_project_modules(project_root)
    model_before = immutable_versions(model)
    for _name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
    budget_cfg = {"max_wall_seconds": int(config["max_wall_seconds"]),
                  "max_forward_calls": int(config["max_forward_calls"]),
                  "max_update_steps": int(config["max_update_steps"])}
    budget = driver.Budget(budget_cfg)
    original_logits = model.logits
    def counted(tokens, mask=None):
        budget.forward(mask if mask is not None else torch.ones_like(tokens))
        return original_logits(tokens, mask)
    model.logits = counted
    tail_runtime, original_tail = driver._install_tail_counter(budget)
    output_dir.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    budget.started = started
    active_arm = None
    current_tx = None
    work_receipts = {}
    qualification_receipt = None
    try:
        driver._atomic_json(output_dir / "protocol.json", {
            "schema": "banking-v12-boundary-protocol-v1", "status": "running",
            "config_sha256": preflight["config_sha256"], "seed": args.seed,
            "foundation_receipt_sha256": preflight["foundation_receipt_sha256"],
            "foundation_checkpoint_sha256": preflight["foundation_checkpoint_sha256"],
            "v10_source_lock_sha256": lock_sha,
            "training_manifest_sha256": config["training_manifest_sha256"],
            "v12_source_hashes": preflight["v12_source_hashes"],
            "test_data_read": False, "claim_scope": config["claim_scope"],
        })
        context = tail_runtime.clone_mutable_context(model)
        wake = AdditiveWakeController(context, periods=v10cfg["periods"],
                                      base_lr=v10cfg["base_lr"], fast_lr=v10cfg["fast_lr"])
        foundation_state = driver._restore_checkpoint(
            foundation_cp, context, wake, expected_sha=receipt["checkpoint_sha256"],
            seed=args.seed, arm="FOUNDATION_CMS_ONLY", cfg=v10cfg, torch=torch)
        if foundation_state.get("step") != wake.ticks:
            raise ValueError("restored foundation wake cursor differs from checkpoint step")

        order = tuple(int(x) for x in manifest["class_orders"][str(args.seed)])
        current_rows = driver.task_rows(train_rows, order, 1)
        by_id = {str(row["id"]): row for row in current_rows}
        epoch_ids = driver.deterministic_epoch_ids(current_rows, seed=args.seed,
                                                    task_id=1, epoch=0)
        cursor = 0
        tick_receipts = []
        def one_prefix_tick():
            nonlocal cursor
            if cursor >= len(epoch_ids):
                raise RuntimeError("task-1 frozen epoch ended before first level-0 due boundary")
            tick_ids = epoch_ids[cursor:cursor + int(v10cfg["batch_size"])]
            cursor += len(tick_ids)
            rows = [by_id[row_id] for row_id in tick_ids]
            metric = driver._train_tick(model, wake.context, wake, rows, tokenizer,
                                        manifest["classes"], v10cfg, torch,
                                        next(model.parameters()).device, budget)
            tick_receipts.append({"task_tick": len(tick_receipts) + 1,
                                  "wake_tick": wake.ticks,
                                  "answer_tokens": metric["answer_tokens"]})
        shared_before = budget.as_dict()
        prefix_ticks, proposals = select_first_due_level0(
            wake, int(config["max_prefix_wake_ticks"]), one_prefix_tick)
        if 0 not in proposals:
            raise RuntimeError("level 0 did not become due within the frozen wake prefix")
        if set(proposals) != {0}:
            raise RuntimeError("first level-0 boundary coincides with another due level; refusing an ambiguous probe")
        proposal = proposals[0]
        preproposal_context = deepcopy(wake.context)
        cms_noreset = _apply_proposal_copy(deepcopy(wake.context), proposal)
        shared_boundary_work = _budget_delta(shared_before, budget.as_dict())
        selected = fixed_teacher_request_rows(driver, train_rows, order,
                                             int(config["fixed_teacher_rollout_requests"]))
        teacher_view = deepcopy(preproposal_context)
        freeze_teacher_context(teacher_view)
        rollout_before = budget.as_dict()
        rollouts, rollout_stats = _make_rollouts(model, teacher_view, selected, tokenizer,
            manifest["classes"], common, config, torch, next(model.parameters()).device,
            tail_runtime, budget, args.seed, output_dir / "progress.json",
            max_sequence_length=int(v10cfg["max_sequence_length"]))
        rollout_work = _budget_delta(rollout_before, budget.as_dict())
        targets_sha = _atomic_torch(output_dir / "teacher-rollouts-private.pt", {
            "schema": "banking-v12-fixed-teacher-rollouts-private-v1",
            "seed": args.seed, "rollouts": rollouts,
        }, torch)

        tx_seed = int(config["receiver_initialization_seed_offset"]) + args.seed * 100003 + int(config["task_id"]) * 1009
        txes = {}
        state_hashes = {}
        checkpoint_contexts = {"TEACHER": teacher_view, "CMS_NO_RESET": deepcopy(cms_noreset)}
        from clinc_lifecycle_v8 import _state_hash
        state_hashes["TEACHER"] = _state_hash(teacher_view)
        state_hashes["CMS_NO_RESET"] = _state_hash(cms_noreset)
        for arm in config["arms"]:
            live_context = deepcopy(preproposal_context)
            tx = BatchableAdditiveSleepTransaction(
                live_context, int(config["sender"]), proposal,
                lr=float(arm["learning_rate"]), weight_decay=float(config["weight_decay"]),
                clip_norm=float(config["clip_norm"]), seed=tx_seed)
            if arm["name"] == "B1_HIGH":
                checkpoint_contexts["RESET_ZERO"] = deepcopy(tx.student_context())
                state_hashes["RESET_ZERO"] = _state_hash(tx.student_context())
            state_hashes[arm["name"]] = _state_hash(tx.student_context())
            txes[arm["name"]] = tx
        if len(set(state_hashes[name] for name in
                   ("B1_HIGH", "B1_LOW", "B32_HIGH", "B32_LOW"))) != 1:
            raise RuntimeError("V12 receiver initialization differs across optimizer arms")

        batch_log = output_dir / "optimizer-batches.jsonl"
        for arm_name in config["execution_order"]:
            arm = next(item for item in config["arms"] if item["name"] == arm_name)
            name = arm["name"]
            active_arm = name
            tx = txes[name]
            current_tx = tx
            arm_work_before = budget.as_dict()
            batch_size, max_requests = arm["effective_batch_size"], arm["max_requests"]
            checkpoints = set(arm["checkpoint_requests"])
            request_cursor = 0
            while request_cursor < max_requests:
                amount = min(batch_size, max_requests - request_cursor)
                start_request = request_cursor
                metrics = _train_request_batch(model, tx, rollouts, request_cursor, amount,
                                               optimize_request_batch, torch,
                                               next(model.parameters()).device,
                                               tail_runtime, budget)
                request_cursor += amount
                with batch_log.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps({
                        "seed": args.seed, "arm": name, "request_start": start_request,
                        "request_count": amount, "requests_completed": request_cursor,
                        "adamw_step": int(tx.counters["adamw_steps"]),
                        "loss_mean": metrics["loss_mean"], "token_count": metrics["token_count"],
                        "preclip_grad_norm": metrics["preclip_grad_norm"],
                        "parameter_update_l2": metrics["parameter_update_l2"],
                        "physical_adamw_steps": metrics["physical_adamw_steps"],
                    }, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                if (args.seed == 17 and name == "B32_LOW" and
                        request_cursor == int(config["restore_qualification"]["snapshot_after_requests"])):
                    qual_dir = output_dir / "qualification"
                    qual_dir.mkdir(exist_ok=False)
                    snap_meta = _save_tx_snapshot(qual_dir / "b32-low-after32.pt", tx,
                        torch=torch, seed=args.seed, arm=name, requests=request_cursor,
                        step=int(tx.counters["adamw_steps"]))
                    snapshot = tx.snapshot()
                    saved = torch.load(qual_dir / "b32-low-after32.pt", map_location="cpu",
                                       weights_only=False)["snapshot"]
                    if sha256_file(qual_dir / "b32-low-after32.pt") != snap_meta["sha256"]:
                        raise RuntimeError("V12 serialized snapshot hash changed during qualification readback")
                    if not _same_nested(snapshot, saved, torch):
                        raise RuntimeError("V12 serialized snapshot differs before restore branch")
                    original_result = _train_request_batch(model, tx, rollouts, 32, 32,
                        optimize_request_batch, torch, next(model.parameters()).device,
                        tail_runtime, budget)
                    restored = BatchableAdditiveSleepTransaction.from_snapshot(
                        deepcopy(tx.live), saved)
                    qualification_before = budget.as_dict()
                    restored_result = _train_request_batch(model, restored, rollouts, 32, 32,
                        optimize_request_batch, torch, next(model.parameters()).device,
                        tail_runtime, budget)
                    qualification_extra_work = _budget_delta(qualification_before, budget.as_dict())
                    restored_snapshot = restored.snapshot()
                    original_snapshot = tx.snapshot()
                    if not _same_nested(restored_snapshot, original_snapshot, torch):
                        raise RuntimeError("V12 optimizer/RNG/resume parity failed after requests 32:64")
                    restored_state_sha = stable_state_sha(restored_snapshot, torch)
                    original_state_sha = stable_state_sha(original_snapshot, torch)
                    if restored_state_sha != original_state_sha:
                        raise RuntimeError("V12 stable snapshot-content hashes differ after resume branch")
                    qualification_receipt = {
                        "status": "pass", "snapshot_sha256": snap_meta["sha256"],
                        "snapshot_content_sha256": stable_state_sha(saved, torch),
                        "post_range_content_sha256": restored_state_sha,
                        "original_post_range_content_sha256": original_state_sha,
                        "original_request_range": [32, 64], "restored_request_range": [32, 64],
                        "extra_request_evaluations": 32, "extra_physical_adamw_steps": 1,
                        "original_metrics": original_result, "restored_metrics": restored_result,
                        "extra_work_delta": qualification_extra_work,
                        "post_range_snapshot_state_equal": True,
                    }
                    _atomic_json(qual_dir / "restore-qualification.json", qualification_receipt)
                    # The original branch is already at request 64. Count its requests only once
                    # in the main arm; only the cloned restore branch is extra work.
                    request_cursor = 64
                if request_cursor in checkpoints:
                    eval_name = f"{name}_{request_cursor}"
                    checkpoint_contexts[eval_name] = deepcopy(tx.student_context())
                    snap_dir = output_dir / "receiver-checkpoints"
                    snap_meta = _save_tx_snapshot(
                        snap_dir / f"{eval_name}.pt", tx, torch=torch, seed=args.seed,
                        arm=name, requests=request_cursor, step=int(tx.counters["adamw_steps"]))
                    _atomic_json(snap_dir / f"{eval_name}.json", snap_meta)
                if request_cursor % 128 == 0 or request_cursor == max_requests:
                    _atomic_json(output_dir / "progress.json", {
                        "status": "running", "seed": args.seed, "arm": name,
                        "requests": request_cursor, "adamw_steps": int(tx.counters["adamw_steps"]),
                        "work": budget.as_dict(), "wall_seconds": time.monotonic() - started,
                    })
            if int(tx.counters["adamw_steps"]) != max_requests // batch_size:
                raise RuntimeError("V12 physical AdamW step count differs from request budget")
            work_receipts[name] = {
                "requests": int(tx.counters.get("v12_committed_requests", 0)),
                "tokens": int(tx.counters.get("v12_committed_tokens", 0)),
                "backward_calls": int(tx.counters.get("v12_backward_calls", 0)),
                "adamw_steps": int(tx.counters["adamw_steps"]),
                "batch_commits": int(tx.counters.get("v12_batch_commits", 0)),
                "learning_rate": float(arm["learning_rate"]),
                "effective_batch_size": batch_size,
                "work_delta_including_restore_validation": _budget_delta(arm_work_before, budget.as_dict()),
                "initial_student_state_sha256": state_hashes[name],
            }
            tx.commit()
        if args.seed == 17 and qualification_receipt is None:
            raise RuntimeError("required seed-17 restore qualification was not completed")

        dev_eval, dev_counts = evaluation_rows(dev_rows, order)
        if len(checkpoint_contexts) != 10:
            raise RuntimeError(f"expected ten shared-evaluation contexts, got {len(checkpoint_contexts)}")
        evaluation_before = budget.as_dict()
        runtime_result = score_candidates(
            model, checkpoint_contexts, dev_eval, tokenizer, manifest["classes"],
            common, tail_runtime, torch=torch, device=next(model.parameters()).device,
            max_sequence_length=int(v10cfg["max_sequence_length"]),
            batch_size=int(config["evaluation"]["candidate_batch_size"]), budget=budget,
            class_order=order, task_id=1, include_greedy=True,
            max_new_tokens=int(v10cfg["max_new_tokens"]))
        evaluation_work = _budget_delta(evaluation_before, budget.as_dict())
        if immutable_versions(model) != model_before:
            raise RuntimeError("frozen base model changed during V12 run")
        costs = budget.as_dict()
        result = {
            "schema": "banking-v12-boundary-result-v1", "status": "complete",
            "seed": args.seed,
            "boundary": {"prefix_wake_ticks": len(tick_receipts),
                "restored_foundation_wake_tick": int(foundation_state["step"]),
                "wake_tick_at_due": int(wake.ticks), "due_levels": [0],
                "proposal_sha256": stable_state_sha(proposal, torch),
                "prefix_tick_receipts": tick_receipts},
            "fixed_teacher_rollouts": {**rollout_stats, "sha256": targets_sha,
                "request_order_digest_sha256": canonical_sha([
                    hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest()
                    for _kind, row, _role in selected]),
                "class_role_counts": {role: sum(item[2] == role for item in selected)
                                      for role in ("old", "current")},
                "gold_labels_used_as_loss_targets": False},
            "shared_boundary_work": shared_boundary_work,
            "fixed_rollout_work": rollout_work,
            "optimizer_arms": work_receipts,
            "restore_qualification": qualification_receipt,
            "development": {"population": "first_task0_plus_task1_development_only",
                "counts": dev_counts, "global_candidate_labels": 77,
                "runtime": runtime_result, "work_delta": evaluation_work},
            "budget": costs, "wall_seconds": time.monotonic() - started,
            "official_test_read": False, "test_data_read": False,
            "claim_scope": config["claim_scope"],
            "initial_state_hashes": state_hashes,
        }
        _atomic_json(output_dir / "development-evaluation.json", result["development"])
        _atomic_json(output_dir / "work.json", {"optimizer_arms": work_receipts,
                                                  "budget": costs,
                                                  "wall_seconds": result["wall_seconds"]})
        _atomic_json(output_dir / "final-receipt.json", result)
        _atomic_json(output_dir / "protocol.json", {
            "schema": "banking-v12-boundary-protocol-v1", "status": "complete",
            "config_sha256": preflight["config_sha256"], "seed": args.seed,
            "foundation_receipt_sha256": preflight["foundation_receipt_sha256"],
            "foundation_checkpoint_sha256": preflight["foundation_checkpoint_sha256"],
            "v10_source_lock_sha256": lock_sha,
            "training_manifest_sha256": config["training_manifest_sha256"],
            "v12_source_hashes": preflight["v12_source_hashes"],
            "final_receipt_sha256": sha256_file(output_dir / "final-receipt.json"),
            "test_data_read": False, "claim_scope": config["claim_scope"],
        })
        _atomic_json(output_dir / "progress.json", {
            "status": "complete", "seed": args.seed, "work": costs,
            "wall_seconds": result["wall_seconds"],
        })
        return result
    except BaseException as exc:
        _atomic_json(output_dir / "failure.json", {
            "schema": "banking-v12-boundary-failure-v1", "status": "failed",
            "seed": args.seed, "error_type": type(exc).__name__,
            "error_message": str(exc)[:500], "work": budget.as_dict(),
            "active_arm": active_arm,
            "active_transaction_counters": dict(current_tx.counters) if current_tx is not None else None,
            "wall_seconds": time.monotonic() - started, "test_data_read": False,
        })
        raise
    finally:
        driver._remove_tail_counter(tail_runtime, original_tail)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("project-root", "model-path", "training-pack", "v10-config", "config",
                "foundation-root", "output-dir"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args(argv)
    preflight = _preflight(args)
    result = _run(args, preflight)
    print(json.dumps({"status": result["status"], "seed": args.seed,
                      "output_dir": str(Path(args.output_dir).resolve()),
                      "official_test_read": False}))


if __name__ == "__main__":
    main()
