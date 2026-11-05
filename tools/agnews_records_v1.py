"""Strict CPU-only validation for frozen AG News train-role records.

The official test split is deliberately outside this schema and is never
accepted by the training data loader.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "nl-sleep-agnews-train-roles-v1"
DEV_SEED = "1907"
MAIN_SEEDS = ("17", "29", "43")
EXPECTED_SEEDS = (DEV_SEED,) + MAIN_SEEDS
ROLE_PHASE = {"old_train": "old", "old_calib": "old", "new_train": "new", "new_calib": "new"}
TRAIN_ROLES = {"old_train", "new_train"}
CALIB_ROLES = {"old_calib", "new_calib"}
SLEEP_CLASS_ORDER = [0, 2, 3, 1, 1, 3, 2, 0] * 2 + [0, 2, 1, 3] * 2
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require_hash(value, label):
    if not isinstance(value, str) or not HEX64.fullmatch(value.lower()):
        raise ValueError(f"{label} must be a 64-character SHA256 hex digest")


def _indices(value, expected_steps, label):
    if not isinstance(value, list) or len(value) != expected_steps:
        raise ValueError(f"{label} must contain exactly {expected_steps} fixed steps")
    for step, batch in enumerate(value):
        if not isinstance(batch, list) or len(batch) != 4 or any(type(i) is not int for i in batch):
            raise ValueError(f"{label}[{step}] must contain four global integer record indices")


def validate_agnews_records(doc: dict) -> dict:
    """Validate provenance, split roles, token boundaries and frozen schedules.

    Returns aggregate metadata only; no prompt token arrays are copied into the
    return value. This function never opens any path other than its input JSON.
    """
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA}")
    # Explicitly reject any accidental official-test payload/path reference.
    forbidden_test_keys = {"test", "test_file", "test_path", "test_split", "test_data",
                           "test_records", "official_test_file", "official_test_path"}
    def contains_test_payload_key(value):
        if isinstance(value, dict):
            return any(str(k).lower() in forbidden_test_keys or contains_test_payload_key(v) for k,v in value.items())
        if isinstance(value, list):
            return any(contains_test_payload_key(x) for x in value)
        return False
    if contains_test_payload_key(doc):
        raise ValueError("official test data/path must not be present in train-role JSON")
    source = doc.get("source")
    if not isinstance(source, dict) or not source.get("provider") or not source.get("revision"):
        raise ValueError("source must include provider and fixed revision")
    train_sha = source.get("source_file_sha256", source.get("train_file_sha256", source.get("trainFileSHA")))
    _require_hash(train_sha, "source.train_file_sha256")
    tokenizer = doc.get("tokenizer")
    inventory = tokenizer.get("file_inventory") if isinstance(tokenizer, dict) else None
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("tokenizer.file_inventory must pin tokenizer files")
    tokenizer_hashes = {}
    for row in inventory:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("tokenizer inventory entries need a filename and SHA256")
        _require_hash(row.get("sha256"), f"tokenizer.file_inventory[{row['name']}].sha256")
        tokenizer_hashes[row["name"]] = row["sha256"].lower()

    names = doc.get("class_names")
    phases = doc.get("phase_class_ids")
    if not isinstance(names, list) or len(names) != 4 or any(not isinstance(x, str) or not x for x in names):
        raise ValueError("class_names must describe the four source classes")
    if not isinstance(phases, dict) or set(phases) != {"old", "new"}:
        raise ValueError("phase_class_ids must explicitly declare old and new class IDs")
    old_ids, new_ids = phases["old"], phases["new"]
    if (not isinstance(old_ids, list) or not isinstance(new_ids, list) or
        len(old_ids) != 2 or len(new_ids) != 2 or
        set(old_ids) & set(new_ids) or set(old_ids + new_ids) != {0, 1, 2, 3}):
        raise ValueError("old/new class groups must be a disjoint partition of IDs 0..3")

    prompt = doc.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("prompt_spec is required to verify record token boundaries")
    prefix = prompt.get("prompt_prefix_token_ids")
    suffix = prompt.get("answer_cue_token_ids")
    if (not isinstance(prefix, list) or not prefix or any(type(x) is not int for x in prefix) or
        not isinstance(suffix, list) or not suffix or any(type(x) is not int for x in suffix)):
        raise ValueError("prompt prefix and answer cue token IDs must be nonempty integer lists")
    if prompt.get("article_token_cap", prompt.get("article_token_limit")) != 96 or prompt.get("max_prompt_tokens") != 128:
        raise ValueError("frozen prompt contract requires up to 96 article tokens and max 128 prompt tokens")
    cue_text = prompt.get("answer_cue", prompt.get("answer_cue_text"))
    if cue_text != "\nAnswer:":
        raise ValueError("all roles must share the frozen newline Answer cue")

    records = doc.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty flat array")
    ids = set()
    record_keys = []
    role_counts = Counter()
    prompt_fingerprints = {}
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"records[{index}] must be an object")
        forbidden = {"text", "article", "raw_text", "content", "title"} & set(row)
        if forbidden:
            raise ValueError(f"records[{index}] contains raw source text fields: {sorted(forbidden)}")
        record_id = row.get("record_id_sha256")
        _require_hash(record_id, f"records[{index}].record_id_sha256")
        if record_id in ids:
            raise ValueError("duplicate normalized article identity across roles")
        ids.add(record_id)
        partition, role, class_id = row.get("partition"), row.get("role"), row.get("class_id")
        if partition not in {"dev", "main"} or role not in ROLE_PHASE:
            raise ValueError(f"records[{index}] has an unknown partition/role")
        if type(class_id) is not int or class_id not in range(4):
            raise ValueError(f"records[{index}].class_id must be an integer 0..3")
        expected_phase = ROLE_PHASE[role]
        if class_id not in phases[expected_phase]:
            raise ValueError(f"records[{index}] class does not match its old/new role")
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError(f"records[{index}].tokens must be a nonempty integer token list")
        article_n = len(tokens) - len(prefix) - len(suffix)
        if (tokens[:len(prefix)] != prefix or tokens[-len(suffix):] != suffix or
            not 1 <= article_n <= 96 or len(tokens) > 128):
            raise ValueError(f"records[{index}] violates prefix + <=96 article + Answer cue token contract")
        prompt_fingerprint = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        if row.get("prompt_sha256") is not None:
            _require_hash(row["prompt_sha256"], f"records[{index}].prompt_sha256")
            if row["prompt_sha256"].lower() != prompt_fingerprint:
                raise ValueError(f"records[{index}] prompt hash does not match prompt tokens")
        if row.get("article_token_count") is not None and row["article_token_count"] != article_n:
            raise ValueError(f"records[{index}] article_token_count disagrees with prompt composition")
        if row.get("article_was_truncated") is not None and type(row["article_was_truncated"]) is not bool:
            raise ValueError(f"records[{index}].article_was_truncated must be a Boolean")
        phase = row.get("phase")
        if phase is not None and phase != expected_phase:
            raise ValueError(f"records[{index}].phase disagrees with role/class")
        if prompt_fingerprint in prompt_fingerprints:
            raise ValueError("duplicate tokenized prompt across roles; split leakage guard refused")
        prompt_fingerprints[prompt_fingerprint] = index
        record_keys.append((partition, role, class_id))
        role_counts[(partition, role, class_id)] += 1

    # Frozen stratified sizes: counts are per class, not total per role.
    for class_id in range(4):
        phase = "old" if class_id in phases["old"] else "new"
        train_role, calib_role = f"{phase}_train", f"{phase}_calib"
        expected = {
            ("dev", train_role, class_id): 128,
            ("dev", calib_role, class_id): 64,
            ("main", train_role, class_id): 256,
            ("main", calib_role, class_id): 64,
        }
        for key, count in expected.items():
            if role_counts[key] != count:
                raise ValueError(f"stratified role count mismatch for {key}: {role_counts[key]} != {count}")

    code_maps = doc.get("code_maps")
    schedules = doc.get("schedules")
    pools = doc.get("sleep_prefix_pools")
    if not all(isinstance(x, dict) for x in (code_maps, schedules, pools)):
        raise ValueError("code_maps, schedules and sleep_prefix_pools are required")
    if any(set(x) != set(EXPECTED_SEEDS) for x in (code_maps, schedules, pools)):
        raise ValueError("seed maps must cover exactly DEV 1907 and main 17/29/43")
    per_seed_schedule_hashes = {}
    for seed in EXPECTED_SEEDS:
        code_map = code_maps[seed]
        if not isinstance(code_map, dict) or type(code_map.get("marker_token_id")) is not int:
            raise ValueError(f"code_maps[{seed}] must provide a marker token")
        classes = code_map.get("class_to_token_ids")
        if (not isinstance(classes, list) or len(classes) != 4 or
            any(type(x) is not int for x in classes) or len(set(classes)) != 4 or
            code_map["marker_token_id"] in classes):
            raise ValueError(f"code_maps[{seed}] must map each class to a distinct second token")
        partition = "dev" if seed == DEV_SEED else "main"
        schedule = schedules[seed]
        if not isinstance(schedule, dict):
            raise ValueError(f"schedules[{seed}] must be an object")
        _indices(schedule.get("foundation512"), 512, f"schedules[{seed}].foundation512")
        _indices(schedule.get("new_primary256"), 256, f"schedules[{seed}].new_primary256")
        for stage, role, class_group in (("foundation512", "old_train", old_ids),
                                          ("new_primary256", "new_train", new_ids)):
            stored_hash = schedule.get(stage + "_sha256")
            if stored_hash is not None:
                _require_hash(stored_hash, f"schedules[{seed}].{stage}_sha256")
                if stored_hash.lower() != canonical_sha256(schedule[stage]):
                    raise ValueError(f"schedules[{seed}].{stage} hash mismatch")
            for step_index, batch in enumerate(schedule[stage]):
                classes_in_batch = Counter()
                for global_index in batch:
                    if not 0 <= global_index < len(records):
                        raise ValueError(f"schedule index outside global records array: {global_index}")
                    rec = records[global_index]
                    if rec["partition"] != partition or rec["role"] != role or rec["class_id"] not in class_group:
                        raise ValueError(f"{seed}/{stage}/{step_index} points outside its train-role partition")
                    classes_in_batch[rec["class_id"]] += 1
                if sorted(classes_in_batch.values()) != [2, 2]:
                    raise ValueError(f"{seed}/{stage}/{step_index} must be balanced 2+2 across phase classes")
        event_pools = pools[seed]
        if not isinstance(event_pools, list) or len(event_pools) != 4:
            raise ValueError(f"sleep_prefix_pools[{seed}] must have four fixed events")
        for event_no, pool in enumerate(event_pools):
            if not isinstance(pool, list) or len(pool) != 24 or any(type(i) is not int for i in pool):
                raise ValueError(f"sleep_prefix_pools[{seed}][{event_no}] must contain 24 record indices")
            actual_classes = []
            for i in pool:
                if not 0 <= i < len(records):
                    raise ValueError("Sleep prefix index outside global records array")
                rec = records[i]
                if rec["partition"] != partition or rec["role"] not in TRAIN_ROLES:
                    raise ValueError("Sleep prefix pool must contain only same-partition train prompts")
                actual_classes.append(rec["class_id"])
            if actual_classes != SLEEP_CLASS_ORDER:
                raise ValueError(f"Sleep prefix classes differ from frozen event schedule at {seed}/{event_no}")
            for start in range(0, 24, 4):
                group = actual_classes[start:start+4]
                if sum(c in old_ids for c in group) != 2 or sum(c in new_ids for c in group) != 2:
                    raise ValueError("each prefix/replay batch must have exactly old/new 2+2 prompts")
        per_seed_schedule_hashes[seed] = canonical_sha256({"schedule": schedule, "sleep_prefix_pools": event_pools})

    return {"schema": SCHEMA, "records": len(records),
            "role_counts": {f"{p}/{r}/class{c}": n for (p,r,c),n in sorted(role_counts.items())},
            "old_class_ids": sorted(old_ids), "new_class_ids": sorted(new_ids),
            "train_file_sha256": train_sha.lower(),
            "tokenizer_files_sha256": tokenizer_hashes,
            "schedules_sha256_by_seed": per_seed_schedule_hashes,
            "test_role_or_file_present": False,
            "raw_source_text_or_token_ids_returned": False}


def load_agnews_records(path: str | Path) -> tuple[dict, dict]:
    """Load only the frozen train-derived role file; never accepts a test path."""
    source = Path(path)
    if not source.is_file() or source.is_symlink() or getattr(source, "is_junction", lambda: False)():
        raise ValueError("records JSON must be an existing regular local file")
    doc = json.loads(source.read_text(encoding="utf-8"))
    return doc, validate_agnews_records(doc)


def scheduled_indices(doc: dict, seed: int, stage: str, step: int) -> list[int]:
    """Return a frozen four-record batch without consuming any RNG state."""
    key = str(seed)
    if stage not in {"foundation512", "new_primary256"}:
        raise ValueError("unknown primary training schedule")
    batches = doc["schedules"][key][stage]
    if type(step) is not int or not 0 <= step < len(batches):
        raise IndexError("primary schedule step is outside the frozen range")
    return list(batches[step])


def extra_replay_indices(doc: dict, seed: int, event: int, batch_number: int) -> list[int]:
    """Cycle the fixed 24-prompt event pool; never advances primary schedules."""
    pools = doc["sleep_prefix_pools"][str(seed)]
    if type(event) is not int or not 0 <= event < len(pools) or type(batch_number) is not int or batch_number < 0:
        raise IndexError("extra-replay event/batch number is outside its fixed pool")
    pool = pools[event]
    start = (batch_number * 4) % len(pool)
    return [pool[(start+i) % len(pool)] for i in range(4)]


def sleep_prefix_records(doc: dict, seed: int, event: int) -> list[dict]:
    """Return the 24 label-free train prompts in the fixed Sleep event pool."""
    return [doc["records"][i] for i in doc["sleep_prefix_pools"][str(seed)][event]]


def collate_record_indices(doc: dict, seed: int, indices: list[int], device, pad_id: int):
    """Build prompt+marker teacher-forced inputs and full-vocabulary targets.

    Returns tensors `(input_ids, attention_mask, prompt_lengths, marker_targets,
    class_targets)`; the class target occupies the next logit after the shared
    marker input. No label token is inserted before the target position.
    """
    import torch
    if not indices or len(indices) != 4:
        raise ValueError("training batches require exactly four global record indices")
    code = doc["code_maps"][str(seed)]
    rows = [doc["records"][i] for i in indices]
    lengths = torch.tensor([len(row["tokens"]) for row in rows], dtype=torch.long, device=device)
    max_prompt = int(lengths.max())
    ids = torch.full((len(rows), max_prompt+1), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    marker_id = int(code["marker_token_id"])
    markers = torch.full((len(rows),), marker_id, dtype=torch.long, device=device)
    classes = torch.tensor([int(code["class_to_token_ids"][row["class_id"]]) for row in rows],
                           dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        n = len(row["tokens"])
        ids[i, :n] = torch.tensor(row["tokens"], dtype=torch.long, device=device)
        ids[i, n] = marker_id
        mask[i, :n+1] = 1
    return ids, mask, lengths, markers, classes


def collate_scheduled_batch(doc: dict, seed: int, stage: str, step: int, device, pad_id: int):
    return collate_record_indices(doc, seed, scheduled_indices(doc, seed, stage, step), device, pad_id)


def collate_extra_replay_batch(doc: dict, seed: int, event: int, batch_number: int, device, pad_id: int):
    return collate_record_indices(doc, seed, extra_replay_indices(doc, seed, event, batch_number), device, pad_id)


def collate_evaluation_batch(doc: dict, seed: int, partition: str, role: str, batch_number: int,
                             device, pad_id: int, *, batch_size=4):
    """Collate a deterministic evaluation slice from a single frozen role."""
    import torch
    if partition not in {"dev", "main"} or role not in ROLE_PHASE or role not in TRAIN_ROLES | CALIB_ROLES:
        raise ValueError("evaluation role/partition is invalid")
    row_indices = [i for i,r in enumerate(doc["records"])
                   if r["partition"] == partition and r["role"] == role]
    if type(batch_number) is not int or batch_number < 0:
        raise IndexError("evaluation batch number must be nonnegative")
    start = batch_number * batch_size
    if start >= len(row_indices):
        raise IndexError("evaluation batch outside frozen role")
    indices = row_indices[start:start+batch_size]
    if len(indices) != batch_size:
        raise ValueError("evaluation role size must be divisible by the requested batch size")
    # Shared class/marker target construction, but preserve batch_size=4 contract.
    return collate_record_indices(doc, seed, indices, device, pad_id)
