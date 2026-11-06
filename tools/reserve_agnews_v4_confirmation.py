"""Reserve an unscored within-AGNews-train confirmation set from public candidate pool.

Tokenizer-only, CPU-only, no model loading/scoring. Sensitive text and labels are
never printed. Outputs are sealed locally under a fresh, non-overwriting folder.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
from collections import defaultdict
from datetime import datetime, timezone

ROOT = pathlib.Path(r"D:\my_paper\NL_Sleep_Public_Candidate")
ISOLATION = ROOT / "outputs" / "agnews-v3-isolation" / "fresh-roles-v3"
CANDIDATES = ISOLATION / "candidates-private.json"
TRAIN_VALIDATION = ISOLATION / "train-validation-private.json"
V3_PROMPTS = ISOLATION / "holdout-prompts-sealed.json"
TOKENIZER_DIR = pathlib.Path(r"D:\my_paper\Nested_Learning_and_Sleep\.cache\models\llama32-3b")
OUT = ROOT / "outputs" / "agnews-v4-confirmation" / "reserved-unscored"
SALT = "nl-sleep-agnews-v4-confirmation-20261106"
SHUFFLE_SEED = 404066
PER_CLASS = 256
CLASSES = range(4)

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha_file(p: pathlib.Path) -> str:
    return sha_bytes(p.read_bytes())

def hash_obj(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha_bytes(raw)

def read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))

def write_new(p: pathlib.Path, obj):
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        f.write("\n")

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    # Verify that the declared tokenizer asset inventory still matches v3.
    candidates = read_json(CANDIDATES)
    trainval = read_json(TRAIN_VALIDATION)
    v3_prompts = read_json(V3_PROMPTS)
    assert len(candidates["records"]) == 16384
    assert len(trainval["records"]) == 1536
    assert len(v3_prompts["records"]) == 1024
    inventory = trainval["tokenizer"]["file_inventory"]
    inventory_hashes = {}
    for item in inventory:
        path = TOKENIZER_DIR / item["name"]
        actual = sha_file(path)
        assert actual == item["sha256"], f"Tokenizer asset hash mismatch: {item['name']}"
        inventory_hashes[item["name"]] = actual

    # Tokenizer-only import and local load. Do not instantiate a model or CUDA.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_DIR), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    prefix = trainval["prompt"]["prompt_prefix_token_ids"]
    cue = trainval["prompt"]["answer_cue_token_ids"]
    assert trainval["prompt"]["article_token_cap"] == 96

    prior_ids = {r["record_id_sha256"] for r in trainval["records"]}
    prior_ids.update(r["record_id_sha256"] for r in v3_prompts["records"])
    prior_prefix_hashes = {r["prompt_sha256"] for r in trainval["records"]}
    prior_prefix_hashes.update(r["prompt_sha256"] for r in v3_prompts["records"])

    eligible_by_prompt = defaultdict(list)
    excluded_id = 0
    excluded_v3_prefix = 0
    for r in candidates["records"]:
        if r["record_id_sha256"] in prior_ids:
            excluded_id += 1
            continue
        article_tokens = tokenizer.encode(r["text"], add_special_tokens=False)[:96]
        prompt_tokens = prefix + article_tokens + cue
        prefix_hash = hash_obj(prompt_tokens)
        if prefix_hash in prior_prefix_hashes:
            excluded_v3_prefix += 1
            continue
        eligible_by_prompt[prefix_hash].append({
            "record_id_sha256": r["record_id_sha256"],
            "source_index": r["source_index"],
            "class_id": r["class_id"],
            "selection_key": sha_bytes((SALT + "\0" + r["record_id_sha256"]).encode()),
            "tokens": prompt_tokens,
            "prompt_sha256": prefix_hash,
        })

    # Drop tokenized-prefix collisions, including conflicts across classes.
    buckets = defaultdict(list)
    dropped_prefix_groups = 0
    dropped_cross_class_groups = 0
    for rows in eligible_by_prompt.values():
        classes = {r["class_id"] for r in rows}
        if len(classes) != 1:
            dropped_cross_class_groups += 1
            continue
        row = min(rows, key=lambda x: (x["selection_key"], x["record_id_sha256"]))
        dropped_prefix_groups += len(rows) - 1
        buckets[row["class_id"]].append(row)

    selected = []
    for class_id in CLASSES:
        ordered = sorted(buckets[class_id], key=lambda r: (r["selection_key"], r["record_id_sha256"]))
        assert len(ordered) >= PER_CLASS, f"Insufficient eligible candidates for class {class_id}"
        selected.extend(ordered[:PER_CLASS])
    random.Random(SHUFFLE_SEED).shuffle(selected)

    selected_ids = {r["record_id_sha256"] for r in selected}
    selected_prefixes = {r["prompt_sha256"] for r in selected}
    assert len(selected) == 1024
    assert len(selected_ids) == len(selected_prefixes) == 1024
    assert not selected_ids.intersection(prior_ids)
    assert not selected_prefixes.intersection(prior_prefix_hashes)
    assert all(sum(r["class_id"] == c for r in selected) == PER_CLASS for c in CLASSES)

    label_map = {r["record_id_sha256"]: r["class_id"] for r in selected}
    prompt_records = [{k: r[k] for k in ("record_id_sha256", "source_index", "tokens", "prompt_sha256")} for r in selected]
    # Ensure class/phase metadata cannot escape in prompt file.
    assert all("class_id" not in r and "phase" not in r for r in prompt_records)

    write_new(OUT / "prompts-sealed.json", {"schema": "agnews-v4-confirmation-prompts-sealed", "records": prompt_records})
    write_new(OUT / "labels-sealed.json", {"schema": "agnews-v4-confirmation-labels-sealed", "labels": label_map})
    manifest = {
        "schema": "agnews-v4-confirmation-reservation-manifest",
        "status": "reserved_unscored",
        "salt": SALT,
        "shuffle_seed": SHUFFLE_SEED,
        "source_files": {
            "candidates_private_sha256": sha_file(CANDIDATES),
            "train_validation_private_sha256": sha_file(TRAIN_VALIDATION),
            "v3_holdout_prompts_sealed_sha256": sha_file(V3_PROMPTS),
        },
        "tokenizer": {"model_directory": "D:/my_paper/Nested_Learning_and_Sleep/.cache/models/llama32-3b", "asset_inventory_sha256": inventory_hashes, "pad_token_id": trainval["tokenizer"].get("pad_token_id"), "article_token_cap": 96},
        "outputs": {
            "prompts_sealed_sha256": sha_file(OUT / "prompts-sealed.json"),
            "labels_sealed_sha256": sha_file(OUT / "labels-sealed.json"),
        },
        "counts": {
            "candidate_records": len(candidates["records"]),
            "v3_train_validation_records_excluded": len(trainval["records"]),
            "v3_holdout_records_excluded": len(v3_prompts["records"]),
            "candidate_normalized_id_overlaps_excluded": excluded_id,
            "candidate_v3_tokenized_prefix_overlaps_excluded": excluded_v3_prefix,
            "eligible_unique_tokenized_prefixes": len(eligible_by_prompt),
            "duplicate_prefix_records_dropped": dropped_prefix_groups,
            "cross_class_prefix_groups_dropped": dropped_cross_class_groups,
            "reserved_total": len(selected),
            "reserved_per_class": {str(c): sum(r["class_id"] == c for r in selected) for c in CLASSES},
        },
        "intersections": {
            "normalized_id_intersection_with_all_v3_roles_zero": not bool(selected_ids.intersection(prior_ids)),
            "tokenized_prefix_intersection_with_all_v3_roles_zero": not bool(selected_prefixes.intersection(prior_prefix_hashes)),
            "selected_normalized_id_unique": len(selected_ids) == len(selected),
            "selected_tokenized_prefix_unique": len(selected_prefixes) == len(selected),
        },
        "interpretation": "Unscored within-AGNews-train confirmation reserve; not external or pretraining-independent confirmation.",
        "scored": False,
        "reserved_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_new(OUT / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output_directory": str(OUT), "counts": manifest["counts"], "intersections": manifest["intersections"], "output_hashes": manifest["outputs"]}, indent=2))

if __name__ == "__main__":
    main()
