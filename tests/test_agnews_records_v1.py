"""CPU-only fixture tests for the root-owned AG News frozen-record contract."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from agnews_records_v1 import EXPECTED_SEEDS, SLEEP_CLASS_ORDER, validate_agnews_records


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_fixture():
    doc = {
        "schema": "nl-sleep-agnews-train-roles-v1",
        "source": {"provider": "ModelScope", "revision": "fixture-revision", "train_file_sha256": _sha("train")},
        "tokenizer": {"file_inventory": [{"name": "tokenizer.json", "sha256": _sha("tokenizer")} ]},
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
        "phase_class_ids": {"old": [0, 1], "new": [2, 3]},
        "prompt": {"prefix": "fixture prefix", "prompt_prefix_token_ids": [1], "article_token_cap": 96,
                   "answer_cue_token_ids": [2, 3], "answer_cue": "\nAnswer:",
                   "max_prompt_tokens": 128},
        "test_opened": False,
        "records": [], "code_maps": {}, "schedules": {}, "sleep_prefix_pools": {}
    }
    lookup = {}
    next_id = 0
    for partition in ("dev", "main"):
        for class_id in range(4):
            phase = "old" if class_id in (0, 1) else "new"
            train_role, calib_role = phase + "_train", phase + "_calib"
            train_n = 128 if partition == "dev" else 256
            for role, n in ((train_role, train_n), (calib_role, 64)):
                for _ in range(n):
                    row = {"record_id_sha256": _sha(f"id-{next_id}"), "partition": partition,
                           "phase": phase, "role": role, "class_id": class_id,
                           "tokens": [1] + [1000 + next_id] * 96 + [2, 3],
                           "article_token_count": 96, "article_was_truncated": True}
                    row["prompt_sha256"] = hashlib.sha256(
                        json.dumps(row["tokens"],separators=(",", ":")).encode()).hexdigest()
                    lookup.setdefault((partition, role, class_id), []).append(len(doc["records"]))
                    doc["records"].append(row)
                    next_id += 1
    for seed in EXPECTED_SEEDS:
        partition = "dev" if seed == "1907" else "main"
        doc["code_maps"][seed] = {"marker_token_id": 70, "class_to_token_ids": [71, 72, 73, 74]}
        schedule = {}
        for stage, role, classes, nsteps in (("foundation512", "old_train", (0, 1), 512),
                                             ("new_primary256", "new_train", (2, 3), 256)):
            pair = lookup[(partition, role, classes[0])][:2] + lookup[(partition, role, classes[1])][:2]
            schedule[stage] = [pair[:] for _ in range(nsteps)]
        doc["schedules"][seed] = schedule
        pool = []
        role_lookup = {c: lookup[(partition, "old_train" if c in (0,1) else "new_train", c)] for c in range(4)}
        for event in range(4):
            per_class_offset = event * 8
            ids = []
            for cls in SLEEP_CLASS_ORDER:
                choices = role_lookup[cls]
                ids.append(choices[per_class_offset % len(choices)])
                per_class_offset += 1
            pool.append(ids)
        doc["sleep_prefix_pools"][seed] = pool
    return doc


def test_complete_frozen_record_contract_and_schedule_integrity():
    report = validate_agnews_records(make_fixture())
    assert report["records"] == 2048
    assert report["old_class_ids"] == [0, 1]
    assert report["new_class_ids"] == [2, 3]
    assert set(report["schedules_sha256_by_seed"]) == set(EXPECTED_SEEDS)
    assert report["test_role_or_file_present"] is False
    assert report["raw_source_text_or_token_ids_returned"] is False


def test_training_sleep_and_extra_replay_batches_consume_frozen_global_indices():
    import torch
    from agnews_records_v1 import (collate_extra_replay_batch, collate_scheduled_batch,
                                   extra_replay_indices, scheduled_indices)
    doc = make_fixture()
    schedule_before = list(doc["schedules"]["1907"]["new_primary256"][0])
    primary_ids = scheduled_indices(doc, 1907, "new_primary256", 0)
    assert primary_ids == schedule_before
    ids, mask, lengths, markers, classes = collate_scheduled_batch(doc, 1907, "new_primary256", 0,
                                                                    torch.device("cpu"), 0)
    assert ids.shape == (4, 100)
    assert torch.all(mask.sum(dim=1) == 100)
    assert torch.all(ids[torch.arange(4), lengths] == markers)
    assert classes.tolist() == [73, 73, 74, 74]
    event_pool = doc["sleep_prefix_pools"]["1907"][0]
    replay_indices = extra_replay_indices(doc, 1907, 0, 0)
    assert replay_indices == event_pool[:4]
    replay = collate_extra_replay_batch(doc, 1907, 0, 0, torch.device("cpu"), 0)
    assert replay[0].shape == (4, 100)
    # Extra replay selection must never consume/shift the primary schedule.
    assert scheduled_indices(doc, 1907, "new_primary256", 0) == schedule_before


@pytest.mark.parametrize("mutation", ["test_payload", "raw_text", "bad_prefix", "bad_cue",
                                         "calibration_in_schedule", "bad_sleep_class_order",
                                         "duplicate_id", "bad_phase_groups", "code_collision"])
def test_invalid_or_leaky_records_are_refused(mutation):
    doc = make_fixture()
    if mutation == "test_payload":
        doc["source"]["official_test_file"] = "should-never-appear.parquet"
    elif mutation == "raw_text":
        doc["records"][0]["text"] = "raw article"
    elif mutation == "bad_prefix":
        doc["records"][0]["tokens"][0] = 9
    elif mutation == "bad_cue":
        doc["records"][0]["tokens"][-1] = 99
    elif mutation == "calibration_in_schedule":
        lookup_i = next(i for i,r in enumerate(doc["records"]) if r["partition"] == "dev" and r["role"] == "old_calib" and r["class_id"] == 0)
        doc["schedules"]["1907"]["foundation512"][0][0] = lookup_i
    elif mutation == "bad_sleep_class_order":
        i = doc["sleep_prefix_pools"]["1907"][0][0]
        doc["records"][i]["class_id"] = 3
    elif mutation == "duplicate_id":
        doc["records"][1]["record_id_sha256"] = doc["records"][0]["record_id_sha256"]
    elif mutation == "bad_phase_groups":
        doc["phase_class_ids"] = {"old": [0, 2], "new": [1, 3]}
    elif mutation == "code_collision":
        doc["code_maps"]["1907"]["class_to_token_ids"][3] = 72
    with pytest.raises(ValueError):
        validate_agnews_records(doc)
