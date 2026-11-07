#!/usr/bin/env python3
"""Compare saved V11 request-cohort digests without exposing row identifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SEEDS = (17, 29, 43)
ARMS = ("LTI_NO_REWARD", "LTI_CANONICAL_TEACHER")
ARM_DIRS = {
    "LTI_NO_REWARD": "lti-no-reward",
    "LTI_CANONICAL_TEACHER": "lti-canonical-teacher",
}
HEX = set("0123456789abcdef")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.isascii()
        and set(value.lower()) <= HEX
    )


def read_run(path: Path, seed: int, arm: str) -> tuple[dict[str, Any], dict[tuple[int, int, int], dict[str, Any]]]:
    log_sha = _sha256_file(path)
    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows += 1
            row = json.loads(line)
            if row.get("seed") != seed or row.get("arm") != arm:
                raise ValueError("metrics log seed/arm does not match its expected run")
            digest = row.get("request_ids_sha256")
            if digest is None:
                continue
            if not _valid_digest(digest):
                raise ValueError("request cohort digest is missing or malformed")
            event_kind = row.get("event")
            if not isinstance(event_kind, str):
                raise ValueError("event kind must be a string")
            level = _strict_int(row.get("level"), "level")
            task_id = _strict_int(row.get("task_id"), "task_id")
            event_index = _strict_int(row.get("event_index"), "event_index")
            selector_index = _strict_int(
                row.get("global_selector_event_index"), "global_selector_event_index"
            )
            global_wake_tick = _strict_int(row.get("global_wake_tick"), "global_wake_tick")
            requests = row.get("requests_by_role")
            if not isinstance(requests, dict) or set(requests) != {"old", "current"}:
                raise ValueError("requests_by_role must contain exactly old/current")
            role_counts = {
                role: _strict_int(requests[role], f"requests_by_role.{role}")
                for role in ("old", "current")
            }
            if any(v < 0 for v in role_counts.values()):
                raise ValueError("role request counts cannot be negative")
            key = (level, task_id, event_index)
            if key in records:
                raise ValueError("duplicate event key within one run")
            records[key] = {
                "digest": digest.lower(),
                "event_kind": event_kind,
                "selector_index": selector_index,
                "global_wake_tick": global_wake_tick,
                "role_counts": role_counts,
            }
    if rows == 0:
        raise ValueError("empty metrics log")
    if _sha256_file(path) != log_sha:
        raise RuntimeError("metrics log changed during cohort analysis")
    return (
        {
            "seed": seed,
            "arm": arm,
            "logical_file": f"seed{seed}/{arm}/metrics.jsonl",
            "sha256": log_sha,
            "jsonl_rows": rows,
            "saved_cohort_events": len(records),
        },
        records,
    )


def build_report(run_root: Path) -> dict[str, Any]:
    run_metadata: dict[tuple[int, str], dict[str, Any]] = {}
    event_maps: dict[tuple[int, str], dict[tuple[int, int, int], dict[str, Any]]] = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = run_root / f"seed{seed}-{ARM_DIRS[arm]}" / "metrics.jsonl"
            meta, events = read_run(path, seed, arm)
            run_metadata[(seed, arm)] = meta
            event_maps[(seed, arm)] = events

    seed_reports = []
    totals = Counter()
    kind_counts = Counter()
    for seed in SEEDS:
        left = event_maps[(seed, ARMS[0])]
        right = event_maps[(seed, ARMS[1])]
        left_keys, right_keys = set(left), set(right)
        shared = left_keys & right_keys
        matched_digest = 0
        matched_plan = 0
        mismatches = Counter()
        role_totals = {
            arm: Counter(old=0, current=0) for arm in ARMS
        }
        for key, event in left.items():
            for role in ("old", "current"):
                role_totals[ARMS[0]][role] += event["role_counts"][role]
            kind_counts[(ARMS[0], event["event_kind"])] += 1
        for key, event in right.items():
            for role in ("old", "current"):
                role_totals[ARMS[1]][role] += event["role_counts"][role]
            kind_counts[(ARMS[1], event["event_kind"])] += 1

        for key in shared:
            a, b = left[key], right[key]
            if a["digest"] == b["digest"]:
                matched_digest += 1
            else:
                mismatches["request_digest"] += 1
            plan_equal = True
            for field in ("event_kind", "selector_index", "global_wake_tick", "role_counts"):
                if a[field] != b[field]:
                    mismatches[field] += 1
                    plan_equal = False
            if plan_equal:
                matched_plan += 1

        missing_left = len(right_keys - left_keys)
        missing_right = len(left_keys - right_keys)
        totals.update(
            paired_keys=len(shared),
            digest_matches=matched_digest,
            plan_matches=matched_plan,
            missing_no_reward=missing_left,
            missing_canonical=missing_right,
        )
        seed_reports.append(
            {
                "seed": seed,
                "events": {
                    ARMS[0]: len(left),
                    ARMS[1]: len(right),
                    "paired_event_keys": len(shared),
                    "request_digest_matches": matched_digest,
                    "full_saved_plan_matches": matched_plan,
                    "missing_from_no_reward": missing_left,
                    "missing_from_canonical_teacher": missing_right,
                    "mismatches_by_field": dict(sorted(mismatches.items())),
                },
                "role_request_totals": {
                    arm: dict(role_totals[arm]) for arm in ARMS
                },
            }
        )

    return {
        "schema": "banking-v11-request-cohort-balance-v1",
        "scope": {
            "compared_arms": list(ARMS),
            "seeds": list(SEEDS),
            "comparison_key": ["seed", "level", "task_id", "event_index"],
            "compared_saved_fields": [
                "request_ids_sha256",
                "event",
                "global_selector_event_index",
                "global_wake_tick",
                "requests_by_role.old/current",
            ],
            "raw_request_ids_or_text_exported": False,
            "token_ids_or_labels_exported": False,
            "no_sampling_or_rng_trajectory_equivalence_claim": True,
        },
        "input_logs": [run_metadata[(seed, arm)] for seed in SEEDS for arm in ARMS],
        "summary": {
            "total_events_per_arm": {
                arm: sum(len(event_maps[(seed, arm)]) for seed in SEEDS)
                for arm in ARMS
            },
            "paired_event_keys": totals["paired_keys"],
            "matching_request_digest_events": totals["digest_matches"],
            "matching_saved_plan_events": totals["plan_matches"],
            "events_missing_from_no_reward": totals["missing_no_reward"],
            "events_missing_from_canonical_teacher": totals["missing_canonical"],
            "event_kind_counts": {
                arm: dict(sorted((kind, count) for (a, kind), count in kind_counts.items() if a == arm))
                for arm in ARMS
            },
        },
        "by_seed": seed_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite an existing cohort report")
    report = build_report(args.run_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "schema": report["schema"],
        "events_per_arm": report["summary"]["total_events_per_arm"],
        "paired_event_keys": report["summary"]["paired_event_keys"],
        "request_digest_matches": report["summary"]["matching_request_digest_events"],
        "saved_plan_matches": report["summary"]["matching_saved_plan_events"],
        "output": str(args.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
