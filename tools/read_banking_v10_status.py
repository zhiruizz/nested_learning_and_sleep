"""Read only the finite Banking V10 queue's small status artifacts."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


def read(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as exc:
        return {"read_error": type(exc).__name__}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    queue = read(root / "queue-status.json")
    result = {"checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "queue_exists": root.is_dir(), "queue": None, "runs": []}
    if queue:
        result["queue"] = {k: queue.get(k) for k in
                           ("status", "pid", "queue_sha256", "automatic_retries", "started_unix", "ended_unix")}
        result["queue"]["jobs"] = [{k: r.get(k) for k in
                                    ("job", "status", "output_dir", "elapsed_wall_seconds", "error")}
                                   for r in queue.get("jobs", [])]
    for seed in (17, 29, 43):
        for arm in ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE"):
            run = root / f"seed{seed}-{arm.lower()}"
            if not run.is_dir():
                run = root / "runs" / f"seed{seed}" / arm
            if not run.is_dir():
                continue
            progress = read(run / "progress.json") or {}
            receipt = read(run / "run-receipt.json") or {}
            failure = read(run / "failure.json")
            work = progress.get("work", {})
            result["runs"].append({"seed": seed, "arm": arm,
                "status": receipt.get("status", progress.get("status")),
                "task_id": progress.get("task_id"), "task_step": progress.get("task_step"),
                "global_wake_tick": progress.get("global_wake_tick"),
                "event_counts": progress.get("event_counts"),
                "checkpoint_files": len(list((run / "checkpoints").glob("*.pt"))),
                "work": {k: work.get(k) for k in ("active_wall_seconds", "full_backbone_forward_calls",
                    "physical_adamw_updates", "wake_updates", "sleep_events")},
                "failure": failure})
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
