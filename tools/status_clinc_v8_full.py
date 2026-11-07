"""Read-only scalar progress of a CLINC grid; no text/prediction output."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def status(root):
    grid = Path(root)
    rows = []
    for mode, dirs in (("foundation", sorted((grid / "foundations").glob("seed*"))),
                       ("continuation", sorted((grid / "runs").glob("seed*/*")))):
        for d in dirs:
            if not d.is_dir():
                continue
            row = {"mode": mode, "path": str(d.relative_to(grid)), "status": "in_progress"}
            log = d / "metrics.jsonl"
            if log.exists():
                with log.open(encoding="utf-8") as f:
                    for line in f:
                        try:
                            j = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if j.get("event") == "wake_update":
                            row.update(task_id=j["task_id"], step=j["step"], wake_tick=j["global_wake_tick"])
                        if j.get("event") == "sleep_commit":
                            row.update(last_sleep_event=j["sleep_event_index"], sleep_tick=j["global_wake_tick"])
                        if j.get("event") == "task_end_clean_development":
                            m = j["result"]["metrics"]
                            row["last_task_end"] = {"task_id": j["task_id"],
                                                    **{k: m[k] for k in ("old", "current", "seen")}}
            receipt = d / ("foundation-receipt.json" if mode == "foundation" else "run-receipt.json")
            if receipt.exists():
                j = json.loads(receipt.read_text())
                row.update(status=j["status"], tasks_completed=j["tasks_completed"],
                           seconds=j["active_wall_seconds"], physical_sleep_adamw_steps=j["physical_sleep_adamw_steps"],
                           forward_calls=j["work"].get("full_backbone_forward_calls"),
                           checkpoint_sha256=j["final_checkpoint_sha256"])
            if (d / "failure.json").exists():
                j = json.loads((d / "failure.json").read_text())
                row["failure"] = {k: j.get(k) for k in ("status", "phase", "error_type", "error", "elapsed_seconds")}
            rows.append(row)
    queue = []
    if (grid / "queue.jsonl").exists():
        for line in (grid / "queue.jsonl").read_text().splitlines():
            try:
                queue.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return {"as_of_utc": datetime.now(timezone.utc).isoformat(), "runs": rows,
            "last_queue_events": queue[-2:], "main_seal_exists": (grid / "main-six-seal.json").exists()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-root", required=True)
    args = p.parse_args()
    print(json.dumps(status(args.grid_root), ensure_ascii=False))
