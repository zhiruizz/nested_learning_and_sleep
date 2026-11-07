#!/usr/bin/env python3
"""Build an article-safe attempt/cost ledger from CLINC V8 run receipts.

This deliberately reads only run metadata JSON/JSONL. It never opens model
checkpoints, dataset files, token arrays, or evaluation predictions. Missing
cost counters remain null; source artifacts are selected explicitly so a
receipt and its mirrored metrics are never counted twice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORK_KEYS = (
    "active_wall_seconds",
    "full_backbone_forward_calls",
    "full_backbone_forward_tokens",
    "tail_calls",
    "tail_backward_calls",
    "tail_backward_tokens",
    "optimizer_or_lifecycle_update_steps",
    "sleep_events",
    "sampled_answer_tokens_including_first_eos",
    "executed_input_tokens_including_padding",
    "padded_input_tokens",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return obj


def work_compact(work: Any) -> dict[str, Any] | None:
    if not isinstance(work, dict):
        return None
    # Keep absent counters absent. Never coerce an omitted field to zero.
    return {k: work[k] for k in WORK_KEYS if k in work}


def artifact_ref(root: Path, output_id: str, relative: str) -> dict[str, str] | None:
    path = root / output_id / relative
    if not path.is_file():
        return None
    return {"output_id": output_id, "file": relative.replace("\\", "/"), "sha256": sha256_file(path)}


def record(root: Path, *, rid: str, output_id: str, phase: str, status: str,
           artifact_file: str, notes: str = "", source_field: str = "work",
           elapsed_field: str = "elapsed_seconds", explicit: dict[str, Any] | None = None) -> dict[str, Any]:
    obj = load_json(root / output_id / artifact_file)
    ref = artifact_ref(root, output_id, artifact_file)
    if obj is None:
        return {
            "id": rid, "output_id": output_id, "phase": phase,
            "status": "artifact_missing", "cost": None, "artifact": None,
            "notes": notes or "Expected receipt missing; cost is unknown.",
        }
    cost = work_compact(obj.get(source_field))
    if cost is None:
        cost = {}
    if elapsed_field in obj:
        cost["reported_elapsed_seconds"] = obj[elapsed_field]
    if explicit:
        cost.update(explicit)
    return {
        "id": rid, "output_id": output_id, "phase": phase,
        "status": status, "cost": cost or None,
        "artifact": ref,
        "error": obj.get("error"),
        "notes": notes,
    }


def get_metric_records(root: Path, output_id: str, relative: str) -> list[dict[str, Any]]:
    path = root / output_id / relative
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # Deliberately discard row-level results, IDs, and any possible text.
            rows.append(obj)
    return rows


def full_run_progress(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_id = "clinc-v8-full-k10-20261107-a"
    run_root = root / output_id
    queue_path = run_root / "queue.jsonl"
    queue = get_metric_records(root, output_id, "queue.jsonl")
    latest_queue = queue[-1] if queue else {}
    # Read receipts as the unique source for shared foundations.
    foundation_rows = []
    foundation_total: dict[str, Any] = {}
    for seed in (17, 29, 43):
        rel = f"foundations/seed{seed}/foundation-receipt.json"
        p = run_root / rel
        d = load_json(p)
        if d is None:
            foundation_rows.append({"seed": seed, "status": "missing", "work": None, "artifact": None})
            continue
        w = work_compact(d.get("work"))
        foundation_rows.append({
            "seed": seed,
            "status": d.get("status", "unknown"),
            "work": w,
            "receipt_elapsed_seconds": d.get("active_wall_seconds"),
            "artifact": {"file": rel, "sha256": sha256_file(p)},
            "qualification": d.get("qualification"),
        })
        if d.get("status") == "complete" and w:
            for k in WORK_KEYS:
                if k in w and isinstance(w[k], (int, float)):
                    foundation_total[k] = foundation_total.get(k, 0) + w[k]
    queue_time = latest_queue.get("utc")
    protocol = load_json(run_root / "protocol.json")
    frozen_sha = None
    if protocol:
        cfg = protocol.get("config_sha256")
        frozen_sha = cfg
    run_rows: list[dict[str, Any]] = []
    runs_dir = run_root / "runs"
    if runs_dir.is_dir():
        for metrics_path in sorted(runs_dir.glob("seed*/*/metrics.jsonl")):
            seed_dir, arm_dir = metrics_path.parent.parent.name, metrics_path.parent.name
            try:
                seed = int(seed_dir.removeprefix("seed"))
            except ValueError:
                seed = None
            relbase = metrics_path.parent.relative_to(run_root).as_posix()
            receipt_path = metrics_path.parent / "run-receipt.json"
            receipt = load_json(receipt_path)
            events = get_metric_records(root, output_id, f"{relbase}/metrics.jsonl")
            wakes = [r for r in events if r.get("event") == "wake_update"]
            starts = [r for r in events if r.get("event") == "task_start"]
            ends = [r for r in events if r.get("event") == "task_end_clean_development"]
            last = wakes[-1] if wakes else {}
            matching_queue = [q for q in queue if q.get("seed") == seed and q.get("arm") == arm_dir]
            run_queue = matching_queue[-1] if matching_queue else {}
            if receipt:
                w = work_compact(receipt.get("work")) or {}
                # Physical totals include the shared foundation; preserve per-process
                # update counters separately so foundation work is not added twice.
                cost = dict(w)
                cost.update({
                    "physical_wake_updates_including_shared_foundation": receipt.get("physical_wake_updates"),
                    "physical_sleep_adamw_steps": receipt.get("physical_sleep_adamw_steps"),
                    "adamw_step_attempts_this_process": receipt.get("adamw_step_attempts_this_process"),
                    "adamw_step_completions_this_process": receipt.get("adamw_step_completions_this_process"),
                    "tasks_completed": receipt.get("tasks_completed"),
                    "reported_elapsed_seconds": receipt.get("active_wall_seconds"),
                })
                status = receipt.get("status", "unknown")
                notes = "Run receipt is canonical for this arm. Its physical_wake_updates includes the shared foundation; work.optimizer_or_lifecycle_update_steps is the run's own per-process continuation counter. Foundation receipt is listed separately and must not be re-added."
            else:
                cost = {
                    "wake_update_events_observed": len(wakes),
                    "optimizer_or_lifecycle_update_steps": len(wakes),
                    "full_backbone_forward_calls": None,
                    "physical_adamw_calls": None,
                    "sleep_events": None,
                    "reported_elapsed_seconds": None,
                }
                status = "in_progress"
                notes = "No terminal run receipt yet. Wake-update event count is observed so far; missing forward/Adam/Sleep/wall totals remain unknown, not zero."
            run_rows.append({
                "id": f"full-grid-seed{seed}-{arm_dir}", "output_id": output_id,
                "phase": "full-grid-continuation", "status": status,
                "seed": seed, "arm": arm_dir, "cost": cost or None,
                "latest_observed": {
                    "task_id": last.get("task_id"), "task_step": last.get("step"),
                    "global_wake_tick": last.get("global_wake_tick"),
                    "wake_updates_seen": len(wakes),
                    "tasks_started": [r.get("task_id") for r in starts],
                    "tasks_cleanly_ended": [r.get("task_id") for r in ends],
                    "last_queue_event": run_queue.get("event"), "last_queue_utc": run_queue.get("utc"),
                },
                "protocol_config_sha256": frozen_sha,
                "metrics_artifact": {"file": f"{relbase}/metrics.jsonl", "sha256": sha256_file(metrics_path)},
                "receipt_artifact": {"file": f"{relbase}/run-receipt.json", "sha256": sha256_file(receipt_path)} if receipt_path.is_file() else None,
                "notes": notes,
            })
    return run_rows, {
        "output_id": output_id,
        "foundations": foundation_rows,
        "unique_shared_foundation_work_total": foundation_total or None,
        "notes": "Each seed's task-0 foundation is a shared starting checkpoint and is counted once, not once per future arm.",
    }


def build(root: Path) -> dict[str, Any]:
    entries = [
        record(root, rid="real3b-tail-qualification", output_id="clinc-v8-real3b-qualification-20261107-a", phase="engineering-qualification", status="complete", artifact_file="check-result.json", explicit={"full_backbone_forward_calls": 3, "physical_adamw_steps_completed": 0}, notes="3 full forwards, no optimizer updates; exact full-vs-tail output and gradient qualification."),
        record(root, rid="real3b-resume-memory", output_id="clinc-v8-real3b-resume-20261107-a", phase="engineering-resume-qualification", status="complete", artifact_file="check-result.json", source_field="", explicit={"full_backbone_forward_calls": 24, "logical_resume_updates": 8, "physical_adamw_steps_completed": 8}, notes="Actual 3B lifecycle resume fixture; two pre-steps, three reference and three restored replay AdamW steps."),
        record(root, rid="real3b-resume-disk-attempt-b", output_id="clinc-v8-real3b-resume-20261107-b-disk", phase="engineering-resume-disk-roundtrip", status="failed", artifact_file="failure.json", explicit={"full_backbone_forward_calls": 6, "logical_resume_updates": 2, "physical_adamw_steps_completed": 2, "wake_ticks_observed": 7}, notes="Disk roundtrip attempt failed with Bad file descriptor during the sleep-pre-updates phase; preserve its partial cost (2 AdamW steps, 6 forwards)."),
        record(root, rid="real3b-resume-disk-attempt-c", output_id="clinc-v8-real3b-resume-20261107-c-disk", phase="engineering-resume-disk-roundtrip", status="complete", artifact_file="check-result.json", source_field="", explicit={"full_backbone_forward_calls": 24, "logical_resume_updates": 8, "physical_adamw_steps_completed": 8}, notes="Fresh disk roundtrip replay qualification after attempt B; counts are for this attempt only."),
        record(root, rid="r1-task0-sleep-failure", output_id="clinc-v8-dev-seed17-seen-replay-20261107-r1", phase="training-attempt", status="failed", artifact_file="failure.json", explicit={"logical_update_counter_decomposition": {"wake_updates": 252, "sleep_update_attempts": 10}, "physical_adamw_steps_completed": 9, "physical_adamw_step_failed_before_update": 1}, notes="R1 run failed the finite differentiable-loss guard at task 0. Work counter 262 comprises 252 wake updates plus 10 Sleep optimizer attempts; 9 physical Sleep AdamW steps completed and the tenth failed before step. R1 evaluation metric (3.33%) is invalid due to row/length alignment and is not a valid score."),
        record(root, rid="r1-corrected-rescore-a", output_id="clinc-v8-r1-hope-corrected-eval-20261107-a", phase="evaluation-repair", status="complete", artifact_file="evaluation.json", source_field="work", notes="No training or optimizer update; rescored sealed R1 checkpoint on clean task-0 development after row alignment correction."),
        record(root, rid="r1-corrected-rescore-b", output_id="clinc-v8-r1-hope-corrected-eval-20261107-b-finalsource", phase="evaluation-repair", status="complete", artifact_file="evaluation.json", source_field="work", notes="Repeated engineering rescore under final source binding; same checkpoint and fixed clean development subset."),
        record(root, rid="r1-corrected-rescore-c", output_id="clinc-v8-r1-hope-corrected-eval-20261107-c-hashdomain", phase="evaluation-repair", status="complete", artifact_file="evaluation.json", source_field="work", notes="Repeated source/hash-domain validation rescore; no training. Three rescoring attempts are separate costs, not independent model seeds."),
        record(root, rid="no-sleep-task1-resume-r2", output_id="clinc-v8-hope-task1-resume-20261107-r2", phase="training-resume", status="complete", artifact_file="development-sealed.json", source_field="work", notes="Seed 17 HOPE task-1 continuation, no Sleep events; 188 update-counter steps."),
        record(root, rid="natural-routing-sleep-failure-replay", output_id="clinc-v8-sparse-sleep-diagnostic-20261107-a", phase="failure-diagnostic-replay", status="failed-expected", artifact_file="routing-diagnostic.json", source_field="r1_failure_work", elapsed_field="elapsed_seconds", explicit={"diagnostic_outer_elapsed_seconds": 23.75, "wake_updates_observed": 64, "physical_adamw_steps_attempted": 9, "physical_adamw_steps_completed": 9}, notes="Bounded deterministic replay isolated natural routing starvation: all 8 answer positions missed receiving slot 3; finite scalar loss had requires_grad=false. Embedded r1_failure_work is this replay's 64-wake prefix, not a second copy of the earlier R1 failure."),
        record(root, rid="controlled-real3b-qualification-a", output_id="clinc-v8-controlled-real3b-qualification-20261107-a", phase="engineering-qualification", status="failed", artifact_file="failure.json", explicit={"full_backbone_forward_calls": 0, "physical_adamw_steps_completed": 0}, notes="Interface NameError before successful qualification; receipt explicitly records zero model forwards and zero AdamW calls."),
        record(root, rid="controlled-real3b-qualification-b", output_id="clinc-v8-controlled-real3b-qualification-20261107-b", phase="engineering-qualification", status="complete", artifact_file="check-result.json", source_field="", explicit={"full_backbone_forward_calls": 24, "logical_resume_updates": 8, "physical_adamw_steps_completed": 8}, notes="Controlled-route 3B resume qualification; 8 AdamW replay fixture steps, 24 forwards, exact state/RNG checks."),
        record(root, rid="controlled-launch-sentinel", output_id="clinc-v8-controlled-launch-qualification-20261107-b", phase="host-launch-qualification", status="intentional-sentinel-stop", artifact_file="qualification-result.json", source_field="", explicit={"full_backbone_forward_calls": 0, "physical_adamw_steps_completed": 0, "model_loaded": False, "gpu_initialized": False, "reported_elapsed_seconds": 0.016}, notes="Startup/source/data binding reached; intentional stop before model load and GPU work. Failure JSON mirrors the zero-work qualification result and is not counted again."),
        record(root, rid="controlled-task1-launch-r1", output_id="clinc-v8-controlled-seed17-task1-20261107-r1", phase="host-launch-failure", status="failed", artifact_file="host-launch-failure.json", source_field="", explicit={"full_backbone_forward_calls": 0, "physical_adamw_steps_completed": 0, "optimizer_or_lifecycle_update_steps": 0}, notes="Python local-name shadowing stopped startup before model load; zero forwards, AdamW calls, and wake updates are explicit in the receipt."),
        record(root, rid="controlled-task1-r2", output_id="clinc-v8-controlled-seed17-task1-20261107-r2", phase="training-probe", status="complete", artifact_file="development-sealed.json", source_field="work", notes="Seed-17 two-arm controlled route probe; shared paired task, 32 Sleep events across both arms. Counters are aggregate for the pair."),
        record(root, rid="controlled-boundary-a", output_id="clinc-v8-controlled-boundaries-20261107-a", phase="boundary-scorer-attempt", status="failed", artifact_file="failure.json", notes="Evaluator API error before any backbone forward; 22.437 seconds recorded."),
        record(root, rid="controlled-boundary-b", output_id="clinc-v8-controlled-boundaries-20261107-b", phase="boundary-scorer-attempt", status="failed-budget-cap", artifact_file="failure.json", notes="Reached configured forward cap before finishing; output retains exactly 10,000 full-backbone forwards and 978,011 forward tokens. Not a completed score."),
        record(root, rid="controlled-boundary-c", output_id="clinc-v8-controlled-boundaries-20261107-c", phase="boundary-scorer-attempt", status="complete", artifact_file="evaluation.json", source_field="work", notes="Read-only boundary score completed with 10,347 full-backbone forwards; this observed count exceeds the earlier failed attempt's 10,000 cap. Zero wake and optimizer updates. Not an independent seed."),
        record(root, rid="transfer-startup-k1", output_id="clinc-v8-transfer-startup-k1-20261107-a", phase="host-launch-qualification", status="intentional-sentinel-stop", artifact_file="qualification-result.json", source_field="", explicit={"full_backbone_forward_calls": 0, "physical_adamw_steps_completed": 0, "model_loaded": False, "reported_elapsed_seconds": 0.0}, notes="k=1 startup binding sentinel; zero model forward and optimizer work."),
        record(root, rid="transfer-startup-k10", output_id="clinc-v8-transfer-startup-k10-20261107-a", phase="host-launch-qualification", status="intentional-sentinel-stop", artifact_file="qualification-result.json", source_field="", explicit={"full_backbone_forward_calls": 0, "physical_adamw_steps_completed": 0, "model_loaded": False, "reported_elapsed_seconds": 0.0}, notes="k=10 startup binding sentinel; zero model forward and optimizer work."),
        record(root, rid="transfer-k1-corrected", output_id="clinc-v8-transfer-k1-corrected-20261107-a", phase="training-probe", status="complete", artifact_file="development-sealed.json", source_field="work", notes="Corrected two-arm transfer probe, k=1, seed 17; 472 aggregate update-counter steps and 32 Sleep events across both arms."),
        record(root, rid="transfer-k10-corrected", output_id="clinc-v8-transfer-k10-corrected-20261107-a", phase="training-probe", status="complete", artifact_file="development-sealed.json", source_field="work", notes="Corrected two-arm transfer probe, k=10, seed 17; 1,336 aggregate update-counter steps and 32 Sleep events across both arms."),
    ]

    progress_rows, foundations = full_run_progress(root)
    entries.extend(progress_rows)
    return {
        "schema": "clinc-v8-attempt-cost-ledger-v1",
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "V8 CLINC development, engineering, correction, and current full-grid execution attempts through the as-of time",
        "source_root_label": "Nested_Learning_and_Sleep/outputs",
        "source_policy": "Explicit JSON/JSONL metadata receipts only; no model checkpoints, test files, dataset text, prompts, predictions, or token arrays read.",
        "accounting_rules": [
            "Each attempt references one canonical artifact; mirrored metrics/receipt views are not added a second time.",
            "optimizer_or_lifecycle_update_steps is the artifact's logical/update counter, not automatically a physical AdamW count.",
            "physical_adamw_steps is populated only where a receipt explicitly reports it; otherwise it remains absent or null.",
            "Repeated qualifications, rescoring passes, and bounded failure replays are separate compute attempts but are not independent scientific seeds.",
            "Missing counters are unknown, never zero. Intentional zero-work sentinel receipts may report exact zeros.",
            "The three task-0 foundation receipts are unique shared starting work, counted once across future arms.",
            "The full-grid current continuation is in progress and excluded from completed-run totals.",
        ],
        "attempts": entries,
        "shared_starting_assets": foundations,
        "known_results": {
            "invalid_original_r1_task0_metric": {"accuracy": 0.03333333333333333, "status": "invalid_row_length_alignment", "replacement_valid_score": 0.7233333333333334, "replacement_score_scope": "300-row clean task-0 development subset, seen 15 classes", "rescore_attempts": 3},
            "natural_routing_failure": {"answer_tokens": 8, "receiver_answer_coverage": 0.0, "loss_finite": True, "loss_requires_grad": False, "physical_adamw_steps_completed": 9, "status": "guard_failure_reproduced"},
            "controlled_boundary_readonly": {"paired_event_count": 32, "optimizer_steps": 0, "wake_updates": 0, "status": "completed_read_only_score"},
        },
        "limitations": [
            "The ledger is an as-of snapshot; in-progress full-grid counters can advance after generation.",
            "Failure replay and engineering qualification costs are reported separately from main scientific runs.",
            "No evaluation/test result is inferred from missing, partial, or failed artifacts.",
            "The full-run receipt's sleep_text_replay availability flag is not treated as actual exposure; actual arm losses and Sleep events govern accounting. HOPE has no replay-loss exposure where no Sleep commit occurred.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True, help="CLINC V8 outputs directory (metadata only)")
    p.add_argument("--out", type=Path, required=True, help="Destination aggregate JSON")
    args = p.parse_args()
    data = build(args.output_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"out": args.out.name, "attempt_count": len(data["attempts"]), "as_of_utc": data["as_of_utc"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
