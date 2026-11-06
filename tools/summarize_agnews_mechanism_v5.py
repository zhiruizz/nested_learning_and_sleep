"""Offline, JSON-only summary of AGNews v5 mechanism diagnostics.

This tool consumes aggregate metric/receipt JSON only. It never loads prompts,
records, checkpoints, models, or sealed data, and it performs no training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


PROBES = (
    "actual",
    "disable_fast_force_warm",
    "force_warm_receiver",
    "mix_warm_receiver",
    "restore_base0_force_warm",
    "restore_base1_force_warm",
    "restore_both_actual_route",
    "restore_both_force_warm",
    "warm_function_control",
)
SEEDS = (17, 29, 43)
GROUPS = ("base0", "base1", "fast_active_expert")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_summary(values):
    xs = [float(x) for x in values]
    return {
        "n": len(xs),
        "mean": statistics.mean(xs) if xs else None,
        "sample_sd": statistics.stdev(xs) if len(xs) > 1 else None,
        "per_seed": xs,
    }


def _metric(row, group, name):
    return float(row["metrics"][group][name])


def build_component_summary(data):
    rows = data["results"]
    cases = sorted({(x["arm"], int(x["step"])) for x in rows})
    if len(cases) != 5:
        raise ValueError(f"expected five (arm, step) cases; got {len(cases)}")
    out = []
    for arm, step in cases:
        subset = [x for x in rows if x["arm"] == arm and int(x["step"]) == step]
        if len(subset) != 27 or {x["probe"] for x in subset} != set(PROBES):
            raise ValueError(f"case {arm}@{step} is not 9 probes x 3 seeds")
        case = {"arm": arm, "step": step, "probes": []}
        for probe in PROBES:
            group = [x for x in subset if x["probe"] == probe]
            group.sort(key=lambda x: int(x["seed"]))
            if [int(x["seed"]) for x in group] != list(SEEDS):
                raise ValueError(f"missing seed for {arm}@{step}/{probe}")
            case["probes"].append({
                "probe": probe,
                "old_em": sample_summary([_metric(x, "old", "full_vocab_two_token_greedy_exact_match") for x in group]),
                "new_em": sample_summary([_metric(x, "new", "full_vocab_two_token_greedy_exact_match") for x in group]),
                "per_seed": [{
                    "seed": int(x["seed"]),
                    "old_em": _metric(x, "old", "full_vocab_two_token_greedy_exact_match"),
                    "new_em": _metric(x, "new", "full_vocab_two_token_greedy_exact_match"),
                    "old_nll": _metric(x, "old", "class_nll"),
                    "new_nll": _metric(x, "new", "class_nll"),
                    "old_receiver_tensors_unchanged": x.get("old_receiver_tensors_unchanged"),
                } for x in group],
            })
        out.append(case)
    warm = data.get("warm_references", [])
    if len(warm) != 3:
        raise ValueError("expected three distinct warm-reference records")
    warm.sort(key=lambda x: int(x["seed"]))
    warm_refs = {
        "definition": "warm_references exact actual-route references; separate from intervention rows",
        "old_em": sample_summary([float(x["metrics"]["old"]["full_vocab_two_token_greedy_exact_match"]) for x in warm]),
        "new_em": sample_summary([float(x["metrics"]["new"]["full_vocab_two_token_greedy_exact_match"]) for x in warm]),
        "per_seed": [{
            "seed": int(x["seed"]),
            "old_em": float(x["metrics"]["old"]["full_vocab_two_token_greedy_exact_match"]),
            "new_em": float(x["metrics"]["new"]["full_vocab_two_token_greedy_exact_match"]),
        } for x in warm],
    }
    return out, warm_refs


def build_route_rows(data):
    rows = []
    for x in data["results"]:
        case = f"{x['arm']}@{x['step']}"
        for bank_index, bank in enumerate(x.get("route_audit") or []):
            for group in ("old", "new"):
                rec = bank.get(group)
                if rec is None:
                    continue
                selected = {str(k): int(v) for k, v in rec.get("selected_slot_counts", {}).items()}
                rows.append({
                    "case": case,
                    "seed": int(x["seed"]),
                    "probe": x["probe"],
                    "bank": bank_index,
                    "group": group,
                    "positions": int(rec.get("positions", 0)),
                    "selected_slots": selected,
                    "controlled_mix_slot": rec.get("controlled_mix_slot"),
                    "controlled_mix_fraction": rec.get("controlled_mix_fraction"),
                    "null_gate_sum": float(rec.get("null_gate_sum", 0.0)),
                    "warm0_gate_sum": float(rec.get("warm0_gate_sum", 0.0)),
                    "latest_gate_sum": float(rec.get("latest_gate_sum", 0.0)),
                })
    return rows


def build_interference(data):
    gradient_rows = []
    trajectory = []
    for run in sorted(data["results"], key=lambda x: int(x["seed"])):
        seed = int(run["seed"])
        if len(run["gradient_probes"]) != 4:
            raise ValueError(f"seed {seed}: expected four gradient-probe states")
        for probe in run["gradient_probes"]:
            for group_name in GROUPS:
                g = probe["groups"][group_name]
                gradient_rows.append({
                    "seed": seed,
                    "step": int(probe["step"]),
                    "tag": probe["tag"],
                    "group": group_name,
                    "cosine": g.get("cosine"),
                    "norm_a": g.get("norm_a"),
                    "norm_b": g.get("norm_b"),
                    "dot": g.get("dot"),
                    "parameter_count": g.get("parameter_count"),
                    "absent_reason": g.get("absent_reason"),
                })
        for snap in run["snapshots"]:
            metrics = snap["metrics"]["metrics"]
            if metrics["all"]["examples"] != 128 or metrics["old"]["examples"] != 64 or metrics["new"]["examples"] != 64:
                raise ValueError("interference trajectory must stay on fixed 128-example subset")
            trajectory.append({
                "seed": seed,
                "step": int(snap["step"]),
                "tag": snap["tag"],
                "n_all": int(metrics["all"]["examples"]),
                "n_old": int(metrics["old"]["examples"]),
                "n_new": int(metrics["new"]["examples"]),
                "old_em": float(metrics["old"]["full_vocab_two_token_greedy_exact_match"]),
                "new_em": float(metrics["new"]["full_vocab_two_token_greedy_exact_match"]),
                "all_em": float(metrics["all"]["full_vocab_two_token_greedy_exact_match"]),
                "old_nll": float(metrics["old"]["class_nll"]),
                "new_nll": float(metrics["new"]["class_nll"]),
            })
        if not all(run["endpoint_match_v4_step16"].values()):
            raise ValueError(f"seed {seed}: terminal state does not exactly match v4 endpoint")
    if len(gradient_rows) != 36 or len(trajectory) != 21:
        raise ValueError("expected 12 gradient states x 3 groups and 21 trajectory snapshots")
    return gradient_rows, trajectory


def _sum_work(work_sources):
    per_source = {}
    by_stage = {}
    grand = {"forward_calls": 0, "forward_input_tokens": 0, "gradient_forward_input_tokens": 0}
    for name, work in work_sources.items():
        totals = {k: 0 for k in grand}
        for stage, counts in work.items():
            stage_total = by_stage.setdefault(stage, {k: 0 for k in grand})
            for field in grand:
                val = int(counts.get(field, 0))
                totals[field] += val
                stage_total[field] += val
                grand[field] += val
        per_source[name] = {"stages": len(work), "totals": totals}
    return {"per_source": per_source, "by_stage_sum_across_sources": by_stage, "grand_total": grand}


def build_work_ledger(components, component_failure, interference, interference_failure, qualification):
    sources = {
        "components_corrected_all3": components["work"],
        "components_failed_initial": component_failure["work"],
        "interference_corrected_all3": interference["work"],
        "interference_failed_initial": interference_failure["work"],
        "anchored_resume_qualification": qualification["work"],
    }
    work = _sum_work(sources)
    failed_grad_stages = {k: v for k, v in interference_failure["work"].items() if "_gradient" in k}
    derived_failed_gradient_calls = sum(int(v.get("forward_calls", 0)) for v in failed_grad_stages.values())
    reported_failed_gradient_calls = int(interference_failure.get("gradient_probe_forward_calls", 0))
    totals = {
        "component_corrected_seconds": float(components["seconds"]),
        "component_failed_seconds": None,
        "component_failed_seconds_status": "not recorded; intentionally unknown",
        "interference_corrected_seconds": float(interference["seconds"]),
        "interference_failed_seconds": float(interference_failure["seconds"]),
        "qualification_seconds": float(qualification["seconds"]),
        "known_elapsed_seconds_sum": sum(float(x) for x in (components["seconds"], interference["seconds"], interference_failure["seconds"], qualification["seconds"])),
        "completed_diagnostic_wake_updates": int(interference["updates"]) + int(qualification["SGD_updates"]),
        "completed_sleep_adam_updates": int(qualification["Sleep_Adam_updates"]),
        "corrected_gradient_probe_forward_calls": int(interference["gradient_probe_forward_calls"]),
        "failed_gradient_probe_reported_calls": reported_failed_gradient_calls,
        "failed_gradient_probe_derived_calls_from_work_stages": derived_failed_gradient_calls,
        "failed_gradient_probe_counter_mismatch": reported_failed_gradient_calls != derived_failed_gradient_calls,
        "all_phase_work_is_forward_token_proxy_not_flops_or_gpu_busy_time": True,
    }
    work["logical_cost_and_elapsed"] = totals
    return work


def _fmt_pct(v):
    return "—" if v is None else f"{100.0 * v:.2f}%"


def _fmt_mean_sd(s):
    if s["mean"] is None:
        return "—"
    return f"{100*s['mean']:.2f} ± {100*s['sample_sd']:.2f}"


def render_component_table(cases):
    lines = [
        "# Component probes",
        "",
        "Full validation denominator: old/new = 256/256 (512 total), across seeds 17/29/43. Values are mean ± sample SD in percentage points; per-seed values follow.",
        "",
        "| Case | Probe | Old EM mean ± SD | Old per seed (17/29/43) | New EM mean ± SD | New per seed (17/29/43) |",
        "|---|---|---:|---|---:|---|",
    ]
    for case in cases:
        for p in case["probes"]:
            old = ", ".join(_fmt_pct(x["old_em"]) for x in p["per_seed"])
            new = ", ".join(_fmt_pct(x["new_em"]) for x in p["per_seed"])
            lines.append(f"| {case['arm']}@{case['step']} | {p['probe']} | {_fmt_mean_sd(p['old_em'])} | {old} | {_fmt_mean_sd(p['new_em'])} | {new} |")
    lines += ["", "All entries are fixed probes, not candidate selection. Route forcing changes both selected expert and effective amplitude; it is not a pure router-only intervention.", ""]
    return "\n".join(lines)


def render_gradient_table(gradient_rows):
    lines = [
        "# Gradient diagnostics",
        "",
        "Gradients are diagnostic probes, not optimizer updates. The three parameter groups are shown separately; absent fast-expert rows are marked rather than treated as measured zero-cosine.",
        "",
        "| Seed | Step | Tag | Group | Cosine | Norm A | Norm B | Dot |",
        "|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in gradient_rows:
        cosine = "absent" if r["cosine"] is None else f"{r['cosine']:.5f}"
        norm_a = "—" if r["norm_a"] is None else f"{r['norm_a']:.5g}"
        norm_b = "—" if r["norm_b"] is None else f"{r['norm_b']:.5g}"
        dot = "—" if r["dot"] is None else f"{r['dot']:.5g}"
        lines.append(f"| {r['seed']} | {r['step']} | {r['tag']} | {r['group']} | {cosine} | {norm_a} | {norm_b} | {dot} |")
    lines += ["", "## 128-example trajectory subset", "", "These snapshots use 32 validation examples per class (old/new = 64/64); they are not pooled with the 512-example component probes.", "", "| Seed | Step | Tag | Old EM | New EM | All EM | Old class NLL | New class NLL |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    return "\n".join(lines)


def render_route_table(route_rows):
    lines = [
        "# Route and gate diagnostics",
        "",
        "Positions and selected-slot counts are accumulated across three seeds. Gate sums are normalized by audited answer positions. Selected winners are under each row's current policy; forced-route counts are not natural top-1 routing.",
        "",
        "| Case | Probe | Bank | Group | Positions | Selected slots (sum) | Null gate / pos | Warm0 gate / pos | Latest gate / pos | Mix slot/fraction |",
        "|---|---|---:|---|---:|---|---:|---:|---:|---|",
    ]
    grouped = {}
    for r in route_rows:
        k = (r["case"], r["probe"], r["bank"], r["group"])
        g = grouped.setdefault(k, {"positions": 0, "selected": {}, "null": 0., "warm0": 0., "latest": 0., "mix": set()})
        g["positions"] += r["positions"]
        for slot, n in r["selected_slots"].items():
            g["selected"][slot] = g["selected"].get(slot, 0) + n
        g["null"] += r["null_gate_sum"]; g["warm0"] += r["warm0_gate_sum"]; g["latest"] += r["latest_gate_sum"]
        g["mix"].add((r["controlled_mix_slot"], r["controlled_mix_fraction"]))
    for (case, probe, bank, group), g in sorted(grouped.items()):
        n = max(g["positions"], 1)
        mix = ",".join(f"{s}/{f}" for s, f in sorted(g["mix"], key=repr))
        selected = json.dumps(g["selected"], sort_keys=True, separators=(",", ":"))
        lines.append(f"| {case} | {probe} | {bank} | {group} | {g['positions']} | `{selected}` | {g['null']/n:.4f} | {g['warm0']/n:.4f} | {g['latest']/n:.4f} | {mix} |")
    lines += ["", "Forced slot policies change the expert output amplitude as well as routing. Gate sums are not semantic-memory measurements and must not be interpreted as causal evidence by themselves.", ""]
    return "\n".join(lines)


def _write_new(path: Path, text: str):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _make_plots(out_dir: Path, cases, gradient_rows):
    import matplotlib.pyplot as plt

    selected = ("actual", "mix_warm_receiver", "restore_both_force_warm", "force_warm_receiver", "warm_function_control")
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.8), sharey=True)
    for ax, case in zip(axes, cases):
        probes = [p for p in case["probes"] if p["probe"] in selected]
        x = list(range(len(probes)))
        old = [100*p["old_em"]["mean"] for p in probes]
        new = [100*p["new_em"]["mean"] for p in probes]
        old_sd = [100*p["old_em"]["sample_sd"] for p in probes]
        new_sd = [100*p["new_em"]["sample_sd"] for p in probes]
        ax.bar([i-.18 for i in x], old, width=.36, label="Old",
               yerr=old_sd, capsize=2, error_kw={"linewidth": .8})
        ax.bar([i+.18 for i in x], new, width=.36, label="New",
               yerr=new_sd, capsize=2, error_kw={"linewidth": .8})
        ax.set_title(f"{case['arm']}\nstep {case['step']}", fontsize=8)
        ax.set_xticks(x, [p["probe"].replace("_", "\n") for p in probes], rotation=0, fontsize=6)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Full-vocabulary EM (%; mean ± sample SD)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Fixed component-policy probes: mean ± sample SD, n=3 seeds; same 512-example validation set")
    fig.tight_layout()
    fig.savefig(out_dir / "component_probes.png", dpi=160)
    fig.savefig(out_dir / "component_probes.svg")
    plt.close(fig)

    early = [x for x in gradient_rows if x["step"] == 1 and x["group"] in ("base0", "fast_active_expert")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = .34
    positions = list(range(3))
    for offset, group, label in ((-.17, "base0", "Fast base0"), (.17, "fast_active_expert", "Fast expert")):
        vals = []
        for seed in SEEDS:
            row = next((x for x in early if x["seed"] == seed and x["group"] == group), None)
            vals.append(float("nan") if row is None or row["cosine"] is None else row["cosine"])
        ax.bar([p+offset for p in positions], vals, width=width, label=label)
    ax.axhline(0., color="black", linewidth=.8)
    ax.set_xticks(positions, [str(s) for s in SEEDS])
    ax.set_ylabel("Gradient cosine at step 1")
    ax.set_title("Early old/new gradient alignment by seed (step 1; base0 vs fast expert)")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(out_dir / "initial_gradient_cosines.png", dpi=160)
    fig.savefig(out_dir / "initial_gradient_cosines.svg")
    plt.close(fig)


def summarize(components_path, interference_path, component_failure_path, interference_failure_path, qualification_path, out_dir, plots=False):
    paths = {
        "components_corrected": Path(components_path),
        "interference_corrected": Path(interference_path),
        "components_failed": Path(component_failure_path),
        "interference_failed": Path(interference_failure_path),
        "qualification": Path(qualification_path),
    }
    data = {k: read_json(p) for k, p in paths.items()}
    comp, warm = build_component_summary(data["components_corrected"])
    routes = build_route_rows(data["components_corrected"])
    gradients, trajectory = build_interference(data["interference_corrected"])
    work = build_work_ledger(
        data["components_corrected"], data["components_failed"], data["interference_corrected"],
        data["interference_failed"], data["qualification"],
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    analysis = {
        "schema": "agnews-mechanism-v5-offline-analysis",
        "development_only": True,
        "confirmation_scored_in_this_summary": False,
        "inputs": {k: {"file": p.name, "sha256": file_sha256(p)} for k, p in paths.items()},
        "component_validation_scope": {"examples_all": 512, "old": 256, "new": 256, "per_class": 128, "cases": comp},
        "warm_reference_separate_from_component_probes": warm,
        "route_rows": routes,
        "interference_validation_scope": {"examples_all": 128, "old": 64, "new": 64, "per_class": 32, "gradient_rows": gradients, "trajectory_rows": trajectory},
        "resume_qualification": {
            "passed": bool(data["qualification"]["passed"]),
            "exact_model_controller_policy_rng": bool(data["qualification"]["exact_model_controller_policy_RNG"]),
            "SGD_updates": int(data["qualification"]["SGD_updates"]),
            "Sleep_Adam_updates": int(data["qualification"]["Sleep_Adam_updates"]),
            "scoring": bool(data["qualification"]["scoring"]),
        },
        "attempt_history": {
            "components_initial_failed": {
                "error_type": data["components_failed"].get("type"),
                "partial_rows": len(data["components_failed"].get("results", [])),
                "optimizer_updates": 0,
                "wall_seconds": None,
                "wall_seconds_note": "not recorded; intentionally unknown",
            },
            "interference_initial_failed": {
                "error_type": data["interference_failed"].get("type"),
                "attempted_wake_updates": int(data["interference_failed"].get("attempted_wake_updates", 0)),
                "reported_gradient_probe_forward_calls": int(data["interference_failed"].get("gradient_probe_forward_calls", 0)),
                "derived_gradient_probe_forward_calls_from_work_stages": work["logical_cost_and_elapsed"]["failed_gradient_probe_derived_calls_from_work_stages"],
                "counter_mismatch_documented": work["logical_cost_and_elapsed"]["failed_gradient_probe_counter_mismatch"],
                "wall_seconds": float(data["interference_failed"]["seconds"]),
            },
            "components_corrected": {"scored_rows": len(data["components_corrected"]["results"]), "states_including_warm_refs": int(data["components_corrected"]["total_scored_states"]), "optimizer_updates": int(data["components_corrected"]["optimizer_updates"]), "seconds": float(data["components_corrected"]["seconds"])},
            "interference_corrected": {"updates": int(data["interference_corrected"]["updates"]), "gradient_probe_forward_calls": int(data["interference_corrected"]["gradient_probe_forward_calls"]), "all_seeds_match_v4_endpoint": all(all(r["endpoint_match_v4_step16"].values()) for r in data["interference_corrected"]["results"]), "seconds": float(data["interference_corrected"]["seconds"])},
            "qualification": {"passed": bool(data["qualification"]["passed"]), "SGD_updates": int(data["qualification"]["SGD_updates"]), "Sleep_Adam_updates": int(data["qualification"]["Sleep_Adam_updates"]), "seconds": float(data["qualification"]["seconds"])},
        },
        "work": work,
        "interpretation_limits": [
            "Component probes use 512 validation examples; interference snapshots use a fixed 128-example subset; their means are not pooled.",
            "Route overrides affect expert amplitude and routing together; forced-slot selected counts are under the intervention policy, not natural routing.",
            "All metrics are exploratory development evidence; no confirmation set was scored here.",
            "Forward-token totals are proxies, not exact FLOPs, GPU busy time, or energy.",
        ],
    }
    _write_new(out_dir / "analysis.json", json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_new(out_dir / "component-table.md", render_component_table(comp))
    gradmd = render_gradient_table(gradients)
    for r in trajectory:
        gradmd += f"| {r['seed']} | {r['step']} | {r['tag']} | {100*r['old_em']:.2f}% | {100*r['new_em']:.2f}% | {100*r['all_em']:.2f}% | {r['old_nll']:.5f} | {r['new_nll']:.5f} |\n"
    _write_new(out_dir / "grad-table.md", gradmd)
    _write_new(out_dir / "route-table.md", render_route_table(routes))
    _write_new(out_dir / "work-ledger.json", json.dumps(work, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if plots:
        _make_plots(out_dir, comp, gradients)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--interference", type=Path, required=True)
    parser.add_argument("--component-failure", type=Path, required=True)
    parser.add_argument("--interference-failure", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    a = parser.parse_args()
    result = summarize(a.components, a.interference, a.component_failure, a.interference_failure, a.qualification, a.out_dir, a.plots)
    print(json.dumps({"schema": result["schema"], "out_dir": str(a.out_dir), "component_cases": len(result["component_validation_scope"]["cases"]), "gradient_rows": len(result["interference_validation_scope"]["gradient_rows"]), "holdout_scored": False}, sort_keys=True))


if __name__ == "__main__":
    main()
