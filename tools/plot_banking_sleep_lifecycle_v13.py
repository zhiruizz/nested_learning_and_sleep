#!/usr/bin/env python3
"""Plot completed aggregate-only V13 lifecycle metrics and exposure/update costs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


OUTPUT_SCHEMA = "banking-v13-batched-lifecycle-aggregate-v1"
SEEDS = (17, 29, 43)
ARMS = ("CMS_ONLY", "B1_HIGH", "B1_LOW", "B32_HIGH", "B32_LOW", "B32_LOW_DOSE")
STAGES = ("event1_before", "event1_after", "event2_before", "event2_after", "postwake15")
POPS = ("old", "current", "all77")
PRIMARY = "teacher_forced_sum"
ARM_COLORS = {"CMS_ONLY": "#666666", "B1_HIGH": "#0072B2", "B1_LOW": "#56B4E9",
              "B32_HIGH": "#D55E00", "B32_LOW": "#E69F00", "B32_LOW_DOSE": "#009E73"}


class PlotError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlotError(f"cannot read JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise PlotError(f"expected JSON object: {path.name}")
    return value


def _read_csv(path: Path) -> dict[tuple[int, str, str, str, str], dict[str, Any]]:
    required = {"seed", "arm", "stage", "metric", "population", "count"}
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise PlotError("V13 CSV schema mismatch")
        for line, raw in enumerate(reader, 2):
            try:
                key = (int(raw["seed"]), raw["arm"], raw["stage"], raw["metric"], raw["population"])
                count = int(raw["count"])
            except (ValueError, TypeError) as exc:
                raise PlotError(f"invalid CSV row {line}") from exc
            if key in rows or key[0] not in SEEDS or key[1] not in ARMS or key[2] not in STAGES or count < 1:
                raise PlotError(f"duplicate or invalid CSV row {line}")
            row = {"count": count}
            for field in ("correct", "invalid_predictions"):
                if raw.get(field, ""):
                    row[field] = int(raw[field])
            for field in ("accuracy", "mean_token_nll"):
                if raw.get(field, ""):
                    row[field] = float(raw[field])
                    if not math.isfinite(row[field]):
                        raise PlotError(f"nonfinite {field} at CSV row {line}")
            rows[key] = row
    expected = set()
    for seed in SEEDS:
        for arm in ARMS:
            for stage in STAGES:
                for metric in ("teacher_forced_sum", "teacher_forced_length_mean", "gold_label_token_mean_nll"):
                    expected.update((seed, arm, stage, metric, pop) for pop in POPS)
                expected.add((seed, arm, stage, "greedy", "all77"))
    if set(rows) != expected:
        raise PlotError(f"V13 metric grid mismatch: expected {len(expected)} rows, got {len(rows)}")
    for seed, arm, stage, metric, pop in expected:
        denom = {"old": 110, "current": 110, "all77": 220}[pop]
        if rows[(seed, arm, stage, metric, pop)]["count"] != denom:
            raise PlotError(f"wrong denominator for {seed}/{arm}/{stage}/{metric}/{pop}")
    return rows


def validate_inputs(aggregate_path: Path, csv_path: Path) -> tuple[dict[str, Any], dict[tuple[int, str, str, str, str], dict[str, Any]]]:
    aggregate = _json(aggregate_path)
    if (aggregate.get("schema") != OUTPUT_SCHEMA or aggregate.get("status") != "complete_descriptive_development_analysis"
            or aggregate.get("test_data_read") is not False or aggregate.get("raw_rows_or_ids_emitted") is not False):
        raise PlotError("aggregate not complete or outside development-only contract")
    design = aggregate.get("design", {})
    if (design.get("seeds") != list(SEEDS) or design.get("arms") != list(ARMS)
            or design.get("stages") != list(STAGES) or design.get("populations") != list(POPS)
            or design.get("candidate_labels") != 77):
        raise PlotError("aggregate design grid mismatch")
    rows = _read_csv(csv_path)
    summary = aggregate.get("metrics_by_arm_stage_family_population")
    if not isinstance(summary, dict):
        raise PlotError("aggregate lacks mean/sample-SD statistics")
    for arm in ARMS:
        for stage in STAGES:
            for metric in ("teacher_forced_sum", "teacher_forced_length_mean", "gold_label_token_mean_nll", "greedy"):
                pops = POPS if metric != "greedy" else ("all77",)
                for pop in pops:
                    field = "mean_token_nll" if metric == "gold_label_token_mean_nll" else "accuracy"
                    key = f"{arm}|{stage}|{metric}|{pop}"
                    entry = summary.get(key, {}).get(field)
                    if not isinstance(entry, dict) or entry.get("n") != 3:
                        raise PlotError(f"missing JSON summary {key}")
                    values = [rows[(seed, arm, stage, metric, pop)][field] for seed in SEEDS]
                    mean, sd = statistics.mean(values), statistics.stdev(values)
                    if not (math.isclose(float(entry.get("mean")), mean, rel_tol=1e-9, abs_tol=1e-9)
                            and math.isclose(float(entry.get("sample_sd")), sd, rel_tol=1e-9, abs_tol=1e-9)):
                        raise PlotError(f"CSV does not reproduce JSON summary {key}")
    return aggregate, rows


def _primary(rows: dict, population: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.7), sharex=True, sharey=True)
    x = list(range(len(STAGES)))
    for ax, pop in zip(axes, POPS):
        for arm in ARMS:
            per_stage = [[100 * rows[(seed, arm, stage, PRIMARY, pop)]["accuracy"] for seed in SEEDS]
                         for stage in STAGES]
            means = [statistics.mean(v) for v in per_stage]
            sds = [statistics.stdev(v) for v in per_stage]
            ax.errorbar(x, means, yerr=sds, color=ARM_COLORS[arm], marker="o", markersize=4,
                        linewidth=1.5, capsize=2.5, label=arm, zorder=3)
            for ix, vals in enumerate(per_stage):
                for seed, value in zip(SEEDS, vals):
                    jitter = {17: -0.055, 29: 0, 43: 0.055}[seed]
                    ax.scatter(ix + jitter, value, s=18, color=ARM_COLORS[arm], marker={17:"o",29:"s",43:"^"}[seed],
                               edgecolor="white", linewidth=0.3, alpha=0.78, zorder=4)
        ax.axvline(0.5, color="#697586", linestyle=":", linewidth=1)
        ax.axvline(2.5, color="#697586", linestyle=":", linewidth=1)
        ax.set_xticks(x, ["Event 1\nbefore", "Event 1\nafter", "Event 2\nbefore", "Event 2\nafter", "After\n15 wake ticks"])
        ax.set_title({"old":"Old (110 rows)", "current":"Current (110 rows)", "all77":"All (220 rows / 77 labels)"}[pop])
        ax.set_ylabel("Teacher-forced sum exact match (%)")
        ax.axhline(0, color="#555", linewidth=.5, linestyle="--")
        ax.axhline(100, color="#555", linewidth=.5, linestyle="--")
        ax.grid(axis="y", color="#d5d9df", linewidth=.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(.5, 1.00))
    fig.suptitle("V13 local batched Sleep lifecycle - development only", y=1.08, fontsize=14, weight="bold")
    fig.text(.5, .015,
        "Three class-order seeds share one initialization; paired arms share their seed-specific foundation. Means +/- sample SD (not CI). "
        "Events are natural boundaries; postwake15 is a fixed stage. Request exposure is not FLOPs: the B1/B32 arms differ in physical updates. "
        "CMS_ONLY is a reference; B32_LOW_DOSE changes both requests and updates. Local reward proxy; not a paper reproduction.",
        ha="center", va="bottom", fontsize=8.1, wrap=True)
    fig.subplots_adjust(top=.82, bottom=.23, left=.06, right=.99, wspace=.16)
    fig.savefig(out.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _costs(aggregate: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for arm in ARMS:
        entries = [x for x in aggregate["costs"] if x["arm"] == arm]
        for seed, entry in zip(SEEDS, sorted(entries, key=lambda x: x["seed"])):
            ax.scatter(entry["requests_recorded"], entry["physical_adamw_steps_recorded"],
                       color=ARM_COLORS[arm], marker={17:"o",29:"s",43:"^"}[seed], s=65,
                       edgecolor="white", linewidth=.5, label=f"{arm} · seed {seed}")
    ax.set_xlabel("Selected replay requests across two events (opportunities)")
    ax.set_ylabel("Committed physical AdamW updates across two events")
    ax.set_title("Recorded exposure and optimizer updates - not FLOPs")
    ax.grid(color="#d5d9df", linewidth=.55)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.text(.5, .01, "Counts distinguish selected requests from committed updates; zero-gradient skips may separate them. No FLOP equivalence is implied.",
             ha="center", fontsize=8)
    fig.subplots_adjust(bottom=.18, right=.99, top=.90)
    fig.savefig(out.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def create_plots(aggregate_path: Path, csv_path: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise PlotError("plot output directory must be fresh")
    aggregate, rows = validate_inputs(aggregate_path, csv_path)
    import matplotlib  # noqa: F401
    out_dir.mkdir(parents=True, exist_ok=False)
    primary = out_dir / "banking-v13-lifecycle-primary"
    costs = out_dir / "banking-v13-lifecycle-costs"
    _primary(rows, "all77", primary)
    _costs(aggregate, costs)
    outputs = []
    for path in (primary.with_suffix(".png"), primary.with_suffix(".svg"), costs.with_suffix(".png"), costs.with_suffix(".svg")):
        outputs.append({"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {"schema": "banking-v13-lifecycle-plot-manifest-v1", "status": "descriptive_development_plots",
        "inputs": {"aggregate_json": {"file": aggregate_path.name, "sha256": sha256_file(aggregate_path)},
                   "metrics_csv": {"file": csv_path.name, "sha256": sha256_file(csv_path)}},
        "provenance": aggregate["provenance"], "plots": outputs,
        "notes": {"n":3,"error_bars":"mean ± sample SD, not confidence intervals",
                  "official_test_read":False,"scope":"two local natural boundaries and postwake stage; local proxy; not paper reproduction"}}
    manifest_path = out_dir / "plot-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    return {"plot_manifest_sha256": sha256_file(manifest_path), "plots": outputs}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(create_plots(args.aggregate, args.metrics_csv, args.out_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
