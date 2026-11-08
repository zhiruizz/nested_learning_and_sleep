#!/usr/bin/env python3
"""Plot a completed, aggregate-only V12 optimizer-boundary analysis.

This reader accepts the reanalyzer's JSON plus its numeric metrics CSV. It does
not read run receipts, checkpoints, model weights, rollout tensors, or examples.
The JSON currently binds the analyzer/config/source graph but does not contain a
CSV digest; the manifest records both artifact hashes and cross-checks the CSV
seed summaries against the JSON's published means and sample standard
deviations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "banking-v12-optimizer-boundary-public-aggregate-v1"
EXPECTED_SEEDS = (17, 29, 43)
CONTEXTS = (
    "TEACHER", "CMS_NO_RESET", "RESET_ZERO", "B1_HIGH_30", "B1_HIGH_960",
    "B1_LOW_30", "B1_LOW_960", "B32_HIGH_960", "B32_LOW_960", "B32_LOW_3200",
)
# Group the plotted exposure contrasts while retaining the complete source grid.
PLOT_CONTEXTS = (
    "TEACHER", "CMS_NO_RESET", "RESET_ZERO", "B1_HIGH_30", "B1_LOW_30",
    "B1_HIGH_960", "B1_LOW_960", "B32_HIGH_960", "B32_LOW_960", "B32_LOW_3200",
)
POPS = ("old", "current", "all77")
PRIMARY = "teacher_forced_sum"

LABELS = (
    "Teacher\nreference", "CMS\nno reset", "Reset\nzero",
    "B1 high\n30 req / 30 Adam", "B1 low\n30 req / 30 Adam",
    "B1 high\n960 req / 960 Adam", "B1 low\n960 req / 960 Adam",
    "B32 high\n960 req / 30 Adam", "B32 low\n960 req / 30 Adam",
    "B32 low\n3200 req / 100 Adam",
)


class PlotError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlotError(f"cannot read JSON artifact {path.name}") from exc
    if not isinstance(value, dict):
        raise PlotError(f"{path.name} must contain an object")
    return value


def _hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise PlotError(f"invalid SHA-256 field: {name}")
    return value.lower()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PlotError(f"invalid finite number: {name}")
    return float(value)


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PlotError(f"invalid count: {name}")
    return value


def _read_metric_csv(path: Path) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    required = {"seed", "context", "metric", "population", "count"}
    rows: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise PlotError("metrics CSV is missing required aggregate columns")
            for line, raw in enumerate(reader, 2):
                try:
                    seed = int(raw["seed"])
                    context, metric, pop = raw["context"], raw["metric"], raw["population"]
                    count = int(raw["count"])
                except (TypeError, ValueError) as exc:
                    raise PlotError(f"invalid metrics CSV row {line}") from exc
                if seed not in EXPECTED_SEEDS or context not in CONTEXTS or pop not in POPS:
                    raise PlotError(f"unexpected seed/context/population at CSV row {line}")
                key = (seed, context, metric, pop)
                if key in rows or count < 1:
                    raise PlotError(f"duplicate key or invalid denominator at CSV row {line}")
                row: dict[str, Any] = {"count": count}
                for field in ("accuracy", "mean_token_nll"):
                    raw_value = raw.get(field, "")
                    if raw_value not in (None, ""):
                        try:
                            row[field] = _finite(float(raw_value), f"CSV row {line}.{field}")
                        except ValueError as exc:
                            raise PlotError(f"invalid {field} in CSV row {line}") from exc
                for field in ("correct", "invalid_predictions"):
                    raw_value = raw.get(field, "")
                    if raw_value not in (None, ""):
                        try:
                            row[field] = _count(int(raw_value), f"CSV row {line}.{field}")
                        except ValueError as exc:
                            raise PlotError(f"invalid {field} in CSV row {line}") from exc
                if "accuracy" in row:
                    if "correct" not in row or not math.isclose(row["accuracy"], row["correct"] / count, abs_tol=1e-12):
                        raise PlotError(f"accuracy/count mismatch at CSV row {line}")
                rows[key] = row
    except OSError as exc:
        raise PlotError(f"cannot read metrics CSV {path.name}") from exc
    expected = set()
    for seed in EXPECTED_SEEDS:
        for context in CONTEXTS:
            for metric, pops in (("teacher_forced_sum", POPS), ("teacher_forced_length_mean", POPS),
                                 ("gold_label_token_mean_nll", POPS), ("greedy", ("all77",))):
                expected.update((seed, context, metric, pop) for pop in pops)
    if set(rows) != expected:
        raise PlotError(f"metrics CSV grid mismatch: expected {len(expected)} rows, found {len(rows)}")
    for seed in EXPECTED_SEEDS:
        for ctx in CONTEXTS:
            for pop, denom in zip(POPS, (110, 110, 220)):
                for metric in ("teacher_forced_sum", "teacher_forced_length_mean", "gold_label_token_mean_nll"):
                    if rows[(seed, ctx, metric, pop)]["count"] != denom:
                        raise PlotError(f"wrong {metric}/{pop} denominator for seed {seed}, {ctx}")
            g = rows[(seed, ctx, "greedy", "all77")]
            if g["count"] != 220:
                raise PlotError(f"wrong greedy denominator for seed {seed}, {ctx}")
    return rows


def validate_inputs(aggregate_path: Path, metrics_path: Path, config_path: Path,
                    analyzer_path: Path) -> tuple[dict[str, Any], dict[tuple[int, str, str, str], dict[str, Any]]]:
    aggregate = _json(aggregate_path)
    config = _json(config_path)
    if aggregate.get("schema") != EXPECTED_SCHEMA or aggregate.get("status") != "complete_descriptive_development_analysis":
        raise PlotError("input is not a completed V12 descriptive aggregate")
    if aggregate.get("official_test_read") is not False or aggregate.get("raw_rows_or_ids_emitted") is not False:
        raise PlotError("aggregate does not affirm the expected data boundary")
    design = aggregate.get("design", {})
    if (design.get("seeds") != list(EXPECTED_SEEDS) or design.get("contexts") != list(CONTEXTS)
            or design.get("candidate_labels") != 77 or design.get("no_selection") is not True):
        raise PlotError("aggregate design does not match the fixed V12 grid")
    provenance = aggregate.get("provenance", {})
    if config.get("schema") != "banking-optimizer-boundary-v12-config-v1" or config.get("status") != "frozen":
        raise PlotError("configuration is not the frozen V12 config")
    if sha256_file(config_path) != _hex64(provenance.get("config_sha256"), "provenance.config_sha256"):
        raise PlotError("config file SHA does not match aggregate provenance")
    actual_analyzer_sha = sha256_file(analyzer_path)
    if actual_analyzer_sha != _hex64(provenance.get("analysis_tool_sha256"), "provenance.analysis_tool_sha256"):
        raise PlotError("analyzer source SHA does not match aggregate provenance")
    sources = provenance.get("source_hashes")
    if not isinstance(sources, dict) or sources != config.get("source_hashes"):
        raise PlotError("aggregate/config source hash maps differ")
    for name, digest in sources.items():
        _hex64(digest, f"source_hashes.{name}")
    runs = provenance.get("runs")
    if not isinstance(runs, list) or {r.get("seed") for r in runs if isinstance(r, dict)} != set(EXPECTED_SEEDS) or len(runs) != 3:
        raise PlotError("provenance does not contain exactly three run hashes")
    for row in runs:
        for key in ("result_sha256", "protocol_sha256", "optimizer_batch_log_sha256", "boundary_proposal_sha256",
                    "foundation_receipt_sha256", "foundation_checkpoint_sha256"):
            _hex64(row.get(key), f"run.{row.get('seed')}.{key}")
    stats = aggregate.get("statistics", {}).get("metrics_by_context_family_population")
    if not isinstance(stats, dict):
        raise PlotError("aggregate summary statistics are absent")
    rows = _read_metric_csv(metrics_path)
    # The analyzer emits per-seed CSV rows and JSON sample summaries. Cross-check
    # every exported mean and sample SD so the plotted CSV cannot silently drift.
    for context in CONTEXTS:
        for metric in ("teacher_forced_sum", "teacher_forced_length_mean", "gold_label_token_mean_nll", "greedy"):
            pops = POPS if metric != "greedy" else ("all77",)
            field = "mean_token_nll" if metric == "gold_label_token_mean_nll" else "accuracy"
            for pop in pops:
                key = f"{context}|{metric}|{pop}"
                summary = stats.get(key)
                if not isinstance(summary, dict) or field not in summary:
                    raise PlotError(f"aggregate lacks summary row {key}")
                vals = [rows[(seed, context, metric, pop)][field] for seed in EXPECTED_SEEDS]
                expected_mean = statistics.mean(vals)
                expected_sd = statistics.stdev(vals)
                if (not math.isclose(_finite(summary[field].get("mean"), f"{key}.mean"), expected_mean, rel_tol=1e-9, abs_tol=1e-9)
                        or not math.isclose(_finite(summary[field].get("sample_sd"), f"{key}.sample_sd"), expected_sd, rel_tol=1e-9, abs_tol=1e-9)
                        or summary[field].get("n") != 3):
                    raise PlotError(f"CSV seed data do not match JSON statistics for {key}")
    return aggregate, rows


def _limits(values: list[float], errors: list[float] | None = None) -> tuple[float, float]:
    lows = values[:]
    highs = values[:]
    if errors is not None:
        lows.extend(v - e for v, e in zip(values, errors))
        highs.extend(v + e for v, e in zip(values, errors))
    lo, hi = min(lows), max(highs)
    pad = max(2.0, (hi - lo) * 0.12)
    # Preserve raw SD endpoints even if they extend beyond the probability range;
    # 0 and 100 are still shown as reference lines.
    return min(-2.0, lo - pad), max(102.0, hi + pad)


def _draw_group_guides(ax: Any) -> None:
    ax.axvspan(-0.5, 2.5, color="#eef2f7", alpha=0.9, zorder=0)
    ax.axvspan(2.5, 4.5, color="#f8f2e8", alpha=0.58, zorder=0)
    ax.axvspan(4.5, 8.5, color="#eaf4ee", alpha=0.58, zorder=0)
    ax.axvspan(8.5, 9.5, color="#f4eaf0", alpha=0.62, zorder=0)
    ax.axvline(2.5, color="#607080", linewidth=1.1)
    ax.axvline(4.5, color="#607080", linewidth=1.1)
    ax.axvline(8.5, color="#607080", linewidth=1.1)
    ax.set_xticks(range(len(CONTEXTS)), LABELS, rotation=32, ha="right", fontsize=8)


def _draw_primary(rows: dict[tuple[int, str, str, str], dict[str, Any]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {17: "#0072B2", 29: "#D55E00", 43: "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.3), sharex=True)
    x = list(range(len(CONTEXTS)))
    for ax, pop in zip(axes, POPS):
        series = [[rows[(seed, ctx, PRIMARY, pop)]["accuracy"] * 100 for seed in EXPECTED_SEEDS]
                  for ctx in PLOT_CONTEXTS]
        means = [statistics.mean(v) for v in series]
        sds = [statistics.stdev(v) for v in series]
        _draw_group_guides(ax)
        ax.axhline(0, color="#39424e", linewidth=0.8, linestyle=":")
        ax.axhline(100, color="#39424e", linewidth=0.8, linestyle=":")
        ax.errorbar(x, means, yerr=sds, fmt="D", markersize=4.2, capsize=3, linewidth=1.5,
                    color="#20252b", ecolor="#4b5563", label="Mean ± sample SD", zorder=4)
        for idx, vals in enumerate(series):
            for seed, value in zip(EXPECTED_SEEDS, vals):
                jitter = {17: -0.11, 29: 0.0, 43: 0.11}[seed]
                ax.scatter(idx + jitter, value, s=28, color=colors[seed], edgecolor="white",
                           linewidth=0.45, label=f"Seed {seed}" if idx == 0 else None, zorder=5)
        endpoints = [v for values in series for v in values]
        ax.set_ylim(*_limits(endpoints, [sd for sd in sds for _ in EXPECTED_SEEDS]))
        ax.set_title({"old": "Old 110", "current": "Current 110", "all77": "All 220 · 77 labels"}[pop])
        ax.set_ylabel("Exact-match accuracy (%)")
        ax.grid(axis="y", color="#d7dce2", linewidth=0.6, alpha=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("V12 optimizer-boundary diagnostic · local partial comparison", y=1.08, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.015,
             "Task-1 development only · n=3 seeded class orders from one shared foundation lineage · one first-due boundary. "
             "Error bars are raw mean ± sample SD (not confidence intervals). Gray: reference contexts; "
             "tan: 30-request contexts; green: 960-request matched-exposure comparisons; mauve: 3200-request dose extension. "
             "Not a full Sleep lifecycle or paper-faithful reproduction.",
             ha="center", va="bottom", fontsize=8.1, wrap=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.31, wspace=0.19)
    fig.savefig(out.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _draw_greedy_delta(rows: dict[tuple[int, str, str, str], dict[str, Any]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {17: "#0072B2", 29: "#D55E00", 43: "#009E73"}
    paired = {
        ctx: [100 * (rows[(seed, ctx, "greedy", "all77")]["accuracy"] -
                    rows[(seed, ctx, PRIMARY, "all77")]["accuracy"])
              for seed in EXPECTED_SEEDS]
        for ctx in PLOT_CONTEXTS
    }
    means = [statistics.mean(paired[c]) for c in PLOT_CONTEXTS]
    sds = [statistics.stdev(paired[c]) for c in PLOT_CONTEXTS]
    fig, ax = plt.subplots(figsize=(13.8, 5.3))
    _draw_group_guides(ax)
    ax.axhline(0, color="#39424e", linewidth=1)
    ax.errorbar(range(10), means, yerr=sds, fmt="D", markersize=4.2, capsize=3,
                color="#20252b", ecolor="#4b5563", label="Mean paired difference ± sample SD", zorder=4)
    for idx, ctx in enumerate(CONTEXTS):
        for seed, value in zip(EXPECTED_SEEDS, paired[ctx]):
            jitter = {17: -0.11, 29: 0.0, 43: 0.11}[seed]
            ax.scatter(idx + jitter, value, s=30, color=colors[seed], edgecolor="white", linewidth=0.45,
                       label=f"Seed {seed}" if idx == 0 else None, zorder=5)
    ax.set_ylim(*_limits([v for vals in paired.values() for v in vals], [sd for sd in sds for _ in EXPECTED_SEEDS]))
    ax.set_ylabel("Greedy − primary exact match (percentage points)")
    ax.set_title("Unconstrained greedy vs teacher-forced sum objective · all 220 development rows")
    ax.grid(axis="y", color="#d7dce2", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.text(0.5, 0.01,
             "Paired within seed and context; fixed-context aggregate metrics only (not streaming batch loss). "
             "Three seeded class orders share one foundation lineage; one boundary; development only. "
             "Descriptive local diagnostic, not a full lifecycle or paper reproduction.",
             ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.25)
    fig.savefig(out.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def create_plots(aggregate_path: Path, metrics_path: Path, config_path: Path,
                 analyzer_path: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise PlotError("plot output directory must be fresh")
    aggregate, rows = validate_inputs(aggregate_path, metrics_path, config_path, analyzer_path)
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise PlotError("Matplotlib is required for PNG/SVG output") from exc
    out_dir.mkdir(parents=True, exist_ok=False)
    p1 = out_dir / "banking-v12-boundary-primary-accuracy"
    p2 = out_dir / "banking-v12-boundary-greedy-delta"
    _draw_primary(rows, p1)
    _draw_greedy_delta(rows, p2)
    artifacts = []
    for path in (p1.with_suffix(".png"), p1.with_suffix(".svg"), p2.with_suffix(".png"), p2.with_suffix(".svg")):
        artifacts.append({"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {
        "schema": "banking-v12-boundary-plot-manifest-v1",
        "status": "descriptive_development_plots",
        "input_artifacts": {
            "aggregate_json": {"file": aggregate_path.name, "sha256": sha256_file(aggregate_path)},
            "metrics_csv": {"file": metrics_path.name, "sha256": sha256_file(metrics_path),
                             "note": "CSV SHA is recorded here; source analyzer currently does not embed it in the JSON."},
            "analyzer_source": {"file": analyzer_path.name, "sha256": sha256_file(analyzer_path)},
            "frozen_config": {"file": config_path.name, "sha256": sha256_file(config_path)},
        },
        "source_provenance": aggregate["provenance"],
        "plots": artifacts,
        "interpretation": {
            "n": 3, "error_bars": "mean plus/minus raw sample standard deviation; not confidence intervals",
            "population": "task-1 development, 110 old + 110 current, 77-label global candidate set",
            "scope": "one first-due boundary; shared foundation lineage; local partial diagnostic; not paper-faithful reproduction",
            "official_test_read": False,
        },
    }
    manifest_path = out_dir / "plot-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                             encoding="utf-8", newline="\n")
    return {"out_dir": str(out_dir), "plot_manifest_sha256": sha256_file(manifest_path), "plots": artifacts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True, help="reanalyzer JSON")
    parser.add_argument("--metrics-csv", type=Path, required=True, help="reanalyzer seed-level numeric CSV")
    parser.add_argument("--config", type=Path, required=True, help="frozen V12 config used by the analyzer")
    parser.add_argument("--analyzer-source", type=Path, required=True, help="exact source file whose SHA is in JSON provenance")
    parser.add_argument("--out-dir", type=Path, required=True, help="must not already exist")
    args = parser.parse_args(argv)
    result = create_plots(args.aggregate, args.metrics_csv, args.config, args.analyzer_source, args.out_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
