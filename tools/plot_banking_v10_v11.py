"""Plot only validated aggregate Banking results; no model or data-row access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

SEEDS = (17, 29, 43)
V10_ARMS = ("CMS_ONLY", "SLEEP_ADDITIVE", "GOLD_ADDITIVE")
V11_ARMS = (*V10_ARMS, "LTI_NO_REWARD", "LTI_CANONICAL_TEACHER")
LABELS = ("CMS", "Sleep", "Gold\nreplay", "No\nreward", "Canonical\nteacher")
COLORS = ("#3478a3", "#df8b35", "#4e936a", "#8b78b2", "#c47694")


def selected_points(rows, *, version, population, subgroup, arms):
    selected = [r for r in rows if (r["version"], r["population"], r["subgroup"], r["metric"]) ==
                (version, population, subgroup, "teacher_forced_sum")]
    points = {}
    denominators = set()
    for arm in arms:
        records = [r for r in selected if r["arm"] == arm]
        if len(records) != 3 or {r["seed"] for r in records} != set(SEEDS):
            raise ValueError("Plot requires exactly three matched seeds for every arm")
        by_seed = {r["seed"]: r for r in records}
        values = [float(by_seed[s]["value"]) * 100 for s in SEEDS]
        if not all(math.isfinite(v) and 0 <= v <= 100 for v in values):
            raise ValueError("Accuracy must be finite and between zero and one")
        denominators.update(r["n"] for r in records)
        points[arm] = {"per_seed_percent": dict(zip(map(str, SEEDS), values)),
                       "mean_percent": statistics.mean(values),
                       "sample_sd_percentage_points": statistics.stdev(values)}
    if len(denominators) != 1:
        raise ValueError("Plot panel mixes populations or denominators")
    return points, denominators.pop()


def panel(ax, points, *, arms, title, n):
    means = [points[a]["mean_percent"] for a in arms]
    sds = [points[a]["sample_sd_percentage_points"] for a in arms]
    xs = list(range(len(arms)))
    ax.bar(xs, means, yerr=sds, capsize=4, width=.66,
           color=COLORS[:len(arms)], alpha=.84,
           error_kw={"linewidth": 1.15, "ecolor": "#263340"}, zorder=2)
    for index, (seed, marker) in enumerate(zip(SEEDS, ("o", "s", "^"))):
        positions = [x + (index - 1) * .13 for x in xs]
        ys = [points[a]["per_seed_percent"][str(seed)] for a in arms]
        ax.scatter(positions, ys, s=24, marker=marker, color="#17232e",
                   edgecolors="white", linewidths=.45, zorder=4, label=f"seed {seed}")
    ax.set_xticks(xs, LABELS[:len(arms)])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title}\n{n} examples per seed", fontsize=11)
    lower = min([0., *[m - s - 2 for m, s in zip(means, sds)]])
    upper = max([100., *[m + s + 2 for m, s in zip(means, sds)]])
    ax.set_ylim(lower, upper)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="#dce1e6", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    if summary.get("schema") != "banking-v10-v11-public-reanalysis-v1":
        raise ValueError("A validated public reanalysis summary is required")
    out = args.output_dir.resolve()
    if out.exists() or not out.parent.is_dir():
        raise ValueError("Plot output must be a fresh child of an existing directory")
    rows = summary["rows"]
    specs = [
        ("banking-v10-official-accuracy", "V10: local 3B controls after all seven tasks",
         "v10", V10_ARMS,
         (("official_test_raw", "all", "Raw official test"),
          ("official_test_train_development_disjoint", "all", "Train/development-disjoint test"))),
    ]
    if "v11" in summary["summaries"]:
        specs.append(("banking-v11-development-accuracy", "V11: task-1 reward ablation (development only)",
                      "v11", V11_ARMS,
                      (("development_task01", "old", "Old task"),
                       ("development_task01", "current", "New task"),
                       ("development_task01", "all77", "Both tasks"))))
    prepared = []
    for stem, title, version, arms, populations in specs:
        panels = []
        for pop, group, label in populations:
            points, n = selected_points(rows, version=version, population=pop, subgroup=group, arms=arms)
            panels.append((points, n, label))
        prepared.append((stem, title, arms, panels))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "svg.hashsalt": "banking-v10-v11",
                         "savefig.facecolor": "white"})
    out.mkdir()
    manifest = {"schema": "banking-v10-v11-aggregate-plot-manifest-v1",
                "input_sha256": hashlib.sha256(raw).hexdigest(),
                "matplotlib_version": matplotlib.__version__, "panels": {}, "artifacts": {},
                "interval": "sample SD across three paired order/data seeds, not a confidence interval",
                "shared_initialization_seed": 1907008}
    for stem, title, arms, panels in prepared:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.8), squeeze=False)
        for ax, (points, n, label) in zip(axes[0], panels):
            panel(ax, points, arms=arms, title=label, n=n)
            manifest["panels"][f"{stem}/{label}"] = {"examples_per_seed": n, "arms": points}
        handles, legend_labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, ncol=3, loc="lower center", bbox_to_anchor=(.5, .045), frameon=False)
        fig.suptitle(title, fontsize=14, y=.97)
        fig.text(.5, .015, "Global 77-label sum score incl. EOS; bars: mean; whiskers: sample SD; one shared initialization.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, .9))
        for extension in ("png", "svg"):
            path = out / f"{stem}.{extension}"
            extra = {"metadata": {"Date": None}} if extension == "svg" else {}
            fig.savefig(path, dpi=200, bbox_inches="tight", **extra)
            manifest["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        plt.close(fig)
    if args.summary.read_bytes() != raw:
        raise RuntimeError("Input summary changed during plot rendering")
    (out / "plot-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "figures": len(prepared), "output_dir": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
