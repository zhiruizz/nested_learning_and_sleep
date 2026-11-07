"""Plot sanitized P2 development aggregates; never loads model or row payloads."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("fresh figure directory required")
    raw = args.aggregate.read_bytes()
    data = json.loads(raw)
    if data["schema"] != "p2-formal-sanitized-science-aggregate-v1":
        raise ValueError("unexpected aggregate schema")
    runs = data["endpoint_and_probe_metrics"]
    assert len(runs) == 4
    assert {(r["arm"], r["seed"]) for r in runs} == {
        (arm, seed) for arm in ("All+N-repeat", "FastNew-SlowOld-k4") for seed in (402, 403)}
    for run in runs:
        assert [p["step"] for p in run["probes"]] == list(range(1, 41))
        for role, expected in (("old", 298), ("new", 300)):
            point = run["endpoint"][role]
            assert point["n"] == expected
            assert abs(point["accuracy"] - point["correct"] / expected) < 1e-12
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "p2-descriptive-20261108"})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    colors = ["#7c7c7c", "#cc7a22", "#208063", "#7561a7"]
    labels = [f"{'All+N' if r['arm'] == 'All+N-repeat' else 'k4'} / {r['seed']}" for r in runs]
    for ax, role in zip(axes[:2], ("old", "new")):
        for run, color, label in zip(runs, colors, labels):
            ax.plot(range(1, 41), [100 * p[role]["first_token_eos_rate"] for p in run["probes"]],
                    color=color, label=label, linewidth=1.7, alpha=0.92)
        ax.set(xlim=(1, 40), ylim=(-3, 103), xlabel="Logical update",
               ylabel="First-token EOS (% of 16 probes)", title=f"{role.capitalize()} development probes")
        ax.grid(axis="y", alpha=0.18)
    x = np.arange(4)
    for role, offset, hatch in (("old", -0.18, ""), ("new", 0.18, "//")):
        values = [100 * r["endpoint"][role]["accuracy"] for r in runs]
        bars = axes[2].bar(x + offset, values, width=0.34, color=colors, hatch=hatch,
                           edgecolor="white", label=role)
        for bar, value in zip(bars, values):
            axes[2].text(bar.get_x() + bar.get_width() / 2, max(0, value) + 2,
                         f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axes[2].set(xticks=x, xticklabels=[s.replace(" / ", "\n") for s in labels], ylim=(0, 100),
                ylabel="Exact match (%)", title="Development endpoints\nOld n=298; new n=300")
    axes[2].grid(axis="y", alpha=0.18)
    axes[2].legend(frameon=False, loc="upper right")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("P2: seed-sensitive collapse — original protocol not satisfied", fontsize=13, y=0.98)
    fig.text(0.5, 0.015, "Two seeds; descriptive only. Missing All+N write-time readback and development endpoint exposure are disclosed.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    args.output_dir.mkdir(parents=False)
    files = []
    for suffix in ("png", "svg"):
        path = args.output_dir / ("ubuntu-p2-development." + suffix)
        fig.savefig(path, dpi=180, metadata={"Date": None} if suffix == "svg" else None)
        files.append({"path": path.name, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    manifest = {"schema": "p2-descriptive-plot-manifest-v1", "aggregate_sha256": hashlib.sha256(raw).hexdigest(),
                "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "files": files}
    (args.output_dir / "plot-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
