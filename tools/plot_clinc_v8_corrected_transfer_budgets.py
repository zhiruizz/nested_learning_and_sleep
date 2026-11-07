#!/usr/bin/env python3
"""Plot corrected CLINC150 development endpoints from frozen JSON summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = ("old", "current", "seen")
COLORS = {
    "HOPE_REFERENCE": "#596579",
    "SLEEP_MIX": "#167d9a",
    "GOLD_CE_MIX": "#db8a24",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def collect_plot_rows(transfer: dict[str, Any], development: dict[str, Any]) -> list[dict[str, Any]]:
    if transfer.get("schema") != "clinc-v8-transfer-budget-analysis-v1":
        raise ValueError("Unexpected corrected transfer analysis schema")
    if development.get("schema") != "clinc-v8-two-task-development-summary-v1":
        raise ValueError("Unexpected development summary schema")
    if development.get("test_payload_read") is not False:
        raise ValueError("Refusing to plot a summary that reports test payload access")
    hope_rows = [row for row in development.get("rows", []) if row.get("arm") == "HOPE"]
    if len(hope_rows) != 1:
        raise ValueError("Expected exactly one HOPE row in the development summary")
    reference = hope_rows[0].get("metrics")
    if not isinstance(reference, dict):
        raise ValueError("Missing HOPE metrics in the development summary")
    embedded_reference = transfer.get("no_sleep_reference", {}).get("endpoint")
    if not isinstance(embedded_reference, dict):
        raise ValueError("Missing same-foundation HOPE reference")
    if any(reference.get(metric, {}).get("correct") != embedded_reference.get(metric, {}).get("correct")
           or reference.get(metric, {}).get("count") != embedded_reference.get(metric, {}).get("count")
           for metric in METRICS):
        raise ValueError("HOPE baseline differs between development summary and transfer analysis")
    rows: list[dict[str, Any]] = [{
        "key": "HOPE_REFERENCE",
        "label": "HOPE\n(no Sleep)",
        "arm": "HOPE_REFERENCE",
        "budget": None,
        "adam_steps": int(hope_rows[0]["physical_receiver_adamw_steps"]),
        "metrics": reference,
    }]
    for budget in ("k1", "k10"):
        budget_data = transfer.get("budgets", {}).get(budget)
        if not isinstance(budget_data, dict):
            raise ValueError(f"Missing budget {budget}")
        endpoints = budget_data.get("endpoint_clean_full_primary")
        costs = budget_data.get("cost", {}).get("per_arm")
        if not isinstance(endpoints, dict) or not isinstance(costs, dict):
            raise ValueError(f"Missing endpoint or arm cost data for {budget}")
        for arm in ("SLEEP_MIX", "GOLD_CE_MIX"):
            if arm not in endpoints or arm not in costs:
                raise ValueError(f"Missing {arm} endpoint or cost for {budget}")
            short = "Sleep" if arm == "SLEEP_MIX" else "Gold CE"
            rows.append({
                "key": f"{budget}_{arm}",
                "label": f"{short}\n{budget}",
                "arm": arm,
                "budget": budget,
                "adam_steps": int(costs[arm]["physical_sleep_adam_steps"]),
                "metrics": endpoints[arm],
            })
    for row in rows:
        for metric in METRICS:
            value = row["metrics"].get(metric)
            if not isinstance(value, dict) or not isinstance(value.get("correct"), int) or not isinstance(value.get("count"), int):
                raise ValueError(f"Invalid {metric} metric in {row['key']}")
            if value["count"] <= 0 or not 0 <= value["correct"] <= value["count"]:
                raise ValueError(f"Out-of-range {metric} metric in {row['key']}")
    return rows


def render(transfer_path: Path, development_path: Path, output_stem: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    transfer = _read_json(transfer_path)
    development = _read_json(development_path)
    rows = collect_plot_rows(transfer, development)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.titlesize": 15,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6), sharey=True)
    x = np.arange(len(rows))
    widths = 0.72
    for ax, metric in zip(axes, METRICS):
        counts = [row["metrics"][metric]["correct"] for row in rows]
        totals = [row["metrics"][metric]["count"] for row in rows]
        values = [100.0 * correct / total for correct, total in zip(counts, totals)]
        ax.bar(x, values, width=widths,
               color=[COLORS[row["arm"]] for row in rows],
               edgecolor="white", linewidth=0.7)
        for xi, value, correct, total, row in zip(x, values, counts, totals, rows):
            ax.text(xi, value + 1.1, f"{correct}/{total}\n{value:.1f}%",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title({"old": "Old-task development", "current": "Current-task development", "seen": "Combined seen tasks"}[metric])
        ax.set_xticks(x, [row["label"] for row in rows])
        ax.set_ylim(0, 88)
        ax.grid(axis="y", color="#d8dde4", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Exact-match accuracy (%)")
    fig.suptitle("Corrected transfer-budget results on the clean development set", y=0.97)
    fig.text(0.5, 0.02,
             "One class-order seed; HOPE shares the same foundation. Gold CE reuses labeled training targets. "
             "Receiver AdamW steps: " + " / ".join(f"{r['label'].replace(chr(10), ' ')}={r['adam_steps']}" for r in rows) +
             ". Development evidence only; not a multi-seed or test result.",
             ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.12, 1, 0.91), w_pad=1.2)

    svg_path = output_stem.with_suffix(".svg")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Title": "CLINC150 corrected transfer-budget development"})
    fig.savefig(png_path, dpi=180, bbox_inches="tight", metadata={"Title": "CLINC150 corrected transfer-budget development"})
    plt.close(fig)
    return svg_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, default=Path("outputs/clinc-v8-preparation/transfer-budget-analysis-v1.json"))
    parser.add_argument("--development", type=Path, default=Path("outputs/clinc-v8-preparation/development-summary-v1.json"))
    parser.add_argument("--output-stem", type=Path, default=Path("outputs/clinc-v8-figures/clinc-v8-corrected-transfer-budgets"))
    args = parser.parse_args()
    paths = render(args.transfer, args.development, args.output_stem)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
