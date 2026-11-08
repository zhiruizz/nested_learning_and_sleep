import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.plot_banking_optimizer_boundary_v12 import (
    CONTEXTS,
    EXPECTED_SEEDS,
    POPS,
    create_plots,
    PlotError,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path):
    config_path = root / "frozen-config.json"
    analyzer_path = root / "reanalyzer.py"
    analyzer_path.write_text("# synthetic test source\n", encoding="utf-8")
    config = {
        "schema": "banking-optimizer-boundary-v12-config-v1",
        "status": "frozen",
        "source_hashes": {"tools/example.py": "a" * 64},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    csv_path = root / "metrics.csv"
    raw_rows = []
    summaries = {}
    for ci, context in enumerate(CONTEXTS):
        for metric, populations in (("teacher_forced_sum", POPS),
                                    ("teacher_forced_length_mean", POPS),
                                    ("gold_label_token_mean_nll", POPS),
                                    ("greedy", ("all77",))):
            for pop in populations:
                denom = {"old": 110, "current": 110, "all77": 220}[pop]
                value_field = "mean_token_nll" if metric == "gold_label_token_mean_nll" else "accuracy"
                vals = []
                for si, seed in enumerate(EXPECTED_SEEDS):
                    row = {"seed": seed, "context": context, "metric": metric,
                           "population": pop, "count": denom}
                    if value_field == "accuracy":
                        correct = 30 + si + ci if denom == 110 else 65 + si + ci
                        row.update(correct=correct, accuracy=correct / denom, invalid_predictions=0)
                        vals.append(row["accuracy"])
                    else:
                        row["mean_token_nll"] = 1.0 + si * 0.1 + ci * 0.01
                        vals.append(row["mean_token_nll"])
                    raw_rows.append(row)
                mean = sum(vals) / len(vals)
                sd = (sum((v - mean) ** 2 for v in vals) / 2) ** 0.5
                summaries[f"{context}|{metric}|{pop}"] = {
                    value_field: {"n": 3, "mean": mean, "sample_sd": sd}
                }

    fields = ["seed", "context", "metric", "population", "count", "correct",
              "invalid_predictions", "accuracy", "mean_token_nll"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(raw_rows)

    hashes = "b" * 64
    aggregate = {
        "schema": "banking-v12-optimizer-boundary-public-aggregate-v1",
        "status": "complete_descriptive_development_analysis",
        "design": {"seeds": list(EXPECTED_SEEDS), "contexts": list(CONTEXTS),
                   "candidate_labels": 77, "no_selection": True},
        "provenance": {
            "config_sha256": _sha(config_path), "analysis_tool_sha256": _sha(analyzer_path),
            "source_hashes": config["source_hashes"],
            "runs": [{"seed": seed, **{key: hashes for key in (
                "result_sha256", "protocol_sha256", "optimizer_batch_log_sha256",
                "boundary_proposal_sha256", "foundation_receipt_sha256",
                "foundation_checkpoint_sha256")}} for seed in EXPECTED_SEEDS],
        },
        "statistics": {"metrics_by_context_family_population": summaries},
        "official_test_read": False,
        "raw_rows_or_ids_emitted": False,
    }
    aggregate_path = root / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    return aggregate_path, csv_path, config_path, analyzer_path


def test_synthetic_aggregate_renders_two_formats_and_fresh_manifest(tmp_path):
    aggregate, metrics, config, analyzer = _fixture(tmp_path)
    out = tmp_path / "plots"
    result = create_plots(aggregate, metrics, config, analyzer, out)
    assert len(result["plots"]) == 4
    assert all((out / name).is_file() for name in (
        "banking-v12-boundary-primary-accuracy.png",
        "banking-v12-boundary-primary-accuracy.svg",
        "banking-v12-boundary-greedy-delta.png",
        "banking-v12-boundary-greedy-delta.svg",
        "plot-manifest.json",
    ))
    with pytest.raises(PlotError, match="fresh"):
        create_plots(aggregate, metrics, config, analyzer, out)
