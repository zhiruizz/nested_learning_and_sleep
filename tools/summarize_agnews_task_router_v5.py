"""Aggregate-only reanalysis of the fixed two-branch task-router control."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

EM = "full_vocab_two_token_greedy_exact_match"
CONDITIONS = ("warm_alone", "new_CMS_reference_alone", "learned_feature_gate", "task_ID_diagnostic")
GROUPS = ("all", "old", "new", "class0", "class1", "class2", "class3")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(values):
    return {"n": len(values), "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}

def task_gate(train, val, phase):
    classes = (0, 1) if phase == "old" else (2, 3)
    return train[phase][EM] >= .80 and val[phase][EM] >= .75 and all(val[f"class{k}"][EM] >= .70 for k in classes)

def analyze(result, plan):
    assert result["complete"] is True and result["actual_forward_calls"] == 1152
    assert result["actual_optimizer_updates"] == 0 and result["actual_gradient_forward_calls"] == 0
    assert result["confirmation_scored"] is False
    assert [r["seed"] for r in result["seeds"]] == [17, 29, 43]
    assert plan["training_group_label_uses"] == 3072
    assert plan["conditions"] == list(CONDITIONS)
    assert sum(v["forward_calls"] for v in result["work"].values()) == 1152
    assert all(v["gradient_forward_input_tokens"] == 0 for v in result["work"].values())
    summary = {}
    gates = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        gates[condition] = []
        for split, counts in (("train", (1024, 512, 256)), ("validation", (512, 256, 128))):
            source = [r["conditions"][split][condition] for r in result["seeds"]]
            for row in source:
                assert row["all"]["examples"] == counts[0]
                assert row["old"]["examples"] == row["new"]["examples"] == counts[1]
                assert all(row[f"class{k}"]["examples"] == counts[2] for k in range(4))
            summary[condition][split] = {g: {m: stats([r[g][m] for r in source])
                for m in source[0][g] if m != "examples"} for g in GROUPS}
        for row in result["seeds"]:
            train, val = (row["conditions"][split][condition] for split in ("train", "validation"))
            old, new = task_gate(train, val, "old"), task_gate(train, val, "new")
            gates[condition].append({"seed": row["seed"], "old": old, "new": new, "both": old and new})
    confusions = []
    for row in result["seeds"]:
        assert row["checkpoint_metric_match"] == {"warm_step0": True, "new_CMS_step448": True}
        assert row["gate"]["learned_selector_uses_validation_labels"] is False
        for split, total in (("training", 1024), ("validation", 512)):
            matrix = row["gate"][f"{split}_confusion"]
            assert sum(matrix.values()) == total
        matrix = row["gate"]["validation_confusion"]
        confusions.append({"seed": row["seed"], "group_accuracy":
            (matrix["actual_old_pred_old"] + matrix["actual_new_pred_new"]) / 512,
            "validation_confusion": matrix})
    return {"schema": "agnews-task-router-analysis-v5", "seeds": [17,29,43],
        "condition_mean_sample_sd": summary, "task_gates": gates,
        "learned_gate_validation": confusions,
        "work": result["work"], "wall_seconds": result["elapsed_seconds"],
        "actual_forward_calls": 1152, "LLM_optimizer_updates": 0, "CPU_ridge_fits": 3,
        "unique_training_group_labels": 1024, "training_group_label_uses": 3072,
        "confirmation_scored": False, "independent_confirmation": False,
        "inference_cost": "Both branches scored every row; this is not single-forward deployed inference",
        "claim_scope": "retained-context capacity/routing development diagnostic, not Sleep benefit"}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    plan_path = args.result.with_name("plan.json")
    result = json.loads(args.result.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert result["plan_sha256"] == sha(plan_path)
    assert result["source_sha256"] == plan["driver_sha256"]
    driver = Path(__file__).with_name("probe_agnews_task_router_v5.py")
    if driver.exists():
        assert sha(driver) == result["source_sha256"]
    analysis = analyze(result, plan)
    analysis["result_sha256"], analysis["plan_sha256"] = sha(args.result), sha(plan_path)
    args.out_dir.mkdir(exist_ok=False, parents=True)
    (args.out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# Two-branch routing/capacity control", "",
        "Exploratory n=3; fixed train-only ridge. Additional group labels and retained adapter state; not Sleep consolidation. Both branches were evaluated for every article. Task-ID selection uses evaluation group identity only as a diagnostic and is not a guaranteed upper bound.", "",
        "| Condition | Old EM % mean ± SD | New EM % mean ± SD | All EM % mean ± SD | Both gates |",
        "|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        metrics = analysis["condition_mean_sample_sd"][condition]["validation"]
        cells = [f"{100*metrics[g][EM]['mean']:.2f} ± {100*metrics[g][EM]['sample_sd']:.2f}" for g in ("old", "new", "all")]
        both = sum(r["both"] for r in analysis["task_gates"][condition])
        lines.append(f"| {condition} | {' | '.join(cells)} | {both}/3 |")
    lines.extend(["", f"1,152 full-model batch forwards; zero LLM optimizer updates; three CPU ridge fits; wall seconds {analysis['wall_seconds']:.3f}. 1,024 unique training group labels, reused in three fits (3,072 label uses).", "",
        "| Seed | Learned gate validation group accuracy | Old→old | Old→new | New→old | New→new |", "|---|---:|---:|---:|---:|---:|"])
    for row in analysis["learned_gate_validation"]:
        m = row["validation_confusion"]
        lines.append(f"| {row['seed']} | {100*row['group_accuracy']:.2f}% | {m['actual_old_pred_old']} | {m['actual_old_pred_new']} | {m['actual_new_pred_old']} | {m['actual_new_pred_new']} |")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"runs": 3, "conditions": 4, "output": str(args.out_dir)}))

if __name__ == "__main__":
    main()
