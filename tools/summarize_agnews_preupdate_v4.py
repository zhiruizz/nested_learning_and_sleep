"""Offline aggregate analysis for AGNews pre-update comparison v4.

Standard library only by default. Reads warm transfer-result.json,
development-sealed.json and adjacent protocol.json; never reads sealed prompts
or labels, model checkpoints, or article text. It does not score or train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
from collections import defaultdict

ARMS = (
    "foundation_CMS_reference", "warm_CMS_no_sleep", "warm_pre_sleep_P64",
    "warm_pre_sleep_P4", "warm_post_sleep_P4", "warm_extra_gold_replay_P4",
)
SEEDS = (17, 29, 43)
STEPS = (0, 16, 32, 64, 112)
EM = "full_vocab_two_token_greedy_exact_match"
NLL = "answer_nll_nats_per_token"
GROUPS = ("old", "new", "all", "class0", "class1", "class2", "class3")
CONTRASTS = (
    ("pre4 - post4: teacher order, same event/slot/Adam budgets", "warm_pre_sleep_P4", "warm_post_sleep_P4"),
    ("pre4 - warm_no_sleep: algorithm bundle with extra compute/capacity", "warm_pre_sleep_P4", "warm_CMS_no_sleep"),
    ("pre4 - pre64: cadence/compute/capacity bundle", "warm_pre_sleep_P4", "warm_pre_sleep_P64"),
    ("warm_no_sleep - foundation_CMS_reference: total boundary transfer package", "warm_CMS_no_sleep", "foundation_CMS_reference"),
    ("warm_gold_replay - warm_no_sleep: added gold labels/SGD/data exposure", "warm_extra_gold_replay_P4", "warm_CMS_no_sleep"),
)

def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))

def mean_sd(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)}

def metric(m, group, key):
    return m[group][key]

def aggregate_runs(seal):
    runs = seal["runs"]
    assert seal.get("complete") is True
    assert seal.get("confirmation_scored") is False
    assert seal.get("no_holdout_input") is True
    assert len(runs) == 18, f"expected 18 runs, got {len(runs)}"
    by_key = {}
    for r in runs:
        key = (int(r["seed"]), r["arm"])
        assert key not in by_key
        assert r["status"] == "complete" and r["primary_updates"] == 112
        assert r["arm"] in ARMS and key[0] in SEEDS
        assert [int(c["primary_step"]) for c in r["validation"]] == list(STEPS)
        assert [int(c["primary_step"]) for c in r["checkpoints"]] == list(STEPS)
        assert len(r["validation"]) == 5
        by_key[key] = r
    assert set(by_key) == {(s, a) for s in SEEDS for a in ARMS}
    assert sum(len(r["validation"]) for r in runs) == 90
    primary_updates = sum(int(r["primary_updates"]) for r in runs)
    assert primary_updates == 18 * 112 == 2016
    assert sum(int(r["extra_replay_updates"]) for r in runs) == 3 * 336
    assert sum(int(r["sleep_Adam_updates"]) for r in runs) == 4104
    for r in runs:
        expected_replay = 336 if r["arm"] == "warm_extra_gold_replay_P4" else 0
        assert r["extra_replay_updates"] == expected_replay
        expected_sleep = 672 if r["arm"] in ("warm_pre_sleep_P4", "warm_post_sleep_P4") else 24 if r["arm"] == "warm_pre_sleep_P64" else 0
        assert r["sleep_Adam_updates"] == expected_sleep
    return by_key

def warm_analysis(warm):
    assert warm.get("all3_qualified") is True and warm.get("phaseB_permitted") is True
    assert warm.get("confirmation_scored") is False
    stages = []
    for stage in warm["stages"]:
        records = []
        for r in stage["records"]:
            tr, va = r["train"], r["validation"]
            teacher_va = warm["teachers"][str(r["seed"])]["validation"]
            teacher_drops_pp = {g: 100 * (teacher_va[g][EM] - va[g][EM]) for g in ("old", "class0", "class1")}
            max_teacher_drop_pp = max(teacher_drops_pp.values())
            gates = {
                "old_train_ge_80pct": tr["old"][EM] >= .80,
                "old_validation_ge_75pct": va["old"][EM] >= .75,
                "class0_validation_ge_70pct": va["class0"][EM] >= .70,
                "class1_validation_ge_70pct": va["class1"][EM] >= .70,
                "teacher_relative_drop_old_le_5pp": teacher_drops_pp["old"] <= 5,
                "teacher_relative_drop_class0_le_5pp": teacher_drops_pp["class0"] <= 5,
                "teacher_relative_drop_class1_le_5pp": teacher_drops_pp["class1"] <= 5,
            }
            computed_qualified = all(gates.values())
            assert bool(r["qualified"]) == computed_qualified, f"warm record qualification mismatch seed={r['seed']} budget={r['budget']}"
            records.append({"seed": r["seed"], "qualified": r["qualified"], "budget": r["budget"],
                "gates": gates, "train_em": {g: tr[g][EM] for g in ("old", "class0", "class1")},
                "validation_em": {g: va[g][EM] for g in ("old", "class0", "class1")},
                "teacher_em_validation": {g: teacher_va[g][EM] for g in ("old", "class0", "class1")},
                "teacher_relative_drops_pp": teacher_drops_pp,
                "teacher_max_drop_old_and_each_class_pp": max_teacher_drop_pp,
                "declared_teacher_drop_bound_pp": r["teacher_max_drop_old_and_each_class_pp"],
                "receiver_disabled_validation_em": {g: r["receiver_disabled_validation"][g][EM] for g in ("old", "class0", "class1")}})
        computed_all_qualified = all(r["qualified"] for r in records)
        assert bool(stage["all_qualified"]) == computed_all_qualified
        stages.append({"budget": stage["budget"], "all_qualified": computed_all_qualified, "seeds": records})
    final_qualified_seeds = {int(r["seed"]) for r in warm["qualified_models"]}
    assert final_qualified_seeds == set(SEEDS)
    final_stage = next(s for s in stages if s["budget"] == warm["chosen_common_budget"])
    assert final_stage["all_qualified"]
    controls = {}
    for seed, c in warm["initial_controls"].items():
        controls[seed] = {}
        for name in ("reset_without_KD_train", "reset_without_KD_validation"):
            x = c[name]
            controls[seed][name] = {g: x[g][EM] for g in x if g in GROUPS}
        controls[seed]["teacher"] = {phase: {g: c["teacher"][phase][g][EM] for g in GROUPS if g in c["teacher"][phase]}
                                      for phase in ("train", "validation")}
    return {"chosen_common_budget": warm["chosen_common_budget"], "stages": stages,
            "initial_controls_em": controls,
            "qualified_models": [{"seed": r["seed"], "warm_updates": r["warm_updates"],
                "train_em": {g: r["train"][g][EM] for g in ("old", "class0", "class1")},
                "validation_em": {g: r["validation"][g][EM] for g in ("old", "class0", "class1")}}
                for r in warm["qualified_models"]]}

def checkpoint_rows(by_key):
    rows = []
    byseedarmstep = {}
    for (seed, arm), r in by_key.items():
        for v in r["validation"]:
            step = int(v["primary_step"]); m = v["metrics"]
            assert set(GROUPS).issubset(m)
            row = {"seed": seed, "arm": arm, "step": step,
                "metrics": {g: {"em": m[g][EM], "candidate_accuracy": m[g]["candidate_accuracy"],
                                 "answer_nll_nats_per_token": m[g][NLL],
                                 "marker_accuracy": m[g]["marker_accuracy"], "examples": m[g]["examples"]}
                             for g in GROUPS}}
            rows.append(row); byseedarmstep[(seed, arm, step)] = m
    rows.sort(key=lambda r: (r["seed"], ARMS.index(r["arm"]), r["step"]))
    assert len(rows) == 90
    return rows, byseedarmstep

def descriptive(byseedarmstep):
    endpoint = {}
    trajectory = {}
    for arm in ARMS:
        endpoint[arm] = {}
        trajectory[arm] = {}
        for seed in SEEDS:
            start = byseedarmstep[(seed, arm, 0)]
            end = byseedarmstep[(seed, arm, 112)]
            endpoint[arm][str(seed)] = {g: {"em": end[g][EM], "answer_nll_nats_per_token": end[g][NLL],
                                                "candidate_accuracy": end[g]["candidate_accuracy"]} for g in GROUPS}
            endpoint[arm][str(seed)]["old_drop_from_own_start"] = start["old"][EM] - end["old"][EM]
        endpoint[arm]["summary"] = {}
        for g in GROUPS:
            endpoint[arm]["summary"][g] = {k: mean_sd([endpoint[arm][str(s)][g][field] for s in SEEDS])
                for k, field in (("em", "em"), ("answer_nll_nats_per_token", "answer_nll_nats_per_token"), ("candidate_accuracy", "candidate_accuracy"))}
        endpoint[arm]["summary"]["old_drop_from_own_start"] = mean_sd([endpoint[arm][str(s)]["old_drop_from_own_start"] for s in SEEDS])
        for step in STEPS:
            trajectory[arm][str(step)] = {}
            for g in GROUPS:
                trajectory[arm][str(step)][g] = {k: mean_sd([byseedarmstep[(s, arm, step)][g][field] for s in SEEDS])
                    for k, field in (("em", EM), ("answer_nll_nats_per_token", NLL), ("candidate_accuracy", "candidate_accuracy"))}
    paired = []
    for label, left_arm, right_arm in CONTRASTS:
        measures = {}
        per_seed = []
        for seed in SEEDS:
            diffs = {"seed": seed}
            for g in GROUPS:
                a = byseedarmstep[(seed, left_arm, 112)][g]
                b = byseedarmstep[(seed, right_arm, 112)][g]
                diffs[g] = {"em_difference": a[EM] - b[EM], "answer_nll_difference": a[NLL] - b[NLL]}
            per_seed.append(diffs)
        for g in GROUPS:
            measures[g] = {
                "em_difference": mean_sd([x[g]["em_difference"] for x in per_seed]),
                "answer_nll_difference": mean_sd([x[g]["answer_nll_difference"] for x in per_seed]),
            }
        paired.append({"contrast": label, "left_arm_minus_right_arm": [left_arm, right_arm], "per_seed": per_seed, "summary": measures})
    return endpoint, trajectory, paired

def sum_work(work):
    fields = ("forward_calls", "forward_input_tokens", "gradient_forward_input_tokens")
    total = {k: 0 for k in fields}
    for counter in work.values():
        for k in fields: total[k] += int(counter.get(k, 0))
    return total

def work_ledger(warm, seal, by_key):
    warm_total = sum_work(warm.get("work", {}))
    warm_by_stage = defaultdict(lambda: {"forward_calls":0,"forward_input_tokens":0,"gradient_forward_input_tokens":0})
    for name, count in warm.get("work", {}).items():
        stage = name.split("_seed",1)[0]
        for k,v in sum_work({name:count}).items(): warm_by_stage[stage][k] += v
    phase_by_arm = {}
    for arm in ARMS:
        phase_by_arm[arm] = {"seeds": {}, "training": {"forward_calls":0,"forward_input_tokens":0,"gradient_forward_input_tokens":0},
                             "validation": {"forward_calls":0,"forward_input_tokens":0,"gradient_forward_input_tokens":0},
                             "total_including_validation": {"forward_calls":0,"forward_input_tokens":0,"gradient_forward_input_tokens":0}}
        for seed in SEEDS:
            work = by_key[(seed,arm)].get("work", {})
            sums = sum_work(work)
            phase_by_arm[arm]["seeds"][str(seed)] = sums
            for k,v in sums.items(): phase_by_arm[arm]["training"][k] += v
    for name, counter in seal.get("work", {}).items():
        matched = [arm for arm in ARMS if arm in name]
        if len(matched) == 1 and "validation" in name:
            arm = matched[0]
            sums = sum_work({name: counter})
            for k,v in sums.items(): phase_by_arm[arm]["validation"][k] += v
            marker = "validation_seed"
            seed = int(name.split(marker, 1)[1].split("_", 1)[0]) if marker in name else None
            if seed in SEEDS:
                for k,v in sums.items():
                    phase_by_arm[arm]["seeds"][str(seed)][k] += v
    for arm in ARMS:
        for k in phase_by_arm[arm]["total_including_validation"]:
            phase_by_arm[arm]["total_including_validation"][k] = phase_by_arm[arm]["training"][k] + phase_by_arm[arm]["validation"][k]
    phase_attributed = {k: sum(v["total_including_validation"][k] for v in phase_by_arm.values()) for k in ("forward_calls", "forward_input_tokens", "gradient_forward_input_tokens")}
    phase_seal_total = sum_work(seal.get("work", {}))
    phase_residue = {k: phase_seal_total[k]-phase_attributed[k] for k in phase_attributed}
    assert all(v == 0 for v in phase_residue.values()), f"Phase B work counters not fully attributed: {phase_residue}"
    expected_arm = {"primary_SGD_updates":2016,"total_SGD_updates":3024,"extra_gold_replay_SGD_updates":1008,"sleep_Adam_updates":4104}
    replay_count = sum(int(r["extra_replay_updates"]) for r in seal["runs"])
    assert replay_count == expected_arm["extra_gold_replay_SGD_updates"]
    primary_count = sum(int(r["primary_updates"]) for r in seal["runs"])
    assert primary_count == 2016
    assert primary_count + replay_count == expected_arm["total_SGD_updates"]
    assert sum(int(r["sleep_Adam_updates"]) for r in seal["runs"]) == expected_arm["sleep_Adam_updates"]
    shared_updates = int(warm["successful_or_attempted_main_Adam_updates"])
    ghost_updates = int(warm["extra_ghost_Adam_updates"])
    assert shared_updates == 4608 and ghost_updates == 2
    return {
        "warm_shared_once": {"main_Adam_updates": shared_updates, "restore_ghost_Adam_updates": ghost_updates,
                             "forward_counters": warm_total, "by_work_stage": dict(warm_by_stage)},
        "phaseB": {"primary_SGD_updates": expected_arm["primary_SGD_updates"], "total_SGD_updates":expected_arm["total_SGD_updates"],
                   "extra_gold_replay_SGD_updates": replay_count, "sleep_Adam_updates": expected_arm["sleep_Adam_updates"],
                   "run_attributed_forward_counters": phase_attributed,
                   "seal_total_including_validation": phase_seal_total,
                   "unattributed_counter_residue": phase_residue,
                   "by_arm": phase_by_arm},
        "method_path_per_seed": {arm:{str(seed):{
            "warm_tax_Adam_updates": 0 if arm == "foundation_CMS_reference" else 1536,
            "phaseB_primary_SGD_updates": int(by_key[(seed,arm)]["primary_updates"]),
            "phaseB_extra_gold_replay_SGD_updates": int(by_key[(seed,arm)]["extra_replay_updates"]),
            "phaseB_sleep_Adam_updates": int(by_key[(seed,arm)]["sleep_Adam_updates"]),
            "forward_counters": phase_by_arm[arm]["seeds"][str(seed)]} for seed in SEEDS} for arm in ARMS},
        "method_path_all3_seeds": {arm:{
            "warm_tax_Adam_updates": 0 if arm == "foundation_CMS_reference" else 4608,
            "phaseB_primary_SGD_updates": sum(int(r["primary_updates"]) for r in seal["runs"] if r["arm"]==arm),
            "phaseB_extra_gold_replay_SGD_updates": sum(int(r["extra_replay_updates"]) for r in seal["runs"] if r["arm"]==arm),
            "phaseB_sleep_Adam_updates": sum(int(r["sleep_Adam_updates"]) for r in seal["runs"] if r["arm"]==arm),
            "forward_counters": phase_by_arm[arm]["total_including_validation"]} for arm in ARMS},
        "method_path_note": "Warm tax is 1536 Adam updates per warm-start seed path, 4608 across all three seeds, and zero for foundation reference. Shared warm total is counted once at experiment level; Phase B counters are reported by arm and by seed.",
    }

def write_text_lf(path, text):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)

def write_tables(outdir, warm_info, endpoint, trajectory, paired):
    ep = ["# AGNews pre-update v4: fixed endpoint table", "", "Development/validation only. Fixed step 112; mean and sample SD across three seeds. No p-values or checkpoint selection.", "", "| Arm | Old EM | New EM | All EM | Old drop from own start | Old NLL | New NLL | All NLL |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        s = endpoint[arm]["summary"]
        fmt=lambda x:f"{100*x['mean']:.2f}% ± {100*x['sample_sd']:.2f} pp"
        nfmt=lambda x:f"{x['mean']:.4f} ± {x['sample_sd']:.4f}"
        ep.append(f"| {arm} | {fmt(s['old']['em'])} | {fmt(s['new']['em'])} | {fmt(s['all']['em'])} | {fmt(s['old_drop_from_own_start'])} | {nfmt(s['old']['answer_nll_nats_per_token'])} | {nfmt(s['new']['answer_nll_nats_per_token'])} | {nfmt(s['all']['answer_nll_nats_per_token'])} |")
    write_text_lf(outdir/"endpoint-table.md", "\n".join(ep)+"\n")
    tr=["# AGNews pre-update v4: trajectory table", "", "Mean ± sample SD of full-vocabulary EM across three seeds at fixed checkpoints.", "", "| Arm | Step | Old | New | All | Class0 | Class1 | Class2 | Class3 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for step in STEPS:
            s=trajectory[arm][str(step)]
            vals=[s[g]["em"] for g in GROUPS]
            tr.append("| {} | {} | {} |".format(arm,step," | ".join(f"{100*x['mean']:.2f}% ± {100*x['sample_sd']:.2f}" for x in vals)))
    write_text_lf(outdir/"trajectory-table.md", "\n".join(tr)+"\n")
    pt=["# AGNews pre-update v4: predefined paired endpoint contrasts", "", "Difference is left arm minus right arm at step 112. Mean ± sample SD across matched seeds; no p-values.", "", "| Contrast | Group | EM difference (pp) | Answer NLL difference |", "|---|---|---:|---:|"]
    for c in paired:
        for g in GROUPS:
            x=c["summary"][g]
            pt.append(f"| {c['contrast']} | {g} | {100*x['em_difference']['mean']:+.2f} ± {100*x['em_difference']['sample_sd']:.2f} | {x['answer_nll_difference']['mean']:+.4f} ± {x['answer_nll_difference']['sample_sd']:.4f} |")
    write_text_lf(outdir/"paired-table.md", "\n".join(pt)+"\n")
    warm_lines=["## Warm teacher transfer stages", "", f"Chosen common budget: {warm_info['chosen_common_budget']} Adam updates. Stage records retain per-seed pass/fail and validation gates.", "", "| Budget | Seed | Qualified | Train old | Val old | Val class0 | Val class1 | Max teacher drop (pp) |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for stage in warm_info["stages"]:
        for r in stage["seeds"]:
            warm_lines.append(f"| {stage['budget']} | {r['seed']} | {r['qualified']} | {100*r['train_em']['old']:.2f}% | {100*r['validation_em']['old']:.2f}% | {100*r['validation_em']['class0']:.2f}% | {100*r['validation_em']['class1']:.2f}% | {r['teacher_max_drop_old_and_each_class_pp']:.2f} |")
    write_text_lf(outdir/"warm-stages.md", "\n".join(warm_lines)+"\n")

def optional_plots(outdir, trajectory, endpoint):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(13,5.5),sharex=True)
    for ax, group, label in zip(axes,("old","new"),("Old-task","New-task")):
        for arm in ARMS:
            means=[trajectory[arm][str(s)][group]["em"]["mean"] for s in STEPS]
            ax.plot(STEPS,[100*x for x in means],marker="o",label=arm)
        ax.set(xlabel="Primary SGD updates",ylabel=f"{label} full-vocabulary EM (%)",title=f"{label} trajectory")
        ax.set_xticks(STEPS);ax.grid(alpha=.25)
    axes[1].legend(fontsize=7);fig.suptitle("Development trajectories, fixed checkpoints");fig.tight_layout()
    fig.savefig(outdir/"old-new-trajectories.png",dpi=160);fig.savefig(outdir/"old-new-trajectories.svg");plt.close(fig)
    fig, ax = plt.subplots(figsize=(11,6))
    x=list(range(len(ARMS)));width=.36
    old=[100*endpoint[a]["summary"]["old"]["em"]["mean"] for a in ARMS]
    old_sd=[100*endpoint[a]["summary"]["old"]["em"]["sample_sd"] for a in ARMS]
    new=[100*endpoint[a]["summary"]["new"]["em"]["mean"] for a in ARMS]
    new_sd=[100*endpoint[a]["summary"]["new"]["em"]["sample_sd"] for a in ARMS]
    ax.bar([i-width/2 for i in x],old,width,yerr=old_sd,capsize=3,label="Old EM")
    ax.bar([i+width/2 for i in x],new,width,yerr=new_sd,capsize=3,label="New EM")
    ax.set_xticks(x,ARMS,rotation=25,ha="right");ax.set_ylim(bottom=0)
    ax.set(ylabel="Full-vocabulary EM (%)",title="Fixed step 112 endpoint tradeoff (development only)")
    ax.grid(axis="y",alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(outdir/"endpoint-tradeoff.png",dpi=160);fig.savefig(outdir/"endpoint-tradeoff.svg");plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warm-result",type=pathlib.Path,required=True)
    p.add_argument("--development-seal",type=pathlib.Path,required=True)
    p.add_argument("--out-dir",type=pathlib.Path,required=True)
    p.add_argument("--plots",action="store_true",help="Optional Matplotlib PNG/SVG charts")
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    warm=read_json(a.warm_result);seal=read_json(a.development_seal)
    protocol_path=a.development_seal.with_name("protocol.json")
    protocol=read_json(protocol_path)
    assert protocol.get("confirmation_input") is False and protocol.get("confirmation_scored") is False
    assert protocol.get("exploratory_only") is True
    assert protocol["warm_result_sha256"]==sha_file(a.warm_result)
    assert seal["protocol_sha256"]==sha_file(protocol_path)
    assert seal["source_lock_sha256"]==protocol["source_lock_sha256"]
    assert seal["warm_result_sha256"]==sha_file(a.warm_result)
    lock=protocol["source_lock"]
    assert lock["paired_contrasts"]==[x[0] for x in CONTRASTS]
    assert lock["validation_adaptivity"]=="warm budget selected by train/validation gate; phaseB reports all arms/endpoints, never selects best checkpoint"
    assert lock["confirmation_scored"] is False
    assert lock["statistics"]=="all3 seeds, means/sampleSD and paired differences, no p-values or independent confirmation claim"
    by_key=aggregate_runs(seal)
    warm_info=warm_analysis(warm)
    rows,byseedarmstep=checkpoint_rows(by_key)
    endpoint,trajectory,paired=descriptive(byseedarmstep)
    ledger=work_ledger(warm,seal,by_key)
    analysis={
        "schema":"agnews-preupdate-analysis-v4",
        "input_hashes":{"warm_result_sha256":sha_file(a.warm_result),"development_seal_sha256":sha_file(a.development_seal),"protocol_sha256":sha_file(protocol_path)},
        "source_lock_sha256":protocol.get("source_lock_sha256"),
        "counts":{"runs":18,"evaluations":90,"arms":list(ARMS),"seeds":list(SEEDS),"steps":list(STEPS)},
        "warm_stages":warm_info,
        "checkpoint_rows":rows,
        "endpoint_summary":endpoint,
        "trajectory_summary":trajectory,
        "predefined_paired_contrasts":paired,
        "work_ledger":ledger,
        "interpretation_flags":{"development_only":True,"validation_adaptive_warm_budget":True,"confirmation_scored":False,"no_pvalues":True,"no_checkpoint_selection":True,"no_independent_confirmation_claim":True},
    }
    write_text_lf(a.out_dir/"analysis.json",json.dumps(analysis,indent=2,ensure_ascii=False)+"\n")
    write_tables(a.out_dir,warm_info,endpoint,trajectory,paired)
    if a.plots: optional_plots(a.out_dir,trajectory,endpoint)
    write_text_lf(a.out_dir/"work-ledger.json",json.dumps(ledger,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":"analysis_written","runs":18,"evaluations":90,"arm_count":6,"seed_count":3,
        "confirmation_scored":False,"output_files":[str(a.out_dir/x) for x in ("analysis.json","endpoint-table.md","trajectory-table.md","paired-table.md","warm-stages.md","work-ledger.json")]},indent=2))

if __name__=="__main__": main()
