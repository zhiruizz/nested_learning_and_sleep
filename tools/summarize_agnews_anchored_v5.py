"""Offline summary of the sealed AG News anchored-wake v5 development run.

Reads only JSON seal/protocol metadata. It never opens checkpoints, datasets,
training prompts, reserved labels, or model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SEEDS=(17,29,43)
ARMS=("warm_CMS_reference","warm_anchorKD","warm_anchorGold",
      "warm_anchorKD_sleepP16_latest","warm_anchorKD_sleepP16_warmmix",
      "warm_self_sleepP16_latest","warm_self_sleepP16_warmmix")
STEPS=(0,16,64,112,256,448)
SLEEP_ARMS=frozenset(("warm_anchorKD_sleepP16_latest","warm_anchorKD_sleepP16_warmmix",
                      "warm_self_sleepP16_latest","warm_self_sleepP16_warmmix"))
KD_ARMS=frozenset(("warm_anchorKD","warm_anchorKD_sleepP16_latest","warm_anchorKD_sleepP16_warmmix"))
METRICS=("full_vocab_two_token_greedy_exact_match","answer_nll_nats_per_token","marker_nll",
         "class_nll","marker_accuracy","class_given_marker_accuracy","candidate_accuracy")
GROUPS=("old","new","all","class0","class1","class2","class3")
CONTRASTS=(
    ("KD - reference","warm_anchorKD","warm_CMS_reference"),
    ("KD latest - self latest","warm_anchorKD_sleepP16_latest","warm_self_sleepP16_latest"),
    ("KD warm - self warm","warm_anchorKD_sleepP16_warmmix","warm_self_sleepP16_warmmix"),
    ("KD latest - KD","warm_anchorKD_sleepP16_latest","warm_anchorKD"),
    ("KD warm - KD","warm_anchorKD_sleepP16_warmmix","warm_anchorKD"),
    ("self latest - reference","warm_self_sleepP16_latest","warm_CMS_reference"),
    ("self warm - reference","warm_self_sleepP16_warmmix","warm_CMS_reference"),
    ("KD warm - KD latest","warm_anchorKD_sleepP16_warmmix","warm_anchorKD_sleepP16_latest"),
    ("self warm - self latest","warm_self_sleepP16_warmmix","warm_self_sleepP16_latest"),
    ("Gold - KD","warm_anchorGold","warm_anchorKD"),
    ("Gold - reference","warm_anchorGold","warm_CMS_reference"),
)
INTERACTIONS=(
    ("(KD warm - self warm) - (KD latest - self latest)",
     (("warm_anchorKD_sleepP16_warmmix","warm_self_sleepP16_warmmix"),
      ("warm_anchorKD_sleepP16_latest","warm_self_sleepP16_latest"))),
    ("(KD latest - self latest) - (KD - reference)",
     (("warm_anchorKD_sleepP16_latest","warm_self_sleepP16_latest"),
      ("warm_anchorKD","warm_CMS_reference"))),
)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def write_text_lf(path,text):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        f.write(text.rstrip("\n")+"\n")

def mean_sd(values):
    values=[float(x) for x in values]
    return {"n":len(values),"mean":statistics.mean(values) if values else None,
            "sample_sd":statistics.stdev(values) if len(values)>1 else (0.0 if values else None)}

def assert_counts(metrics,total,old,new,per_class):
    assert metrics["all"]["examples"]==total
    assert metrics["old"]["examples"]==old and metrics["new"]["examples"]==new
    assert sum(metrics[f"class{k}"]["examples"] for k in range(4))==total
    assert all(metrics[f"class{k}"]["examples"]==per_class for k in range(4))

def merge_work(runs):
    """Sum exact run.work keys; do not re-filter by arm-name prefixes."""
    out={}
    for run in runs:
        for stage,counters in run["work"].items():
            dest=out.setdefault(stage,{})
            for key,value in counters.items():
                if not isinstance(value,(int,float)) or isinstance(value,bool):
                    raise ValueError(f"nonnumeric WorkMeter counter: {stage}/{key}")
                dest[key]=dest.get(key,0)+value
    return out

def work_residual(run_union,global_work):
    residual={}
    if set(run_union)!=set(global_work):
        residual["__stage_keys__"]={"missing_from_run_union":sorted(set(global_work)-set(run_union)),
                                    "extra_in_run_union":sorted(set(run_union)-set(global_work))}
    for stage in sorted(set(run_union)|set(global_work)):
        a=run_union.get(stage,{})
        b=global_work.get(stage,{})
        if set(a)!=set(b):
            residual.setdefault(stage,{})["__counter_keys__"]={"missing_from_run_union":sorted(set(b)-set(a)),
                                                                  "extra_in_run_union":sorted(set(a)-set(b))}
        for key in sorted(set(a)|set(b)):
            delta=a.get(key,0)-b.get(key,0)
            if delta!=0:residual.setdefault(stage,{})[key]=delta
    return residual

def metric_values(metrics):
    return {g:{m:float(metrics[g][m]) for m in METRICS} for g in GROUPS}

def source_check(seal_path,seal):
    protocol_path=Path(seal_path).parent/"protocol.json"
    if not protocol_path.is_file():raise FileNotFoundError("adjacent protocol.json is required")
    protocol=read(protocol_path); protocol_hash=sha(protocol_path)
    if seal.get("protocol_sha256")!=protocol_hash:raise ValueError("seal/protocol SHA mismatch")
    if seal.get("source_sha256")!=protocol.get("source_sha256"):raise ValueError("seal/protocol source hashes differ")
    if seal.get("config_sha256")!=protocol.get("config_sha256"):raise ValueError("seal/protocol config hashes differ")
    cfg=protocol.get("config",{})
    if cfg.get("schema")!="agnews-anchored-comparison-config-v5":raise ValueError("unexpected embedded config schema")
    if cfg.get("review_status")!="FROZEN_DEVELOPMENT_AFTER_REVIEW":raise ValueError("config is not marked frozen")
    if cfg.get("arms")!=list(ARMS) or cfg.get("evaluation_steps")!=list(STEPS):raise ValueError("config arms/steps mismatch")
    if cfg.get("primary_steps")!=448 or cfg.get("capacity")!=32 or cfg.get("sleep_period")!=16:
        raise ValueError("config budget/capacity mismatch")
    if cfg.get("confirmation_input") is not False or cfg.get("confirmation_scored") is not False:
        raise ValueError("protocol permits confirmation input/scoring")
    expected_pairs=[[a,b] for _,a,b in CONTRASTS]
    if cfg.get("paired_contrasts_left_minus_right")!=expected_pairs:
        raise ValueError("frozen paired contrast arms/sign differ from analyzer")
    expected_interactions=["(KDwarm-selfwarm)-(KDlatest-selflatest)","(KDlatest-selflatest)-(KD-reference)"]
    if cfg.get("factorial_interactions")!=expected_interactions:
        raise ValueError("frozen interaction signs differ from analyzer")
    driver_hash=protocol.get("source_sha256",{}).get("driver")
    local_sources={"driver":"run_agnews_anchored_comparison_v5.py","common":"agnews_v3_common.py",
      "v4_boundary":"agnews_preupdate_sleep_v4.py","wake_helper":"agnews_anchored_wake_v5.py",
      "v4_engine":"run_agnews_retention_v2.py"}
    for key,filename in local_sources.items():
        expected=protocol.get("source_sha256",{}).get(key); path=Path(__file__).with_name(filename)
        if expected and path.is_file() and sha(path)!=expected:
            raise ValueError(f"available {key} source does not match protocol hash")
    candidates=[]
    for key in ("config_path","config_file"):
        if protocol.get(key):
            path=Path(protocol[key]); candidates.append(path if path.is_absolute() else protocol_path.parent/path)
    candidates.extend([protocol_path.parent/"config.json",protocol_path.parent/"agnews-anchored-comparison-v5.json"])
    candidates.extend(Path(__file__).resolve().parents[1].glob("configs/*anchored*comparison*v5*.json"))
    config_files=[p for p in candidates if p.is_file()]
    config_check="not_available"
    if config_files:
        config_check="matched" if any(sha(p)==seal["config_sha256"] for p in config_files) else "mismatch"
        if config_check!="matched":raise ValueError("available raw config file does not match sealed config hash")
    return {"protocol_filename":protocol_path.name,"protocol_sha256":protocol_hash,
            "config_sha256":seal["config_sha256"],"raw_config_check":config_check,
            "runner_sha256":driver_hash,"no_data_or_checkpoint_reads":True},protocol

def validate_runs(seal,protocol):
    if seal.get("schema")!="agnews-anchored-development-seal-v5" or seal.get("complete") is not True:
        raise ValueError("expected a complete anchored v5 development seal")
    if seal.get("confirmation_scored") is not False or seal.get("no_holdout_input") is not True:
        raise ValueError("sealed run reports confirmation/holdout access")
    runs=seal.get("runs",[])
    if len(runs)!=21:raise ValueError(f"expected 21 runs, found {len(runs)}")
    bykey={}
    for r in runs:
        key=(int(r["seed"]),r["arm"])
        if key in bykey or key[0] not in SEEDS or key[1] not in ARMS:raise ValueError(f"duplicate/unknown run {key}")
        if r.get("status")!="complete" or r.get("primary_updates")!=448:raise ValueError(f"incomplete run {key}")
        if [int(x["primary_step"]) for x in r["validation"]]!=list(STEPS):raise ValueError(f"validation steps mismatch {key}")
        if [int(x["primary_step"]) for x in r["checkpoints"]]!=list(STEPS):raise ValueError(f"checkpoint steps mismatch {key}")
        if len(r["validation"])!=6 or r.get("train_final") is None:raise ValueError(f"missing scores {key}")
        for point in r["validation"]:
            assert_counts(point["metrics"],512,256,256,128)
        assert_counts(r["train_final"],1024,512,512,256)
        gates=r.get("qualification",{}).get("gates",{})
        if set(gates)!={"old","new"}:raise ValueError(f"old/new task gates absent {key}")
        final_val=r["validation"][-1]["metrics"]
        for phase in ("old","new"):
            g=gates[phase]
            raw_train=float(r["train_final"][phase][METRICS[0]])
            raw_val=float(final_val[phase][METRICS[0]])
            if not math.isclose(float(g["train_phase_em"]),raw_train,rel_tol=0,abs_tol=1e-12):
                raise ValueError(f"train gate EM differs from raw metric {key}/{phase}")
            if not math.isclose(float(g["validation_phase_em"]),raw_val,rel_tol=0,abs_tol=1e-12):
                raise ValueError(f"validation gate EM differs from raw metric {key}/{phase}")
            train_pass=raw_train>=.80; val_pass=raw_val>=.75
            classes=(0,1) if phase=="old" else (2,3)
            class_em={str(k):float(final_val[f"class{k}"][METRICS[0]]) for k in classes}
            for k in classes:
                if not math.isclose(float(g["validation_class_em"][str(k)]),class_em[str(k)],rel_tol=0,abs_tol=1e-12):
                    raise ValueError(f"class gate EM differs from raw metric {key}/{phase}/class{k}")
            class_pass={str(k):class_em[str(k)]>=.70 for k in classes}
            if bool(g["train_phase_pass"])!=train_pass or bool(g["validation_phase_pass"])!=val_pass:
                raise ValueError(f"phase gate component mismatch {key}/{phase}")
            if g["validation_class_pass"]!=class_pass:raise ValueError(f"class gate detail mismatch {key}/{phase}")
            computed=(train_pass and val_pass and all(class_pass.values()))
            if bool(g["pass"])!=computed:raise ValueError(f"gate detail inconsistent {key}/{phase}")
        if bool(r["qualification"].get("all_tasks_pass"))!=(gates["old"]["pass"] and gates["new"]["pass"]):
            raise ValueError(f"collapsed all-task gate mismatch {key}")
        expected_events=28 if r["arm"] in SLEEP_ARMS else 0
        if len(r.get("events",[]))!=expected_events or r.get("sleep_Adam_updates")!=expected_events*24:
            raise ValueError(f"Sleep event/update count mismatch {key}")
        if r["arm"] in SLEEP_ARMS:
            slots=r.get("final_active_parameters",{}).get("active_slots_by_bank")
            if slots!=[0,29]:raise ValueError(f"Sleep receiver capacity mismatch {key}: {slots}")
        elif r.get("sleep_Adam_updates")!=0:raise ValueError(f"unexpected Sleep updates {key}")
        bykey[key]=r
    if set(bykey)!={(s,a) for s in SEEDS for a in ARMS}:raise ValueError("seed/arm grid incomplete")
    if sum(r["primary_updates"] for r in runs)!=9408 or seal.get("actual_primary_updates")!=9408:
        raise ValueError("primary update total mismatch")
    if sum(len(r["events"]) for r in runs)!=336 or seal.get("actual_sleep_Adam_updates")!=8064:
        raise ValueError("global Sleep event/update total mismatch")
    if sum(r["sleep_Adam_updates"] for r in runs)!=8064:raise ValueError("row Sleep update total mismatch")
    if seal.get("actual_kd_old_aux_steps")!=4032 or seal.get("actual_gold_old_aux_steps")!=1344:
        raise ValueError("old auxiliary target/update totals mismatch")
    if len(KD_ARMS)*len(SEEDS)*448!=4032 or 1*len(SEEDS)*448!=1344:
        raise ValueError("arm-derived old auxiliary schedule totals mismatch")
    if protocol.get("planned_primary_updates")!=9408 or protocol.get("planned_sleep_events")!=336 or protocol.get("planned_sleep_adam_updates")!=8064:
        raise ValueError("protocol budget mismatch")
    if protocol.get("planned_kd_old_aux_steps")!=4032 or protocol.get("planned_gold_old_aux_steps")!=1344:
        raise ValueError("protocol auxiliary budget mismatch")
    union=merge_work(runs); residual=work_residual(union,seal.get("work",{}))
    if residual:raise ValueError(f"run work counters do not sum to seal: {residual}")
    return bykey,union

def sample_stats(values_by_seed):
    return {"per_seed":{str(s):v for s,v in values_by_seed.items()},**mean_sd(list(values_by_seed.values()))}

def score_at(bykey,seed,arm,step,group,metric):
    run=bykey[(seed,arm)]
    point=next(x for x in run["validation"] if x["primary_step"]==step)
    return float(point["metrics"][group][metric])

def contrast_table(bykey,contrasts,steps):
    out=[]
    for label,a,b in contrasts:
        rows=[]
        for step in steps:
            groups={}
            for group in GROUPS:
                groups[group]={}
                for metric in METRICS:
                    vals={s:score_at(bykey,s,a,step,group,metric)-score_at(bykey,s,b,step,group,metric) for s in SEEDS}
                    groups[group][metric]=sample_stats(vals)
            rows.append({"step":step,"metrics":groups})
        out.append({"contrast":label,"minuend":a,"subtrahend":b,"paired_n":3,"trajectory":rows})
    return out

def interaction_table(bykey):
    out=[]
    for label,((a,b),(c,d)) in INTERACTIONS:
        rows=[]
        for step in STEPS:
            groups={}
            for group in GROUPS:
                groups[group]={}
                for metric in METRICS:
                    vals={s:(score_at(bykey,s,a,step,group,metric)-score_at(bykey,s,b,step,group,metric))-
                              (score_at(bykey,s,c,step,group,metric)-score_at(bykey,s,d,step,group,metric)) for s in SEEDS}
                    groups[group][metric]=sample_stats(vals)
            rows.append({"step":step,"metrics":groups})
        out.append({"interaction":label,"paired_n":3,"trajectory":rows})
    return out

def analyze(seal,protocol,provenance):
    bykey,work=validate_runs(seal,protocol); trajectory={}; endpoints={}; per_seed=[]; gate_summary={}; retention={}
    for arm in ARMS:
        trajectory[arm]={}; endpoints[arm]={}
        for step in STEPS:
            groups={}
            for group in GROUPS:
                groups[group]={}
                for metric in METRICS:
                    vals={s:score_at(bykey,s,arm,step,group,metric) for s in SEEDS}
                    groups[group][metric]=sample_stats(vals)
            trajectory[arm][str(step)]=groups
        endpoints[arm]=trajectory[arm][str(STEPS[-1])]
        deltas={s:{"initial_old_em":score_at(bykey,s,arm,0,"old",METRICS[0]),
                   "final_old_em":score_at(bykey,s,arm,448,"old",METRICS[0]),
                   "old_em_forgetting_step0_minus_step448":score_at(bykey,s,arm,0,"old",METRICS[0])-score_at(bykey,s,arm,448,"old",METRICS[0]),
                   "initial_new_em":score_at(bykey,s,arm,0,"new",METRICS[0]),
                   "final_new_em":score_at(bykey,s,arm,448,"new",METRICS[0]),
                   "new_em_change_step448_minus_step0":score_at(bykey,s,arm,448,"new",METRICS[0])-score_at(bykey,s,arm,0,"new",METRICS[0])} for s in SEEDS}
        retention[arm]={k:sample_stats({s:v[k] for s,v in deltas.items()}) for k in next(iter(deltas.values()))}
        gate_summary[arm]={"old_pass_seeds":0,"new_pass_seeds":0,"both_pass_seeds":0,"per_seed":{}}
        for s in SEEDS:
            r=bykey[(s,arm)]; gates=r["qualification"]["gates"]
            both=bool(gates["old"]["pass"] and gates["new"]["pass"])
            gate_summary[arm]["old_pass_seeds"]+=int(bool(gates["old"]["pass"]))
            gate_summary[arm]["new_pass_seeds"]+=int(bool(gates["new"]["pass"]))
            gate_summary[arm]["both_pass_seeds"]+=int(both)
            gate_summary[arm]["per_seed"][str(s)]={"old":gates["old"],"new":gates["new"],"both_pass":both}
            per_seed.append({"seed":s,"arm":arm,"trajectory":{str(x["primary_step"]):metric_values(x["metrics"]) for x in r["validation"]},
                "train_final":metric_values(r["train_final"]),"qualification":r["qualification"],
                "initial_old_em":deltas[s]["initial_old_em"],"final_old_em":deltas[s]["final_old_em"],
                "old_em_forgetting_step0_minus_step448":deltas[s]["old_em_forgetting_step0_minus_step448"],
                "initial_new_em":deltas[s]["initial_new_em"],"final_new_em":deltas[s]["final_new_em"],
                "new_em_change_step448_minus_step0":deltas[s]["new_em_change_step448_minus_step0"],
                "primary_updates":r["primary_updates"],"sleep_Adam_updates":r["sleep_Adam_updates"],
                "sleep_events":len(r["events"]),"seconds":r.get("seconds"),
                "final_active_slots_by_bank":r.get("final_active_parameters",{}).get("active_slots_by_bank"),
                "work":r["work"]})
    arm_cost={}
    for arm in ARMS:
        selected=[bykey[(s,arm)] for s in SEEDS]
        arm_cost[arm]={"runs":len(selected),"primary_updates":sum(r["primary_updates"] for r in selected),
            "sleep_events":sum(len(r["events"]) for r in selected),"sleep_Adam_updates":sum(r["sleep_Adam_updates"] for r in selected),
            "mean_seconds":mean_sd([r["seconds"] for r in selected if r.get("seconds") is not None]),
            "meter_work":merge_work(selected)}
    return {"schema":"agnews-anchored-analysis-v5","seal_sha256":provenance["seal_sha256"],
        "protocol":provenance,"seeds":list(SEEDS),"arms":list(ARMS),"evaluation_steps":list(STEPS),
        "validation_states":126,"validation_examples_per_state":{"all":512,"old":256,"new":256,"each_class":128},
        "per_seed_runs":per_seed,"trajectory_mean_sample_sd":trajectory,"endpoint_step448_mean_sample_sd":endpoints,
        "retention_and_acquisition_diagnostics":retention,"task_gates":gate_summary,
        "paired_contrasts":contrast_table(bykey,CONTRASTS,STEPS),
        "paired_endpoint_train_contrasts":train_contrast_table(bykey),"paired_interactions":interaction_table(bykey),
        "cost":{"primary_updates":9408,"sleep_events":336,"sleep_Adam_updates":8064,
          "kd_old_aux_steps":4032,"gold_old_aux_steps":1344,"sleep_arm_run_count":12,"events_per_sleep_run":28,
          "kd_old_prefix_rows":16128,"gold_old_prefix_rows":5376,"gold_old_answer_targets":5376,
          "capacity":32,"sleep_final_active_slots_by_bank":[0,29],"work_union":work,"global_work":seal["work"],
          "by_arm":arm_cost,
          "work_residual":{},"work_union_equals_seal":True,"external_costs_not_in_seal":
            "Keep the separate 48-update interference diagnostic, 18-update qualification, and any earlier failed-probe costs in the external report ledger.",
          "compute_interpretation":"WorkMeter forward calls/input tokens are proxies, not FLOPs or GPU busy time; soft-KD and Gold auxiliary paths have different computation.",
          "arm_semantics":"Old auxiliary samples are balanced using class metadata. Anchor KD targets frozen-teacher soft distributions rather than old gold answers, but class labels still determine balanced sampling; it is not wholly label-free. Anchor Gold uses labeled old gold answers. Comparisons bundle target and compute differences."},
        "inference_limits":{"n_seeds":3,"sample_sd_ddof":1,"paired_seed_differences":True,
            "p_values":False,"independence_claim":False,"endpoint_selection":False,
            "claim_scope":"exploratory train/validation development comparison"}}

def train_contrast_table(bykey):
    out=[]
    for label,a,b in CONTRASTS:
        groups={}
        for group in GROUPS:
            groups[group]={}
            for metric in METRICS:
                vals={s:float(bykey[(s,a)]["train_final"][group][metric])-float(bykey[(s,b)]["train_final"][group][metric]) for s in SEEDS}
                groups[group][metric]=sample_stats(vals)
        out.append({"contrast":label,"minuend":a,"subtrahend":b,"paired_n":3,"train_final_metrics":groups})
    return out

def plot_results(analysis,out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    labels={"warm_CMS_reference":"Reference","warm_anchorKD":"Anchor KD","warm_anchorGold":"Anchor Gold",
      "warm_anchorKD_sleepP16_latest":"KD-Sleep latest","warm_anchorKD_sleepP16_warmmix":"KD-Sleep warm",
      "warm_self_sleepP16_latest":"Self-Sleep latest","warm_self_sleepP16_warmmix":"Self-Sleep warm"}
    colors=plt.cm.tab10.colors; steps=STEPS
    fig,axes=plt.subplots(1,2,figsize=(16,6),sharex=True,sharey=True)
    for ax,group,title in zip(axes,("old","new"),("Old classes: World/Sports","New classes: Business/Sci-Tech")):
        for i,arm in enumerate(ARMS):
            stats=[analysis["trajectory_mean_sample_sd"][arm][str(s)][group][METRICS[0]] for s in steps]
            means=[x["mean"] for x in stats];sd=[x["sample_sd"] for x in stats]
            ax.errorbar(steps,means,yerr=sd,label=labels[arm],marker="o",capsize=2,color=colors[i%len(colors)])
        ax.set_title(title);ax.set_xlabel("New-task primary wake updates");ax.set_ylabel("Full-vocabulary exact two-token EM")
        ax.set_xticks(steps);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.25)
    axes[1].legend(fontsize=9,loc="upper center",bbox_to_anchor=(.5,-.18),ncol=4)
    fig.tight_layout();fig.savefig(Path(out_dir)/"trajectory_em.png",dpi=160);fig.savefig(Path(out_dir)/"trajectory_em.svg");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(16,6),sharey=True);x=list(range(len(ARMS)));width=.68
    for ax,group,title in zip(axes,("old","new"),("Old classes, step 448","New classes, step 448")):
        stats=[analysis["endpoint_step448_mean_sample_sd"][arm][group][METRICS[0]] for arm in ARMS]
        ax.bar(x,[s["mean"] for s in stats],width,yerr=[s["sample_sd"] for s in stats],capsize=3,color=colors[:len(ARMS)])
        ax.set_title(title);ax.set_xticks(x);ax.set_xticklabels([labels[a] for a in ARMS],rotation=25,ha="right")
        ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis="y",alpha=.25)
    fig.tight_layout();fig.savefig(Path(out_dir)/"endpoint_em.png",dpi=160);fig.savefig(Path(out_dir)/"endpoint_em.svg");plt.close(fig)

def fmt(stat,percent=False,digits=3):
    if stat["mean"] is None:return "—"
    scale=100 if percent else 1
    return f"{stat['mean']*scale:.{digits}f} ± {stat['sample_sd']*scale:.{digits}f}"

LABELS={"warm_CMS_reference":"CMS reference","warm_anchorKD":"Anchor KD","warm_anchorGold":"Anchor Gold",
 "warm_anchorKD_sleepP16_latest":"KD Sleep latest","warm_anchorKD_sleepP16_warmmix":"KD Sleep warmmix",
 "warm_self_sleepP16_latest":"Self Sleep latest","warm_self_sleepP16_warmmix":"Self Sleep warmmix"}

def markdown_deliverables(analysis):
    endpoint=["# Endpoint metrics at step 448","","EM and accuracy are percentages; NLL is nats per answer token. Entries are mean ± sample SD over the three paired seeds. Gates retain all seed outcomes.","",
      "| Arm | Old EM | New EM | All EM | Old NLL | New NLL | Old pass | New pass | Both pass |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        e=analysis["endpoint_step448_mean_sample_sd"][arm];g=analysis["task_gates"][arm]
        endpoint.append(f"| {LABELS[arm]} | {fmt(e['old'][METRICS[0]],True,1)} | {fmt(e['new'][METRICS[0]],True,1)} | {fmt(e['all'][METRICS[0]],True,1)} | {fmt(e['old'][METRICS[1]])} | {fmt(e['new'][METRICS[1]])} | {g['old_pass_seeds']}/3 | {g['new_pass_seeds']}/3 | {g['both_pass_seeds']}/3 |")
    trajectory=["# Validation trajectory","","Exactly 42 arm-by-step mean rows; metric uncertainty is available in analysis.json as sample SD over seeds. The validation state count is 126 (21 runs × 6 checkpoints), with 512 examples per state.","",
      "| Arm | Step | Old EM | New EM | All EM | Class 0 | Class 1 | Class 2 | Class 3 | Old NLL | New NLL |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
      for step in STEPS:
        x=analysis["trajectory_mean_sample_sd"][arm][str(step)]
        vals=[fmt(x[g][METRICS[0]],True,1) for g in ("old","new","all","class0","class1","class2","class3")]
        trajectory.append("| "+LABELS[arm]+f" | {step} | "+" | ".join(vals+[fmt(x['old'][METRICS[1]]),fmt(x['new'][METRICS[1]])])+" |")
    perseed=["# Per-seed endpoint values","","Validation endpoint at step 448 and the task gates for each seed; no seed is selected or dropped.","",
      "| Seed | Arm | Old EM | New EM | All EM | Old NLL | New NLL | Initial old EM | Old EM forgetting | Old gate | New gate | Both |","|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for row in analysis["per_seed_runs"]:
        m=row["trajectory"]["448"];g=row["qualification"]["gates"]
        perseed.append(f"| {row['seed']} | {LABELS[row['arm']]} | {m['old'][METRICS[0]]*100:.1f}% | {m['new'][METRICS[0]]*100:.1f}% | {m['all'][METRICS[0]]*100:.1f}% | {m['old'][METRICS[1]]:.3f} | {m['new'][METRICS[1]]:.3f} | {row['initial_old_em']*100:.1f}% | {row['old_em_forgetting_step0_minus_step448']*100:.1f} pp | {bool(g['old']['pass'])} | {bool(g['new']['pass'])} | {bool(row['qualification']['all_tasks_pass'])} |")
    paired=["# Paired endpoint contrasts","","Validation step 448. Each cell is the paired first-minus-second difference mean ± sample SD across seeds; interactions use the frozen signed formulas. No p-values are reported.","",
      "| Contrast | Old EM Δ | New EM Δ | All EM Δ | Old NLL Δ | New NLL Δ | n |","|---|---:|---:|---:|---:|---:|---:|"]
    contrast_rows=[(x["contrast"],x["trajectory"][-1]["metrics"]) for x in analysis["paired_contrasts"]]
    contrast_rows += [(x["interaction"],x["trajectory"][-1]["metrics"]) for x in analysis["paired_interactions"]]
    for label,m in contrast_rows:
        paired.append(f"| {label} | {fmt(m['old'][METRICS[0]],True,1)} | {fmt(m['new'][METRICS[0]],True,1)} | {fmt(m['all'][METRICS[0]],True,1)} | {fmt(m['old'][METRICS[1]])} | {fmt(m['new'][METRICS[1]])} | 3 |")
    work=["# Work ledger","","Source counters are retained by explicit row.work keys and checked against the global seal, including zero-valued counters. These token/call counts are compute proxies, not FLOPs.","",
      f"- Primary wake updates: {analysis['cost']['primary_updates']}",f"- Sleep events / Adam updates: {analysis['cost']['sleep_events']} / {analysis['cost']['sleep_Adam_updates']}",
      f"- KD old auxiliary steps / context rows: {analysis['cost']['kd_old_aux_steps']} / {analysis['cost']['kd_old_prefix_rows']}",
      f"- Gold old auxiliary steps / context rows / answer targets: {analysis['cost']['gold_old_aux_steps']} / {analysis['cost']['gold_old_prefix_rows']} / {analysis['cost']['gold_old_answer_targets']}",
      "- The old subset is balanced using class metadata. KD uses frozen-teacher soft targets rather than old gold answers, but the balanced sampling still uses class labels; it is not wholly label-free.",
      "- KD and Gold auxiliary paths have different computation; comparisons do not isolate target type at equal FLOPs.",
      "- External run ledger must separately include the 48-update interference diagnostic, 18-update qualification and earlier failed probes.","",
      "## Per-arm measured work","","| Arm | Primary updates | Sleep events | Sleep Adam updates | Seconds mean +/- SD | Metered stages/counters |","|---|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        x=analysis["cost"]["by_arm"][arm]
        work.append(f"| {LABELS[arm]} | {x['primary_updates']} | {x['sleep_events']} | {x['sleep_Adam_updates']} | {fmt(x['mean_seconds'])} | `{json.dumps(x['meter_work'],sort_keys=True,separators=(',',':'))}` |")
    return {"endpoint-table.md":"\n".join(endpoint),"trajectory-table.md":"\n".join(trajectory),
            "per-seed-endpoints.md":"\n".join(perseed),"paired-endpoints.md":"\n".join(paired),
            "work-ledger.md":"\n".join(work),"work-ledger.json":None}

def write_markdown_deliverables(analysis,out_dir):
    for name,text in markdown_deliverables(analysis).items():
        if text is not None:write_text_lf(Path(out_dir)/name,text)
    ledger={"schema":"agnews-anchored-work-ledger-v5","cost":analysis["cost"],
        "external_costs_not_in_seal":analysis["cost"]["external_costs_not_in_seal"]}
    write_json(Path(out_dir)/"work-ledger.json",ledger)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seal",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--plots",action="store_true")
    a=p.parse_args();seal=read(a.seal);provenance,protocol=source_check(a.seal,seal)
    provenance["seal_sha256"]=sha(a.seal);analysis=analyze(seal,protocol,provenance)
    a.out_dir.mkdir(parents=True,exist_ok=False)
    dest=a.out_dir/"analysis.json"
    write_json(dest,analysis)
    write_markdown_deliverables(analysis,a.out_dir)
    plot_status="not_requested"
    if a.plots:
        try:plot_results(analysis,a.out_dir);plot_status="written"
        except ImportError:plot_status="matplotlib_unavailable"
    print(json.dumps({"analysis":str(dest),"plots":plot_status,"runs":21,"validation_states":126}))

if __name__=="__main__":main()
