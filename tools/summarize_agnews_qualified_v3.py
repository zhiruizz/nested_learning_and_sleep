"""Summarize an already sealed AG News v3 comparison and evaluation summary.

This offline tool reads only the seal, evaluation summary and frozen protocol.
It never opens model checkpoints, token records, or holdout files.
"""
from __future__ import annotations
import argparse
import json
import math
import hashlib
import csv
from pathlib import Path

ARMS = ("frozen", "sft", "cms", "cms_sleep", "cms_extra_replay")
SEEDS = (17, 29, 43)
ACCURACY_KEYS = ("full_vocab_two_token_greedy_exact_match", "candidate_accuracy")
NLL_KEYS = ("answer_nll_nats_per_token",)

def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)

def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def mean_sd(values):
    values = [float(x) for x in values]
    if not values:
        return {"n": 0, "mean": None, "sample_sd": None}
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1) if len(values) > 1 else None
    return {"n": len(values), "mean": mean, "sample_sd": math.sqrt(variance) if variance is not None else None}

def metric_value(metrics, group, key):
    part = metrics.get(group)
    if not isinstance(part, dict):
        return None
    value = part.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None

def find_accuracy_key(metrics):
    key="full_vocab_two_token_greedy_exact_match"
    return key if all(metric_value(metrics, group, key) is not None for group in ("old", "new", "all")) else None

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def validate_inputs(seal, evaluation, seal_path, protocol, protocol_path):
    if seal.get("schema") != "agnews-qualified-training-seal-v3" or seal.get("complete") is not True:
        raise ValueError("training seal is missing or incomplete")
    if evaluation.get("schema") != "agnews-qualified-evaluation-v3":
        raise ValueError("evaluation summary schema mismatch")
    if seal.get("old_all_qualified") is not True or seal.get("heldout_read") is not False:
        raise ValueError("training seal does not certify qualified train-only training")
    if evaluation.get("all_foundations_qualified") is not True or evaluation.get("trained_after_holdout_open") is not False:
        raise ValueError("evaluation summary does not certify the frozen qualified evaluation")
    if file_sha256(seal_path) != evaluation.get("provenance", {}).get("seal_sha256"):
        raise ValueError("evaluation provenance does not match the training seal SHA256")
    if seal.get("records_sha256") != evaluation.get("provenance", {}).get("records_sha256"):
        raise ValueError("training/evaluation records provenance mismatch")
    if file_sha256(protocol_path)!=seal.get('protocol_sha256'):raise ValueError('protocol SHA mismatch')
    cfg=protocol['config'];provenance=evaluation['provenance']
    if protocol['config_sha256']!=seal['config_sha256']:raise ValueError('config SHA mismatch')
    if cfg['records_sha256']!=seal['records_sha256']:raise ValueError('config records mismatch')
    for key,cfg_key in [('evaluator_sha256','evaluator_sha256'),('helper_sha256','helper_source_sha256'),('holdout_prompts_sha256','holdout_prompts_sha256'),('holdout_labels_sha256','holdout_labels_sha256')]:
        if provenance.get(key)!=cfg[cfg_key]:raise ValueError(f'frozen evaluation provenance mismatch: {key}')
    if evaluation.get('unique_holdout_examples')!=1024 or evaluation.get('examples_per_class')!=256:raise ValueError('holdout denominator mismatch')
    runs = seal.get("runs")
    results = evaluation.get("results")
    expected = {(s, a) for s in SEEDS for a in ARMS}
    if not isinstance(runs, list) or {(r.get("seed"), r.get("arm")) for r in runs} != expected or len(runs) != 15:
        raise ValueError("seal must contain exactly the 15 complete seed/arm runs")
    if not isinstance(results, dict) or set(results) != {f"{s}/{a}" for s, a in expected}:
        raise ValueError("evaluation results must contain all 15 seed/arm groups")
    n=seal.get("new_steps")
    if not isinstance(n,int) or n not in (512,1024,2048):raise ValueError("invalid new_steps")
    expected_steps={"frozen":[0,n], **{a:[0,n//4,n//2,3*n//4,n] for a in ARMS if a!="frozen"}}
    for row in runs:
        arm=row.get("arm")
        if row.get("status") != "complete":
            raise ValueError(f"run incomplete: {row.get('seed')}/{row.get('arm')}")
        expected_updates=0 if arm=="frozen" else n
        expected_replay=12*(n//64) if arm=="cms_extra_replay" else 0
        expected_sleep=24*(n//64) if arm=="cms_sleep" else 0
        if (row.get("primary_updates"),row.get("extra_replay_updates"),row.get("sleep_adam_updates")) != (expected_updates,expected_replay,expected_sleep):
            raise ValueError(f"run budget mismatch: {row.get('seed')}/{arm}")
        points = results[f"{row['seed']}/{arm}"]
        cps=row.get("checkpoints",[])
        if [cp.get("base_step") for cp in cps] != expected_steps[arm]: raise ValueError("sealed checkpoint steps mismatch")
        if [cp.get("sha256") for cp in cps] != [p.get("checkpoint_sha256") for p in points]: raise ValueError("checkpoint SHA fields mismatch")
        if [cp.get("base_step") for cp in cps] != [p.get("base_step") for p in points]: raise ValueError("evaluation endpoints do not match sealed checkpoints")
        if not isinstance(points, list) or not points:
            raise ValueError("each run must have at least one evaluated checkpoint")
        steps = [p.get("base_step") for p in points]
        if steps != sorted(set(steps)):
            raise ValueError("evaluation checkpoint steps must be unique and increasing")
        if any(not isinstance(p.get("metrics"), dict) for p in points):
            raise ValueError("checkpoint metrics are missing")
        for point in points:
            for group,count in {'old':512,'new':512,'all':1024,**{f'class{i}':256 for i in range(4)}}.items():
                if point['metrics'].get(group,{}).get('examples')!=count:raise ValueError(f'metric denominator mismatch: {group}')
                if metric_value(point['metrics'],group,'full_vocab_two_token_greedy_exact_match')is None:raise ValueError('full vocabulary EM missing')
    return runs, results

def summarize(run_dir, evaluation_dir):
    seal = read_json(Path(run_dir) / "training-sealed.json")
    evaluation = read_json(Path(evaluation_dir) / "evaluation-summary.json")
    protocol_path=Path(run_dir)/'protocol.json'
    if not protocol_path.exists():protocol_path=Path(run_dir)/'training-protocol.json'
    runs, results = validate_inputs(seal, evaluation, Path(run_dir) / "training-sealed.json",read_json(protocol_path),protocol_path)
    n = seal.get("new_steps")
    if not isinstance(n, int) or n not in (512, 1024, 2048):
        raise ValueError("sealed new_steps must be 512, 1024, or 2048")
    run_index = {(r["seed"], r["arm"]): r for r in runs}
    per_arm = {}
    for arm in ARMS:
        endpoint_rows = []
        trajectories = {}
        accuracy_key = None
        for seed in SEEDS:
            points = results[f"{seed}/{arm}"]
            final = points[-1]
            metrics = final["metrics"]
            key = find_accuracy_key(metrics)
            if key is None:
                raise ValueError(f"accuracy metric absent for {seed}/{arm}")
            if accuracy_key is not None and accuracy_key != key:
                raise ValueError("accuracy metric key differs across runs")
            accuracy_key = key
            trajectories[str(seed)] = [{"base_step": p["base_step"], "metrics": p["metrics"]} for p in points]
            row = {"seed": seed, "endpoint_step": final["base_step"], "metrics": metrics}
            endpoint_rows.append(row)
        summaries = {}
        for group in ("old", "new", "all", "class0", "class1", "class2", "class3"):
            summaries[group] = {}
            keys = set.intersection(*(set(r["metrics"].get(group, {})) for r in endpoint_rows)) if endpoint_rows else set()
            for key in sorted(keys):
                values = [metric_value(r["metrics"], group, key) for r in endpoint_rows]
                if all(v is not None for v in values):
                    summaries[group][key] = mean_sd(values)
        old_drops = []
        tradeoffs = []
        for seed in SEEDS:
            points = results[f"{seed}/{arm}"]
            baseline = next((p for p in points if p["base_step"] == 0), None)
            if baseline is None:
                continue
            endpoint = points[-1]
            base_old = metric_value(baseline["metrics"], "old", accuracy_key)
            end_old = metric_value(endpoint["metrics"], "old", accuracy_key)
            end_new = metric_value(endpoint["metrics"], "new", accuracy_key)
            end_all = metric_value(endpoint["metrics"], "all", accuracy_key)
            if None not in (base_old, end_old):
                old_drops.append({"seed": seed, "pre": base_old, "post": end_old, "drop": base_old-end_old})
            if None not in (end_old, end_new, end_all):
                tradeoffs.append({"seed": seed, "old": end_old, "new": end_new, "all": end_all,
                                  "old_new_gap": end_new-end_old, "all_minus_macro_old_new": end_all-(end_old+end_new)/2})
        costs = []
        for seed in SEEDS:
            r = run_index[(seed, arm)]
            work = r.get("work", {})
            costs.append({"seed": seed, "wall_seconds": r.get("seconds"),
                "primary_updates": r.get("primary_updates", 0),
                "extra_replay_updates": r.get("extra_replay_updates", 0),
                "logical_sgd_updates": r.get("primary_updates", 0)+r.get("extra_replay_updates", 0),
                "sleep_adam_updates": r.get("sleep_adam_updates", 0),
                "forward_input_tokens": sum(v.get("forward_input_tokens", 0) for v in work.values()),
                "gradient_forward_input_tokens": sum(v.get("gradient_forward_input_tokens", 0) for v in work.values()),
                "forward_calls": sum(v.get("forward_calls", 0) for v in work.values())})
        per_arm[arm] = {"accuracy_metric": accuracy_key, "endpoint_metrics_mean_sample_sd": summaries,
            "old_pre_post": old_drops, "old_drop_mean_sample_sd": mean_sd([x["drop"] for x in old_drops]),
            "endpoint_old_new_all_tradeoff": tradeoffs, "trajectory_by_seed": trajectories,
            "logical_training_cost_by_seed": costs,
            "logical_training_cost_mean_sample_sd": {k: mean_sd([x[k] for x in costs if isinstance(x.get(k),(int,float))])
                for k in ("wall_seconds","logical_sgd_updates","sleep_adam_updates","forward_input_tokens","gradient_forward_input_tokens","forward_calls")}}
    amendments = seal.get("operational_amendments", [])
    recovery = {"amendments_referenced_in_seal": amendments,
        "interruption_cost": seal.get("interruption_cost", "not recorded in the seal; see referenced operational amendments if present"),
        "duplicate_interrupted_work": seal.get("interrupted_duplicate_work_not_in_logical_counts", "not recorded")}
    return {"schema":"agnews-qualified-analysis-v3", "training_seal_records_sha256":seal["records_sha256"],
        "evaluation_provenance":evaluation.get("provenance"), "new_steps":n, "arms":per_arm,
        "evaluation_cost":{"seconds":evaluation.get("seconds"),"resumed_scores":evaluation.get("resumed_scores"),"work":evaluation.get("work")},
        "recovery_operational_cost_separate":recovery,
        "interpretation":{"seed_statistics":"Three-seed descriptive mean and sample SD; not a significance test.",
          "evaluation_scope":"Fresh within-AG News train-derived holdout; not external or pretraining-independent confirmation.",
          "balanced_all":"All-role metric is class-balanced by the frozen 256-per-class holdout; macro old/new mean is shown as a contrast, not substituted for the reported all metric.",
          "compute":"Forward-input tokens and optimizer-step counts are operational proxies, not FLOPs or GPU utilization."}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--evaluation-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--plots",action="store_true")
    args=parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory must be new")
    args.out_dir.mkdir(parents=True,exist_ok=False)
    report=summarize(args.run_dir,args.evaluation_dir)
    write_json(args.out_dir/"analysis-summary.json",report)
    em='full_vocab_two_token_greedy_exact_match'
    table=['| Arm | Old EM % | New EM % | All EM % | Old drop pp | Answer NLL |','|---|---:|---:|---:|---:|---:|']
    with (args.out_dir/'per-seed-endpoints.csv').open('x',encoding='utf-8',newline='')as stream:
        writer=csv.writer(stream);writer.writerow(['seed','arm','old_em','new_em','all_em','old_drop','answer_nll','primary_updates','extra_replay_updates','sleep_adam_updates','forward_tokens','gradient_forward_tokens','wall_seconds'])
        for arm,summary in report['arms'].items():
            values=[]
            for group in ('old','new','all'):
                x=summary['endpoint_metrics_mean_sample_sd'][group][em];values.append(f"{100*x['mean']:.2f} ± {100*x['sample_sd']:.2f}")
            drop=summary['old_drop_mean_sample_sd'];nll=summary['endpoint_metrics_mean_sample_sd']['all']['answer_nll_nats_per_token']
            table.append(f"| {arm} | {' | '.join(values)} | {100*drop['mean']:.2f} ± {100*drop['sample_sd']:.2f} | {nll['mean']:.3f} ± {nll['sample_sd']:.3f} |")
            for cost,trade,old_drop in zip(summary['logical_training_cost_by_seed'],summary['endpoint_old_new_all_tradeoff'],summary['old_pre_post']):
                seed=cost['seed'];endpoint=summary['trajectory_by_seed'][str(seed)][-1]['metrics']
                writer.writerow([seed,arm,trade['old'],trade['new'],trade['all'],old_drop['drop'],endpoint['all']['answer_nll_nats_per_token'],cost['primary_updates'],cost['extra_replay_updates'],cost['sleep_adam_updates'],cost['forward_input_tokens'],cost['gradient_forward_input_tokens'],cost['wall_seconds']])
    (args.out_dir/'table.md').write_text('\n'.join(table)+'\n',encoding='utf-8',newline='\n')
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10.4,4.5),sharex=True,sharey=True)
        colors=['#666666','#0072B2','#009E73','#D55E00','#CC79A7']
        names=['Frozen','Adapter SFT','CMS','CMS + Sleep (mix .5)','CMS + extra replay']
        for arm,color,label in zip(ARMS,colors,names):
            traj=report['arms'][arm]['trajectory_by_seed'];steps=[x['base_step']for x in traj['17']]
            for ax,group in zip(axes,('old','new')):
                ys=[[100*p['metrics'][group][em]for p in traj[str(seed)]]for seed in SEEDS]
                ms=[mean_sd([y[i]for y in ys])for i in range(len(steps))]
                means=[x['mean']for x in ms];sd=[x['sample_sd']for x in ms]
                ax.plot(steps,means,color=color,label=label,linewidth=2)
                ax.fill_between(steps,[max(0,m-s)for m,s in zip(means,sd)],[min(100,m+s)for m,s in zip(means,sd)],color=color,alpha=.1)
        for ax,title in zip(axes,('Old classes: World / Sports','New classes: Business / Sci-Tech')):
            ax.set(title=title,xlabel='Primary new-task minibatches',ylim=(0,100));ax.grid(alpha=.2);ax.set_xticks([0,report['new_steps']//4,report['new_steps']//2,3*report['new_steps']//4,report['new_steps']])
        axes[0].set_ylabel('Full-vocabulary two-token EM (%)')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
        fig.suptitle('Qualified AG News v3 · 3 seeds · mean ± sample SD',fontsize=12)
        fig.tight_layout(rect=(0,.14,1,.94))
        fig.savefig(args.out_dir/'retention-trajectory.png',dpi=180)
        fig.savefig(args.out_dir/'retention-trajectory.svg')
        plt.close(fig)
if __name__=="__main__": main()
