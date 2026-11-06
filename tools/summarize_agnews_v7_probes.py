"""Strict standard-library aggregation for v7 stress and probe aggregates."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path,obj):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def mean_sd(xs):
    if not xs: return {"n":0,"mean":None,"sample_sd":None}
    m=sum(xs)/len(xs); sd=math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1)) if len(xs)>1 else 0.0
    return {"n":len(xs),"mean":m,"sample_sd":sd}

def validate(stress,sp,diag,dp):
    assert stress["schema"]=="agnews-single-receiver-wake-stress-v7" and stress["complete"] and stress["passed"]
    assert diag["schema"]=="agnews-single-receiver-diagnostics-v7" and diag["complete"]
    assert stress["confirmation_scored"] is False and diag["confirmation_scored"] is False
    assert stress["actual_CMS_ticks"]==1536 and stress["qualification"] is False
    assert diag["optimizer_updates"]==0 and diag["independent_gradient_states"]==93
    assert len(stress["rows"])==12 and len(diag["gradient_rows"])==120 and len(diag["lesion_rows"])==120
    assert sum(r["updates"] for r in stress["rows"])==1536
    assert sp["schema"]=="agnews-single-receiver-wake-stress-plan-v7"
    assert dp["schema"]=="agnews-single-receiver-diagnostic-plan-v7"
    assert stress["main_seal_sha256"]==sp["main_seal_sha256"]==dp["main_seal_sha256"]
    assert sp["steps"]==128 and sp["evaluation_steps"]==[0,4,16,64,128]
    assert dp["steps"]==[0,64,256,1024] and dp["lesions"]==[[],[0],[1],[0,1]]
    assert set(dp["source_sha256"]) >= {"probe_agnews_single_receiver_v7.py","single_receiver_probes_v7.py","run_agnews_single_receiver_v7.py","agnews_tail_cache_v7.py","matmul_work_v7.py"}
    stress_keys={(r["seed"],r["source_arm"],r["base_writes"]) for r in stress["rows"]}
    assert stress_keys=={(s,a,w) for s in (17,29,43) for a in ("split_receiver","split_shared") for w in (True,False)}
    gradient_keys={(r["seed"],r["arm"],r["step"]) for r in diag["gradient_rows"]}
    assert len(gradient_keys)==120
    arms=("current_receiver","current_shared","split_receiver","split_shared","split_narrow_receiver","split_narrow_shared","gold_receiver","gold_shared","new_only_receiver","new_only_shared")
    assert gradient_keys=={(s,a,k) for s in (17,29,43) for a in arms for k in (0,64,256,1024)}
    assert sum(not r["reused_common_initial_probe"] for r in diag["gradient_rows"])==93
    lesion_keys={(r["seed"],r["arm"],tuple(r["disabled_slow_outputs"])) for r in diag["lesion_rows"]}
    assert len(lesion_keys)==120
    assert lesion_keys=={(s,a,tuple(x)) for s in (17,29,43) for a in arms for x in ([],[0],[1],[0,1])}
    for row in stress["rows"]:
        assert row["status"]=="complete" and row["updates"]==128 and len(row["evaluations"])==5
        assert row["base_writes"] in (True,False) and row["seed"] in (17,29,43)
        assert all(x["validation"]["all"]["examples"]==512 for x in row["evaluations"])
        assert row["slow_and_frozen_scope_guard_passed"] is True
    for row in diag["gradient_rows"]:
        assert row["unchanged_model_policy_rng"] is True and row["seed"] in (17,29,43)
        assert row["arm"] in ("current_receiver","current_shared","split_receiver","split_shared","split_narrow_receiver","split_narrow_shared","gold_receiver","gold_shared","new_only_receiver","new_only_shared") and row["step"] in (0,64,256,1024)
    for row in diag["lesion_rows"]:
        assert row["model_policy_rng_unchanged"] is True and row["seed"] in (17,29,43)
        assert row["arm"] in ("current_receiver","current_shared","split_receiver","split_shared","split_narrow_receiver","split_narrow_shared","gold_receiver","gold_shared","new_only_receiver","new_only_shared") and row["disabled_slow_outputs"] in ([],[0],[1],[0,1])

def hash_obj(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def _metrics(row,step,which="validation"):
    ev=next(x for x in row["evaluations"] if x["step"]==step)
    return ev[which]

def aggregate(stress,diag):
    arms=("current_receiver","current_shared","split_receiver","split_shared","split_narrow_receiver","split_narrow_shared","gold_receiver","gold_shared","new_only_receiver","new_only_shared")
    steps=[0,4,16,64,128]; stress_cells=[]
    for arm in ("split_receiver","split_shared"):
      for writes in (True,False):
       for step in steps:
        rs=[r for r in stress["rows"] if r["source_arm"]==arm and r["base_writes"] is writes]
        cell={"source_arm":arm,"scope":arm.rsplit("_",1)[1],"base_writes":writes,"step":step}
        for group in ("old","new","all"):
            cell[group+"_em"]=mean_sd([_metrics(r,step)[group]["em"] for r in rs])
        stress_cells.append(cell)
    stress_contrasts=[]
    for arm in ("split_receiver","split_shared"):
      for seed in (17,29,43):
        pair={r["base_writes"]:r for r in stress["rows"] if r["source_arm"]==arm and r["seed"]==seed}
        for step in steps:
          a=_metrics(pair[True],step);b=_metrics(pair[False],step)
          stress_contrasts.append({"arm":arm,"seed":seed,"step":step,"written_minus_skipped_old_em":a["old"]["em"]-b["old"]["em"],"written_minus_skipped_new_em":a["new"]["em"]-b["new"]["em"]})
    grad=[]; arms=("current_receiver","current_shared","split_receiver","split_shared","split_narrow_receiver","split_narrow_shared","gold_receiver","gold_shared","new_only_receiver","new_only_shared")
    for arm in arms:
      for step in (64,256,1024,0):
       for group in ("receiver","base0","base1"):
        rows=[r for r in diag["gradient_rows"] if r["arm"]==arm and r["step"]==step]
        subset=[r["gradient"][group] for r in rows]
        grad.append({"arm":arm,"scope":arm.rsplit("_",1)[1],"step":step,"group":group,"initial_probe_reused_across_arms":step==0,**{k:mean_sd([float(x[k]) for x in subset if x[k] is not None]) for k in ("cosine","old_norm","new_norm")},"null_cosine_count":sum(x["cosine"] is None for x in subset)})
    initial_common=[]
    for group in ("receiver","base0","base1"):
        subset=[r["gradient"][group] for r in diag["gradient_rows"] if r["step"]==0 and not r["reused_common_initial_probe"]]
        initial_common.append({"group":group,**{k:mean_sd([float(x[k]) for x in subset if x[k] is not None]) for k in ("cosine","old_norm","new_norm")}})
    lesions=[]
    for lesion in ([],[0],[1],[0,1]):
      for arm in arms:
        subset=[r for r in diag["lesion_rows"] if r["disabled_slow_outputs"]==lesion and r["arm"]==arm]
        cell={"disabled_slow_outputs":lesion,"arm":arm}
        for g in ("old","new","all"):
          cell[g+"_em"]=mean_sd([r["validation"][g]["em"] for r in subset])
        lesions.append(cell)
    routes=[]
    for arm in arms:
      for disabled in ([],[0],[1],[0,1]):
        subset=[r for r in diag["lesion_rows"] if r["arm"]==arm and r["disabled_slow_outputs"]==disabled]
        audits=[r["semantic_position_route_audit"] for r in subset if r["semantic_position_route_audit"] is not None]
        if audits:
          npos=sum(float(a["loss_positions"]) for a in audits)
          routes.append({"arm":arm,"disabled_slow_outputs":disabled,"n":len(audits),"mean_loss_positions":npos/len(audits),"natural_old_fraction":sum(float(a["natural_old"]) for a in audits)/npos,"natural_new_fraction":sum(float(a["natural_new"]) for a in audits)/npos,"mean_null_probability":sum(float(a["null_probability_sum"]) for a in audits)/npos,"mean_executed_old_coefficient":sum(float(a["executed_old_coefficient_sum"]) for a in audits)/npos,"mean_executed_new_coefficient":sum(float(a["executed_new_coefficient_sum"]) for a in audits)/npos})
    fast=[]
    for arm in ("split_receiver","split_shared"):
     for writes in (True,False):
      subset=[r for r in stress["rows"] if r["source_arm"]==arm and r["base_writes"] is writes]
      cell={"source_arm":arm,"scope":arm.rsplit("_",1)[1],"base_writes":writes}
      for g in ("old","new","all"):
        cell[g+"_fast_masked_em"]=mean_sd([r["fast_output_masked_validation"][g]["em"] for r in subset])
        cell[g+"_unmasked_em"]=mean_sd([r["full_validation"][g]["em"] for r in subset])
        cell[g+"_offline_baseline_em"]=mean_sd([_metrics(r,0)[g]["em"] for r in subset])
        cell[g+"_fast_masked_minus_offline_baseline"]=mean_sd([r["fast_output_masked_validation"][g]["em"]-_metrics(r,0)[g]["em"] for r in subset])
      fast.append(cell)
    return {"stress_validation_trajectory":stress_cells,"base_write_effects_by_seed":stress_contrasts,"fast_masked_offline_baseline":fast,"gradient_group_trajectory":grad,"initial_common_gradient":initial_common,"lesion_validation":lesions,"natural_route_coefficients":routes,
        "work_lower_bound":{"stress":stress["work"],"diagnostics":diag["work"],"note":"Separate observed matmul lower bounds; do not add shared preparation work across phases."},
        "counts":{"stress_branches":len(stress["rows"]),"stress_ticks":stress["actual_CMS_ticks"],"gradient_rows":len(diag["gradient_rows"]),"independent_gradient_states":diag["independent_gradient_states"],"lesion_rows":len(diag["lesion_rows"]),"optimizer_updates":diag["optimizer_updates"],"confirmation_scored":False}}

def markdown(out):
    lines=["# AGNews v7 stress and probe aggregates","","Descriptive development diagnostics only; no confirmation scoring or optimizer updates in probes. FLOPs are observed matmul lower bounds. Gradient cosine is a local comparison for these objectives and parameters; it does not establish causal uniqueness.","","## Stress validation EM","","| Source arm / scope | Base writes | Step | Old mean (SD) | New mean (SD) | All mean (SD) |","|---|---|---:|---:|---:|---:|---:|"]
    for x in out["stress_validation_trajectory"]:
        def f(v):return "NA" if v["mean"] is None else f'{v["mean"]:.4f} ({v["sample_sd"]:.4f})'
        lines.append(f'| {x["source_arm"]} / {x["scope"]} | {x["base_writes"]} | {x["step"]} | {f(x["old_em"])} | {f(x["new_em"])} | {f(x["all_em"])} |')
    lines += ["","## Base-write contrasts","","Each entry is written minus skipped within the same seed and arm.","","| Arm | Seed | Step | Old EM delta | New EM delta |","|---|---:|---:|---:|---:|"]
    for x in out["base_write_effects_by_seed"]:
        lines.append(f'| {x["arm"]} | {x["seed"]} | {x["step"]} | {x["written_minus_skipped_old_em"]:.4f} | {x["written_minus_skipped_new_em"]:.4f} |')
    lines += ["","## Fast-output lesion versus offline baseline","","| Source arm / scope | Base writes | Group | Fast-masked EM | Step-0 baseline EM | Difference |","|---|---|---|---:|---:|---:|"]
    for x in out["fast_masked_offline_baseline"]:
        for g in ("old","new","all"):
            lines.append(f'| {x["source_arm"]} / {x["scope"]} | {x["base_writes"]} | {g} | {x[g+"_fast_masked_em"]["mean"]:.4f} | {x[g+"_offline_baseline_em"]["mean"]:.4f} | {x[g+"_fast_masked_minus_offline_baseline"]["mean"]:.4f} |')
    lines += ["","## Gradient groups","","| Arm / scope | Step | Group | Cosine mean (SD) | Old norm | New norm |","|---|---:|---|---:|---:|---:|"]
    for x in out["gradient_group_trajectory"]:
        def f(v):return "NA" if v["mean"] is None else f'{v["mean"]:.4f} ({v["sample_sd"]:.4f})'
        lines.append(f'| {x["arm"]} / {x["scope"]} | {x["step"]} | {x["group"]} | {f(x["cosine"])} | {f(x["old_norm"])} | {f(x["new_norm"])} |')
    lines += ["","## Slow-output lesions","","| Disabled slots | Arm | Old EM | New EM | All EM |","|---|---|---:|---:|---:|"]
    for x in out["lesion_validation"]:
        lines.append(f'| {x["disabled_slow_outputs"]} | {x["arm"]} | {x["old_em"]["mean"]:.4f} | {x["new_em"]["mean"]:.4f} | {x["all_em"]["mean"]:.4f} |')
    lines += ["","## Natural route coefficients","","| Arm | Natural old | Natural new | Null probability | Executed old | Executed new |","|---|---:|---:|---:|---:|---:|"]
    for x in out["natural_route_coefficients"]:
        if x["disabled_slow_outputs"]==[]:
            lines.append(f'| {x["arm"]} | {x["natural_old_fraction"]:.4f} | {x["natural_new_fraction"]:.4f} | {x["mean_null_probability"]:.4f} | {x["mean_executed_old_coefficient"]:.4f} | {x["mean_executed_new_coefficient"]:.4f} |')
    return "\n".join(lines)+"\n"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stress-dir",type=Path,required=True);p.add_argument("--diagnostics-dir",type=Path,required=True);p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); sp=read(a.stress_dir/"plan.json");sr=read(a.stress_dir/"result.json");dp=read(a.diagnostics_dir/"plan.json");dr=read(a.diagnostics_dir/"result.json")
    assert sr["plan_sha256"]==sha(a.stress_dir/"plan.json") and dr["plan_sha256"]==sha(a.diagnostics_dir/"plan.json")
    validate(sr,sp,dr,dp)
    root=Path(__file__).resolve().parents[1]
    for rel,h in dp["source_sha256"].items(): assert sha(root/"tools"/rel)==h,(rel,"source hash mismatch")
    assert sha(root/"tools"/"stress_agnews_single_receiver_v7.py")==sp["source_sha256"]
    result=aggregate(sr,dr);result["input_sha256"]={"stress_result":sha(a.stress_dir/"result.json"),"stress_plan":sha(a.stress_dir/"plan.json"),"diagnostics_result":sha(a.diagnostics_dir/"result.json"),"diagnostics_plan":sha(a.diagnostics_dir/"plan.json")}
    a.out_dir.mkdir(parents=True,exist_ok=False);write(a.out_dir/"aggregate.json",result);(a.out_dir/"aggregate.md").write_text(markdown(result),encoding="utf-8",newline="\n")

if __name__=="__main__":main()
