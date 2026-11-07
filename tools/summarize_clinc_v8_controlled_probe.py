"""Read-only summary for a completed CLINC controlled-routing development probe.
Reads only protocol.json, metrics.jsonl, and development-sealed.json; never loads checkpoints or data.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values):
    xs=[float(x) for x in values if x is not None]
    if not xs: return {"n":0,"mean":None,"sd":None}
    mean=sum(xs)/len(xs)
    return {"n":len(xs),"mean":mean,"sd":math.sqrt(sum((x-mean)**2 for x in xs)/(len(xs)-1)) if len(xs)>1 else None}


def _subvalue(item, phase, key):
    value=item.get(phase)
    if isinstance(value,dict): return value.get(key)
    return value if key=="objective" and isinstance(value,(int,float)) else None


def summarize(run_dir):
    root=Path(run_dir)
    paths={name:root/name for name in ("protocol.json","metrics.jsonl","development-sealed.json")}
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing: raise ValueError(f"incomplete input; missing files: {missing}")
    protocol=json.loads(paths["protocol.json"].read_text(encoding="utf-8"))
    seal=json.loads(paths["development-sealed.json"].read_text(encoding="utf-8"))
    rows=[json.loads(line) for line in paths["metrics.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
    commits=[r for r in rows if r.get("event")=="controlled_sleep_commit"]
    complete={r["arm"]:r["receipt"] for r in rows if r.get("event")=="run_complete"}
    by=defaultdict(list)
    for event in commits:
        for item in event.get("losses",[]):
            role=item.get("paired_prompt_selection",{}).get("role","unknown")
            owner=item.get("sample_owner", "gold_label" if item.get("uses_gold_intent") else "unknown")
            by[(event["arm"],role,owner,item.get("kind","unknown"))].append(item)
    aggregates={}
    for (arm,role,owner,kind),items in sorted(by.items()):
        key="|".join((arm,role,owner,kind))
        bkl=[_subvalue(x,"same_sample_fit_before","kl") for x in items]
        akl=[_subvalue(x,"same_sample_fit_after","kl") for x in items]
        before_obj=[_subvalue(x,"same_sample_fit_before","objective") for x in items]
        after_obj=[_subvalue(x,"same_sample_fit_after","objective") for x in items]
        before_ce=[_subvalue(x,"same_sample_fit_before","student_ce") for x in items]
        after_ce=[_subvalue(x,"same_sample_fit_after","student_ce") for x in items]
        gold_scalar=(arm=="GOLD_CE_MIX")
        valid=[int(x["sampled_label_valid"]) for x in items if isinstance(x.get("sampled_label_valid"),bool)]
        matches=[int(x["sampled_label_matches_prompt_gold_diagnostic"]) for x in items if isinstance(x.get("sampled_label_matches_prompt_gold_diagnostic"),bool)]
        aggregates[key]={"n":len(items),"sample_owner":owner,"kind":kind,
            "same_sample_kl_before":_stats(bkl),"same_sample_kl_after":_stats(akl),
            "same_sample_kl_delta_after_minus_before":_stats([a-b for a,b in zip(akl,bkl) if a is not None and b is not None]),
            "same_sample_objective_before":_stats(before_obj),"same_sample_objective_after":_stats(after_obj),
            "same_sample_ce_before":_stats(before_obj if gold_scalar else before_ce),
            "same_sample_ce_after":_stats(after_obj if gold_scalar else after_ce),
            "sampled_answer_valid_rate":_stats(valid),"sampled_answer_matches_prompt_gold_diagnostic_rate":_stats(matches),
            "natural_top1_fresh_receiver_coverage":_stats([x.get("natural_vs_controlled_route_before",{}).get("natural_top1_receiver_coverage") for x in items]),
            "controlled_fresh_branch_executed_coverage":_stats([x.get("natural_vs_controlled_route_before",{}).get("fresh_branch_executed_answer_coverage") for x in items]),
            "prompt_id_hashes":[x.get("prompt_id_sha256") for x in items]}
        if gold_scalar: aggregates[key]["sampling_diagnostic_note"]="not applicable: gold CE emits no sampled-answer diagnostic fields"
    exposure={}
    for arm in ("SLEEP_MIX","GOLD_CE_MIX"):
        items=[x for event in commits if event.get("arm")==arm for x in event.get("losses",[])]
        counts=Counter(x.get("paired_prompt_selection",{}).get("role") for x in items)
        classes=defaultdict(set)
        for x in items:
            meta=x.get("paired_prompt_selection",{}); classes[meta.get("role")].add(meta.get("class_index"))
        exposure[arm]={"calls":len(items),"role_counts":dict(counts),
            "covered_class_indices":{k:sorted(v) for k,v in classes.items()},
            "unique_prompt_hashes":len({x.get("prompt_id_sha256") for x in items}),
            "prompt_id_sequence_sha256":hashlib.sha256(json.dumps([x.get("prompt_id_sha256") for x in items],separators=(",",":")).encode()).hexdigest()}
    endpoints={}
    for arm in ("SLEEP_MIX","GOLD_CE_MIX"):
        for row in rows:
            if row.get("arm")==arm and row.get("event")=="task1_end_eval_clean_full":
                endpoints[arm]={k:{"accuracy":v["accuracy"],"correct":round(v["accuracy"]*v["count"]),"count":v["count"],"invalid":v["invalid"],"predictions_sha256":v["predictions_sha256"]} for k,v in row["result"]["metrics"].items()}
    teacher_items=[x for (a,_r,o,_k),items in by.items() if a=="SLEEP_MIX" and o=="teacher" for x in items]
    teacher_valid=[x["sampled_label_valid"] for x in teacher_items if isinstance(x.get("sampled_label_valid"),bool)]
    teacher_match=[x["sampled_label_matches_prompt_gold_diagnostic"] for x in teacher_items if isinstance(x.get("sampled_label_matches_prompt_gold_diagnostic"),bool)]
    work=seal.get("work",{})
    complete_status=bool(seal.get("complete") and len(complete)==2 and len(commits)==32 and seal.get("test_data_read") is False)
    return {"schema":"clinc-v8-controlled-probe-analysis-v1","status":"complete" if complete_status else "pending_or_incomplete",
        "source":{"run_dir":str(root.resolve()),"protocol_sha256":_sha(paths["protocol.json"]),"seal_sha256":_sha(paths["development-sealed.json"]),"metrics_sha256":_sha(paths["metrics.jsonl"]),"source_lock_sha256":protocol.get("source_lock_sha256"),"source_hashes":protocol.get("source_hashes"),"training_manifest_sha256":protocol.get("training_manifest_sha256"),"parent_checkpoint_sha256":protocol.get("parent_task0_checkpoint_sha256")},
        "scope":{"test_data_read":seal.get("test_data_read"),"seed":protocol.get("config",{}).get("seed"),"arms":list(complete),"sleep_commits":len(commits),"sampled_sleep_requests":sum(len(x.get("losses",[])) for x in commits),"source_arm":"qualified HOPE task0; task1 continuation only"},
        "exposure":exposure,"by_arm_role_owner_kind":aggregates,
        "teacher_sample_summary":{"owner":"teacher only; excludes student-owned GKD samples","n":len(teacher_valid),"canonical_label_valid_count":sum(teacher_valid),"canonical_label_valid_rate":sum(teacher_valid)/len(teacher_valid) if teacher_valid else None,"matches_prompt_gold_count":sum(teacher_match),"matches_prompt_gold_rate":sum(teacher_match)/len(teacher_match) if teacher_match else None},
        "endpoint_seen_clean_development":endpoints,
        "cost":{"total_full_backbone_forward_calls":work.get("full_backbone_forward_calls"),"full_backbone_forward_tokens":work.get("full_backbone_forward_tokens"),"total_tail_calls":work.get("tail_calls"),"tail_grad_enabled_calls":work.get("tail_grad_enabled_calls"),"tail_backward_calls":work.get("tail_backward_calls"),"optimizer_or_lifecycle_update_steps":work.get("optimizer_or_lifecycle_update_steps"),"sleep_events":work.get("sleep_events"),"active_wall_seconds":work.get("active_wall_seconds"),"per_arm":{a:{"active_seconds":r.get("active_seconds"),"forward_calls":r.get("work_arm_delta",{}).get("full_backbone_forward_calls"),"forward_tokens":r.get("work_arm_delta",{}).get("full_backbone_forward_tokens"),"physical_sleep_adamw_steps":r.get("physical_sleep_adamw_steps"),"sleep_events":r.get("sleep_event_count")} for a,r in complete.items()}},
        "observations":["Single seed and development data only; diagnostic, not a multi-seed claim.","Same-sample loss changes show local fit, not held-out transfer.","Natural top-1 receiver coverage is distinct from forced 0.5 branch execution.","GOLD_CE_MIX uses extra gold labels and is not FLOP- or information-matched.","Only protocol, metrics, and development seal were read; no test data or checkpoint contents."]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True); parser.add_argument('--out',required=True)
    args=parser.parse_args(argv); result=summarize(args.run_dir)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps({'status':result['status'],'out':str(out),'arms':result['scope']['arms'],'test_data_read':result['scope']['test_data_read']}))
if __name__=='__main__': main()
