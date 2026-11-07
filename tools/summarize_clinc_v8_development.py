"""Reanalyze the three qualified, two-task development endpoints; no model/data read."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def events(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def metric(value):
    n, accuracy = value["count"], value["accuracy"]
    if type(n) is not int or n < 1 or not 0 <= accuracy <= 1:
        raise ValueError("invalid accuracy denominator")
    correct = round(n * accuracy)
    if abs(correct / n - accuracy) > 1e-12:
        raise ValueError("accuracy is inconsistent with an integer numerator")
    invalid = value["invalid"]
    if type(invalid) is not int or not 0 <= invalid <= n:
        raise ValueError("invalid-generation count out of range")
    return {"correct":correct,"count":n,"accuracy":accuracy,"invalid":invalid,
            "predictions_sha256":value["predictions_sha256"]}


def one(rows, *, arm, event):
    selected=[r for r in rows if r.get("arm")==arm and r.get("event")==event]
    if len(selected)!=1:
        raise ValueError(f"expected one {arm}/{event}, got {len(selected)}")
    return selected[0]


def summarize(no_sleep, controlled, foundation):
    no_sleep, controlled, foundation = Path(no_sleep),Path(controlled),Path(foundation)
    np, ns = read(no_sleep/"protocol.json"),read(no_sleep/"development-sealed.json")
    cp, cs = read(controlled/"protocol.json"),read(controlled/"development-sealed.json")
    q=read(foundation)
    if q.get("status")!="complete" or q.get("training_only") is not True or q["selection"]["test_data_read"] is not False:
        raise ValueError("qualified foundation must be development-only and complete")
    if not ns.get("complete") or not cs.get("complete") or not ns.get("immutable_base_unchanged") or not cs.get("immutable_base_unchanged"):
        raise ValueError("all endpoint runs must be complete with unchanged backbone")
    if len(ns["runs"])!=1 or len(cs["runs"])!=2:
        raise ValueError("expected one no-Sleep and two controlled arms")
    for field in ("training_manifest_sha256","model_revision","model_assets_sha256"):
        if np[field]!=cp[field] or cp[field]!=q[field]:
            raise ValueError(f"inconsistent shared asset binding: {field}")
    checkpoint=q["arms"]["HOPE"]["checkpoint"]["sha256"]
    if ns["runs"][0]["resume_parent_checkpoint_sha256"]!=checkpoint or cp["parent_task0_checkpoint_sha256"]!=checkpoint:
        raise ValueError("endpoints must share the qualified task-0 asset")
    wake_fields=("seed","initialization_seed","rank","capacity","levels","periods",
                 "epochs_per_task","examples_per_task","batch_size","max_steps_per_task",
                 "base_lr","fast_lr","max_sequence_length")
    if any(np["config"][k]!=cp["config"][k] for k in wake_fields):
        raise ValueError("wake schedules/initialization differ")
    if np["config"]["seed"]!=17 or np["config"]["tasks"]!=[0,1] or cp.get("test_data_read") is not False:
        raise ValueError("this reanalysis is restricted to the development pilot")
    nr,cr=events(no_sleep/"metrics.jsonl"),events(controlled/"metrics.jsonl")
    rows=[]
    for arm, records, receipt in [("HOPE",nr,ns["runs"][0])]+[(a,cr,next(r for r in cs["runs"] if r["arm"]==a)) for a in ("SLEEP_MIX","GOLD_CE_MIX")]:
        event_name="task_end_eval_clean_full" if arm=="HOPE" else "task1_end_eval_clean_full"
        endpoint=one(records,arm=arm,event=event_name)
        if endpoint.get("task_id")!=1 or endpoint.get("seed")!=17:
            raise ValueError("endpoint task/seed mismatch")
        ms={name:metric(endpoint["result"]["metrics"][name]) for name in ("old","current","seen")}
        if ms["old"]["count"]+ms["current"]["count"]!=ms["seen"]["count"] or ms["old"]["correct"]+ms["current"]["correct"]!=ms["seen"]["correct"]:
            raise ValueError("old/current do not partition the seen endpoint")
        work=receipt["work_arm_delta"]
        rows.append({"arm":arm,"seed":17,"metrics":ms,"physical_new_wake_updates":188,
                     "physical_receiver_adamw_steps":receipt.get("physical_sleep_adamw_steps",0),
                     "active_seconds":receipt["active_seconds"],
                     "full_backbone_forward_calls":work["full_backbone_forward_calls"],
                     "work":work,"extra_gold_targets":arm=="GOLD_CE_MIX"})
    old=metric(q["arms"]["HOPE"]["task0_full_clean_development_seen15"]["metrics"]["current"])
    if old["accuracy"]<np["config"]["task0_acquisition_gate"]:
        raise ValueError("old task was not acquired")
    source_paths={"no_sleep_protocol":no_sleep/"protocol.json","no_sleep_seal":no_sleep/"development-sealed.json",
                  "no_sleep_metrics":no_sleep/"metrics.jsonl","controlled_protocol":controlled/"protocol.json",
                  "controlled_seal":controlled/"development-sealed.json","controlled_metrics":controlled/"metrics.jsonl",
                  "corrected_foundation_evaluation":foundation}
    return {"schema":"clinc-v8-two-task-development-summary-v1","complete":True,
            "claim_scope":"one class-order seed, two tasks, exploratory clean development; not final test or full-paper replication",
            "test_payload_read":False,"model_revision":np["model_revision"],
            "training_manifest_sha256":np["training_manifest_sha256"],
            "shared_foundation":{"checkpoint_sha256":checkpoint,"old_development":old,
                                 "compute_note":"shared historical HOPE task0 asset; original asset attempt costs are separate, not charged three times"},
            "rows":rows,"source_files":{k:{"file":p.name,"sha256":sha(p)} for k,p in source_paths.items()},
            "limitations":["CMS-inspired last-MLP adapter, not native HOPE","controlled persistent mix adds expert computation",
                           "gold control uses additional label targets; update/capacity matched, not FLOP or supervision matched",
                           "frozen development proxy uses distance/z0; corrected formula is a separate objective",
                           "all originally reported R1 accuracies invalid; foundation was independently rescored",
                           "class-order seed17 is reused for development; no independent confirmation claim"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-sleep-run",required=True);p.add_argument("--controlled-run",required=True)
    p.add_argument("--foundation-evaluation",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    result=summarize(a.no_sleep_run,a.controlled_run,a.foundation_evaluation)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"complete":True,"rows":len(result["rows"]),"test_payload_read":False}))

if __name__=="__main__":main()
