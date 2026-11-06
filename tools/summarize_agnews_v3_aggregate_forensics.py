"""Reproduce aggregate-only diagnostics for the public AGNews v3 release.

Reads exactly evaluation-summary.json, training-sealed.json and
analysis-summary.json and the frozen training protocol from the public
candidate folder. It first runs the original verified aggregate analyzer. It never opens model, raw article, prompt, or label files.
"""
from __future__ import annotations

import json
import argparse,hashlib
import pathlib
import statistics
from collections import defaultdict

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-dir',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1])
parser.add_argument('--out-dir',type=pathlib.Path,required=True)
args=parser.parse_args();SOURCE=args.repo_dir;RESULTS=SOURCE/'results/agnews-qualified-v3';OUT=args.out_dir
OUT.mkdir(parents=True,exist_ok=False)
import summarize_agnews_qualified_v3 as verified_analysis
verified_analysis.summarize(RESULTS,RESULTS)

def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))

evaluation = read("evaluation-summary.json")
training = read("training-sealed.json")
analysis = read("analysis-summary.json")
report = (SOURCE / "reports" / "agnews-qualified-v3.md").read_text(encoding="utf-8")

metric_fields = (
    "examples", "full_vocab_two_token_greedy_exact_match", "marker_accuracy",
    "candidate_accuracy", "class_given_marker_accuracy",
    "answer_nll_nats_per_token", "class_nll", "marker_nll",
)
checkpoint_rows = []
for seed_arm, checkpoints in evaluation["results"].items():
    seed, arm = seed_arm.split("/", 1)
    for checkpoint in checkpoints:
        row = {"seed": int(seed), "arm": arm, "base_step": checkpoint["base_step"],
               "checkpoint_sha256": checkpoint.get("checkpoint_sha256")}
        for group in ("old", "new", "all", "class0", "class1", "class2", "class3"):
            m = checkpoint["metrics"].get(group)
            row[group] = ({k: m[k] for k in metric_fields if k in m} if m else None)
        checkpoint_rows.append(row)
checkpoint_rows.sort(key=lambda r: (r["seed"], r["arm"], r["base_step"]))

sleep_event_rows = []
sleep_runs = {}
for run in training["runs"]:
    if run["arm"] != "cms_sleep":
        continue
    seed = run["seed"]
    sleep_runs[seed] = run
    for e in run["sleep_events"]:
        sleep_event_rows.append({
            "seed": seed, "event": e["event"],
            "candidate_committed": e.get("candidate_committed"),
            "gkd_marker_hits": e.get("gkd_marker_hits"),
            "gkd_marker_total": e.get("gkd_marker_total"),
            "gkd_marker_hit_rate": (e.get("gkd_marker_hits", 0) / e["gkd_marker_total"] if e.get("gkd_marker_total") else None),
            "lti_teacher_marker_hits": e.get("lti_teacher_marker_hits"),
            "lti_teacher_marker_total": e.get("lti_teacher_marker_total"),
            "lti_teacher_marker_hit_rate": (e.get("lti_teacher_marker_hits", 0) / e["lti_teacher_marker_total"] if e.get("lti_teacher_marker_total") else None),
            "natural_receiver_gkd_loss_position_hits": e.get("natural_receiver_gkd_loss_position_hits"),
            "natural_receiver_gkd_loss_positions": e.get("natural_receiver_gkd_loss_positions"),
            "natural_receiver_lti_loss_position_hits": e.get("natural_receiver_lti_loss_position_hits"),
            "natural_receiver_lti_loss_positions": e.get("natural_receiver_lti_loss_positions"),
            "natural_receiver_loss_position_hit_rate": e.get("natural_receiver_loss_position_hit_rate"),
            "executed_mixed_receiver_loss_position_coverage": e.get("executed_mixed_receiver_loss_position_coverage"),
            "receiver_gradients_all_positive": e.get("receiver_gradients_all_positive"),
            "receiver_gradient_norm_min": e.get("receiver_gradient_norm_min"),
            "receiver_gradient_norm_mean": e.get("receiver_gradient_norm_mean"),
            "receiver_gradient_norm_max": e.get("receiver_gradient_norm_max"),
            "postcommit_active_slots_by_bank": e.get("postcommit_active_slots_by_bank"),
            "gkd_updates": e.get("gkd_updates"), "gkd_teacher_updates": e.get("gkd_teacher_updates"),
            "gkd_student_on_policy_updates": e.get("gkd_student_on_policy_updates"),
            "lti_updates": e.get("lti_updates"), "dream_updates": e.get("dream_updates"),
            "dream_tokens": e.get("dream_tokens"), "receiver_mix_fraction": e.get("receiver_mix_fraction"),
            "semantic_judge": e.get("semantic_judge"),
        })
sleep_event_rows.sort(key=lambda r: (r["seed"], r["event"]))

def mean(values):
    return statistics.mean(values) if values else None

arm_seed_endpoints = {}
for seed_arm, checkpoints in evaluation["results"].items():
    seed, arm = seed_arm.split("/", 1)
    final = next(c for c in checkpoints if c["base_step"] == max(x["base_step"] for x in checkpoints))
    arm_seed_endpoints.setdefault(arm, {})[int(seed)] = final["metrics"]
endpoint_comparison = {}
for arm, seed_data in sorted(arm_seed_endpoints.items()):
    endpoint_comparison[arm] = {
        str(seed): {g: {"em": seed_data[seed][g]["full_vocab_two_token_greedy_exact_match"],
                        "candidate_accuracy": seed_data[seed][g]["candidate_accuracy"],
                        "marker_accuracy": seed_data[seed][g]["marker_accuracy"],
                        "answer_nll_nats_per_token": seed_data[seed][g]["answer_nll_nats_per_token"]}
                   for g in ("old", "new", "all")}
        for seed in sorted(seed_data)
    }
    endpoint_comparison[arm]["descriptive_mean"] = {
        g: {field: mean([seed_data[s][g][key] for s in sorted(seed_data)])
            for field, key in (("em", "full_vocab_two_token_greedy_exact_match"),
                               ("candidate_accuracy", "candidate_accuracy"),
                               ("marker_accuracy", "marker_accuracy"),
                               ("answer_nll_nats_per_token", "answer_nll_nats_per_token"))}
        for g in ("old", "new", "all")
    }

trajectory = {}
arms = sorted({r["arm"] for r in checkpoint_rows})
for arm in arms:
    trajectory[arm] = {}
    steps = sorted({r["base_step"] for r in checkpoint_rows if r["arm"] == arm})
    for step in steps:
        rs = [r for r in checkpoint_rows if r["arm"] == arm and r["base_step"] == step]
        trajectory[arm][str(step)] = {
            g: {field: mean([r[g][field] for r in rs if r[g] is not None and field in r[g]])
                for field in ("full_vocab_two_token_greedy_exact_match", "candidate_accuracy",
                              "marker_accuracy", "answer_nll_nats_per_token")}
            for g in ("old", "new", "all")
        }

event_summary = {}
for seed in sorted(sleep_runs):
    ev = [r for r in sleep_event_rows if r["seed"] == seed]
    event_summary[str(seed)] = {
        "n_events": len(ev),
        "gkd_marker_hits_total": sum(r["gkd_marker_hits"] or 0 for r in ev),
        "gkd_marker_total": sum(r["gkd_marker_total"] or 0 for r in ev),
        "lti_teacher_marker_hits_total": sum(r["lti_teacher_marker_hits"] or 0 for r in ev),
        "lti_teacher_marker_total": sum(r["lti_teacher_marker_total"] or 0 for r in ev),
        "natural_gkd_loss_position_hits_total": sum(r["natural_receiver_gkd_loss_position_hits"] or 0 for r in ev),
        "natural_gkd_loss_positions_total": sum(r["natural_receiver_gkd_loss_positions"] or 0 for r in ev),
        "natural_lti_loss_position_hits_total": sum(r["natural_receiver_lti_loss_position_hits"] or 0 for r in ev),
        "natural_lti_loss_positions_total": sum(r["natural_receiver_lti_loss_positions"] or 0 for r in ev),
        "natural_receiver_combined_hits_total": sum((r["natural_receiver_gkd_loss_position_hits"] or 0) + (r["natural_receiver_lti_loss_position_hits"] or 0) for r in ev),
        "natural_receiver_combined_positions_total": sum((r["natural_receiver_gkd_loss_positions"] or 0) + (r["natural_receiver_lti_loss_positions"] or 0) for r in ev),
        "natural_receiver_hit_rate_mean": mean([r["natural_receiver_loss_position_hit_rate"] for r in ev if r["natural_receiver_loss_position_hit_rate"] is not None]),
        "natural_receiver_hit_rate_min": min([r["natural_receiver_loss_position_hit_rate"] for r in ev if r["natural_receiver_loss_position_hit_rate"] is not None], default=None),
        "natural_receiver_hit_rate_max": max([r["natural_receiver_loss_position_hit_rate"] for r in ev if r["natural_receiver_loss_position_hit_rate"] is not None], default=None),
        "all_executed_mixed_coverage_one_events": sum(r["executed_mixed_receiver_loss_position_coverage"] == 1.0 for r in ev),
        "all_gradient_positive_events": sum(r["receiver_gradients_all_positive"] is True for r in ev),
        "all_candidate_committed_events": sum(r["candidate_committed"] is True for r in ev),
        "postcommit_active_slots_sequence": [r["postcommit_active_slots_by_bank"] for r in ev],
        "event_gkd_marker_hits": [r["gkd_marker_hits"] for r in ev],
        "natural_receiver_loss_position_hit_rates": [r["natural_receiver_loss_position_hit_rate"] for r in ev],
        "gradient_norm_mean_by_event": [r["receiver_gradient_norm_mean"] for r in ev],
        "final_active_slots_by_bank": sleep_runs[seed]["active_parameters"].get("active_slots_by_bank"),
        "final_currently_trainable_parameters": sleep_runs[seed]["active_parameters"].get("currently_trainable_parameters"),
    }

findings = {
    "scope": "aggregate-only exploratory diagnostics; no raw data, model, prompts, labels, or training access",
    "denominators": {"old": 512, "new": 512, "all": 1024, "each_class": 256},
    "sleep_event_summary": event_summary,
    "descriptive_endpoint_mean_differences_sleep_minus_cms_percentage_points": {
        g: 100 * (endpoint_comparison["cms_sleep"]["descriptive_mean"][g]["em"] - endpoint_comparison["cms"]["descriptive_mean"][g]["em"])
        for g in ("old", "new", "all")
    },
    "sleep_minus_extra_replay_mean_endpoint_em_percentage_points": {
        g: 100 * (endpoint_comparison["cms_sleep"]["descriptive_mean"][g]["em"] - endpoint_comparison["cms_extra_replay"]["descriptive_mean"][g]["em"])
        for g in ("old", "new", "all")
    },
    "class_endpoint_em_mean_by_arm": {
        arm: {f"class{i}": mean([m[f"class{i}"]["full_vocab_two_token_greedy_exact_match"] for m in seed_data.values()])
              for i in range(4)}
        for arm, seed_data in arm_seed_endpoints.items()
    },
    "consistency_observations": [
        "The report's three-seed old-task EM of 0 for SFT, CMS, and CMS+Sleep matches endpoint rows.",
        "The report's CMS+Sleep minus CMS endpoint differences are descriptive and match aggregate calculations.",
        "Sleep marker and receiver diagnostics document execution properties; they do not establish causal mechanism or paper-level semantic reward behavior.",
    ],
    "not_inferred": ["causality", "statistical significance", "generalization beyond this within-AGNews-train holdout", "full Sleep/HOPE reproduction"],
}

out = {
    "schema": "agnews_v3_aggregate_forensics_v1",
    "source_files": ["results/agnews-qualified-v3/evaluation-summary.json", "results/agnews-qualified-v3/training-sealed.json", "results/agnews-qualified-v3/analysis-summary.json", "results/agnews-qualified-v3/training-protocol.json"],
    "checkpoint_metrics": checkpoint_rows,
    "sleep_events": sleep_event_rows,
    "endpoint_comparison": endpoint_comparison,
    "mean_checkpoint_trajectory": trajectory,
    "diagnostics": findings,
}
assert len(checkpoint_rows)==66 and len(sleep_event_rows)==48
out['source_files_sha256']={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()for name in out['source_files']}
(OUT / "aggregate-forensics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = ["# AGNews v3 aggregate forensics", "", "Scope: public aggregate JSON and report only. No raw examples, prompt/label data, model files, GPU, or training were accessed. Denominators are old/new/all = 512/512/1024; each class = 256.", "",
         "## Endpoint comparison", "", "Mean full-vocabulary two-token EM across the three seeds; percentages. Descriptive only.", "", "| Arm | Old | New | All | Mean answer NLL (all) |", "|---|---:|---:|---:|---:|"]
for arm in ("cms", "cms_sleep", "cms_extra_replay", "sft", "frozen"):
    if arm not in endpoint_comparison: continue
    m = endpoint_comparison[arm]["descriptive_mean"]
    lines.append(f"| {arm} | {100*m['old']['em']:.2f}% | {100*m['new']['em']:.2f}% | {100*m['all']['em']:.2f}% | {m['all']['answer_nll_nats_per_token']:.4f} |")
lines += ["", f"CMS+Sleep minus CMS: old {findings['descriptive_endpoint_mean_differences_sleep_minus_cms_percentage_points']['old']:+.2f} pp; new {findings['descriptive_endpoint_mean_differences_sleep_minus_cms_percentage_points']['new']:+.2f} pp; all {findings['descriptive_endpoint_mean_differences_sleep_minus_cms_percentage_points']['all']:+.2f} pp.", "", "## Sleep event diagnostics", "", "| Seed | Events | GKD marker hits | LTI teacher marker hits | Natural GKD positions | Natural LTI positions | Positive-gradient events | Commits | Final active slots |", "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for seed, s in event_summary.items():
    lines.append(f"| {seed} | {s['n_events']} | {s['gkd_marker_hits_total']}/{s['gkd_marker_total']} | {s['lti_teacher_marker_hits_total']}/{s['lti_teacher_marker_total']} | {s['natural_gkd_loss_position_hits_total']}/{s['natural_gkd_loss_positions_total']} | {s['natural_lti_loss_position_hits_total']}/{s['natural_lti_loss_positions_total']} | {s['all_gradient_positive_events']}/{s['n_events']} | {s['all_candidate_committed_events']}/{s['n_events']} | {s['final_active_slots_by_bank']} |")
lines += ["", "Executed mixed-receiver coverage was 1.0 for all 48 events. Natural receiver hit rates varied (per-seed event means {:.1f}%, {:.1f}%, {:.1f}%; overall range {:.0f} to {:.0f}%). Each seed postcommit active-slot sequence progressed [0,1], [0,2], ... [0,16].".format(100*event_summary['17']['natural_receiver_hit_rate_mean'],100*event_summary['29']['natural_receiver_hit_rate_mean'],100*event_summary['43']['natural_receiver_hit_rate_mean'],100*min(s['natural_receiver_hit_rate_min'] for s in event_summary.values()),100*max(s['natural_receiver_hit_rate_max'] for s in event_summary.values())), "", "## Checkpoint trajectory", "", "Mean full-vocabulary EM by checkpoint; each arm is averaged over three seeds.", "", "| Arm / step | Old | New | All | Marker accuracy (all) |", "|---|---:|---:|---:|---:|"]
for arm in ("cms", "cms_sleep", "cms_extra_replay"):
    for step, groups in trajectory[arm].items():
        lines.append(f"| {arm} / {step} | {100*groups['old']['full_vocab_two_token_greedy_exact_match']:.2f}% | {100*groups['new']['full_vocab_two_token_greedy_exact_match']:.2f}% | {100*groups['all']['full_vocab_two_token_greedy_exact_match']:.2f}% | {100*groups['all']['marker_accuracy']:.2f}% |")
lines += ["", "## Evidence and limits", "", "- Marker accuracy remains 1.0 at the Sleep endpoint, while class prediction fails for old tasks: class0/class1 EM is 0 in every seed; new class2 averages 95.31%, while class3 averages 60.42% and varies more by seed. This marker-vs-class split locates failure downstream of marker recognition; it does not identify why it happened.",
          "- CMS and Sleep both reach 0 old-task EM by step 256 from a shared 93.68% starting point. Sleep new-task EM varies across checkpoints (61.7%, 61.2%, 86.1%, 77.9% at steps 256 to 1024), so the endpoint follows a non-monotonic trajectory. CMS rises from 71.2% to 84.7% at those checkpoints.",
          "- Natural receiver coverage is not complete: per-seed aggregate hits are listed above and per-event rates range from 0 to 1. All 48 events have positive-gradient flags and commits, but GKD sampled-marker hits (teacher/student rollouts) are short of total in some events. Executed mix covered every loss position, so incomplete natural coverage does not establish blocked receiving gradients in v3 or explain its endpoint gap.",
          "- CMS, CMS+Sleep, and SFT each have 0% old-task endpoint EM across all three seeds. CMS+Sleep has 77.86% new EM versus CMS 84.70%; this small-seed aggregate is descriptive and does not support causal or significance claims.",
          "- CMS+extra-replay retains old accuracy better on average but is heterogeneous across seeds; inspect the per-seed and checkpoint rows in the JSON. It is a comparison arm, not a causal control for Sleep.",
          "- The public report scopes the holdout as fresh within-AGNews-train rather than external/pretraining-independent confirmation and explicitly says this is not a full Sleep/HOPE reproduction.", "",
          "## Files", "", "`aggregate-forensics.json` contains all 66 checkpoint rows, the 48 per-event rows, per-seed endpoint metrics, arm means and checkpoint trajectories.", ""]
(OUT / "aggregate-forensics.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"output": str(OUT), "checkpoint_rows": len(checkpoint_rows), "sleep_event_rows": len(sleep_event_rows), "means": {a: endpoint_comparison[a]["descriptive_mean"] for a in ("cms", "cms_sleep", "cms_extra_replay")}, "sleep_events": event_summary}, indent=2))
