"""Summarize a completed zero-update CLINC boundary counterfactual evaluation (JSON only)."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path

STAGES=('teacher_pre','base_only_reconstruction','initial_reset_expanded_reconstruction','optimized_post')
COHORTS=('old','current','seen')

def summarize(path):
 p=Path(path); d=json.loads(p.read_text(encoding='utf-8'))
 if d.get('schema')!='clinc-controlled-boundary-evaluation-v3' or d.get('status')!='complete':raise ValueError('boundary evaluation must be complete schema v3')
 if d.get('test_data_read') is not False or d.get('scoring_scope',{}).get('test_data_read') is not False:raise ValueError('test data must remain unread')
 if d.get('zero_optimizer_steps') is not True or d.get('zero_wake_updates_or_resume') is not True:raise ValueError('boundary analysis requires zero optimizer/wake updates')
 pairs=d.get('paired_events',[])
 if len(pairs)!=32:raise ValueError(f'expected 32 paired events, found {len(pairs)}')
 totals=defaultdict(lambda:{'correct':0,'count':0,'invalid':0,'pairs':0})
 pairrows=[]
 for pair in pairs:
  metrics=pair.get('metrics',{})
  if set(metrics)!=set(STAGES):raise ValueError('pair lacks exact four predeclared contexts')
  if pair.get('post_safe_to_resume_next_wake') not in (True,False):raise ValueError('pair lacks checkpoint safety annotation')
  for stage in STAGES:
   if set(metrics[stage])!=set(COHORTS):raise ValueError('pair cohort metrics missing')
   for cohort in COHORTS:
    m=metrics[stage][cohort]; key=(pair['arm'],pair['sender_level'],stage,cohort); row=totals[key]
    row['correct']+=round(float(m['accuracy'])*int(m['count'])); row['count']+=int(m['count']); row['invalid']+=int(m['invalid']); row['pairs']+=1
  # Exact teacher/pre-base identity should be checkable using prediction digests.
  same={c:metrics['teacher_pre'][c]['predictions_sha256']==metrics['base_only_reconstruction'][c]['predictions_sha256'] for c in COHORTS}
  pairrows.append({'arm':pair['arm'],'event_index':pair['event_index'],'wake_step':pair['wake_step'],'global_wake_tick':pair['global_wake_tick'],'sender_level':pair['sender_level'],'pre_checkpoint_sha256':pair.get('pre_checkpoint_sha256'),'post_checkpoint_sha256':pair.get('post_checkpoint_sha256'),'post_safe_to_resume_next_wake':pair.get('post_safe_to_resume_next_wake'),'other_due_levels_pending':pair.get('post_other_due_levels_pending'),'teacher_equals_base_only_predictions':same,'metrics':metrics})
 aggregate={}
 overall=defaultdict(lambda:{'correct':0,'count':0,'invalid':0,'pairs':0})
 for (arm,level,stage,cohort),v in sorted(totals.items()):
  aggregate[f'{arm}|sender{level}|{stage}|{cohort}']={**v,'accuracy':v['correct']/v['count'] if v['count'] else None}
 for (arm,level,stage,cohort),v in totals.items():
  row=overall[(arm,stage,cohort)];row['correct']+=v['correct'];row['count']+=v['count'];row['invalid']+=v['invalid'];row['pairs']+=v['pairs']
 aggregate_all={f'{arm}|{stage}|{cohort}':{**v,'accuracy':v['correct']/v['count'] if v['count'] else None} for (arm,stage,cohort),v in sorted(overall.items())}
 identity={arm:{'teacher_pre_equals_base_only_by_all_cohort_digests':sum(all(p['teacher_equals_base_only_predictions'].values()) for p in pairrows if p['arm']==arm),'pair_count':sum(p['arm']==arm for p in pairrows)} for arm in sorted({p['arm'] for p in pairrows})}
 return {'schema':'clinc-v8-boundary-counterfactual-summary-v1','status':'complete','source':{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'evaluation_source_sha256':d.get('evaluator_source_sha256'),'config_sha256':d.get('config_sha256'),'probe_protocol_sha256':d.get('probe_protocol_sha256'),'probe_metrics_sha256':d.get('probe_metrics_sha256'),'probe_seal_sha256':d.get('probe_seal_sha256'),'source_lock_sha256':d.get('probe_source_lock_sha256'),'training_manifest_sha256':d.get('training_manifest_sha256'),'parent_task0_checkpoint_sha256':d.get('parent_task0_checkpoint_sha256'),'model_revision':d.get('model_revision'),'model_assets_sha256':d.get('model_assets_sha256')},'scope':{'test_data_read':False,'pairs':len(pairs),'arms':sorted({p['arm'] for p in pairs}),'scoring_scope':d.get('scoring_scope'),'counterfactuals_are_reconstructions_not_training_checkpoints':d.get('counterfactuals_are_reconstructions_not_training_checkpoints'),'zero_optimizer_steps':True,'zero_wake_updates_or_resume':True,'immutable_backbone_unchanged':d.get('immutable_backbone_unchanged')},'cost':{'full_backbone_forward_calls':d['work'].get('full_backbone_forward_calls'),'full_backbone_forward_tokens':d['work'].get('full_backbone_forward_tokens'),'active_wall_seconds':d.get('elapsed_seconds'),'optimizer_or_lifecycle_update_steps':d['work'].get('optimizer_or_lifecycle_update_steps'),'tail_backward_calls':d['work'].get('tail_backward_calls'),'wall_cap_seconds':d.get('wall_cap_seconds'),'forward_cap':d.get('forward_cap')},'aggregate_by_arm_stage_cohort':aggregate_all,'aggregate_by_arm_sender_stage_cohort':aggregate,'teacher_base_only_digest_identity':identity,'pairs':pairrows,'limitations':['The four contexts include two reconstructions and two captured boundary states; reconstructed contexts are not training checkpoints.','Metrics are fixed development probes only and cannot establish a training trajectory or causal retention effect.','This is a single development seed; no official test data was read.']}

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--evaluation',required=True);ap.add_argument('--out',required=True);a=ap.parse_args(argv)
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);result=summarize(a.evaluation);out.write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8');print(json.dumps({'status':result['status'],'pairs':result['scope']['pairs'],'out':str(out)}))
if __name__=='__main__':main()
