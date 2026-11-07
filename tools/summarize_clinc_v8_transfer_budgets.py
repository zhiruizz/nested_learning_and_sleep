"""Read-only aggregation of CLINC v8 corrected transfer-budget runs.
Consumes only protocol/metrics/seal JSON from k=1 and k=10; never reads checkpoints, rows, or test data.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter,defaultdict
from pathlib import Path

EXPECTED_OBJECTIVE='paper-bounded-token-edit-v1'
EXPECTED_ARMS={'SLEEP_MIX','GOLD_CE_MIX'}

def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _norm(x):
 if isinstance(x,(tuple,list)):return [_norm(v) for v in x]
 if isinstance(x,dict):return {k:_norm(v) for k,v in x.items()}
 return x
def _stats(xs):
 xs=[float(x) for x in xs if x is not None]
 if not xs:return {'n':0,'mean':None,'sd':None}
 m=sum(xs)/len(xs)
 return {'n':len(xs),'mean':m,'sd':math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1)) if len(xs)>1 else None}
def _metric(x,phase,key):
 d=x.get(phase)
 if isinstance(d,dict):return d.get(key)
 return d if key=='objective' and isinstance(d,(float,int)) else None

def _load(run_dir,expected_k):
 root=Path(run_dir); files={n:root/n for n in ('protocol.json','metrics.jsonl','development-sealed.json')}
 if any(not p.is_file() for p in files.values()):raise ValueError(f'{root}: incomplete; requires protocol/metrics/seal')
 protocol=json.loads(files['protocol.json'].read_text(encoding='utf-8')); seal=json.loads(files['development-sealed.json'].read_text(encoding='utf-8'))
 rows=[json.loads(line) for line in files['metrics.jsonl'].read_text(encoding='utf-8').splitlines() if line.strip()]
 cfg=protocol.get('config',{}); k=cfg.get('inner_cycles_per_event')
 if protocol.get('schema')!='clinc-v8-controlled-transfer-probe' or k!=expected_k:raise ValueError(f'{root}: budget tag/config mismatch')
 if cfg.get('objective_version')!=EXPECTED_OBJECTIVE:raise ValueError(f'{root}: wrong objective version')
 if cfg.get('sleep_prompt_selector')!='role-balanced-rr-v2-budgeted':raise ValueError(f'{root}: wrong selector tag')
 expected_caps={1:(1800,20000,600),10:(3600,40000,1600)}[expected_k]
 if (cfg.get('probe_max_wall_seconds'),cfg.get('probe_max_forward_calls'),cfg.get('probe_max_update_steps'))!=expected_caps:raise ValueError(f'{root}: budget caps/tag mismatch')
 expected=(2*k,k)
 if (cfg.get('gkd_calls_per_event'),cfg.get('lti_calls_per_event'))!=expected:raise ValueError(f'{root}: triplet budget config mismatch')
 if seal.get('complete') is not True or seal.get('test_data_read') is not False:raise ValueError(f'{root}: missing complete development-only seal')
 completes={r.get('arm'):r.get('receipt') for r in rows if r.get('event')=='run_complete'}
 commits=[r for r in rows if r.get('event')=='controlled_sleep_commit']
 if set(completes)!=EXPECTED_ARMS or len(commits)!=32:raise ValueError(f'{root}: not all 2 arms x 16 events are complete')
 if seal.get('arms_completed')!=2 or set(seal.get('arms_expected',[]))!=EXPECTED_ARMS:raise ValueError(f'{root}: seal arm set mismatch')
 for arm in EXPECTED_ARMS:
  losses=[x for e in commits if e.get('arm')==arm for x in e.get('losses',[])]
  if len(losses)!=48*k:raise ValueError(f'{root}/{arm}: expected {48*k} loss requests, found {len(losses)}')
  rec=completes[arm]
  if rec.get('task1_physical_wake_updates')!=188 or rec.get('sleep_event_count')!=16 or rec.get('physical_sleep_adamw_steps')!=48*k:raise ValueError(f'{root}/{arm}: wake/event/Adam count mismatch')
  if not any(r.get('arm')==arm and r.get('event')=='task1_end_eval_clean_full' for r in rows):raise ValueError(f'{root}/{arm}: missing clean primary endpoint')
 seq_by_arm={}
 for arm in EXPECTED_ARMS:
  arm_events=sorted([e for e in commits if e.get('arm')==arm],key=lambda x:x.get('step',-1)); seq=[]
  for e in arm_events:
   losses=e.get('losses',[])
   if len(losses)!=3*expected_k:raise ValueError(f'{root}/{arm}: per-event request count mismatch')
   roles=Counter(x.get('paired_prompt_selection',{}).get('role') for x in losses)
   expected_roles=[{'old':(3*expected_k)//2,'current':(3*expected_k+1)//2},{'old':(3*expected_k+1)//2,'current':(3*expected_k)//2}]
   if dict(roles) not in expected_roles:raise ValueError(f'{root}/{arm}: per-event roles unbalanced')
   covered=defaultdict(set)
   for x in losses:
    meta=x.get('paired_prompt_selection',{}); covered[meta.get('role')].add(meta.get('class_index'))
   if expected_k==10 and any(covered[r]!=set(range(15)) for r in ('old','current')):raise ValueError(f'{root}/{arm}: k10 event lacks all-class role coverage')
   seq.extend(x.get('prompt_id_sha256') for x in losses)
  seq_by_arm[arm]=seq
 if seq_by_arm['SLEEP_MIX']!=seq_by_arm['GOLD_CE_MIX']:raise ValueError(f'{root}: arms do not use identical request IDs/order')
 return {'root':root,'protocol':protocol,'seal':seal,'rows':rows,'completes':completes,'commits':commits,'files':files}

def _summarize_one(item,k):
 p=item['protocol']; seal=item['seal']; rows=item['rows']; commits=item['commits']; completes=item['completes']; by=defaultdict(list)
 for ev in commits:
  for x in ev.get('losses',[]):
   meta=x.get('paired_prompt_selection',{}); owner=x.get('sample_owner') or ('gold_label' if x.get('uses_gold_intent') else 'unknown')
   by[(ev['arm'],meta.get('role','unknown'),owner,x.get('kind','unknown'))].append(x)
 grouped={}
 for (arm,role,owner,kind),xs in sorted(by.items()):
  key='|'.join((arm,role,owner,kind)); bk=[_metric(x,'same_sample_fit_before','kl') for x in xs]; ak=[_metric(x,'same_sample_fit_after','kl') for x in xs]
  gold=arm=='GOLD_CE_MIX'; beforeobj=[_metric(x,'same_sample_fit_before','objective') for x in xs]; afterobj=[_metric(x,'same_sample_fit_after','objective') for x in xs]
  valid=[int(x['sampled_label_valid']) for x in xs if isinstance(x.get('sampled_label_valid'),bool)]
  match=[int(x['sampled_label_matches_prompt_gold_diagnostic']) for x in xs if isinstance(x.get('sampled_label_matches_prompt_gold_diagnostic'),bool)]
  grouped[key]={'n':len(xs),'same_sample_kl_before':_stats(bk),'same_sample_kl_after':_stats(ak),'same_sample_kl_delta':_stats([a-b for a,b in zip(ak,bk) if a is not None and b is not None]),
   'same_sample_ce_before':_stats(beforeobj if gold else [_metric(x,'same_sample_fit_before','student_ce') for x in xs]),
   'same_sample_ce_after':_stats(afterobj if gold else [_metric(x,'same_sample_fit_after','student_ce') for x in xs]),
   'same_sample_objective_before':_stats(beforeobj),'same_sample_objective_after':_stats(afterobj),
   'sampled_answer_valid_rate':_stats(valid),'sampled_answer_matches_prompt_gold_rate':_stats(match),
   'natural_top1_fresh_receiver_coverage':_stats([x.get('natural_vs_controlled_route_before',{}).get('natural_top1_receiver_coverage') for x in xs]),
   'forced_fresh_branch_coverage':_stats([x.get('natural_vs_controlled_route_before',{}).get('fresh_branch_executed_answer_coverage') for x in xs]),
   'prompt_hashes':[x.get('prompt_id_sha256') for x in xs]}
  if gold:grouped[key]['sampling_diagnostics']='not applicable: no sampled-output fields (n=0; not zero rate)'
 exposure={}; endpoints={}; per_arm={}
 for arm in sorted(EXPECTED_ARMS):
  evs=[e for e in commits if e.get('arm')==arm]; xs=[x for e in evs for x in e.get('losses',[])]; role_count=Counter(); role_classes=defaultdict(set); per_event=[]
  for e in evs:
   evrole=Counter(); evclasses=defaultdict(set); ids=defaultdict(list)
   for x in e.get('losses',[]):
    m=x.get('paired_prompt_selection',{}); role=m.get('role','unknown'); role_count[role]+=1; evrole[role]+=1; role_classes[role].add(m.get('class_index')); evclasses[role].add(m.get('class_index')); ids[role].append(x.get('prompt_id_sha256'))
   per_event.append({'step':e.get('step'),'role_counts':dict(evrole),'class_coverage':{r:len(c) for r,c in evclasses.items()},'unique_prompts_by_role':{r:len(set(v)) for r,v in ids.items()}})
  exposure[arm]={'request_count':len(xs),'role_counts':dict(role_count),'class_indices_covered':{r:sorted(v) for r,v in role_classes.items()},'unique_prompt_hashes':len({x.get('prompt_id_sha256') for x in xs}),'per_event':per_event,
    'gold_known_label_exposures':len(xs) if arm=='GOLD_CE_MIX' else 0,'gold_target_tokens':sum(int(x.get('token_count',0)) for x in xs) if arm=='GOLD_CE_MIX' else 0,
    'prompt_id_sequence_sha256':hashlib.sha256(json.dumps([x.get('prompt_id_sha256') for x in xs],separators=(',',':')).encode()).hexdigest(),
    'gold_label_note':'reuses known training labels; not newly annotated' if arm=='GOLD_CE_MIX' else None}
  er=[r for r in rows if r.get('arm')==arm and r.get('event')=='task1_end_eval_clean_full']
  m=er[-1]['result']['metrics']
  endpoints[arm]={name:{'correct':round(v['accuracy']*v['count']),'count':v['count'],'accuracy':v['accuracy'],'invalid':v['invalid'],'prediction_sha256':v['predictions_sha256']} for name,v in m.items()}
  rec=completes[arm]; w=rec.get('work_arm_delta',{})
  per_arm[arm]={'wake_updates':rec.get('task1_physical_wake_updates'),'physical_sleep_adam_steps':rec.get('physical_sleep_adamw_steps'),'sleep_events':rec.get('sleep_event_count'),'full_backbone_forward_calls':w.get('full_backbone_forward_calls'),'full_backbone_forward_tokens':w.get('full_backbone_forward_tokens'),'executed_input_tokens_including_padding':w.get('executed_input_tokens_including_padding'),'padded_input_tokens':w.get('padded_input_tokens'),'sampled_answer_tokens_including_first_eos':w.get('sampled_answer_tokens_including_first_eos'),'active_seconds':rec.get('active_seconds'),'sleep_request_count':len(xs),'gold_known_label_exposures':exposure[arm]['gold_known_label_exposures'],'gold_target_tokens':exposure[arm]['gold_target_tokens']}
 work=seal.get('work',{})
 teacher=[x for (a,_r,o,_k),vals in by.items() if a=='SLEEP_MIX' and o=='teacher' for x in vals]
 tv=[x['sampled_label_valid'] for x in teacher if isinstance(x.get('sampled_label_valid'),bool)]; tm=[x['sampled_label_matches_prompt_gold_diagnostic'] for x in teacher if isinstance(x.get('sampled_label_matches_prompt_gold_diagnostic'),bool)]
 return {'k':k,'objective_version':p['config'].get('objective_version'),'config_sha256':hashlib.sha256(json.dumps(p['config'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
  'protocol_sha256':_sha(item['files']['protocol.json']),'source_lock_sha256':p.get('source_lock_sha256'),'source_hashes':p.get('source_hashes'),'parent_source_lock_sha256':p.get('parent_r1_source_lock_sha256'),'parent_checkpoint_sha256':p.get('parent_task0_checkpoint_sha256'),
  'model_revision':p.get('model_revision'),'model_assets_sha256':p.get('model_assets_sha256'),'training_manifest_sha256':p.get('training_manifest_sha256'),'source_task0_clean_accuracy':p.get('source_task0_clean_accuracy'),
  'source_asset_config_binding':p.get('source_asset_config_binding'),'seal_sha256':_sha(item['files']['development-sealed.json']),'metrics_sha256':_sha(item['files']['metrics.jsonl']),
  'scope':{'status':'complete','test_data_read':seal.get('test_data_read'),'seed':p.get('config',{}).get('seed'),'per_arm_wake_updates':188,'events_per_arm':16,'requests_per_arm':48*k,'source_arm':p.get('source_arm')},
  'exposure':exposure,'same_sample_groups':grouped,'teacher_owned_sample_quality':{'n':len(tv),'valid_count':sum(tv),'valid_rate':sum(tv)/len(tv) if tv else None,'matches_prompt_gold_count':sum(tm),'matches_prompt_gold_rate':sum(tm)/len(tm) if tm else None},
  'endpoint_clean_full_primary':endpoints,'cost':{'total_full_backbone_forward_calls':work.get('full_backbone_forward_calls'),'total_full_backbone_forward_tokens':work.get('full_backbone_forward_tokens'),'total_executed_input_tokens_including_padding':work.get('executed_input_tokens_including_padding'),'total_padded_input_tokens':work.get('padded_input_tokens'),'total_sampled_answer_tokens':work.get('sampled_answer_tokens_including_first_eos'),'total_update_attempts':work.get('optimizer_or_lifecycle_update_steps'),'total_sleep_events':work.get('sleep_events'),'active_wall_seconds_sum':work.get('active_wall_seconds'),'per_arm':per_arm}}

def summarize(k1_dir,k10_dir,reference_path=None):
 one=_load(k1_dir,1); ten=_load(k10_dir,10)
 p1,p10=one['protocol'],ten['protocol']
 # They must be matched sources/data/context/objective despite distinct budget configs.
 for key, a,b in [('source_lock_sha256',p1.get('source_lock_sha256'),p10.get('source_lock_sha256')),('training_manifest_sha256',p1.get('training_manifest_sha256'),p10.get('training_manifest_sha256')),('parent_task0_checkpoint_sha256',p1.get('parent_task0_checkpoint_sha256'),p10.get('parent_task0_checkpoint_sha256')),('model_revision',p1.get('model_revision'),p10.get('model_revision')),('model_assets_sha256',p1.get('model_assets_sha256'),p10.get('model_assets_sha256'))]:
  if _norm(a)!=_norm(b):raise ValueError(f'cannot pool mismatched runs: {key}')
 if p1['config'].get('objective_version')!=p10['config'].get('objective_version'):raise ValueError('cannot compare runs with different objective tags')
 reference=None
 if reference_path is not None:
  rp=Path(reference_path); ref=json.loads(rp.read_text(encoding='utf-8'))
  if ref.get('complete') is not True or ref.get('test_payload_read') is not False:raise ValueError('no-Sleep reference is incomplete or read test payload')
  if ref.get('training_manifest_sha256')!=p1.get('training_manifest_sha256') or ref.get('shared_foundation',{}).get('checkpoint_sha256')!=p1.get('parent_task0_checkpoint_sha256'):
   raise ValueError('no-Sleep reference has a different manifest or parent foundation')
  base=next((r for r in ref.get('rows',[]) if r.get('arm')=='HOPE' and r.get('seed')==17),None)
  if base is None:raise ValueError('no-Sleep reference lacks seed17 HOPE task1 row')
  reference={'summary_sha256':_sha(rp),'schema':ref.get('schema'),'claim_scope':ref.get('claim_scope'),
   'foundation_sha256':ref['shared_foundation']['checkpoint_sha256'],
   'endpoint':{name:{'correct':v['correct'],'count':v['count'],'accuracy':v['accuracy'],'invalid':v['invalid']} for name,v in base['metrics'].items()},
   'physical_new_wake_updates':base.get('physical_new_wake_updates'),'physical_receiver_adamw_steps':base.get('physical_receiver_adamw_steps'),'active_seconds':base.get('active_seconds'),
   'note':'historical no-Sleep HOPE reference from the same foundation/manifest; reported separately, not pooled with either corrected transfer objective'}
 return {'schema':'clinc-v8-transfer-budget-analysis-v1','status':'complete','budgets':{'k1':_summarize_one(one,1),'k10':_summarize_one(ten,10)},'no_sleep_reference':reference,
  'comparability':{'same_training_manifest':True,'same_parent_task0_checkpoint':True,'same_source_lock':True,'same_model_assets':True,'same_objective_version':EXPECTED_OBJECTIVE,'k1_historical_oldproxy_excluded':True},
  'dataset_cache_limitation':'The replay selector uses four prompts per class as its working set, but the process retains the full local 15k training rows/cache; this is not a physically bounded 4-per-class learner.',
  'interpretation':['Do not pool k=1 and k=10 same-sample metrics: each is a separate fixed objective/budget condition, although both use the corrected objective tag.','Gold CE reuses known old/current training labels and exposes more labeled targets than sampled Sleep; equal event/update counts do not match information or FLOPs.','A single seed is exploratory. Endpoint changes cannot identify a pure optimizer-step effect apart from extra sampled sequences, forwards, and gold-label exposure.','The earlier historical k=1 run used a different edit proxy and is not substituted for the corrected k=1 arm.']}

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--k1-dir',required=True);ap.add_argument('--k10-dir',required=True);ap.add_argument('--no-sleep-reference');ap.add_argument('--out',required=True);a=ap.parse_args(argv)
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);result=summarize(a.k1_dir,a.k10_dir,a.no_sleep_reference);out.write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8');print(json.dumps({'status':result['status'],'out':str(out)}))
if __name__=='__main__':main()
