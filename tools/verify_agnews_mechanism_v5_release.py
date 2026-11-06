"""Verify frozen v5 diagnostic sources and aggregate bytes in the Git index."""
import hashlib,json,subprocess
BASE='results/agnews-mechanism-v5/'
def blob(p):return subprocess.check_output(['git','show',':'+p])
def sha(p):return hashlib.sha256(blob(p)).hexdigest()
def read(p):return json.loads(blob(p))
def basename(p):return p.replace('\\','/').rsplit('/',1)[-1]
def verify():
 cfg=read('configs/agnews-anchored-comparison-v5.json')
 for key,name in [('driver_source_sha256','run_agnews_anchored_comparison_v5.py'),('wake_helper_sha256','agnews_anchored_wake_v5.py'),('v4_boundary_sha256','agnews_preupdate_sleep_v4.py'),('common_sha256','agnews_v3_common.py'),('engine_sha256','run_agnews_retention_v2.py'),('qualification_driver_sha256','qualify_agnews_anchored_resume_v5.py')]:assert cfg[key]==sha('tools/'+name),key
 for name,h in cfg['nested_source_sha256'].items():
  p='src/nested_sleep/__init__.py'if name=='nested_sleep'else'src/'+name.replace('.','/')+'.py'
  assert sha(p)==h,p
 assert cfg['planned_primary_SGD_updates']==9408 and cfg['planned_Sleep_Adam_updates']==8064
 assert cfg['review_status']=='FROZEN_DEVELOPMENT_AFTER_REVIEW'and not cfg['confirmation_input']and not cfg['confirmation_scored']
 cp,cr=read(BASE+'components/plan.json'),read(BASE+'components/result.json')
 assert cp['source_sha256']==cr['source_sha256']==sha('tools/probe_agnews_components_v5.py')
 assert cr['plan_sha256']==sha(BASE+'components/plan.json')and cr['complete']and cr['optimizer_updates']==0
 assert len(cr['results'])==135 and len(cr['warm_references'])==3 and cr['total_scored_states']==138
 refs={x['seed']:x['metrics']for x in cr['warm_references']}
 v4=read('results/agnews-preupdate-v4/development-sealed.json')
 for row in cr['results']:
  assert row['old_receiver_tensors_unchanged']
  if row['probe']=='warm_function_control':
   for g,m in row['metrics'].items():
    assert m['full_vocab_two_token_greedy_exact_match']==refs[row['seed']][g]['full_vocab_two_token_greedy_exact_match']
    assert abs(m['answer_nll_nats_per_token']-refs[row['seed']][g]['answer_nll_nats_per_token'])<1e-5
  if row['probe']=='actual':
   r=next(x for x in v4['runs']if x['seed']==row['seed']and x['arm']==row['arm'])
   assert row['metrics']==next(x for x in r['validation']if x['primary_step']==row['step'])['metrics']
 assert cfg['component_diagnostic_result_sha256']==sha(BASE+'components/result.json')
 for folder,source in [('interference','tools/diagnose_agnews_interference_v5.py'),('failures/interference',BASE+'failures/interference/source-executed.py')]:
  p=read(BASE+folder+'/plan.json')
  for old,h in p['source_sha256'].items():
   name=basename(old);target=source if name=='diagnose_agnews_interference_v5.py'else'tools/'+name
   assert sha(target)==h,(target,old)
 dr=read(BASE+'interference/diagnostic-result.json')
 assert dr['complete']and dr['updates']==48 and dr['gradient_probe_forward_calls']==66
 assert dr['plan_sha256']==sha(BASE+'interference/plan.json')and not dr['confirmation_scored']
 assert len(dr['results'])==3 and all(all(x['endpoint_match_v4_step16'].values())for x in dr['results'])
 assert cfg['interference_diagnostic_result_sha256']==sha(BASE+'interference/diagnostic-result.json')
 failed=read(BASE+'failures/components/plan.json')
 assert failed['source_sha256']==sha(BASE+'failures/components/source-executed.py')
 assert read(BASE+'failures/interference/failure.json')['attempted_wake_updates']==0
 qp,q=read(BASE+'qualification/plan.json'),read(BASE+'qualification/qualification.json')
 assert q['source_sha256']==qp['source_sha256']==sha('tools/qualify_agnews_anchored_resume_v5.py')
 assert q['anchor_helper_sha256']==qp['anchor_helper_sha256']==cfg['wake_helper_sha256']
 assert q['plan_sha256']==cfg['qualification_plan_sha256']==sha(BASE+'qualification/plan.json')
 assert cfg['qualification_result_sha256']==sha(BASE+'qualification/qualification.json')
 assert q['passed']and q['exact_model_controller_policy_RNG']and q['branches'][0]==q['branches'][1]
 assert q['SGD_updates']==18 and q['Sleep_Adam_updates']==48 and q['scoring']is False
 assert q['private_checkpoint']['step']==14 and q['private_checkpoint']['controller_counts']==[2,14]
 a=read(BASE+'analysis/analysis.json')
 files={'components_corrected':'components/result.json','interference_corrected':'interference/diagnostic-result.json','components_failed':'failures/components/failure.json','interference_failed':'failures/interference/failure.json','qualification':'qualification/qualification.json'}
 for k,p in files.items():assert a['inputs'][k]['sha256']==sha(BASE+p)
 assert not a['confirmation_scored_in_this_summary']
 changed=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines()
 assert not subprocess.check_output(['git','diff','--cached','--name-only','--','src','results/agnews-preupdate-v4','results/agnews-qualified-v3','results/agnews-sleep-forensics-v3'])
 assert not any(p.endswith(('.pt','.bin','.safetensors','.parquet','.arrow','.pkl','.npz','.npy'))for p in changed)
 def clean(v):
  if isinstance(v,dict):
   assert not({'tokens','input_ids','raw_reference','sampled_candidate','teacher_data'}&set(v))
   for x in v.values():clean(x)
  elif isinstance(v,list):
   for x in v:clean(x)
 for p in changed:
  if p.startswith(BASE)and p.endswith('.json'):clean(read(p))
 print('PASS: v5 diagnostic/correction/qualification/config Git hash chains; old evidence/core unchanged; no raw data/weights')
if __name__=='__main__':verify()
