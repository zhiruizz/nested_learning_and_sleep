"""Fixed development comparison after all-seed competent-teacher transfer.

No confirmation input or score-based early stopping/selection. Core v3 remains
unchanged. No recorded failure may be resumed by this entry point.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time,random
import torch
import agnews_v3_common as c
import agnews_preupdate_sleep_v4 as boundary
import qualify_agnews_boundary_transfer_v4 as transfer
from nested_sleep.optim import CMSController
from nested_sleep.lm import execution_state

ARMS=('foundation_CMS_reference','warm_CMS_no_sleep','warm_pre_sleep_P64','warm_pre_sleep_P4','warm_post_sleep_P4','warm_extra_gold_replay_P4')

def event_pools(pack,seed):
 ids={k:[i for i,r in enumerate(pack['records'])if r['role']=='train'and r['class_id']==k]for k in range(4)}
 order=[0,2,3,1,1,3,2,0]*2+[0,2,1,3]*2;pools=[]
 for step in range(4,113,4):
  rng=random.Random(f'nl-sleep-v4-boundary:{seed}:{step}');pool=[rng.choice(ids[k])for k in order]
  assert all(sum(pack['records'][i]['class_id']==k for i in pool)==6 for k in range(4));pools.append(pool)
 return pools

def main():
 p=argparse.ArgumentParser()
 for name in('records','run-dir','warm-dir','model-dir','config','source-lock','out-dir'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();pack=c.load_pack(a.records);cfg=c.read(a.config);lock=c.read(a.source_lock);warm=c.read(a.warm_dir/'transfer-result.json');seal=c.read(a.run_dir/'training-sealed.json')
 assert lock['driver_source_sha256']==c.sha(__file__)and lock['config_sha256']==c.sha(a.config)
 assert lock['warm_result_sha256']==c.sha(a.warm_dir/'transfer-result.json')
 assert cfg['boundary_helper_sha256']==lock['boundary_helper_sha256']==c.sha(Path(boundary.__file__))
 assert cfg['warm_source_sha256']==warm['source_sha256']==c.sha(Path(transfer.__file__))
 assert cfg['nested_source_sha256']=={k:v['sha256']for k,v in c.engine.ensure_sources().items()}
 assert warm['all3_qualified']and warm['phaseB_permitted']and warm['confirmation_scored']is False
 assert len(warm['qualified_models'])==3 and {r['seed']for r in warm['qualified_models']}==set(c.SEEDS)
 assert cfg['phaseB_arms']==list(ARMS)and cfg['phaseB_primary_steps']==112 and cfg['phaseB_checkpoints']==[0,16,32,64,112]
 for row in warm['qualified_models']:
  assert transfer.transfer_passes(warm['teachers'][str(row['seed'])]['validation'],row['train'],row['validation'])
  assert c.sha(a.warm_dir/row['checkpoint']['path'])==row['checkpoint']['sha256']
 c.preflight(a.model_dir,pack);a.out_dir.mkdir(exist_ok=False,parents=True);(a.out_dir/'checkpoints').mkdir();start=time.monotonic()
 def check():
  if time.monotonic()-start>=cfg['phaseB_elapsed_limit_seconds']:raise TimeoutError('development comparison cooperative elapsed limit')
 pools={s:event_pools(pack,s)for s in c.SEEDS}
 c.write(a.out_dir/'protocol.json',{'schema':'agnews-preupdate-comparison-protocol-v4','config':cfg,'config_sha256':c.sha(a.config),
  'source_lock':lock,'source_lock_sha256':c.sha(a.source_lock),'driver_source_sha256':c.sha(__file__),
  'warm_result_sha256':c.sha(a.warm_dir/'transfer-result.json'),'chosen_shared_warm_budget':warm['chosen_common_budget'],
  'pool_hash_by_seed':{str(s):c.engine.shared.tensor_tree_hash(pools[s])for s in c.SEEDS},
  'confirmation_input':False,'confirmation_scored':False,'exploratory_only':True,'automatic_retry':False,
  'created_utc':datetime.now(timezone.utc).isoformat()})
 proto=c.sha(a.out_dir/'protocol.json');meter=c.engine.WorkMeter();token=c.engine.ACTIVE_METER.set(meter);runs=[];sgd_attempts=0;current_event=False
 try:
  for seed in c.SEEDS:
   check();m=c.model(a.model_dir,seed);all_rows=c.rows(pack,seed);validation=c.rows(pack,seed,'validation')
   f=next(r for r in seal['foundations']if r['seed']==seed);assert c.sha(a.run_dir/f['checkpoint'])==f['checkpoint_sha256']
   foundation=torch.load(a.run_dir/f['checkpoint'],map_location='cpu',weights_only=False)
   wr=next(r for r in warm['qualified_models']if r['seed']==seed);warm_cp=torch.load(a.warm_dir/wr['checkpoint']['path'],map_location='cpu',weights_only=False)
   first_warm_metrics=None;first_pre4_student=None
   for arm in ARMS:
    check();base=foundation if arm=='foundation_CMS_reference'else warm_cp;c.restore(m,base);c.engine.restore_rng(base['rng'])
    assert c.fingerprint(m)==(f['adapter_fingerprint']if arm=='foundation_CMS_reference'else wr['checkpoint']['adapter_fingerprint'])
    ctl=CMSController(m,(4,16),cfg['base_lr'],cfg['expert_lr']);row={'seed':seed,'arm':arm,'status':'running','primary_updates':0,'extra_replay_updates':0,'sleep_Adam_updates':0,'events':[],'checkpoints':[],'validation':[]}
    runs.append(row);begin=time.monotonic()
    def save(step):
     check();before=c.fingerprint(m);policy=execution_state(m);rng=c.engine.shared.tensor_tree_hash(c.engine.capture_rng())
     metrics=c.score(m,validation,pack,seed,f'validation_seed{seed}_{arm}_step{step}',meter)
     assert c.fingerprint(m)==before and execution_state(m)==policy and c.engine.shared.tensor_tree_hash(c.engine.capture_rng())==rng
     path=a.out_dir/'checkpoints'/f'seed-{seed}-{arm}-step-{step}.pt'
     info=c.checkpoint(path,m,ctl,seed,step,{'arm':arm,'primary_cursor':step,'extra_replay_updates':row['extra_replay_updates'],
      'sleep_Adam_updates':row['sleep_Adam_updates'],'protocol_sha256':proto,'driver_source_sha256':c.sha(__file__),
      'config_sha256':c.sha(a.config),'new_schedule_sha256':pack['schedules'][str(seed)]['new_sha256']})
     info['path']=str(path.relative_to(a.out_dir));info['primary_step']=step;row['checkpoints'].append(info)
     row['validation'].append({'primary_step':step,'metrics':metrics,'adapter_fingerprint':before,'execution_policy':policy})
     return metrics
    initial=save(0)
    if arm=='foundation_CMS_reference':assert initial['old']==f['validation']['old']
    else:
     assert initial['old']==wr['validation']['old']
     if first_warm_metrics is None:first_warm_metrics=initial
     else:assert initial==first_warm_metrics,'warm-start arms do not share identical step0 metrics'
    for step in range(1,113):
     check();indices=pack['schedules'][str(seed)]['new'][step-1];sleep_now=(arm in('warm_pre_sleep_P4','warm_post_sleep_P4')and step%4==0)or(arm=='warm_pre_sleep_P64'and step%64==0)
     sgd_attempts+=1
     if sleep_now:
      event=c.engine.shared.step(m,ctl,[all_rows[i]for i in indices],pack['tokenizer']['pad_token_id'],meter,f'adapt_seed{seed}_{arm}_primary',defer_due=True)
      pool=[all_rows[i]for i in pools[seed][step//4-1]];current_event=True
      m,ctl,summary=boundary.consolidate(m,ctl,event['proposals'],pool,420000000+seed*1000+step,meter,f'adapt_seed{seed}_{arm}_sleep{step}',apply_fast_before_teacher=arm=='warm_post_sleep_P4')
      current_event=False;summary['primary_boundary']=step;row['events'].append(summary);row['sleep_Adam_updates']+=24
      if step==4 and arm=='warm_pre_sleep_P4':first_pre4_student=summary['initial_student_fingerprint']
      if step==4 and arm=='warm_post_sleep_P4':assert summary['initial_student_fingerprint']==first_pre4_student,'pre/post first boundary initial students differ'
     else:c.engine.supervised_update(m,ctl,c.batch(pack,seed,indices),pack['tokenizer']['pad_token_id'],meter,f'adapt_seed{seed}_{arm}_primary')
     row['primary_updates']=step
     if arm=='warm_extra_gold_replay_P4'and step%4==0:
      pool=pools[seed][step//4-1]
      for batch in range(12):
       check();sgd_attempts+=1;chosen=[pool[(batch*4+i)%24]for i in range(4)]
       c.engine.supervised_update(m,ctl,c.batch(pack,seed,chosen),pack['tokenizer']['pad_token_id'],meter,f'adapt_seed{seed}_{arm}_extra_gold_replay');row['extra_replay_updates']+=1
     if step in cfg['phaseB_checkpoints']:save(step)
     if step%16==0:print(json.dumps({'seed':seed,'arm':arm,'primary_step':step,'sleep_updates':row['sleep_Adam_updates'],'extra_replay_updates':row['extra_replay_updates']}),flush=True)
    assert ctl.counts==[0,0]
    expected_Adam=672 if arm in('warm_pre_sleep_P4','warm_post_sleep_P4')else 24 if arm=='warm_pre_sleep_P64'else 0
    assert row['sleep_Adam_updates']==expected_Adam and row['extra_replay_updates']==(336 if arm=='warm_extra_gold_replay_P4'else 0)
    row.update(status='complete',seconds=time.monotonic()-begin,final_fingerprint=c.fingerprint(m),
      final_active_parameters=c.engine.shared.active_parameter_counts(m),work={k:v.copy()for k,v in meter.counts.items()if k.startswith(f'adapt_seed{seed}_{arm}_')})
    if arm in('warm_pre_sleep_P4','warm_post_sleep_P4'):assert row['final_active_parameters']['active_slots_by_bank']==[0,29]
    (a.out_dir/'progress.json').write_text(json.dumps({'schema':'agnews-preupdate-progress-v4','runs':runs,'confirmation_scored':False},indent=2)+'\n',encoding='utf-8',newline='\n')
   del m,ctl,foundation,warm_cp;torch.cuda.empty_cache()
  assert len(runs)==18 and all(r['status']=='complete'for r in runs)
  assert sum(r['primary_updates']+r['extra_replay_updates']for r in runs)==3024 and sum(r['sleep_Adam_updates']for r in runs)==4104
  check();c.write(a.out_dir/'development-sealed.json',{'schema':'agnews-preupdate-development-seal-v4','complete':True,'runs':runs,
   'protocol_sha256':proto,'source_lock_sha256':c.sha(a.source_lock),'config_sha256':c.sha(a.config),'driver_source_sha256':c.sha(__file__),
   'warm_result_sha256':c.sha(a.warm_dir/'transfer-result.json'),'chosen_shared_warm_budget':warm['chosen_common_budget'],
   'actual_supervised_updates':sgd_attempts,'actual_Sleep_Adam_updates':4104,'work':meter.counts,'seconds':time.monotonic()-start,
   'exploratory_only':True,'confirmation_scored':False,'no_holdout_input':True,'completed_utc':datetime.now(timezone.utc).isoformat()})
  print('ALL18_DEVELOPMENT_ARMS_SEALED',flush=True)
 except BaseException as error:
  if'm'in locals():c.checkpoint(a.out_dir/'failure-state-private.pt',m,ctl if'ctl'in locals()else None,seed,row['primary_updates']if'row'in locals()else 0,{'incomplete_event':current_event})
  c.write(a.out_dir/'failure.json',{'error':str(error),'type':type(error).__name__,'runs':runs,'supervised_update_attempts':sgd_attempts,
   'completed_event_Adam_updates':sum(r['sleep_Adam_updates']for r in runs),'incomplete_event_update_bounds':[0,24]if current_event else[0,0],
   'work':meter.counts,'confirmation_scored':False});raise
 finally:c.engine.ACTIVE_METER.reset(token)
if __name__=='__main__':main()
