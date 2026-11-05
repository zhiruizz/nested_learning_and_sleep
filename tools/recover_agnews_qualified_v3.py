"""Operational continuation of the frozen comparison after process loss.

Completed arms are retained byte-for-byte. No scientific parameter or heldout
boundary changes. New recovery checkpoints accompany every64 primary batches.
"""
import argparse,os,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import agnews_v3_common as c
import run_agnews_qualified_comparison_v3 as definition
import torch
from nested_sleep.optim import CMSController

def main():
 p=argparse.ArgumentParser()
 for name in('run-dir','records','qualification-dir','model-dir','config'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();run=a.run_dir
 assert not(run/'training-sealed.json').exists()and not(run/'failure.json').exists()
 assert not list(run.glob('recovery-failure-*.json')),'engineering failures cannot be retried automatically'
 cfg=c.read(a.config);protocol=c.read(run/'protocol.json');pack=c.load_pack(a.records)
 assert c.sha(run/'protocol.json')=='9b562c1aac3f6a57ff343edacf6c4a3d5f370113d13228f52ab02abb7fa44ed4'
 assert protocol['started_utc']=='2026-11-05T23:37:09.838236+00:00'
 assert protocol['hard_seconds']==3600 and protocol['automatic_retry']is False
 assert protocol['source_sha256']==cfg['comparison_source_sha256'] and protocol['helper_sha256']==cfg['helper_source_sha256']
 assert {k:v['sha256']for k,v in protocol['sources'].items()}==cfg['nested_source_sha256']
 cert=c.read(a.qualification_dir/'qualified-foundations.json');qualification=c.read(a.qualification_dir/'qualification-result.json')
 definition.validate_qualification(cert,qualification,cfg)
 assert c.sha(a.config)==protocol['config_sha256']
 assert cfg['comparison_source_sha256']==c.sha(Path(definition.__file__))
 assert cfg['helper_source_sha256']==c.sha(Path(c.__file__))
 assert cfg['nested_source_sha256']=={k:v['sha256']for k,v in c.engine.ensure_sources().items()}
 assert cfg['qualification_result_sha256']==c.sha(a.qualification_dir/'qualification-result.json')
 c.preflight(a.model_dir,pack);n=cfg['new_steps'];source=c.sha(__file__)
 original_deadline=datetime.fromisoformat(protocol['started_utc']).timestamp()+3600
 class Clock:
  def check(self):
   if time.time()>=original_deadline:raise TimeoutError('original main3600s wall cap reached; not extended')
 clock=Clock();clock.check()
 progress=c.read(run/'recovery-progress.json')if(run/'recovery-progress.json').exists()else c.read(run/'progress.json')
 completed={(r['seed'],r['arm']):r for r in progress['runs']if r['status']=='complete'}
 for r in completed.values():
  for cp in r['checkpoints']:assert c.sha(run/cp['path'])==cp['sha256']
 recovery=run/'recovery';recovery.mkdir(exist_ok=True)
 attempt=1+len(list(recovery.glob('attempt-*')));adir=recovery/f'attempt-{attempt}';adir.mkdir(exist_ok=False)
 amendment={'schema':'agnews-operational-recovery-v3','attempt':attempt,'pid':os.getpid(),
  'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source,
  'frozen_protocol_sha256':c.sha(run/'protocol.json'),'config_sha256':c.sha(a.config),
  'scientific_parameters_changed':False,'completed_arms_preserved':len(completed),'heldout_read':False,
  'reason':'prior process/session disappeared without exception or exit receipt; cause unknown',
  'first_discarded_sleep_prefix_work_bounds':{'supervised_min':64,'supervised_max':256,'sleep_adam_min':24,'sleep_adam_max':96},
  'bound_note':'last printed boundary64 completed; no postzero scientific Sleep checkpoint existed before next256 boundary; exact interrupted work unavailable',
  'original_deadline_utc':datetime.fromtimestamp(original_deadline,timezone.utc).isoformat()}
 c.write(adir/'amendment.json',amendment)
 meter=c.engine.WorkMeter();token=c.engine.ACTIVE_METER.set(meter)
 for r in completed.values():meter.counts.update(r['work'])
 def save_progress():
  (run/'recovery-progress.json').write_text(c.engine.json.dumps(progress,indent=2)+'\n',encoding='utf-8',newline='\n')
 def scientific(m,ctl,seed,arm,step,replay_events):
  return c.engine.save_checkpoint(run,m,ctl,seed,arm,step,c.DATA_SHA,c.sha(a.config),source,
    pack['schedules'][str(seed)]['new_sha256'],primary_cursor=0 if arm=='frozen'else step,replay_events=replay_events)
 try:
  for seed in c.SEEDS:
   if all((seed,x)in completed for x in definition.ARMS):continue
   clock.check();m=c.model(a.model_dir,seed)
   foundation=next(r for r in cert['records']if r['seed']==seed)
   dest=run/'checkpoints'/f'seed-{seed}-foundation.pt'
   if not dest.exists():shutil.copyfile(a.qualification_dir/foundation['checkpoint']['path'],dest)
   assert c.sha(dest)==foundation['checkpoint']['sha256']
   base=torch.load(dest,map_location='cpu',weights_only=False)
   if not any(f['seed']==seed for f in progress['foundations']):
    progress['foundations'].append({**foundation,'checkpoint':str(dest.relative_to(run)),
      'checkpoint_sha256':c.sha(dest),'adapter_fingerprint':foundation['checkpoint']['adapter_fingerprint']})
   for arm in definition.ARMS:
    if(seed,arm)in completed:continue
    clock.check();c.restore(m,base);c.engine.restore_rng(base['rng'])
    assert c.fingerprint(m)==foundation['checkpoint']['adapter_fingerprint']
    ctl=None if arm=='frozen'else CMSController(m,periods=(1,1)if arm=='sft'else(4,16),base_lr=c.LR[0],expert_lr=c.LR[1])
    row={'seed':seed,'arm':arm,'status':'running','primary_updates':0,'extra_replay_updates':0,
      'sleep_adam_updates':0,'sleep_events':[],'checkpoints':[]};previous=0;prior_seconds=0
    receipts=[]
    for receipt_path in recovery.glob(f'attempt-*/seed{seed}-{arm}-*.receipt.json'):
     receipt=c.read(receipt_path);checkpoint_path=run/receipt['checkpoint']
     assert checkpoint_path.resolve().is_relative_to(run.resolve())
     assert receipt['source_sha256']==source and receipt['protocol_sha256']==c.sha(run/'protocol.json')
     assert receipt['seed']==seed and receipt['arm']==arm
     assert receipt['config_sha256']==c.sha(a.config)and c.sha(checkpoint_path)==receipt['checkpoint_sha256']
     receipts.append((receipt['primary_step'],receipt,checkpoint_path))
    if receipts:
     previous,receipt,checkpoint_path=max(receipts,key=lambda x:x[0])
     state=torch.load(checkpoint_path,map_location='cpu',weights_only=False);c.restore(m,state);c.engine.restore_rng(state['rng'])
     assert state['seed']==seed and state['step']==previous and state['records_sha256']==c.DATA_SHA
     assert state['recovery_row']['seed']==seed and state['recovery_row']['arm']==arm
     ctl=CMSController.restore(m,state['controller']);row=state['recovery_row'];prior_seconds=state['arm_seconds']
     meter.counts.update(state['arm_work'])
    else:
     initial=run/'checkpoints'/f'seed-{seed}-arm-{arm}-step-000.pt'
     if initial.exists():
      cp=torch.load(initial,map_location='cpu',weights_only=False);c.restore(m,cp);c.engine.restore_rng(cp['rng'])
      assert c.fingerprint(m)==foundation['checkpoint']['adapter_fingerprint']
      row['checkpoints'].append({'base_step':0,'path':str(initial.relative_to(run)),
       'sha256':c.sha(initial),'adapter_fingerprint':c.fingerprint(m),'primary_cursor':0,'replay_events':0})
     else:row['checkpoints'].append(scientific(m,ctl,seed,arm,0,0))
    progress['runs']=[r for r in progress['runs']if(r['seed'],r['arm'])!=(seed,arm)]+[row]
    begin=time.monotonic();all_rows=c.rows(pack,seed)
    for step in range(previous+1,n+1)if arm!='frozen'else():
     clock.check();indices=pack['schedules'][str(seed)]['new'][step-1]
     if arm=='cms_sleep'and step%64==0:
      event=c.engine.shared.step(m,ctl,[all_rows[i]for i in indices],pack['tokenizer']['pad_token_id'],meter,f'seed{seed}_{arm}_primary',defer_due=True)
      pool=[all_rows[i]for i in pack['sleep_prefix_pools'][str(seed)][step//64-1]]
      m,ctl,summary=c.engine.sleep_event(m,ctl,event['proposals'],pool,step//64-1,seed,meter,clock,pack['code_maps'][str(seed)]['marker_token_id'])
      row['sleep_events'].append(summary);row['sleep_adam_updates']+=24
     else:c.engine.supervised_update(m,ctl,c.batch(pack,seed,indices),pack['tokenizer']['pad_token_id'],meter,f'seed{seed}_{arm}_primary')
     row['primary_updates']=step
     if arm=='cms_extra_replay'and step%64==0:
      pool=pack['sleep_prefix_pools'][str(seed)][step//64-1]
      for b in range(12):
       clock.check();c.engine.supervised_update(m,ctl,c.batch(pack,seed,[pool[(b*4+i)%24]for i in range(4)]),pack['tokenizer']['pad_token_id'],meter,f'seed{seed}_{arm}_replay')
       row['extra_replay_updates']+=1
     if step in(n//4,n//2,3*n//4,n):row['checkpoints'].append(scientific(m,ctl,seed,arm,step,step//64 if arm=='cms_extra_replay'else 0))
     if step%64==0:
      cp=adir/f'seed{seed}-{arm}-{step}.pt'
      c.checkpoint(cp,m,ctl,seed,step,{'recovery_row':row,'arm_seconds':prior_seconds+time.monotonic()-begin,
        'arm_work':{k:v.copy()for k,v in meter.counts.items()if k.startswith(f'seed{seed}_{arm}_')}})
      c.write(cp.with_suffix('.receipt.json'),{'primary_step':step,'checkpoint':str(cp.relative_to(run)),
       'checkpoint_sha256':c.sha(cp),'config_sha256':c.sha(a.config),'source_sha256':source,
       'protocol_sha256':c.sha(run/'protocol.json'),'seed':seed,'arm':arm})
      save_progress();print(c.engine.json.dumps({'seed':seed,'arm':arm,'primary_step':step,'recovery_checkpoint':True}),flush=True)
    if arm=='frozen':row['checkpoints'].append(scientific(m,None,seed,arm,n,0))
    assert ctl is None or ctl.counts==[0,0]
    if arm=='sft':assert c.fingerprint(m)==qualification['new_sft_records'][str(seed)]['checkpoint']['adapter_fingerprint']
    row.update(status='complete',seconds=prior_seconds+time.monotonic()-begin,
      work={k:v.copy()for k,v in meter.counts.items()if k.startswith(f'seed{seed}_{arm}_')},
      final_adapter_fingerprint=c.fingerprint(m),active_parameters=c.engine.shared.active_parameter_counts(m))
    completed[(seed,arm)]=row;save_progress()
   del m;torch.cuda.empty_cache()
  assert len(progress['runs'])==15 and all(r['status']=='complete'for r in progress['runs'])
  assert {(r['seed'],r['arm'])for r in progress['runs']}=={(s,a)for s in c.SEEDS for a in definition.ARMS}
  for r in progress['runs']:
   assert r['primary_updates']==(0 if r['arm']=='frozen'else n)
   assert r['extra_replay_updates']==(12*(n//64)if r['arm']=='cms_extra_replay'else 0)
   assert r['sleep_adam_updates']==(24*(n//64)if r['arm']=='cms_sleep'else 0)
   assert [x['base_step']for x in r['checkpoints']]==([0,n]if r['arm']=='frozen'else[0,n//4,n//2,3*n//4,n])
   for cp in r['checkpoints']:assert c.sha(run/cp['path'])==cp['sha256']
  assert sum(r['primary_updates']+r['extra_replay_updates']for r in progress['runs'])==cfg['main_adaptation_supervised_updates']
  assert sum(r['sleep_adam_updates']for r in progress['runs'])==cfg['main_sleep_adam_updates']
  c.write(run/'training-sealed.json',{'schema':'agnews-qualified-training-seal-v3','complete':True,'new_steps':n,'old_steps':cfg['old_steps'],
   'foundations':progress['foundations'],'runs':progress['runs'],'old_all_qualified':True,'heldout_read':False,
   'records_sha256':c.DATA_SHA,'config_sha256':c.sha(a.config),'source_sha256':cfg['comparison_source_sha256'],
   'protocol_sha256':c.sha(run/'protocol.json'),'work_total':meter.counts,'seconds':time.time()-(original_deadline-3600),
   'qualification_result_sha256':cfg['qualification_result_sha256'],'sealed_utc':datetime.now(timezone.utc).isoformat(),
   'operational_amendments':[{'path':str(p.relative_to(run)),'sha256':c.sha(p)}for p in recovery.glob('attempt-*/amendment.json')],
   'completion_source_sha256':source,'interrupted_duplicate_work_not_in_logical_counts':True})
  print('ALL15_QUALIFIED_COMPARISON_SEALED',flush=True)
 except BaseException as e:
  c.write(run/f'recovery-failure-{attempt}.json',{'error':str(e),'type':type(e).__name__,'progress':progress,'work':meter.counts,'heldout_read':False});raise
 finally:c.engine.ACTIVE_METER.reset(token)
if __name__=='__main__':main()
