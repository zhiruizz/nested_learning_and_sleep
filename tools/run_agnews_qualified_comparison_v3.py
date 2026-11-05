"""Frozen comparison starts only from all-seed qualified fresh foundations."""
import argparse,json,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import agnews_v3_common as c
import torch
from nested_sleep.optim import CMSController
ARMS=('frozen','sft','cms','cms_sleep','cms_extra_replay')
def validate_qualification(cert,qualification,cfg):
 assert cert['all_qualified'] and qualification['old_all_qualified'] and qualification['new_sft_all_qualified']
 assert len(cert['records'])==3 and {r['seed']for r in cert['records']}==set(c.SEEDS)
 assert set(qualification['new_sft_records'])=={str(s)for s in c.SEEDS}
 for r in cert['records']:assert c.qualifies(r['train'],r['validation'],'old')
 for r in qualification['new_sft_records'].values():assert c.qualifies(r['train'],r['validation'],'new')
 assert cfg['new_steps']==qualification['new_steps'] and cfg['new_steps']in(512,1024,2048)

def main():
 p=argparse.ArgumentParser()
 for n in ('records','qualification-dir','model-dir','out-dir','config'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();pack=c.load_pack(a.records);cfg=c.read(a.config)
 qualification=c.read(a.qualification_dir/'qualification-result.json')
 cert=c.read(a.qualification_dir/'qualified-foundations.json')
 assert cfg['schema']=='agnews-qualified-comparison-config-v3' and cfg['records_sha256']==c.DATA_SHA
 assert cfg['comparison_source_sha256']==c.sha(__file__) and cfg['helper_source_sha256']==c.sha(Path(c.__file__))
 assert cfg['nested_source_sha256']=={k:v['sha256']for k,v in c.engine.ensure_sources().items()}
 assert cfg['qualification_result_sha256']==c.sha(a.qualification_dir/'qualification-result.json')
 assert qualification['old_all_qualified'] and qualification['new_sft_all_qualified'] and cert['all_qualified']
 validate_qualification(cert,qualification,cfg)
 assert cert['model_config']=={'rank':16,'capacity':32,'levels':2}
 assert cert['records_sha256']==c.DATA_SHA and cert['learning_rates']==list(c.LR)
 n=cfg['new_steps'];assert n==qualification['new_steps'] and n in (512,1024,2048)
 assert cfg['old_steps']==qualification['old_steps']==cert['steps']
 assert cfg['arms']==list(ARMS) and cfg['seeds']==list(c.SEEDS)
 for record in cert['records']:
  assert c.qualifies(record['train'],record['validation'],'old')
  assert record['step']==cert['steps'] and c.sha(a.qualification_dir/record['checkpoint']['path'])==record['checkpoint']['sha256']
 c.preflight(a.model_dir,pack);a.out_dir.mkdir(parents=True,exist_ok=False)
 (a.out_dir/'checkpoints').mkdir();start=time.monotonic();deadline=start+3600
 class Clock:
  def check(self):
   if time.monotonic()>deadline:raise TimeoutError('formal comparison3600s cap; no retry')
 clock=Clock();checkpoints_at=(0,n//4,n//2,3*n//4,n);cfgsha=c.sha(a.config);sourcesha=c.sha(__file__)
 meter=c.engine.WorkMeter();token=c.engine.ACTIVE_METER.set(meter)
 progress={'schema':'agnews-qualified-progress-v3','foundations':[],'runs':[],'status':'running','heldout_read':False}
 c.write(a.out_dir/'protocol.json',{'config':cfg,'config_sha256':cfgsha,'source_sha256':sourcesha,
  'engine_sha256':c.ENGINE_SHA,'helper_sha256':c.sha(Path(c.__file__)),'sources':c.engine.SOURCE_HASHES,
  'qualification_certificate_sha256':c.sha(a.qualification_dir/'qualified-foundations.json'),
  'old_all_qualified':True,'heldout_read':False,'hard_seconds':3600,'automatic_retry':False,
  'started_utc':datetime.now(timezone.utc).isoformat()})
 def save(m,ctl,seed,arm,step,primary,replays):
  r=c.engine.save_checkpoint(a.out_dir,m,ctl,seed,arm,step,c.DATA_SHA,cfgsha,sourcesha,
   pack['schedules'][str(seed)]['new_sha256'],primary_cursor=primary,replay_events=replays)
  return r
 try:
  for seed in c.SEEDS:
   ctl=None
   clock.check();m=c.model(a.model_dir,seed)
   record=next(r for r in cert['records']if r['seed']==seed)
   origin=a.qualification_dir/record['checkpoint']['path'];copy=a.out_dir/'checkpoints'/f'seed-{seed}-foundation.pt'
   shutil.copyfile(origin,copy);assert c.sha(copy)==record['checkpoint']['sha256']
   base=torch.load(copy,map_location='cpu',weights_only=False);c.restore(m,base)
   assert c.fingerprint(m)==record['checkpoint']['adapter_fingerprint']
   progress['foundations'].append({**record,'checkpoint':str(copy.relative_to(a.out_dir)),
     'checkpoint_sha256':c.sha(copy),'adapter_fingerprint':c.fingerprint(m)})
   for arm in ARMS:
    clock.check();c.restore(m,base);c.engine.restore_rng(base['rng']);assert c.fingerprint(m)==record['checkpoint']['adapter_fingerprint']
    ctl=None if arm=='frozen'else CMSController(m,periods=(1,1)if arm=='sft'else(4,16),base_lr=c.LR[0],expert_lr=c.LR[1])
    row={'seed':seed,'arm':arm,'status':'running','primary_updates':0,'extra_replay_updates':0,
      'sleep_adam_updates':0,'sleep_events':[],'checkpoints':[]};progress['runs'].append(row)
    begin=time.monotonic();row['checkpoints'].append(save(m,ctl,seed,arm,0,0,0));replays=0
    if arm!='frozen':
     all_rows=c.rows(pack,seed)
     for step in range(1,n+1):
      clock.check();indices=pack['schedules'][str(seed)]['new'][step-1]
      if arm=='cms_sleep' and step%64==0:
       event=c.engine.shared.step(m,ctl,[all_rows[i]for i in indices],pack['tokenizer']['pad_token_id'],
            meter,f'seed{seed}_{arm}_primary',defer_due=True)
       pool=[all_rows[i]for i in pack['sleep_prefix_pools'][str(seed)][step//64-1]]
       m,ctl,summary=c.engine.sleep_event(m,ctl,event['proposals'],pool,step//64-1,seed,meter,clock,
          pack['code_maps'][str(seed)]['marker_token_id'])
       row['sleep_events'].append(summary);row['sleep_adam_updates']+=24
      else:c.engine.supervised_update(m,ctl,c.batch(pack,seed,indices),pack['tokenizer']['pad_token_id'],meter,f'seed{seed}_{arm}_primary')
      row['primary_updates']=step
      if arm=='cms_extra_replay' and step%64==0:
       pool=pack['sleep_prefix_pools'][str(seed)][step//64-1]
       for b in range(12):
        clock.check();ix=[pool[(b*4+i)%24]for i in range(4)]
        c.engine.supervised_update(m,ctl,c.batch(pack,seed,ix),pack['tokenizer']['pad_token_id'],meter,f'seed{seed}_{arm}_replay')
        row['extra_replay_updates']+=1
       replays+=1
      if step in checkpoints_at:row['checkpoints'].append(save(m,ctl,seed,arm,step,step,replays))
      if step%64==0:print(json.dumps({'seed':seed,'arm':arm,'primary_step':step}),flush=True)
    else:row['checkpoints'].append(save(m,None,seed,arm,n,0,0))
    assert ctl is None or ctl.counts==[0,0]
    if arm=='sft':assert c.fingerprint(m)==qualification['new_sft_records'][str(seed)]['checkpoint']['adapter_fingerprint'],'formal SFT diverged from its fixed validation calibration trajectory'
    row.update(status='complete',seconds=time.monotonic()-begin,
      work={k:v.copy()for k,v in meter.counts.items()if k.startswith(f'seed{seed}_{arm}_')},
      final_adapter_fingerprint=c.fingerprint(m),active_parameters=c.engine.shared.active_parameter_counts(m))
    # Progress snapshots are informational; the exclusive terminal seal is authoritative.
    (a.out_dir/'progress.json').write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8',newline='\n')
   del m;torch.cuda.empty_cache()
  assert len(progress['runs'])==15 and len(progress['foundations'])==3
  assert sum(r['sleep_adam_updates']for r in progress['runs'])==3*(n//64)*24
  seal={'schema':'agnews-qualified-training-seal-v3','complete':True,'new_steps':n,'old_steps':cert['steps'],
   'foundations':progress['foundations'],'runs':progress['runs'],'old_all_qualified':True,'heldout_read':False,
   'records_sha256':c.DATA_SHA,'config_sha256':cfgsha,'source_sha256':sourcesha,
   'protocol_sha256':c.sha(a.out_dir/'protocol.json'),'work_total':meter.counts,'seconds':time.monotonic()-start,
   'qualification_result_sha256':cfg['qualification_result_sha256'],'sealed_utc':datetime.now(timezone.utc).isoformat()}
  c.write(a.out_dir/'training-sealed.json',seal)
  print(json.dumps({'status':'ALL15_QUALIFIED_COMPARISON_SEALED','seconds':seal['seconds']}),flush=True)
 except BaseException as e:
  if 'm'in locals():c.checkpoint(a.out_dir/'failure-state.pt',m,ctl if 'ctl'in locals()else None,seed,step if 'step'in locals()else 0)
  c.write(a.out_dir/'failure.json',{'error':str(e),'progress':progress,'work':meter.counts,'heldout_read':False});raise
 finally:c.engine.ACTIVE_METER.reset(token)
if __name__=='__main__':main()
