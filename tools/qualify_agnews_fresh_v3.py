"""Train/validation-only global qualification, then new-task budget calibration.

All three seeds use the first common passing budget; heldout files are not inputs.
"""
import argparse,json,time
from pathlib import Path
import agnews_v3_common as c
import torch
from nested_sleep.optim import CMSController

OLD_STAGES=(512,1024,2048,4096);NEW_STAGES=(512,1024,2048)
def main():
 p=argparse.ArgumentParser()
 for n in ('records','model-dir','out-dir'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();pack=c.load_pack(a.records);c.preflight(a.model_dir,pack)
 a.out_dir.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+3600
 protocol={'schema':'agnews-fresh-qualification-protocol-v3','records_sha256':c.DATA_SHA,
  'seeds':c.SEEDS,'capacity':c.CAPACITY,'rank':16,'learning_rates':c.LR,
  'old_global_budget_ladder':OLD_STAGES,'new_sft_global_budget_ladder':NEW_STAGES,
  'gate':{'train_em_min':.80,'validation_em_min':.75,'each_validation_class_em_min':.70},
  'all_seeds_required':True,'heldout_input':False,'source_sha256':c.sha(__file__),
  'helper_sha256':c.sha(Path(c.__file__)),'hard_seconds':3600}
 protocol['nested_source_hashes']=c.engine.SOURCE_HASHES
 protocol['weight_sha256']=c.engine.WEIGHT_SHA256
 protocol['tokenizer_inventory']=pack['tokenizer']['file_inventory']
 c.write(a.out_dir/'protocol.json',protocol)
 meter=c.engine.WorkMeter();token=c.engine.ACTIVE_METER.set(meter)
 old_records={};new_records={};old_target=None;new_target=None
 def check():
  if time.monotonic()>deadline:raise TimeoutError('qualification3600s cap')
 def run_phase(phase,stages,registry):
  previous=0
  for target in stages:
   stage=[]
   for seed in c.SEEDS:
    check();m=c.model(a.model_dir,seed)
    if previous:
     prev=registry[str(seed)];cp=torch.load(a.out_dir/prev['checkpoint']['path'],map_location='cpu',weights_only=False)
     c.restore(m,cp);c.engine.restore_rng(cp['rng']);ctl=CMSController.restore(m,cp['controller'])
    elif phase=='new':
     base=old_records[str(seed)];cp=torch.load(a.out_dir/base['checkpoint']['path'],map_location='cpu',weights_only=False)
     c.restore(m,cp);c.engine.restore_rng(cp['rng']);ctl=CMSController(m,periods=(1,1),base_lr=c.LR[0],expert_lr=c.LR[1])
    else:ctl=CMSController(m,periods=(4,16),base_lr=c.LR[0],expert_lr=c.LR[1])
    schedule=pack['schedules'][str(seed)]['foundation'if phase=='old'else'new']
    for step in range(previous,target):
     check();event=c.engine.supervised_update(m,ctl,c.batch(pack,seed,schedule[step]),
       pack['tokenizer']['pad_token_id'],meter,f'{phase}_seed{seed}_train')
     if (step+1)%256==0:print(json.dumps({'phase':phase,'seed':seed,'step':step+1,'loss':event['loss']}),flush=True)
    assert ctl.counts==[0,0]
    train=c.score(m,c.rows(pack,seed,'train',phase),pack,seed,f'{phase}_seed{seed}_score',meter)
    val=c.score(m,c.rows(pack,seed,'validation',phase),pack,seed,f'{phase}_seed{seed}_score',meter)
    qualified=c.qualifies(train,val,phase)
    filename=f'{phase}-seed-{seed}-step-{target}.pt'
    cpinfo=c.checkpoint(a.out_dir/filename,m,ctl,seed,target,{'phase':phase,'source_sha256':protocol['source_sha256']})
    record={'seed':seed,'step':target,'phase':phase,'checkpoint':cpinfo,'train':train,'validation':val,'qualified':qualified}
    c.write(a.out_dir/f'{phase}-seed-{seed}-step-{target}.json',record)
    registry[str(seed)]=record;stage.append(record)
    print(json.dumps({'phase':phase,'seed':seed,'step':target,'train_em':train[phase][c.EM],
      'validation_em':val[phase][c.EM],'perclass_validation':[val[f'class{x}'][c.EM]for x in ([0,1]if phase=='old'else[2,3])],
      'qualified':qualified}),flush=True)
    del m,ctl;torch.cuda.empty_cache()
   c.write(a.out_dir/f'{phase}-stage-{target}.json',{'all_qualified':all(r['qualified']for r in stage),'records':stage})
   if all(r['qualified']for r in stage):return target
   previous=target
  return None
 try:
  old_target=run_phase('old',OLD_STAGES,old_records)
  if old_target is not None:
   c.write(a.out_dir/'qualified-foundations.json',{'schema':'agnews-qualified-foundations-v3','all_qualified':True,
    'records_sha256':c.DATA_SHA,'model_config':{'rank':16,'capacity':c.CAPACITY,'levels':2},
    'steps':old_target,'learning_rates':c.LR,'records':list(old_records.values()),'heldout_read':False,
    'protocol_sha256':c.sha(a.out_dir/'protocol.json')})
   new_target=run_phase('new',NEW_STAGES,new_records)
  result={'schema':'agnews-qualification-result-v3','old_all_qualified':old_target is not None,
   'new_sft_all_qualified':new_target is not None,'old_steps':old_target,'new_steps':new_target,
   'old_records':old_records,'new_sft_records':new_records,'work':meter.counts,'seconds':time.monotonic()-start,
   'heldout_read':False,'formal_comparison_started':False}
  c.write(a.out_dir/'qualification-result.json',result)
  print(json.dumps({k:v for k,v in result.items()if k not in ('old_records','new_sft_records','work')}),flush=True)
 except BaseException as e:
  c.write(a.out_dir/'failure.json',{'error':str(e),'type':type(e).__name__,'old_records':old_records,
      'new_records':new_records,'work':meter.counts,'heldout_read':False});raise
 finally:c.engine.ACTIVE_METER.reset(token)
if __name__=='__main__':main()
