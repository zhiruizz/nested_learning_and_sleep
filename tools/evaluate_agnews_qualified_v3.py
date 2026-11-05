"""Decode the fresh sealed holdout only after the complete qualified comparison."""
import argparse,time
from pathlib import Path
from datetime import datetime,timezone
import agnews_v3_common as c
import torch
from nested_sleep.lm import execution_state

def verify_seal(run):
 s=c.read(run/'training-sealed.json');assert s['schema']=='agnews-qualified-training-seal-v3' and s['complete']
 assert s['old_all_qualified'] and not s['heldout_read'] and s['records_sha256']==c.DATA_SHA
 n=s['new_steps'];assert n in(512,1024,2048)
 assert len(s['runs'])==15 and {(r['seed'],r['arm'])for r in s['runs']}=={(x,a)for x in c.SEEDS for a in('frozen','sft','cms','cms_sleep','cms_extra_replay')}
 assert len(s['foundations'])==3
 for f in s['foundations']:
  assert c.qualifies(f['train'],f['validation'],'old')
  path=(run/f['checkpoint']).resolve();assert path.is_relative_to(run.resolve())and c.sha(path)==f['checkpoint_sha256']
 for r in s['runs']:
  arm=r['arm'];assert r['status']=='complete'
  assert r['primary_updates']==(0 if arm=='frozen'else n)
  assert r['extra_replay_updates']==(12*(n//64)if arm=='cms_extra_replay'else 0)
  assert r['sleep_adam_updates']==(24*(n//64)if arm=='cms_sleep'else 0)
  assert [x['base_step']for x in r['checkpoints']]==([0,n]if arm=='frozen'else[0,n//4,n//2,3*n//4,n])
  for cp in r['checkpoints']:
   p=(run/cp['path']).resolve();assert p.is_relative_to(run.resolve())and c.sha(p)==cp['sha256']
 assert c.sha(run/'protocol.json')==s['protocol_sha256']
 return s
def main():
 p=argparse.ArgumentParser()
 for name in ('run-dir','records','isolation-dir','model-dir','out-dir'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--resume',action='store_true');a=p.parse_args()
 # No holdout prompt/label file is read before this full checkpoint/budget gate.
 seal=verify_seal(a.run_dir);cfg=c.read(a.run_dir/'protocol.json')['config']
 source=c.sha(__file__);assert cfg['evaluator_sha256']==source
 assert cfg['helper_source_sha256']==c.sha(Path(c.__file__))
 assert c.sha(a.run_dir/'protocol.json')==seal['protocol_sha256']
 pack=c.load_pack(a.records);c.preflight(a.model_dir,pack)
 assert c.sha(a.isolation_dir/'holdout-prompts-sealed.json')==cfg['holdout_prompts_sha256']
 assert c.sha(a.isolation_dir/'holdout-labels-sealed.json')==cfg['holdout_labels_sha256']
 prompts=c.read(a.isolation_dir/'holdout-prompts-sealed.json')['records']
 labels=c.read(a.isolation_dir/'holdout-labels-sealed.json')['labels']
 assert len(prompts)==len(labels)==1024
 assert set(labels)=={r['record_id_sha256']for r in prompts}
 held=[{**r,'class_id':labels[r['record_id_sha256']]}for r in prompts]
 assert all(sum(r['class_id']==x for r in held)==256 for x in range(4))
 assert len({r['record_id_sha256']for r in held})==len({r['prompt_sha256']for r in held})==1024
 assert not({r['record_id_sha256']for r in held}&{r['record_id_sha256']for r in pack['records']})
 assert not({r['prompt_sha256']for r in held}&{r['prompt_sha256']for r in pack['records']})
 if a.out_dir.exists()and not a.resume:raise FileExistsError('evaluation output already exists; explicit read-only resume required')
 a.out_dir.mkdir(parents=True,exist_ok=True);started=time.monotonic();deadline=started+3600
 provenance={'seal_sha256':c.sha(a.run_dir/'training-sealed.json'),'evaluator_sha256':source,
   'holdout_prompts_sha256':cfg['holdout_prompts_sha256'],'holdout_labels_sha256':cfg['holdout_labels_sha256'],
   'helper_sha256':c.sha(Path(c.__file__)),'records_sha256':c.DATA_SHA}
 if a.resume:assert c.read(a.out_dir/'evaluation-provenance.json')==provenance
 else:c.write(a.out_dir/'evaluation-provenance.json',provenance)
 meter=c.engine.WorkMeter();token=c.engine.ACTIVE_METER.set(meter);results={};resumed=0
 try:
  m=c.model(a.model_dir,0)
  for run in seal['runs']:
   key=f'{run["seed"]}/{run["arm"]}';results[key]=[]
   for cp in run['checkpoints']:
    if time.monotonic()>deadline:raise TimeoutError('readonly evaluation3600s cap')
    name=f'seed{run["seed"]}-{run["arm"]}-{cp["base_step"]}.json';dest=a.out_dir/name
    if dest.exists():
     assert a.resume
     receipt=c.read(a.out_dir/(name+'.receipt'))
     assert receipt['sha256']==c.sha(dest)and receipt['provenance']==provenance
     result=c.read(dest);assert result['checkpoint_sha256']==cp['sha256'];resumed+=1
    else:
     state=torch.load(a.run_dir/cp['path'],map_location='cpu',weights_only=False)
     assert state['seed']==run['seed']and state['records_sha256']==c.DATA_SHA
     c.restore(m,state);before=c.fingerprint(m);policy=execution_state(m)
     metrics=c.score(m,held,pack,run['seed'],f'{key}_step{cp["base_step"]}',meter)
     assert c.fingerprint(m)==before and execution_state(m)==policy
     assert c.sha(a.run_dir/cp['path'])==cp['sha256']
     result={'base_step':cp['base_step'],'checkpoint_sha256':cp['sha256'],'metrics':metrics}
     c.write(dest,result);c.write(a.out_dir/(name+'.receipt'),{'sha256':c.sha(dest),'provenance':provenance})
    results[key].append(result)
    print(c.engine.json.dumps({'seed':run['seed'],'arm':run['arm'],'step':cp['base_step'],'status':'scored_or_verified'}),flush=True)
  assert len(results)==15
  c.write(a.out_dir/'evaluation-summary.json',{'schema':'agnews-qualified-evaluation-v3','results':results,
   'provenance':provenance,'all_foundations_qualified':True,'examples_per_class':256,'unique_holdout_examples':1024,
   'resumed_scores':resumed,'work':meter.counts,'seconds':time.monotonic()-started,
   'completed_utc':datetime.now(timezone.utc).isoformat(),'trained_after_holdout_open':False,
   'holdout_scope':'fresh within-AGNews-train, not external/pretraining-independent confirmation'})
 except BaseException as e:
  c.write(a.out_dir/f'failure-{int(time.time())}.json',{'error':str(e),'completed_results':results,'work':meter.counts,'training_resumed':False});raise
 finally:c.engine.ACTIVE_METER.reset(token)
if __name__=='__main__':main()
