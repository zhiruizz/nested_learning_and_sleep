"""Four real-3B DEV transactions and mixed-policy restore; never official test."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,random,sys,time
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import run_x870e_synthetic_dev_v2 as shared
from nested_sleep.lm import execution_state,load_execution_state


def main():
 p=argparse.ArgumentParser()
 for name in ('records','foundation','model-dir','out-dir'):p.add_argument('--'+name,type=Path,required=True)
 args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();clock=shared.Clock(start+1200)
 assert shared.sha256(args.records)=='7b5169daadcf7d4ccaefc9d0bb79f1a4eef92fc275fae0c3a7d593e06b7e9e59'
 pack=json.loads(args.records.read_text(encoding='utf-8'));assert pack['test_opened']is False
 code=pack['code_maps']['1907']
 rows=[{**r,'marker':code['marker_token_id'],'label':code['class_to_token_ids'][r['class_id']]}for r in pack['records']]
 cp=torch.load(args.foundation,map_location='cpu',weights_only=False);assert cp['seed']==1907 and cp['candidate']==0
 torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
 torch.manual_seed(1907);torch.cuda.manual_seed_all(1907);np.random.seed(1907);random.seed(1907)
 model=shared.LlamaCMSLM(AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,
  trust_remote_code=False,dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval(),rank=16,capacity=8,levels=2)
 shared.promote_mutable_to_fp32(model);shared.load_snapshot(model,cp['model'])
 controller=shared.CMSController(model,periods=(4,16),base_lr=.005,expert_lr=.01)
 meter=shared.WorkMeter();token=shared.ACTIVE_METER.set(meter)
 events=[];saved=None;wake_restored=False;restore=None;attempts=0
 def checkpoint():return shared.compact_cpu({'model':model.adapter_state_dict(),'execution':execution_state(model),
   'controller':controller.state_dict(),'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
   'numpy':np.random.get_state(),'python':random.getstate()}})
 def restore_wake(s):
  shared.load_snapshot(model,s['model']);load_execution_state(model,s['execution'])
  torch.set_rng_state(s['rng']['torch']);torch.cuda.set_rng_state_all(s['rng']['cuda'])
  np.random.set_state(s['rng']['numpy']);random.setstate(s['rng']['python'])
  return shared.CMSController.restore(model,s['controller'])
 try:
  for step,indices in enumerate(pack['schedules']['1907']['new_primary256'],1):
   clock.check()
   if step==128:saved=checkpoint();torch.save(saved,args.out_dir/'wake127-mixed.pt')
   event=shared.step(model,controller,[rows[i]for i in indices],pack['tokenizer']['pad_token_id'],meter,
      'dev_new_wake',defer_due=step%64==0);attempts+=1
   if step==128:
    reference=(shared.fingerprint(model),shared.tensor_tree_hash(controller.state_dict()),
        shared.tensor_tree_hash(event['proposals']),shared.tensor_tree_hash(execution_state(model)))
    controller=restore_wake(torch.load(args.out_dir/'wake127-mixed.pt',map_location='cpu',weights_only=False))
    event=shared.step(model,controller,[rows[i]for i in indices],pack['tokenizer']['pad_token_id'],meter,
       'dev_wake128_restore_repeat',defer_due=True);attempts+=1
    actual=(shared.fingerprint(model),shared.tensor_tree_hash(controller.state_dict()),
        shared.tensor_tree_hash(event['proposals']),shared.tensor_tree_hash(execution_state(model)))
    assert actual==reference;wake_restored=True
   if step%64:continue
   index=step//64-1;assert controller.counts==[4,16]
   before=(shared.fingerprint(model),shared.tensor_tree_hash(controller.state_dict()),deepcopy(execution_state(model)))
   candidate=shared.isolated_clone(model,share_backbone=True)
   cc=shared.CMSController.restore(candidate,controller.state_dict());cc.apply(candidate,1,event['proposals'][1])
   session=shared.SleepSession(candidate,0,event['proposals'][0],seed=19070000+index,lr=.003,dream_tokens=2,
      share_backbone=True,reject_zero_gradient=True,receiver_mix_fraction=.5)
   assert execution_state(session.teacher)==execution_state(candidate)
   assert session.student.banks()[1].route_override=={'mix':session.slot,'fraction':.5}
   session.student.banks()[1].audit_routes=True
   pool=[rows[i]for i in pack['sleep_prefix_pools']['1907'][index]]
   natural_hits=positions=0
   for i,row in enumerate(pool):
    clock.check();prefix=torch.tensor([row['tokens']],device='cuda')
    meter.stage=f'dev_sleep_event{index+1}'
    if i==0 and index==1:
     restore=shared.alias_restore_fixture(candidate,prefix,session,args.out_dir,meter,code['marker_token_id'])
     # Also inspect the serialized non-tensor execution policy after a real-3B update.
     snap=torch.load(args.out_dir/'dev907-sleep-restore.pt',map_location='cpu',weights_only=False)
     check=shared.SleepSession.restore(candidate,snap,mutable_only=True)
     assert execution_state(check.teacher)==execution_state(session.teacher)
     assert execution_state(check.student)==execution_state(session.student)
     del check
    elif i<16:session.seed_step(prefix,owner=None)
    else:session.lti_step(prefix)
    receiver=session.student.banks()[1]
    n=session.loss_context['prefix_length'];selected=receiver.last_route_audit['selected'][:,n-1:-1]
    positions+=selected.numel();natural_hits+=int((selected==session.slot).sum())
   assert len(session.records)==24 and all(r['grad_norm']>0 for r in session.records)
   session.finish_consolidation();committed=session.commit();cc.consume(0)
   assert before==(shared.fingerprint(model),shared.tensor_tree_hash(controller.state_dict()),execution_state(model))
   assert cc.counts==[0,0] and session.slot==index
   assert execution_state(committed)['routing'][1]=={'mix':index,'fraction':.5}
   events.append({'event':index+1,'receiver_slot':index,'gkd_updates':16,'lti_updates':8,
     'positive_receiver_gradients':True,'natural_receiver_loss_hits':natural_hits,'loss_positions':positions,
     'executed_mix_receiver_coverage':1.0,'postcommit_execution':execution_state(committed),
     'minimum_gradient_norm':min(r['grad_norm']for r in session.records)})
   model,controller=committed,cc
   print(json.dumps(events[-1]),flush=True)
  result={'schema':'agnews-controlled-mix-dev-v2','status':'REAL3B_FOUR_EVENT_MIX_AND_RECOVERY_PASS',
   'dev_seed':1907,'new_wake_updates':256,'new_wake_update_attempts':attempts,'sleep_gkd':64,'sleep_lti':32,
   'ghost_restore_gkd_updates':2,'wake127to128_with_mixed_policy_restored_exact':wake_restored,
   'mixed_sleep_restore':restore,'events':events,'work':meter.counts,'seconds':time.monotonic()-start,
   'source_sha256':shared.sha256(Path(__file__)),'nested_sources':shared.SOURCE_HASHES,
   'prior_fixture_attempt':'output-parent mkdir failed before model load/CUDA forward; zero updates',
   'test_opened':False,'main_started':False,'prior_main_v1_not_resumed':True}
  (args.out_dir/'fixture-result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
  print(json.dumps({'status':result['status'],'seconds':result['seconds']}),flush=True)
 except BaseException as e:
  (args.out_dir/'failure.json').write_text(json.dumps({'error':str(e),'type':type(e).__name__,
     'events_completed':events,'wake_attempts':attempts,'work':meter.counts,'test_opened':False},indent=2),encoding='utf-8')
  raise
 finally:shared.ACTIVE_METER.reset(token)


if __name__=='__main__':main()
