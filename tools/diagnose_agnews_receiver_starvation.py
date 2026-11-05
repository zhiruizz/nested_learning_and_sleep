"""Two isolated train-only forensic branches; no v1 resume/test access."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path('D:/my_paper/NL_Sleep_Public_Candidate')
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import torch
from transformers import AutoModelForCausalLM
import run_x870e_synthetic_dev_v2 as shared
from nested_sleep.lm import LlamaCMSLM,promote_mutable_to_fp32,load_execution_state,isolated_clone
from nested_sleep.lifecycle import SleepSession,fingerprint
from nested_sleep.optim import CMSController
from nested_sleep.accounting import ACTIVE_METER,WorkMeter
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
start=time.monotonic();deadline=start+180
run=ROOT/'outputs/agnews-retention-v1/main-v1-seeds17-29-43'
out=ROOT/'outputs/agnews-retention-v1/receiver-routing-forensics-v1-attempt2'
out.mkdir(exist_ok=False)
pack=json.loads((ROOT/'outputs/agnews-retention-v1/data/agnews-train-roles-v1.json').read_text())
cp=torch.load(run/'failure-state.pt',map_location='cpu',weights_only=False)
assert cp['seed']==17 and cp['arm']=='cms_sleep' and cp['base_step']==127
assert cp['controller']['events']==128 and cp['controller']['counts']==[4,16]
model=LlamaCMSLM(AutoModelForCausalLM.from_pretrained(
 'D:/my_paper/Nested_Learning_and_Sleep/.cache/models/llama32-3b',local_files_only=True,
 trust_remote_code=False,dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval(),rank=16,capacity=8,levels=2)
promote_mutable_to_fp32(model);shared.load_snapshot(model,cp['model']);load_execution_state(model,cp['execution_state'])
controller=CMSController.restore(model,cp['controller'])
assert controller.counts==[4,16]
params=dict(model.named_parameters())
proposals={i:{n:params[n].detach()-a for n,a in pending.items()}for i,pending in enumerate(controller.pending)}
candidate=isolated_clone(model,share_backbone=True)
cctrl=CMSController.restore(candidate,controller.state_dict());cctrl.apply(candidate,1,proposals[1])
prefix=torch.tensor([pack['records'][pack['sleep_prefix_pools']['17'][1][0]]['tokens']],device='cuda')
before=fingerprint(model);ctrl_before=shared.tensor_tree_hash(controller.state_dict())
meter=WorkMeter();token=ACTIVE_METER.set(meter);results={}
try:
 for mode in ('natural','persistent_mix_0.5'):
  if time.monotonic()>deadline: raise TimeoutError('forensic180s cap')
  session=SleepSession(candidate,0,proposals[0],seed=170001,lr=.003,dream_tokens=2,
                       share_backbone=True,reject_zero_gradient=True)
  receiver=session.student.banks()[1];receiver.audit_routes=True
  if mode!='natural':receiver.route_override={'mix':session.slot,'fraction':.5}
  meter.stage=mode
  try:
   session.seed_step(prefix,owner=None)
   result={'status':'passed','record':session.records[-1]}
  except RuntimeError as e:
   result={'status':'rejected','error':str(e),'successful_adam_updates':0}
  audit=receiver.last_route_audit;n=prefix.shape[1]
  result.update({'new_receiver_slot':session.slot,
    'natural_winners_at_two_loss_positions':audit['selected'][0,n-1:n+1].cpu().tolist(),
    'new_slot_natural_token_count':int((audit['selected']==session.slot).sum()),
    'natural_route_score_abs_max':audit['score_abs_max'],
    'executed_route_policy':receiver.route_override,
    'new_slot_parameter_gradient_norms':[None if p.grad is None else float(p.grad.float().norm())for p in(session.new_slot.a,session.new_slot.b)]})
  results[mode]=result
  del session
 assert fingerprint(model)==before and shared.tensor_tree_hash(controller.state_dict())==ctrl_before
 summary={'schema':'agnews-receiver-routing-forensics-v1','private_source_checkpoint_sha256':shared.sha256(run/'failure-state.pt'),
  'main_v1_resumed':False,'official_test_read':False,'branches':results,'work':meter.counts,
  'prior_forensic_attempt':'precondition expected cursor128 but snapshot stores last-completed-loop127; stopped before loading model/CUDA forward, zero updates',
  'original_live_model_controller_unchanged':True,'seconds':time.monotonic()-start,
  'forensic_successful_adam_updates':sum(x['status']=='passed'for x in results.values()),
  'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (out/'diagnostic.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print(json.dumps(summary),flush=True)
finally:ACTIVE_METER.reset(token)
