"""Exploratory, train/validation-only replay of v3's first Sleep boundary.

No final-holdout path, model downloads, method selection or production edits.
Extra no-grad observations must leave weights, policy and RNG unchanged.
"""
from pathlib import Path
import argparse,time,json
from datetime import datetime,timezone
import agnews_v3_common as c
import torch
from nested_sleep.lm import isolated_clone,execution_state,load_execution_state
from nested_sleep.lifecycle import SleepSession
from nested_sleep.optim import CMSController

def load_recovery(run,seed,arm,seal,protocol):
 path=run/'recovery/attempt-1'/f'seed{seed}-{arm}-64.pt';receipt=c.read(path.with_suffix('.receipt.json'))
 assert receipt['seed']==seed and receipt['arm']==arm and receipt['primary_step']==64
 assert receipt['checkpoint_sha256']==c.sha(path) and receipt['config_sha256']==seal['config_sha256']
 assert receipt['protocol_sha256']==seal['protocol_sha256'] and receipt['source_sha256']==seal['completion_source_sha256']
 state=torch.load(path,map_location='cpu',weights_only=False)
 assert state['seed']==seed and state['step']==64 and state['records_sha256']==c.DATA_SHA
 return state

def independent_greedy_check(model,records,pack,seed,meter):
 subset=[r for cid in range(4)for r in [x for x in records if x['class_id']==cid][:4]]
 before=meter.counts.get(meter.stage,{}).get('forward_calls',0)
 device=next(model.parameters()).device;code=pack['code_maps'][str(seed)];hits=0
 with torch.no_grad():
  for off in range(0,len(subset),8):
   chosen=[{**r,'marker':code['marker_token_id'],'label':code['class_to_token_ids'][r['class_id']]}for r in subset[off:off+8]]
   ids,mask,lens,markers,labels=c.engine.shared.collate(chosen,device,pack['tokenizer']['pad_token_id']);ix=torch.arange(len(chosen),device=device)
   first=model.logits(ids,mask)[ix,lens-1].argmax(-1)
   ids,mask=c.engine.shared.append_markers(ids,mask,lens,first,pack['tokenizer']['pad_token_id'])
   second=model.logits(ids,mask)[ix,lens].argmax(-1)
   hits+=int(((first==markers)&(second==labels)).sum())
 forced=c.score(model,subset,pack,seed,meter.stage,meter)['all'][c.EM]
 assert hits/len(subset)==forced,'literal autoregressive greedy EM differs from frozen scorer'
 calls=meter.counts.get(meter.stage,{}).get('forward_calls',0)-before
 if c.engine.ACTIVE_METER.get()is meter:assert calls==3*((len(subset)+7)//8),'extra model forwards were not metered'
 return {'examples':len(subset),'literal_greedy_exact_matches':hits,'frozen_scorer_EM':forced,'exact_agreement':True,'metered_model_forward_calls':calls}

def observe(model,records,pack,seed,stage,meter,clock):
 clock.check();before=c.fingerprint(model);policy=execution_state(model)
 rng=c.engine.shared.tensor_tree_hash(c.engine.capture_rng());old_stage=meter.stage
 hooks=[];flags=[];stats={}
 for bi,bank in enumerate(model.banks()):
  flags.append((bank,bank.audit_routes));bank.audit_routes=True;cursor=[0]
  def hook(module,args,output,bi=bi,cursor=cursor):
   batch=records[cursor[0]:cursor[0]+output.shape[0]];cursor[0]+=len(batch)
   for ri,row in enumerate(batch):
    group='old'if row['class_id']<2 else'new';item=stats.setdefault(f'bank{bi}/{group}',{'positions':0,'selected_slots':{},'bank_output_l2_sum':0.,'mixed_component_l2_sum':0.})
    positions=torch.tensor([len(row['tokens'])-1,len(row['tokens'])],device=output.device)
    vec=output[ri,positions].float();item['positions']+=2;item['bank_output_l2_sum']+=float(vec.norm(dim=-1).sum())
    audit=module.last_route_audit
    if audit is not None:
     for slot in audit['selected'][ri,positions].cpu().tolist():
      name=str(slot);item['selected_slots'][name]=item['selected_slots'].get(name,0)+1
    override=module.route_override or{}
    if 'mix'in override:
     with torch.no_grad():part=module.slots[override['mix']](args[0][ri,positions]).float()*override['fraction']
     item['mixed_component_l2_sum']+=float(part.norm(dim=-1).sum())
  hooks.append(bank.register_forward_hook(hook))
 try:metrics=c.score(model,records,pack,seed,stage,meter)
 finally:
  for hook in hooks:hook.remove()
  for bank,flag in flags:bank.audit_routes=flag
  meter.stage=old_stage
 meter.stage=stage+'_independent_greedy'
 try:greedy=independent_greedy_check(model,records,pack,seed,meter)
 finally:meter.stage=old_stage
 assert c.fingerprint(model)==before and execution_state(model)==policy
 assert c.engine.shared.tensor_tree_hash(c.engine.capture_rng())==rng
 for item in stats.values():
  item['mean_bank_output_l2']=item.pop('bank_output_l2_sum')/item['positions']
  item['mean_mixed_component_l2']=item.pop('mixed_component_l2_sum')/item['positions']
 clock.check()
 return {'stage':stage,'metrics':metrics,'route_and_component_diagnostics':stats,'independent_autoregressive_check':greedy,'adapter_fingerprint':before,
  'execution_policy':policy,'observation_unchanged_weights_policy_rng':True}

def sampled_target_summary(session,row,kind,loss):
 context=session.loss_context;sequence=context['sequence'];n=len(row['tokens']);p=context['teacher_logp']
 marker_ok=int(sequence[0,n])==row['marker'];sample_ok=int(sequence[0,n+1])==row['label']
 labels=session._diagnostic_labels
 with torch.no_grad():
  q=session.student.logits(sequence)[:,context['prefix_length']-1:-1].float().log_softmax(-1)
  kl=float((p.exp()*(p-q)).sum(-1).mean())
  teacher_class=p[0,-1].exp();student_class=q[0,-1].exp()
  objective_after=kl
  if kind=='lti_reward_plus_kl':
   lp=q.gather(-1,sequence[:,context['prefix_length']:,None]).squeeze(-1).mean()
   objective_after=float(.5*kl-.5*context['reward']*lp)
  return {'phase':'old'if row['class_id']<2 else'new','kind':kind,'sample_marker_correct':marker_ok,
   'sampled_class_correct':sample_ok,'teacher_greedy_class_correct_given_actual_prefix':int(p[0,-1].argmax())==row['label'],
   'teacher_gold_class_probability_given_actual_prefix':float(teacher_class[row['label']]),
   'teacher_old_code_mass_given_actual_prefix':float(teacher_class[labels[:2]].sum()),
   'teacher_new_code_mass_given_actual_prefix':float(teacher_class[labels[2:]].sum()),
   'student_gold_class_probability_after_update_given_actual_prefix':float(student_class[row['label']]),
   'loss_before_update':float(loss.detach()),'same_sample_objective_after_update':objective_after,
   'teacher_student_KL_after_update':kl,'gold_class_probability_is_label_diagnostic_not_training_target':True}

class BoundaryTelemetry:
 def __init__(self,pack,seed,validation,pool,meter,clock):
  self.pack,self.seed,self.validation,self.pool,self.meter,self.clock=pack,seed,validation,pool,meter,clock
  self.stages=[];self.samples=[];self.updates=0
 def observe(self,m,name):
  result=observe(m,self.validation,self.pack,self.seed,f'diagnostic_seed{self.seed}_{name}',self.meter,self.clock)
  self.stages.append(result);return result
 def factory(self,*args,**kwargs):
  session=SleepSession(*args,**kwargs);session._diagnostic_labels=self.pack['code_maps'][str(self.seed)]['class_to_token_ids']
  self.observe(session.teacher,'first_teacher_after_slow_before_fast_reset')
  self.observe(session.student,'first_student_after_fast_reset_mix_before_training')
  def observer(s,when,loss,kind,tokens):
   if when!='after':return
   self.clock.check();row=self.pool[self.updates];self.updates+=1;previous=self.meter.stage
   self.meter.stage=f'diagnostic_seed{self.seed}_sample_observation'
   self.samples.append(sampled_target_summary(s,row,kind,loss))
   if self.updates in(16,24):self.observe(s.student,f'first_student_after_{self.updates}_sleep_updates')
   # The engine's following route audit must see its own sequence, not the
   # final validation batch. This no-grad call also leaves state/RNG unchanged.
   with torch.no_grad():s.student.logits(s.loss_context['sequence'])
   self.meter.stage=previous
  session.step_observer=observer
  return session

def main():
 parser=argparse.ArgumentParser()
 for name in('records','run-dir','model-dir','out-dir'):parser.add_argument('--'+name,type=Path,required=True)
 a=parser.parse_args();a.out_dir.mkdir(exist_ok=False,parents=True);start=time.monotonic()
 class Clock:
  def check(self):
   if time.monotonic()-start>900:raise TimeoutError('900-second cooperative elapsed limit; operation-boundary checks')
 clock=Clock();pack=c.load_pack(a.records);protocol=c.read(a.run_dir/'protocol.json');seal=c.read(a.run_dir/'training-sealed.json')
 assert c.sha(a.run_dir/'protocol.json')==seal['protocol_sha256']
 assert protocol['config']['nested_source_sha256']=={k:v['sha256']for k,v in c.engine.ensure_sources().items()}
 c.preflight(a.model_dir,pack)
 for seed in c.SEEDS:
  load_recovery(a.run_dir,seed,'cms_sleep',seal,protocol)
  if seed in(29,43):load_recovery(a.run_dir,seed,'cms',seal,protocol)
 clock.check()
 c.write(a.out_dir/'diagnostic-plan.json',{'schema':'agnews-sleep-forensics-plan-v3','source_sha256':c.sha(__file__),
  'created_utc':datetime.now(timezone.utc).isoformat(),'base_public_commit':'f82c5b9030c3d9dff9b7ee4bf6c5ac6f65d5740b',
  'main_seal_sha256':c.sha(a.run_dir/'training-sealed.json'),'records_sha256':c.DATA_SHA,
  'seeds':list(c.SEEDS),'supervised_replay_steps':192,'sleep_adam_replay_steps':72,'elapsed_limit_seconds':900,
  'time_enforcement':'cooperative checks between operations and after observations; no forced interruption of an in-flight CUDA/kernel/load',
  'trace_steps':[0,8,16,32,63,64],'scoring_role':'validation only; 128/class; all512',
  'final_route_probes':['actual_v3_mix','natural','disable_all_slow','force_oldest_slow','force_latest_slow'],
  'foundation_probe':'disable_fast_bank','heldout_input':False,'exploratory_after_v3_holdout_access':True,
  'no_endpoint_selection_or_confirmatory_claim':True,'automatic_retry':False})
 meter=c.engine.WorkMeter();token=c.engine.ACTIVE_METER.set(meter);results=[];telemetries=[];sgd=adam=0
 try:
  for seed in c.SEEDS:
   clock.check();m=c.model(a.model_dir,seed);foundation=next(f for f in seal['foundations']if f['seed']==seed)
   assert c.sha(a.run_dir/foundation['checkpoint'])==foundation['checkpoint_sha256']
   base=torch.load(a.run_dir/foundation['checkpoint'],map_location='cpu',weights_only=False)
   c.restore(m,base);c.engine.restore_rng(base['rng']);validation=c.rows(pack,seed,'validation');all_rows=c.rows(pack,seed)
   pool=[all_rows[i]for i in pack['sleep_prefix_pools'][str(seed)][0]]
   telemetry=BoundaryTelemetry(pack,seed,validation,pool,meter,clock);telemetries.append(telemetry);stages=telemetry.stages
   telemetry.observe(m,'foundation_step0')
   policy=execution_state(m);m.banks()[0].route_override={'disable':[i for i,s in enumerate(m.banks()[0].slots)if bool(s.active)]}
   telemetry.observe(m,'foundation_disable_fast_bank');load_execution_state(m,policy)
   ctl=CMSController(m,periods=(4,16),base_lr=c.LR[0],expert_lr=c.LR[1])
   for step in range(1,64):
    clock.check();sgd+=1;c.engine.supervised_update(m,ctl,c.batch(pack,seed,pack['schedules'][str(seed)]['new'][step-1]),pack['tokenizer']['pad_token_id'],meter,f'replay_seed{seed}_wake')
    if step in(8,16,32,63):telemetry.observe(m,f'wake_step{step}')
   sgd+=1;event=c.engine.shared.step(m,ctl,[all_rows[i]for i in pack['schedules'][str(seed)]['new'][63]],pack['tokenizer']['pad_token_id'],meter,f'replay_seed{seed}_wake',defer_due=True)
   telemetry.observe(m,'boundary64_after_collect_before_base_commits')
   full=isolated_clone(m,share_backbone=True);fullctl=CMSController.restore(full,ctl.state_dict())
   for level in(1,0):fullctl.apply(full,level,event['proposals'][level])
   telemetry.observe(full,'boundary64_CMS_no_reset_control')
   control_path=a.run_dir/'recovery/attempt-1'/f'seed{seed}-cms-64.pt'
   control_match=None
   if control_path.exists():
    control=load_recovery(a.run_dir,seed,'cms',seal,protocol)
    control_match=(c.engine.shared.tensor_tree_hash(full.adapter_state_dict())==c.engine.shared.tensor_tree_hash(control['model'])and execution_state(full)==control['execution_state'])
    assert control_match,'first64 CMS replay diverged'
   del full,fullctl
   original=c.engine.SleepSession;c.engine.SleepSession=telemetry.factory
   try:m,ctl,event_summary=c.engine.sleep_event(m,ctl,event['proposals'],pool,0,seed,meter,clock,pack['code_maps'][str(seed)]['marker_token_id'])
   finally:c.engine.SleepSession=original
   adam+=telemetry.updates;assert telemetry.updates==24;telemetry.observe(m,'committed_first_sleep_step64')
   expected=load_recovery(a.run_dir,seed,'cms_sleep',seal,protocol)
   matches={'model':c.engine.shared.tensor_tree_hash(m.adapter_state_dict())==c.engine.shared.tensor_tree_hash(expected['model']),
    'controller':c.engine.shared.tensor_tree_hash(ctl.state_dict())==c.engine.shared.tensor_tree_hash(expected['controller']),
    'execution_policy':execution_state(m)==expected['execution_state'],
    'global_RNG':c.engine.shared.tensor_tree_hash(c.engine.capture_rng())==c.engine.shared.tensor_tree_hash(expected['rng'])}
   assert all(matches.values()),f'exploratory replay diverged: {matches}'
   final=next(r for r in seal['runs']if r['seed']==seed and r['arm']=='cms_sleep')['checkpoints'][-1]
   assert c.sha(a.run_dir/final['path'])==final['sha256'];cp=torch.load(a.run_dir/final['path'],map_location='cpu',weights_only=False);c.restore(m,cp)
   actual=execution_state(m);active=[i for i,x in enumerate(m.banks()[1].slots)if bool(x.active)]
   probes={'actual_v3_mix':actual['routing'][1],'natural':None,'disable_all_slow':{'disable':active},'force_oldest_slow':{'force':min(active)},'force_latest_slow':{'force':max(active)}}
   final_probes=[]
   for name,override in probes.items():
    m.banks()[1].route_override=override;final_probes.append(observe(m,validation,pack,seed,f'diagnostic_seed{seed}_endpoint_{name}',meter,clock))
   load_execution_state(m,actual);assert c.fingerprint(m)==final['adapter_fingerprint']
   row={'seed':seed,'first_sleep_exact_replay_matches':matches,'CMS64_control_exact_replay_matches_when_available':control_match,
    'validation_trajectory_and_first_boundary':stages,'first_event_generated_target_diagnostics':telemetry.samples,
    'first_event_original_summary':event_summary,'endpoint_route_probes_validation_only':final_probes}
   results.append(row);c.write(a.out_dir/f'seed-{seed}.json',row)
   print(json.dumps({'seed':seed,'status':'EXACT_FIRST64_REPLAY_PASS','sgd_so_far':sgd,'sleep_adam_so_far':adam}),flush=True)
   del m,ctl;torch.cuda.empty_cache()
  assert sgd==192 and adam==72
  clock.check();c.write(a.out_dir/'diagnostic-result.json',{'schema':'agnews-sleep-forensics-v3','exploratory_not_confirmation':True,
   'heldout_read':False,'validation_examples_per_class':128,'results':results,'work':meter.counts,
   'supervised_replay_steps':sgd,'sleep_adam_replay_steps':adam,'seconds':time.monotonic()-start,
   'source_sha256':c.sha(__file__),'completed_utc':datetime.now(timezone.utc).isoformat(),
   'note':'Route hooks add unmetered low-rank component-norm probes; forward token totals are diagnostic proxies, not FLOPs.'})
 except BaseException as error:
  c.write(a.out_dir/'diagnostic-failure.json',{'error':str(error),'type':type(error).__name__,'completed_seeds':len(results),
   'supervised_replay_update_attempts':sgd,'observed_completed_sleep_adam_updates':sum(t.updates for t in telemetries),'work':meter.counts,'heldout_read':False});raise
 finally:c.engine.ACTIVE_METER.reset(token)
if __name__=='__main__':main()
