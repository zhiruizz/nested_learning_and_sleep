"""Development-only transfer of a competent old teacher before new-task data.

No confirmation/holdout input. All seeds share the first qualifying budget.
Frozen v3 library sources remain unchanged; snapshots are private and mutable-only.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time,random
import torch
import agnews_v3_common as c
from nested_sleep.lifecycle import SleepSession
from nested_sleep.lm import execution_state,load_execution_state

BUDGETS=(24,96,384,1536)

def transfer_passes(teacher,train,validation):
 return c.qualifies(train,validation,'old')and all(validation[g][c.EM]>=teacher[g][c.EM]-.05-1e-12 for g in('old','class0','class1'))

def compact_session(session):
 state=session.state_dict();state['teacher']=session.teacher.adapter_state_dict();state['student']=session.student.adapter_state_dict()
 return c.engine.shared.compact_cpu(state)

def make_pools(pack,seed):
 indices={k:[i for i,r in enumerate(pack['records'])if r['role']=='train'and r['class_id']==k]for k in(0,1)}
 rng=random.Random(f'nl-sleep-v4-old-transfer:{seed}');order=[0,1,1,0]*4+[0,1]*4
 pools=[[rng.choice(indices[k])for k in order]for _ in range(64)]
 assert all(len(p)==24 and all(pack['records'][i]['role']=='train'and pack['records'][i]['class_id']<2 for i in p)for p in pools)
 return pools

def state_hash(session):
 return c.engine.shared.tensor_tree_hash({'model':session.student.adapter_state_dict(),'optimizer':session.optimizer.state_dict(),
  'policy':execution_state(session.student),'generator':session.generator.get_state(),'records':session.records})

def restore_probe(live,saved,prefix,meter):
 before=c.engine.shared.tensor_tree_hash(saved);out=[]
 for _ in range(2):
  branch=SleepSession.restore(live,saved['session'],mutable_only=True);c.engine.restore_rng(saved['rng']);meter.stage='boundary_transfer_restore_ghost'
  branch.seed_step(prefix,owner=None);out.append(state_hash(branch));del branch
 assert out[0]==out[1] and c.engine.shared.tensor_tree_hash(saved)==before
 c.engine.restore_rng(saved['rng']);return {'two_independent_restores_identical':True,'saved_snapshot_not_mutated':True,'extra_Adam_updates':2}

def main():
 p=argparse.ArgumentParser()
 for name in('records','run-dir','model-dir','config','out-dir'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();cfg=c.read(a.config);pack=c.load_pack(a.records);seal=c.read(a.run_dir/'training-sealed.json')
 assert cfg['warm_source_sha256']==c.sha(__file__)and cfg['records_sha256']==c.DATA_SHA
 assert tuple(cfg['warm_budget_ladder'])==BUDGETS and cfg['teacher_max_validation_drop_pp']==5
 assert cfg['old_train_gate']==.8 and cfg['old_validation_gate']==.75 and cfg['old_each_class_validation_gate']==.7
 assert cfg['require_drop_bound_each_old_class']and cfg['confirmation_scoring']is False
 assert c.sha(a.run_dir/'training-sealed.json')==cfg['v3_main_seal_sha256']
 assert cfg['nested_source_sha256']=={k:v['sha256']for k,v in c.engine.ensure_sources().items()}
 c.preflight(a.model_dir,pack);a.out_dir.mkdir(exist_ok=False,parents=True);start=time.monotonic()
 def check():
  if time.monotonic()-start>=cfg['warm_elapsed_limit_seconds']:raise TimeoutError('warm transfer cooperative elapsed limit')
 pools={s:make_pools(pack,s)for s in c.SEEDS}
 protocol={'schema':'agnews-boundary-transfer-protocol-v4','config':cfg,'config_sha256':c.sha(a.config),
  'source_sha256':c.sha(__file__),'v3_main_seal_sha256':c.sha(a.run_dir/'training-sealed.json'),
  'teacher_scope':'qualified old foundation; immutable; identity prospective fast-base weights; no new-task training',
  'created_utc':datetime.now(timezone.utc).isoformat(),'confirmation_input':False,'confirmation_scoring':False,
  'pool_hash_by_seed':{str(s):c.engine.shared.tensor_tree_hash(pools[s])for s in c.SEEDS}}
 c.write(a.out_dir/'protocol.json',protocol);protocol_sha=c.sha(a.out_dir/'protocol.json')
 meter=c.engine.WorkMeter();token=c.engine.ACTIVE_METER.set(meter);registry={};all_stages=[];initial_controls={};teachers={};chosen=None;ghost=None;attempted=0
 try:
  previous=0
  for target in BUDGETS:
   stage=[]
   for seed in c.SEEDS:
    check();m=c.model(a.model_dir,seed);f=next(r for r in seal['foundations']if r['seed']==seed)
    assert c.sha(a.run_dir/f['checkpoint'])==f['checkpoint_sha256']
    foundation=torch.load(a.run_dir/f['checkpoint'],map_location='cpu',weights_only=False);c.restore(m,foundation);c.engine.restore_rng(foundation['rng'])
    original=c.fingerprint(m);policy=execution_state(m);old_train=c.rows(pack,seed,'train','old');old_val=c.rows(pack,seed,'validation','old');rows=c.rows(pack,seed)
    if previous:
     prior=registry[str(seed)];assert c.sha(a.out_dir/prior['session_snapshot']['path'])==prior['session_snapshot']['sha256']
     saved=torch.load(a.out_dir/prior['session_snapshot']['path'],map_location='cpu',weights_only=False)
     assert saved['source_sha256']==c.sha(__file__)and saved['protocol_sha256']==protocol_sha and saved['seed']==seed and saved['budget']==previous
     session=SleepSession.restore(m,saved['session'],mutable_only=True);c.engine.restore_rng(saved['rng'])
    else:
     t_train=c.score(m,old_train,pack,seed,f'teacher_seed{seed}_train',meter);t_val=c.score(m,old_val,pack,seed,f'teacher_seed{seed}_validation',meter)
     assert t_train==f['train']and t_val==f['validation']and c.qualifies(t_train,t_val,'old')
     teachers[str(seed)]={'train':t_train,'validation':t_val,'foundation_sha256':f['checkpoint_sha256'],'fingerprint':original}
     params=dict(m.named_parameters());identity={n:params[n].detach().clone()for n in m.level_descriptors()[0]['base']}
     session=SleepSession(m,0,identity,seed=410000+seed,lr=.003,dream_tokens=2,share_backbone=True,reject_zero_gradient=True,receiver_mix_fraction=.5)
     del params,identity
     initial_controls[str(seed)]={'teacher':teachers[str(seed)],'reset_without_KD_train':c.score(session.student,old_train,pack,seed,f'initial_seed{seed}_train',meter),
      'reset_without_KD_validation':c.score(session.student,old_val,pack,seed,f'initial_seed{seed}_validation',meter),'initial_student_fingerprint':c.fingerprint(session.student)}
    for update in range(previous,target):
     check();cycle,position=divmod(update,24);row=rows[pools[seed][cycle][position]]
     prefix=torch.tensor([row['tokens']],dtype=torch.long,device='cuda');meter.stage=f'warm_seed{seed}_distillation';attempted+=1
     if position<16:session.seed_step(prefix,owner=None)
     else:session.lti_step(prefix)
     if(update+1)%96==0:print(json.dumps({'phase':'old_boundary_transfer','seed':seed,'updates':update+1}),flush=True)
    train=c.score(session.student,old_train,pack,seed,f'warm_seed{seed}_train',meter);val=c.score(session.student,old_val,pack,seed,f'warm_seed{seed}_validation',meter)
    qualified=transfer_passes(teachers[str(seed)]['validation'],train,val)
    student_policy=execution_state(session.student);session.student.banks()[1].route_override={'disable':[session.slot]}
    disabled=c.score(session.student,old_val,pack,seed,f'warm_seed{seed}_receiver_disabled_validation',meter);load_execution_state(session.student,student_policy)
    assert c.fingerprint(m)==original and execution_state(m)==policy and c.fingerprint(session.teacher)==original
    saved={'schema':'agnews-boundary-transfer-session-v4','session':compact_session(session),'rng':c.engine.capture_rng(),
      'seed':seed,'budget':target,'records_sha256':c.DATA_SHA,'source_sha256':c.sha(__file__),'protocol_sha256':protocol_sha}
    name=f'seed-{seed}-budget-{target}-session.pt';torch.save(saved,a.out_dir/name)
    if seed==17 and target==24:
     prefix=torch.tensor([rows[pools[seed][1][0]]['tokens']],device='cuda');ghost=restore_probe(m,saved,prefix,meter)
    record={'seed':seed,'budget':target,'train':train,'validation':val,'receiver_disabled_validation':disabled,'qualified':qualified,
     'teacher_max_drop_old_and_each_class_pp':5,'teacher_fingerprint':original,'student_fingerprint':c.fingerprint(session.student),
     'session_snapshot':{'path':name,'sha256':c.sha(a.out_dir/name)},'gradients_positive':sum(r['grad_norm']>0 for r in session.records),
     'zero_gradient_updates':sum(r['grad_norm']==0 for r in session.records),'near_zero_objective_updates':sum(abs(r['loss'])<=1e-6 for r in session.records)}
    registry[str(seed)]=record;stage.append(record);c.write(a.out_dir/f'seed-{seed}-budget-{target}.json',record)
    print(json.dumps({'phase':'old_boundary_transfer','seed':seed,'budget':target,'train_old_EM':train['old'][c.EM],
     'validation_old_EM':val['old'][c.EM],'teacher_old_EM':teachers[str(seed)]['validation']['old'][c.EM],'qualified':qualified}),flush=True)
    del m,session,saved;torch.cuda.empty_cache()
   all_stages.append({'budget':target,'all_qualified':all(x['qualified']for x in stage),'records':stage})
   c.write(a.out_dir/f'stage-{target}.json',all_stages[-1])
   if all(x['qualified']for x in stage):chosen=target;break
   previous=target
  qualified_models=[]
  if chosen is not None:
   for seed in c.SEEDS:
    check();m=c.model(a.model_dir,seed);f=next(r for r in seal['foundations']if r['seed']==seed)
    base=torch.load(a.run_dir/f['checkpoint'],map_location='cpu',weights_only=False);c.restore(m,base)
    row=registry[str(seed)];assert c.sha(a.out_dir/row['session_snapshot']['path'])==row['session_snapshot']['sha256']
    saved=torch.load(a.out_dir/row['session_snapshot']['path'],map_location='cpu',weights_only=False)
    session=SleepSession.restore(m,saved['session'],mutable_only=True);c.engine.restore_rng(saved['rng']);before=c.fingerprint(session.student)
    session.finish_consolidation();model=session.commit();assert c.fingerprint(model)==before==row['student_fingerprint']
    checkpoint=c.checkpoint(a.out_dir/f'seed-{seed}-qualified-transfer.pt',model,None,seed,chosen,
      {'warm_source_sha256':c.sha(__file__),'protocol_sha256':protocol_sha,'config_sha256':c.sha(a.config),'old_teacher_fingerprint':teachers[str(seed)]['fingerprint']})
    qualified_models.append({'seed':seed,'checkpoint':checkpoint,'warm_updates':chosen,'train':row['train'],'validation':row['validation']})
    del m,session,model,saved;torch.cuda.empty_cache()
  check();c.write(a.out_dir/'transfer-result.json',{'schema':'agnews-boundary-transfer-result-v4','all3_qualified':chosen is not None,'chosen_common_budget':chosen,
   'teachers':teachers,'initial_controls':initial_controls,'stages':all_stages,'qualified_models':qualified_models,
   'successful_or_attempted_main_Adam_updates':attempted,'restore_ghost_test':ghost,'extra_ghost_Adam_updates':2 if ghost else 0,
   'work':meter.counts,'seconds':time.monotonic()-start,'config_sha256':c.sha(a.config),'protocol_sha256':protocol_sha,
   'source_sha256':c.sha(__file__),'exploratory_only':True,'confirmation_scored':False,'phaseB_permitted':chosen is not None,
   'completed_utc':datetime.now(timezone.utc).isoformat()})
  print(json.dumps({'status':'TRANSFER_QUALIFIED'if chosen is not None else'TRANSFER_GATE_FAILED_NO_PHASEB','chosen_budget':chosen,'main_updates':attempted,'ghost_updates':2 if ghost else 0}),flush=True)
 except BaseException as error:
  c.write(a.out_dir/'failure.json',{'type':type(error).__name__,'error':str(error),'main_Adam_attempts':attempted,
   'completed_stages':all_stages,'registry':registry,'work':meter.counts,'confirmation_scored':False});raise
 finally:c.engine.ACTIVE_METER.reset(token)
if __name__=='__main__':main()
