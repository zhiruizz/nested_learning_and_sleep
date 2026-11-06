"""Predeclared new-only wake stress: apply or skip shared CMS base writes."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import copy,time
import torch
import agnews_v3_common as c
import run_agnews_single_receiver_v7 as v7
import agnews_tail_cache_v7 as cache_helper
from matmul_work_v7 import MatmulWork
from nested_sleep.optim import CMSController
from nested_sleep.lm import execution_state


@contextmanager
def mask_fast(model):
    handles=[s.register_forward_hook(lambda _m,_a,o:torch.zeros_like(o)) for s in model.banks()[0].slots]
    try:yield
    finally:
        for h in handles:h.remove()


def wake_step(model,controller,features,train,indices,pack,seed,write_bases,work):
    controller.prepare_wake(model)
    device=next(model.parameters()).device
    batch={k:torch.cat([features[i][k] for i in indices]).to(device) for k in ('y','residual','lengths')}
    work.phase='forward';z=v7.cached_tail_logits(model,batch,118)
    targets=torch.tensor(v7.gold_answer_targets(pack,seed,train,indices),device=device)
    loss=v7.answer_ce(z,targets)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite stress CE')
    work.phase='backward';loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite stress gradient')
    proposals=controller.collect(model)
    for level in sorted(proposals,reverse=True):
        if write_bases:controller.apply(model,level,proposals[level])
        else:controller.consume(level)
    return {'loss':float(loss.detach()),'gradient_norm':float(norm),'due_levels':sorted(proposals)}


def state_hash(model,controller):
    return c.engine.shared.tensor_tree_hash({'model':model.adapter_state_dict(),'policy':execution_state(model),
        'controller':controller.state_dict(),'rng':c.engine.capture_rng()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('records','model-dir','run-dir','out-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--qualify',action='store_true');p.add_argument('--qualification',type=Path)
    a=p.parse_args();assert not a.out_dir.exists()
    pack=c.load_pack(a.records);c.preflight(a.model_dir,pack)
    seal=c.read(a.run_dir/'development-sealed.json');assert seal['complete'] and len(seal['runs'])==30
    assert seal['protocol_sha256']==c.sha(a.run_dir/'protocol.json')
    protocol=c.read(a.run_dir/'protocol.json')
    assert protocol['driver_sha256']==c.sha(Path(v7.__file__))
    assert protocol['cache_helper_sha256']==c.sha(Path(cache_helper.__file__))
    assert protocol['records_sha256']==c.DATA_SHA
    source=c.sha(Path(__file__))
    if not a.qualify:
        assert a.qualification is not None
        q=c.read(a.qualification)
        assert q['passed'] and q['source_sha256']==source and q['main_seal_sha256']==c.sha(a.run_dir/'development-sealed.json')
    a.out_dir.mkdir(parents=True)
    plan={'schema':'agnews-single-receiver-wake-stress-plan-v7','main_seal_sha256':c.sha(a.run_dir/'development-sealed.json'),
        'main_protocol_sha256':seal['protocol_sha256'],'source_sha256':source,'records_sha256':c.DATA_SHA,
        'selected_main_arms':['split_receiver','split_shared'],'seeds':list(v7.SEEDS),
        'base_writes':[True,False],'steps':128,'evaluation_steps':[0,4,16,64,128],
        'checkpoint_steps':[16,64,128],'periods':[4,16],'base_lr':.02,'expert_lr':.04,
        'qualified':a.qualify,'qualification_sha256':None if a.qualify else c.sha(a.qualification),
        'no_old_targets_in_training':True,'confirmation_scored':False,'whole_cap_seconds':7200,
        'design_note':'same new CE computation and data; numeric gradients may diverge after the first applied base write; skip branch consumes clocks; not a novel third task'}
    c.write(a.out_dir/'plan.json',plan);plan_sha=c.sha(a.out_dir/'plan.json')
    work=MatmulWork();started=time.monotonic();rows=[];attempted=0
    def guard():
        if time.monotonic()-started>7200:raise TimeoutError('wake stress cooperative cap')
    def save(path,m,ctrl,extra):
        state=c.engine.shared.compact_cpu({'model':m.adapter_state_dict(),'execution_state':execution_state(m),
            'controller':ctrl.state_dict(),'rng':c.engine.capture_rng(),**extra})
        torch.save(state,path)
        return {'path':path.name,'sha256':c.sha(path)}
    try:
      with work:
        for seed in ((17,) if a.qualify else v7.SEEDS):
          guard();m=c.model(a.model_dir,seed)
          cache_path=a.run_dir/'cache-private'/f'seed-{seed}.pt'
          source_rows=[r for r in seal['runs'] if r['seed']==seed]
          assert c.sha(cache_path)==source_rows[0]['cache_sha256']
          data=torch.load(cache_path,map_location='cpu',weights_only=False);features=data['student_features']
          train=c.rows(pack,seed,'train');val=c.rows(pack,seed,'validation');allrows=train+val
          sched=v7.class_schedule(train,seed,'new_only_receiver',steps=128)
          for arm in (('split_receiver',) if a.qualify else ('split_receiver','split_shared')):
            parent=next(r for r in source_rows if r['arm']==arm)
            meta=next(x for x in parent['checkpoints'] if x['step']==1024)
            cp,_=v7._load_cp(a.run_dir,meta);assert cp['protocol_sha256']==seal['protocol_sha256']
            first_branch_hashes=[]
            for writes in (True,False):
              guard();c.restore(m,cp);c.engine.restore_rng(cp['rng'])
              ctrl=CMSController(m,periods=(4,16),base_lr=.02,expert_lr=.04)
              ctrl.prepare_wake(m)
              allowed={n for d in m.level_descriptors() for n in d['base']} if writes else set()
              allowed.update(f'{m.level_descriptors()[0]["bank"]}.slots.0.{x}' for x in ('a','b'))
              frozen=v7.model_guards(m,allowed);initial=c.fingerprint(m)
              if writes:common_initial=initial
              else:assert initial==common_initial
              row={'seed':seed,'source_arm':arm,'base_writes':writes,'updates':0,'source_checkpoint_sha256':meta['sha256'],
                   'schedule_sha256':v7.hash_json(sched),'initial_model_sha256':initial,'evaluations':[],
                   'first_three_model_sha256':[],'checkpoints':[],'status':'running'}
              rows.append(row);work.stage=f'{seed}:{arm}:{writes}:training'
              if a.qualify:
                for step in (1,2):
                    guard();wake_step(m,ctrl,features,train,sched[step-1],pack,seed,writes,work);attempted+=1
                pending_path=a.out_dir/f'pending-{writes}-step2.pt'
                pending=save(pending_path,m,ctrl,{'step':2,'seed':seed,'plan_sha256':plan_sha})
                final=[]
                for restore in (False,True):
                  if restore:
                    saved=torch.load(pending_path,map_location='cpu',weights_only=False)
                    c.restore(m,saved);ctrl=CMSController.restore(m,saved['controller']);c.engine.restore_rng(saved['rng'])
                  for step in (3,4,5):
                    guard();wake_step(m,ctrl,features,train,sched[step-1],pack,seed,writes,work);attempted+=1
                  final.append(state_hash(m,ctrl))
                assert final[0]==final[1]
                row.update({'status':'complete','resume_exact_model_controller_policy_rng':True,'pending_checkpoint':pending,'actual_updates':8})
                continue
              work.phase='forward';work.stage=f'{seed}:{arm}:{writes}:score0'
              with torch.no_grad():v0=v7.evaluate_cached(m,features,allrows,list(range(1024,1536)),pack,seed,118)
              assert all(v0[g]['em']==parent['evaluations'][-1]['validation'][g]['em'] for g in v0)
              row['evaluations'].append({'step':0,'validation':v0})
              for step,idx in enumerate(sched,1):
                guard();work.stage=f'{seed}:{arm}:{writes}:training'
                tick=wake_step(m,ctrl,features,train,idx,pack,seed,writes,work);attempted+=1;row['updates']+=1
                if step<=3:row['first_three_model_sha256'].append(c.fingerprint(m))
                if step in (4,16,64,128):
                    guard();work.stage=f'{seed}:{arm}:{writes}:evaluation';work.phase='forward'
                    with torch.no_grad():
                        tr=v7.evaluate_cached(m,features,allrows,list(range(1024)),pack,seed,118)
                        va=v7.evaluate_cached(m,features,allrows,list(range(1024,1536)),pack,seed,118)
                    row['evaluations'].append({'step':step,'train':tr,'validation':va})
                    if step in (16,64,128):
                        path=a.out_dir/f'{seed}-{arm}-{writes}-{step}.pt'
                        row['checkpoints'].append({**save(path,m,ctrl,{'step':step,'seed':seed,'source_arm':arm,
                            'base_writes':writes,'plan_sha256':plan_sha}),'step':step})
                    print({'seed':seed,'arm':arm,'base_writes':writes,'step':step},flush=True)
              if writes:first_branch_hashes=row['first_three_model_sha256']
              else:assert row['first_three_model_sha256']==first_branch_hashes
              assert v7.model_guards(m,allowed)==frozen
              work.phase='forward';work.stage=f'{seed}:{arm}:{writes}:full_parity'
              z=v7._full_fixed_logits(m,val,pack,seed,118);full=v7.score_logits(z,val,pack,seed);del z
              assert all(full[g]['em']==row['evaluations'][-1]['validation'][g]['em'] and
                  abs(full[g]['nll']-row['evaluations'][-1]['validation'][g]['nll'])<=.001 for g in full)
              work.stage=f'{seed}:{arm}:{writes}:fast_output_lesion'
              with mask_fast(m):
                z=v7._full_fixed_logits(m,val,pack,seed,118);fast_masked=v7.score_logits(z,val,pack,seed);del z
              if not writes:
                expected=parent['evaluations'][-1]['validation']
                assert all(fast_masked[g]['em']==expected[g]['em'] and abs(fast_masked[g]['nll']-expected[g]['nll'])<=.001 for g in expected)
              row.update({'status':'complete','slow_and_frozen_scope_guard_passed':True,'full_validation':full,
                          'fast_output_masked_validation':fast_masked,'controller_counts':ctrl.counts})
              c.engine.write_atomic(a.out_dir/'progress.json',{'rows':rows,'complete':False})
          del m,data,features,cp
      guard()
      result={'schema':'agnews-single-receiver-wake-stress-v7','passed':True,'complete':True,'rows':rows,
              'source_sha256':source,'main_seal_sha256':plan['main_seal_sha256'],'plan_sha256':plan_sha,
              'actual_CMS_ticks':attempted,'work':work.state_dict(),'seconds':time.monotonic()-started,
              'qualification':a.qualify,'confirmation_scored':False}
      c.write(a.out_dir/('qualification.json' if a.qualify else 'result.json'),result)
    except BaseException as e:
      c.write(a.out_dir/'failure.json',{'type':type(e).__name__,'error':str(e),'rows':rows,'actual_CMS_ticks':attempted,
          'work':work.state_dict(),'seconds':time.monotonic()-started})
      raise

if __name__=='__main__':main()
