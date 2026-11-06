"""Bounded, read-only-intervention replay from v4-qualified AG News warm starts.

Only the 16 ordinary new-task wake updates are trainable. All observations,
gradient probes, and component ablations are explicitly meter-tagged diagnostics.
No test/holdout input is accepted by this CLI.
"""
from __future__ import annotations

import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import agnews_v3_common as c
from nested_sleep.lm import execution_state, immutable_versions, isolated_clone
from nested_sleep.optim import CMSController

SEEDS=(17,29,43); TRACE=(0,1,4,8,16)
ABLATIONS=('actual','restore_warm_base0','restore_warm_base1','restore_both_bases',
           'disable_fast_bank','force_original_slow_receiver0','disable_original_receiver0')

def tensor_hash(value): return c.engine.shared.tensor_tree_hash(value)

def vector_summary(a,b):
    """Summarize flattened gradient lists without retaining or exposing tensors."""
    if not a or len(a)!=len(b): raise ValueError('gradient vectors must be nonempty and paired')
    dot=sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))
    na=sum(float(x.double().square().sum()) for x in a)**.5
    nb=sum(float(y.double().square().sum()) for y in b)**.5
    return {'dot':dot,'norm_a':na,'norm_b':nb,'cosine':dot/(na*nb) if na and nb else None}

def mutable_state(model):
    return model.adapter_state_dict() if hasattr(model,'adapter_state_dict') else model.state_dict()

def guarded_observation(model, fn, policy_getter, rng_getter):
    """Run a read-only observation and reject any state, gradient, policy or RNG drift."""
    params=list(model.parameters()); state={k:v.detach().cpu().clone() for k,v in mutable_state(model).items()}
    frozen=immutable_versions(model)
    flags=[p.requires_grad for p in params]; grads=[None if p.grad is None else p.grad.detach().clone() for p in params]
    modes=[(m,m.training) for m in model.modules()]; policy=policy_getter(model); rng=rng_getter()
    result=fn()
    assert tensor_hash(state)==tensor_hash(mutable_state(model)),'observation changed mutable model tensors'
    assert immutable_versions(model)==frozen,'observation changed frozen parameter versions'
    assert flags==[p.requires_grad for p in params] and all((g is None and p.grad is None) or
        (g is not None and p.grad is not None and torch.equal(g,p.grad)) for g,p in zip(grads,params)), 'observation changed grad state'
    assert all(m.training==flag for m,flag in modes),'observation changed module mode'
    assert policy_getter(model)==policy,'observation changed execution policy'
    assert tensor_hash(rng_getter())==tensor_hash(rng),'observation changed RNG'
    return result

def balanced_indices(pack,role,classes,per_class):
    out=[]
    for cls in classes:
        got=[i for i,r in enumerate(pack['records']) if r['role']==role and r['class_id']==cls][:per_class]
        if len(got)!=per_class: raise ValueError(f'insufficient {role} rows for class {cls}')
        out.extend(got)
    return out

def named_base(model,level):
    names=model.level_descriptors()[level]['base']; params=dict(model.named_parameters())
    return [(n,params[n]) for n in names]

def active_fast_params(model):
    bank=model.banks()[0]
    return [p for i,slot in enumerate(bank.slots) if bool(slot.active) for p in slot.parameters()]

def absent_group_summary():
    return {'dot':0.0,'norm_a':0.0,'norm_b':0.0,'cosine':None,
            'parameter_count':0,'parameter_l2_norm':0.0,'gradient_requires_grad_scope':[],
            'ordinary_wake_requires_grad_before_probe':[],'probe_scope_is_diagnostic_only':True,
            'absent_reason':'no active fast expert; skipped without changing routing or parameters'}

def attach_fast_group(groups,fast_params):
    if fast_params: groups['fast_active_expert']=fast_params
    return not bool(fast_params)

def make_gradient_call_counter():
    calls=0
    def add(value):
        nonlocal calls
        calls+=int(value)
        return calls
    return add

def grad_probe(model,pack,seed,meter,stage,groups):
    """Old/new CE gradients for mutable groups; computes no optimizer update."""
    banks=model.banks(); fast_params=active_fast_params(model)
    all_named=dict(model.named_parameters())
    group_params={'base0':[p for _,p in named_base(model,0)],'base1':[p for _,p in named_base(model,1)]}
    skip_fast=attach_fast_group(group_params,fast_params)
    by_id={id(p):n for n,p in all_named.items()}
    res={}; old_class_deltas={}; named_by_id={id(p):n for n,p in all_named.items()}; forward_calls=0
    all_parameters=list(model.parameters())
    for group,params in group_params.items():
        if not params: raise RuntimeError(f'empty mutable gradient group {group}')
        ids={id(p) for p in params}; flags=[p.requires_grad for p in all_parameters]
        saved_grads=[None if p.grad is None else p.grad.detach().clone() for p in all_parameters]
        for p in all_parameters: p.requires_grad_(id(p) in ids)
        try:
            vectors={}
            for name,classes in (('old',(0,1)),('new',(2,3))):
                ix=balanced_indices(pack,'train',classes,2); batch=c.batch(pack,seed,ix)
                meter.stage=f'{stage}_{group}_{name}_gradient'
                ids_t,mask,lens,markers,labels=batch; logits=model.logits(ids_t,mask); forward_calls+=1
                ri=torch.arange(len(labels),device=ids_t.device)
                loss=.5*(F.cross_entropy(logits[ri,lens-1].float(),markers)+F.cross_entropy(logits[ri,lens].float(),labels))
                grads=torch.autograd.grad(loss,params,allow_unused=True)
                vectors[name]=[torch.zeros_like(p,dtype=torch.float32) if g is None else g.detach().float().clone()
                               for p,g in zip(params,grads)]
                if name=='old':
                    delta=[]
                    warm=model._v5_warm_adapter
                    for n,p in zip([by_id[id(p)] for p in params],params):
                        delta.append((warm[n].to(device=p.device,dtype=torch.float32)-p.detach().float()).clone())
                    old_class_deltas[group]=vector_summary(vectors['old'],delta)['dot']
            stats=vector_summary(vectors['old'],vectors['new'])
            stats['directional_old_loss_delta_if_restored_warm']=old_class_deltas[group]
            stats['parameter_count']=sum(p.numel() for p in params)
            stats['parameter_l2_norm']=sum(float(p.detach().float().square().sum()) for p in params)**.5
            stats['gradient_requires_grad_scope']=[named_by_id[id(p)] for p in params]
            stats['ordinary_wake_requires_grad_before_probe']=[named_by_id[id(p)] for p,f in zip(all_parameters,flags) if f]
            stats['probe_scope_is_diagnostic_only']=True
            res[group]=stats
        finally:
            for p,flag,g in zip(all_parameters,flags,saved_grads):
                p.requires_grad_(flag); p.grad=None if g is None else g
    if skip_fast: res['fast_active_expert']=absent_group_summary()
    return {'groups':res,'forward_calls':forward_calls,'absent_groups':[k for k,v in res.items() if v.get('absent_reason')]}

def restore_base(model,warm_state,level):
    params=dict(model.named_parameters())
    with torch.no_grad():
        for name in model.level_descriptors()[level]['base']:
            params[name].copy_(warm_state[name].to(device=params[name].device,dtype=params[name].dtype))

def ablation_metrics(model,warm_state,pack,seed,validation,meter,stage,check):
    outputs={}
    for ab in ABLATIONS:
        check()
        clone=isolated_clone(model,share_backbone=True); clone.eval().requires_grad_(False)
        if ab in ('restore_warm_base0','restore_both_bases'): restore_base(clone,warm_state,0)
        if ab in ('restore_warm_base1','restore_both_bases'): restore_base(clone,warm_state,1)
        b0,b1=clone.banks()
        if ab=='disable_fast_bank': b0.route_override={'disable':[i for i,s in enumerate(b0.slots) if bool(s.active)]}
        if ab=='force_original_slow_receiver0': b1.route_override={'force':0}
        if ab=='disable_original_receiver0': b1.route_override={'disable':[0]}
        outputs[ab]=guarded_observation(clone,lambda:score_with_route_audit(clone,validation,pack,seed,f'{stage}_ablation_{ab}',meter),
            execution_state,lambda:c.engine.capture_rng())
        del clone
    return outputs

def score_with_route_audit(model,records,pack,seed,stage,meter):
    """Attach passive hooks to the scorer's existing forwards; never adds a forward."""
    banks=model.banks(); prior=[]; counts={}; forced=False
    for b in banks:
        override=b.route_override or {}; forced=forced or override.get('force') is not None
        prior.append((b,getattr(b,'audit_routes',None),getattr(b,'last_route_audit',None),
                      getattr(b,'last_routes',None),getattr(b,'audit_history',None)))
    cursors={i:0 for i in range(len(banks))}; hooks=[]
    for bi,bank in enumerate(banks):
        if not hasattr(bank,'audit_routes') or not hasattr(bank,'last_route_audit'): continue
        bank.audit_routes=True
        if hasattr(bank,'audit_history'): bank.audit_history=[]
        def hook(module,args,out,bi=bi):
            audit=module.last_route_audit
            if not isinstance(audit,dict) or not torch.is_tensor(audit.get('selected')): return
            for ri in range(out.shape[0]):
                if cursors[bi]+ri>=len(records): break
                row=records[cursors[bi]+ri]; group='old' if row['class_id']<2 else 'new'
                item=counts.setdefault(f'bank{bi}/{group}',{'loss_positions':0,'selected_under_policy_slots':{}})
                lens=len(row['tokens']); pos=(lens-1,lens)
                for slot in audit['selected'][ri,list(pos)].detach().cpu().reshape(-1).tolist():
                    key=str(int(slot));item['selected_under_policy_slots'][key]=item['selected_under_policy_slots'].get(key,0)+1
                    item['loss_positions']+=1
            cursors[bi]+=out.shape[0]
        hooks.append(bank.register_forward_hook(hook))
    try: metrics=c.score(model,records,pack,seed,stage,meter)
    finally:
        for h in hooks:h.remove()
        for bank,value,last_audit,last_routes,history in prior:
            if value is not None:bank.audit_routes=value
            if hasattr(bank,'last_route_audit'):bank.last_route_audit=last_audit
            if hasattr(bank,'last_routes'):bank.last_routes=last_routes
            if hasattr(bank,'audit_history'):bank.audit_history=history
    return {'metrics':metrics,'selected_under_policy_winner_counts':counts,
            'route_policy_by_bank':[b.route_override for b in banks],
            'winner_counts_are_selected_under_policy':True,'has_force_override':forced,
            'unforced_natural_top1_available':not forced}

def train_pool(pack,seed):
    return [pack['records'][i] for cls in range(4) for i in balanced_indices(pack,'train',(cls,),6)]

def endpoint_record(seal,seed):
    runs=[r for r in seal['runs'] if r['seed']==seed and r['arm']=='warm_CMS_no_sleep']
    if len(runs)!=1: raise ValueError(f'expected one v4 warm_CMS_no_sleep row for seed {seed}')
    cps=[x for x in runs[0]['checkpoints'] if x.get('primary_step',x.get('step'))==16]
    if len(cps)!=1: raise ValueError(f'missing unique v4 step16 checkpoint for seed {seed}')
    return runs[0],cps[0]

def resolve_phaseb_checkpoint(run_dir,ref):
    path=run_dir/ref['path']
    if not path.exists(): path=run_dir/'checkpoints'/ref['path']
    if not path.is_file(): raise FileNotFoundError(f'v4 checkpoint missing: {ref["path"]}')
    return path

def main():
    p=argparse.ArgumentParser()
    for name in ('records','v3-run-dir','warm-dir','v4-run-dir','model-dir','out-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); pack=c.load_pack(a.records); c.preflight(a.model_dir,pack)
    warm=c.read(a.warm_dir/'transfer-result.json'); v4=c.read(a.v4_run_dir/'development-sealed.json')
    v4proto=c.read(a.v4_run_dir/'protocol.json'); v3seal=c.read(a.v3_run_dir/'training-sealed.json')
    if not warm.get('all3_qualified') or not warm.get('phaseB_permitted') or warm.get('confirmation_scored'):
        raise ValueError('warm source must be all-seed qualified and train/validation only')
    if not v4.get('complete') or v4.get('confirmation_scored') or v4.get('no_holdout_input') is not True:
        raise ValueError('v4 comparison must be complete without holdout input/scoring')
    if {r['seed'] for r in warm['qualified_models']}!=set(SEEDS): raise ValueError('warm seed set mismatch')
    warm_files={};endpoint_refs={}
    for seed in SEEDS:
        wr=next(x for x in warm['qualified_models'] if x['seed']==seed)
        wp=a.warm_dir/wr['checkpoint']['path']
        if c.sha(wp)!=wr['checkpoint']['sha256']: raise ValueError(f'warm checkpoint SHA mismatch seed={seed}')
        warm_files[str(seed)]={'path':str(wp.resolve()),'sha256':c.sha(wp)}
        _,ref=endpoint_record(v4,seed); ep=resolve_phaseb_checkpoint(a.v4_run_dir,ref)
        if ref.get('sha256') and c.sha(ep)!=ref['sha256']: raise ValueError(f'v4 endpoint SHA mismatch seed={seed}')
        endpoint_refs[str(seed)]={'path':str(ep.resolve()),'sha256':c.sha(ep),'step':16}
    cfg=v4proto['config']; out=a.out_dir; out.mkdir(parents=True,exist_ok=False)
    source_files=[Path(__file__),Path(c.__file__),Path(c.engine.__file__),Path(c.engine.shared.__file__)]
    sources={str(x.resolve()):c.sha(x) for x in source_files}
    plan={'schema':'agnews-interference-diagnostic-plan-v5','source_sha256':sources,
      'input_sha256':{'records':c.sha(a.records),'v3_seal':c.sha(a.v3_run_dir/'training-sealed.json'),
        'warm_result':c.sha(a.warm_dir/'transfer-result.json'),'v4_seal':c.sha(a.v4_run_dir/'development-sealed.json'),
        'v4_protocol':c.sha(a.v4_run_dir/'protocol.json'),'warm_checkpoints':warm_files,
        'v4_step16_checkpoints':endpoint_refs},'seeds':list(SEEDS),'new_wake_updates_per_seed':16,
      'total_new_wake_updates':48,'trace_steps':list(TRACE),'ablation_steps':[1,4,16],
      'validation':'fixed first 32 rows/class from train-derived validation (128 total); all role/class metrics reported consistently',
      'gradient_probe':'fixed train-only 4-row batches: 2 each from old classes 0/1 and new classes 2/3; autograd.grad, no optimizer step',
      'teacher_candidate_pool':'fixed train-only 24 rows, 6/class', 'no_holdout_input':True,
      'elapsed_limit_seconds':1200,
      'endpoint_match_required':'v4 warm_CMS_no_sleep step16 model/controller/execution policy/RNG',
      'created_utc':datetime.now(timezone.utc).isoformat()}
    c.write(out/'plan.json',plan); plan_sha=c.sha(out/'plan.json')
    meter=c.engine.WorkMeter(); tok=c.engine.ACTIVE_METER.set(meter); all_results=[]; updates=0
    add_grad_calls=make_gradient_call_counter(); started=time.monotonic()
    def check():
        if time.monotonic()-started>=1200: raise TimeoutError('v5 diagnostic exceeded fixed 1200-second elapsed limit')
    try:
      for seed in SEEDS:
        check()
        model=c.model(a.model_dir,seed); check(); wr=next(x for x in warm['qualified_models'] if x['seed']==seed)
        warm_path=a.warm_dir/wr['checkpoint']['path']
        if c.sha(warm_path)!=wr['checkpoint']['sha256']: raise ValueError('warm checkpoint SHA mismatch')
        warm_cp=torch.load(warm_path,map_location='cpu',weights_only=False); c.restore(model,warm_cp); c.engine.restore_rng(warm_cp['rng'])
        warm_state={k:v.detach().cpu().clone() for k,v in warm_cp['model'].items()}; model._v5_warm_adapter=warm_state
        ctl=CMSController(model,(4,16),cfg['base_lr'],cfg['expert_lr'])
        validation_all=c.rows(pack,seed,'validation')
        validation=[r for cls in range(4) for r in [x for x in validation_all if x['class_id']==cls][:32]]
        if len(validation)!=128: raise ValueError('validation role must provide 32 rows per class')
        train=c.rows(pack,seed,'train'); rows=c.rows(pack,seed)
        rng_at_warm=tensor_hash(c.engine.capture_rng()); fp0=c.fingerprint(model)
        snaps=[]; gradients=[]; ablations=[]; candidate_scores=[]
        def observe(step,tag):
            check()
            result=guarded_observation(model,lambda:score_with_route_audit(model,validation,pack,seed,f'v5_seed{seed}_{tag}',meter),execution_state,lambda:c.engine.capture_rng())
            check()
            snaps.append({'step':step,'tag':tag,'metrics':result,'adapter_fingerprint':c.fingerprint(model),
                          'controller_counts':list(ctl.counts),'policy':execution_state(model)})
            if step in (0,1,4,16) and tag=='post':
                check()
                probe=guarded_observation(model,lambda:grad_probe(model,pack,seed,meter,f'v5_seed{seed}_{tag}',('base0','base1','fast_active_expert')),
                    execution_state,lambda:c.engine.capture_rng())
                add_grad_calls(probe['forward_calls'])
                gradients.append({'step':step,'tag':tag,**probe})
                check()
            if step in (1,4,16) and tag=='post':
                ablations.append({'step':step,'metrics':ablation_metrics(model,warm_state,pack,seed,validation,meter,f'v5_seed{seed}_step{step}',check)})
            print(json.dumps({'phase':'interference_v5','seed':seed,'step':step,'tag':tag,
                'elapsed_seconds':time.monotonic()-started,'gradient_probe_forward_calls':add_grad_calls(0)}),flush=True)
        observe(0,'post')
        new_schedule=pack['schedules'][str(seed)]['new']
        for step in range(1,17):
            check()
            indices=new_schedule[step-1]; updates+=1
            if step in (4,16):
                ev=c.engine.shared.step(model,ctl,[rows[i] for i in indices],pack['tokenizer']['pad_token_id'],meter,f'v5_seed{seed}_wake',defer_due=True)
                if not ev.get('proposals'): raise RuntimeError(f'no due proposal at step {step}')
                observe(step,'prebase')
                candidate=isolated_clone(model,share_backbone=True); cc=CMSController.restore(candidate,ctl.state_dict())
                pool=train_pool(pack,seed)
                before=guarded_observation(model,lambda:c.score(model,pool,pack,seed,f'v5_seed{seed}_candidate{step}_before',meter),
                    execution_state,lambda:c.engine.capture_rng())
                check()
                slow_only=None
                if 1 in ev['proposals']:
                    cc.apply(candidate,1,ev['proposals'][1])
                    slow_only=guarded_observation(candidate,lambda:c.score(candidate,pool,pack,seed,f'v5_seed{seed}_candidate{step}_slow_only',meter),
                        execution_state,lambda:c.engine.capture_rng())
                    check()
                if 0 in ev['proposals']: cc.apply(candidate,0,ev['proposals'][0])
                all_due=guarded_observation(candidate,lambda:c.score(candidate,pool,pack,seed,f'v5_seed{seed}_candidate{step}_all_due',meter),
                    execution_state,lambda:c.engine.capture_rng())
                candidate_scores.append({'step':step,'prebase_live_counterfactual_teacher':before,
                    'after_slow_base_only_counterfactual_teacher':slow_only,
                    'after_all_due_bases_counterfactual':all_due,'due_levels':sorted(ev['proposals']),
                    'pool_rows':len(pool),'not_an_actual_Sleep_trajectory':True})
                del candidate,cc
                check()
                for level in sorted(ev['proposals'],reverse=True): ctl.apply(model,level,ev['proposals'][level])
                observe(step,'post')
            else:
                c.engine.supervised_update(model,ctl,c.batch(pack,seed,indices),pack['tokenizer']['pad_token_id'],meter,f'v5_seed{seed}_wake')
                if step in (1,8): observe(step,'post')
        run,cpref=endpoint_record(v4,seed); cp_path=resolve_phaseb_checkpoint(a.v4_run_dir,cpref)
        endpoint=torch.load(cp_path,map_location='cpu',weights_only=False)
        now={'model':model.adapter_state_dict(),'controller':ctl.state_dict(),'execution_state':execution_state(model),'rng':c.engine.capture_rng()}
        expected={'model':endpoint['model'],'controller':endpoint['controller'],'execution_state':endpoint['execution_state'],'rng':endpoint['rng']}
        match={k:tensor_hash(now[k])==tensor_hash(expected[k]) for k in now}
        if not all(match.values()): raise RuntimeError(f'v4 step16 endpoint mismatch seed={seed}: {match}')
        all_results.append({'seed':seed,'warm_start_checkpoint_sha256':wr['checkpoint']['sha256'],'warm_step0_fingerprint':fp0,
          'warm_rng_hash':rng_at_warm,'snapshots':snaps,'gradient_probes':gradients,'ablations':ablations,
          'teacher_candidate_scores':candidate_scores,'updates':16,'endpoint_match_v4_step16':match,
          'endpoint_fingerprint':c.fingerprint(model),'endpoint_controller_counts':list(ctl.counts)})
        del model,ctl,warm_cp,endpoint; torch.cuda.empty_cache()
      if updates!=48 or len(all_results)!=3: raise RuntimeError('incomplete fixed diagnostic update budget')
      c.write(out/'diagnostic-result.json',{'schema':'agnews-interference-diagnostic-v5','complete':True,
        'plan_sha256':plan_sha,'source_sha256':sources,'records_sha256':c.sha(a.records),'results':all_results,
        'updates':updates,'gradient_probe_forward_calls':add_grad_calls(0),'work':meter.counts,
        'seconds':time.monotonic()-started,'elapsed_limit_seconds':1200,
        'no_holdout_input':True,'confirmation_scored':False,
        'completed_utc':datetime.now(timezone.utc).isoformat()})
    except BaseException as e:
      c.write(out/'failure.json',{'schema':'agnews-interference-failure-v5','type':type(e).__name__,'error':str(e),
        'completed_seeds':[r['seed'] for r in all_results],'attempted_wake_updates':updates,'work':meter.counts,
        'gradient_probe_forward_calls':add_grad_calls(0),'seconds':time.monotonic()-started,
        'elapsed_limit_seconds':1200,'plan_sha256':plan_sha,
        'no_holdout_input':True,'automatic_retry':False})
      raise
    finally: c.engine.ACTIVE_METER.reset(tok)

if __name__=='__main__': main()
