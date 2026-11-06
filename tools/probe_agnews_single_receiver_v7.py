"""Predeclared read-only gradient and full-model output-lesion diagnostics."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import time
import torch
import agnews_v3_common as c
import run_agnews_single_receiver_v7 as v7
import agnews_tail_cache_v7 as cache_helper
from agnews_tail_cache_v7 import tail_logits_padded
from matmul_work_v7 import MatmulWork
from single_receiver_probes_v7 import gradient_conflict,slow_output_lesion
from nested_sleep.lm import execution_state


@contextmanager
def route_audit(model,records):
    """Observe actual router decisions only at the two semantic loss positions."""
    bank=model.banks()[1]
    previous=(bank.audit_routes,bank.last_route_audit,bank.audit_history)
    bank.audit_routes=True
    totals={'loss_positions':0,'natural_old':0,'natural_new':0,'null_probability_sum':0.,
            'executed_old_coefficient_sum':0.,'executed_new_coefficient_sum':0.}
    offset=0
    def observe(_module,inputs,output):
        nonlocal offset
        count=inputs[0].shape[0];batch=records[offset:offset+count];offset+=count
        lengths=torch.tensor([len(r['tokens']) for r in batch],device=inputs[0].device)
        rows=torch.arange(count,device=lengths.device)[:,None]
        pos=torch.stack((lengths-1,lengths),1)
        audit=bank.last_route_audit
        selected=audit['selected'][rows,pos];probs=audit['probabilities'][rows,pos]
        active=audit['active'];assert active==[0,1] and bank.route_override=={'mix':1,'fraction':.5}
        totals['loss_positions']+=selected.numel()
        totals['natural_old']+=int((selected==0).sum());totals['natural_new']+=int((selected==1).sum())
        totals['null_probability_sum']+=float(probs[...,-1].sum())
        totals['executed_old_coefficient_sum']+=float((.5*probs[...,0]*(selected==0)).sum())
        totals['executed_new_coefficient_sum']+=float((.5+.5*probs[...,1]*(selected==1)).sum())
    handle=bank.register_forward_hook(observe)
    try:yield totals
    finally:
        handle.remove();bank.audit_routes,bank.last_route_audit,bank.audit_history=previous


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('records','model-dir','run-dir','v6-dir','out-dir'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.out_dir.exists()
    pack=c.load_pack(a.records);c.preflight(a.model_dir,pack)
    seal=c.read(a.run_dir/'development-sealed.json');protocol=c.read(a.run_dir/'protocol.json')
    assert seal['complete'] and len(seal['runs'])==30
    assert seal['protocol_sha256']==c.sha(a.run_dir/'protocol.json')
    assert protocol['driver_sha256']==c.sha(Path(v7.__file__))
    assert protocol['cache_helper_sha256']==c.sha(Path(cache_helper.__file__))
    assert protocol['records_sha256']==c.DATA_SHA
    assert {(r['seed'],r['arm']) for r in seal['runs']}=={(s,arm) for s in v7.SEEDS for arm in v7.ARMS}
    assert all(r['status']=='complete' and r['updates']==1024 for r in seal['runs'])
    v6=c.read(a.v6_dir/'development-sealed.json');assert c.sha(a.v6_dir/'development-sealed.json')==protocol['v6_seal_sha256']
    a.out_dir.mkdir(parents=True)
    plan={'schema':'agnews-single-receiver-diagnostic-plan-v7','main_seal_sha256':c.sha(a.run_dir/'development-sealed.json'),
          'main_protocol_sha256':seal['protocol_sha256'],'records_sha256':c.DATA_SHA,
          'steps':[0,64,256,1024],'lesions':[[],[0],[1],[0,1]],'full_model_lesion_validation':True,
          'gradient_batch':'fixed first2 train examples per class; B8 same matrix shape',
          'gradient_objectives':'warm old KL / current new KL; each averages its four example positions',
          'global_seconds_cap':7200,'confirmation_scored':False,'development_only':True,
          'source_sha256':{x:c.sha(Path(__file__).parent/x) for x in
              ('probe_agnews_single_receiver_v7.py','single_receiver_probes_v7.py','run_agnews_single_receiver_v7.py','agnews_tail_cache_v7.py','matmul_work_v7.py')}}
    c.write(a.out_dir/'plan.json',plan)
    work=MatmulWork();started=time.monotonic();gradient_rows=[];lesion_rows=[];initial_probes={}
    def guard():
        if time.monotonic()-started>7200:raise TimeoutError('diagnostic cooperative cap')
    def invariant(m):
        return {'model':c.fingerprint(m),'policy':c.engine.shared.tensor_tree_hash(execution_state(m)),
                'rng':c.engine.shared.tensor_tree_hash(c.engine.capture_rng())}
    try:
        with work:
          for seed in v7.SEEDS:
            guard();m=c.model(a.model_dir,seed)
            cache_path=a.run_dir/'cache-private'/f'seed-{seed}.pt'
            source_rows=[r for r in seal['runs'] if r['seed']==seed]
            assert len({r['cache_sha256'] for r in source_rows})==1
            assert c.sha(cache_path)==source_rows[0]['cache_sha256']
            data=torch.load(cache_path,map_location='cpu',weights_only=False);features=data['student_features']
            train=c.rows(pack,seed,'train');val=c.rows(pack,seed,'validation');allrows=train+val
            indices=[i for cls in range(4) for i in [j for j,r in enumerate(train) if r['class_id']==cls][:2]]
            assert len(indices)==8 and all(train[i]['role']=='train' for i in indices)
            y=torch.cat([features[i]['y'] for i in indices]).cuda()
            residual=torch.cat([features[i]['residual'] for i in indices]).cuda()
            lengths=torch.cat([features[i]['lengths'] for i in indices]).cuda()
            old_t=torch.cat([data['warm_old_teacher_train_logits'][i] for i in indices[:4]]).cuda().float()
            new_t=torch.cat([data['new_teacher_train_logits'][i] for i in indices[4:]]).cuda().float()
            old_lp=old_t.log_softmax(-1).detach();new_lp=new_t.log_softmax(-1).detach()
            newrun=next(r for r in v6['runs'] if r['seed']==seed and r['arm']=='aux_new_CE')
            newmeta=v7._checkpoint_row(newrun,1024)
            for row in source_rows:
              for step in plan['steps']:
                guard()
                if step==0:
                    checkpoint,_=v7._load_cp(a.v6_dir,newmeta);c.restore(m,checkpoint);v7.reset_student_context(m)
                    assert c.fingerprint(m)==seal['common_initial_by_seed'][str(seed)]['adapter_sha256']
                    assert c.engine.shared.tensor_tree_hash(execution_state(m))==seal['common_initial_by_seed'][str(seed)]['policy_sha256']
                else:
                    meta=next(x for x in row['checkpoints'] if x['step']==step)
                    checkpoint,_=v7._load_cp(a.run_dir,meta);assert checkpoint['seed']==seed and checkpoint['arm']==row['arm']
                    assert checkpoint['protocol_sha256']==seal['protocol_sha256'];c.restore(m,checkpoint)
                before=invariant(m)
                if step==0 and seed in initial_probes:
                    stats,objective_values=initial_probes[seed];reused=True
                else:
                    objective_values={}
                    params=dict(m.named_parameters());desc=m.level_descriptors()
                    groups={'receiver':[(f'{desc[1]["bank"]}.slots.1.{x}',params[f'{desc[1]["bank"]}.slots.1.{x}']) for x in ('a','b')],
                            'base0':[(n,params[n]) for n in desc[0]['base']],
                            'base1':[(n,params[n]) for n in desc[1]['base']]}
                    def loss_old():
                        q=tail_logits_padded(m,y,residual,lengths,118).float()[:4].log_softmax(-1)
                        loss=(old_lp.exp()*(old_lp-q)).sum(-1).mean()
                        objective_values['old_warm_teacher_KL']=float(loss.detach())
                        return loss
                    def loss_new():
                        q=tail_logits_padded(m,y,residual,lengths,118).float()[4:].log_softmax(-1)
                        loss=(new_lp.exp()*(new_lp-q)).sum(-1).mean()
                        objective_values['new_current_teacher_KL']=float(loss.detach())
                        return loss
                    work.stage=f'{seed}:{row["arm"]}:{step}:gradient'
                    work.phase='diagnostic_combined_forward_backward'
                    stats=gradient_conflict(m,loss_old,loss_new,groups);reused=False
                    if step==0:initial_probes[seed]=(stats,objective_values)
                assert invariant(m)==before
                gradient_rows.append({'seed':seed,'arm':row['arm'],'step':step,'gradient':stats,
                    'objective_values':objective_values,
                    'reused_common_initial_probe':reused,'fixed_batch_indices_sha256':v7.hash_json(indices),
                    'unchanged_model_policy_rng':True})
              for disabled in plan['lesions']:
                guard();before=invariant(m)
                work.stage=f'{seed}:{row["arm"]}:lesion:{disabled}';work.phase='forward'
                with slow_output_lesion(m,disabled):
                    if not disabled:
                        with route_audit(m,val) as routes:
                            logits=v7._full_fixed_logits(m,val,pack,seed,118)
                    else:logits=v7._full_fixed_logits(m,val,pack,seed,118)
                    metrics=v7.score_logits(logits,val,pack,seed);del logits
                assert invariant(m)==before
                if not disabled:
                    expected=row['evaluations'][-1]['validation']
                    assert all(metrics[g]['em']==expected[g]['em'] and abs(metrics[g]['nll']-expected[g]['nll'])<=.001 for g in metrics)
                lesion_rows.append({'seed':seed,'arm':row['arm'],'disabled_slow_outputs':disabled,
                    'validation':metrics,'model_policy_rng_unchanged':True,
                    'semantic_position_route_audit':dict(routes) if not disabled else None})
              c.engine.write_atomic(a.out_dir/'progress.json',{'gradient_rows':gradient_rows,'lesion_rows':lesion_rows,'complete':False})
              print({'seed':seed,'arm':row['arm'],'probe_complete':True},flush=True)
            del m,data,features,y,residual,lengths,old_t,new_t,old_lp,new_lp,checkpoint
        guard()
        c.write(a.out_dir/'result.json',{'schema':'agnews-single-receiver-diagnostics-v7','complete':True,
            'plan_sha256':c.sha(a.out_dir/'plan.json'),'gradient_rows':gradient_rows,'lesion_rows':lesion_rows,
            'independent_gradient_states':len(gradient_rows)-sum(r['reused_common_initial_probe'] for r in gradient_rows),
            'work':work.state_dict(),'seconds':time.monotonic()-started,'optimizer_updates':0,'confirmation_scored':False})
    except BaseException as e:
        c.write(a.out_dir/'failure.json',{'type':type(e).__name__,'error':str(e),'gradient_rows':gradient_rows,
            'lesion_rows':lesion_rows,'work':work.state_dict(),'seconds':time.monotonic()-started,'optimizer_updates':0})
        raise

if __name__=='__main__':main()
