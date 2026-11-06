"""Full versus same-shape cached training qualification, four losses/two scopes."""
import argparse
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import agnews_v3_common as c
from agnews_tail_cache_v7 import capture,tail_logits_padded
from matmul_work_v7 import MatmulWork
from nested_sleep.lm import isolated_clone,execution_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('records','model-dir','v6-dir','warm-dir','out-dir'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.out_dir.exists()
    pack=c.load_pack(a.records);c.preflight(a.model_dir,pack)
    seal=c.read(a.v6_dir/'development-sealed.json'); warm=c.read(a.warm_dir/'transfer-result.json')
    trial=next(r for r in seal['runs'] if r['seed']==17 and r['arm']=='aux_new_CE')
    newcp=next(x for x in trial['checkpoints'] if x['primary_step']==1024)
    oldcp=next(r['checkpoint'] for r in warm['qualified_models'] if r['seed']==17)
    newpath=a.v6_dir/newcp['path'];oldpath=a.warm_dir/oldcp['path']
    assert c.sha(newpath)==newcp['sha256'] and c.sha(oldpath)==oldcp['sha256']
    a.out_dir.mkdir(parents=True)
    plan={'fixed_T':118,'losses':['current','split','gold','new_only'],'scopes':['receiver','shared'],
        'updates_per_case':2,'actual_updates_expected':16,'require_exact_model_adam_rng':True,
        'max_logit_abs':.05,'max_loss_abs':.001,'max_grad_relative':.01,
        'teacher_logits_detached':True,'KL':'mean positions KL(teacher||student), temperature1, no LTI/RL',
        'records_sha256':c.DATA_SHA,'new_checkpoint_sha256':newcp['sha256'],'warm_checkpoint_sha256':oldcp['sha256'],
        'source_sha256':{f:c.sha(Path(__file__).parent/f) for f in ('qualify_agnews_v7_loss_paths.py','agnews_tail_cache_v7.py','matmul_work_v7.py')}}
    c.write(a.out_dir/'plan.json',plan)
    work=MatmulWork();started=time.monotonic();results=[];updates=0
    try:
        with work:
            m=c.model(a.model_dir,17);newstate=torch.load(newpath,map_location='cpu',weights_only=False)
            oldstate=torch.load(oldpath,map_location='cpu',weights_only=False)
            batches={}
            for kind,classes in [('balanced',(0,0,1,1,2,2,3,3)),('new_only',(2,2,2,2,3,3,3,3))]:
                pools={cl:[i for i,r in enumerate(pack['records']) if r['role']=='train' and r['class_id']==cl] for cl in range(4)}
                used={cl:0 for cl in range(4)};indices=[]
                for cl in classes:indices.append(pools[cl][used[cl]]);used[cl]+=1
                ids,mask,lens,markers,labels=c.batch(pack,17,indices)
                pad=118-ids.shape[1];assert pad>=0
                ids=F.pad(ids,(0,pad),value=pack['tokenizer']['pad_token_id']);mask=F.pad(mask,(0,pad),value=0)
                c.restore(m,newstate);work.stage='prepare_'+kind+'_current';work.phase='forward'
                with torch.no_grad():cache=capture(m,ids,mask,lens)
                c.restore(m,oldstate);work.stage='prepare_'+kind+'_warm'
                with torch.no_grad():oldcache=capture(m,ids,mask,lens)
                assert torch.equal(cache['y'],oldcache['y']) and torch.equal(cache['residual'],oldcache['residual'])
                split=cache['logits'].float().clone()
                if kind=='balanced':split[:4]=oldcache['logits'][:4].float()
                batches[kind]=(ids,mask,lens,torch.stack((markers,labels),1),cache,split.detach())
            c.restore(m,newstate);m.banks()[0].reset_active();slot=m.banks()[1].activate();assert slot==1
            m.banks()[1].route_override={'mix':1,'fraction':.5};start_hash=c.fingerprint(m)
            rng=c.engine.capture_rng()
            for family in plan['losses']:
                batch=batches['new_only' if family=='new_only' else 'balanced']
                ids,mask,lens,targets,cache,split=batch
                teacher=split if family=='split' else cache['logits'].float().detach()
                for scope in plan['scopes']:
                    records=[];gradients=[];values=[]
                    for branch in ('full','cached'):
                        pupil=isolated_clone(m,share_backbone=True);c.engine.restore_rng(rng)
                        names=[pupil.level_descriptors()[1]['bank']+f'.slots.1.{x}' for x in ('a','b')]
                        if scope=='shared':names += [n for d in pupil.level_descriptors() for n in d['base']]
                        parameters=dict(pupil.named_parameters());train=[parameters[n] for n in names]
                        pupil.requires_grad_(False)
                        for t in train:t.requires_grad_(True)
                        opt=torch.optim.AdamW(train,lr=.003,weight_decay=0)
                        work.stage=f'{family}_{scope}_{branch}';work.phase='forward'
                        if branch=='full':
                            logits=pupil.logits(ids,mask);rows=torch.arange(8,device='cuda')
                            logits=torch.stack((logits[rows,lens-1],logits[rows,lens]),1).float()
                        else:logits=tail_logits_padded(pupil,cache['y'],cache['residual'],lens,118).float()
                        if family in ('gold','new_only'):loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.flatten())
                        else:
                            lp=teacher.log_softmax(-1);lq=logits.log_softmax(-1)
                            loss=(lp.exp()*(lp-lq)).sum(-1).mean()
                        values.append((logits.detach().clone(),float(loss.detach())))
                        work.phase='backward';loss.backward();gradients.append([t.grad.detach().clone() for t in train])
                        torch.nn.utils.clip_grad_norm_(train,1);opt.step();updates+=1;opt.zero_grad(set_to_none=True)
                        records.append({'model':c.fingerprint(pupil),
                            'adam':c.engine.shared.tensor_tree_hash(opt.state_dict()),
                            'rng':c.engine.shared.tensor_tree_hash(c.engine.capture_rng()),'policy':execution_state(pupil)})
                        del pupil,opt,logits,loss,parameters,train
                    rel=[float((x-y).norm()/x.norm().clamp_min(1e-12)) for x,y in zip(*gradients)]
                    maximum=float((values[0][0]-values[1][0]).abs().max());ldiff=abs(values[0][1]-values[1][1])
                    row={'family':family,'scope':scope,'max_logit_abs':maximum,'loss_abs':ldiff,
                         'gradient_relative_l2':rel,'model_adam_rng_exact':records[0]==records[1]}
                    results.append(row);c.engine.write_atomic(a.out_dir/'progress.json',{'results':results,'updates':updates})
                    assert maximum<=.05 and ldiff<=.001 and max(rel)<=.01 and records[0]==records[1]
                    print(row,flush=True)
                    del records,gradients,values
            assert c.fingerprint(m)==start_hash and updates==16
        c.write(a.out_dir/'qualification.json',{'passed':True,'plan_sha256':c.sha(a.out_dir/'plan.json'),
            'results':results,'actual_Adam_updates':updates,'work':work.state_dict(),
            'seconds':time.monotonic()-started,'confirmation_scored':False})
    except BaseException as e:
        c.write(a.out_dir/'failure.json',{'type':type(e).__name__,'error':str(e),'results':results,
            'actual_Adam_updates':updates,'work':work.state_dict(),'seconds':time.monotonic()-started})
        raise

if __name__=='__main__':main()
