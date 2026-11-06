"""Real-model no-update qualification of the frozen last-MLP feature cache."""
import argparse
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import agnews_v3_common as c
from agnews_tail_cache_v7 import capture, tail_logits_padded
from matmul_work_v7 import MatmulWork


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('records','model-dir','v6-dir','out-dir'): p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); assert not a.out_dir.exists()
    pack=c.load_pack(a.records);c.preflight(a.model_dir,pack)
    seal=c.read(a.v6_dir/'development-sealed.json')
    run=next(r for r in seal['runs'] if r['seed']==17 and r['arm']=='aux_new_CE')
    cp=next(x for x in run['checkpoints'] if x['primary_step']==1024)
    path=(a.v6_dir/cp['path']).resolve();assert path.is_relative_to(a.v6_dir.resolve()) and c.sha(path)==cp['sha256']
    a.out_dir.mkdir(parents=True)
    config={'logit_max_abs':.05,'loss_abs':.001,'gradient_relative_l2':.01,
            'padded_sequence_length':max(len(r['tokens']) for r in pack['records'])+1,
            'require_marker_and_class_argmax_exact':True,'optimizer_updates':0,
            'records_sha256':c.DATA_SHA,'checkpoint_sha256':cp['sha256'],
            'source_sha256':{f:c.sha(Path(__file__).parent/f) for f in
                ('qualify_agnews_tail_cache_v7.py','agnews_tail_cache_v7.py','matmul_work_v7.py')}}
    c.write(a.out_dir/'plan.json',config)
    started=time.monotonic(); work=MatmulWork();m=None
    try:
        with work:
            work.stage='load';m=c.model(a.model_dir,17)
            state=torch.load(path,map_location='cpu',weights_only=False);c.restore(m,state)
            indices=[next(i for i,r in enumerate(pack['records']) if r['role']=='train' and r['class_id']==cl)
                     for cl in (0,0,1,1,2,2,3,3)]
            ids,mask,lens,markers,labels=c.batch(pack,17,indices)
            size=config['padded_sequence_length']-ids.shape[1]
            assert size>=0
            ids=F.pad(ids,(0,size),value=pack['tokenizer']['pad_token_id']);mask=F.pad(mask,(0,size),value=0)
            work.stage='capture';work.phase='forward'
            with torch.no_grad(): cache=capture(m,ids,mask,lens)
            m.banks()[0].reset_active();slot=m.banks()[1].activate()
            assert slot==1
            m.banks()[1].route_override={'mix':slot,'fraction':.5}
            before=c.fingerprint(m);results=[]
            rows=torch.arange(8,device='cuda');targets=torch.stack((markers,labels),dim=1)
            for scope in ('receiver','shared'):
                m.requires_grad_(False)
                names=[m.level_descriptors()[1]['bank']+f'.slots.{slot}.{k}' for k in ('a','b')]
                if scope=='shared': names += [n for d in m.level_descriptors() for n in d['base']]
                params=dict(m.named_parameters());train=[params[n] for n in names]
                for t in train:t.requires_grad_(True)
                work.stage=scope+'_full';work.phase='forward'
                all_logits=m.logits(ids,mask)
                full=torch.stack((all_logits[rows,lens-1],all_logits[rows,lens]),dim=1).float()
                loss=F.cross_entropy(full.reshape(-1,full.shape[-1]),targets.flatten())
                work.phase='backward';loss.backward();fg=[x.grad.detach().clone() for x in train]
                full_value=full.detach().clone();full_loss=float(loss.detach());m.zero_grad(set_to_none=True)
                del all_logits,full,loss
                work.stage=scope+'_tail';work.phase='forward'
                tail=tail_logits_padded(m,cache['y'],cache['residual'],lens,config['padded_sequence_length']).float()
                loss=F.cross_entropy(tail.reshape(-1,tail.shape[-1]),targets.flatten())
                work.phase='backward';loss.backward()
                rels=[float((x.grad-g).norm()/g.norm().clamp_min(1e-12)) for x,g in zip(train,fg)]
                maximum=float((tail.detach()-full_value).abs().max())
                loss_diff=abs(float(loss.detach())-full_loss)
                argmax=bool(torch.equal(tail.detach().argmax(-1),full_value.argmax(-1)))
                record={'scope':scope,'logit_max_abs':maximum,'loss_abs':loss_diff,
                    'gradient_relative_l2':rels,'argmax_exact':argmax,'trainable_parameters':sum(x.numel() for x in train)}
                results.append(record)
                assert maximum<=config['logit_max_abs'] and loss_diff<=config['loss_abs'] and argmax
                assert all(x<=config['gradient_relative_l2'] for x in rels)
                m.zero_grad(set_to_none=True);del tail,loss,fg,full_value
            assert c.fingerprint(m)==before
        c.write(a.out_dir/'qualification.json',{'passed':True,'scope':'real3B forward/gradient equivalence; no optimizer update',
            'config':config,'results':results,'matmul_work_lower_bound':work.state_dict(),
            'wall_seconds':time.monotonic()-started,'confirmation_scored':False,'optimizer_updates':0})
        print({'passed':True,'results':results,'seconds':time.monotonic()-started},flush=True)
    except BaseException as e:
        c.write(a.out_dir/'failure.json',{'type':type(e).__name__,'error':str(e),
            'partial_results':locals().get('results',[]),
            'work':work.state_dict(),'seconds':time.monotonic()-started,'optimizer_updates':0})
        raise

if __name__=='__main__':main()
