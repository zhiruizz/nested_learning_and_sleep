"""Real-3B correctness qualification of anchored wake + Sleep checkpoint resume.

No validation/scoring or method selection. Two independent restores cross the
same pending CMS boundary and must agree exactly. Raw states remain private.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,time
import torch
import agnews_v3_common as c
from agnews_anchored_wake_v5 import anchored_step
import agnews_anchored_wake_v5 as anchor_source
import agnews_preupdate_sleep_v4 as boundary
from nested_sleep.optim import CMSController
from nested_sleep.lm import isolated_clone,execution_state

def state_hashes(model,ctl):
    return {k:c.engine.shared.tensor_tree_hash(v)for k,v in{
        'model':model.adapter_state_dict(),'controller':ctl.state_dict(),
        'execution_policy':execution_state(model),'RNG':c.engine.capture_rng()}.items()}

def main():
    p=argparse.ArgumentParser()
    for n in('records','warm-dir','model-dir','out-dir'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();pack=c.load_pack(a.records);warm=c.read(a.warm_dir/'transfer-result.json');assert warm['all3_qualified']
    c.preflight(a.model_dir,pack);a.out_dir.mkdir(exist_ok=False,parents=True);seed=17
    ref=next(x for x in warm['qualified_models']if x['seed']==seed);wp=a.warm_dir/ref['checkpoint']['path'];assert c.sha(wp)==ref['checkpoint']['sha256']
    plan={'schema':'agnews-anchored-resume-plan-v5','source_sha256':c.sha(__file__),'anchor_helper_sha256':c.sha(anchor_source.__file__),
      'boundary_helper_sha256':c.sha(boundary.__file__),'warm_checkpoint_sha256':c.sha(wp),'records_sha256':c.DATA_SHA,
      'seed':seed,'prefix_steps':14,'resume_steps':[15,16],'branches':2,'expected_SGD_updates':18,'expected_Sleep_Adam_updates':48,
      'old_aux':'fixed first2/train/class0and1','Sleep_pool':'fixed first6/train/class0to3','scoring':False,
      'confirmation_input':False,'automatic_retry':False,'elapsed_limit_seconds':600,'created_utc':datetime.now(timezone.utc).isoformat()}
    c.write(a.out_dir/'plan.json',plan);start=time.monotonic();meter=c.engine.WorkMeter();token=c.engine.ACTIVE_METER.set(meter);sgd=0;adam=0;outputs=[]
    def check():
        if time.monotonic()-start>=600:raise TimeoutError('resume qualification elapsed limit')
    try:
        model=c.model(a.model_dir,seed);cp=torch.load(wp,map_location='cpu',weights_only=False);c.restore(model,cp);c.engine.restore_rng(cp['rng'])
        teacher=isolated_clone(model,share_backbone=True).eval().requires_grad_(False);teacher_fp=c.fingerprint(teacher)
        ctl=CMSController(model,(4,16),.02,.04);rows=c.rows(pack,seed)
        old=[r for k in(0,1)for r in[r for r in rows if r['role']=='train'and r['class_id']==k][:2]]
        per_class={k:[r for r in rows if r['role']=='train'and r['class_id']==k][:6]for k in range(4)}
        used={k:0 for k in range(4)};pool=[]
        for k in([0,2,3,1,1,3,2,0]*2+[0,2,1,3]*2):pool.append(per_class[k][used[k]]);used[k]+=1
        def step(index):
            nonlocal model,ctl,sgd,adam
            check();chosen=[rows[i]for i in pack['schedules'][str(seed)]['new'][index-1]];sgd+=1
            event,proposals=anchored_step(model,ctl,chosen,old,pack['tokenizer']['pad_token_id'],meter,f'qualify_seed17_step{index}',teacher,defer_due=index%16==0)
            if index%16==0:
                model,ctl,summary=boundary.consolidate(model,ctl,proposals,pool,505170016,meter,'qualify_seed17_Sleep16')
                adam+=24;model.banks()[1].route_override={'mix':0,'fraction':.5}
            assert c.fingerprint(teacher)==teacher_fp
        for index in range(1,15):step(index)
        assert ctl.counts==[2,14]
        path=a.out_dir/'checkpoint14-private.pt';info=c.checkpoint(path,model,ctl,seed,14,{'plan_sha256':c.sha(a.out_dir/'plan.json')})
        frozen_sha=c.sha(path)
        for branch in range(2):
            check();saved=torch.load(path,map_location='cpu',weights_only=False);before=c.engine.shared.tensor_tree_hash(saved)
            c.restore(model,saved);ctl=CMSController.restore(model,saved['controller']);c.engine.restore_rng(saved['rng'])
            assert ctl.counts==[2,14]
            for index in(15,16):step(index)
            assert ctl.counts==[0,0]and c.engine.shared.tensor_tree_hash(saved)==before
            outputs.append(state_hashes(model,ctl));assert c.sha(path)==frozen_sha
        assert outputs[0]==outputs[1]and sgd==18 and adam==48;check()
        c.write(a.out_dir/'qualification.json',{'schema':'agnews-anchored-resume-qualified-v5','passed':True,'plan_sha256':c.sha(a.out_dir/'plan.json'),
          'source_sha256':c.sha(__file__),'anchor_helper_sha256':c.sha(anchor_source.__file__),'exact_model_controller_policy_RNG':True,
          'branches':outputs,'private_checkpoint':info,'SGD_updates':sgd,'Sleep_Adam_updates':adam,'work':meter.counts,
          'seconds':time.monotonic()-start,'confirmation_scored':False,'scoring':False})
        print('REAL3B_ANCHORED_SLEEP_RESUME_EXACT',flush=True)
    except BaseException as e:
        c.write(a.out_dir/'failure.json',{'error':str(e),'type':type(e).__name__,'SGD_attempts':sgd,'completed_Sleep_Adam':adam,'work':meter.counts,'confirmation_scored':False});raise
    finally:c.engine.ACTIVE_METER.reset(token)

if __name__=='__main__':main()
