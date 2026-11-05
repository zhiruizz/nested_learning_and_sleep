"""Continue a qualified DEV foundation; validate actual3B Sleep and recovery."""
from datetime import datetime
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_x870e_synthetic_dev_v2 as shared
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--foundation-run',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    args=parser.parse_args()
    parent=json.loads((args.foundation_run/'protocol.json').read_text())
    assert shared.sha256(args.records)==parent['records_sha256']
    assert shared.sha256(Path(shared.__file__))==parent['helper_sha256']
    elapsed=time.time()-datetime.fromisoformat(parent['started_utc']).timestamp()
    clock=shared.Clock(time.monotonic()+max(0.,1200-elapsed))
    clock.check()
    out=args.foundation_run/'sleep-fixture'
    out.mkdir(exist_ok=False)
    def write(name,value):
        with (out/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    pack=json.loads(args.records.read_text())
    assert pack['test_opened'] is False
    mapping=pack['code_maps']['1907'];marker=mapping['marker_token_id']
    rows=[{**r,'marker':marker,'label':mapping['class_to_token_ids'][r['class_id']]}for r in pack['records']]
    schedule=pack['schedules']['1907']['new_primary256'][:64]
    pool=[rows[i]for i in pack['sleep_prefix_pools']['1907'][0]]
    assert [r['class_id']for r in pool]==[0,2,3,1,1,3,2,0]*2+[0,2,1,3]*2
    assert all(r['partition']=='dev' and r['role'].endswith('_train')for r in pool)
    qualified=json.loads((args.foundation_run/'candidate-00-qualification.json').read_text())
    assert qualified['qualifies'] is True
    checkpoint=torch.load(args.foundation_run/'candidate-00-foundation.pt',map_location='cpu',weights_only=False)
    assert checkpoint['candidate']==0 and checkpoint['seed']==1907
    assert checkpoint['records_sha256']==parent['records_sha256']
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    backbone=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
    model=shared.LlamaCMSLM(backbone,rank=16,capacity=8,levels=2)
    shared.promote_mutable_to_fp32(model)
    def restore(state):
        shared.load_snapshot(model,state['model'])
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
        random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
        return shared.CMSController.restore(model,state['controller'])
    controller=restore(checkpoint)
    assert controller.counts==[0,0] and controller.events==512
    meter=shared.WorkMeter();mtoken=shared.ACTIVE_METER.set(meter)
    pad=pack['tokenizer']['pad_token_id'];attempts=0;started=time.time()
    try:
        for i,indices in enumerate(schedule):
            clock.check()
            if i==63:
                saved=shared.compact_cpu({'model':shared.snapshot(model),'controller':controller.state_dict(),
                   'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
                          'python':random.getstate(),'numpy':np.random.get_state()}})
                torch.save(saved,out/'wake63.pt')
            event=shared.step(model,controller,[rows[j]for j in indices],pad,meter,'fixture_new_wake',defer_due=(i==63))
            attempts+=1
        reference=(shared.fingerprint(model),shared.tensor_tree_hash(controller.state_dict()),
                   shared.tensor_tree_hash(event['proposals']))
        saved=torch.load(out/'wake63.pt',map_location='cpu',weights_only=False)
        controller=restore(saved)
        clock.check()
        event=shared.step(model,controller,[rows[j]for j in schedule[63]],pad,meter,'wake64_restore_repeat',defer_due=True)
        attempts+=1
        actual=(shared.fingerprint(model),shared.tensor_tree_hash(controller.state_dict()),
                shared.tensor_tree_hash(event['proposals']))
        assert actual==reference,'restored real3B wake63to64 diverged'
        assert controller.counts==[4,16]
        original_model=shared.fingerprint(model);original_controller=shared.tensor_tree_hash(controller.state_dict())
        candidate=shared.isolated_clone(model,share_backbone=True)
        cc=shared.CMSController.restore(candidate,controller.state_dict())
        cc.apply(candidate,1,event['proposals'][1])
        assert cc.counts==[4,0]
        session=shared.SleepSession(candidate,0,event['proposals'][0],seed=8907,lr=.003,
            dream_tokens=2,share_backbone=True,reject_zero_gradient=True)
        def prefix(row):return torch.tensor([row['tokens']],device='cuda',dtype=torch.long)
        meter.stage='dev_sleep_transaction'
        clock.check()
        recovery=shared.alias_restore_fixture(candidate,prefix(pool[0]),session,out,meter,marker)
        for i in range(1,16):
            clock.check();session.seed_step(prefix(pool[i]))
        marker_hits=0
        for i in range(16,24):
            clock.check();p=prefix(pool[i]);session.lti_step(p)
            marker_hits+=int(session.loss_context['sequence'][0,p.shape[1]].item()==marker)
        assert len(session.records)==24 and all(r['grad_norm']>0 for r in session.records)
        session.finish_consolidation();committed=session.commit();cc.consume(0)
        assert cc.counts==[0,0]
        assert shared.fingerprint(model)==original_model
        assert shared.tensor_tree_hash(controller.state_dict())==original_controller
        sleep_proxy=sum(v['forward_input_tokens']+2*v['gradient_forward_input_tokens']
            for k,v in meter.counts.items()if k=='dev_sleep_transaction')
        clock.check()
        shared.step(committed,cc,pool[:4],pad,meter,'extra_replay_calibration')
        b=meter.counts['extra_replay_calibration'];batch_proxy=b['forward_input_tokens']+2*b['gradient_forward_input_tokens']
        extra=max(4,round(sleep_proxy/batch_proxy/4)*4)
        result={'status':'REAL3B_WAKE_AND_SLEEP_RECOVERY_PASS','new_wake_updates':64,
            'extra_restore_wake_update_attempts':attempts-64,'wake63to64_next_state_and_proposals_identical':True,
            'sleep_optimizer_restore':recovery,'gkd_steps':16,'lti_steps':8,'dream_steps':0,
            'all_receiver_gradients_positive':True,'original_live_and_controller_unchanged':True,
            'lti_teacher_marker_hits':marker_hits,'lti_teacher_marker_total':8,
            'sleep_work_proxy':sleep_proxy,'replay_batch_work_proxy':batch_proxy,
            'extra_replay_batches_per_event':extra,'replay_to_sleep_proxy_ratio':extra*batch_proxy/sleep_proxy,
            'work':deepcopy(meter.counts),'elapsed_seconds':time.time()-started,
            'cumulative_dev_seconds':time.time()-datetime.fromisoformat(parent['started_utc']).timestamp(),
            'source_sha256':shared.sha256(Path(__file__)),'test_opened':False,'main_started':False}
        write('sleep-fixture-result.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        write('failure.json',{'status':'FIXTURE_FAILED_NO_MAIN_RELEASE','error_type':type(exc).__name__,
          'error':str(exc),'new_wake_update_attempts':attempts,'work':meter.counts,'test_opened':False})
        raise
    finally:shared.ACTIVE_METER.reset(mtoken)


if __name__=='__main__':main()
