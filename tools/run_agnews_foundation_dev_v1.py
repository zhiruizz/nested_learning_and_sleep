"""Bounded AG News foundation feasibility; never opens official test data."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import run_x870e_synthetic_dev_v2 as shared
import numpy as np
import torch
from transformers import AutoModelForCausalLM

DATA_SHA = '7b5169daadcf7d4ccaefc9d0bb79f1a4eef92fc275fae0c3a7d593e06b7e9e59'
HELPER_SHA = '6c7bf0a079a052eb87458010c1b55e52db1d9daa6dce99e55708b2da98385948'
WEIGHTS = {'model-00001-of-00002.safetensors':'584d8d3e3f82f7964955174dfe5e3b1cf117a9d859f022cfdf7fcb884856e002',
           'model-00002-of-00002.safetensors':'4719a04514ec2f060240711b7c33ab21187cac730ecaba3040b7a0fd95a9cefb'}


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    assert shared.sha256(args.records) == DATA_SHA
    assert shared.sha256(Path(shared.__file__)) == HELPER_SHA
    pack = json.loads(args.records.read_text(encoding='utf-8'))
    assert pack['schema'] == 'nl-sleep-agnews-train-roles-v1' and pack['test_opened'] is False
    assert pack['phase_class_ids'] == {'old':[0,1], 'new':[2,3]}
    palette = pack['code_maps']['1907']
    marker, codes = palette['marker_token_id'], palette['class_to_token_ids']
    assert len(codes)==len(set(codes))==4
    rows = [{**r, 'marker':marker, 'label':codes[r['class_id']]} for r in pack['records']]
    train = [r for r in rows if r['partition']=='dev' and r['role']=='old_train']
    calibration = [r for r in rows if r['partition']=='dev' and r['role']=='old_calib']
    batches = pack['schedules']['1907']['foundation512']
    assert len(train)==256 and len(calibration)==128 and len(batches)==512
    assert shared.json_hash(batches)==pack['schedules']['1907']['foundation512_sha256']
    for batch in batches:
        assert len(batch)==4 and sorted(rows[i]['class_id'] for i in batch)==[0,0,1,1]
        assert all(rows[i]['partition']=='dev' and rows[i]['role']=='old_train' for i in batch)
    for item in pack['tokenizer']['file_inventory']:
        assert shared.sha256(args.model_dir/item['name'])==item['sha256']
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = time.monotonic()+1200
    def check():
        if time.monotonic()>=deadline:
            raise TimeoutError('AGNews development1200second hard cap; no extension')
    protocol = {'schema':'agnews-foundation-dev-v1', 'seed':1907, 'rank':16, 'capacity':8,
        'levels':2, 'periods':[4,16], 'batch_size':4, 'foundation_updates_per_candidate':512,
        'lr_pairs':[[.005,.01],[.02,.04],[.05,.1]], 'train_em_min':.8, 'calibration_em_min':.75,
        'hard_seconds':1200, 'records_sha256':DATA_SHA, 'helper_sha256':HELPER_SHA,
        'runner_sha256':shared.sha256(Path(__file__)), 'started_utc':datetime.now(timezone.utc).isoformat(),
        'official_test_opened':False, 'main_started':False, 'raw_data_published':False,
        'previous_synthetic_dev_updates':3072, 'previous_synthetic_dev_wall_seconds':1030.0394}
    write_new(args.out_dir/'protocol.json', protocol)
    print(json.dumps({'status':'started','pid':os.getpid(),'output_dir':str(args.out_dir),
                      'hard_seconds':1200,'runner_sha256':protocol['runner_sha256']}),flush=True)
    for filename, expected in WEIGHTS.items():
        assert shared.sha256(args.model_dir/filename)==expected
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.manual_seed(1907); torch.cuda.manual_seed_all(1907); np.random.seed(1907); random.seed(1907)
    props = torch.cuda.get_device_properties(0)
    write_new(args.out_dir/'runtime.json', {'python':sys.version,'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu_name':props.name,'gpu_uuid':str(getattr(props,'uuid','')),
        'gpu_total_memory_bytes':props.total_memory,'weights_sha256':WEIGHTS,
        'nested_source_hashes':shared.SOURCE_HASHES,'model_source':'ModelScope LLM-Research/Llama-3.2-3B',
        'model_revision':'48d84dd14886f01964f72e7f742a404dd86f9fb6','frozen_backbone':True,
        'attention_backend':'eager','backbone_dtype':'BF16','mutable_dtype':'FP32',
        'gradient_clip_norm':1.,'CMS_pending_aggregation':'sum', 'test_opened':False})
    backbone = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval()
    model = shared.LlamaCMSLM(backbone, rank=16, capacity=8, levels=2)
    shared.promote_mutable_to_fp32(model)
    model.eval().requires_grad_(False)
    initial = shared.snapshot(model)
    meter = shared.WorkMeter()
    token = shared.ACTIVE_METER.set(meter)
    candidates, completed = [], 0
    current = None
    try:
        for index, (base_lr, expert_lr) in enumerate(protocol['lr_pairs']):
            check()
            shared.load_snapshot(model, initial)
            torch.manual_seed(1907); torch.cuda.manual_seed_all(1907); np.random.seed(1907); random.seed(1907)
            controller = shared.CMSController(model, periods=(4,16), base_lr=base_lr, expert_lr=expert_lr)
            current = {'index':index, 'base_lr':base_lr, 'expert_lr':expert_lr, 'updates_completed':0}
            cstart = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            meter.stage=f'candidate{index}_score'
            before = {'train':shared.score(model,train,pack['tokenizer']['pad_token_id']),
                      'calibration':shared.score(model,calibration,pack['tokenizer']['pad_token_id'])}
            for step, indices in enumerate(batches,1):
                check()
                event = shared.step(model,controller,[rows[i] for i in indices],pack['tokenizer']['pad_token_id'],
                                     meter,f'candidate{index}_foundation')
                completed += 1; current['updates_completed']=step
                if step%128==0:
                    print(json.dumps({'candidate':index,'step':step,'loss':event['loss']}),flush=True)
            meter.stage=f'candidate{index}_score'
            after = {'train':shared.score(model,train,pack['tokenizer']['pad_token_id']),
                     'calibration':shared.score(model,calibration,pack['tokenizer']['pad_token_id'])}
            torch.cuda.synchronize()
            score = {**current, 'before':before, 'after':after, 'seconds':time.perf_counter()-cstart,
                     'work':{k:deepcopy(v)for k,v in meter.counts.items()if k.startswith(f'candidate{index}_')},
                     'controller_counts':list(controller.counts)}
            score['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            write_new(args.out_dir/f'candidate-{index:02d}-scores.json',score)
            torch.save(shared.compact_cpu({'model':shared.snapshot(model),'controller':controller.state_dict(),
                'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
                       'python':random.getstate(),'numpy':np.random.get_state()},
                'seed':1907,'candidate':index,'records_sha256':DATA_SHA}),
                args.out_dir/f'candidate-{index:02d}-foundation.pt')
            score['qualifies']=(after['train']['full_vocab_two_token_greedy_exact_match']>=.8
                           and after['calibration']['full_vocab_two_token_greedy_exact_match']>=.75)
            candidates.append(score)
            write_new(args.out_dir/f'candidate-{index:02d}-qualification.json',score)
            print(json.dumps({'candidate':index,'train_em':after['train']['full_vocab_two_token_greedy_exact_match'],
                              'calibration_em':after['calibration']['full_vocab_two_token_greedy_exact_match'],
                              'qualifies':score['qualifies']}),flush=True)
        qualified = [r for r in candidates if r['qualifies']]
        selected = qualified[0]['index'] if qualified else None
        result = {'status':'FOUNDATION_QUALIFIED_PENDING_SLEEP_FIXTURE' if qualified else 'VALID_NO_FOUNDATION_QUALIFIED',
                  'selected_candidate':selected,'candidates':candidates,'updates_completed':completed,
                  'elapsed_seconds':time.time()-started,'work_total':meter.counts,
                  'official_test_opened':False,'main_started':False,'sleep_fixture_run':False}
        write_new(args.out_dir/'foundation-result.json',result)
    except BaseException as exc:
        write_new(args.out_dir/'failure.json',{'status':'ENGINEERING_OR_BUDGET_FAILURE',
             'error_type':type(exc).__name__,'error':str(exc),'updates_completed':completed,
             'current_candidate':current,'work_total':meter.counts,'elapsed_seconds':time.time()-started,
             'official_test_opened':False,'main_started':False,'automatic_retry':False})
        raise
    finally:
        shared.ACTIVE_METER.reset(token)


if __name__=='__main__':
    main()
