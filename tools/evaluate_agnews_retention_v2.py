"""Post-seal official-test evaluation. No training, selection, or optimizer updates.

Run select with a PyArrow environment; tokenize/score with the model environment.
Every stage refuses an incomplete training seal before reading test content.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import unicodedata

TEST_SHA = '71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb'
TRAIN_PACK_SHA = '7b5169daadcf7d4ccaefc9d0bb79f1a4eef92fc275fae0c3a7d593e06b7e9e59'
SALT = 'nl-sleep-agnews-officialtest-v1'
SEEDS = (17, 29, 43)
ARMS = ('frozen', 'sft', 'cms', 'cms_sleep', 'cms_extra_replay')
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def obj_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def require_seal(run_dir):
    """Verify all 15 arms, budgets, endpoints, and every private checkpoint hash."""
    seal_path = run_dir / 'training-sealed.json'
    seal = read(seal_path)
    if seal.get('schema') != 'agnews-retention-training-seal-v2' or seal.get('complete') is not True:
        raise RuntimeError('official test stays sealed: main block incomplete')
    if seal.get('records_sha256') != TRAIN_PACK_SHA or seal.get('official_test_opened') is not False:
        raise RuntimeError('training seal provenance mismatch')
    protocol = read(run_dir / 'training-protocol.json')
    if sha(run_dir / 'training-protocol.json') != seal['protocol_sha256']:
        raise RuntimeError('training protocol hash mismatch')
    if protocol.get('status') != 'training_sealed' or protocol.get('completed_runs') != 15:
        raise RuntimeError('unsealed training protocol')
    if isinstance(seal['runs'], list):
        runs = seal['runs']
        seal['runs'] = {f'{r["seed"]}/{r["arm"]}':r for r in runs}
        if len(seal['runs']) != len(runs):
            raise RuntimeError('duplicate runs')
    if isinstance(seal['foundation_checkpoints'], list):
        foundations = seal['foundation_checkpoints']
        seal['foundation_checkpoints'] = {str(r['seed']):r for r in foundations}
        if len(seal['foundation_checkpoints']) != len(foundations):
            raise RuntimeError('duplicate foundations')
    if set(seal['runs']) != {f'{s}/{a}' for s in SEEDS for a in ARMS}:
        raise RuntimeError('missing/extra seed-arm runs')
    if set(seal['foundation_checkpoints']) != {str(s) for s in SEEDS}:
        raise RuntimeError('missing foundation')
    def checked_path(relative, expected):
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()) or sha(path) != expected:
            raise RuntimeError(f'checkpoint outside run directory or hash mismatch: {relative}')
    wake, sleep = 0, 0
    for foundation in seal['foundation_checkpoints'].values():
        if foundation['status'] != 'complete' or foundation['updates'] != 512:
            raise RuntimeError('incomplete foundation')
        checked_path(foundation['checkpoint'], foundation['checkpoint_sha256'])
        wake += 512
    for key, run in seal['runs'].items():
        arm = key.split('/')[1]
        primary = 0 if arm == 'frozen' else 256
        replay = 48 if arm == 'cms_extra_replay' else 0
        adam = 96 if arm == 'cms_sleep' else 0
        if (run['status'], run['primary_updates'], run['extra_replay_updates'], run['sleep_adam_updates']) != (
                'complete', primary, replay, adam):
            raise RuntimeError(f'incomplete budget in {key}')
        expected_steps = [0, 256] if arm == 'frozen' else [0, 64, 128, 192, 256]
        if [c['base_step'] for c in run['checkpoints']] != expected_steps:
            raise RuntimeError(f'checkpoint trajectory mismatch: {key}')
        foundation = seal['foundation_checkpoints'][key.split('/')[0]]
        if run['checkpoints'][0]['adapter_fingerprint'] != foundation['adapter_fingerprint']:
            raise RuntimeError('arm did not fork common foundation')
        for cp in run['checkpoints']:
            checked_path(cp['path'], cp['sha256'])
        wake += primary + replay
        sleep += adam
    if (wake, sleep) != (4752, 288):
        raise RuntimeError('whole-block budget mismatch')
    return seal, sha(seal_path)


def load_train_pack(path):
    if sha(path) != TRAIN_PACK_SHA:
        raise RuntimeError('train role pack changed')
    return read(path)


def select_test(args, seal_hash):
    import pyarrow.parquet as pq
    if sha(args.test_parquet) != TEST_SHA:
        raise RuntimeError('official test source SHA mismatch')
    pack = load_train_pack(args.records)
    excluded = {r['record_id_sha256'] for r in pack['records']}
    groups = defaultdict(list)
    table = pq.read_table(args.test_parquet, columns=['text', 'label'])
    if len(table) != 7600:
        raise RuntimeError('unexpected official test size')
    for index, row in enumerate(table.to_pylist()):
        text, label = row['text'], int(row['label'])
        if not isinstance(text, str) or label not in range(4):
            raise RuntimeError('test schema mismatch')
        normalized = ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()
        rid = hashlib.sha256(normalized.encode()).hexdigest()
        groups[rid].append({'record_id_sha256': rid, 'source_index': index,
                            'class_id': label, 'text': text})
    records, conflicts, excluded_count = [], 0, 0
    for rid, rows in groups.items():
        if len({r['class_id'] for r in rows}) != 1:
            conflicts += 1
            continue
        if rid in excluded:
            excluded_count += 1
            continue
        row = min(rows, key=lambda r: r['source_index'])
        row['selection_key'] = hashlib.sha256((SALT+'\0'+rid).encode()).hexdigest()
        records.append(row)
    result = {'schema': 'agnews-postseal-test-candidates-private-v2', 'seal_sha256': seal_hash,
              'source_test_sha256': TEST_SHA, 'records_sha256': TRAIN_PACK_SHA,
              'source_rows': len(table), 'text_duplicate_rows': len(table)-len(groups),
              'text_conflict_groups': conflicts, 'excluded_training_texts': excluded_count,
              'salt': SALT, 'records': records, 'evaluator_sha256': sha(__file__)}
    write_new(args.out_dir/'test-candidates-private.json', result)
    write_new(args.out_dir/'test-candidates-receipt.json', {
        'seal_sha256':seal_hash,'evaluator_sha256':sha(__file__),
        'candidates_sha256':sha(args.out_dir/'test-candidates-private.json')})
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)


def tokenize_test(args, seal_hash):
    from transformers import AutoTokenizer
    pack = load_train_pack(args.records)
    receipt = read(args.out_dir/'test-candidates-receipt.json')
    if receipt != {'seal_sha256':seal_hash,'evaluator_sha256':sha(__file__),
                   'candidates_sha256':sha(args.out_dir/'test-candidates-private.json')}:
        raise RuntimeError('test candidates changed after selection')
    source = read(args.out_dir/'test-candidates-private.json')
    if source['seal_sha256'] != seal_hash or source['source_test_sha256'] != TEST_SHA:
        raise RuntimeError('test candidates are not bound to this sealed training block')
    if source['evaluator_sha256'] != sha(__file__):
        raise RuntimeError('evaluator changed between test preparation stages')
    for entry in pack['tokenizer']['file_inventory']:
        if sha(args.model_dir/entry['name']) != entry['sha256']:
            raise RuntimeError('tokenizer source mismatch')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    prompt = pack['prompt']
    prefix, cue = prompt['prompt_prefix_token_ids'], prompt['answer_cue_token_ids']
    groups = defaultdict(list)
    for row in source['records']:
        article = tokenizer.encode(row['text'], add_special_tokens=False)
        tokens = prefix + article[:96] + cue
        psha = obj_sha(tokens)
        groups[psha].append({**{k:row[k] for k in ('record_id_sha256','class_id','source_index','selection_key')},
                             'tokens': tokens, 'prompt_sha256': psha})
    excluded = {r['prompt_sha256'] for r in pack['records']}
    buckets, conflicts, excluded_count = defaultdict(list), 0, 0
    for psha, rows in groups.items():
        if len({r['class_id'] for r in rows}) != 1:
            conflicts += 1
            continue
        if psha in excluded:
            excluded_count += 1
            continue
        buckets[rows[0]['class_id']].append(min(rows, key=lambda r: r['selection_key']))
    selected = []
    for label in range(4):
        rows = sorted(buckets[label], key=lambda r:r['selection_key'])[:128]
        if len(rows) != 128:
            raise RuntimeError('not enough disjoint test prompts; no fallback')
        for row in rows:
            row.pop('selection_key')
        selected.extend(rows)
    result = {'schema': 'agnews-postseal-test-pack-private-v2', 'seal_sha256': seal_hash,
              'source_test_sha256': TEST_SHA, 'train_records_sha256': TRAIN_PACK_SHA,
              'salt': SALT, 'examples_per_class': 128, 'total_examples': 512,
              'prompt_conflict_groups': conflicts, 'excluded_training_prefixes': excluded_count,
              'text_selection_audit': {k:v for k,v in source.items() if k != 'records'},
              'records': selected, 'evaluator_sha256': sha(__file__)}
    write_new(args.out_dir/'test-pack-private.json', result)
    write_new(args.out_dir/'test-pack-receipt.json', {
        'seal_sha256':seal_hash,'evaluator_sha256':sha(__file__),
        'pack_sha256':sha(args.out_dir/'test-pack-private.json')})
    print(json.dumps({'status':'TEST_PACK_FROZEN_AFTER_SEAL','examples':len(selected),
                      'sha256':sha(args.out_dir/'test-pack-private.json'),
                      'excluded_training_prefixes':excluded_count}), flush=True)


def score_test(args, seal, seal_hash):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    sys.path.insert(0, str(ROOT/'src'))
    sys.path.insert(0, str(ROOT/'tools'))
    import time
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    import run_x870e_synthetic_dev_v2 as shared
    from nested_sleep.lm import LlamaCMSLM, load_execution_state, promote_mutable_to_fp32
    from nested_sleep.lifecycle import fingerprint
    from nested_sleep.accounting import WorkMeter, ACTIVE_METER
    for entry in shared.SOURCE_HASHES.values():
        if not Path(entry['path']).resolve().is_relative_to((ROOT/'src').resolve()):
            raise RuntimeError('external implementation import refused')
    training_protocol = read(args.run_dir/'training-protocol.json')
    expected_sources = training_protocol['nested_source_hashes']
    for name, entry in expected_sources.items():
        if name in shared.SOURCE_HASHES and shared.SOURCE_HASHES[name]['sha256'] != entry['sha256']:
            raise RuntimeError('implementation changed since training')
    pack = load_train_pack(args.records)
    receipt = read(args.out_dir/'test-pack-receipt.json')
    if receipt != {'seal_sha256':seal_hash,'evaluator_sha256':sha(__file__),
                   'pack_sha256':sha(args.out_dir/'test-pack-private.json')}:
        raise RuntimeError('test pack changed after freeze')
    test_pack = read(args.out_dir/'test-pack-private.json')
    if test_pack['seal_sha256'] != seal_hash or test_pack['source_test_sha256'] != TEST_SHA:
        raise RuntimeError('test pack/seal mismatch')
    if test_pack['evaluator_sha256'] != sha(__file__):
        raise RuntimeError('evaluator changed after test pack freeze')
    records = test_pack['records']
    if len(records)!=512 or Counter(r['class_id'] for r in records)!={c:128 for c in range(4)}:
        raise RuntimeError('frozen test class counts changed')
    ids, prompts = {r['record_id_sha256'] for r in records}, {r['prompt_sha256'] for r in records}
    if len(ids)!=512 or len(prompts)!=512:
        raise RuntimeError('duplicate frozen test IDs/prompts')
    if ids & {r['record_id_sha256'] for r in pack['records']} or prompts & {r['prompt_sha256'] for r in pack['records']}:
        raise RuntimeError('test pack overlaps a training/calibration role')
    prefix, cue = pack['prompt']['prompt_prefix_token_ids'],pack['prompt']['answer_cue_token_ids']
    for row in records:
        tokens=row['tokens']
        if obj_sha(tokens)!=row['prompt_sha256'] or not len(prefix)+len(cue)<len(tokens)<=len(prefix)+96+len(cue):
            raise RuntimeError('test prompt hash/length mismatch')
        if tokens[:len(prefix)]!=prefix or tokens[-len(cue):]!=cue:
            raise RuntimeError('test prompt composition mismatch')
    for name, expected in training_protocol['weight_sha256'].items():
        if sha(args.model_dir/name) != expected:
            raise RuntimeError('base model weight mismatch')
    for entry in pack['tokenizer']['file_inventory']:
        if sha(args.model_dir/entry['name']) != entry['sha256']:
            raise RuntimeError('tokenizer assets changed')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = LlamaCMSLM(AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager').eval(),
        rank=16, capacity=8, levels=2)
    promote_mutable_to_fp32(model)
    model.eval().requires_grad_(False)
    meter = WorkMeter()
    meter_token = ACTIVE_METER.set(meter)
    started = time.monotonic()
    deadline = started+3600
    def evaluate_rows(rows, seed, stage):
        code = pack['code_maps'][str(seed)]
        enriched = [{**r, 'marker':code['marker_token_id'],
                     'label':code['class_to_token_ids'][r['class_id']]} for r in rows]
        data = []
        meter.stage = stage
        with torch.no_grad():
            for offset in range(0,len(rows),8):
                if time.monotonic() > deadline:
                    raise TimeoutError('post-seal evaluation cap; no score-dependent extension')
                batch = enriched[offset:offset+8]
                ids, mask, lengths, markers, labels = shared.collate(batch, 'cuda', pack['tokenizer']['pad_token_id'])
                full, full_mask = shared.append_markers(ids, mask, lengths, markers, pack['tokenizer']['pad_token_id'])
                logits = model.logits(full, full_mask)
                idx = torch.arange(len(batch), device=ids.device)
                first, second = logits[idx,lengths-1].float(), logits[idx,lengths].float()
                ml = F.cross_entropy(first,markers,reduction='none').cpu().tolist()
                cl = F.cross_entropy(second,labels,reduction='none').cpu().tolist()
                mp, cp = first.argmax(-1),second.argmax(-1)
                candidates = second[:,code['class_to_token_ids']].argmax(-1).cpu().tolist()
                markers_ok, classes_ok = (mp==markers).cpu().tolist(),(cp==labels).cpu().tolist()
                for row,m,c,mok,cok,candidate in zip(batch,ml,cl,markers_ok,classes_ok,candidates):
                    data.append({'class_id':row['class_id'],'marker_nll':m,'class_nll':c,
                                 'marker_accuracy':int(mok),'class_token_accuracy_given_gold_marker':int(cok),
                                 'full_vocab_two_token_greedy_exact_match':int(mok and cok),
                                 'four_candidate_accuracy_given_gold_marker':int(candidate==row['class_id'])})
        groups = {'old':[0,1],'new':[2,3],'all':[0,1,2,3], **{f'class{c}':[c] for c in range(4)}}
        result = {}
        for key, classes in groups.items():
            part = [r for r in data if r['class_id'] in classes]
            if part:
                result[key] = {'examples':len(part), **{metric:sum(r[metric] for r in part)/len(part)
                    for metric in data[0] if metric!='class_id'}}
                result[key]['answer_nll_nats_per_token'] = (result[key]['marker_nll']+result[key]['class_nll'])/2
        return result
    results, qualifications = {}, {}
    def restore(relative, expected):
        path = args.run_dir/relative
        if sha(path) != expected:
            raise RuntimeError('checkpoint changed immediately before scoring')
        cp = torch.load(path,map_location='cpu',weights_only=False)
        if cp['records_sha256'] != TRAIN_PACK_SHA or cp['config_sha256'] != seal['config_sha256']:
            raise RuntimeError('checkpoint provenance mismatch')
        shared.load_snapshot(model,cp['model'])
        load_execution_state(model,cp['execution_state'])
        model.eval().requires_grad_(False)
        return path, fingerprint(model)
    try:
        for seed in SEEDS:
            foundation = seal['foundation_checkpoints'][str(seed)]
            path,before = restore(foundation['checkpoint'],foundation['checkpoint_sha256'])
            old_train = [r for r in pack['records'] if r['partition']=='main' and r['role']=='old_train']
            old_cal = [r for r in pack['records'] if r['partition']=='main' and r['role']=='old_calib']
            tr = evaluate_rows(old_train,seed,f'{seed}_foundation_train')
            ca = evaluate_rows(old_cal,seed,f'{seed}_foundation_calibration')
            qualifications[str(seed)] = {'train':tr,'calibration':ca,
                'qualified_fixed_dev_criterion':tr['old']['full_vocab_two_token_greedy_exact_match']>=.8
                    and ca['old']['full_vocab_two_token_greedy_exact_match']>=.75}
            if fingerprint(model)!=before or sha(path)!=foundation['checkpoint_sha256']:
                raise RuntimeError('foundation evaluation mutated checkpoint/model')
            for arm in ARMS:
                key=f'{seed}/{arm}'
                results[key]=[]
                for cp in seal['runs'][key]['checkpoints']:
                    path,before=restore(cp['path'],cp['sha256'])
                    metrics=evaluate_rows(test_pack['records'],seed,f'{key}_step{cp["base_step"]}')
                    if fingerprint(model)!=before or sha(path)!=cp['sha256']:
                        raise RuntimeError('evaluation mutated checkpoint/model')
                    item={'base_step':cp['base_step'],'checkpoint_sha256':cp['sha256'], 'metrics':metrics}
                    results[key].append(item)
                    write_new(args.out_dir/f'score-{seed}-{arm}-{cp["base_step"]:03d}.json',item)
                    print(json.dumps({'seed':seed,'arm':arm,'step':cp['base_step'],'status':'scored'}),flush=True)
        summary={'schema':'agnews-retention-postseal-evaluation-v2','seal_sha256':seal_hash,
                 'test_pack_sha256':sha(args.out_dir/'test-pack-private.json'),'test_source_sha256':TEST_SHA,
                 'evaluator_sha256':sha(__file__),'sample_counts':{str(c):128 for c in range(4)},
                 'results':results,'foundation_qualifications':qualifications,
                 'evaluation_seconds':time.monotonic()-started,'evaluation_work':meter.counts,
                 'seed_independent_test_samples':False,'no_seeds_excluded':True,
                 'raw_text_or_token_ids_in_summary':False,'trained_after_test_open':False,
                 'created_utc':datetime.now(timezone.utc).isoformat()}
        write_new(args.out_dir/'evaluation-summary.json',summary)
    finally:
        ACTIVE_METER.reset(meter_token)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['select','tokenize','score'],required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--test-parquet',type=Path)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    seal,seal_hash=require_seal(args.run_dir)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=='select':
        if args.test_parquet is None: parser.error('--test-parquet required for select')
        select_test(args,seal_hash)
    elif args.stage=='tokenize':
        if args.model_dir is None: parser.error('--model-dir required for tokenize')
        tokenize_test(args,seal_hash)
    else:
        if args.model_dir is None: parser.error('--model-dir required for score')
        score_test(args,seal,seal_hash)


if __name__=='__main__':
    main()
