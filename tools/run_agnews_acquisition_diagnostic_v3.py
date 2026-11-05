"""Bounded train+calibration-only AG News old-phase acquisition diagnostic.
No official-test path exists. Reads only the frozen train-role JSON.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PUBLIC_ROOT=Path(r"D:/my_paper/NL_Sleep_Public_Candidate")
PUBLIC_SRC=(PUBLIC_ROOT/"src").resolve(); PUBLIC_TOOLS=(PUBLIC_ROOT/"tools").resolve()
sys.path.insert(0,str(PUBLIC_TOOLS)); sys.path.insert(0,str(PUBLIC_SRC))
import numpy as np, torch
from transformers import AutoModelForCausalLM
import run_x870e_synthetic_dev_v2 as shared
import agnews_records_v1 as records
from nested_sleep.optim import CMSController
DATA_SHA="7b5169daadcf7d4ccaefc9d0bb79f1a4eef92fc275fae0c3a7d593e06b7e9e59"
HELPER_SHA="6c7bf0a079a052eb87458010c1b55e52db1d9daa6dce99e55708b2da98385948"
RECORD_HELPER_SHA="afa35ba68366bd268b7c1a9ec4c1460fcf8f64da942d6cdba38fc4ecd8b9ee2e"
WEIGHTS={"model-00001-of-00002.safetensors":"584d8d3e3f82f7964955174dfe5e3b1cf117a9d859f022cfdf7fcb884856e002","model-00002-of-00002.safetensors":"4719a04514ec2f060240711b7c33ab21187cac730ecaba3040b7a0fd95a9cefb"}
SEEDS=(17,29,43); LR_PAIRS=((.005,.01),(.02,.04)); UPDATES=512; TRAIN_GATE=.80; CAL_GATE=.75; PERIODS=(4,16); HARD_SECONDS=5400
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def write_new(path,obj):
 with Path(path).open('x',encoding='utf-8',newline='\n') as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def labeled(doc,seed):
 m=doc['code_maps'][str(seed)]; return [{**r,'marker':m['marker_token_id'],'label':m['class_to_token_ids'][r['class_id']]} for r in doc['records']]
def class_metrics(model,rows,pad_id):
 overall=shared.score(model,rows,pad_id); out={}
 for c in (0,1):
  s=shared.score(model,[r for r in rows if r['class_id']==c],pad_id)
  out[str(c)]={'examples':s['examples'],'two_token_exact_match':s['full_vocab_two_token_greedy_exact_match'],'marker_accuracy':s['marker_greedy_accuracy'],'class_token_accuracy':s['class_token_greedy_accuracy'],'full_vocab_nll_nats_per_answer_token':s['full_vocab_nll_nats_per_answer_token']}
 overall['per_class']=out; return overall
def qualifies(train,cal):
 return train['full_vocab_two_token_greedy_exact_match']>=TRAIN_GATE and cal['full_vocab_two_token_greedy_exact_match']>=CAL_GATE and all(train['per_class'][str(c)]['two_token_exact_match']>0 for c in (0,1))
def permutation_targets(doc,seed,class_ids):
 mapping=doc['code_maps'][str(seed)]['class_to_token_ids']; return [mapping[c] for c in class_ids]
def train_step(model,ctl,rows,pad_id,meter,stage):
 device=next(model.parameters()).device; ctl.prepare_wake(model); meter.stage=stage
 ids,mask,lens,markers,labels=shared.collate(rows,device,pad_id); full,fullmask=shared.append_markers(ids,mask,lens,markers,pad_id); logits=model.logits(full,fullmask); ix=torch.arange(len(labels),device=device)
 first=logits[ix,lens-1].float(); second=logits[ix,lens].float(); loss=.5*(torch.nn.functional.cross_entropy(first,markers)+torch.nn.functional.cross_entropy(second,labels))
 if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss at {stage}')
 loss.backward(); norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0))
 proposals=ctl.collect(model)
 for level in sorted(proposals,reverse=True):ctl.apply(model,level,proposals[level])
 return {'loss':float(loss.detach()),'grad_norm':norm,'zero_grad':norm==0.,'marker_predictions':first.detach().argmax(-1).cpu().tolist(),'marker_targets':markers.detach().cpu().tolist(),'class_predictions':second.detach().argmax(-1).cpu().tolist(),'class_targets':labels.detach().cpu().tolist(),'nonpad_tokens':int(fullmask.sum())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--records',type=Path,required=True); ap.add_argument('--model-dir',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/agnews-acquisition-diagnostic-v3.json'); a=ap.parse_args()
 if a.out_dir.exists(): raise FileExistsError(a.out_dir)
 if sha(a.records)!=DATA_SHA: raise RuntimeError('frozen train-role records SHA mismatch')
 if sha(PUBLIC_TOOLS/'run_x870e_synthetic_dev_v2.py')!=HELPER_SHA: raise RuntimeError('pinned public shared helper SHA mismatch')
 if sha(Path(records.__file__).resolve())!=RECORD_HELPER_SHA: raise RuntimeError('pinned public records helper SHA mismatch')
 for name,meta in shared.SOURCE_HASHES.items():
  if not Path(meta['path']).resolve().is_relative_to(PUBLIC_SRC): raise RuntimeError(f'external nested_sleep binding refused: {name}')
 doc,summary=records.load_agnews_records(a.records)
 if doc['test_opened'] is not False or doc['phase_class_ids']!={'old':[0,1],'new':[2,3]}: raise RuntimeError('train-only data contract mismatch')
 cfg=json.loads(a.config.read_text(encoding='utf-8'))
 if cfg.get('schema')!='agnews-acquisition-diagnostic-v3' or cfg.get('seeds')!=list(SEEDS) or cfg.get('lr_pairs')!=[list(x) for x in LR_PAIRS] or cfg.get('updates_per_candidate')!=UPDATES or cfg.get('gates')!={'train_exact_match_min':TRAIN_GATE,'calibration_exact_match_min':CAL_GATE}: raise RuntimeError('config differs from frozen v3 plan')
 if cfg.get('records_sha256')!=DATA_SHA or cfg.get('official_test_in_runner') is not False: raise RuntimeError('config data/test boundary mismatch')
 for f,h in WEIGHTS.items():
  if sha(a.model_dir/f)!=h: raise RuntimeError(f'weight hash mismatch: {f}')
 for x in doc['tokenizer']['file_inventory']:
  if sha(a.model_dir/x['name'])!=x['sha256']: raise RuntimeError('tokenizer hash mismatch')
 a.out_dir.mkdir(parents=True,exist_ok=False); start=time.monotonic(); deadline=start+HARD_SECONDS
 write_new(a.out_dir/'protocol.json',{'schema':'agnews-acquisition-diagnostic-run-v3','started_utc':datetime.now(timezone.utc).isoformat(),'seeds':SEEDS,'lr_pairs':LR_PAIRS,'updates_per_candidate':UPDATES,'periods':PERIODS,'gates':cfg['gates'],'records_sha256':DATA_SHA,'record_summary':summary,'runner_sha256':sha(__file__),'shared_helper_sha256':HELPER_SHA,'records_helper_sha256':RECORD_HELPER_SHA,'nested_source_hashes':shared.SOURCE_HASHES,'weight_sha256':WEIGHTS,'official_test_opened':False,'automatic_retry':False,'budget':{'max_updates':len(SEEDS)*len(LR_PAIRS)*UPDATES,'hard_wall_seconds':HARD_SECONDS}})
 torch.use_deterministic_algorithms(True); torch.set_num_threads(4)
 if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no candidate training started')
 results=[]; meter=shared.WorkMeter(); token=shared.ACTIVE_METER.set(meter)
 try:
  for seed in SEEDS:
   for ci,(blr,elr) in enumerate(LR_PAIRS):
    if time.monotonic()>=deadline: raise TimeoutError('fixed v3 wall cap reached')
    seed_all(seed); backbone=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
    model=shared.LlamaCMSLM(backbone,rank=16,capacity=8,levels=2); shared.promote_mutable_to_fp32(model); model.eval().requires_grad_(False); initial=shared.snapshot(model); rows=labeled(doc,seed)
    train=[r for r in rows if r['partition']=='main' and r['role']=='old_train']; cal=[r for r in rows if r['partition']=='main' and r['role']=='old_calib']; sched=doc['schedules'][str(seed)]['foundation512']; pad=int(doc['tokenizer']['pad_token_id']); ctl=CMSController(model,periods=PERIODS,base_lr=blr,expert_lr=elr)
    code_map=doc['code_maps'][str(seed)]; before_fp=shared.fingerprint(model); meter.stage=f'seed{seed}_candidate{ci}_before_eval'; before=class_metrics(model,train,pad); before_cal=class_metrics(model,cal,pad); events=[]; t0=time.perf_counter(); marker_counts={}; class_target_counts={}; confusion={}; zero=0; checkpoints=[]
    for step,indices in enumerate(sched):
     if time.monotonic()>=deadline: raise TimeoutError(f'fixed cap seed={seed} candidate={ci} step={step}')
     ev=train_step(model,ctl,[rows[i] for i in indices],pad,meter,f'seed{seed}_candidate{ci}_train'); events.append(ev); zero+=int(ev['zero_grad'])
     for target in ev['marker_targets']: marker_counts[str(target)]=marker_counts.get(str(target),0)+1
     for pred,target in zip(ev['class_predictions'],ev['class_targets']):
      class_target_counts[str(target)]=class_target_counts.get(str(target),0)+1
      k=f'{target}:{pred}'; confusion[k]=confusion.get(k,0)+1
     if (step+1)%128==0: checkpoints.append({'step':step+1,'adapter_fingerprint':shared.fingerprint(model),'mean_loss_last_128':sum(x['loss'] for x in events[-128:])/128,'mean_grad_norm_last_128':sum(x['grad_norm'] for x in events[-128:])/128,'zero_grad_last_128':sum(x['zero_grad'] for x in events[-128:])})
    if ctl.counts!=[0,0]: raise RuntimeError(f'unconsumed CMS clocks: {ctl.counts}')
    meter.stage=f'seed{seed}_candidate{ci}_after_eval'
    final=class_metrics(model,train,pad); final_cal=class_metrics(model,cal,pad); state=shared.snapshot(model); parameter_names=set(dict(model.named_parameters())); delta=sum(float((state[k].float()-initial[k].float()).pow(2).sum()) for k in state if k in parameter_names)**.5
    per_class_pass=all(final['per_class'][str(c)]['two_token_exact_match']>0 for c in (0,1)); qualified=qualifies(final,final_cal)
    row={'seed':seed,'candidate':ci,'base_lr':blr,'expert_lr':elr,'updates':len(events),'code_map_audit':{'class_to_token_ids':code_map['class_to_token_ids'],'marker_token_id':code_map['marker_token_id'],'permutation_is_bijective':len(set(code_map['class_to_token_ids']))==4 and code_map['marker_token_id'] not in code_map['class_to_token_ids']},'before_fingerprint':before_fp,'after_fingerprint':shared.fingerprint(model),'parameter_delta_l2':delta,'train_before':before,'calibration_before':before_cal,'train_after':final,'calibration_after':final_cal,'train_gate':TRAIN_GATE,'calibration_gate':CAL_GATE,'per_class_nonzero_gate':per_class_pass,'qualifies':qualified,'loss_first_32':sum(x['loss'] for x in events[:32])/32,'loss_last_32':sum(x['loss'] for x in events[-32:])/32,'grad_norm_min':min(x['grad_norm'] for x in events),'grad_norm_mean':sum(x['grad_norm'] for x in events)/len(events),'grad_norm_max':max(x['grad_norm'] for x in events),'zero_grad_steps':zero,'marker_target_counts':marker_counts,'class_target_counts':class_target_counts,'class_confusion_counts':confusion,'fingerprint_by_128':checkpoints,'controller_counts':list(ctl.counts),'seconds':time.perf_counter()-t0,'work':{k:deepcopy(v) for k,v in meter.counts.items() if k.startswith(f'seed{seed}_candidate{ci}_')},'official_test_opened':False}
    from nested_sleep.lm import execution_state
    ck=a.out_dir/f'seed-{seed}-lr-{ci}-foundation.pt'; torch.save(shared.compact_cpu({'model':state,'execution_state':execution_state(model),'controller':ctl.state_dict(),'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()},'seed':seed,'candidate':ci,'base_lr':blr,'expert_lr':elr,'updates':UPDATES,'records_sha256':DATA_SHA,'runner_sha256':sha(__file__),'official_test_opened':False}),ck); row['checkpoint']=ck.name; row['checkpoint_sha256']=sha(ck); write_new(a.out_dir/f'seed-{seed}-lr-{ci}-result.json',row); results.append(row)
    print(json.dumps({'seed':seed,'candidate':ci,'train_em':final['full_vocab_two_token_greedy_exact_match'],'cal_em':final_cal['full_vocab_two_token_greedy_exact_match'],'qualified':qualified}),flush=True); del ctl,model,backbone
  global_pass=[i for i in range(len(LR_PAIRS)) if all(x['qualifies']for x in results if x['candidate']==i)]
  write_new(a.out_dir/'result.json',{'status':'GLOBAL_LR_QUALIFIED' if global_pass else 'NO_GLOBAL_LR_QUALIFIED','globally_qualified_lr_indices':global_pass,'candidates':results,'work_total':meter.counts,'official_test_opened':False,'next_budget_authorized':False})
 except BaseException as e:
  write_new(a.out_dir/'failure.json',{'status':'FAILED_UNSEALED','error_type':type(e).__name__,'error':str(e),'candidates_completed':len(results),'official_test_opened':False,'automatic_retry':False,'work_total':meter.counts}); raise
 finally: shared.ACTIVE_METER.reset(token)
if __name__=='__main__': main()
