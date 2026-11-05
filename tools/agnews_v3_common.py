"""Frozen task mechanics for fresh-role qualification, comparison and scoring."""
from pathlib import Path
import hashlib,json,os,random,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
import torch.nn.functional as F
import run_agnews_retention_v2 as engine
from nested_sleep.lm import execution_state,load_execution_state
from nested_sleep.lifecycle import fingerprint
DATA_SHA='41d967950b9472eb791e858102e560a5601799fca789ac53d8b42395b5e93fc7'
ENGINE_SHA='498c317d876d7e37fdead90ad81057a2f164d24fe7867e3d00bb6abca449216b'
CORE_SHA='1fd3164e705c79d30d9c5666c055ef64b027672e547b8374708263066275df02'
SEEDS=(17,29,43);CAPACITY=32;LR=(.02,.04);EM='full_vocab_two_token_greedy_exact_match'
def sha(path):return engine.sha256(Path(path))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def check_sources():
 assert sha(ROOT/'tools/run_agnews_retention_v2.py')==ENGINE_SHA
 assert sha(ROOT/'src/nested_sleep/lifecycle.py')==CORE_SHA
 engine.ensure_sources()
def load_pack(path):
 assert sha(path)==DATA_SHA
 d=read(path);assert d['schema']=='agnews-fresh-train-validation-v3'
 assert len(d['records'])==1536 and d['final_holdout_path_or_labels_in_training_pack']is False
 for c in range(4):
  assert sum(r['class_id']==c and r['role']=='train'for r in d['records'])==256
  assert sum(r['class_id']==c and r['role']=='validation'for r in d['records'])==128
 return d
def preflight(model_dir,pack):
 check_sources()
 for n,h in engine.WEIGHT_SHA256.items():assert sha(model_dir/n)==h
 for x in pack['tokenizer']['file_inventory']:assert sha(model_dir/x['name'])==x['sha256']
 torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
def model(model_dir,seed):
 from transformers import AutoModelForCausalLM
 engine.seed_all(seed)
 m=engine.LlamaCMSLM(AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
  dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval(),rank=16,capacity=CAPACITY,levels=2)
 engine.promote_mutable_to_fp32(m);m.eval().requires_grad_(False);return m
def rows(pack,seed,role=None,phase=None):
 code=pack['code_maps'][str(seed)]
 return [{**r,'marker':code['marker_token_id'],'label':code['class_to_token_ids'][r['class_id']]}for r in pack['records']
  if (role is None or r['role']==role)and(phase is None or r['phase']==phase)]
def batch(pack,seed,indices,device='cuda'):
 code=pack['code_maps'][str(seed)]
 chosen=[{**pack['records'][i],'marker':code['marker_token_id'],
          'label':code['class_to_token_ids'][pack['records'][i]['class_id']]}for i in indices]
 ids,mask,lengths,markers,labels=engine.shared.collate(chosen,device,pack['tokenizer']['pad_token_id'])
 ids,mask=engine.shared.append_markers(ids,mask,lengths,markers,pack['tokenizer']['pad_token_id'])
 return ids,mask,lengths,markers,labels
def restore(m,cp):
 assert set(cp['model'])==set(m.adapter_state_dict()),'mutable checkpoint topology mismatch'
 engine.shared.load_snapshot(m,cp['model']);load_execution_state(m,cp['execution_state'])
 m.eval().requires_grad_(False)
def checkpoint(path,m,controller,seed,step,extra=None):
 state=engine.shared.compact_cpu({'model':m.adapter_state_dict(),'execution_state':execution_state(m),
  'controller':None if controller is None else controller.state_dict(),'rng':engine.capture_rng(),
  'seed':seed,'step':step,'records_sha256':DATA_SHA,'model_config':{'rank':16,'capacity':CAPACITY,'levels':2},
  **(extra or{})})
 torch.save(state,path);return {'path':Path(path).name,'sha256':sha(path),'adapter_fingerprint':fingerprint(m),
   'step':step,'controller_counts':None if controller is None else list(controller.counts)}
def score(m,records,pack,seed,stage,meter):
 meter.stage=stage;data=[];code=pack['code_maps'][str(seed)];labels4=code['class_to_token_ids']
 m.eval();device=next(m.parameters()).device
 with torch.no_grad():
  for off in range(0,len(records),8):
   b=[{**r,'marker':code['marker_token_id'],'label':labels4[r['class_id']]}for r in records[off:off+8]]
   ids,mask,lens,markers,labels=engine.shared.collate(b,device,pack['tokenizer']['pad_token_id'])
   ids,mask=engine.shared.append_markers(ids,mask,lens,markers,pack['tokenizer']['pad_token_id'])
   logits=m.logits(ids,mask);i=torch.arange(len(b),device=device)
   first,second=logits[i,lens-1].float(),logits[i,lens].float()
   ml=F.cross_entropy(first,markers,reduction='none').cpu().tolist();cl=F.cross_entropy(second,labels,reduction='none').cpu().tolist()
   mok=(first.argmax(-1)==markers).cpu().tolist();cok=(second.argmax(-1)==labels).cpu().tolist()
   candidates=second[:,labels4].argmax(-1).cpu().tolist()
   for r,x,y,a,b,c in zip(b,ml,cl,mok,cok,candidates):data.append({'class_id':r['class_id'],EM:int(a and b),
    'marker_accuracy':int(a),'class_given_marker_accuracy':int(b),'candidate_accuracy':int(c==r['class_id']),
    'marker_nll':x,'class_nll':y,'answer_nll_nats_per_token':(x+y)/2})
 out={}
 for name,classes in {'old':[0,1],'new':[2,3],'all':[0,1,2,3],**{f'class{c}':[c]for c in range(4)}}.items():
  subset=[r for r in data if r['class_id']in classes]
  if subset:out[name]={'examples':len(subset),**{k:sum(r[k]for r in subset)/len(subset)for k in data[0]if k!='class_id'}}
 return out
def qualifies(train,validation,phase='old'):
 classes=[0,1]if phase=='old'else[2,3]
 return train[phase][EM]>=.80 and validation[phase][EM]>=.75 and all(validation[f'class{c}'][EM]>=.70 for c in classes)
