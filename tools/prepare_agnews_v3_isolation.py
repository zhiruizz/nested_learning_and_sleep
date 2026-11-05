"""Freeze fresh train/validation and sealed holdout roles; no model scoring.

select: PyArrow environment. tokenize: pinned local tokenizer environment.
Old official test content is used only as a blacklist, never as new confirmation.
"""
import argparse,hashlib,json,random,unicodedata
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path

TRAIN_SHA='fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee'
OLD_PACK_SHA='7b5169daadcf7d4ccaefc9d0bb79f1a4eef92fc275fae0c3a7d593e06b7e9e59'
SALT='nl-sleep-agnews-v3-20261105-fresh'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def h(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,separators=(',',':'),allow_nan=False);f.write('\n')
def select(a):
 import pyarrow.parquet as pq
 assert sha(a.train)==TRAIN_SHA and sha(a.old_pack)==OLD_PACK_SHA
 old=read(a.old_pack);seen={r['record_id_sha256']for r in old['records']}
 official=read(a.old_test_candidates);assert official['source_test_sha256']=='71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb'
 assert official['source_rows']==7600 and len(official['records'])==7600
 seen|={r['record_id_sha256']for r in official['records']}
 groups=defaultdict(list);table=pq.read_table(a.train,columns=['text','label']);assert len(table)==120000
 for i,r in enumerate(table.to_pylist()):
  normalized=' '.join(unicodedata.normalize('NFKC',r['text']).split()).casefold()
  rid=hashlib.sha256(normalized.encode()).hexdigest()
  if rid not in seen:groups[rid].append({'record_id_sha256':rid,'source_index':i,'class_id':int(r['label']),'text':r['text']})
 buckets=defaultdict(list);conflicts=0
 for rid,rows in groups.items():
  if len({r['class_id']for r in rows})!=1:conflicts+=1;continue
  r=min(rows,key=lambda x:x['source_index']);r['selection_key']=hashlib.sha256((SALT+'\0'+rid).encode()).hexdigest();buckets[r['class_id']].append(r)
 records=[]
 for c in range(4):
  chosen=sorted(buckets[c],key=lambda r:r['selection_key'])[:4096];assert len(chosen)==4096;records+=chosen
 a.out.mkdir(parents=True,exist_ok=False)
 write(a.out/'candidates-private.json',{'schema':'agnews-v3-candidates-private','salt':SALT,'train_sha256':TRAIN_SHA,
  'old_pack_sha256':OLD_PACK_SHA,'old_test_candidates_sha256':sha(a.old_test_candidates),
  'excluded_old_role_ids':len(old['records']),'excluded_old_official_test_ids':7600,
  'normalized_conflicting_groups_dropped':conflicts,'records':records})
 write(a.out/'candidate-receipt.json',{'candidates_sha256':sha(a.out/'candidates-private.json'),
  'old_test_candidates_sha256':sha(a.old_test_candidates),'preparer_sha256':sha(__file__)})
 print(json.dumps({'status':'FRESH_CANDIDATES_SELECTED','candidates':len(records),'no_model_scoring':True}),flush=True)
def tokenize(a):
 from transformers import AutoTokenizer
 old=read(a.old_pack);assert sha(a.old_pack)==OLD_PACK_SHA
 receipt=read(a.out/'candidate-receipt.json')
 assert receipt['candidates_sha256']==sha(a.out/'candidates-private.json')
 assert receipt['old_test_candidates_sha256']==sha(a.old_test_candidates) and receipt['preparer_sha256']==sha(__file__)
 for item in old['tokenizer']['file_inventory']:assert sha(a.model_dir/item['name'])==item['sha256']
 tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=False)
 prefix=old['prompt']['prompt_prefix_token_ids'];cue=old['prompt']['answer_cue_token_ids']
 prior={r['prompt_sha256']for r in old['records']}
 official=read(a.old_test_candidates)
 for r in official['records']:
  prior.add(h(prefix+tokenizer.encode(r['text'],add_special_tokens=False)[:96]+cue))
 groups=defaultdict(list)
 for r in read(a.out/'candidates-private.json')['records']:
  article=tokenizer.encode(r['text'],add_special_tokens=False);tokens=prefix+article[:96]+cue;ps=h(tokens)
  if ps in prior:continue
  groups[ps].append({k:r[k]for k in ('record_id_sha256','source_index','class_id','selection_key')}|
    {'tokens':tokens,'prompt_sha256':ps})
 buckets=defaultdict(list);conflicts=0
 for rows in groups.values():
  if len({r['class_id']for r in rows})!=1:conflicts+=1;continue
  row=min(rows,key=lambda r:r['selection_key']);buckets[row['class_id']].append(row)
 records=[];holdout=[];labels={}
 for c in range(4):
  rows=sorted(buckets[c],key=lambda r:r['selection_key']);assert len(rows)>=640
  for start,count,role in ((0,256,'train'),(256,128,'validation'),(384,256,'holdout')):
   for row in rows[start:start+count]:
    row=dict(row);row.pop('selection_key');row['phase']='old'if c<2 else'new';row['role']=role
    if role=='holdout':
     labels[row['record_id_sha256']]=row.pop('class_id');row.pop('phase');holdout.append(row)
    else:records.append(row)
 assert len(records)==1536 and len(holdout)==1024
 all_rows=records+holdout
 assert len({r['record_id_sha256']for r in all_rows})==len(all_rows)
 assert len({r['prompt_sha256']for r in all_rows})==len(all_rows)
 assert not({r['prompt_sha256']for r in all_rows}&prior)
 random.Random(330055).shuffle(holdout)
 schedules={};pools={};maps={}
 for seed in (17,29,43):
  maps[str(seed)]=old['code_maps'][str(seed)]
  indexes={c:[i for i,r in enumerate(records)if r['class_id']==c and r['role']=='train']for c in range(4)}
  schedules[str(seed)]={}
  for name,classes,offset in [('foundation',[0,1],1),('new',[2,3],2)]:
   rng=random.Random(seed*10000+offset+330000);batches=[]
   for _ in range(4096):
    batch=[rng.choice(indexes[c])for c in classes for _ in range(2)];rng.shuffle(batch);batches.append(batch)
   schedules[str(seed)][name]=batches;schedules[str(seed)][name+'_sha256']=h(batches)
  pools[str(seed)]=[]
  for event in range(64,2049,64):
   rng=random.Random(seed*10000+event+333000)
   pools[str(seed)].append([rng.choice(indexes[c])for c in [0,2,3,1,1,3,2,0]*2+[0,2,1,3]*2])
 pack={'schema':'agnews-fresh-train-validation-v3','created_utc':datetime.now(timezone.utc).isoformat(),
  'source_train_sha256':TRAIN_SHA,'salt':SALT,'old_pack_sha256':OLD_PACK_SHA,'old_test_used_for_blacklist_only':True,
  'phase_class_ids':{'old':[0,1],'new':[2,3]},'tokenizer':old['tokenizer'],'prompt':old['prompt'],
  'records':records,'code_maps':maps,'schedules':schedules,'sleep_prefix_pools':pools,
  'final_holdout_path_or_labels_in_training_pack':False}
 write(a.out/'train-validation-private.json',pack)
 write(a.out/'holdout-prompts-sealed.json',{'schema':'agnews-v3-heldout-prompts','records':holdout})
 write(a.out/'holdout-labels-sealed.json',{'schema':'agnews-v3-heldout-labels','labels':labels})
 manifest={'schema':'agnews-v3-isolation-manifest','salt':SALT,'train_validation_sha256':sha(a.out/'train-validation-private.json'),
  'holdout_prompts_sha256':sha(a.out/'holdout-prompts-sealed.json'),'holdout_labels_sha256':sha(a.out/'holdout-labels-sealed.json'),
  'train_per_class':256,'validation_per_class':128,'holdout_per_class':256,
  'normalized_id_intersections_zero':True,'tokenized_prompt_intersections_zero':True,
  'excluded_old_role_count':2048,'excluded_old_official_test_count':7600,
  'prefix_conflict_groups_dropped':conflicts,'no_model_scoring':True,
  'interpretation':'fresh within-AGNews-train holdout; development informed by old official-test results; not external or pretraining-independent confirmation',
  'prepared_utc':datetime.now(timezone.utc).isoformat(),'preparer_sha256':sha(__file__)}
 write(a.out/'isolation-manifest.json',manifest);print(json.dumps(manifest),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['select','tokenize'],required=True)
 for n in ('train','old-pack','old-test-candidates','out'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--model-dir',type=Path);a=p.parse_args();select(a)if a.stage=='select'else tokenize(a)
if __name__=='__main__':main()
