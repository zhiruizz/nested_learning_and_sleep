"""Build a training-derived, group-disjoint Banking77 pack; no test interface."""
from __future__ import annotations
import argparse,csv,hashlib,json,random,unicodedata
from collections import Counter,defaultdict
from pathlib import Path

SOURCE_SHA='b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b'
SOURCE_REVISION='9d081458ff52e53cf7e848f414e6e9344e4e6696'
SEEDS=(17,29,43)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def normalize_text(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def canonical_label(text):return unicodedata.normalize('NFKC',text).casefold().strip()
def rank_key(label,text_key,seed):return sha(f'banking77-v10-split:{seed}:{label}:{text_key}'.encode())

def split_records(records,*,dev_per_class=10,split_seed=20261107):
    """Group duplicate text first, quarantine conflicts, then split within class."""
    groups=defaultdict(list)
    for row in records:groups[normalize_text(row['text'])].append(row)
    unique=[];conflict_groups=conflict_rows=duplicate_rows=0
    for key,rows in groups.items():
        labels={canonical_label(row['source_label']) for row in rows}
        if len(labels)!=1:
            conflict_groups+=1;conflict_rows+=len(rows);continue
        representative=min(rows,key=lambda row:row['id'])
        unique.append({**representative,'normalized_text_sha256':sha(key.encode()),
                       'canonical_label':next(iter(labels))})
        duplicate_rows+=len(rows)-1
    classes=tuple(sorted({canonical_label(row['source_label']) for row in records}))
    by_class=defaultdict(list)
    for row in unique:by_class[row['canonical_label']].append(row)
    if any(len(by_class[label])<dev_per_class+16 for label in classes):
        raise ValueError('At least one class cannot support fixed dev plus 16 training rows')
    train=[];dev=[]
    for cid,label in enumerate(classes):
        ordered=sorted(by_class[label],key=lambda row:rank_key(label,row['normalized_text_sha256'],split_seed))
        for role,selected in [('development',ordered[:dev_per_class]),('train',ordered[dev_per_class:])]:
            for row in selected:
                packed={**row,'intent_id':cid,'role':role,'text_sha256':sha(row['text'].encode())}
                (dev if role=='development' else train).append(packed)
    train.sort(key=lambda row:row['id']);dev.sort(key=lambda row:row['id'])
    assert {r['normalized_text_sha256'] for r in train}.isdisjoint(r['normalized_text_sha256'] for r in dev)
    return classes,train,dev,{'source_rows':len(records),'unique_nonconflicting_rows':len(unique),
      'same_label_duplicate_rows_excluded':duplicate_rows,'conflicting_text_groups_quarantined':conflict_groups,
      'conflicting_source_rows_quarantined':conflict_rows,'train_counts_per_class':{label:len(by_class[label])-dev_per_class for label in classes},
      'dev_per_class':dev_per_class,'unused_unique_training_source_rows':len(unique)-len(train)-len(dev),
      'unique_counts_per_class':{label:len(by_class[label]) for label in classes}}

def prepare(source,out,*,dev_per_class=10,split_seed=20261107):
    if out.exists():raise FileExistsError('Output pack exists; no overwrite')
    raw=source.read_bytes()
    if sha(raw)!=SOURCE_SHA or len(raw)!=839073:raise ValueError('Original training source SHA/size mismatch')
    records=[]
    with source.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if set(reader.fieldnames or [])!={'text','category'}:raise ValueError('Unexpected original CSV columns')
        for index,row in enumerate(reader):
            if not row['text'].strip() or not row['category'].strip():raise ValueError('Empty text or source label')
            identity=sha(json.dumps([SOURCE_SHA,index,row['text'],row['category']],ensure_ascii=False,separators=(',',':')).encode())
            records.append({'id':identity,'text':row['text'],'source_label':row['category'],'source_row':index})
    source_labels=sorted({row['source_label'] for row in records})
    if len(records)!=10003 or len(source_labels)!=77:raise ValueError('Original corpus count/class mismatch')
    if len(set(map(canonical_label,source_labels)))!=77:raise ValueError('Source labels collide after canonicalization')
    classes,train,dev,audit=split_records(records,dev_per_class=dev_per_class,split_seed=split_seed)
    loose=lambda s:' '.join(s.replace('_',' ').split())
    if len(set(map(loose,classes)))!=77:raise ValueError('Prediction-normalized labels collide')
    out.mkdir(parents=True)
    files={}
    for role,rows in [('train',train),('development',dev)]:
        payload=''.join(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n' for row in rows).encode()
        name=f'{role}.jsonl'
        with (out/name).open('xb') as stream:stream.write(payload)
        files[role]={'path':name,'sha256':sha(payload),'rows':len(rows)}
    orders={}
    for seed in SEEDS:
        ids=list(range(77));random.Random(f'banking77-cil-v10:{seed}').shuffle(ids);orders[str(seed)]=ids
    manifest={'schema':'banking-cil-train-dev-manifest-v10','test_data_included':False,'derived_from_training_only':True,
      'num_classes':77,'classes':list(classes),'source_label_to_id':{label:classes.index(canonical_label(label)) for label in source_labels},
      'class_orders':orders,'files':files,
      'source':{'dataset':'PolyAI/banking77 original','revision':SOURCE_REVISION,'train_csv_sha256':SOURCE_SHA,
       'train_csv_bytes':len(raw),'train_csv_rows':10003,'license':'CC BY 4.0'},
      'split_policy':{'split_seed':split_seed,'normalized_text':'NFKC,casefold,whitespace-collapse; punctuation/underscore preserved',
       'grouped_before_split':True,'conflicting_label_groups':'quarantine all','same_label_duplicates':'one stable-ID representative',
       'development_per_class':dev_per_class,'training_selection':'all remaining unique nonconflicting training-source rows; no balancing, oversampling or truncation; minimum16 per class',
       'class_order_seeds':list(SEEDS)},'audit':audit}
    payload=(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
    with (out/'pack-manifest.json').open('xb') as stream:stream.write(payload)
    return {'status':'prepared','manifest_sha256':sha(payload),'train_rows':len(train),'development_rows':len(dev),
      'train_per_class_min':min(audit['train_counts_per_class'].values()),'train_per_class_max':max(audit['train_counts_per_class'].values()),'class_count':77,'audit':audit,
      'test_payload_accessed':False,'model_loaded':False,'optimizer_updates':0}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-train',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--dev-per-class',type=int,default=10)
    p.add_argument('--split-seed',type=int,default=20261107);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    if a.receipt.exists():raise FileExistsError(a.receipt)
    result=prepare(a.source_train,a.out_dir,dev_per_class=a.dev_per_class,split_seed=a.split_seed)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='audit'}))
if __name__=='__main__':main()
