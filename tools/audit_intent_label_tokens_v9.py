"""Audit public label metadata with a local tokenizer; no dataset rows or model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().replace('_',' ').split())
def audit(labels,tokenizer):
    sizes=[]
    failures=0
    for label in labels:
        ids=tokenizer.encode(label,add_special_tokens=False)
        sizes.append(len(ids))
        failures+=int(norm(tokenizer.decode(ids,skip_special_tokens=True))!=norm(label))
    return {'class_count':len(labels),'label_token_length_histogram':dict(sorted(Counter(sizes).items())),
      'longest_label_tokens':max(sizes),'longest_label_plus_eos':max(sizes)+1,
      'labels_longer_than_v8_sleep_cap8':sum(n>8 for n in sizes),
      'labels_plus_eos_longer_than_cap8':sum(n+1>8 for n in sizes),
      'normalized_label_roundtrip_failures':failures,
      'class_names_unique_after_normalization':len(set(map(norm,labels)))==len(labels)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True,type=Path)
    p.add_argument('--clinc-manifest',required=True,type=Path)
    p.add_argument('--banking-metadata',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True,trust_remote_code=False,use_fast=True)
    clinc=json.loads(a.clinc_manifest.read_text(encoding='utf-8'))
    assert clinc['schema']=='clinc-cil-training-manifest-v8' and clinc['test_data_included'] is False
    bank=json.loads(a.banking_metadata.read_text(encoding='utf-8'))['default']
    banklabels=bank['features']['label']['names']
    assert len(clinc['classes'])==150 and len(banklabels)==77
    output={'schema':'intent-label-token-metadata-audit-v9','model_loaded':False,'gpu_initialized':False,
      'dataset_payload_rows_read':False,'optimizer_steps':0,'development_diagnostic_only':True,
      'clinc':audit(clinc['classes'],tok),'banking77':audit(banklabels,tok),
      'source_metadata_sha256':{'clinc_train_development_manifest':sha(a.clinc_manifest),
       'banking77_dataset_infos':sha(a.banking_metadata),'local_tokenizer_json':sha(a.model_path/'tokenizer.json')},
      'interpretation':['Token-length coverage is a necessary reachability check, not an accuracy result.',
       'V8 primary greedy evaluation uses longest canonical label plus EOS; its Sleep rollout cap is separately eight.',
       'Banking metadata is public schema only; this does not download or parse any train/test payload.',
       'Any future task-specific token cap must be frozen before validation scoring; do not tune it on exposed CLINC test.']}
    import torch
    assert not torch.cuda.is_initialized()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'clinc':output['clinc'],'banking77':output['banking77']}))

if __name__=='__main__':main()
