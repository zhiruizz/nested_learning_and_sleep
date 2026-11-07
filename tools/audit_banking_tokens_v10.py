"""Tokenizer-only Banking train/dev audit. No model, GPU, or test interface."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path

HEAD='Classify the intent of this user request.\nRequest: '
TAIL='\nIntent:'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-pack',required=True,type=Path)
    p.add_argument('--model-path',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True,trust_remote_code=False)
    raw=a.training_pack.read_bytes();m=json.loads(raw)
    assert m['schema']=='banking-cil-train-dev-manifest-v10' and m['test_data_included'] is False
    enc=lambda s:t.encode(s,add_special_tokens=False)
    lengths=[len(enc(s)) for s in m['classes']]
    roundtrips=[t.decode(enc(s))==s for s in m['classes']]
    fixed=len(enc(HEAD))+len(enc(TAIL));body_budget=128-fixed-max(lengths)-1
    assert body_budget>0 and all(roundtrips)
    counts={};normalized={};prefix_mismatch=0;decoded_mismatch=0;decoded_no_cleanup_mismatch=0;rows=0
    for role in ('train','development'):
        f=a.training_pack.parent/m['files'][role]['path'];b=f.read_bytes()
        assert hashlib.sha256(b).hexdigest()==m['files'][role]['sha256']
        body=[];groups=set()
        for line in b.splitlines():
            r=json.loads(line);text=r['text'];body.append(len(enc(text)))
            groups.add(r['normalized_text_sha256']);rows+=1
            prompt=HEAD+text+TAIL
            segmented=enc(HEAD)+enc(text)+enc(TAIL)
            prefix_mismatch+=int(segmented!=enc(prompt))
            decoded_mismatch+=int(t.decode(segmented)!=prompt)
            decoded_no_cleanup_mismatch+=int(t.decode(segmented,clean_up_tokenization_spaces=False)!=prompt)
        assert len(body)==m['files'][role]['rows']
        counts[role]={'rows':len(body),'body_token_min':min(body),'body_token_max':max(body),
                      'requests_over_reserved_body_budget':sum(n>body_budget for n in body)}
        normalized[role]=groups
    assert normalized['train'].isdisjoint(normalized['development'])
    result={'schema':'banking-tokenization-audit-v10','training_manifest_sha256':hashlib.sha256(raw).hexdigest(),
       'class_count':len(lengths),'label_token_length_histogram':dict(sorted(Counter(lengths).items())),
       'label_roundtrip_count':sum(roundtrips),'label_length_max_with_eos':max(lengths)+1,
       'reserved_body_budget':body_budget,'max_sequence_length':128,'roles':counts,
       'segmented_vs_joint_different_token_ids':prefix_mismatch,'segmented_decoded_text_mismatches':decoded_mismatch,
       'decoded_text_mismatches_cleanup_disabled':decoded_no_cleanup_mismatch,
       'rows_audited':rows,'normalized_train_dev_overlap':0,'prepend_bos':False,
       'model_loaded':False,'gpu_initialized':False,'test_payload_accessed':False,
       'optimizer_updates':0}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))
if __name__=='__main__':main()
