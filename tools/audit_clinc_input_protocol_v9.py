"""Count CLINC train/dev prompt reachability; no model, GPU, or test input."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model-path','manifest','master-config','out']:
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import clinc_cil_common_v8 as common
    import run_clinc_cil_v8_full as full
    from transformers import AutoTokenizer
    cfg=full.validate_master_config(json.loads(a.master_config.read_text(encoding='utf-8')))
    assert common.sha256_file(a.manifest)==cfg['training_manifest_sha256']
    view,train,dev,clean,_partition,dupes,_r2=full._read_inputs(a.manifest,cfg)
    tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True,trust_remote_code=False,use_fast=True)
    enc=lambda text:tok.encode(text,add_special_tokens=False)
    head,tail=enc(common.PROMPT_HEAD),enc(common.PROMPT_TAIL)
    longest=max(len(enc(label)) for label in view.classes)
    body_budget=cfg['max_sequence_length']-len(head)-len(tail)-longest-1
    result={}
    for name,rows in [('train',train),('development',dev)]:
        lengths=[len(enc(row['text'])) for row in rows]
        result[name]={'rows':len(rows),'body_tokens_min':min(lengths),
          'body_tokens_mean':statistics.mean(lengths),'body_tokens_max':max(lengths),
          'rows_with_truncated_body':sum(n>body_budget for n in lengths)}
    # Stable dev rows, chosen without predictions, check the actual shared train/eval prefix.
    samples=sorted(clean,key=lambda row:row['id'])[:100]
    train_eval_match=joint_diff=decoded_diff=0
    for row in samples:
        prefix=common.tokenize_prompt(row['text'],tok,view.classes,tok.eos_token_id,cfg['max_sequence_length'])
        supervised=common.tokenize_row(row,tok,view.classes,tok.eos_token_id,cfg['max_sequence_length'])
        ids=prefix['input_ids']
        train_eval_match+=int(supervised['input_ids'][:supervised['prompt_length']]==ids)
        joint=enc(common.build_prompt(row['text']))
        joint_diff+=int(joint!=ids)
        decoded_diff+=int(tok.decode(joint,skip_special_tokens=True)!=tok.decode(ids,skip_special_tokens=True))
    import torch
    assert not torch.cuda.is_initialized()
    record={'schema':'clinc-v9-input-protocol-audit-v1','development_only':True,
      'test_payload_read':False,'model_loaded':False,'gpu_initialized':False,'optimizer_steps':0,
      'training_manifest_sha256':common.sha256_file(a.manifest),
      'master_config_sha256':common.sha256_file(a.master_config),
      'max_sequence_length':cfg['max_sequence_length'],'max_label_plus_eos':longest+1,
      'prompt_head_tokens':len(head),'prompt_tail_tokens':len(tail),'body_budget_tokens':body_budget,
      'prepend_bos':False,'normalized_train_dev_overlap_rows_excluded':dupes,
      'row_length_audit':result,'prefix_probe':{'sample_rows':len(samples),
       'actual_train_eval_prefix_matches':train_eval_match,
       'segmented_vs_joint_tokenization_different':joint_diff,
       'segmented_vs_joint_decoded_text_different':decoded_diff},
      'interpretation':['No accuracy or optimization performed; length checks do not establish a cause of low scores.',
       'Both frozen training and generation use the same segmented, gold-independent prompt prefix.',
       'Different token segmentation from standard joint encoding is a model-format assumption; its performance effect requires a new train/dev control.',
       'Any prompt or BOS change would be a new protocol, not a correction to the exposed v8 test result.']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(record))

if __name__=='__main__':main()
