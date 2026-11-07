"""Zero-update, fixed-dev allocator timing probe. Never modifies V10 code/config."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--training-pack',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--foundation-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allocator',choices=['default','gc70'],required=True)
    p.add_argument('--fraction',type=float,default=1.0)
    p.add_argument('--per-class',type=int,choices=[2,10],default=2)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if json.loads((a.foundation_root/'queue-status.json').read_text())['status']!='complete':
        raise ValueError('Wait for the running foundation queue; no concurrent GPU benchmark')
    if a.allocator=='gc70':os.environ['PYTORCH_CUDA_ALLOC_CONF']='garbage_collection_threshold:0.7'
    else:os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import run_banking_cil_v10 as d
    import banking_cil_common_v10 as common
    import banking_cil_v10_runtime as runtime
    import clinc_tail_runtime_v8 as tail
    from clinc_lifecycle_v8 import _state_hash
    raw_cfg=json.loads(a.config.read_text(encoding='utf-8'));cfg=d.validate_config(raw_cfg,phase='foundation')
    hashes,lock,manifest,train,dev=d._source_prepare(a.project_root,Path(__file__).resolve().parents[1],a.model_path,a.training_pack,cfg)
    rpath=a.foundation_root/'foundations/seed17/foundation-receipt.json';r=json.loads(rpath.read_text())
    assert r['status']=='qualified' and r['source_lock_sha256']==lock and r['config_sha256']==d._config_sha(cfg)
    cp=rpath.parent/r['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==r['checkpoint_sha256']
    import torch
    state=torch.load(cp,map_location='cpu',weights_only=False)
    if not 0<a.fraction<=1:raise ValueError('Memory fraction must be in (0,1]')
    if a.fraction<1:torch.cuda.set_per_process_memory_fraction(a.fraction,device=cfg['device'])
    d._init_rng(cfg['initialization_seed'],torch)
    torch,tok,model,immutable=d._load_model(a.project_root,a.model_path,cfg)
    model.requires_grad_(False);context=tail.clone_mutable_context(model);context.load_state_dict(state['context'])
    for param in context.parameters():param.requires_grad_(False)
    before=_state_hash(context);versions=immutable(model)
    order=manifest['class_orders']['17'];allowed=set(common.task_class_ids(order,0))
    if a.per_class==10:rows=[r for r in dev if r['intent_id']in allowed]
    else:
        rows=[]
        for cid in sorted(allowed):rows.extend(sorted((r for r in dev if r['intent_id']==cid),key=lambda r:r['id'])[:2])
    assert len(rows)==11*a.per_class
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    result=runtime.score_candidates(model,{'foundation':context},rows,tok,manifest['classes'],common,tail,
        torch=torch,device=cfg['device'],max_sequence_length=128,batch_size=32,include_greedy=False)
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    assert _state_hash(context)==before and immutable(model)==versions
    out={'schema':'banking-allocator-fixed-dev-benchmark-v10','allocator':a.allocator,
      'allocator_environment':os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),'rows':len(rows),'candidate_batch_size':32,
      'per_process_memory_fraction':a.fraction,
      'candidate_policy_unchanged':True,'metric_selection':False,'source_lock_sha256':lock,
      'training_manifest_sha256':cfg['training_manifest_sha256'],'context_sha256':before,
      'optimizer_updates':0,'test_payload_accessed':False,'measured_scoring_seconds':elapsed,
      'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
      'end_allocated_bytes':torch.cuda.memory_allocated(),'end_reserved_bytes':torch.cuda.memory_reserved(),
      'result':result,'limitations':'Timing on 22 fixed development rows only; allocator changes storage management, not model math, shapes, targets, or optimizer.'}
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in out.items()if k not in('result','limitations')}))
if __name__=='__main__':main()
