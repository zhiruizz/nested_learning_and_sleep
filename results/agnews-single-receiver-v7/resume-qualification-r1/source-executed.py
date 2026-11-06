"""Candidate v7 single-context Sleep-mechanism diagnostic (not paper reproduction).

This candidate is intentionally not frozen or runnable until its config is
reviewed and the separate real-model qualification passes. It contains no
confirmation-data path. MatmulWork reports only observed matrix-multiply FLOPs.
"""
from __future__ import annotations

import argparse, hashlib, json, random
from pathlib import Path

SEEDS=(17,29,43)
SCOPES=("receiver","shared")
BASE_ARMS=("current","split","split_narrow","gold","new_only")
ARMS=tuple(f"{arm}_{scope}" for arm in BASE_ARMS for scope in SCOPES)
STEPS=1024
EVAL_STEPS=(0,64,256,1024)
BATCH_CLASSES=(0,0,1,1,2,2,3,3)
NARROW_OLD_PER_CLASS=16
_QUAL_UPDATES=0
_QUAL_METER=None
_OPTIMIZER_ATTEMPTS=0
_STUDY_METER=None
_STUDY_PREFIX='startup'


def audit_stage(stage,phase='forward'):
    if _STUDY_METER is not None:
        _STUDY_METER.stage=_STUDY_PREFIX+':'+stage
        _STUDY_METER.phase=phase


def validate_input_locks(args,cfg):
    import agnews_v3_common as c
    import agnews_tail_cache_v7 as helper
    import matmul_work_v7 as counter
    for file,expected in ((Path(__file__),cfg['driver_sha256']),(Path(helper.__file__),cfg['cache_helper_sha256']),
                          (Path(counter.__file__),cfg['matmul_counter_sha256']),(args.records,cfg['records_sha256']),
                          (args.warm_dir/'transfer-result.json',cfg['warm_result_sha256']),
                          (args.v6_dir/'development-sealed.json',cfg['v6_seal_sha256'])):
        if c.sha(file)!=expected:raise ValueError(f'input/source lock mismatch: {file.name}')
    warm=c.read(args.warm_dir/'transfer-result.json');v6=c.read(args.v6_dir/'development-sealed.json')
    for seed in SEEDS:
        lock=cfg['checkpoint_sha256_by_seed'][str(seed)]
        old=next(x['checkpoint'] for x in warm['qualified_models'] if x['seed']==seed)
        run=next(x for x in v6['runs'] if x['seed']==seed and x['arm']=='aux_new_CE')
        new=_checkpoint_row(run,1024)
        if lock!={'warm':old['sha256'],'current_v6_1024':new['sha256']}:raise ValueError('frozen checkpoint metadata mismatch')


def class_token(classmap, cls):
    cls=int(cls)
    if isinstance(classmap,dict):
        return int(classmap[cls] if cls in classmap else classmap[str(cls)])
    return int(classmap[cls])

def hash_json(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def class_schedule(train_rows, seed, arm, steps=STEPS):
    """Deterministic balanced batch schedule; narrow maps only old pools to 16/class."""
    pools={c:[i for i,r in enumerate(train_rows) if int(r["class_id"])==c] for c in range(4)}
    if any(r.get("role")!="train" for r in train_rows): raise ValueError("schedule input must be train-only rows")
    narrow="split_narrow" in arm
    new_only=arm.startswith("new_only_")
    if narrow:
        for cls in (0,1):
            pools[cls]=sorted(pools[cls],key=lambda i:hashlib.sha256(f"v7-narrow:{seed}:{cls}:{i}".encode()).digest())[:NARROW_OLD_PER_CLASS]
    if any(len(pools[c]) < 16 for c in range(4)):
        raise ValueError("training class pool too small")
    pattern=(2,2,2,2,3,3,3,3) if new_only else BATCH_CLASSES
    out=[]
    for step in range(steps):
        ids=[None]*len(pattern);processed=set()
        for cls in pattern:
            if cls in processed: continue
            processed.add(cls)
            count=pattern.count(cls)
            # A class-local stream ensures narrow old sampling cannot change
            # the new-class examples; sample without replacement per batch.
            rng=random.Random(f"v7-schedule:{seed}:{step}:{cls}")
            selected=rng.sample(pools[cls],count)
            positions=[j for j,x in enumerate(pattern) if x==cls]
            for j,v in zip(positions,selected): ids[j]=v
        out.append(ids)
    return out

def auxiliary_classes(arm):
    if arm.startswith("new_only_"): return (2,3)
    return (0,1,2,3)

def gold_answer_targets(pack,seed,rows,indices):
    code=pack["code_maps"][str(seed)]; marker=int(code["marker_token_id"]); classmap=code["class_to_token_ids"]
    return [[marker,class_token(classmap,rows[i]['class_id'])] for i in indices]

def validate_arm_scope(arm, trainable_names):
    """Fail closed: only explicitly permitted slow/base adapter tensors train."""
    try: family,scope=arm.rsplit("_",1)
    except ValueError as e: raise ValueError("unknown arm") from e
    if family not in BASE_ARMS or scope not in SCOPES: raise ValueError("unknown arm")
    names=set(trainable_names)
    if not names: raise ValueError("empty trainable scope")
    # The model adapter naming is pinned by the root freeze config; never accept
    # heads, backbone, or old slow slot zero even if their names seem adapter-like.
    # Actual names include the frozen transformer backbone; permission is
    # allowlist-based, so no substring ban on `backbone` is appropriate.
    if any(".slots.0." in n for n in names): raise ValueError("old slow slot zero is forbidden")
    if any(not (".slots.1.a" in n or ".slots.1.b" in n or n.endswith(".memory.0.a") or n.endswith(".memory.0.b") or n.endswith(".memory.1.a") or n.endswith(".memory.1.b")) for n in names):
        raise ValueError("scope contains a parameter outside the new slow slot/base")
    if scope=="receiver" and any(n.endswith((".memory.0.a",".memory.0.b",".memory.1.a",".memory.1.b")) for n in names): raise ValueError("receiver scope contains shared base")
    return True

def parameter_scope(model, scope):
    """Return exact approved parameter names for the second (slow) level."""
    desc=model.level_descriptors()
    if len(desc)<2: raise ValueError("model does not expose two memory levels")
    d=desc[1]; bank=d["bank"]
    allowed={f"{bank}.slots.1.{ab}" for ab in ("a","b")}
    if scope=="shared":
        allowed.update(desc[0].get("base",[])); allowed.update(d.get("base",[]))
    elif scope!="receiver": raise ValueError("unknown trainable scope")
    names={n for n,_ in model.named_parameters() if n in allowed}
    if names!=allowed: raise ValueError(f"adapter allowlist mismatch: expected {sorted(allowed)}, found {sorted(names)}")
    return allowed

def set_parameter_scope(model, scope):
    allowed=parameter_scope(model,scope)
    for name,p in model.named_parameters(): p.requires_grad_(name in allowed)
    actual={n for n,p in model.named_parameters() if p.requires_grad}
    if actual!=allowed: raise AssertionError("trainable scope did not match exact allowlist")
    return allowed

def training_classes(arm):
    return (2,3) if arm.startswith("new_only_") else (0,1,2,3)

def cached_tail_logits(model, cache, total_length):
    """Single point of indirection for parity-sensitive tail evaluation."""
    import agnews_tail_cache_v7 as cache_helper
    fn=getattr(cache_helper,"tail_logits_padded",None)
    if fn is None: raise RuntimeError("R3 parity helper tail_logits_padded is required")
    return fn(model,cache["y"],cache["residual"],cache["lengths"],total_length)

def capture_cached(model, ids, mask, lengths, device="cpu"):
    import agnews_tail_cache_v7 as cache_helper
    result=cache_helper.capture(model,ids,mask,lengths,device=device)
    if set(result)!={"y","residual","logits"}: raise ValueError("cache helper schema mismatch")
    return result

def make_model_batch(rows, pack, seed, device, total_length):
    """Build marker-conditioned inputs, then pad every phase to the locked T."""
    import torch
    import agnews_v3_common as c
    code=pack["code_maps"][str(seed)]
    tagged=[{**r,"marker":code["marker_token_id"],"label":code["class_to_token_ids"][int(r["class_id"])]} for r in rows]
    ids,mask,lengths,markers,labels=c.engine.shared.collate(tagged,device,pack["tokenizer"]["pad_token_id"])
    ids,mask=c.engine.shared.append_markers(ids,mask,lengths,markers,pack["tokenizer"]["pad_token_id"])
    if ids.shape[1]>total_length: raise ValueError("input exceeds locked padded length")
    pad=total_length-ids.shape[1]
    if pad:
        ids=torch.nn.functional.pad(ids,(0,pad),value=pack["tokenizer"]["pad_token_id"])
        mask=torch.nn.functional.pad(mask,(0,pad),value=0)
    return ids,mask,lengths,markers,labels

def score_logits(logits, rows, pack, seed):
    """Canonical Python-float metrics for cached-tail and fixed-T full logits."""
    import torch
    import torch.nn.functional as F
    code=pack["code_maps"][str(seed)]; ids4=[class_token(code['class_to_token_ids'],k) for k in range(4)]
    labels=torch.tensor([class_token(code['class_to_token_ids'],r['class_id']) for r in rows],device=logits.device)
    marker=torch.full_like(labels,int(code["marker_token_id"]))
    first=logits[:,0].float(); second=logits[:,1].float()
    mce=F.cross_entropy(first,marker,reduction="none").detach().cpu().tolist()
    cce=F.cross_entropy(second,labels,reduction="none").detach().cpu().tolist()
    mok=(first.argmax(-1)==marker).detach().cpu().tolist(); cok=(second.argmax(-1)==labels).detach().cpu().tolist()
    cand=second[:,ids4].argmax(-1).detach().cpu().tolist()
    vals=[]
    for i,r in enumerate(rows):
        cls=int(r["class_id"]); vals.append({"class_id":cls,"em":int(mok[i] and cok[i]),"marker":int(mok[i]),"class":int(cok[i]),"candidate":int(cand[i]==cls),'marker_nll':float(mce[i]),'class_nll':float(cce[i]),"nll":(float(mce[i])+float(cce[i]))/2})
    out={}
    for name,cs in {"old":(0,1),"new":(2,3),"all":(0,1,2,3),**{f"class{k}":(k,) for k in range(4)}}.items():
        subset=[x for x in vals if x["class_id"] in cs]
        if subset: out[name]={"examples":len(subset),**{k:sum(float(x[k]) for x in subset)/len(subset) for k in ("em","marker","class","candidate","marker_nll","class_nll","nll")}}
    return out

def evaluate_cached(model, feature_cache, rows, indices, pack, seed, total_length, batch_size=8, meter=None):
    import torch
    parts=[]
    audit_stage('evaluation')
    for st in range(0,len(indices),batch_size):
        ids=indices[st:st+batch_size]
        y=torch.cat([feature_cache[i]["y"] for i in ids]).to(next(model.parameters()).device)
        residual=torch.cat([feature_cache[i]["residual"] for i in ids]).to(y.device)
        lengths=torch.cat([feature_cache[i]["lengths"] for i in ids]).to(y.device)
        with torch.no_grad():
            if meter is None: z=cached_tail_logits(model,{"y":y,"residual":residual,"lengths":lengths},total_length)
            else:
                meter.stage="evaluation";meter.phase="forward"
                with meter: z=cached_tail_logits(model,{"y":y,"residual":residual,"lengths":lengths},total_length)
        parts.append(z.detach())
    return score_logits(torch.cat(parts),[rows[i] for i in indices],pack,seed)

def reset_student_context(model):
    """Build the shared v7 start context from the v6 checkpoint."""
    banks=model.banks()
    if len(banks)!=2: raise ValueError("expected exactly fast and slow banks")
    banks[0].reset_active()
    if bool(banks[1].slots[1].active):
        raise ValueError("v6 slow slot 1 must be inactive at the common start")
    banks[1].slots[0].requires_grad_(False)
    banks[1].activate(1)
    banks[1].route_override={"mix":1,"fraction":0.5}
    model.eval()
    return model

def make_features(model, rows, pack, seed, total_length, batch_size=8, meter=None):
    """Capture immutable prefix features once; store CPU tensors and lengths."""
    import torch
    cache={}; device=next(model.parameters()).device
    audit_stage('cache_capture')
    for start in range(0,len(rows),batch_size):
        batch_rows=rows[start:start+batch_size]
        ids,mask,lengths,_,_=make_model_batch(batch_rows,pack,seed,device,total_length)
        if meter is None: cap=capture_cached(model,ids,mask,lengths,device="cpu")
        else:
            meter.stage="cache_capture";meter.phase="forward"
            with meter: cap=capture_cached(model,ids,mask,lengths,device="cpu")
        for j in range(len(batch_rows)):
            cache[start+j]={k:cap[k][j:j+1].contiguous() for k in ("y","residual","logits")}
            cache[start+j]["lengths"]=lengths[j:j+1].detach().cpu().long()
    return cache

def fit_one_batch(model, optimizer, rows, indices, cache, teacher_logits, pack, seed, arm, total_length, meter=None):
    """One AdamW update, with explicit old/new teacher routing and gold controls."""
    import torch
    audit_stage('train')
    device=next(model.parameters()).device
    y=torch.cat([cache[i]["y"] for i in indices]).to(device)
    residual=torch.cat([cache[i]["residual"] for i in indices]).to(device)
    lengths=torch.cat([cache[i]["lengths"] for i in indices]).to(device)
    student=cached_tail_logits(model,{"y":y,"residual":residual,"lengths":lengths},total_length)
    family=arm.rsplit("_",1)[0]
    if family in ("current","split","split_narrow"):
        target=torch.cat([teacher_logits[i] for i in indices],dim=0).to(device)
        loss=distill_kl(student,target)
    else:
        targets=torch.tensor(gold_answer_targets(pack,seed,rows,indices),device=device)
        loss=answer_ce(student,targets)
    if not torch.isfinite(loss): raise RuntimeError('nonfinite distillation/CE loss')
    optimizer.zero_grad(set_to_none=True)
    if meter is not None: meter.stage="train";meter.phase="backward"
    audit_stage('train','backward')
    loss.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(params,1.0)
    if not torch.isfinite(norm): raise RuntimeError('nonfinite training gradient')
    if float(norm)==0.0 and abs(float(loss.detach()))>1e-6: raise RuntimeError('nonzero objective without trainable gradient')
    global _OPTIMIZER_ATTEMPTS
    _OPTIMIZER_ATTEMPTS+=1
    audit_stage('optimizer','optimizer')
    optimizer.step()
    return float(loss.detach().cpu())


def model_guards(model,allowed):
    import agnews_v3_common as c
    from nested_sleep.lm import execution_state,immutable_versions
    state=model.adapter_state_dict()
    return {'frozen_adapters_sha256':c.engine.shared.tensor_tree_hash({k:v for k,v in state.items() if k not in allowed}),
            'policy_sha256':c.engine.shared.tensor_tree_hash(execution_state(model)),
            'immutable_parameter_versions':immutable_versions(model)}


def old_teacher_passes(train,val):
    return train['old']['em']>=.80 and val['old']['em']>=.75 and all(val[f'class{k}']['em']>=.70 for k in (0,1))

def _checkpoint_row(run, step):
    xs=[x for x in run.get("checkpoints",[]) if int(x.get("primary_step",x.get("step",-1)))==step]
    if len(xs)!=1: raise ValueError(f"expected one checkpoint at step {step}")
    return xs[0]

def _load_cp(base, meta):
    import torch, agnews_v3_common as c
    path=Path(meta["path"]); path=path if path.is_absolute() else Path(base)/path
    if c.sha(path)!=meta["sha256"]: raise ValueError("checkpoint SHA mismatch")
    return torch.load(path,map_location="cpu",weights_only=False),path

def _full_fixed_logits(model, rows, pack, seed, total_length, meter=None):
    import torch
    device=next(model.parameters()).device; parts=[]
    audit_stage('full_model_parity')
    for st in range(0,len(rows),8):
        ids,mask,lengths,_,_=make_model_batch(rows[st:st+8],pack,seed,device,total_length)
        if meter is None: cap=capture_cached(model,ids,mask,lengths,device=device)
        else:
            meter.stage="full_model_parity";meter.phase="forward"
            with meter: cap=capture_cached(model,ids,mask,lengths,device=device)
        parts.append(cap["logits"].detach())
    return torch.cat(parts)

def _run_resume_qualification(args):
    import torch, copy
    import agnews_v3_common as c
    import agnews_tail_cache_v7 as ch
    from matmul_work_v7 import MatmulWork
    from nested_sleep.lm import execution_state,load_execution_state
    import time
    global _QUAL_UPDATES,_QUAL_METER
    _QUAL_UPDATES=0; qstart=time.monotonic()
    cfg=c.read(args.config);pack=c.load_pack(args.records);warm=c.read(args.warm_dir/"transfer-result.json");v6=c.read(args.v6_dir/"development-sealed.json")
    validate_input_locks(args,cfg)
    if c.sha(Path(__file__))!=cfg["driver_sha256"] or c.sha(args.records)!=cfg["records_sha256"]: raise ValueError("qualification source/records lock mismatch")
    if c.sha(args.warm_dir/"transfer-result.json")!=cfg["warm_result_sha256"] or c.sha(args.v6_dir/"development-sealed.json")!=cfg["v6_seal_sha256"]: raise ValueError("qualification teacher seal lock mismatch")
    c.preflight(args.model_dir,pack)
    T=int(cfg["padded_sequence_length"]);seed=17
    nr=next(x for x in v6["runs"] if x["seed"]==seed and x["arm"]=="aux_new_CE")
    cp,_=_load_cp(args.v6_dir,_checkpoint_row(nr,1024));wr=next(x for x in warm["qualified_models"] if x["seed"]==seed);wcp,_=_load_cp(args.warm_dir,wr["checkpoint"])
    qmeter=MatmulWork();_QUAL_METER=qmeter;train=c.rows(pack,seed,role="train"); nt=c.model(args.model_dir,seed);c.restore(nt,cp);nt.eval().requires_grad_(False)
    nc=make_features(nt,train,pack,seed,T,meter=qmeter);feats=nc;teacher={i:nc[i]["logits"] for i in range(len(train))};del nt
    wm=c.model(args.model_dir,seed);c.restore(wm,wcp);wm.eval().requires_grad_(False);old=[i for i,r in enumerate(train) if int(r["class_id"])<2];wt={}
    for st in range(0,len(old),8):
        audit_stage('qualification_warm_teacher')
        ix=old[st:st+8];ids,mask,lens,_,_=make_model_batch([train[i] for i in ix],pack,seed,next(wm.parameters()).device,T);qmeter.stage="qualification_warm_teacher";qmeter.phase="forward"
        with qmeter: cap=capture_cached(wm,ids,mask,lens,"cpu")
        wt.update({i:cap["logits"][j:j+1] for j,i in enumerate(ix)})
    del wm
    schedule=class_schedule(train,seed,"split_receiver",steps=4);results=[]
    for scope in SCOPES:
        model=c.model(args.model_dir,seed);c.restore(model,cp);c.engine.restore_rng(cp["rng"]);reset_student_context(model);set_parameter_scope(model,scope)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003,weight_decay=0.)
        def update(target_model,target_opt,step):
            global _QUAL_UPDATES
            if time.monotonic()-qstart>=600: raise TimeoutError("v7 resume qualification 600s active-work cap")
            idx=schedule[step-1];targets={i:(wt[i] if int(train[i]["class_id"])<2 else teacher[i]) for i in idx}
            qmeter.stage="qualification_training";qmeter.phase="forward"
            with qmeter: fit_one_batch(target_model,target_opt,train,idx,feats,targets,pack,seed,"split_"+scope,T,qmeter)
            _QUAL_UPDATES+=1
        update(model,opt,1);update(model,opt,2)
        saved={"model":c.engine.shared.compact_cpu(model.adapter_state_dict()),"execution_state":execution_state(model),"optimizer":copy.deepcopy(opt.state_dict()),"rng":c.engine.capture_rng()}
        pending_path=args.out_dir/f"pending-{scope}-step2.pt";torch.save(saved,pending_path);pending_sha=c.sha(pending_path);del model,opt
        branch_states=[]
        for _ in range(2):
            saved=torch.load(pending_path,map_location="cpu",weights_only=False)
            branch=c.model(args.model_dir,seed);c.restore(branch,cp);c.engine.restore_rng(cp["rng"]);reset_student_context(branch);set_parameter_scope(branch,scope)
            load_execution_state(branch,saved["execution_state"]);c.engine.shared.load_snapshot(branch,saved["model"])
            bo=torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad],lr=.003,weight_decay=0.);bo.load_state_dict(copy.deepcopy(saved["optimizer"]));c.engine.restore_rng(saved["rng"])
            update(branch,bo,3);update(branch,bo,4)
            branch_states.append({"model_sha256":c.engine.shared.tensor_tree_hash(branch.adapter_state_dict()),"optimizer_sha256":c.engine.shared.tensor_tree_hash(bo.state_dict()),"execution_state_sha256":c.engine.shared.tensor_tree_hash(execution_state(branch)),"rng_sha256":c.engine.shared.tensor_tree_hash(c.engine.capture_rng())})
            del branch,bo
        exact=branch_states[0]==branch_states[1]
        results.append({"scope":scope,"exact_model_optimizer_policy_rng":exact,"exact_model_optimizer_rng":exact,"branches":branch_states,"actual_updates":6,"pending_checkpoint":{"path":pending_path.name,"sha256":pending_sha}})
    result={"schema":"agnews-single-receiver-resume-qualified-v7","passed":all(x["exact_model_optimizer_policy_rng"] for x in results),"exact_model_optimizer_policy_rng":all(x["exact_model_optimizer_policy_rng"] for x in results),"exact_model_optimizer_rng":all(x["exact_model_optimizer_policy_rng"] for x in results),"config_sha256":c.sha(args.config),"driver_sha256":c.sha(Path(__file__)),"actual_Adam_updates":12,"matmul_work_lower_bound":qmeter.state_dict(),"scoring":False,"confirmation_scored":False,"results":results}
    c.write(args.out_dir/"qualification.json",result)

def _run_main(args):
    import torch, torch.nn.functional as F
    import agnews_v3_common as c
    import agnews_tail_cache_v7 as cache_helper
    from nested_sleep.lm import execution_state
    from matmul_work_v7 import MatmulWork
    cfg=c.read(args.config); pack=c.load_pack(args.records)
    validate_input_locks(args,cfg)
    if cfg.get("status")!="frozen_development": raise ValueError("config is not frozen")
    if tuple(cfg.get("seeds",()))!=SEEDS or tuple(cfg.get("arms",()))!=ARMS: raise ValueError("config arms/seeds mismatch")
    if cfg.get("steps")!=STEPS or tuple(cfg.get("evaluation_steps",()))!=EVAL_STEPS: raise ValueError("fixed-step config mismatch")
    if float(cfg.get("learning_rate",.003))!=.003 or float(cfg.get("weight_decay",0.0))!=0.0 or float(cfg.get("gradient_clip",1.0))!=1.0: raise ValueError("optimizer contract mismatch")
    if float(cfg.get("elapsed_limit_seconds",28800))!=28800 or float(cfg.get("per_run_elapsed_limit_seconds",900))!=900: raise ValueError("frozen elapsed caps mismatch")
    total_length=int(cfg["padded_sequence_length"])
    if total_length!=118: raise ValueError("expected locked fixed T=118")
    warm=c.read(args.warm_dir/"transfer-result.json"); v6=c.read(args.v6_dir/"development-sealed.json")
    q=c.read(args.qualification)
    if not q.get("passed") or not q.get("exact_model_optimizer_policy_rng"): raise ValueError("resume qualification gate failed")
    if q.get("config_sha256")!=c.sha(args.config): raise ValueError("qualification config hash mismatch")
    if q.get("driver_sha256")!=c.sha(Path(__file__)): raise ValueError("resume qualification source mismatch")
    for key in ("cache","loss_paths"):
        item=cfg["numerical_qualifications"][key];qp=Path(item["path"])
        if c.sha(qp)!=item["sha256"]: raise ValueError(f"{key} qualification SHA mismatch")
        qv=c.read(qp)
        if not qv.get("passed"): raise ValueError(f"{key} numerical qualification failed")
    if c.sha(Path(__file__))!=cfg["driver_sha256"]: raise ValueError("driver source lock mismatch")
    if c.sha(Path(cache_helper.__file__))!=cfg["cache_helper_sha256"]: raise ValueError("cache helper lock mismatch")
    if cfg["records_sha256"]!=c.sha(args.records): raise ValueError("records lock mismatch")
    if cfg["warm_result_sha256"]!=c.sha(args.warm_dir/"transfer-result.json") or cfg["v6_seal_sha256"]!=c.sha(args.v6_dir/"development-sealed.json"): raise ValueError("teacher/result source lock mismatch")
    c.preflight(args.model_dir,pack)
    args.out_dir.mkdir(parents=True,exist_ok=False); (args.out_dir/"checkpoints").mkdir();(args.out_dir/"cache-private").mkdir()
    protocol={"schema":"agnews-single-receiver-protocol-v7","status":"frozen_development","config_sha256":c.sha(args.config),"driver_sha256":c.sha(Path(__file__)),"cache_helper_sha256":c.sha(Path(cache_helper.__file__)),"records_sha256":c.sha(args.records),"warm_result_sha256":c.sha(args.warm_dir/"transfer-result.json"),"v6_seal_sha256":c.sha(args.v6_dir/"development-sealed.json"),"qualification_sha256":c.sha(args.qualification),"numerical_qualifications":cfg["numerical_qualifications"],"seeds":list(SEEDS),"arms":list(ARMS),"steps":STEPS,"evaluation_steps":list(EVAL_STEPS),"padded_sequence_length":total_length,"confirmation_input":False,"confirmation_scored":False,"exploratory_only":True}
    c.write(args.out_dir/"protocol.json",protocol); protocol_sha=c.sha(args.out_dir/"protocol.json")
    def progress(value):
        with (args.out_dir/"progress.json").open("w",encoding="utf-8",newline="\n") as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    def time_guard(run_start=None):
        now=__import__('time').monotonic()
        if now-start>=limit: raise TimeoutError("v7 global active-work cap")
        if run_start is not None and now-run_start>=900: raise TimeoutError("v7 per-run active-work cap")
    runs=[]; teacher_quality=[];preparation_meters={};common_initial={}
    start=__import__('time').monotonic(); limit=float(cfg.get("elapsed_limit_seconds",28800))
    def checkpoint(model,opt,seed,arm,step,row):
        path=args.out_dir/"checkpoints"/f"{seed}-{arm}-{step}.pt"
        torch.save({"schema":"agnews-single-receiver-checkpoint-v7","model":c.engine.shared.compact_cpu(model.adapter_state_dict()),"execution_state":c.engine.shared.compact_cpu(execution_state(model)),"optimizer":c.engine.shared.compact_cpu(opt.state_dict()),"rng":c.engine.capture_rng(),"seed":seed,"arm":arm,"step":step,"schedule_sha256":row["schedule_sha256"],"protocol_sha256":protocol_sha},path)
        row.setdefault("checkpoints",[]).append({"path":str(path.relative_to(args.out_dir)),"sha256":c.sha(path),"step":step})
    try:
      for seed in SEEDS:
        global _STUDY_PREFIX
        _STUDY_PREFIX=f'seed{seed}:preparation'
        print(f"v7 cache seed={seed}",flush=True)
        train=c.rows(pack,seed,role="train"); val=c.rows(pack,seed,role="validation")
        newrun=next(x for x in v6["runs"] if x["seed"]==seed and x["arm"]=="aux_new_CE")
        newmeta=_checkpoint_row(newrun,1024); newcp,_=_load_cp(args.v6_dir,newmeta)
        warmrow=next(x for x in warm["qualified_models"] if x["seed"]==seed)
        warmcp,_=_load_cp(args.warm_dir,warmrow["checkpoint"])
        lock=cfg["checkpoint_sha256_by_seed"][str(seed)]
        if lock["warm"]!=warmrow["checkpoint"]["sha256"] or lock["current_v6_1024"]!=newmeta["sha256"]: raise ValueError("checkpoint input lock mismatch")
        new_teacher_model=c.model(args.model_dir,seed); c.restore(new_teacher_model,newcp); c.engine.restore_rng(newcp["rng"]); new_teacher_model.eval().requires_grad_(False)
        print(f"v7 capture seed={seed} T={total_length}",flush=True)
        prep_meter=MatmulWork()
        preparation_meters[str(seed)]=prep_meter
        cache=make_features(new_teacher_model,train+val,pack,seed,total_length,meter=prep_meter)
        ntrain=len(train); trainidx=list(range(ntrain)); validx=list(range(ntrain,ntrain+len(val)))
        new_teacher={i:cache[i]["logits"].clone() for i in trainidx}
        new_teacher_val_metrics=score_logits(torch.cat([cache[i]["logits"] for i in validx]),val,pack,seed)
        new_teacher_train_metrics=score_logits(torch.cat([new_teacher[i] for i in trainidx]),train,pack,seed)
        del new_teacher_model
        # Warm teacher outputs are needed only on old training rows in split arms.
        warm_model=c.model(args.model_dir,seed); c.restore(warm_model,warmcp); c.engine.restore_rng(warmcp["rng"]); warm_model.eval().requires_grad_(False)
        old_ids=[i for i,r in enumerate(train) if int(r["class_id"]) in (0,1)]
        old_teacher={}
        for st in range(0,len(old_ids),8):
            audit_stage('warm_teacher_cache')
            ix=old_ids[st:st+8]; ids,mask,lens,_,_=make_model_batch([train[i] for i in ix],pack,seed,next(warm_model.parameters()).device,total_length)
            prep_meter.stage="warm_teacher_cache";prep_meter.phase="forward"
            with prep_meter: cap=capture_cached(warm_model,ids,mask,lens,device="cpu")
            old_teacher.update({i:cap["logits"][j:j+1].clone() for j,i in enumerate(ix)})
        old_val_ids=[i for i,r in enumerate(val) if int(r["class_id"]) in (0,1)];old_val_teacher={}
        for st in range(0,len(old_val_ids),8):
            audit_stage('warm_teacher_validation')
            ix=old_val_ids[st:st+8];ids,mask,lens,_,_=make_model_batch([val[i] for i in ix],pack,seed,next(warm_model.parameters()).device,total_length)
            prep_meter.stage="warm_teacher_validation";prep_meter.phase="forward"
            with prep_meter: cap=capture_cached(warm_model,ids,mask,lens,device="cpu")
            old_val_teacher.update({i:cap["logits"][j:j+1].clone() for j,i in enumerate(ix)})
        del warm_model
        old_train_rows=[train[i] for i in old_ids]; old_val_rows=[val[i] for i in old_val_ids]
        warm_train_metrics=score_logits(torch.cat([old_teacher[i] for i in old_ids]),old_train_rows,pack,seed)
        warm_val_metrics=score_logits(torch.cat([old_val_teacher[i] for i in old_val_ids]),old_val_rows,pack,seed)
        if not old_teacher_passes(warm_train_metrics,warm_val_metrics): raise RuntimeError('fixed warm teacher lost competence; no seed replacement')
        teacher_quality.append({"seed":seed,"current_newteacher":{"train":new_teacher_train_metrics,"validation":new_teacher_val_metrics},"warm_oldteacher":{"train_old_only":warm_train_metrics,"validation_old_only":warm_val_metrics},'warm_old_gate_passed':True})
        cache_disk={i:{k:v for k,v in z.items() if k!="logits"} for i,z in cache.items()}
        cache_path=args.out_dir/"cache-private"/f"seed-{seed}.pt";torch.save({"seed":seed,"T":total_length,"student_features":cache_disk,"new_teacher_train_logits":new_teacher,"warm_old_teacher_train_logits":old_teacher},cache_path);cache_sha=c.sha(cache_path)
        cache=cache_disk
        del old_val_teacher,cap
        schedules={a:class_schedule(train,seed,a) for a in ARMS}
        for arm in ARMS:
            _STUDY_PREFIX=f'seed{seed}:{arm}'
            time_guard()
            run_start=__import__('time').monotonic()
            m=c.model(args.model_dir,seed); c.restore(m,newcp); c.engine.restore_rng(newcp["rng"]); reset_student_context(m)
            scope=arm.rsplit("_",1)[1]; allowed=set_parameter_scope(m,scope)
            initial={'adapter_sha256':c.fingerprint(m),'policy_sha256':c.engine.shared.tensor_tree_hash(execution_state(m))}
            if str(seed) not in common_initial:common_initial[str(seed)]=initial
            if initial!=common_initial[str(seed)]: raise AssertionError('arms do not share identical initialization')
            frozen_before=model_guards(m,allowed)
            torch.cuda.reset_peak_memory_stats()
            opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.003,weight_decay=0.0)
            sched=schedules[arm]; row={"seed":seed,"arm":arm,"status":"running","updates":0,"schedule_sha256":hash_json(sched),"cache_sha256":cache_sha,"trainable_parameter_count":sum(p.numel() for p in m.parameters() if p.requires_grad),"evaluations":[],"loss_sum":0.0,"loss_min":None,"loss_max":None,"work":{},"student_forward_passes":0,"student_backward_passes":0,"student_input_tokens":0}
            if hash_json(sched)!=cfg['schedule_sha256_by_seed_arm'][f'{seed}:{arm}']: raise AssertionError('schedule source lock mismatch')
            draws=[i for batch in sched for i in batch]
            row.update({'initial':initial,'allowed_parameter_names':sorted(allowed),'cache_file_bytes':cache_path.stat().st_size,
                'model_tensor_bytes':sum(v.numel()*v.element_size() for v in m.adapter_state_dict().values()),
                'unique_training_examples':len(set(draws)), 'class_draws':{str(k):sum(train[i]['class_id']==k for i in draws) for k in range(4)},
                'class_unique_examples':{str(k):len({i for i in draws if train[i]['class_id']==k}) for k in range(4)},
                'answer_token_targets':len(draws)*2,'student_padded_positions':len(draws)*total_length,
                'direct_gold_answer_draws':len(draws) if arm.startswith(('gold_','new_only_')) else 0,
                'teacher_origin_selection_draws':len(draws) if arm.startswith('split') else 0})
            allrows=train+val
            eval_meter=MatmulWork()
            runs.append(row);time_guard(run_start)
            row["evaluations"].append({"step":0,"train":evaluate_cached(m,cache,allrows,trainidx,pack,seed,total_length,meter=eval_meter),"validation":evaluate_cached(m,cache,allrows,validx,pack,seed,total_length,meter=eval_meter)})
            time_guard(run_start)
            for step,idx in enumerate(sched,1):
                time_guard(run_start)
                fam=arm.rsplit("_",1)[0]
                teacher=new_teacher
                if fam=="split" or fam=="split_narrow":
                    teacher={i:(old_teacher[i] if int(train[i]["class_id"])<2 else new_teacher[i]) for i in idx}
                use_teacher={i:new_teacher[i] for i in idx} if fam=="current" else teacher
                if fam in ("gold","new_only"):
                    # Gold arms use the selected real class targets, no teacher outputs.
                    use_teacher={}
                meter=MatmulWork("train","forward")
                with meter: loss=fit_one_batch(m,opt,train,idx,cache,use_teacher,pack,seed,arm,total_length,meter)
                mt=meter.state_dict().get("train",{})
                for phase,rec in mt.items():
                    dest=row["work"].setdefault(phase,{"flops":0,"calls":0,"ops":{}})
                    dest["flops"]+=rec["flops"];dest["calls"]+=rec["calls"]
                    for op,stat in rec["ops"].items():
                        od=dest["ops"].setdefault(op,{"flops":0,"calls":0});od["flops"]+=stat["flops"];od["calls"]+=stat["calls"]
                row["student_forward_passes"]+=1;row["student_backward_passes"]+=1
                row["student_input_tokens"]+=sum(int(cache[i]["lengths"].item())+1 for i in idx)
                row["updates"]+=1; row["loss_sum"]+=loss; row["loss_min"]=loss if row["loss_min"] is None else min(row["loss_min"],loss); row["loss_max"]=loss if row["loss_max"] is None else max(row["loss_max"],loss)
                if step in EVAL_STEPS[1:]:
                    time_guard(run_start)
                    valm=evaluate_cached(m,cache,allrows,validx,pack,seed,total_length,meter=eval_meter)
                    row["evaluations"].append({"step":step,"train":evaluate_cached(m,cache,allrows,trainidx,pack,seed,total_length,meter=eval_meter),"validation":valm})
                    if step in (64,256,1024): checkpoint(m,opt,seed,arm,step,row)
                    time_guard(run_start)
                    print(f"v7 seed={seed} arm={arm} step={step}",flush=True)
                if step%64==0:
                    progress({"runs":runs,"complete":False,"confirmation_scored":False})
            row["status"]="training_complete"; row["train_final"]=row['evaluations'][-1]['train']
            if arm.endswith("receiver") or arm.endswith("shared"):
                # Full-model fixed-T endpoint parity guard.
                time_guard(run_start)
                parity_meter=MatmulWork();full=_full_fixed_logits(m,val,pack,seed,total_length,parity_meter)
                tail=row['evaluations'][-1]['validation']
                fullm=score_logits(full,val,pack,seed); row["parity"]={"full":fullm,"tail":tail}
                for group in ("old","new","all",*(f"class{k}" for k in range(4))):
                    if fullm[group]["em"]!=tail[group]["em"] or abs(fullm[group]["nll"]-tail[group]["nll"])>1e-3: raise AssertionError("cached-tail/full-LM parity failed")
                del full,fullm
                row["endpoint_matmul_work_lower_bound"]=parity_meter.state_dict()
                time_guard(run_start)
            row["evaluation_matmul_work_lower_bound"]=eval_meter.state_dict()
            if model_guards(m,allowed)!=frozen_before: raise AssertionError('frozen adapters/backbone/policy changed')
            row.update({'frozen_scope_guard_passed':True,'frozen_adapters_sha256':frozen_before['frozen_adapters_sha256'],
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                'seconds':__import__('time').monotonic()-run_start,'status':'complete'})
            del m,opt
            time_guard(run_start)
            progress({"runs":runs,"complete":False,"confirmation_scored":False})
      time_guard()
      seal={"schema":"agnews-single-receiver-development-v7","complete":True,"protocol_sha256":protocol_sha,"teacher_quality_by_seed":teacher_quality,"runs":runs,"confirmation_input":False,"confirmation_scored":False,
        'common_initial_by_seed':common_initial,'preparation_matmul_work_lower_bound':{k:v.state_dict() for k,v in preparation_meters.items()},
        'seconds':__import__('time').monotonic()-start,'optimizer_attempts':_OPTIMIZER_ATTEMPTS}
      c.write(args.out_dir/"development-sealed.json",seal)
    except Exception as e:
      c.write(args.out_dir/"fail.json",{"schema":"agnews-single-receiver-failure-v7","error_type":type(e).__name__,"error":str(e),"completed_runs":sum(x["status"]=="complete" for x in runs),"runs":runs,"confirmation_scored":False,
        'teacher_quality_by_seed':teacher_quality,'preparation_matmul_work_lower_bound':{k:v.state_dict() for k,v in preparation_meters.items()},
        'optimizer_attempts':_OPTIMIZER_ATTEMPTS,'seconds':__import__('time').monotonic()-start})
      raise

def answer_ce(logits, labels):
    """Gold CE on two answer positions, preserving standard tensor semantics."""
    import torch.nn.functional as F
    if logits.ndim!=3 or logits.shape[1]!=2 or labels.shape!=(logits.shape[0],2):
        raise ValueError("expected [B,2,V] logits and [B,2] labels")
    return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.reshape(-1))

def distill_kl(student_logits, teacher_logits):
    """Mean full-vocabulary KL over the two marker-conditioned answer positions."""
    import torch
    import torch.nn.functional as F
    if student_logits.shape!=teacher_logits.shape or student_logits.ndim!=3 or student_logits.shape[1]!=2:
        raise ValueError("teacher/student logits must be matching [B,2,V]")
    tl=F.log_softmax(teacher_logits.float(),dim=-1)
    sl=F.log_softmax(student_logits.float(),dim=-1)
    return (tl.exp()*(tl-sl)).sum(-1).mean()

def validate_restore_contract(before, left, right):
    """Qualification contract: all three state fingerprints must agree exactly."""
    keys=("model_sha256","adam_sha256","rng_sha256")
    if any(not all(x.get(k) for x in (before,left,right)) for k in keys): return False
    return all(before[k]==left[k]==right[k] for k in keys)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True); p.add_argument("--records",type=Path,required=True)
    p.add_argument("--warm-dir",type=Path,required=True); p.add_argument("--v6-dir",type=Path,required=True)
    p.add_argument("--qualification",type=Path); p.add_argument("--qualify",action="store_true"); p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    from matmul_work_v7 import MatmulWork
    from agnews_v3_common import write,sha,engine
    import time
    global _STUDY_METER,_STUDY_PREFIX
    _STUDY_METER=MatmulWork();_STUDY_PREFIX='qualification' if args.qualify else 'startup'
    started=time.monotonic();out_preexists=args.out_dir.exists()
    if out_preexists:raise FileExistsError(args.out_dir)
    try:
        with _STUDY_METER:
            if args.qualify:
                args.out_dir.mkdir(parents=True,exist_ok=False)
                try: _run_resume_qualification(args)
                except Exception as exc:
                    write(args.out_dir/"qualification-failure.json",{"schema":"agnews-single-receiver-qualification-failure-v7","error_type":type(exc).__name__,"error":str(exc),"actual_Adam_updates":_QUAL_UPDATES,"optimizer_attempts":_OPTIMIZER_ATTEMPTS,"matmul_work_lower_bound":None if _QUAL_METER is None else _QUAL_METER.state_dict(),"confirmation_scored":False,"driver_sha256":sha(Path(__file__))})
                    raise
            else:
                if args.qualification is None:p.error('--qualification required for main')
                _run_main(args)
    finally:
        if args.out_dir.exists():
            engine.write_atomic(args.out_dir/'execution-work.json',{
                'schema':'agnews-single-receiver-authoritative-matmul-work-v7',
                'work':_STUDY_METER.state_dict(),'seconds':time.monotonic()-started,
                'optimizer_attempts':_OPTIMIZER_ATTEMPTS,'driver_sha256':sha(Path(__file__)),
                'scope':'Executed mm/addmm/bmm/mv/dot shapes, MAC=2; arithmetic lower bound. Per-run counters overlap this authoritative ledger; never add them to it.'})

if __name__=="__main__": main()
