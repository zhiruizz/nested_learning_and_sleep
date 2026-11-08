"""Finite two-boundary V13 execution; no test reader or whole-job resume CLI."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import time


def work_delta(before, after):
    return {k: after[k]-before.get(k,0) for k in after
            if isinstance(after[k],(int,float)) and not isinstance(after[k],bool)}


def summarize_objectives(result):
    requests=result['request_summaries']
    n=int(result['opportunities']['opportunities'])
    if len(requests)!=n:
        raise ValueError('objective diagnostics must cover every selected request')
    roles={r:sum(x['metadata']['role']==r for x in requests) for r in ('old','current')}
    owners={r:sum(x['owner']==r for x in requests) for r in ('teacher','student')}
    rewards=[x['metadata']['lti_reward'] for x in requests if x['kind']=='lti']
    return {'request_count':n,'method_counts':{k:result['opportunities'][k] for k in ('gkd','lti')},
            'role_counts':roles,'sampling_owner_counts':owners,
            'lti_reward_count':len(rewards),'lti_reward_sum':sum(rewards),
            'lti_zero_reward_count':sum(x==0 for x in rewards),
            'loss_mean':sum(x['loss_mean']*x['request_count'] for x in result['batches'])/n,
            'backward_calls':result['backward_calls'],
            'physical_adamw_steps':result['physical_adamw_steps'],
            'skipped_exact_zero_batches':result['skipped_zero_gradient_batches']}


def run_lifecycle(args, preflight):
    import torch
    import run_banking_sleep_lifecycle_v13 as api
    import run_banking_cil_v10 as old
    import banking_cil_common_v10 as common
    import banking_cil_v10_runtime as runtime
    import clinc_tail_runtime_v8 as tails
    from banking_cil_v10_lifecycle import AdditiveWakeController
    from banking_cil_v13_batching import BatchableAdditiveSleepTransactionV13
    from banking_sleep_v13_runtime import run_batched_sleep_event
    from clinc_lifecycle_v8 import _state_hash
    from run_banking_optimizer_boundary_v12 import _same_nested, stable_state_sha

    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    base=json.loads(Path(args.v10_config).read_text(encoding='utf-8'))
    api.validate_config(cfg,require_frozen=True)
    if api.sha256_file(args.config)!=preflight['v13_config_sha256']:
        raise ValueError('config changed after CPU preflight')
    root=Path(__file__).resolve().parents[1]
    for name,digest in cfg['source_hashes'].items():
        if api.sha256_file(root/name)!=digest:raise RuntimeError('execution source changed: '+name)
    if not torch.cuda.is_initialized():raise RuntimeError('launcher must apply reviewed CUDA policy')
    arm=next(x for x in cfg['arms'] if x['name']==args.arm)
    output=Path(args.output_dir).resolve()
    if output.exists() or not output.parent.is_dir():raise ValueError('fresh output with existing parent required')
    protocol={'schema':'banking-v13-batched-lifecycle-protocol-v1','status':'running',
        'seed':args.seed,'arm':args.arm,'config_sha256':preflight['v13_config_sha256'],
        'source_hashes':cfg['source_hashes'],'v10_source_lock_sha256':preflight['source_lock_sha256'],
        'training_manifest_sha256':cfg['training_manifest_sha256'],
        'foundation_receipt_sha256':preflight['foundation_receipt_sha256'],
        'foundation_checkpoint_sha256':preflight['foundation_checkpoint_sha256'],'test_data_read':False}
    budget=old.Budget({k:cfg[k] for k in ('max_wall_seconds','max_forward_calls','max_update_steps')})
    model=original_logits=original_tail=None
    stage_states={}
    stages=[]
    events=[]
    checkpoints=[]
    wake_work={}
    qualification=None
    evaluation_work={}
    started=time.monotonic()
    active_phase='prepare'
    owned_output=False

    def progress(**extra):
        old._atomic_json(output/'progress.json',{'status':'running','seed':args.seed,'arm':args.arm,
            'phase':active_phase,'work':budget.as_dict(),**extra})

    try:
        active_phase='create_outputs'
        output.mkdir()
        owned_output=True
        for name in ('checkpoints','stages','events'):(output/name).mkdir()
        old._atomic_json(output/'protocol.json',protocol)
        progress()
        old._init_rng(cfg['initialization_seed'],torch)
        _,_,manifest,train,dev=old._source_prepare(Path(args.project_root),root,
            Path(args.model_path),Path(args.training_pack),base)
        active_phase='model_load'
        torch,tokenizer,model,immutable_versions=old._load_model(Path(args.project_root),Path(args.model_path),base)
        old._verify_loaded_project_modules(Path(args.project_root))
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        fixed_versions=immutable_versions(model)
        adapter_hash=stable_state_sha(model.adapter_state_dict(),torch)
        original_logits=model.logits
        def counted(tokens,mask=None):
            budget.forward(mask if mask is not None else torch.ones_like(tokens))
            return original_logits(tokens,mask)
        model.logits=counted
        tails,original_tail=old._install_tail_counter(budget)
        context=tails.clone_mutable_context(model)
        wake=AdditiveWakeController(context,periods=base['periods'],base_lr=base['base_lr'],fast_lr=base['fast_lr'])
        fdir=Path(preflight['foundation_run_dir'])
        freceipt=json.loads((fdir/'foundation-receipt.json').read_text(encoding='utf-8'))
        old._restore_checkpoint(fdir/freceipt['checkpoint'],context,wake,
            expected_sha=preflight['foundation_checkpoint_sha256'],seed=args.seed,
            arm='FOUNDATION_CMS_ONLY',cfg=base,torch=torch)
        initial_tick=wake.ticks
        order=tuple(int(x) for x in manifest['class_orders'][str(args.seed)])
        current=old.task_rows(train,order,1)
        lookup={str(x['id']):x for x in current}
        ids=old.deterministic_epoch_ids(current,seed=args.seed,task_id=1,epoch=0)
        ids+=old.deterministic_epoch_ids(current,seed=args.seed,task_id=1,epoch=1)
        horizon=cfg['expected_total_wake_ticks_by_seed'][str(args.seed)]
        ids=ids[:horizon*32]
        if len(ids)!=horizon*32:raise ValueError('two epochs do not cover the frozen wake budget')
        wake_digest=old._canonical_sha(ids)
        wake_cfg={**base,'batch_size':32}
        pools=old._group_sleep_pools(train,order,1)
        device=next(model.parameters()).device

        def save_stage_state(name):
            state=deepcopy(wake.context)
            stage_states[name]=(wake.ticks,state)
            payload={'schema':'banking-v13-state-checkpoint-v1','seed':args.seed,'arm':args.arm,
                'stage':name,'config_sha256':preflight['v13_config_sha256'],
                'context':state.state_dict(),'wake':wake.snapshot(),'rng':old._rng_state(torch),
                'source_hashes':cfg['source_hashes'],'training_manifest_sha256':cfg['training_manifest_sha256']}
            path=output/'checkpoints'/f'{name}.pt'
            digest=old._atomic_torch(path,payload,torch)
            loaded=torch.load(path,map_location='cpu',weights_only=False)
            if not _same_nested(payload,loaded,torch):raise RuntimeError('checkpoint destination readback mismatch')
            record={'file':path.relative_to(output).as_posix(),'sha256':digest,'bytes':path.stat().st_size,
                'stage':name,'global_wake_tick':wake.ticks,'context_state_sha256':_state_hash(state),
                'destination_deserialized_readback':True}
            checkpoints.append(record)
            old._atomic_json(path.with_suffix('.json'),record)
            return record

        for tick in range(1,horizon+1):
            active_phase='wake'
            before=budget.as_dict()
            rows=[lookup[x] for x in ids[(tick-1)*32:tick*32]]
            old._train_tick(model,wake.context,wake,rows,tokenizer,manifest['classes'],wake_cfg,torch,device,budget)
            delta=work_delta(before,budget.as_dict())
            wake_work={k:wake_work.get(k,0)+v for k,v in delta.items()}
            due=sorted(wake.proposals())
            if due:
                event_index=len(events)
                expected=cfg['prefix_ticks_by_seed'][str(args.seed)]+event_index*16
                if due!=[0] or event_index>=2 or tick!=expected:
                    raise RuntimeError('actual due levels/tick differ from frozen two-event plan')
                active_phase='boundary_checkpoint'
                pre=save_stage_state(f'event{event_index+1}_before')
                before=budget.as_dict()
                proposal=wake.proposals()[0]
                selected=[]
                result=None
                event_qual=None
                if not arm['sleep']:
                    wake.apply(0,proposal)
                else:
                    active_phase='sleep'
                    seed=cfg['initialization_seed']+args.seed*100003+1009+event_index*1009
                    tx=BatchableAdditiveSleepTransactionV13(wake.context,0,proposal,
                        lr=arm['learning_rate'],weight_decay=cfg['weight_decay'],clip_norm=cfg['clip_norm'],seed=seed)
                    kinds=[cfg['sleep_method_cycle'][i%3] for i in range(cfg['sleep_selector_event_stride'])]
                    selected=old.make_sleep_request_schedule(pools['old'],pools['current'],
                        event_index=event_index,task_id=1,method_kinds=kinds)[:arm['requests_per_event']]
                    def builder_factory(transaction):
                        return api.build_v10_equivalent_objective(model=model,tx=transaction,tokenizer=tokenizer,
                            classes=manifest['classes'],cfg=base,torch=torch,device=device,budget=budget,
                            tail_runtime=tails,common=common,sample_tokens=old._sample_tokens,lti_reward=old._lti_reward_adapter)
                    qcallback=None
                    if args.seed==17 and args.arm=='B32_HIGH' and event_index==0:
                        from banking_v13_restore_qualification import make_b32_restore_qualification_callback
                        qcallback=make_b32_restore_qualification_callback(tx=tx,selected=selected,
                            objective_builder_factory=builder_factory,budget=budget,
                            output_dir=output/'qualification',torch=torch,seed=args.seed,arm=args.arm)
                    def on_batch(index,next_offset,next_gkd,batch):
                        if qcallback is not None and not qcallback.done:qcallback(index,next_offset,next_gkd,batch)
                        if next_offset%128==0 or next_offset==len(selected) or next_offset==64:
                            progress(event_index=event_index,request_offset=next_offset,
                                physical_adamw_steps=tx.counters['adamw_steps'])
                    result=run_batched_sleep_event(selected,tx,microbatch_size=arm['effective_batch_size'],
                        objective_builder=builder_factory(tx),budget=budget,on_batch_committed=on_batch)
                    if qcallback is not None:
                        if not qcallback.done:raise RuntimeError('real restore qualification did not complete')
                        event_qual=json.loads(qcallback.receipt_path.read_text(encoding='utf-8'))
                        if event_qual.get('status')!='pass':raise RuntimeError('real restore qualification did not pass')
                        qualification=event_qual
                    wake.bind_context(tx.commit())
                    wake.consume(0)
                    budget.sleep_events+=1
                event_work=work_delta(before,budget.as_dict())
                if event_qual:
                    for k,v in event_qual['extra_work_delta'].items():
                        if k in event_work and isinstance(v,(int,float)):event_work[k]-=v
                active_phase='boundary_checkpoint'
                post=save_stage_state(f'event{event_index+1}_after')
                record={'level':0,'event_index':event_index,'selected_requests':len(selected),
                    'effective_batch_size':arm['effective_batch_size'],'physical_adamw_steps':result['physical_adamw_steps'] if result else 0,
                    'objective_aggregates':summarize_objectives(result) if result else {},
                    'work_delta':event_work,'selected_request_digest':old._canonical_sha([x[1]['id'] for x in selected]),
                    'pre_checkpoint_sha256':pre['sha256'],'post_checkpoint_sha256':post['sha256'],
                    'qualification_extra_charged_separately':bool(event_qual)}
                events.append(record)
                old._atomic_json(output/'events'/f'event{event_index}.json',record)
            if tick%4==0 or due:progress(task_wake_tick=tick,global_wake_tick=wake.ticks,events_completed=len(events))
        if len(events)!=2 or wake.ticks!=preflight['event_ticks_and_endpoint'][2] or wake.proposals():
            raise RuntimeError('terminal clocks/events differ from frozen plan')
        save_stage_state('postwake15')
        # Readouts are delayed until training and the internal restore gate end.
        # They cannot provide online feedback or alter event budgets.
        active_phase='development_evaluation'
        allowed=set(order[:22])
        eval_rows=sorted((x for x in dev if int(x['intent_id']) in allowed),key=lambda x:(int(x['intent_id']),str(x['id'])))
        if len(eval_rows)!=220:raise ValueError('development denominator is not 220')
        for name in cfg['evaluation']['stages']:
            tick,state=stage_states[name]
            before_hash=_state_hash(state)
            before=budget.as_dict()
            metrics=runtime.score_candidates(model,{name:state},eval_rows,tokenizer,manifest['classes'],common,tails,
                torch=torch,device=device,max_sequence_length=128,batch_size=32,budget=budget,
                class_order=order,task_id=1,include_greedy=True,max_new_tokens=10)
            if _state_hash(state)!=before_hash:raise RuntimeError('evaluation mutated saved context')
            delta=work_delta(before,budget.as_dict())
            evaluation_work={k:evaluation_work.get(k,0)+v for k,v in delta.items()}
            item={'name':name,'global_wake_tick':tick,'runtime':metrics,'work_delta':delta,
                'context_state_sha256':before_hash,'evaluation_after_training':True}
            stages.append(item)
            old._atomic_json(output/'stages'/f'{name}.json',item)
            progress(stages_completed=len(stages))
        if immutable_versions(model)!=fixed_versions or stable_state_sha(model.adapter_state_dict(),torch)!=adapter_hash:
            raise RuntimeError('frozen backbone/model-owned adapters changed')
        for name,digest in cfg['source_hashes'].items():
            if api.sha256_file(root/name)!=digest:raise RuntimeError('source changed during execution')
        if api.sha256_file(args.config)!=preflight['v13_config_sha256']:raise RuntimeError('config changed during execution')
        result={'schema':'banking-v13-batched-lifecycle-result-v1','status':'complete','seed':args.seed,'arm':args.arm,
            'config_sha256':preflight['v13_config_sha256'],'source_hashes':cfg['source_hashes'],
            'training_manifest_sha256':cfg['training_manifest_sha256'],
            'foundation_receipt_sha256':preflight['foundation_receipt_sha256'],
            'foundation_checkpoint_sha256':preflight['foundation_checkpoint_sha256'],
            'initial_wake_tick':initial_tick,'final_wake_tick':wake.ticks,'wake_request_digest':wake_digest,
            'stages':stages,'events':events,'checkpoint_manifest':checkpoints,
            'restore_qualification':qualification,'work_phases':{'wake':wake_work,'evaluation':evaluation_work},
            'budget':budget.as_dict(),'wall_seconds':time.monotonic()-started,
            'model_immutable_versions_unchanged':True,'model_owned_adapter_sha256_before_after':adapter_hash,
            'official_test_read':False,'test_data_read':False,'P2_accessed':False,
            'claim_scope':cfg['claim_scope'],'raw_rows_or_trajectories_written_to_json':False}
        old._atomic_json(output/'final-receipt.json',result)
        protocol.update(status='complete',final_receipt_sha256=api.sha256_file(output/'final-receipt.json'))
        old._atomic_json(output/'protocol.json',protocol)
        old._atomic_json(output/'progress.json',{'status':'complete','seed':args.seed,'arm':args.arm,'work':budget.as_dict()})
        return result
    except BaseException as exc:
        if owned_output:
            try:
                old._atomic_json(output/'failure.json',{'schema':'banking-v13-failure-v1','status':'failed',
                    'seed':args.seed,'arm':args.arm,'phase':active_phase,'error_type':type(exc).__name__,
                    'work':budget.as_dict(),'events_completed':len(events),'stages_completed':len(stages),'test_data_read':False})
            except OSError:
                pass  # The independent launcher WAL still records filesystem failures.
        raise
    finally:
        if model is not None and original_logits is not None:model.logits=original_logits
        old._remove_tail_counter(tails,original_tail)
