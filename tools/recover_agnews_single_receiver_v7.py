"""Complete only the interrupted tail of the frozen v7 grid; keep attempts separate."""
import argparse,copy,time,json
from pathlib import Path
import torch
import agnews_v3_common as c
import run_agnews_single_receiver_v7 as v7
import agnews_tail_cache_v7 as helper
from nested_sleep.lm import execution_state
from matmul_work_v7 import MatmulWork


class PhaseProxy:
    def __init__(self,work):self.work=work
    @property
    def stage(self):return self.work.stage
    @stage.setter
    def stage(self,value):pass
    @property
    def phase(self):return self.work.phase
    @phase.setter
    def phase(self,value):self.work.phase=value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('records','model-dir','run-dir','warm-dir','v6-dir','config','out-dir'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--qualify',action='store_true');p.add_argument('--qualification',type=Path)
    a=p.parse_args();assert not a.out_dir.exists()
    cfg=c.read(a.config);pack=c.load_pack(a.records);v7.validate_input_locks(a,cfg);c.preflight(a.model_dir,pack)
    protocol=c.read(a.run_dir/'protocol.json');partial=c.read(a.run_dir/'progress.json');loss=c.read(a.run_dir/'process-loss.json')
    assert protocol['config_sha256']==c.sha(a.config) and protocol['driver_sha256']==c.sha(Path(v7.__file__))
    assert c.sha(a.run_dir/'progress.json')==loss['original_progress_sha256']
    assert c.sha(a.run_dir/'protocol.json')==loss['original_protocol_sha256']
    done=[r for r in partial['runs'] if r['status']=='complete'];unfinished=[r for r in partial['runs'] if r['status']!='complete']
    assert len(done)==28 and len(unfinished)==1
    old=unfinished[0];assert (old['seed'],old['arm'],old['updates'])==(43,'new_only_receiver',704)
    assert not (a.run_dir/'development-sealed.json').exists()
    source=c.sha(Path(__file__))
    if not a.qualify:
        q=c.read(a.qualification);assert q['passed'] and q['source_sha256']==source
        assert q['progress_sha256']==loss['original_progress_sha256']
    a.out_dir.mkdir(parents=True)
    plan={'schema':'agnews-v7-interruption-recovery-plan','source_sha256':source,'config_sha256':c.sha(a.config),
        'original_progress_sha256':loss['original_progress_sha256'],'original_protocol_sha256':loss['original_protocol_sha256'],
        'source_checkpoint_step':256,'last_observed_step':704,'original_unrecorded_updates_inclusive':[0,64],
        'known_replayed_updates':448,'remaining_logical_arms':['43:new_only_receiver','43:new_only_shared'],
        'qualification':a.qualify,'qualification_sha256':None if a.qualify else c.sha(a.qualification),
        'original_exit_cause':'unknown; no terminal receipt','do_not_sum_overlapping_local_and_global_work':True}
    c.write(a.out_dir/'plan.json',plan)
    started=time.monotonic();work=MatmulWork();proxy=PhaseProxy(work);attempts=0;rows=[];quality=[]
    def guard():
        if time.monotonic()-started>3600:raise TimeoutError('bounded recovery cap')
    def save(m,opt,row,step):
        path=a.out_dir/f'{row["seed"]}-{row["arm"]}-{step}.pt'
        torch.save(c.engine.shared.compact_cpu({'schema':'agnews-single-receiver-checkpoint-v7',
            'model':m.adapter_state_dict(),'execution_state':execution_state(m),'optimizer':opt.state_dict(),
            'rng':c.engine.capture_rng(),'seed':row['seed'],'arm':row['arm'],'step':step,
            'schedule_sha256':row['schedule_sha256'],'protocol_sha256':loss['original_protocol_sha256'],
            'recovery_source_sha256':source}),path)
        meta={'path':path.relative_to(a.run_dir).as_posix(),'sha256':c.sha(path),'step':step}
        return meta
    def initial_from_v6(m,seed):
        v6=c.read(a.v6_dir/'development-sealed.json')
        nr=next(r for r in v6['runs'] if r['seed']==seed and r['arm']=='aux_new_CE')
        cp,_=v7._load_cp(a.v6_dir,v7._checkpoint_row(nr,1024));c.restore(m,cp);c.engine.restore_rng(cp['rng'])
        v7.reset_student_context(m);return cp
    def restore_training(m,cp,scope):
        c.restore(m,cp);c.engine.restore_rng(cp['rng']);v7.set_parameter_scope(m,scope)
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.003,weight_decay=0)
        opt.load_state_dict(copy.deepcopy(cp['optimizer']));return opt
    try:
      with work:
        seed=43;train=c.rows(pack,seed,'train');val=c.rows(pack,seed,'validation');allrows=train+val
        cache_path=a.run_dir/'cache-private'/'seed-43.pt';assert c.sha(cache_path)==old['cache_sha256']
        data=torch.load(cache_path,map_location='cpu',weights_only=False);features=data['student_features']
        meta=next(x for x in old['checkpoints'] if x['step']==256)
        cp,_=v7._load_cp(a.run_dir,meta);assert cp['protocol_sha256']==loss['original_protocol_sha256']
        sched=v7.class_schedule(train,43,old['arm']);assert v7.hash_json(sched)==old['schedule_sha256']
        m=c.model(a.model_dir,43)
        initial_from_v6(m,43)
        assert c.fingerprint(m)==old['initial']['adapter_sha256']
        expected_frozen=v7.model_guards(m,v7.parameter_scope(m,'receiver'))
        c.restore(m,cp)
        assert v7.model_guards(m,v7.parameter_scope(m,'receiver'))==expected_frozen
        if a.qualify:
            states=[]
            for branch in range(2):
                opt=restore_training(m,cp,'receiver')
                for step in (257,258):
                    guard();work.stage='independent_restore_proof';work.phase='forward'
                    attempts+=1;v7.fit_one_batch(m,opt,train,sched[step-1],features,{},pack,43,old['arm'],118,meter=proxy)
                states.append({'model':c.fingerprint(m),'optimizer':c.engine.shared.tensor_tree_hash(opt.state_dict()),
                    'policy':c.engine.shared.tensor_tree_hash(execution_state(m)),'rng':c.engine.shared.tensor_tree_hash(c.engine.capture_rng())})
                del opt
            assert states[0]==states[1]
            c.write(a.out_dir/'qualification.json',{'passed':True,'source_sha256':source,'progress_sha256':loss['original_progress_sha256'],
                'independent_restores_exact':True,'branches':states,'optimizer_attempts':attempts,
                'work':work.state_dict(),'seconds':time.monotonic()-started})
            return
        for arm in ('new_only_receiver','new_only_shared'):
            guard();scope=arm.rsplit('_',1)[1];resume=scope=='receiver'
            if resume:
                row=copy.deepcopy(old);opt=restore_training(m,cp,scope);start_step=256
                original_observed_work=row.pop('work')
                row['original_attempt_training_work_observed']=original_observed_work
                row['original_attempt_loss_statistics']={k:row[k] for k in ('loss_sum','loss_min','loss_max')}
                row['loss_sum']=None;row['loss_min']=None;row['loss_max']=None
            else:
                original=initial_from_v6(m,43);v7.set_parameter_scope(m,scope);start_step=0
                opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.003,weight_decay=0)
                row=copy.deepcopy(old)
                row.update({'arm':arm,'updates':0,'evaluations':[],'checkpoints':[],'student_forward_passes':0,
                    'student_backward_passes':0,'student_input_tokens':0,'loss_sum':0.,'loss_min':None,'loss_max':None,
                    'work':{},'trainable_parameter_count':294912,'allowed_parameter_names':sorted(v7.parameter_scope(m,scope))})
            sched=v7.class_schedule(train,43,arm);row['schedule_sha256']=v7.hash_json(sched)
            assert row['schedule_sha256']==cfg['schedule_sha256_by_seed_arm'][f'43:{arm}']
            allowed=v7.parameter_scope(m,scope);frozen=v7.model_guards(m,allowed)
            if not resume:
                work.stage=arm+':initial_score';work.phase='forward'
                with torch.no_grad():
                    row['evaluations'].append({'step':0,'train':v7.evaluate_cached(m,features,allrows,list(range(1024)),pack,43,118),
                        'validation':v7.evaluate_cached(m,features,allrows,list(range(1024,1536)),pack,43,118)})
                assert {'adapter_sha256':c.fingerprint(m),'policy_sha256':c.engine.shared.tensor_tree_hash(execution_state(m))}==old['initial']
            row['status']='recovery_running';row['recovered_from_step']=start_step;row['recovery_actual_updates']=0
            row['logical_student_forward_passes']=1024;row['logical_student_backward_passes']=1024
            row['exposure_counters_describe_logical_schedule']=True
            if resume:row['original_unrecorded_optimizer_attempts_inclusive']=[0,64]
            rows.append(row);runstart=time.monotonic()
            for step in range(start_step+1,1025):
                guard();work.stage=arm+':training';work.phase='forward';attempts+=1
                value=v7.fit_one_batch(m,opt,train,sched[step-1],features,{},pack,43,arm,118,meter=proxy)
                row['updates']=step;row['recovery_actual_updates']+=1
                row['student_forward_passes']+=1;row['student_backward_passes']+=1
                row['student_input_tokens']+=sum(int(features[i]['lengths'].item())+1 for i in sched[step-1])
                if not resume:
                    row['loss_sum']+=value;row['loss_min']=value if row['loss_min'] is None else min(row['loss_min'],value)
                    row['loss_max']=value if row['loss_max'] is None else max(row['loss_max'],value)
                if step in (64,256,1024) and step>start_step:
                    work.stage=arm+':evaluation';work.phase='forward'
                    with torch.no_grad():
                        row['evaluations'].append({'step':step,'train':v7.evaluate_cached(m,features,allrows,list(range(1024)),pack,43,118),
                            'validation':v7.evaluate_cached(m,features,allrows,list(range(1024,1536)),pack,43,118)})
                if step%64==0:
                    cpmeta=save(m,opt,row,step)
                    if step in (64,256,1024):row['checkpoints'].append(cpmeta)
                    else:row.setdefault('extra_recovery_checkpoints',[]).append(cpmeta)
                    c.engine.write_atomic(a.out_dir/'progress.json',{'rows':rows,'optimizer_attempts':attempts})
                    c.engine.write_atomic(a.out_dir/'execution-work.json',{'work':work.state_dict(),'optimizer_attempts':attempts,'complete':False})
                    print({'arm':arm,'logical_step':step,'actual_recovery_updates':row['recovery_actual_updates']},flush=True)
            work.stage=arm+':full_parity';work.phase='forward'
            full_logits=v7._full_fixed_logits(m,val,pack,43,118);full=v7.score_logits(full_logits,val,pack,43);del full_logits
            tail=row['evaluations'][-1]['validation']
            assert all(full[g]['em']==tail[g]['em'] and abs(full[g]['nll']-tail[g]['nll'])<=.001 for g in full)
            assert v7.model_guards(m,allowed)==frozen
            row.update({'status':'complete','train_final':row['evaluations'][-1]['train'],'parity':{'full':full,'tail':tail},
                'frozen_scope_guard_passed':True,'seconds_recovery':time.monotonic()-runstart,
                'execution_note':'Observed training counters include the original704 plus replay768 for recovered receiver; logical trajectory ends1024.'})
            del opt
        del m,data,features,cp
        # The original teacher-quality list was not flushed. Re-evaluate the
        # same frozen inputs, reporting this new work as a recovery operation.
        warm=c.read(a.warm_dir/'transfer-result.json');v6=c.read(a.v6_dir/'development-sealed.json')
        for seed in v7.SEEDS:
            guard();m=c.model(a.model_dir,seed);cache=torch.load(a.run_dir/'cache-private'/f'seed-{seed}.pt',map_location='cpu',weights_only=False)
            train=c.rows(pack,seed,'train');val=c.rows(pack,seed,'validation')
            nr=next(r for r in v6['runs'] if r['seed']==seed and r['arm']=='aux_new_CE')
            current,_=v7._load_cp(a.v6_dir,v7._checkpoint_row(nr,1024));c.restore(m,current)
            work.stage=f'teacher_quality_recomputed:{seed}:current';work.phase='forward'
            z=v7._full_fixed_logits(m,val,pack,seed,118);newval=v7.score_logits(z,val,pack,seed);del z
            nt=cache['new_teacher_train_logits'];newtrain=v7.score_logits(torch.cat([nt[i] for i in range(1024)]),train,pack,seed)
            wr=next(r for r in warm['qualified_models'] if r['seed']==seed);warmcp,_=v7._load_cp(a.warm_dir,wr['checkpoint']);c.restore(m,warmcp)
            oldidx=[i for i,r in enumerate(train) if r['class_id']<2];oldval=[r for r in val if r['class_id']<2]
            work.stage=f'teacher_quality_recomputed:{seed}:warm'
            z=v7._full_fixed_logits(m,oldval,pack,seed,118);wval=v7.score_logits(z,oldval,pack,seed);del z
            wt=cache['warm_old_teacher_train_logits'];wtrain=v7.score_logits(torch.cat([wt[i] for i in oldidx]),[train[i] for i in oldidx],pack,seed)
            assert v7.old_teacher_passes(wtrain,wval)
            quality.append({'seed':seed,'current_newteacher':{'train':newtrain,'validation':newval},
                'warm_oldteacher':{'train_old_only':wtrain,'validation_old_only':wval},'warm_old_gate_passed':True,'recomputed_after_process_loss':True})
            del m,cache,nt,wt,current,warmcp
        combined=copy.deepcopy(done)+rows
        assert len(combined)==30 and all(r['status']=='complete' and r['updates']==1024 for r in combined)
        init={str(s):next(r['initial'] for r in combined if r['seed']==s) for s in v7.SEEDS}
        result={'schema':'agnews-single-receiver-development-v7','complete':True,'protocol_sha256':loss['original_protocol_sha256'],
            'runs':combined,'teacher_quality_by_seed':quality,'common_initial_by_seed':init,'confirmation_scored':False,
            'recovery':{'source_sha256':source,'plan_sha256':c.sha(a.out_dir/'plan.json'),'qualification_sha256':c.sha(a.qualification),
                'original_attempt_process_loss_sha256':c.sha(a.run_dir/'process-loss.json'),'original_global_work_missing':True,
                'actual_recovery_optimizer_updates':attempts,'known_replayed_updates':448,'unrecorded_original_updates_inclusive':[0,64],
                'recovery_execution_work_path':(a.out_dir/'execution-work.json').relative_to(a.run_dir).as_posix()}}
        c.write(a.run_dir/'development-sealed.json',result)
    except BaseException as e:
        c.write(a.out_dir/'failure.json',{'type':type(e).__name__,'error':str(e),'rows':rows,'optimizer_attempts':attempts,'work':work.state_dict()})
        raise
    finally:
        c.engine.write_atomic(a.out_dir/'execution-work.json',{'schema':'agnews-v7-recovery-execution-work',
            'work':work.state_dict(),'optimizer_attempts':attempts,'seconds':time.monotonic()-started,
            'source_sha256':source,'complete':(a.run_dir/'development-sealed.json').exists() if not a.qualify else (a.out_dir/'qualification.json').exists()})

if __name__=='__main__':main()
