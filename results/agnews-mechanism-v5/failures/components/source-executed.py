"""Read-only component interventions on existing v4 checkpoints (development).

No optimizer, new training, holdout, or checkpoint writes. Restoring bases and
forcing a receiver are diagnostic interventions, not deployed methods.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,time
import torch
import agnews_v3_common as c
from nested_sleep.lm import execution_state

CASES=(('warm_CMS_no_sleep',16),('warm_CMS_no_sleep',112),
       ('warm_pre_sleep_P4',16),('warm_pre_sleep_P4',112),('warm_post_sleep_P4',112))
PROBES=('actual','force_warm_receiver','disable_fast_force_warm',
        'restore_base0_force_warm','restore_base1_force_warm',
        'restore_both_force_warm','restore_both_actual_route','warm_function_control')


def intervene(model,warm,probe):
    if probe not in PROBES:raise ValueError(probe)
    params=dict(model.named_parameters())
    levels=(0,1)if probe in('restore_both_force_warm','restore_both_actual_route','warm_function_control')else(0,)if probe=='restore_base0_force_warm'else(1,)if probe=='restore_base1_force_warm'else()
    with torch.no_grad():
        for level in levels:
            for name in model.level_descriptors()[level]['base']:
                params[name].copy_(warm['model'][name])
    if probe not in('actual','restore_both_actual_route'):
        model.banks()[1].route_override={'force':0}
    if probe in('disable_fast_force_warm','warm_function_control'):
        model.banks()[0].route_override={'disable':[i for i,s in enumerate(model.banks()[0].slots)if bool(s.active)]}


def observe(model,rows,pack,seed,stage,meter):
    before=c.fingerprint(model);policy=execution_state(model);rng=c.engine.shared.tensor_tree_hash(c.engine.capture_rng())
    metrics=c.score(model,rows,pack,seed,stage,meter)
    assert c.fingerprint(model)==before and execution_state(model)==policy
    assert c.engine.shared.tensor_tree_hash(c.engine.capture_rng())==rng
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('records','warm-dir','v4-run-dir','model-dir','out-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();pack=c.load_pack(a.records);warm=c.read(a.warm_dir/'transfer-result.json');seal=c.read(a.v4_run_dir/'development-sealed.json')
    assert warm['all3_qualified'] and seal['complete'] and not warm['confirmation_scored'] and not seal['confirmation_scored']
    c.preflight(a.model_dir,pack);a.out_dir.mkdir(exist_ok=False,parents=True)
    c.write(a.out_dir/'plan.json',{'schema':'agnews-components-plan-v5','source_sha256':c.sha(__file__),
      'warm_result_sha256':c.sha(a.warm_dir/'transfer-result.json'),'v4_development_seal_sha256':c.sha(a.v4_run_dir/'development-sealed.json'),
      'data_sha256':c.DATA_SHA,'cases':list(CASES),'probes':list(PROBES),'seeds':list(c.SEEDS),'scoring_role':'existing validation,all512',
      'expected_probe_states':120,'warm_reference_states':3,'total_scored_states':123,'optimizer_updates':0,'elapsed_limit_seconds':1200,'time_enforcement':'cooperative,between operations',
      'confirmation_input':False,'confirmation_scored':False,'development_only':True,'automatic_retry':False,
      'interpretation':'component interventions and functional positive control; no chosen deployment policy',
      'created_utc':datetime.now(timezone.utc).isoformat()})
    start=time.monotonic();meter=c.engine.WorkMeter();token=c.engine.ACTIVE_METER.set(meter);results=[];references=[]
    def check():
        if time.monotonic()-start>=1200:raise TimeoutError('component probe elapsed limit')
    try:
        for seed in c.SEEDS:
            check();model=c.model(a.model_dir,seed);validation=c.rows(pack,seed,'validation')
            w=next(x for x in warm['qualified_models']if x['seed']==seed);assert c.sha(a.warm_dir/w['checkpoint']['path'])==w['checkpoint']['sha256']
            wc=torch.load(a.warm_dir/w['checkpoint']['path'],map_location='cpu',weights_only=False);c.restore(model,wc);c.engine.restore_rng(wc['rng'])
            warm_metrics=observe(model,validation,pack,seed,f'seed{seed}_warm_reference',meter)
            references.append({'seed':seed,'metrics':warm_metrics})
            for arm,step in CASES:
                run=next(x for x in seal['runs']if x['seed']==seed and x['arm']==arm)
                cp=next(x for x in run['checkpoints']if x['primary_step']==step);path=a.v4_run_dir/cp['path']
                assert path.resolve().is_relative_to(a.v4_run_dir.resolve())and c.sha(path)==cp['sha256']
                state=torch.load(path,map_location='cpu',weights_only=False)
                slow=model.level_descriptors()[1]['bank']+'.slots.0.'
                old_names=[n for n in wc['model']if n.startswith(slow)]
                assert old_names and all(torch.equal(wc['model'][n],state['model'][n])for n in old_names),'warm receiver tensors changed'
                for probe in PROBES:
                    check();c.restore(model,state);c.engine.restore_rng(state['rng']);assert c.fingerprint(model)==cp['adapter_fingerprint']
                    intervene(model,wc,probe);metrics=observe(model,validation,pack,seed,f'seed{seed}_{arm}_{step}_{probe}',meter)
                    if probe=='actual':assert metrics==next(x for x in run['validation']if x['primary_step']==step)['metrics']
                    if probe=='warm_function_control':
                        for group in metrics:
                            assert metrics[group][c.EM]==warm_metrics[group][c.EM]
                            assert abs(metrics[group]['answer_nll_nats_per_token']-warm_metrics[group]['answer_nll_nats_per_token'])<1e-5
                    results.append({'seed':seed,'arm':arm,'step':step,'probe':probe,'metrics':metrics,'old_receiver_tensors_unchanged':True})
                c.restore(model,state);assert c.fingerprint(model)==cp['adapter_fingerprint']
                print({'seed':seed,'arm':arm,'step':step,'probes':8},flush=True)
            del model,wc,state;torch.cuda.empty_cache()
        assert len(results)==120;check()
        c.write(a.out_dir/'result.json',{'schema':'agnews-component-probes-v5','complete':True,'source_sha256':c.sha(__file__),
          'plan_sha256':c.sha(a.out_dir/'plan.json'),'results':results,'warm_references':references,'total_scored_states':123,'work':meter.counts,'seconds':time.monotonic()-start,
          'optimizer_updates':0,'observations_preserved_weights_policy_RNG':True,'confirmation_scored':False,'development_only':True})
        print('ALL120_COMPONENT_PROBES_COMPLETE',flush=True)
    except BaseException as error:
        c.write(a.out_dir/'failure.json',{'error':str(error),'type':type(error).__name__,'results':results,'work':meter.counts,'confirmation_scored':False});raise
    finally:c.engine.ACTIVE_METER.reset(token)


if __name__=='__main__':main()
