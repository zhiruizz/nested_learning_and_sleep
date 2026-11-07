"""Create a main release only from actual qualified foundations/probe evidence."""
from __future__ import annotations
import argparse,datetime,hashlib,json,sys
from pathlib import Path

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','model-path','training-pack','config','foundation-root','probe-result','probe-preflight','result-review','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import run_banking_cil_v10 as d
    cfg=d.validate_config(read(a.config),phase='main');cfg_sha=d._config_sha(cfg)
    sources,lock,manifest,train,dev=d._source_prepare(a.project_root,Path(__file__).resolve().parents[1],a.model_path,a.training_pack,cfg)
    probe=read(a.probe_result);preflight=read(a.probe_preflight);review=read(a.result_review)
    if (review.get('status'),review.get('probe_artifact_sha256'))!=('approved_for_main_release',sha(a.probe_result)):
        raise ValueError('Actual probe result has not passed the scoped independent review')
    if (preflight.get('status'),preflight.get('cuda_initialized'),preflight.get('model_loaded'))!=('preflight_pass',False,False):
        raise ValueError('Real CPU probe preflight is missing')
    if (probe.get('schema'),probe.get('status'),probe.get('seed'),probe.get('test_data_read'),
            probe.get('model_immutable_versions_unchanged'))!=('banking-cil-v10-transfer-probe-v1','complete',17,False,True):
        raise ValueError('Probe did not complete with immutable model/test isolation')
    for k,want in [('config_sha256',cfg_sha),('banking_driver_source_lock_sha256',lock),('training_manifest_sha256',cfg['training_manifest_sha256'])]:
        if probe.get(k)!=want or preflight.get(k)!=want:raise ValueError('Probe/preflight binding mismatch: '+k)
    if probe.get('source_lock_sha256')!=preflight.get('probe_source_lock_sha256'):
        raise ValueError('Probe executed a different source version from its preflight')
    arms=probe.get('arms',{})
    if set(arms)!={'SLEEP_ADDITIVE','GOLD_ADDITIVE','SLEEP_FIXEDMIX','GOLD_FIXEDMIX'}:
        raise ValueError('Incomplete diagnostic four-arm comparison')
    for arm,r in arms.items():
        if r.get('opportunity_count')!=30 or set(r.get('metrics_after_opportunities',{}))!={'0','3','30'}:
            raise ValueError('Incomplete fixed 3/30 trajectory: '+arm)
    parity=arms['SLEEP_ADDITIVE'].get('disk_resume_equivalence',{})
    if parity.get('exact_model_adam_tx_rng_process_rng')is not True or parity.get('continuous_snapshot_sha256')!=parity.get('restored_snapshot_sha256'):
        raise ValueError('Real model/Adam/sample/process RNG replay did not match')
    foundations={}
    for seed in (17,29,43):
        root=(a.foundation_root/f'seed{seed}').resolve();rpath=root/'foundation-receipt.json';r=read(rpath)
        if r.get('status')!='qualified':raise ValueError('Unqualified foundation')
        cp=(root/r['checkpoint']).resolve()
        if root not in cp.parents or sha(cp)!=r['checkpoint_sha256']:raise ValueError('Foundation checkpoint changed')
        foundations[str(seed)]={k:r[k]for k in('status','config_sha256','source_lock_sha256','training_manifest_sha256','checkpoint_sha256')}
        foundations[str(seed)].update(run_dir=str(root),receipt_sha256=sha(rpath))
    if (probe['foundation_checkpoint_sha256'],probe['foundation_receipt_sha256'])!=(foundations['17']['checkpoint_sha256'],foundations['17']['receipt_sha256']):
        raise ValueError('Probe did not use this exact qualified seed17 foundation')
    result={'schema':'banking-cil-v10-main-release-v1','status':'ready','config_sha256':cfg_sha,
      'source_lock_sha256':lock,'training_manifest_sha256':cfg['training_manifest_sha256'],
      'foundations':foundations,'probe_source_lock_sha256':probe['source_lock_sha256'],
      'transfer_probe':{'status':'complete','resume_parity':True,'source_lock_sha256':probe['source_lock_sha256'],
        'artifact_path':str(a.probe_result.resolve()),'artifact_sha256':sha(a.probe_result)},
      'independent_result_review_sha256':sha(a.result_review),'probe_cpu_preflight_sha256':sha(a.probe_preflight),
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'test_data_included':False,'test_scores_used_for_release':False,
      'budget_decision':'Retain the frozen 30-opportunity budget; diagnostic trajectories are reported, not used to select a best checkpoint.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix('.candidate.json')
    if temporary.exists():raise FileExistsError(temporary)
    payload=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    with temporary.open('xb')as f:f.write(payload)
    for seed in (17,29,43):d._validate_foundation_release(temporary,cfg,cfg_sha,lock,cfg['training_manifest_sha256'],seed,a.foundation_root/f'seed{seed}')
    with a.output.open('xb')as f:f.write(payload)
    temporary.unlink()
    print(json.dumps({'status':'main_release_verified','release_sha256':sha(a.output),'seeds':[17,29,43],'gpu_started':False}))
if __name__=='__main__':main()
