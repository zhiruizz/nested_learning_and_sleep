"""Bind the reviewed Banking v10 numerical protocol to actual source bytes."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--training-pack',type=Path,required=True)
    p.add_argument('--asset-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review-receipt',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import run_banking_cil_v10 as driver
    review=json.loads(a.review_receipt.read_text(encoding='utf-8'))
    if review.get('status')!='approved_for_foundation':raise ValueError('Foundation review approval is missing')
    source=driver._source_hashes(a.project_root,Path(__file__).resolve().parents[1])
    if review.get('source_hashes')!=source:raise ValueError('Source changed after foundation review')
    pack_sha=hashlib.sha256(a.training_pack.read_bytes()).hexdigest()
    if pack_sha!='ed9f4b047768021cbf16a3fa042db12baf212038f89f33a00099f2e32b7300f3':raise ValueError('Unexpected training split')
    old=json.loads(a.asset_config.read_text(encoding='utf-8'))
    cfg={'schema':driver.SCHEMA,'status':'frozen','source_revision':'f0e5be08b377ef8105d15b528cb940d6010b4779+reviewed-banking-v10',
      'source_hashes':source,'model_revision':old['model_revision'],'model_assets_sha256':old['model_assets_sha256'],
      'training_manifest_sha256':pack_sha,'initialization_seed':1907008,'seeds':[17,29,43],
      'tasks':7,'classes_per_task':11,'arms':['CMS_ONLY','SLEEP_ADDITIVE','GOLD_ADDITIVE'],
      'rank':64,'capacity':8,'levels':5,'periods':[16,64,256,1024,None],'batch_size':8,
      'foundation_epochs':2,'epochs_per_task':2,'base_lr':.001,'fast_lr':.1,'sleep_lr':.001,
      'sleep_weight_decay':0.,'sleep_clip_norm':1.,'temperature':.8,'max_new_tokens':10,
      'lti_edit_distance_z0':8,'lti_semantic_weight':.5,'lti_edit_weight':.5,
      'gkd_calls_per_event':20,'lti_calls_per_event':10,'eval_examples_per_class':2,
      'max_sequence_length':128,'task0_global_candidate_gate':.5,'task0_class_macro_gate':.4,
      'foundation_max_wall_seconds':1800,'foundation_max_forward_calls':20000,'foundation_max_update_steps':1000,
      'main_max_wall_seconds':7200,'main_max_forward_calls':150000,'main_max_update_steps':15000,
      'device':'cuda:0','sleep_prompt_policy':'balanced_seen_class_cycle','candidate_batch_size':32,
      'checkpoint_every_steps':64,'objective_version':'canonical_agreement_plus_thresholded_length_normalized_edit_v1',
      'proposal':'identity-prospective transfer probe uses fixed 3/30 call checkpoints before main release',
      'lti_zero_objective_policy':'skip only exact zero complete objective; zero reward retains KL',
      'cost_matching':'same prompt identities, receiver/reset and call opportunities; report actual physical steps, information and work',
      'review_receipt_sha256':hashlib.sha256(a.review_receipt.read_bytes()).hexdigest(),
      'main_training_requires_separate_release':True,'test_payload_included':False,
      'protocol_stage':'foundation_and_development_probe_only; separate main release remains required'}
    driver.validate_config(cfg,phase='foundation')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(cfg,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(payload)
    print(json.dumps({'status':'frozen_foundation_config_created','sha256':hashlib.sha256(payload).hexdigest(),
                      'source_files':len(source),'main_authorized_by_this_artifact':False,'gpu_started':False}))
if __name__=='__main__':main()
