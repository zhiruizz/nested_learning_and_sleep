import json
import pytest
from tools.summarize_clinc_v8_controlled_boundaries import summarize,STAGES,COHORTS

def fixture(tmp_path):
 p=tmp_path/'eval.json'
 pairs=[]
 for i in range(32):
  metric={}
  for j,stage in enumerate(STAGES):
   metric[stage]={c:{'accuracy':.5+j*.1,'count':30,'invalid':3,'predictions_sha256':('same' if stage in ('teacher_pre','base_only_reconstruction') else f'{stage}-{i}-{c}')} for c in COHORTS}
  pairs.append({'arm':'SLEEP_MIX' if i<16 else 'GOLD_CE_MIX','event_index':i%16,'wake_step':4,'global_wake_tick':4,'sender_level':i%3,'pre_checkpoint_sha256':'a'*64,'post_checkpoint_sha256':'b'*64,'post_safe_to_resume_next_wake':True,'post_other_due_levels_pending':[],'metrics':metric})
 data={'schema':'clinc-controlled-boundary-evaluation-v3','status':'complete','test_data_read':False,'scoring_scope':{'test_data_read':False,'old_count':30,'current_count':30,'seen_count':60},'zero_optimizer_steps':True,'zero_wake_updates_or_resume':True,'immutable_backbone_unchanged':True,'counterfactuals_are_reconstructions_not_training_checkpoints':True,'evaluator_source_sha256':'e'*64,'config_sha256':'c'*64,'probe_protocol_sha256':'p'*64,'probe_metrics_sha256':'m'*64,'probe_seal_sha256':'s'*64,'probe_source_lock_sha256':'l'*64,'training_manifest_sha256':'t'*64,'parent_task0_checkpoint_sha256':'q'*64,'model_revision':'model','model_assets_sha256':{},'wall_cap_seconds':900,'forward_cap':16000,'elapsed_seconds':2,'work':{'full_backbone_forward_calls':10,'full_backbone_forward_tokens':30,'optimizer_or_lifecycle_update_steps':0,'tail_backward_calls':0},'paired_events':pairs}
 p.write_text(json.dumps(data),encoding='utf-8');return p

def test_aggregates_all_four_stages_and_checks_predigest_identity(tmp_path):
 r=summarize(fixture(tmp_path));assert r['scope']['pairs']==32
 a=r['aggregate_by_arm_sender_stage_cohort']['SLEEP_MIX|sender0|teacher_pre|old']
 assert a['correct']==90 and a['count']==180 and a['accuracy']==.5
 assert r['pairs'][0]['teacher_equals_base_only_predictions']['old'] is True

def test_refuses_test_read_or_missing_stage(tmp_path):
 p=fixture(tmp_path);d=json.loads(p.read_text());d['test_data_read']=True;p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='test data'):summarize(p)
 d=json.loads(fixture(tmp_path).read_text());d['paired_events'][0]['metrics'].pop('optimized_post');p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='exact four'):summarize(p)
