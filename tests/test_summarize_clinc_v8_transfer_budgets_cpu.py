import json
from pathlib import Path
import pytest
from tools.summarize_clinc_v8_transfer_budgets import summarize


def _make_run(root,k):
 root.mkdir(parents=True)
 cfg={'inner_cycles_per_event':k,'objective_version':'paper-bounded-token-edit-v1','gkd_calls_per_event':2*k,'lti_calls_per_event':k,'seed':17,'sleep_prompt_selector':'role-balanced-rr-v2-budgeted','probe_max_wall_seconds':1800 if k==1 else 3600,'probe_max_forward_calls':20000 if k==1 else 40000,'probe_max_update_steps':600 if k==1 else 1600}
 common={'schema':'clinc-v8-controlled-transfer-probe','config':cfg,'source_lock_sha256':'s'*64,'source_hashes':{},'training_manifest_sha256':'t'*64,'parent_task0_checkpoint_sha256':'p'*64,'parent_r1_source_lock_sha256':'r'*64,'model_revision':'model@rev','model_assets_sha256':{'w':'m'*64},'source_arm':'HOPE','source_task0_clean_accuracy':.72,'source_asset_config_binding':{'r2_resume_config_differences':{}}}
 (root/'protocol.json').write_text(json.dumps(common),encoding='utf-8')
 records=[]
 for arm in ('SLEEP_MIX','GOLD_CE_MIX'):
  for event in range(16):
   losses=[]
   for call in range(3*k):
    global_call=event*(3*k)+call; role='old' if (global_call+1)%2==0 else 'current'; exposure=global_call//2
    meta={'role':role,'class_index':exposure%15,'example_index':(exposure//15)%4}
    x={'kind':'gold_ce' if arm=='GOLD_CE_MIX' else ('lti' if call%3==2 else 'gkd'),'paired_prompt_selection':meta,'prompt_id_sha256':f'id-{event}-{call}','token_count':2,'natural_vs_controlled_route_before':{'natural_top1_receiver_coverage':.25,'fresh_branch_executed_answer_coverage':1.0}}
    if arm=='GOLD_CE_MIX': x.update({'uses_gold_intent':True,'same_sample_fit_before':2.0,'same_sample_fit_after':1.5})
    else:x.update({'sample_owner':'teacher' if call%3!=1 else 'student','sampled_label_valid':call%3==0,'sampled_label_matches_prompt_gold_diagnostic':False,'same_sample_fit_before':{'kl':2.,'student_ce':1.,'objective':2.},'same_sample_fit_after':{'kl':1.5,'student_ce':.8,'objective':1.5}})
    losses.append(x)
   records.append({'event':'controlled_sleep_commit','arm':arm,'step':event,'losses':losses})
  # clean primary and secondary must not be confused
  metrics={g:{'accuracy':acc,'count':count,'invalid':0,'predictions_sha256':'h'*64} for g,acc,count in [('old',.2,300),('current',.3,300),('seen',.25,600)]}
  records.append({'event':'task1_end_eval_clean_full','arm':arm,'result':{'metrics':metrics}})
  bad={g:{'accuracy':0.,'count':count,'invalid':count,'predictions_sha256':'x'*64} for g,count in [('old',300),('current',300),('seen',600)]}
  records.append({'event':'task1_end_eval_development_full_seen','arm':arm,'result':{'metrics':bad}})
  records.append({'event':'run_complete','arm':arm,'receipt':{'task1_physical_wake_updates':188,'sleep_event_count':16,'physical_sleep_adamw_steps':48*k,'active_seconds':1.,'work_arm_delta':{'full_backbone_forward_calls':10,'full_backbone_forward_tokens':20,'executed_input_tokens_including_padding':24,'padded_input_tokens':4,'sampled_answer_tokens_including_first_eos':5}}})
 (root/'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
 (root/'development-sealed.json').write_text(json.dumps({'complete':True,'arms_completed':2,'arms_expected':['SLEEP_MIX','GOLD_CE_MIX'],'test_data_read':False,'work':{'full_backbone_forward_calls':20,'full_backbone_forward_tokens':40,'executed_input_tokens_including_padding':48,'padded_input_tokens':8,'sampled_answer_tokens_including_first_eos':10,'optimizer_or_lifecycle_update_steps':1,'sleep_events':32,'active_wall_seconds':2.}}),encoding='utf-8')
 return root


def test_aggregates_clean_primary_missing_diagnostics_and_gold_ce(tmp_path):
 r=summarize(_make_run(tmp_path/'k1',1),_make_run(tmp_path/'k10',10))
 k1=r['budgets']['k1']; assert k1['endpoint_clean_full_primary']['SLEEP_MIX']['old']['correct']==60
 assert k1['endpoint_clean_full_primary']['SLEEP_MIX']['old']['accuracy']==.2
 gold=k1['same_sample_groups']['GOLD_CE_MIX|old|gold_label|gold_ce']
 assert gold['sampled_answer_valid_rate']=={'n':0,'mean':None,'sd':None}
 assert gold['same_sample_ce_before']['mean']==2 and gold['same_sample_ce_after']['mean']==1.5
 teacher=k1['same_sample_groups']['SLEEP_MIX|current|teacher|gkd']
 assert teacher['sampled_answer_valid_rate']['n']>0 and teacher['sampled_answer_valid_rate']['mean']==1
 assert k1['exposure']['GOLD_CE_MIX']['gold_known_label_exposures']==48
 assert k1['exposure']['GOLD_CE_MIX']['gold_target_tokens']==96


def test_rejects_wrong_objective_and_budget_tags(tmp_path):
 a=_make_run(tmp_path/'a',1); b=_make_run(tmp_path/'b',10)
 protocol=json.loads((b/'protocol.json').read_text());protocol['config']['objective_version']='legacy-proxy';(b/'protocol.json').write_text(json.dumps(protocol))
 with pytest.raises(ValueError,match='wrong objective'):
  summarize(a,b)


def test_rejects_k_or_selector_tag_mismatch(tmp_path):
 a=_make_run(tmp_path/'a',1); b=_make_run(tmp_path/'b',10)
 protocol=json.loads((b/'protocol.json').read_text());protocol['config']['inner_cycles_per_event']=1;(b/'protocol.json').write_text(json.dumps(protocol))
 with pytest.raises(ValueError,match='budget tag/config mismatch'):
  summarize(a,b)
 protocol=json.loads((b/'protocol.json').read_text());protocol['config']['inner_cycles_per_event']=10;protocol['config']['sleep_prompt_selector']='legacy-selector';(b/'protocol.json').write_text(json.dumps(protocol))
 with pytest.raises(ValueError,match='wrong selector tag'):
  summarize(a,b)


def test_rejects_mixed_source_checkpoint_or_model(tmp_path):
 a=_make_run(tmp_path/'a',1);b=_make_run(tmp_path/'b',10)
 protocol=json.loads((b/'protocol.json').read_text());protocol['parent_task0_checkpoint_sha256']='different';(b/'protocol.json').write_text(json.dumps(protocol))
 with pytest.raises(ValueError,match='parent_task0_checkpoint_sha256'):
  summarize(a,b)
