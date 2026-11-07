import json
from pathlib import Path
from tools.summarize_clinc_v8_controlled_probe import summarize


def _fixture(tmp_path):
    root=tmp_path/'run'; root.mkdir()
    (root/'protocol.json').write_text(json.dumps({'config':{'seed':17},'source_lock_sha256':'a'*64,'source_hashes':{},'training_manifest_sha256':'b'*64,'parent_task0_checkpoint_sha256':'c'*64}),encoding='utf-8')
    events=[]
    for arm in ('SLEEP_MIX','GOLD_CE_MIX'):
        for e in range(16):
            losses=[]
            for call,kind in enumerate(('gkd','gkd','lti')):
                role='old' if call%2 else 'current'
                meta={'role':role,'class_index':e%15,'example_index':0}
                common={'kind':'gold_ce' if arm=='GOLD_CE_MIX' else kind,'paired_prompt_selection':meta,
                    'prompt_id_sha256':f'{arm}-{e}-{call}','natural_vs_controlled_route_before':{'natural_top1_receiver_coverage':0.0,'fresh_branch_executed_answer_coverage':1.0}}
                if arm=='SLEEP_MIX':
                    common.update({'sample_owner':'teacher' if call!=1 else 'student','sampled_label_valid':call==0,'sampled_label_matches_prompt_gold_diagnostic':False,
                        'same_sample_fit_before':{'kl':2.0,'student_ce':1.0,'objective':2.0},'same_sample_fit_after':{'kl':1.5,'student_ce':0.8,'objective':1.5}})
                else:
                    common.update({'uses_gold_intent':True,'same_sample_fit_before':2.0,'same_sample_fit_after':1.5})
                losses.append(common)
            events.append({'event':'controlled_sleep_commit','arm':arm,'losses':losses})
        events.append({'event':'run_complete','arm':arm,'receipt':{'active_seconds':1,'work_arm_delta':{'full_backbone_forward_calls':2},'physical_sleep_adamw_steps':48,'sleep_event_count':16}})
    (root/'metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in events)+'\n',encoding='utf-8')
    (root/'development-sealed.json').write_text(json.dumps({'complete':True,'arms_completed':2,'test_data_read':False,'work':{'full_backbone_forward_calls':4}}),encoding='utf-8')
    return root


def test_missing_sampling_diagnostics_are_na_and_gold_scalar_ce_is_preserved(tmp_path):
    result=summarize(_fixture(tmp_path))
    gold=result['by_arm_role_owner_kind']['GOLD_CE_MIX|current|gold_label|gold_ce']
    assert gold['sampled_answer_valid_rate']=={'n':0,'mean':None,'sd':None}
    assert gold['sampled_answer_matches_prompt_gold_diagnostic_rate']['n']==0
    assert gold['same_sample_ce_before']['mean']==2.0
    assert gold['same_sample_ce_after']['mean']==1.5


def test_sleep_boolean_rates_use_present_fields_and_natural_differs_from_forced(tmp_path):
    result=summarize(_fixture(tmp_path))
    teacher=result['by_arm_role_owner_kind']['SLEEP_MIX|current|teacher|gkd']
    assert teacher['sampled_answer_valid_rate']['n']==16
    assert teacher['sampled_answer_valid_rate']['mean']==1.0
    assert teacher['sampled_answer_matches_prompt_gold_diagnostic_rate']['mean']==0.0
    assert teacher['natural_top1_fresh_receiver_coverage']['mean']==0.0
    assert teacher['controlled_fresh_branch_executed_coverage']['mean']==1.0
    assert result['teacher_sample_summary']['n']==32


def test_primary_endpoint_uses_clean_cohort_even_if_secondary_appears_later(tmp_path):
    root=_fixture(tmp_path)
    def event(kind,accuracy):
        return {'event':kind,'arm':'SLEEP_MIX','result':{'metrics':{
            'old':{'accuracy':accuracy,'count':300,'invalid':1,'predictions_sha256':'d'*64}}}}
    with (root/'metrics.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event('task1_end_eval_clean_full',.5))+'\n')
        stream.write(json.dumps(event('task1_end_eval_development_full_seen',.2))+'\n')
    out=summarize(root)['endpoint_seen_clean_development']['SLEEP_MIX']['old']
    assert out['accuracy']==.5 and out['correct']==150
