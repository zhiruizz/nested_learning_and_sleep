import json
from pathlib import Path
import pytest
from tools.run_clinc_controlled_transfer_v8 import (
    _select_event_prompts,_sleep_method_schedule,_prompt_selection_meta,
    bounded_token_edit_similarity,transfer_lti_reward,validate_probe_config,_source_asset_config,
    _assert_no_prior_sleep)
from tools import run_clinc_cil_v8_r2 as r2


def _pools():
    old=[{'id':f'o-{c}-{i}','intent_id':c} for c in range(15) for i in range(4)]
    new=[{'id':f'n-{c}-{i}','intent_id':c+15} for c in range(15) for i in range(4)]
    return {'old':old,'current':new}


def test_k1_is_legacy_48_request_schedule_and_paired_ids():
    cfg={'inner_cycles_per_event':1,'gkd_calls_per_event':2,'lti_calls_per_event':1}
    pool=_pools()
    actual=[x for e in range(16) for x in _select_event_prompts(pool,event=e,task_id=1,cfg=cfg,r2=r2)]
    by_role=[sorted([r for r in pool[role]],key=lambda row:(row['intent_id'],row['id'])) for role in ('old','current')]
    assert len(actual)==48
    assert [kind for kind,_ in actual[:3]]==['gkd','gkd','lti']
    assert sum(_prompt_selection_meta(event=i//3,call=i%3,task_id=1)['role']=='old' for i in range(48))==24
    expected=[]
    for i in range(48):
        meta=_prompt_selection_meta(event=i//3,call=i%3,task_id=1)
        expected.append(by_role[meta['role_index']][meta['class_index']*4+meta['example_index']]['id'])
    assert [row['id'] for _,row in actual]==expected


def test_k10_triples_cover_15_classes_per_role_each_event_and_pool_ids():
    cfg={'inner_cycles_per_event':10,'gkd_calls_per_event':20,'lti_calls_per_event':10}
    pool=_pools()
    calls=[]
    for e in range(16):
        one=_select_event_prompts(pool,event=e,task_id=1,cfg=cfg,r2=r2)
        assert len(one)==30
        assert [kind for kind,_ in one]==['gkd','gkd','lti']*10
        role_classes={'old':set(),'current':set()}
        for c,(_kind,row) in enumerate(one):
            meta=_prompt_selection_meta(event=e,call=c,task_id=1,method_count=30)
            role_classes[meta['role']].add(meta['class_index'])
        assert role_classes=={'old':set(range(15)),'current':set(range(15))}
        calls.extend(one)
    assert len(calls)==480
    for role in ('old','current'):
        ids=set()
        for i,(_kind,row) in enumerate(calls):
            meta=_prompt_selection_meta(event=i//30,call=i%30,task_id=1,method_count=30)
            if meta['role']==role:ids.add(row['id'])
        assert len(ids)==60


def test_both_locked_budget_variants_and_triplet_update_totals():
    expected={1:(1800,20000,600,472),10:(3600,40000,1600,1336)}
    for k,(wall,fw,updates,total_updates) in expected.items():
        path=Path(f'configs/clinc_cil_v8_controlled_transfer_k{k}_seed17.json')
        cfg=validate_probe_config(json.loads(path.read_text(encoding='utf-8')))
        assert (cfg['probe_max_wall_seconds'],cfg['probe_max_forward_calls'],cfg['probe_max_update_steps'])==(wall,fw,updates)
        assert 2*(188+16*3*k)==total_updates


def test_bounded_paper_edit_term_cutoff_normalization_and_empty():
    assert bounded_token_edit_similarity([],[],8)==1.0
    assert bounded_token_edit_similarity([1,2,3],[1,2,4],1)==pytest.approx(2/3)
    assert bounded_token_edit_similarity([1,2,3],[4,5,6],2)==0.0
    assert transfer_lti_reward('', '', teacher_tokens=[],student_tokens=[],z0=8,
        canonical_labels=['x'])==pytest.approx(.5)
    assert transfer_lti_reward('alpha','alpha',teacher_tokens=[1],student_tokens=[1],z0=8,
        canonical_labels=['alpha'])==1.0
    assert transfer_lti_reward('card swallowed','card_swallowed',teacher_tokens=[1],student_tokens=[1],z0=8,
        canonical_labels=['card_swallowed'])==1.0


def test_budget_config_only_changes_dormant_gkd_lti_source_fields(tmp_path):
    k10=json.loads(Path('configs/clinc_cil_v8_controlled_transfer_k10_seed17.json').read_text())
    target=validate_probe_config(k10)
    base=json.loads(Path('configs/clinc_cil_v8_hope_task1_resume_r2.json').read_text())
    parent_cfg=dict(base,schema='clinc-cil-v8-pilot',arms=['HOPE'])
    # The actual R1 protocol predates scoped evaluation and saved less often.
    parent_cfg.pop('full_development_scope',None)
    parent_cfg['checkpoint_every_steps']=64
    parent=tmp_path/'parent'; parent.mkdir()
    (parent/'protocol.json').write_text(json.dumps({'config':parent_cfg}),encoding='utf-8')
    source,note=_source_asset_config(parent,target,r2)
    assert source['gkd_calls_per_event']==2 and source['lti_calls_per_event']==1
    assert note['target_gkd_lti']=={'gkd_calls_per_event':20,'lti_calls_per_event':10}
    assert note['allowed_nontraining_metadata_differences']['full_development_scope']['parent'] is None
    assert note['allowed_nontraining_metadata_differences']['checkpoint_every_steps']['parent']==64
    parent_cfg['base_lr']*=2
    (parent/'protocol.json').write_text(json.dumps({'config':parent_cfg}),encoding='utf-8')
    with pytest.raises(ValueError,match='outside dormant budget allowlist'):
        _source_asset_config(parent,target,r2)


def test_parent_receipt_must_be_a_never_slept_hope_checkpoint():
    _assert_no_prior_sleep({'sleep_events':0,'physical_sleep_adamw_steps':0,
                            'work_arm_delta':{'sleep_events':0}})
    _assert_no_prior_sleep({'sleep_event_count':0,'physical_sleep_adamw_steps':0,
                            'work_arm_delta':{'sleep_events':0}})
    with pytest.raises(ValueError,match='zero prior Sleep'):
        _assert_no_prior_sleep({'sleep_event_count':1,'physical_sleep_adamw_steps':3,
                                'work_arm_delta':{'sleep_events':1}})
    with pytest.raises(ValueError,match='zero prior Sleep'):
        _assert_no_prior_sleep({'sleep_events':0,'sleep_event_count':1,'physical_sleep_adamw_steps':0,
                                'work_arm_delta':{'sleep_events':0}})
