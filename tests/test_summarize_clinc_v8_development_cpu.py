import json
import pytest
from tools.summarize_clinc_v8_development import summarize, metric


def fixture(tmp_path):
    no,control=tmp_path/'no',tmp_path/'control'
    no.mkdir();control.mkdir()
    cfg=dict(seed=17,tasks=[0,1],initialization_seed=1907008,rank=64,capacity=8,levels=5,
             periods=[16,64,256,1024,None],epochs_per_task=1,examples_per_task=1500,
             batch_size=8,max_steps_per_task=188,base_lr=.001,fast_lr=.1,
             max_sequence_length=128,task0_acquisition_gate=.5)
    asset=dict(training_manifest_sha256='a'*64,model_revision='fixture',model_assets_sha256={'x':'b'*64})
    def write(p,x):p.write_text(json.dumps(x),encoding='utf-8')
    write(no/'protocol.json',dict(asset,config=cfg))
    write(control/'protocol.json',dict(asset,config=dict(cfg),test_data_read=False,parent_task0_checkpoint_sha256='c'*64))
    def m(correct,n):return dict(accuracy=correct/n,count=n,invalid=0,predictions_sha256='d'*64)
    ms=dict(old=m(1,4),current=m(2,4),seen=m(3,8))
    def receipt(arm):return dict(arm=arm,resume_parent_checkpoint_sha256='c'*64,active_seconds=1,
                                work_arm_delta={'full_backbone_forward_calls':2},physical_sleep_adamw_steps=0 if arm=='HOPE' else 48)
    write(no/'development-sealed.json',dict(complete=True,immutable_base_unchanged=True,runs=[receipt('HOPE')]))
    write(control/'development-sealed.json',dict(complete=True,immutable_base_unchanged=True,runs=[receipt(x) for x in ('SLEEP_MIX','GOLD_CE_MIX')]))
    (no/'metrics.jsonl').write_text(json.dumps(dict(arm='HOPE',seed=17,task_id=1,event='task_end_eval_clean_full',result={'metrics':ms}))+'\n',encoding='utf-8')
    (control/'metrics.jsonl').write_text('\n'.join(json.dumps(dict(arm=a,seed=17,task_id=1,event='task1_end_eval_clean_full',result={'metrics':ms})) for a in ('SLEEP_MIX','GOLD_CE_MIX'))+'\n',encoding='utf-8')
    q=tmp_path/'q.json'
    write(q,dict(asset,status='complete',training_only=True,selection={'test_data_read':False},
                 arms={'HOPE':{'checkpoint':{'sha256':'c'*64},'task0_full_clean_development_seen15':{'metrics':{'current':m(3,4)}}}}))
    return no,control,q


def test_shared_asset_and_integer_endpoints_are_reanalyzed_without_data(tmp_path):
    result=summarize(*fixture(tmp_path))
    assert result['complete'] and not result['test_payload_read']
    assert [r['metrics']['seen']['correct'] for r in result['rows']]==[3,3,3]
    assert result['shared_foundation']['old_development']['accuracy']==.75


def test_changed_wake_recipe_is_rejected_before_comparing_methods(tmp_path):
    no,control,q=fixture(tmp_path)
    p=control/'protocol.json'; obj=json.loads(p.read_text());obj['config']['fast_lr']=.2
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='wake schedules'):
        summarize(no,control,q)


def test_fraction_must_correspond_to_integer_correct_count():
    with pytest.raises(ValueError,match='integer numerator'):
        metric(dict(count=4,accuracy=.333,invalid=0,predictions_sha256='x'))
