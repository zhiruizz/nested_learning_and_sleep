from pathlib import Path
import sys
import pytest
import torch
pytest.importorskip('transformers')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import probe_agnews_components_v5 as p
from nested_sleep.lm import execution_state
from nested_sleep.native import NativeConfig
from nested_sleep.lm import NativeResearchLM

def model():
    m=NativeResearchLM(NativeConfig(vocab_size=9,dim=4,hidden=5,layers=1,cms_periods=(4,16)),rank=1,capacity=3)
    m.requires_grad_(False);m.banks()[0].activate(0);m.banks()[1].activate(0)
    return m.eval()

def test_component_intervention_restores_only_requested_base():
    m=model();warm={'model':{n:v.clone()for n,v in m.state_dict().items()},'execution_state':execution_state(m)}
    base0=m.level_descriptors()[0]['base'];base1=m.level_descriptors()[1]['base'];params=dict(m.named_parameters())
    with torch.no_grad():
        for n in base0+base1:params[n].add_(.2)
    before={n:v.clone()for n,v in m.state_dict().items()};p.intervene(m,warm,'restore_base0_force_warm')
    assert all(torch.equal(params[n],warm['model'][n])for n in base0)
    assert all(torch.equal(params[n],before[n])for n in base1)
    assert m.banks()[1].route_override=={'force':0}

def test_functional_control_removes_fast_and_preserves_receiver_weights():
    m=model();m.banks()[0].deactivate(0);m.banks()[1].route_override={'mix':0,'fraction':.5}
    with torch.no_grad():m.banks()[1].slots[0].b.fill_(.04)
    warm={'model':{n:v.clone()for n,v in m.state_dict().items()},'execution_state':execution_state(m)}
    receiver={n:v.clone()for n,v in m.banks()[1].slots[0].state_dict().items()}
    ids=torch.tensor([[1,2,3]]);expected=m.logits(ids).detach().clone()
    m.banks()[0].activate(0);m.banks()[1].activate(1)
    with torch.no_grad():
        m.banks()[0].slots[0].b.fill_(.2);m.banks()[1].slots[1].b.fill_(.1)
        for n,param in m.named_parameters():
            if n in sum([x['base']for x in m.level_descriptors()],[]):param.add_(.05)
    p.intervene(m,warm,'warm_function_control')
    assert m.banks()[0].route_override is None
    assert m.banks()[1].route_override=={'mix':0,'fraction':.5}
    assert [int(s.active)for s in m.banks()[1].slots]==[1,0,0]
    assert all(torch.equal(v,receiver[n])for n,v in m.banks()[1].slots[0].state_dict().items())
    assert torch.equal(m.logits(ids),expected)

def test_audit_counts_only_loss_positions_and_preserves_state(monkeypatch):
    m=model();rows=[{'tokens':[1,2],'class_id':0},{'tokens':[1,2],'class_id':2}]
    monkeypatch.setattr(p.c.engine,'capture_rng',lambda:{'fixture':1})
    def score(model,*args):
        for bank in model.banks():bank(torch.ones(2,3,4))
        return {'fixture':True}
    monkeypatch.setattr(p.c,'score',score)
    metrics,audits=p.observe(m,rows,{},17,'fixture',object())
    assert metrics=={'fixture':True}
    assert len(audits)==2
    for a in audits:
        assert a['old']['positions']==a['new']['positions']==2
        assert sum(a['old']['selected_slot_counts'].values())==2
