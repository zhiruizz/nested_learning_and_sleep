import sys
from pathlib import Path

import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import diagnose_agnews_interference_v5 as diag

def test_gradient_group_summary_and_zero_norm():
    a=[torch.tensor([1.,2.]),torch.tensor([3.])]
    b=[torch.tensor([2.,4.]),torch.tensor([6.])]
    got=diag.vector_summary(a,b)
    assert got['dot']==28.0 and got['cosine']==pytest.approx(1.0)
    zero=diag.vector_summary([torch.zeros(2)],[torch.ones(2)])
    assert zero['norm_a']==0 and zero['cosine'] is None

def test_nested_gradient_call_counter_accumulates_across_probe_states():
    add=diag.make_gradient_call_counter()
    assert add(4)==4
    assert add(2)==6
    assert add(0)==6

def test_guarded_observation_preserves_model_grads_mode_policy_and_rng():
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh()).eval()
    model[0].weight.grad=torch.full_like(model[0].weight,0.25)
    policy=lambda m: {'mode':m.training,'tag':'fixed'}
    rng=lambda:torch.get_rng_state().clone()
    before={k:v.clone() for k,v in model.state_dict().items()}
    out=diag.guarded_observation(model,lambda:float(model(torch.ones(2,3)).sum().detach()),policy,rng)
    assert isinstance(out,float)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert torch.all(model[0].weight.grad==0.25) and model.training is False

def test_guard_uses_mutable_adapter_and_frozen_versions_without_full_state_copy(monkeypatch):
    class AdapterOnly(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=torch.nn.Parameter(torch.ones(3)); self.adapter=torch.nn.Parameter(torch.zeros(2))
        def adapter_state_dict(self): return {'adapter':self.adapter.detach().clone()}
        def state_dict(self,*args,**kwargs): raise AssertionError('full-model state_dict must not be cloned')
    model=AdapterOnly().eval()
    monkeypatch.setattr(diag,'immutable_versions',lambda m:tuple((id(p),p._version) for n,p in m.named_parameters() if n=='backbone'))
    out=diag.guarded_observation(model,lambda:float(model.adapter.sum().detach()),lambda m:None,lambda:torch.get_rng_state())
    assert out==0.0

def test_inactive_warm_fast_group_is_reported_absent_without_activation():
    class Slot:
        active=False
        def parameters(self): return iter(())
    class Bank: slots=[Slot()]
    class M:
        def banks(self): return [Bank()]
    model=M()
    fast=diag.active_fast_params(model); groups={'base0':[],'base1':[]}
    assert fast==[] and diag.attach_fast_group(groups,fast) is True
    assert 'fast_active_expert' not in groups
    summary=diag.absent_group_summary()
    assert summary['norm_a']==0 and summary['norm_b']==0 and summary['cosine'] is None
    assert summary['gradient_requires_grad_scope']==[] and summary['absent_reason']

def test_guard_rejects_observation_that_changes_parameter():
    model=torch.nn.Linear(2,2).eval()
    with pytest.raises(AssertionError,match='changed mutable model tensors'):
        diag.guarded_observation(model,lambda: model.weight.data.add_(1),lambda m:None,lambda:torch.get_rng_state())
