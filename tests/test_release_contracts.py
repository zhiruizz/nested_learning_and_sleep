from dataclasses import asdict
import importlib
import json
from pathlib import Path
import pytest
import torch
from nested_sleep.repairability import Assumptions, classify_position, bounds, UNREACHABLE, REACHABLE, UNKNOWN
from nested_sleep.execution import select_executor, AcceptanceUnavailable
from nested_sleep.llama_entry import validate
from nested_sleep.experts import SparseExperts
from nested_sleep.lm import NativeResearchLM
from nested_sleep.native import NativeConfig
from nested_sleep.lifecycle import SleepSession, fingerprint

def test_all_exported_modules_import_without_optional_llama_dependency():
    import nested_sleep
    for name in ('accounting','clocks','consolidation','dreaming','execution','executor_v13','experts','lifecycle','llama_entry','lm','memory','native','objectives','optim','repairability'):
        importlib.import_module('nested_sleep.'+name)
    assert torch.version.cuda is None
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('premise',list(asdict(Assumptions())))
def test_every_missing_structural_premise_returns_unknown(premise):
    good={k:True for k in asdict(Assumptions())}
    good[premise]=False
    assert classify_position(False,good)['status']==UNKNOWN

def test_zero_gradient_does_not_imply_unreachable():
    a=Assumptions(**{k:True for k in asdict(Assumptions())})
    assert classify_position(True,a,local_gradient_norm=0)['status']==REACHABLE
    assert classify_position(None,a)['status']==UNKNOWN

def test_lower_bound_keeps_repetitions_unknown_mass_and_signed_roundoff():
    d={'KL_FP32':[.4,.2,-1e-9,.1],'statuses':[UNREACHABLE,REACHABLE,UNKNOWN,UNREACHABLE]}
    r=bounds([d,d],'ALL4')
    assert r['N_draws']==2 and r['positions_with_positive_loss_weight']=={'unreachable':4,'reachable':2,'unknown':2}
    assert r['LB_actual_objective_FP32_estimate']==pytest.approx(.125)
    assert r['unknown_position_weight_mass']==.25 and r['negative_FP32_values_preserved']==2
    assert r['LB_actual_objective_FP64_normalized_estimate'] is None

@pytest.mark.parametrize('mode',['CACHE','HOST','BOTH'])
def test_absent_acceptance_never_grants_acceleration(mode):
    assert select_executor('auto',scope={'unknown':True})['executor']=='REF'
    with pytest.raises(AcceptanceUnavailable):
        select_executor(mode,scope={'unknown':True})

def test_sparse_reset_retains_basis_and_activates_fresh_receiver():
    b=SparseExperts(3,rank=2,capacity=2)
    b.activate(0)
    a=b.slots[0].a.detach().clone()
    with torch.no_grad(): b.slots[0].b.fill_(.2)
    assert b.reset_active()==[0]
    assert torch.equal(a,b.slots[0].a) and b.slots[0].b.abs().sum()==0
    assert not bool(b.slots[0].active)
    assert b.activate(1)==1 and bool(b.slots[1].active)

def test_fixed_sequence_distillation_changes_receiver_without_live_mutation():
    torch.manual_seed(71)
    live=NativeResearchLM(NativeConfig(vocab_size=12,dim=4,hidden=5),rank=2,capacity=3).eval()
    live.banks()[0].activate()
    before=fingerprint(live)
    names=dict(live.named_parameters())
    proposal={n:names[n].detach().clone()+.01 for n in live.level_descriptors()[0]['base']}
    session=SleepSession(live,0,proposal,lr=.01)
    initial=session.new_slot.b.detach().clone()
    # Fixed synthetic IDs, no autoregressive generation or external tokenizer.
    session.teacher_forced_step(torch.tensor([[1,2,3,4]]),2)
    session.teacher_forced_step(torch.tensor([[1,2,3,4]]),2)
    assert not torch.equal(initial,session.new_slot.b)
    assert fingerprint(live)==before and fingerprint(session.teacher)==before
    assert session.work['generated_tokens']==0
    session.finish_consolidation();session.commit()

def test_terminal_sender_is_rejected():
    live=NativeResearchLM(NativeConfig(vocab_size=12,dim=4,hidden=5),rank=2,capacity=3)
    with pytest.raises(ValueError,match='UNSPECIFIED'):
        SleepSession(live,1,{})

def test_llama_metadata_only_and_missing_assets_refused(tmp_path):
    with pytest.raises(ValueError): validate(None)
    with pytest.raises(ValueError): validate(str(tmp_path/'missing'))
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'llama','hidden_size':8}))
    assert validate(str(tmp_path))['status']=='NOT_RUN'
    with pytest.raises(ValueError): validate(str(tmp_path),levels=3)
    with pytest.raises(AcceptanceUnavailable): validate(str(tmp_path),executor='CACHE')

def test_real_structural_counterexamples():
    import runpy
    run=runpy.run_path(str(Path(__file__).parents[1]/'examples/structural_counterexamples.py'))['run']
    assert run()['reachable_zero_A_gradient']

def test_bundled_reanalysis_keeps_complete_denominators():
    import runpy
    root=Path(__file__).parents[1]
    summarize=runpy.run_path(str(root/'tools/reanalyze.py'))['summarize']
    output=summarize(root/'results')
    assert '0/6' in output and '2/3' in output and 'UNAVAILABLE_COORDINATOR_INTERRUPTION' in output
