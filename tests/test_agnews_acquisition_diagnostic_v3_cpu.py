import importlib.util
import json
from pathlib import Path
import pytest
pytest.importorskip('transformers')

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("diag_v3",ROOT/"tools/run_agnews_acquisition_diagnostic_v3.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def metric(overall,c0,c1):
 return {"full_vocab_two_token_greedy_exact_match":overall,"per_class":{"0":{"two_token_exact_match":c0},"1":{"two_token_exact_match":c1}}}

def test_fixed_gate_requires_train_calibration_and_both_old_classes():
 assert mod.qualifies(metric(.80,.25,.25),metric(.75,.20,.20))
 assert not mod.qualifies(metric(.799,.50,.50),metric(.90,.5,.5))
 assert not mod.qualifies(metric(.90,0.,.9),metric(.90,.8,.8))
 assert not mod.qualifies(metric(.90,.9,.9),metric(.749,.9,.9))

def test_class_codes_are_permutation_safe_and_distinct():
 doc={"code_maps":{"17":{"class_to_token_ids":[41,42,43,44]}}}
 labels=mod.permutation_targets(doc,17,[1,0,1,0])
 assert labels==[42,41,42,41]
 assert len(set(mod.permutation_targets(doc,17,[0,1,2,3])))==4

def test_config_freezes_small_paired_budget_and_test_boundary():
 cfg=json.loads((ROOT/"configs/agnews-acquisition-diagnostic-v3.json").read_text())
 assert cfg["seeds"]==[17,29,43] and cfg["lr_pairs"]==[[.005,.01],[.02,.04]]
 assert cfg["updates_per_candidate"]==512 and cfg["maximum_updates"]==3072
 assert cfg["official_test_in_runner"] is False

def test_actual_train_step_consumes_both_due_clocks_repeatedly():
 import torch
 from nested_sleep.lm import NativeResearchLM
 from nested_sleep.native import NativeConfig
 from nested_sleep.optim import CMSController
 torch.manual_seed(47)
 model=NativeResearchLM(NativeConfig(vocab_size=32,dim=16,hidden=24,layers=1,cms_periods=(4,16)),rank=2,capacity=2)
 model.eval().requires_grad_(False)
 controller=CMSController(model,periods=(4,16),base_lr=.02,expert_lr=.04)
 meter=mod.shared.WorkMeter();token=mod.shared.ACTIVE_METER.set(meter)
 before=mod.shared.fingerprint(model)
 try:
  for _ in range(16):mod.train_step(model,controller,[{'tokens':[1,2,3],'marker':4,'label':5}]*4,0,meter,'contract')
 finally:mod.shared.ACTIVE_METER.reset(token)
 assert controller.counts==[0,0] and controller.events==16
 assert mod.shared.fingerprint(model)!=before
 assert meter.counts['contract']['forward_calls']==16
