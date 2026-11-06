import importlib.util
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("summary_v3",ROOT/"tools/summarize_agnews_qualified_v3.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ARMS=("frozen","sft","cms","cms_sleep","cms_extra_replay")
SEEDS=(17,29,43)
KEY="full_vocab_two_token_greedy_exact_match"
def metrics(v):
 return {g:{"examples":count,"full_vocab_two_token_greedy_exact_match":v,"answer_nll_nats_per_token":1.0+v} for g,count in {'old':512,'new':512,'all':1024,**{f'class{i}':256 for i in range(4)}}.items()}
def fixture(tmp_path):
 run=tmp_path/"run"; ev=tmp_path/"eval"; run.mkdir();ev.mkdir()
 runs=[];results={}
 for seed in SEEDS:
  for arm in ARMS:
   steps=[0,512]if arm=='frozen'else[0,128,256,384,512]
   cps=[{'base_step':s,'sha256':f'fixture-{seed}-{arm}-{s}'}for s in steps]
   runs.append({"seed":seed,"arm":arm,"status":"complete","checkpoints":cps,"primary_updates":0 if arm=="frozen" else 512,"extra_replay_updates":96 if arm=="cms_extra_replay" else 0,"sleep_adam_updates":192 if arm=="cms_sleep" else 0,"seconds":100+seed,"work":{"train":{"forward_input_tokens":1000,"gradient_forward_input_tokens":900,"forward_calls":10}}})
   results[f"{seed}/{arm}"]=[{"base_step":s,'checkpoint_sha256':cp['sha256'],"metrics":metrics(.8 if s==0 else .7+(seed-17)/100)}for s,cp in zip(steps,cps)]
 seal={"schema":"agnews-qualified-training-seal-v3","complete":True,"new_steps":512,"old_all_qualified":True,"heldout_read":False,"records_sha256":"abc","runs":runs,"operational_amendments":[],"interrupted_duplicate_work_not_in_logical_counts":True}
 evaluation={"schema":"agnews-qualified-evaluation-v3","all_foundations_qualified":True,"trained_after_holdout_open":False,"provenance":{"records_sha256":"abc"},"results":results,"seconds":30,"resumed_scores":0,"work":{}}
 (run/"training-sealed.json").write_text(json.dumps(seal));(ev/"evaluation-summary.json").write_text(json.dumps(evaluation))
 protocol={'config_sha256':'config-fixture','config':{'records_sha256':'abc','evaluator_sha256':'e','helper_source_sha256':'h','holdout_prompts_sha256':'p','holdout_labels_sha256':'l'}}
 (run/'protocol.json').write_text(json.dumps(protocol));seal['protocol_sha256']=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest();seal['config_sha256']='config-fixture'
 (run/'training-sealed.json').write_text(json.dumps(seal))
 evaluation['provenance'].update(seal_sha256=hashlib.sha256((run/'training-sealed.json').read_bytes()).hexdigest(),evaluator_sha256='e',helper_sha256='h',holdout_prompts_sha256='p',holdout_labels_sha256='l')
 evaluation.update(unique_holdout_examples=1024,examples_per_class=256)
 (ev/'evaluation-summary.json').write_text(json.dumps(evaluation))
 return run,ev
def test_descriptive_summary_and_separate_costs():
 with TemporaryDirectory(dir=ROOT) as td:
  run,ev=fixture(Path(td)); out=mod.summarize(run,ev)
  arm=out["arms"]["cms_sleep"]
  assert arm["endpoint_metrics_mean_sample_sd"]["old"][KEY]["n"]==3
  assert arm["old_drop_mean_sample_sd"]["mean"]==pytest.approx(-0.0266666666666666)
  assert arm["logical_training_cost_mean_sample_sd"]["logical_sgd_updates"]["mean"]==512
  assert arm["logical_training_cost_mean_sample_sd"]["sleep_adam_updates"]["mean"]==192
  assert out["recovery_operational_cost_separate"]["duplicate_interrupted_work"] is True
  assert "not a significance test" in out["interpretation"]["seed_statistics"]
def test_reject_incomplete_seal():
 with TemporaryDirectory(dir=ROOT) as td:
  run,ev=fixture(Path(td)); p=run/"training-sealed.json"; doc=json.loads(p.read_text());doc["complete"]=False;p.write_text(json.dumps(doc))
  with pytest.raises(ValueError,match="incomplete"):
   mod.summarize(run,ev)

def test_reject_wrong_endpoint_and_seal_provenance(tmp_path):
 run,ev=fixture(tmp_path);p=ev/'evaluation-summary.json';doc=json.loads(p.read_text())
 doc['results']['17/sft'][-1]['base_step']=511;p.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match='endpoints'):mod.summarize(run,ev)
 doc['results']['17/sft'][-1]['base_step']=512;doc['provenance']['seal_sha256']='wrong';p.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match='seal SHA256'):mod.summarize(run,ev)

def test_reject_wrong_dataset_provenance_and_denominator(tmp_path):
 run,ev=fixture(tmp_path);p=ev/'evaluation-summary.json';d=json.loads(p.read_text())
 d['provenance']['holdout_prompts_sha256']='wrong';p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='evaluation provenance'):mod.summarize(run,ev)
 d['provenance']['holdout_prompts_sha256']='p';d['results']['17/frozen'][0]['metrics']['old']['examples']=256;p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='denominator'):mod.summarize(run,ev)
