"""A recovery snapshot must preserve pending CMS sums, policy and all RNGs."""
from pathlib import Path
import random,sys
import numpy as np
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import agnews_v3_common as c
from nested_sleep.lm import LlamaCMSLM,execution_state
from transformers import LlamaConfig,LlamaForCausalLM
from nested_sleep.optim import CMSController

def test_nonboundary_checkpoint_resumes_exact_updates_and_rng(tmp_path,monkeypatch):
 monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
 monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
 monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda states:None if states==[] else pytest.fail('CPU fixture got CUDA states'))
 def model():
  cfg=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128)
  cfg._attn_implementation='eager'
  return LlamaCMSLM(LlamaForCausalLM(cfg).eval(),rank=2,capacity=4,levels=2)
 c.engine.seed_all(17);m=model();ctl=CMSController(m,(4,16),.02,.04)
 pack={'records':[{'tokens':[1,2,3+i],'class_id':i}for i in range(4)],'code_maps':{'17':{'marker_token_id':11,'class_to_token_ids':[24,23,22,21]}},'tokenizer':{'pad_token_id':0}}
 meter=c.engine.WorkMeter()
 def update(m,ctl):c.engine.supervised_update(m,ctl,c.batch(pack,17,[0,1,2,3],'cpu'),0,meter,'fixture')
 for _ in range(5):update(m,ctl)
 assert ctl.counts==[1,5]
 path=tmp_path/'recovery.pt';c.checkpoint(path,m,ctl,17,5,{'recovery_row':{'seed':17,'arm':'cms','primary_updates':5}})
 for _ in range(11):update(m,ctl)
 expected=(c.fingerprint(m),c.engine.shared.tensor_tree_hash(ctl.state_dict()),execution_state(m),(random.random(),np.random.random(),torch.rand(3)))
 c.engine.seed_all(17);branch=model();state=torch.load(path,map_location='cpu',weights_only=False)
 c.restore(branch,state);branch_ctl=CMSController.restore(branch,state['controller']);c.engine.restore_rng(state['rng'])
 assert branch_ctl.counts==[1,5] and state['recovery_row']['primary_updates']==5
 for _ in range(11):update(branch,branch_ctl)
 actual=(c.fingerprint(branch),c.engine.shared.tensor_tree_hash(branch_ctl.state_dict()),execution_state(branch),(random.random(),np.random.random(),torch.rand(3)))
 assert expected[:3]==actual[:3] and expected[3][:2]==actual[3][:2] and torch.equal(expected[3][2],actual[3][2])
