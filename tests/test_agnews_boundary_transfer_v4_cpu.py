from pathlib import Path
import sys,copy
import pytest,torch
pytest.importorskip('transformers')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import qualify_agnews_boundary_transfer_v4 as w
from nested_sleep.lifecycle import SleepSession
from nested_sleep.lm import LlamaCMSLM
from transformers import LlamaConfig,LlamaForCausalLM

def test_old_transfer_gate_applies_drop_tolerance_to_each_class():
 em=w.c.EM;teacher={'old':{em:.9},'class0':{em:.8},'class1':{em:1.}}
 train={'old':{em:.9},'class0':{em:.9},'class1':{em:.9}}
 val={'old':{em:.85},'class0':{em:.75},'class1':{em:.95}}
 assert w.transfer_passes(teacher,train,val)
 val['class1'][em]=.949
 assert not w.transfer_passes(teacher,train,val)

def test_mutable_only_inflight_snapshot_restores_without_mutation(monkeypatch):
 monkeypatch.setattr(torch.cuda,'is_available',lambda:False);monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[]);monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda x:None)
 cfg=LlamaConfig(vocab_size=32,hidden_size=8,intermediate_size=12,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64);cfg._attn_implementation='eager'
 m=LlamaCMSLM(LlamaForCausalLM(cfg).eval(),rank=2,capacity=4,levels=2);m.eval().requires_grad_(False);m.banks()[0].activate()
 with torch.no_grad():m.banks()[0].slots[0].b.fill_(.02)
 params=dict(m.named_parameters());proposal={n:params[n].detach().clone()for n in m.level_descriptors()[0]['base']}
 s=SleepSession(m,0,proposal,seed=19,lr=.003,dream_tokens=2,share_backbone=True,reject_zero_gradient=True,receiver_mix_fraction=.5)
 prefix=torch.tensor([[1,2,3]]);s.seed_step(prefix,owner=None)
 saved={'session':w.compact_session(s),'rng':w.c.engine.capture_rng()}
 assert set(saved['session']['teacher'])==set(m.adapter_state_dict())
 assert not any('embed_tokens' in name for name in saved['session']['teacher'])
 before=w.c.engine.shared.tensor_tree_hash(saved)
 result=w.restore_probe(m,saved,prefix,w.c.engine.WorkMeter())
 assert result['two_independent_restores_identical']and result['saved_snapshot_not_mutated']
 assert w.c.engine.shared.tensor_tree_hash(saved)==before
