from pathlib import Path
import sys,pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import diagnose_agnews_sleep_v3 as d
from transformers import LlamaConfig,LlamaForCausalLM
from nested_sleep.lm import LlamaCMSLM,execution_state
from nested_sleep.optim import CMSController

class Clock:
 def check(self):pass

def test_observer_and_route_hooks_leave_original_sleep_result_exact(monkeypatch):
 monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
 monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
 monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda x:None)
 d.c.engine.seed_all(17)
 cfg=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)
 cfg._attn_implementation='eager';m=LlamaCMSLM(LlamaForCausalLM(cfg).eval(),rank=2,capacity=4,levels=2)
 pack={'records':[{'tokens':[1,2,3+i],'class_id':i}for i in range(4)],'code_maps':{'17':{'marker_token_id':11,'class_to_token_ids':[24,23,22,21]}},'tokenizer':{'pad_token_id':0}}
 rows=d.c.rows(pack,17);pool=[rows[i%4]for i in range(24)];meter=d.c.engine.WorkMeter()
 ctl=CMSController(m,(4,16),.02,.04)
 for _ in range(15):d.c.engine.supervised_update(m,ctl,d.c.batch(pack,17,[0,1,2,3],'cpu'),0,meter,'toy')
 event=d.c.engine.shared.step(m,ctl,rows,0,meter,'toy',defer_due=True)
 snapshot=d.c.engine.shared.compact_cpu({'model':m.adapter_state_dict(),'execution_state':execution_state(m),'rng':d.c.engine.capture_rng()})
 state=ctl.state_dict();original=d.c.engine.SleepSession
 # Tiny random teachers can produce numerical zero objectives; the production
 # guard stays on in the real replay. This fixture compares observation only.
 def factory(*a,**kw):kw['reject_zero_gradient']=False;return original(*a,**kw)
 monkeypatch.setattr(d.c.engine,'SleepSession',factory)
 # Exercise all GKD/LTI updates using fixed generated data so the fixture does
 # not fail merely because a random32-token toy teacher misses the marker.
 import nested_sleep.lm as lm
 def sample(self,prefix,count,**kw):return torch.cat([prefix,torch.full((len(prefix),count),11,dtype=torch.long)],-1),[]
 monkeypatch.setattr(lm.SamplingMixin,'sample',sample)
 plain,plainctl,_=d.c.engine.sleep_event(m,ctl,event['proposals'],pool,0,17,meter,Clock(),11)
 expected=(d.c.fingerprint(plain),d.c.engine.shared.tensor_tree_hash(plainctl.state_dict()),execution_state(plain),d.c.engine.shared.tensor_tree_hash(d.c.engine.capture_rng()))
 d.c.restore(m,snapshot);d.c.engine.restore_rng(snapshot['rng']);ctl=CMSController.restore(m,state)
 telemetry=d.BoundaryTelemetry(pack,17,rows,pool,meter,Clock())
 def observed(*a,**kw):kw['reject_zero_gradient']=False;return telemetry.factory(*a,**kw)
 monkeypatch.setattr(d.c.engine,'SleepSession',observed)
 observed_m,observed_ctl,_=d.c.engine.sleep_event(m,ctl,event['proposals'],pool,0,17,meter,Clock(),11)
 actual=(d.c.fingerprint(observed_m),d.c.engine.shared.tensor_tree_hash(observed_ctl.state_dict()),execution_state(observed_m),d.c.engine.shared.tensor_tree_hash(d.c.engine.capture_rng()))
 assert actual==expected and telemetry.updates==24 and len(telemetry.samples)==24
 assert len(telemetry.stages)==4
 assert all(x['observation_unchanged_weights_policy_rng']for x in telemetry.stages)
