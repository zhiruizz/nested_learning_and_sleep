import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"tools"))
import summarize_agnews_v7_probes as s

def metric(x): return {g:{"examples":512,"em":x} for g in ("old","new","all")}
def fixture():
    sp={"schema":"agnews-single-receiver-wake-stress-plan-v7","main_seal_sha256":"seal","steps":128,"evaluation_steps":[0,4,16,64,128]}
    stressrows=[]
    for seed in (17,29,43):
      for arm in ("split_receiver","split_shared"):
       for writes in (True,False):
        stressrows.append({"seed":seed,"source_arm":arm,"base_writes":writes,"status":"complete","updates":128,"slow_and_frozen_scope_guard_passed":True,"evaluations":[{"step":k,"validation":metric(.1*seed+(0.01 if writes else 0))} for k in sp["evaluation_steps"]],"fast_output_masked_validation":metric(.2),"full_validation":metric(.3)})
    stress={"schema":"agnews-single-receiver-wake-stress-v7","complete":True,"passed":True,"confirmation_scored":False,"qualification":False,"actual_CMS_ticks":1536,"plan_sha256":s.hash_obj(sp),"main_seal_sha256":"seal","rows":stressrows,"work":{}}
    dp={"schema":"agnews-single-receiver-diagnostic-plan-v7","main_seal_sha256":"seal","steps":[0,64,256,1024],"lesions":[[],[0],[1],[0,1]],"source_sha256":{k:"hash" for k in ("probe_agnews_single_receiver_v7.py","single_receiver_probes_v7.py","run_agnews_single_receiver_v7.py","agnews_tail_cache_v7.py","matmul_work_v7.py")}}
    arms=("current_receiver","current_shared","split_receiver","split_shared","split_narrow_receiver","split_narrow_shared","gold_receiver","gold_shared","new_only_receiver","new_only_shared")
    gr=[]
    for seed in (17,29,43):
      for arm in arms:
       for step in (0,64,256,1024):
         gr.append({"seed":seed,"arm":arm,"step":step,"unchanged_model_policy_rng":True,"reused_common_initial_probe":False,"gradient":{g:{"cosine":.5,"old_norm":2.,"new_norm":4.} for g in ("receiver","base0","base1")}})
    for i,r in enumerate(x for x in gr if x["step"]==0):
        r["reused_common_initial_probe"]=i%10!=0
    lr=[]
    for i in range(120):
      seed=(17,29,43)[i%3];arm=arms[(i//3)%10];lesion=([],[0],[1],[0,1])[(i//30)%4]
      lr.append({"seed":seed,"arm":arm,"disabled_slow_outputs":lesion,"model_policy_rng_unchanged":True,"validation":metric(.5),"semantic_position_route_audit":None if lesion else {"loss_positions":1024,"natural_old":500,"natural_new":524,"null_probability_sum":100.,"executed_old_coefficient_sum":200.,"executed_new_coefficient_sum":300.}})
    diag={"schema":"agnews-single-receiver-diagnostics-v7","complete":True,"confirmation_scored":False,"optimizer_updates":0,"independent_gradient_states":93,"plan_sha256":s.hash_obj(dp),"gradient_rows":gr,"lesion_rows":lr,"work":{}}
    return stress,sp,diag,dp

class AggregateTests(unittest.TestCase):
    def test_full_contract_and_sample_sd(self):
        st,sp,d,dp=fixture();s.validate(st,sp,d,dp);a=s.aggregate(st,d)
        self.assertEqual(a["counts"]["stress_ticks"],1536)
        self.assertEqual(a["counts"]["independent_gradient_states"],93)
        self.assertEqual(a["gradient_group_trajectory"][0]["cosine"]["n"],3)
        self.assertEqual(a["initial_common_gradient"][0]["cosine"]["n"],3)
        self.assertAlmostEqual(a["stress_validation_trajectory"][0]["old_em"]["sample_sd"],s.mean_sd([1.7,2.9,4.3])["sample_sd"],places=10)

    def test_fails_closed_on_missing_tick_or_confirmation(self):
        st,sp,d,dp=fixture();st["rows"][0]["updates"]=127
        with self.assertRaises(AssertionError):s.validate(st,sp,d,dp)
        st,sp,d,dp=fixture();d["confirmation_scored"]=True
        with self.assertRaises(AssertionError):s.validate(st,sp,d,dp)

if __name__=="__main__":unittest.main()
