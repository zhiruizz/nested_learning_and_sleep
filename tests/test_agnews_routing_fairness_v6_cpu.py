"""Small CPU-only checks for the v6 shared-backbone router driver."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools import run_agnews_routing_fairness_v6 as v6
from tools import probe_agnews_task_router_v5 as v5


class RoutingFairnessV6Tests(unittest.TestCase):
    def test_phase_fit_reproduces_v5_group_fit_without_class_argument(self):
        rng=np.random.default_rng(421)
        phases=["old"]*8+["new"]*8
        x=rng.normal(size=(16,12)).astype(np.float32)
        x[:8,:3]-=0.4;x[8:,:3]+=0.4
        actual=v6.fit_phase_ridge(x,phases)
        classes=np.asarray([0,1]*4+[2,3]*4,dtype=np.int64)
        reference=v5.fit_group_ridge(x,classes)
        self.assertEqual(actual["model_sha256"],reference["model_sha256"])
        self.assertEqual(actual["normalizer_sha256"],reference["normalizer_sha256"])
        with self.assertRaises(ValueError):
            v6.phase_targets([0,1,2,3])

    def test_permutation_is_fixed_and_preserves_group_balance(self):
        x=np.eye(8,dtype=np.float32)
        phases=["old"]*4+["new"]*4
        a,sha_a=v6.permuted_phase_fit(x,phases)
        b,sha_b=v6.permuted_phase_fit(x,phases)
        self.assertEqual(sha_a,sha_b)
        self.assertEqual(a["model_sha256"],b["model_sha256"])
        targets=v6.permuted_phase_targets(phases)
        self.assertEqual(set(targets.tolist()),{-1.0,1.0})
        self.assertEqual(int(np.sum(targets<0)),4)
        self.assertEqual(sha_a,hashlib.sha256(targets.astype("<f8").tobytes()).hexdigest())

    def test_zero_ridge_score_ties_to_old_like_live_helper(self):
        gate={"mean":np.zeros(3),"scale":np.ones(3),"weight":np.zeros(3),
              "bias":0.0,"feature_dim":3}
        mask=v6.ridge_mask(np.zeros((4,3),dtype=np.float64),gate)
        np.testing.assert_array_equal(mask,np.zeros(4,dtype=bool))

    def test_hash_random_depends_only_on_tokens_and_is_repeatable(self):
        rows=[{"tokens":[1,2,3],"phase":"old","class_id":0},
              {"tokens":[4,5,6],"phase":"new","class_id":2}]
        a=v6.hash_random_mask(rows)
        b=v6.hash_random_mask([{"tokens":r["tokens"]} for r in rows])
        np.testing.assert_array_equal(a,b)

    def test_full_vocab_two_token_metrics_use_all_vocab_and_gold_only_for_scoring(self):
        # Four rows, two-token labels [11,12,13,14], common marker 10.
        logits=torch.full((4,4,20),-5.0)
        labels=[11,12,13,14]
        rows=[];marker=torch.full((4,),10,dtype=torch.long);target=torch.tensor(labels)
        for i,label in enumerate(labels):
            logits[i,1,10]=5.0;logits[i,2,label]=5.0
            rows.append({"class_id":i,"phase":"old" if i<2 else "new"})
        got=v6.metric_rows_from_logits(logits,rows,torch.full((4,),2,dtype=torch.long),marker,target,labels)
        self.assertTrue(all(r[v6.c.EM]==1 for r in got))
        self.assertEqual(v6.aggregate_row_metrics(got)["all"]["examples"],4)
        self.assertEqual(v6.aggregate_row_metrics(got)["old"]["examples"],2)
        self.assertEqual(v6.aggregate_row_metrics(got)["new"]["examples"],2)

    def test_answer_nll_averages_python_floats_like_canonical_probe(self):
        generator=torch.Generator().manual_seed(0)
        logits=torch.randn((4,3,100),generator=generator)
        logits[:,2,:]=logits[:,2,:]*0.913+0.037
        labels=[11,12,13,14]
        rows=[{"class_id":i,"phase":"old" if i<2 else "new"}for i in range(4)]
        marker=torch.tensor([3,7,5,8]);target=torch.tensor(labels)
        lengths=torch.full((4,),2,dtype=torch.long)
        got=v6.metric_rows_from_logits(logits,rows,lengths,marker,target,labels)
        m= torch.nn.functional.cross_entropy(logits[torch.arange(4),1],marker,reduction="none")
        cl=torch.nn.functional.cross_entropy(logits[torch.arange(4),2],target,reduction="none")
        expected=[(float(a)+float(b))/2 for a,b in zip(m,cl)]
        old_fp32_average=[float((a+b)/2) for a,b in zip(m,cl)]
        self.assertTrue(any(a!=b for a,b in zip(expected,old_fp32_average)),
                        "fixture must distinguish Python-float and FP32 averaging")
        self.assertEqual([r["answer_nll_nats_per_token"]for r in got],expected)
        first=logits[torch.arange(4),1];second=logits[torch.arange(4),2]
        marker_ok=(first.argmax(-1)==marker).tolist();class_ok=(second.argmax(-1)==target).tolist()
        candidates=second[:,labels].argmax(-1).tolist()
        canonical=[]
        for i in range(4):
            ml,closs=float(m[i]),float(cl[i])
            canonical.append({"class_id":i,v6.c.EM:int(marker_ok[i]and class_ok[i]),
                "marker_accuracy":int(marker_ok[i]),"class_given_marker_accuracy":int(class_ok[i]),
                "candidate_accuracy":int(candidates[i]==i),"marker_nll":ml,"class_nll":closs,
                "answer_nll_nats_per_token":(ml+closs)/2})
        for actual,expected_row in zip(got,canonical):
            for key in canonical[0]: self.assertEqual(actual[key],expected_row[key])
        self.assertEqual(v6.aggregate_row_metrics(got),v5.aggregate_rows(canonical))

    def test_posthoc_selector_menu_is_eight_fixed_conditions(self):
        phases=["old","old","new","new"]
        records=[{"phase":p,"class_id":i,"tokens":[i,2]} for i,p in enumerate(phases)]
        features=np.asarray([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
        gate=v6.fit_phase_ridge(features,phases)
        perm,_=v6.permuted_phase_fit(features,phases)
        rows=[]
        for i,p in enumerate(phases):
            rows.append({"class_id":i,"phase":p,v6.c.EM:1,"marker_accuracy":1,
                "class_given_marker_accuracy":1,"candidate_accuracy":1,"marker_nll":0.1,
                "class_nll":0.2,"answer_nll_nats_per_token":0.15,
                "marker_logp":0.0,"selfjoint_logp":-0.4})
        post,masks=v6.posthoc_conditions(rows,rows,records,features,gate,perm)
        self.assertEqual(tuple(post),v6.CONDITIONS)
        self.assertEqual(len(masks["correct_phase_ridge"]),4)
        self.assertEqual(post["warm_alone"]["gate_fraction_new"],0.0)
        self.assertEqual(post["new_CMS_reference_alone"]["gate_fraction_new"],1.0)

    def test_metric_compare_rejects_greedy_or_nll_mismatch(self):
        metrics={"all":{"examples":4,v6.c.EM:1,"marker_nll":0.1,"class_nll":0.2,
                         "answer_nll_nats_per_token":0.15}}
        v6.compare_metrics(metrics,metrics)
        bad={"all":dict(metrics["all"],**{v6.c.EM:0})}
        with self.assertRaises(AssertionError):v6.compare_metrics(metrics,bad)

    def test_plan_hashes_seal_inside_continuation_directory(self):
        with tempfile.TemporaryDirectory(dir=v6.ROOT) as td:
            root=Path(td);continuation=root/"continuation";initial=root/"initial"
            continuation.mkdir();initial.mkdir()
            seal=continuation/"development-sealed.json";seal.write_text('{"complete":true}\n',encoding="utf-8")
            (initial/"result.json").write_text('{"complete":true}\n',encoding="utf-8")
            args=SimpleNamespace(v5_router_gate_sha256="gate-sha")
            plan=v6._plan(args,{"records_sha256":"records"},continuation=continuation,initial=initial)
            expected=hashlib.sha256(seal.read_bytes()).hexdigest()
            self.assertEqual(plan["continuation_seal_sha256"],expected)
            self.assertEqual(plan["phase"],"continuation")


if __name__=="__main__":
    unittest.main()
