"""CPU-only contract tests for v7 scheduling, losses and fail-closed scopes."""
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import run_agnews_single_receiver_v7 as v7

class V7Contracts(unittest.TestCase):
    def test_real_list_codebook_marker_and_class_positions(self):
        import torch
        pack={'code_maps':{'17':{'marker_token_id':9,'class_to_token_ids':[10,11,12,13]}}}
        rows=[{'class_id':c} for c in range(4)]
        self.assertEqual(v7.gold_answer_targets(pack,17,rows,[0,1,2,3]),[[9,10],[9,11],[9,12],[9,13]])
        logits=torch.full((4,2,15),-10.)
        logits[:,0,9]=10.
        for i in range(4):logits[i,1,10+i]=10.
        scores=v7.score_logits(logits,rows,pack,17)
        self.assertEqual(scores['all']['em'],1.)
        self.assertEqual(scores['old']['examples'],2)
        self.assertEqual(scores['new']['examples'],2)

    def rows(self):
        return [{"class_id":c,"role":"train"} for c in range(4) for _ in range(256)]

    def test_fixed_arms_and_schedule_determinism(self):
        self.assertEqual(len(v7.ARMS),10)
        rows=self.rows()
        a=v7.class_schedule(rows,17,"split_narrow_receiver",steps=9)
        self.assertEqual(a,v7.class_schedule(rows,17,"split_narrow_receiver",steps=9))
        for batch in a:
            self.assertEqual([rows[i]["class_id"] for i in batch],list(v7.BATCH_CLASSES))
        no=v7.class_schedule(rows,17,"new_only_receiver",steps=4)
        for batch in no:
            self.assertEqual([rows[i]["class_id"] for i in batch],[2,2,2,2,3,3,3,3])
        split=v7.class_schedule(rows,17,"split_receiver",steps=4)
        no_shared=v7.class_schedule(rows,17,"new_only_shared",steps=4)
        self.assertEqual(no,no_shared)
        narrow=v7.class_schedule(rows,17,"split_narrow_receiver",steps=4)
        self.assertEqual([split[s][4:] for s in range(4)],[narrow[s][4:] for s in range(4)])
        self.assertEqual(v7.auxiliary_classes("new_only_shared"),(2,3))

    def test_scope_rejects_old_slot_head_and_shared_base_in_receiver(self):
        self.assertTrue(v7.validate_arm_scope("current_receiver",["backbone.model.layers.31.mlp.memory.1.experts.slots.1.a"]))
        self.assertTrue(v7.validate_arm_scope("current_shared",["backbone.model.layers.31.mlp.memory.1.experts.slots.1.a","backbone.model.layers.31.mlp.memory.0.a"]))
        for bad in ("backbone.model.layers.31.mlp.memory.0.slots.0.a","backbone.lm_head.weight","backbone.model.layers.1.weight","backbone.model.layers.31.mlp.memory.0.a"):
            with self.assertRaises(ValueError): v7.validate_arm_scope("current_receiver",[bad])

    def test_interleaved_roles_cannot_enter_schedule(self):
        rows=[]
        for i in range(128):
            rows.append({"class_id":i%4,"role":"train"})
            rows.append({"class_id":i%4,"role":"validation"})
        train=[r for r in rows if r["role"]=="train"]
        sch=v7.class_schedule(train,29,"split_receiver",steps=3)
        for batch in sch: self.assertTrue(all(train[i]["role"]=="train" for i in batch))
        with self.assertRaises(ValueError): v7.class_schedule(rows,29,"split_receiver",steps=1)

    def test_gold_targets_are_marker_then_class(self):
        pack={"code_maps":{"17":{"marker_token_id":9,"class_to_token_ids":{"0":20,"1":21}}}}
        rows=[{"class_id":0},{"class_id":1}]
        self.assertEqual(v7.gold_answer_targets(pack,17,rows,[0,1]),[[9,20],[9,21]])

    def test_common_context_routes_new_slow_slot_and_preserves_old_slot(self):
        class Slot:
            def __init__(self,active): self.active=active
            def requires_grad_(self,value): return self
        class Bank:
            def __init__(self,active): self.slots=[Slot(active),Slot(False)];self.route_override=None
            def reset_active(self):
                for s in self.slots:s.active=False
            def activate(self,i):
                if self.slots[i].active: raise ValueError()
                self.slots[i].active=True
        class Model:
            def __init__(self):self.bs=[Bank(True),Bank(True)]
            def banks(self):return self.bs
            def eval(self):return self
        m=Model();v7.reset_student_context(m)
        self.assertFalse(m.bs[0].slots[0].active)
        self.assertTrue(m.bs[1].slots[0].active)
        self.assertTrue(m.bs[1].slots[1].active)
        self.assertEqual(m.bs[1].route_override,{"mix":1,"fraction":0.5})

    def test_restore_requires_all_exact_hashes(self):
        s={"model_sha256":"a","adam_sha256":"b","rng_sha256":"c"}
        self.assertTrue(v7.validate_restore_contract(s,dict(s),dict(s)))
        wrong=dict(s); wrong["adam_sha256"]="x"
        self.assertFalse(v7.validate_restore_contract(s,wrong,dict(s)))

    def test_kl_and_ce_shape_contracts(self):
        import torch
        logits=torch.zeros(3,2,7,requires_grad=True)
        target=torch.tensor([[0,1],[2,3],[4,5]])
        ce=v7.answer_ce(logits,target)
        self.assertAlmostEqual(float(ce.detach()),torch.log(torch.tensor(7.)).item(),places=5)
        kl=v7.distill_kl(logits.detach(),logits.detach())
        self.assertAlmostEqual(float(kl),0.,places=6)

if __name__=="__main__": unittest.main()
