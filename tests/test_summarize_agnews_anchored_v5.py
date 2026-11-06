"""Standard-library fixtures for the offline v5 seal analyzer."""
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import summarize_agnews_anchored_v5 as summary

def metrics(seed,arm,step,counts):
    index=summary.ARMS.index(arm);base=.80+index*.005+seed*index*.000001+step*.00001
    result={}
    for g in summary.GROUPS:
        n=counts[g]
        em=base-.3 if seed==29 and g in ("new","class2","class3") else base
        result[g]={"examples":n,"full_vocab_two_token_greedy_exact_match":em,
          "answer_nll_nats_per_token":2-base,"marker_nll":1-base/2,"class_nll":3-base,
          "marker_accuracy":base+.1,"class_given_marker_accuracy":base+.2,"candidate_accuracy":base+.3}
    return result

def fixture():
    runs=[];global_work={};counts={"all":512,"old":256,"new":256,"class0":128,"class1":128,"class2":128,"class3":128}
    train_counts={"all":1024,"old":512,"new":512,"class0":256,"class1":256,"class2":256,"class3":256}
    for seed in summary.SEEDS:
      for arm in summary.ARMS:
        sleep=arm in summary.SLEEP_ARMS;stage=f"run_{seed}_{arm}";work={stage:{"forward_calls":2,"forward_input_tokens":20,"gradient_forward_input_tokens":0,"zero_counter":0}}
        global_work[stage]=dict(work[stage])
        checkpoints=[];validation=[]
        for step in summary.STEPS:
            checkpoints.append({"primary_step":step})
            validation.append({"primary_step":step,"metrics":metrics(seed,arm,step,counts)})
        train_final=metrics(seed,arm,448,train_counts);val_final=validation[-1]["metrics"];gates={}
        for phase in ("old","new"):
            classes=(0,1) if phase=="old" else (2,3)
            train_em=train_final[phase][summary.METRICS[0]];val_em=val_final[phase][summary.METRICS[0]]
            class_em={str(k):val_final[f"class{k}"][summary.METRICS[0]] for k in classes}
            train_pass=train_em>=.8;val_pass=val_em>=.75;class_pass={k:v>=.7 for k,v in class_em.items()}
            gates[phase]={"train_phase_em":train_em,"train_phase_pass":train_pass,
              "validation_phase_em":val_em,"validation_phase_pass":val_pass,
              "validation_class_em":class_em,"validation_class_pass":class_pass,
              "pass":train_pass and val_pass and all(class_pass.values())}
        runs.append({"seed":seed,"arm":arm,"status":"complete","primary_updates":448,
          "sleep_Adam_updates":672 if sleep else 0,"events":[{} for _ in range(28 if sleep else 0)],
          "checkpoints":checkpoints,"validation":validation,"train_final":train_final,
          "qualification":{"gates":gates,"all_tasks_pass":gates["old"]["pass"] and gates["new"]["pass"]},
          "final_active_parameters":{"active_slots_by_bank":[0,29] if sleep else [0,1]},
          "seconds":12.0,"work":work})
    seal={"schema":"agnews-anchored-development-seal-v5","complete":True,"runs":runs,
      "protocol_sha256":"testprotocol","config_sha256":"testconfig","source_sha256":{"driver":"test"},
      "actual_primary_updates":9408,"actual_sleep_Adam_updates":8064,"actual_kd_old_aux_steps":4032,
      "actual_gold_old_aux_steps":1344,"work":global_work,"confirmation_scored":False,"no_holdout_input":True}
    protocol={"config_sha256":"testconfig","source_sha256":{"driver":"test"},
      "config":{"schema":"agnews-anchored-comparison-config-v5","arms":list(summary.ARMS),
        "evaluation_steps":list(summary.STEPS),"primary_steps":448,"capacity":32,"sleep_period":16,
        "review_status":"FROZEN_DEVELOPMENT_AFTER_REVIEW",
        "paired_contrasts_left_minus_right":[[a,b] for _,a,b in summary.CONTRASTS],
        "factorial_interactions":["(KDwarm-selfwarm)-(KDlatest-selflatest)","(KDlatest-selflatest)-(KD-reference)"],
        "confirmation_input":False,"confirmation_scored":False},
      "planned_primary_updates":9408,"planned_sleep_events":336,"planned_sleep_adam_updates":8064,
      "planned_kd_old_aux_steps":4032,"planned_gold_old_aux_steps":1344}
    return seal,protocol

class AnchoredSummaryTests(unittest.TestCase):
    def test_primary_sleep_work_union_gates_and_summary_shapes(self):
        seal,protocol=fixture();bykey,union=summary.validate_runs(seal,protocol)
        self.assertEqual(len(bykey),21);self.assertEqual(len(union),21)
        result=summary.analyze(seal,protocol,{"seal_sha256":"fixture"})
        self.assertEqual(result["validation_states"],126)
        self.assertEqual(result["cost"]["sleep_events"],336)
        self.assertEqual(result["cost"]["sleep_Adam_updates"],8064)
        self.assertEqual(result["task_gates"]["warm_anchorKD"]["new_pass_seeds"],2)
        self.assertEqual(result["task_gates"]["warm_anchorKD"]["both_pass_seeds"],2)
        self.assertIn("old_em_forgetting_step0_minus_step448",result["retention_and_acquisition_diagnostics"]["warm_anchorKD"])
        self.assertEqual(result["cost"]["work_residual"],{})

    def test_paired_contrast_sign_n3_sd_and_fixed_count(self):
        seal,protocol=fixture();bykey,_=summary.validate_runs(seal,protocol)
        vals={s:summary.score_at(bykey,s,"warm_anchorKD",448,"old",summary.METRICS[0])-
                 summary.score_at(bykey,s,"warm_CMS_reference",448,"old",summary.METRICS[0]) for s in summary.SEEDS}
        stats=summary.sample_stats(vals)
        self.assertEqual(stats["n"],3);self.assertGreater(stats["mean"],0)
        self.assertGreater(stats["sample_sd"],0)
        self.assertEqual(len(summary.CONTRASTS),11);self.assertEqual(len(summary.INTERACTIONS),2)

    def test_work_residue_is_not_hidden(self):
        seal,protocol=fixture();seal["work"]["unattributed"]={"forward_calls":1}
        with self.assertRaisesRegex(ValueError,"work counters"):
            summary.validate_runs(seal,protocol)

    def test_gate_components_are_recomputed_not_merely_copied(self):
        seal,protocol=fixture();seal["runs"][0]["qualification"]["gates"]["old"]["train_phase_pass"]=False
        with self.assertRaisesRegex(ValueError,"phase gate component"):
            summary.validate_runs(seal,protocol)

    def test_gate_must_match_raw_train_and_validation_metrics(self):
        seal,protocol=fixture();seal["runs"][0]["train_final"]["old"][summary.METRICS[0]]=.1
        with self.assertRaisesRegex(ValueError,"differs from raw metric"):
            summary.validate_runs(seal,protocol)

    def test_zero_work_counter_shape_is_preserved(self):
        seal,protocol=fixture();first=seal["runs"][0];stage=next(iter(first["work"]))
        del first["work"][stage]["zero_counter"]
        with self.assertRaisesRegex(ValueError,"work counters"):
            summary.validate_runs(seal,protocol)

    def test_protocol_hash_config_and_runner_source_validation(self):
        seal,protocol=fixture()
        d=Path(tempfile.mkdtemp(dir=ROOT,prefix="anchored_summary_fixture_"))
        try:
            cfg=d/"config.json";proto=d/"protocol.json";sp=d/"seal.json"
            runner=Path(summary.__file__).with_name("run_agnews_anchored_comparison_v5.py")
            protocol["source_sha256"].update({"driver":summary.sha(runner),
              "common":summary.sha(Path(summary.__file__).with_name("agnews_v3_common.py")),
              "v4_boundary":summary.sha(Path(summary.__file__).with_name("agnews_preupdate_sleep_v4.py")),
              "wake_helper":summary.sha(Path(summary.__file__).with_name("agnews_anchored_wake_v5.py")),
              "v4_engine":summary.sha(Path(summary.__file__).with_name("run_agnews_retention_v2.py"))})
            seal["source_sha256"]=dict(protocol["source_sha256"])
            cfg.write_text(json.dumps(protocol["config"],sort_keys=True),encoding="utf-8",newline="\n")
            h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            seal["config_sha256"]=protocol["config_sha256"]=h(cfg)
            proto.write_text(json.dumps(protocol,sort_keys=True),encoding="utf-8",newline="\n")
            seal["protocol_sha256"]=h(proto)
            sp.write_text(json.dumps(seal),encoding="utf-8")
            checked,loaded=summary.source_check(sp,seal)
            self.assertEqual(checked["raw_config_check"],"matched")
            self.assertEqual(checked["protocol_filename"],"protocol.json")
            self.assertNotIn("protocol_path",checked)
            loaded["config"]["paired_contrasts_left_minus_right"][0].reverse()
            proto.write_text(json.dumps(loaded,sort_keys=True),encoding="utf-8",newline="\n")
            seal["protocol_sha256"]=h(proto)
            with self.assertRaisesRegex(ValueError,"paired contrast arms/sign"):
                summary.source_check(sp,seal)
            loaded["config"]["paired_contrasts_left_minus_right"]=[[a,b] for _,a,b in summary.CONTRASTS]
            loaded["config"]["capacity"]=16
            proto.write_text(json.dumps(loaded,sort_keys=True),encoding="utf-8",newline="\n")
            seal["protocol_sha256"]=h(proto)
            with self.assertRaisesRegex(ValueError,"capacity mismatch"):
                summary.source_check(sp,seal)
        finally:
            for p in d.iterdir():p.unlink()
            d.rmdir()

    def test_json_output_uses_lf(self):
        d=Path(tempfile.mkdtemp(dir=ROOT,prefix="anchored_summary_lf_"))
        try:
            p=d/"analysis.json";summary.write_json(p,{"ok":True})
            raw=p.read_bytes();self.assertTrue(raw.endswith(b"\n"));self.assertNotIn(b"\r",raw)
        finally:
            for p in d.iterdir():p.unlink()
            d.rmdir()

    def test_markdown_tables_row_counts_and_lf(self):
        seal,protocol=fixture();analysis=summary.analyze(seal,protocol,{"seal_sha256":"fixture"})
        d=Path(tempfile.mkdtemp(dir=ROOT,prefix="anchored_summary_tables_"))
        try:
            summary.write_markdown_deliverables(analysis,d)
            expected={"endpoint-table.md":7,"trajectory-table.md":42,"per-seed-endpoints.md":21,"paired-endpoints.md":13}
            for name,count in expected.items():
                raw=(d/name).read_bytes();self.assertNotIn(b"\r",raw)
                table=[x for x in raw.decode("utf-8").splitlines() if x.startswith("|")]
                self.assertEqual(len(table)-2,count,name)
            self.assertTrue((d/"work-ledger.json").is_file())
            self.assertTrue((d/"work-ledger.md").read_bytes().endswith(b"\n"))
        finally:
            for p in d.iterdir():p.unlink()
            d.rmdir()

if __name__=="__main__":unittest.main()
