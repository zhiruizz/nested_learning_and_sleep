from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / ".pytest-v12-analysis"
TMP_ROOT.mkdir(exist_ok=True)
SPEC = importlib.util.spec_from_file_location(
    "reanalyze_banking_optimizer_boundary_v12",
    ROOT / "tools" / "reanalyze_banking_optimizer_boundary_v12.py",
)
ANALYZER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(ANALYZER)


def _metrics_context(correct_offset: int):
    result = {}
    for family, offset in (("teacher_forced_sum", 0), ("teacher_forced_length_mean", 2)):
        all_correct = 110 + 2 * correct_offset + 2 * offset
        base_correct, extra = divmod(all_correct, 22)
        per_class = {str(i): {"count": 10 if i < 22 else 0,
                              "correct": (base_correct + (i < extra)) if i < 22 else 0,
                              "accuracy": ((base_correct + (i < extra)) / 10 if i < 22 else None)}
                     for i in range(77)}
        result[family] = {
            "old": {"count": 110, "correct": 50 + correct_offset + offset, "accuracy": (50 + correct_offset + offset) / 110, "invalid_predictions": 0},
            "current": {"count": 110, "correct": 60 + correct_offset + offset, "accuracy": (60 + correct_offset + offset) / 110, "invalid_predictions": 0},
            "all77": {"count": 220, "correct": all_correct, "accuracy": all_correct / 220, "invalid_predictions": 0},
            "per_class": per_class,
        }
    result["gold_label_token_mean_nll"] = {
        "old": {"count": 110, "mean_token_nll": 1.0 + correct_offset / 100},
        "current": {"count": 110, "mean_token_nll": 1.2 + correct_offset / 100},
        "all77": {"count": 220, "mean_token_nll": 1.1 + correct_offset / 100},
    }
    return result


def _run_receipt(seed: int):
    metric = _metrics_context(seed - 17)
    candidates, dup, greedy = {}, {}, {}
    for context in ANALYZER.CONTEXTS:
        candidates[context] = metric
        dup[context] = {"candidate_count_per_example": 77,
                        "gold_correct_sum": metric["teacher_forced_sum"]["all77"]["correct"],
                        "gold_correct_length_mean": metric["teacher_forced_length_mean"]["all77"]["correct"]}
        greedy[context] = {"count": 220, "correct": 100 + seed - 17, "accuracy": (100 + seed - 17)/220,
                           "invalid_predictions": 0, "candidate_constrained": False}
    runtime = {"schema": "banking-cil-v10-runtime-v1", "classes": 77,
               "candidate_sequences": 220*77, "candidate_metrics": candidates,
               "candidate_correct_counts": dup, "greedy_generation_metrics": greedy}
    arms = {}
    for name, spec in {"B1_HIGH": (1, 960), "B1_LOW": (1, 960), "B32_HIGH": (32, 960), "B32_LOW": (32, 3200)}.items():
        extra = {"full_backbone_forward_calls": 4, "tail_calls": 5} if seed == 17 and name == "B32_LOW" else {"full_backbone_forward_calls": 0, "tail_calls": 0}
        extra.update({"physical_adamw_updates": spec[1]//spec[0] + (1 if seed == 17 and name == "B32_LOW" else 0),
                      "update_attempts": spec[1]//spec[0] + (1 if seed == 17 and name == "B32_LOW" else 0)})
        arms[name] = {"requests": spec[1], "adamw_steps": spec[1]//spec[0], "effective_batch_size": spec[0],
                      "work_delta_including_restore_validation": extra}
    work = {k: 0 for k in ANALYZER.WORK_KEYS}
    work.update({"physical_adamw_updates": 2050 + (1 if seed == 17 else 0),
                 "update_attempts": 2050 + (1 if seed == 17 else 0), "wake_updates": 4,
                 "full_backbone_forward_calls": 155 if seed == 17 else 151, "tail_calls": 5 if seed == 17 else 0,
                 "active_wall_seconds": 2.0})
    qual = None
    if seed == 17:
        qual = {"status": "pass", "post_range_snapshot_state_equal": True,
                "original_request_range": [32,64], "restored_request_range": [32,64],
                "extra_request_evaluations": 32, "extra_physical_adamw_steps": 1,
                "original_metrics": {"loss_mean": .5, "token_count": 320, "preclip_grad_norm": .1, "parameter_update_l2": .02},
                "restored_metrics": {"loss_mean": .5, "token_count": 320, "preclip_grad_norm": .1, "parameter_update_l2": .02},
                "extra_work_delta": {"full_backbone_forward_calls": 4, "tail_calls": 5,
                                     "physical_adamw_updates": 1,"update_attempts":1}}
    return {"schema": ANALYZER.SCHEMA, "status": "complete", "seed": seed,
            "official_test_read": False, "test_data_read": False, "claim_scope": "three_seed_development_fixed_teacher_single_boundary_mechanism_diagnostic",
            "development": {"population": "first_task0_plus_task1_development_only",
                "counts": {"old":110,"current":110},"global_candidate_labels":77,"runtime":runtime,
                "work_delta": {"full_backbone_forward_calls": 100}},
            "fixed_teacher_rollouts": {"requests":3200,"gold_labels_used_as_loss_targets":False,
                "class_role_counts":{"old":1600,"current":1600},"generated_answer_tokens":1000},
            "boundary":{"prefix_wake_ticks":4,"due_levels":[0]},
            "optimizer_arms":arms,"restore_qualification":qual,"budget":work,
            "shared_boundary_work":{"full_backbone_forward_calls":1,"wake_updates":4},
            "fixed_rollout_work":{"full_backbone_forward_calls":50}}


def _write_batch_logs(run: Path, seed: int):
    rows=[]
    specs={"B1_HIGH":(1,960),"B1_LOW":(1,960),"B32_HIGH":(32,960),"B32_LOW":(32,3200)}
    for arm,(batch,total) in specs.items():
        cursor=0
        while cursor<total:
            if seed==17 and arm=="B32_LOW" and cursor==32:
                cursor=64
                continue
            end=cursor+batch
            rows.append({"seed":seed,"arm":arm,"request_start":cursor,"request_count":batch,
                         "requests_completed":end,"adamw_step":end//batch,"loss_mean":.5,
                         "token_count":320,"preclip_grad_norm":.1,"parameter_update_l2":.02,
                         "physical_adamw_steps":1})
            cursor=end
    (run/"optimizer-batches.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows),encoding="utf-8")


class BankingOptimizerBoundaryReanalysisTests(unittest.TestCase):
    def _fixture(self, root: Path):
        config=json.loads((ROOT/"configs"/"banking_v12_optimizer_boundary_frozen_b.json").read_text(encoding="utf-8"))
        config_path=root/"config.json"; config_path.write_text(json.dumps(config),encoding="utf-8")
        for seed in ANALYZER.SEEDS:
            run=root/f"seed{seed}";run.mkdir()
            result=_run_receipt(seed)
            rpath=run/"final-receipt.json"; raw=json.dumps(result,sort_keys=True).encode();rpath.write_bytes(raw)
            protocol={"schema":ANALYZER.PROTOCOL_SCHEMA,"status":"complete","seed":seed,
                      "final_receipt_sha256":__import__('hashlib').sha256(raw).hexdigest(),
                      "config_sha256":ANALYZER.sha256_file(config_path),
                      "v12_source_hashes":config["source_hashes"],
                      "training_manifest_sha256":config["training_manifest_sha256"],
                      "v10_source_lock_sha256":"a"*64,
                      "foundation_receipt_sha256":"b"*64,"foundation_checkpoint_sha256":"c"*64}
            (run/"protocol.json").write_text(json.dumps(protocol),encoding="utf-8")
            _write_batch_logs(run,seed)
        return config_path

    def test_three_seed_statistics_and_qualification_are_separate(self):
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            root=Path(tmp); cfg=self._fixture(root); out=root/"result"
            result=ANALYZER.analyze(root,cfg,out)
            summary=result["summary"]
            self.assertEqual(summary["cost_accounting"]["core_request_exposures"],18240)
            self.assertEqual(summary["cost_accounting"]["core_adamw_steps"],6150)
            self.assertEqual(summary["cost_accounting"]["additional_seed17_restore_qualification_requests"],32)
            self.assertEqual(summary["cost_accounting"]["additional_seed17_restore_qualification_adamw_steps"],1)
            seed17=summary["cost_accounting"]["seeds"][0]
            self.assertEqual(seed17["optimizer_batch_log_summary"]["seed17_original_gap_requests_recovered_from_qualification"],32)
            self.assertEqual(seed17["optimizer_batch_log_summary"]["per_arm"]["B32_LOW"]["logged_batch_count"],99)
            stat=summary["statistics"]["metrics_by_context_family_population"]["TEACHER|teacher_forced_sum|old"]["accuracy"]
            self.assertEqual(stat["n"],3)
            self.assertGreater(stat["sample_sd"],0)
            self.assertEqual(len(summary["statistics"]["paired_contrasts"]),8*(3+3+3+1))
            self.assertTrue(result["aggregate_path"].is_file())

    def test_missing_context_rejected(self):
        runtime=_run_receipt(17)["development"]["runtime"]
        del runtime["candidate_metrics"]["TEACHER"]
        with self.assertRaises(ANALYZER.AnalysisError):
            ANALYZER._validate_metric_context(runtime,"TEACHER")

    def test_unexpected_raw_batch_field_rejected(self):
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            p=Path(tmp)/"optimizer-batches.jsonl"
            row={"seed":17,"arm":"B1_HIGH","request_start":0,"request_count":1,"requests_completed":1,
                 "adamw_step":1,"loss_mean":.1,"token_count":10,"preclip_grad_norm":.1,
                 "parameter_update_l2":.1,"physical_adamw_steps":1,"row_id":"forbidden"}
            p.write_text(json.dumps(row)+"\n",encoding="utf-8")
            with self.assertRaises(ANALYZER.AnalysisError):
                ANALYZER._read_batch_log(p,17,{"B1_HIGH":{"batch":1,"requests":1},"B1_LOW":{"batch":1,"requests":1},
                    "B32_HIGH":{"batch":32,"requests":32},"B32_LOW":{"batch":32,"requests":32}},None)


if __name__ == "__main__":
    unittest.main()
