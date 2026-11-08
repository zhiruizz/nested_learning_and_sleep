from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_banking_v12_fixed_teacher_fit as fit  # noqa: E402
import banking_cil_v10_lifecycle as lifecycle  # noqa: E402
import banking_cil_v10_routing as routing  # noqa: E402
import banking_cil_v12_batching as batching  # noqa: E402
from clinc_lifecycle_v8 import _state_hash  # noqa: E402
import clinc_tail_runtime_v8 as tail_runtime  # noqa: E402
from nested_sleep.experts import LowRankMemoryLevel  # noqa: E402


class TinyContext:
    """Small production-bank context exercising real V10/V12 snapshot code."""
    def __init__(self):
        self.memory = nn.ModuleList([
            LowRankMemoryLevel(dim=6, rank=3, capacity=4),
            LowRankMemoryLevel(dim=6, rank=3, capacity=4),
        ])
        self.readout_a = nn.Parameter(torch.randn(3, 6))
        self.readout_b = nn.Parameter(torch.zeros(9, 3))
        self.readout_fp32_accumulation = True
        self.banks()[0].activate(0)
        for bank in self.banks():
            for slot in bank.slots:
                slot.route = slot.route.to(torch.bfloat16)
                slot.priority = slot.priority.to(torch.bfloat16)

    def banks(self):
        return [level.experts for level in self.memory]

    def parameters(self):
        yield from self.memory.parameters()
        yield self.readout_a
        yield self.readout_b

    def named_parameters(self):
        yield from self.memory.named_parameters(prefix="memory")
        yield "readout_a", self.readout_a
        yield "readout_b", self.readout_b

    def state_dict(self):
        return {
            "memory": {k: v.detach().cpu().clone() for k, v in self.memory.state_dict().items()},
            "readout_a": self.readout_a.detach().cpu().clone(),
            "readout_b": self.readout_b.detach().cpu().clone(),
            "readout_fp32_accumulation": self.readout_fp32_accumulation,
            "routing": [deepcopy(bank.route_override) for bank in self.banks()],
            "trainable": {k: bool(v.requires_grad) for k, v in self.named_parameters()},
        }

    def load_state_dict(self, state, strict=True):
        self.memory.load_state_dict(state["memory"], strict=strict)
        with torch.no_grad():
            self.readout_a.copy_(state["readout_a"])
            self.readout_b.copy_(state["readout_b"])
        for bank, policy in zip(self.banks(), state["routing"]):
            bank.route_override = deepcopy(policy)
        self.readout_fp32_accumulation = bool(state["readout_fp32_accumulation"])
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(bool(state["trainable"][name]))


def make_tx(seed):
    torch.manual_seed(seed)
    live = TinyContext()
    routing.install_additive_routes(live)
    prospective = {
        "memory.0.a": live.memory[0].a.detach().clone() + 0.03,
        "memory.0.b": live.memory[0].b.detach().clone() - 0.02,
    }
    return batching.BatchableAdditiveSleepTransaction(
        live, 0, prospective, lr=0.001, weight_decay=0.0, clip_norm=1.0, seed=seed)


def test_plan_is_fixed_and_rejects_changed_cohort(tmp_path):
    plan = {
        "schema": fit.PLAN_SCHEMA,
        "status": "fixed_input_plan_before_development_scores_are_read",
        "training_config_sha256": "a" * 64,
        "seeds": [17, 29, 43],
        "trajectory_source": "matching_seed_private_fixed_teacher_training_rollouts",
        "trajectory_indices": {"start_inclusive": 0, "stop_exclusive": 64},
        "selection": "fixed_prefix_no_score_or_content_filter",
        "role_counts_from_frozen_alternation": {"old": 32, "current": 32},
        "objective": "teacher_to_student_full_vocabulary_KL_T1_answer_token_mean_per_request_then_request_mean",
        "contexts": [*fit.BASELINE_CONTEXTS, *(x[0] for x in fit.CHECKPOINT_CONTEXTS)],
        "optimizer_updates": 0, "generation_updates": 0,
        "new_generation": False, "official_test_accessed": False,
        "heldout_or_confirmatory_claim": False,
        "claim_scope": "supplementary_training_fit_diagnostic_on_identical_teacher_trajectories_not_heldout_accuracy",
        "max_full_backbone_forwards": 192, "max_tail_calls": 2112,
        "max_active_wall_seconds": 1800,
    }
    assert fit.validate_plan(plan, "a" * 64) == plan
    plan["trajectory_indices"]["stop_exclusive"] = 65
    with pytest.raises(ValueError, match="exactly the first 64"):
        fit.validate_plan(plan, "a" * 64)


def test_checked_in_plan_and_training_config_match_validator():
    plan_path = ROOT / "configs" / "banking_v12_fixed_teacher_fit_eval_plan.json"
    config_path = ROOT / "configs" / "banking_v12_optimizer_boundary_frozen_b.json"
    plan, plan_sha, config_sha = fit.verify_plan_config(plan_path, config_path)
    assert plan["schema"] == "banking-v12-fixed-teacher-fit-evaluation-plan-v1"
    assert len(plan_sha) == len(config_sha) == 64


def test_prefix_role_schedule_is_balanced_and_private():
    rows = [{"prompt_length": 2, "answer_tokens": 1} for _ in range(64)]
    assert fit.validate_prefix_roles(rows) == {"old": 32, "current": 32}
    with pytest.raises(ValueError, match="exactly 64"):
        fit.validate_prefix_roles(rows[:-1])


def test_fixed_fit_uses_same_prefix_and_request_mean_kl_for_every_context():
    class Context:
        def __init__(self, logits):
            self.logits = torch.tensor(logits, dtype=torch.float32)

    class TinyTail:
        @staticmethod
        def capture(_model, ids, _mask, device=None):
            return ids.to(device=device)

        @staticmethod
        def tail_logits(_model, context, cache):
            return context.logits.view(1, 1, -1).expand(cache.shape[0], cache.shape[1], -1)

        causal_token_kl = staticmethod(tail_runtime.causal_token_kl)

    context_names = [*fit.BASELINE_CONTEXTS, *(x[0] for x in fit.CHECKPOINT_CONTEXTS)]
    contexts = {name: Context([0.0, 1.0, -0.5]) for name in context_names}
    contexts["B1_HIGH_30"] = Context([1.0, -0.5, 0.0])
    trajectories = [{"input_ids": [11, 12, 13], "prompt_length": 2, "answer_tokens": 1}
                    for _ in range(64)]
    counters = fit._new_fit_counters()
    metrics = fit._score_seed(object(), contexts, trajectories, torch, torch.device("cpu"),
                              TinyTail, None, counters)
    expected, expected_count = tail_runtime.causal_token_kl(
        contexts["B1_HIGH_30"].logits.view(1, 1, -1).expand(1, 3, -1),
        contexts["TEACHER"].logits.view(1, 1, -1).expand(1, 3, -1),
        torch.tensor([[False, False, True]]), temperature=1.0)
    assert metrics["B1_HIGH_30"]["all"]["request_count"] == 64
    assert metrics["B1_HIGH_30"]["old"]["request_count"] == 32
    assert metrics["B1_HIGH_30"]["current"]["request_count"] == 32
    assert metrics["B1_HIGH_30"]["all"]["answer_tokens"] == 64
    assert metrics["B1_HIGH_30"]["all"]["mean_request_token_mean_kl"] == pytest.approx(float(expected))
    assert expected_count == 1
    assert metrics["_teacher_self_kl_diagnostic"]["passed"] is True
    assert counters["full_backbone_forwards"] == 64
    assert counters["tail_calls"] == 640
    assert counters["optimizer_updates"] == counters["generation_updates"] == 0


def test_real_snapshot_rebuilds_ten_contexts_and_receipt_hashes():
    seed = 17
    tx_seed = 120000 + seed * 100003 + 1 * 1009
    endpoint_specs = fit.checkpoint_specs()
    snapshots = {}
    first_tx = make_tx(tx_seed)
    for name, spec in endpoint_specs.items():
        tx = make_tx(tx_seed)
        snap = tx.snapshot()
        snap.payload.counters["adamw_steps"] = spec["adamw_steps"]
        # Make the saved trained receiver differ from its zero-update state,
        # while leaving the frozen-state hash scope intact.
        receiver_b = f"1.experts.slots.{snap.receiver_slot}.b"
        with torch.no_grad():
            snap.payload.student_state["memory"][receiver_b].fill_(
                0.01 + spec["requests"] / 100000.0)
        snapshots[name] = snap

    teacher_hash = _state_hash(first_tx.teacher)
    cms = deepcopy(first_tx.teacher)
    params_cms = dict(cms.named_parameters())
    params_proposal = dict(first_tx.student.named_parameters())
    with torch.no_grad():
        for pname in lifecycle._base_names(0):
            params_cms[pname].copy_(params_proposal[pname])
    wake = lifecycle.AdditiveWakeController(cms, periods=(16, None), base_lr=0.001, fast_lr=0.1)
    wake.prepare_wake()
    cms_hash = _state_hash(wake.context)
    reset_hash = _state_hash(first_tx.student)
    final_receipt = {
        "seed": seed,
        "initial_state_hashes": {
            "TEACHER": teacher_hash, "CMS_NO_RESET": cms_hash, "RESET_ZERO": reset_hash,
            "B1_HIGH": reset_hash, "B1_LOW": reset_hash,
            "B32_HIGH": reset_hash, "B32_LOW": reset_hash,
        },
        "optimizer_arms": {arm: {"effective_batch_size": size} for arm, size in
                            (("B1_HIGH", 1), ("B1_LOW", 1), ("B32_HIGH", 32), ("B32_LOW", 32))},
    }
    prototype = fit.cpu_prototype_from_snapshot(snapshots["B1_HIGH_30"], {"periods": (16, None)})
    assert _state_hash(prototype) == teacher_hash
    contexts, receipt = fit.reconstruct_context_set(
        prototype, snapshots, {
            "periods": (16, None), "base_lr": 0.001, "fast_lr": 0.1,
            "receiver_initialization_seed_offset": 120000, "task_id": 1,
            "sender": 0, "weight_decay": 0.0, "clip_norm": 1.0,
        }, final_receipt, lifecycle, routing, batching)
    assert list(contexts) == [*fit.BASELINE_CONTEXTS, *endpoint_specs]
    assert receipt["initial_state_hashes_verified"] is True
    assert _state_hash(contexts["TEACHER"]) == teacher_hash
    assert _state_hash(contexts["CMS_NO_RESET"]) == cms_hash
    assert _state_hash(contexts["RESET_ZERO"]) == reset_hash
    assert contexts["RESET_ZERO"] is not contexts["B32_LOW_960"]
    assert _state_hash(contexts["B32_LOW_960"]) == receipt["student_state_sha256"]["B32_LOW_960"]
    assert _state_hash(contexts["B32_LOW_960"]) != reset_hash
    assert all(_state_hash(contexts[name]) == receipt["student_state_sha256"][name]
               for name in endpoint_specs)


def test_run_failure_writes_terminal_aggregate_only_ledger_and_never_overwrites(tmp_path, monkeypatch):
    plan_path = ROOT / "configs" / "banking_v12_fixed_teacher_fit_eval_plan.json"
    v12_path = ROOT / "configs" / "banking_v12_optimizer_boundary_frozen_b.json"
    v10_path = ROOT / "configs" / "banking_v10_frozen_k10.json"
    output = tmp_path / "fresh-fit-output"
    args = type("Args", (), {"plan": plan_path, "v12_config": v12_path,
                              "v10_config": v10_path, "output_dir": output})()
    prepared = {
        "status": "preflight_pass",
        "snapshot_reconstruction": {"status": "cpu_snapshot_reconstruction_pass", "seeds": {}},
        "plan_sha256": fit.sha256_file(plan_path),
        "training_config_sha256": fit.sha256_file(v12_path),
        "input_manifest_v10_config_sha256": fit.sha256_file(v10_path),
        "evaluator_sha256": fit.sha256_file(fit.__file__),
        "input_manifest": [{"logical_path": "synthetic-input", "sha256": "a" * 64, "bytes": 1}],
        "seed_metadata": {17: {}, 29: {}, 43: {}},
    }

    def fail_body(_args, _prepared, output_dir, runtime_state):
        runtime_state["phase"] = "score_fixed_prefix"
        runtime_state["active_seed"] = 17
        runtime_state["active_request_ordinal"] = 5
        runtime_state["completed_request_ordinal"] = 4
        runtime_state["work_provider"] = lambda: {
            "full_backbone_forward_calls": 5, "full_backbone_forward_tokens": 109,
            "tail_calls": 41, "tail_backward_calls": 0, "tail_backward_tokens": 0,
            "physical_adamw_updates": 0, "wake_updates": 0, "sampled_output_tokens": 0,
            "active_wall_seconds": 2.5,
        }
        fit._write_completed_seed_aggregate(
            output_dir, 17,
            {"context_counts": {"TEACHER": {"all": {"request_count": 64}}}},
            {"final_receipt_sha256": "b" * 64})
        fit._append_progress(output_dir, runtime_state, "requests_started")
        raise RuntimeError("synthetic failure; do not persist example contents")

    monkeypatch.setattr(fit, "_run_fit_evaluation_body", fail_body)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        fit.run_fit_evaluation(args, prepared)
    seed_result_path = output / "seed-17-fixed-teacher-fit-aggregate.json"
    assert {p.name for p in output.iterdir()} == {
        "run-start.json", "progress.jsonl", "failure.json", seed_result_path.name}
    start = json.loads((output / "run-start.json").read_text(encoding="utf-8"))
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    seed_result = json.loads(seed_result_path.read_text(encoding="utf-8"))
    progress = [json.loads(line) for line in
                (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert start["status"] == "started"
    assert failure["status"] == "failed" and failure["terminal"] is True
    assert failure["phase"] == "score_fixed_prefix"
    assert failure["active_seed"] == 17 and failure["active_request_ordinal"] == 5
    assert failure["completed_request_ordinal"] == 4
    assert failure["best_known_spent_work"]["tail_calls"] == 41
    assert "synthetic failure" not in (output / "failure.json").read_text(encoding="utf-8")
    assert seed_result["status"] == "complete" and seed_result["seed"] == 17
    assert seed_result["aggregate"]["context_counts"]["TEACHER"]["all"]["request_count"] == 64
    assert progress[-1]["event"] == "failed"
    assert progress[-1]["work"]["full_backbone_forward_calls"] == 5
    failure_bytes = (output / "failure.json").read_bytes()
    with pytest.raises(FileExistsError, match="must be fresh"):
        fit.run_fit_evaluation(args, prepared)
    assert (output / "failure.json").read_bytes() == failure_bytes


def test_final_input_manifest_serializes_list_sections_explicitly():
    scientific_inputs = [{"logical_path": "seed17/protocol.json", "sha256": "a" * 64, "bytes": 12}]
    model_assets = [{"logical_path": "model-assets/config.json", "sha256": "b" * 64, "bytes": 8}]
    result = fit.build_input_manifest_document(scientific_inputs, model_assets, "c" * 64)
    assert result == {"scientific_inputs": scientific_inputs,
                      "model_assets": model_assets,
                      "v12_evaluator_sha256": "c" * 64}

