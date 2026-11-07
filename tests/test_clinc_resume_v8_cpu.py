"""CPU-only tests for strict task-0 checkpoint continuation in the V8 pilot."""
from __future__ import annotations

from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from tools import run_clinc_cil_v8_r2 as driver
from tools.clinc_lifecycle_v8 import WakeController


class _Slot(nn.Module):
    def __init__(self, active=False):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([0.2]))
        self.b = nn.Parameter(torch.tensor([0.0]))
        self.register_buffer("active", torch.tensor(active))
        self.register_buffer("priority", torch.tensor(1.0 if active else 0.0))
        self.register_buffer("generation", torch.tensor(1 if active else 0))


class _Bank(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([_Slot(True), _Slot(), _Slot()])
        self.route_override = None

    def activate(self, index=None):
        if index is None:
            index = next(i for i, slot in enumerate(self.slots) if not bool(slot.active))
        with torch.no_grad():
            self.slots[index].active.fill_(True)
            self.slots[index].priority.fill_(max(float(x.priority) for x in self.slots) + 1)
            self.slots[index].generation.add_(1)
        return index


class _Level(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([0.4]))
        self.b = nn.Parameter(torch.tensor([0.1]))
        self.experts = _Bank()


class _Context:
    def __init__(self):
        self.memory = nn.ModuleList([_Level() for _ in range(5)])
        self.readout_a = nn.Parameter(torch.tensor([0.7]))
        self.readout_b = nn.Parameter(torch.tensor([0.0]))
        self.readout_fp32_accumulation = True

    def parameters(self):
        yield from self.memory.parameters()
        yield self.readout_a
        yield self.readout_b

    def named_parameters(self):
        for name, parameter in self.memory.named_parameters():
            yield f"memory.{name}", parameter
        yield "readout_a", self.readout_a
        yield "readout_b", self.readout_b

    def banks(self):
        return [level.experts for level in self.memory]

    def state_dict(self):
        return {"memory": deepcopy(self.memory.state_dict()),
                "readout_a": self.readout_a.detach().clone(),
                "readout_b": self.readout_b.detach().clone(),
                "readout_fp32_accumulation": self.readout_fp32_accumulation,
                "routing": [deepcopy(bank.route_override) for bank in self.banks()],
                "trainable": {name: p.requires_grad for name, p in self.named_parameters()}}

    def load_state_dict(self, state):
        self.memory.load_state_dict(state["memory"])
        with torch.no_grad():
            self.readout_a.copy_(state["readout_a"])
            self.readout_b.copy_(state["readout_b"])
        self.readout_fp32_accumulation = state["readout_fp32_accumulation"]
        for bank, route in zip(self.banks(), state["routing"]):
            bank.route_override = deepcopy(route)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(state["trainable"][name])


def _resume_cfg(**overrides):
    cfg = {
        "seed": 17, "initialization_seed": 1907008, "tasks": [0, 1],
        "arms": ["HOPE"], "rank": 64, "capacity": 8, "levels": 5,
        "periods": [16, 64, 256, 1024, None], "epochs_per_task": 1,
        "examples_per_task": 1500, "batch_size": 8, "max_steps_per_task": 188,
        "base_lr": 0.001, "fast_lr": 0.1, "sleep_lr": 0.001,
        "sleep_weight_decay": 0.0, "sleep_clip_norm": 1.0,
        "sleep_prompt_policy": "seen_replay", "gkd_calls_per_event": 2,
        "lti_calls_per_event": 1, "max_new_tokens": 8,
        "lti_edit_distance_z0": 8, "lti_semantic_weight": 0.5,
        "lti_edit_weight": 0.5, "temperature": 0.8,
        "max_sequence_length": 128, "expected_normalized_train_dev_duplicates": 3,
        "task0_acquisition_gate": 0.5, "model_revision": "model@revision",
        "model_assets_sha256": {"model.safetensors": "a" * 64},
        "device": "cpu",
    }
    cfg.update(overrides)
    return cfg


def _rescore_fixture(cp_sha):
    source_map = {"r1.py": "6" * 64}
    return {
        "schema": "clinc-v8-checkpoint-rescore-v1", "status": "complete",
        "training_only": True,
        "selection": {"test_data_read": False},
        "r1_protocol_sha256": "b" * 64,
        "r1_source_lock_sha256": driver._canonical_sha(source_map),
        "r1_source_hashes_verified": source_map,
        "r2_source_sha256": "d" * 64,
        "evaluation_source_hashes": {"driver": "d" * 64},
        "training_manifest_sha256": "e" * 64,
        "manifest_source_sha256": "f" * 64,
        "model_revision": "model@revision",
        "model_assets_sha256": {"model.safetensors": "a" * 64},
        "arms": {"HOPE": {
            "checkpoint": {"sha256": cp_sha, "tasks_completed": 1},
            "task0_full_clean_development_seen15": {
                "metrics": {"current": {"accuracy": 0.72, "count": 300}}}}},
    }


def test_resume_contract_uses_corrected_seen_task0_rescore_and_rejects_science_changes():
    cfg = _resume_cfg()
    parent_cfg = dict(cfg, arms=["HOPE", "HOPE_SLEEP"], eval_full_development=True)
    protocol = {"schema": "clinc-cil-v8-pilot-protocol",
                "training_manifest_sha256": "e" * 64,
                "manifest_source_sha256": "f" * 64,
                "source_hashes": {"r1.py": "6" * 64},
                "source_lock_sha256": driver._canonical_sha({"r1.py": "6" * 64}),
                "model_revision": "model@revision",
                "model_assets_sha256": {"model.safetensors": "a" * 64},
                "config": parent_cfg}
    receipt = {"arm": "HOPE", "status": "stopped_task0_acquisition_gate",
               "tasks_completed": 1, "ticks": 188,
               "final_checkpoint_sha256": "1" * 64,
               "common_init_sha256": "9" * 64}
    q = _rescore_fixture("1" * 64)
    out = driver._validate_resume_contract(protocol, [receipt], q, cfg=cfg,
        current_r2_source_sha256="d" * 64,
        training_manifest_sha256="e" * 64, manifest_source_sha256="f" * 64,
        parent_protocol_sha256="b" * 64)
    assert out["HOPE"]["task0_clean_accuracy"] == 0.72
    bad_cfg = dict(cfg, fast_lr=0.2)
    with pytest.raises(ValueError, match="science parameters"):
        driver._validate_resume_contract(protocol, [receipt], q, cfg=bad_cfg,
            current_r2_source_sha256="d" * 64,
            training_manifest_sha256="e" * 64, manifest_source_sha256="f" * 64,
            parent_protocol_sha256="b" * 64)
    bad_q = _rescore_fixture("1" * 64)
    bad_q["arms"]["HOPE"]["task0_full_clean_development_seen15"]["metrics"]["current"]["accuracy"] = 0.49
    with pytest.raises(ValueError, match="gate failed"):
        driver._validate_resume_contract(protocol, [receipt], bad_q, cfg=cfg,
            current_r2_source_sha256="d" * 64,
            training_manifest_sha256="e" * 64, manifest_source_sha256="f" * 64,
            parent_protocol_sha256="b" * 64)


def test_resume_input_load_binds_receipt_and_rejects_checkpoint_path_escape(tmp_path):
    cfg = _resume_cfg()
    parent = tmp_path / "r1"
    checkpoint = parent / "checkpoints" / "HOPE" / "final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-checkpoint-bytes")
    cp_sha = driver._sha256(checkpoint)
    protocol = {"schema": "clinc-cil-v8-pilot-protocol",
                "training_manifest_sha256": "e" * 64,
                "manifest_source_sha256": "f" * 64,
                "source_hashes": {"r1.py": "6" * 64},
                "source_lock_sha256": driver._canonical_sha({"r1.py": "6" * 64}),
                "model_revision": "model@revision",
                "model_assets_sha256": {"model.safetensors": "a" * 64},
                "config": dict(cfg, arms=["HOPE", "HOPE_SLEEP"])}
    receipt = {"arm": "HOPE", "status": "stopped_task0_acquisition_gate",
               "tasks_completed": 1, "ticks": 188,
               "final_checkpoint": "checkpoints/HOPE/final.pt",
               "final_checkpoint_sha256": cp_sha, "common_init_sha256": "9" * 64}
    (parent / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (parent / "metrics.jsonl").write_text(
        json.dumps({"event": "run_complete", "receipt": receipt}) + "\n",
        encoding="utf-8")
    q = _rescore_fixture(cp_sha)
    q["r1_protocol_sha256"] = driver._sha256(parent / "protocol.json")
    q_path = tmp_path / "corrected-rescore.json"
    q_path.write_text(json.dumps(q), encoding="utf-8")
    result = driver._load_resume_inputs(parent, q_path, cfg=cfg,
        current_source_lock_sha256="c" * 64, current_r2_source_sha256="d" * 64,
        training_manifest_sha256="e" * 64, manifest_source_sha256="f" * 64)
    assert result["arms"]["HOPE"]["receipt"]["final_checkpoint_sha256"] == cp_sha
    assert result["arms"]["HOPE"]["checkpoint_path"] == str(checkpoint.resolve())
    assert result["r2_source_lock_sha256"] == "c" * 64

    receipt["final_checkpoint"] = "../../escape.pt"
    (parent / "metrics.jsonl").write_text(
        json.dumps({"event": "run_complete", "receipt": receipt}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="escapes parent run"):
        driver._load_resume_inputs(parent, q_path, cfg=cfg,
            current_source_lock_sha256="c" * 64, current_r2_source_sha256="d" * 64,
            training_manifest_sha256="e" * 64, manifest_source_sha256="f" * 64)


def test_restore_task0_context_pending_clocks_and_all_rngs(tmp_path):
    torch.manual_seed(271828)
    random.seed(314159)
    np.random.seed(161803)
    source = _Context()
    wake = WakeController(source, periods=(16, 64, 256, 1024, None), base_lr=0.001, fast_lr=0.1)
    wake.prepare_wake()
    params = dict(source.named_parameters())
    for _ in range(188):
        wake.prepare_wake()
        grads = {name: torch.full_like(parameter, 0.01)
                 for name, parameter in params.items() if parameter.requires_grad}
        wake.record_tick(grads, token_count=3)
        for level in sorted(wake.proposals()):
            wake.apply(level)
    assert wake.ticks == 188
    assert wake.clocks == {0: 12, 1: 60, 2: 188, 3: 188}

    state = {"schema": "clinc-cil-v8-final-state", "arm": "HOPE", "seed": 17,
             "context": source.state_dict(), "wake": wake.snapshot(),
             "rng": driver._rng_state(torch)}
    path = tmp_path / "task0-final.pt"
    torch.save(state, path)
    cp_sha = driver._sha256(path)
    receipt = {"arm": "HOPE", "seed": 17, "final_checkpoint_sha256": cp_sha}
    qualification = {"checkpoint": {"sha256": cp_sha}}

    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(1)
    restored = _Context()
    restored_wake = WakeController(restored, periods=(16, 64, 256, 1024, None),
                                   base_lr=0.001, fast_lr=0.1)
    restored_wake.prepare_wake()
    result = driver._restore_parent_task0_checkpoint(restored, restored_wake, path,
        receipt=receipt, qualification=qualification, cfg=_resume_cfg(), torch=torch)
    assert result["restored_ticks"] == 188
    assert result["restored_clocks"] == {0: 12, 1: 60, 2: 188, 3: 188}
    assert driver._mutable_context_sha(restored) == driver._mutable_context_sha(source)
    assert restored_wake.pending.keys() == wake.pending.keys()
    for key in wake.pending:
        assert torch.equal(restored_wake.pending[key], wake.pending[key])
    restored_wake.prepare_wake()
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
