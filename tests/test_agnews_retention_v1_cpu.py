"""CPU-only contracts for the fixed AG News main training protocol."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def test_main_config_is_fixed_and_has_no_test_input_or_evaluator():
    config_path = ROOT / "configs" / "agnews-retention-v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["training"]["seeds"] == [17, 29, 43]
    assert config["training"]["arms"] == ["frozen", "sft", "cms", "cms_sleep", "cms_extra_replay"]
    assert config["training"]["sleep_boundaries_by_primary_step"] == [64, 128, 192, 256]
    assert config["training"]["cms_periods"] == [4, 16]
    assert config["training"]["extra_replay_batches_per_event"] == 12
    assert config["training"]["total_planned_wake_updates"] == 4752
    assert config["training"]["total_planned_sleep_adam_updates"] == 288
    assert config["data"]["official_test_in_runner"] is False
    assert config["outputs"]["seal_contains_test_scores"] is False
    assert "test_path" not in json.dumps(config).lower()


def test_runner_rejects_protocol_config_edits():
    from run_agnews_retention_v1 import validate_config
    config = json.loads((ROOT / "configs" / "agnews-retention-v1.json").read_text(encoding="utf-8"))
    validated = validate_config(config)
    assert validated["schema"] == config["schema"]
    mutated = json.loads(json.dumps(config))
    mutated["training"]["extra_replay_batches_per_event"] = 11
    with pytest.raises(ValueError, match="extra_replay_batches_per_event"):
        validate_config(mutated)


def test_real_deferred_sleep_branch_and_four_cycle_receiver_capacity():
    import torch
    import torch.nn.functional as F
    from nested_sleep.lm import NativeResearchLM, isolated_clone
    from nested_sleep.native import NativeConfig
    from nested_sleep.lifecycle import SleepSession
    from nested_sleep.optim import CMSController
    from nested_sleep.lifecycle import fingerprint
    from run_x870e_synthetic_dev_v2 import tensor_tree_hash

    torch.manual_seed(170043)
    model = NativeResearchLM(NativeConfig(vocab_size=32, dim=16, hidden=24, layers=1,
                                           cms_periods=(4, 16)), rank=2, capacity=8)
    controller = CMSController(model, periods=(4, 16), base_lr=.005, expert_lr=.01)
    prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
    sleep_events = 0
    for wake in range(1, 65):
        controller.prepare_wake(model)
        tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        logits = model.logits(tokens)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1))
        loss.backward()
        proposals = controller.collect(model)
        if wake % 16:
            expected = [0] if wake % 4 == 0 else []
            assert sorted(proposals) == expected
            for level in sorted(proposals, reverse=True):
                controller.apply(model, level, proposals[level])
            continue

        assert sorted(proposals) == [0, 1]
        assert controller.counts == [4, 16]
        live_fingerprint = fingerprint(model)
        live_controller_hash = tensor_tree_hash(controller.state_dict())
        candidate = isolated_clone(model)
        candidate_controller = CMSController.restore(candidate, controller.state_dict())
        candidate_controller.apply(candidate, 1, proposals[1])
        assert candidate_controller.counts == [4, 0]
        session = SleepSession(candidate, 0, proposals[0], seed=170043 + wake,
                               lr=.003, dream_tokens=2, reject_zero_gradient=True)
        assert fingerprint(session.teacher) == fingerprint(candidate)
        for _ in range(16):
            session.seed_step(prefix, owner=None)
        for _ in range(8):
            session.lti_step(prefix)
        assert len(session.records) == 24
        assert all(row["grad_norm"] > 0 for row in session.records)
        session.finish_consolidation()
        model = session.commit()
        candidate_controller.consume(0)
        controller = candidate_controller
        assert controller.counts == [0, 0]
        assert fingerprint(model) != live_fingerprint
        assert tensor_tree_hash(controller.state_dict()) != live_controller_hash
        sleep_events += 1
        active_receiver_slots = sum(bool(slot.active) for slot in model.banks()[1].slots)
        assert active_receiver_slots == sleep_events

    assert sleep_events == 4
    assert sum(bool(slot.active) for slot in model.banks()[1].slots) == 4
    assert len(model.banks()[1].slots) == 8
    assert controller.counts == [0, 0]


def test_driver_source_has_no_test_path_argument_or_network_download():
    source = (ROOT / "tools" / "run_agnews_retention_v1.py").read_text(encoding="utf-8")
    assert "official-test" in source.lower()  # explicit refusal/documentation only
    assert "--test" not in source
    assert "from_pretrained(args.model_dir, local_files_only=True" in source
    assert "requests.get" not in source and "snapshot_download" not in source
