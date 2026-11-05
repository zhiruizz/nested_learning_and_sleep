"""Small CPU contracts for the separate AG News controlled-mix v2 runner."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

pytest.importorskip("transformers", reason="optional Llama CLI dependency")
from run_agnews_retention_v2 import validate_config


def test_v2_config_is_new_protocol_and_pins_receiver_mix():
    config = json.loads((ROOT / "configs" / "agnews-retention-v2.json").read_text(encoding="utf-8"))
    assert config["schema"] == "agnews-retention-main-config-v2"
    assert config["training"]["sleep_receiver_mix_fraction"] == 0.5
    assert config["training"]["sleep_boundaries_by_primary_step"] == [64, 128, 192, 256]
    assert config["outputs"]["private_output_root"] == "outputs/agnews-retention-v2/"
    assert validate_config(config)["schema"] == config["schema"]


@pytest.mark.parametrize("change", ["fraction", "seal_schema", "seed_order", "budget"])
def test_v2_runner_rejects_unregistered_protocol_mutations(change):
    config = json.loads((ROOT / "configs" / "agnews-retention-v2.json").read_text(encoding="utf-8"))
    if change == "fraction":
        config["training"]["sleep_receiver_mix_fraction"] = 0.25
    elif change == "seal_schema":
        config["schema"] = "agnews-retention-main-config-v1"
    elif change == "seed_order":
        config["training"]["seeds"] = [29, 17, 43]
    else:
        config["training"]["hard_wall_seconds"] = 7200
    with pytest.raises(ValueError):
        validate_config(config)


def test_route_audit_reuses_existing_sleep_logits_without_extra_forward():
    source = (ROOT / "tools" / "run_agnews_retention_v2.py").read_text(encoding="utf-8")
    assert "receiver_mix_fraction=SLEEP_RECEIVER_MIX_FRACTION" in source
    assert "receiver_bank.audit_routes = True" in source
    assert "natural_receiver_gkd_loss_position_hits" in source
    assert "natural_receiver_lti_loss_position_hits" in source
    assert "natural top-1 receiver hits at answer-loss positions from existing student logits; no extra forward" in source
    assert "--test" not in source
