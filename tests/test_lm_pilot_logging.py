import importlib.util
import builtins
from pathlib import Path


def test_json_console_emitter_serializes_before_flushing(monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    source = repository / "tools" / "run_wikitext2_lm_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_logging_test", source)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    calls = []
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: calls.append((args, kwargs)))

    pilot.emit_json_line({"seed": 17, "status": "completed"})

    assert calls == [(('{"seed": 17, "status": "completed"}',), {"flush": True})]


def test_expected_checkpoint_inventory_includes_initials_arms_and_fixture():
    repository = Path(__file__).resolve().parents[1]
    source = repository / "tools" / "run_wikitext2_lm_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_inventory_test", source)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    config = {"training": {"seeds": [17, 29],
                           "methods": ["frozen", "sequential_sft", "cms_wake", "cms_wake_ks_gkd_lti"]}}
    assert pilot.expected_checkpoint_count(config) == 11
