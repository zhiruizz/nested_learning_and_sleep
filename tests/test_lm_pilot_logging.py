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
