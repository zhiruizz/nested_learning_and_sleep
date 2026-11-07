"""Exercise actual host assets/pack/config startup, stopping at the model loader."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master-config", required=True)
    p.add_argument("--training-pack", required=True)
    p.add_argument("--project-root", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "tools/run_clinc_cil_v8_full.py"
    spec = importlib.util.spec_from_file_location("clinc_full_startup", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    original = m._read_inputs
    sentinel_calls = []

    class LoaderSentinel(Exception):
        pass

    def sentinel(*args, **kwargs):
        sentinel_calls.append(True)
        raise LoaderSentinel("actual startup reached the model loader; intentionally no model loaded")

    def read_then_patch(*args, **kwargs):
        values = original(*args, **kwargs)
        values[-1]._load_common = sentinel
        return values

    m._read_inputs = read_then_patch
    args = SimpleNamespace(**vars(a), mode="foundation", seed=17, arm=None,
                           foundation_dir=None, resume_checkpoint=None)
    try:
        m.run_training(args)
    except LoaderSentinel:
        if len(sentinel_calls) != 1:
            raise AssertionError("model loader sentinel count mismatch")
        failure = json.loads((Path(a.output_dir) / "failure.json").read_text())
        assert failure["phase"] == "model_load" and failure["work"] is None
        result = {"schema": "clinc-v8-full-host-startup-qualification-v1",
                  "passed": True, "real_model_assets_verified": True,
                  "real_training_pack_verified": True, "test_data_accessed": False,
                  "model_loaded": False, "forward_calls": 0, "optimizer_steps": 0,
                  "master_config_sha256": m.sha256_file(a.master_config),
                  "runner_sha256": m.sha256_file(path)}
        out = Path(a.output_dir) / "startup-qualification.json"
        out.write_bytes((json.dumps(result, indent=2) + "\n").encode())
        print(json.dumps(result))
    else:
        raise AssertionError("startup passed beyond model loader sentinel")


if __name__ == "__main__":
    main()
