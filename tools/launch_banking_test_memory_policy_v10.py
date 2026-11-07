"""Source-bound, process-local CUDA memory launcher for the Banking v10 test evaluator.

Unlike the general V10 launcher, this wrapper binds ``--training-root`` to
the sealed main-run directory, independently of the train/dev pack location.
It runs the frozen evaluator's complete CPU preflight before registering the
deferred CUDA memory-fraction callback. It never reads the test CSV itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping


SCHEMA = "banking-v10-test-memory-launch-v1"
ALLOCATOR_CONF = "garbage_collection_threshold:0.7"
MEMORY_FRACTION = 0.8
TORCH_VERSION = "2.8.0+cu128"
EVALUATOR_KEY = "evaluate_banking_test_v10.py"
TEST_SOURCE_REVISION = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
TEST_SOURCE_BYTES = 239_961
TEST_SOURCE_SHA256 = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
TRAIN_SOURCE_BYTES = 839_073
TRAIN_SOURCE_SHA256 = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
CONTRACT_KEYS = {
    "schema", "launcher_sha256", "project_root", "task_root", "model_dir",
    "training_pack", "training_pack_sha256", "config_path", "config_file_sha256", "config_sha256",
    "source_lock_sha256", "source_hashes", "training_manifest_path",
    "training_manifest_sha256", "training_root", "seal_path", "seal_sha256",
    "main_release_path", "main_release_sha256", "foundation_root",
    "evaluation_contract_path", "evaluation_contract_sha256", "evaluator_sha256",
    "contract_builder_sha256", "publisher_train_csv_path", "publisher_train_csv_sha256",
    "publisher_train_csv_bytes", "test_csv_path", "test_csv_expected_sha256",
    "test_csv_expected_bytes", "test_csv_revision", "preflight_output_dir",
    "output_dir", "launch_record_path", "device", "allocator_conf",
    "memory_fraction", "torch_version",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_contract_shape(contract: Mapping, *, launcher_sha256: str) -> None:
    if set(contract) != CONTRACT_KEYS or contract.get("schema") != SCHEMA:
        raise ValueError("memory-launch contract schema/fields mismatch")
    if contract.get("launcher_sha256") != launcher_sha256:
        raise ValueError("memory launcher source hash mismatch")
    if contract.get("allocator_conf") != ALLOCATOR_CONF or contract.get("memory_fraction") != MEMORY_FRACTION:
        raise ValueError("memory policy differs from fixed process-only policy")
    if contract.get("torch_version") != TORCH_VERSION:
        raise ValueError("memory launcher is pinned to an unsupported PyTorch version")
    if contract.get("device") not in ("cuda:0", "cuda:1"):
        raise ValueError("memory launcher requires an explicit supported CUDA device")
    for key in ("config_file_sha256", "config_sha256", "source_lock_sha256",
                "training_pack_sha256", "training_manifest_sha256", "seal_sha256", "main_release_sha256",
                "evaluation_contract_sha256", "evaluator_sha256", "contract_builder_sha256"):
        if not _hex64(contract.get(key)):
            raise ValueError(f"contract {key} must be a SHA-256 hex digest")
    source_hashes = contract.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or len(source_hashes) != 24:
        raise ValueError("contract must bind the exact 24-file frozen evaluator source graph")
    if any(not isinstance(k, str) or not _hex64(v) for k, v in source_hashes.items()):
        raise ValueError("malformed source-hash graph")
    if (contract.get("publisher_train_csv_sha256") != TRAIN_SOURCE_SHA256 or
            contract.get("publisher_train_csv_bytes") != TRAIN_SOURCE_BYTES):
        raise ValueError("publisher train CSV must match the verified fixed source pin")
    if (contract.get("test_csv_expected_sha256") != TEST_SOURCE_SHA256 or
            contract.get("test_csv_expected_bytes") != TEST_SOURCE_BYTES or
            contract.get("test_csv_revision") != TEST_SOURCE_REVISION):
        raise ValueError("test CSV path must bind the fixed publisher revision/byte/hash pin")


def evaluator_argv(contract: Mapping, *, mode: str) -> list[str]:
    if mode not in ("preflight", "evaluate"):
        raise ValueError("unsupported evaluator mode")
    args = [
        "--mode", mode,
        "--project-root", str(Path(contract["project_root"]).resolve()),
        "--model-dir", str(Path(contract["model_dir"]).resolve()),
        "--config", str(Path(contract["config_path"]).resolve()),
        "--training-manifest", str(Path(contract["training_manifest_path"]).resolve()),
        # The main run root is intentionally independent of training_pack.parent.
        "--training-root", str(Path(contract["training_root"]).resolve()),
        "--seal", str(Path(contract["seal_path"]).resolve()),
        "--main-release-artifact", str(Path(contract["main_release_path"]).resolve()),
        "--foundation-root", str(Path(contract["foundation_root"]).resolve()),
        "--evaluation-contract", str(Path(contract["evaluation_contract_path"]).resolve()),
        "--output-dir", str(Path(contract["preflight_output_dir"] if mode == "preflight"
                                  else contract["output_dir"]).resolve()),
        "--device", str(contract["device"]),
    ]
    if mode == "evaluate":
        args.extend(("--test-csv", str(Path(contract["test_csv_path"]).resolve())))
    return args


def validate_test_csv_location(publisher_train_csv: str | Path, test_csv: str | Path) -> None:
    train = Path(publisher_train_csv).expanduser().resolve()
    test = Path(test_csv).expanduser().resolve()
    if (train.name != "train.csv" or train.parent.name != "raw" or
            train.parent.parent.name != "banking77-original-9d08145"):
        raise ValueError("publisher training CSV is outside the pinned original raw cache")
    if test != train.parent / "test.csv":
        raise ValueError("test CSV must be the fixed raw/test.csv sibling of the verified publisher train CSV")


def _install_memory_policy(torch_module, device: str) -> dict:
    if torch_module.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before the memory policy was queued")
    state = {"registered": False, "called": False, "error": None}

    def set_fraction_on_init():
        state["called"] = True
        try:
            torch_module.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device=device)
        except Exception as exc:  # surfaced in terminal launch record
            state["error"] = f"{type(exc).__name__}: {exc}"
            raise

    torch_module.cuda._lazy_call(set_fraction_on_init)
    state["registered"] = True
    if torch_module.cuda.is_initialized():
        raise RuntimeError("registering the memory callback unexpectedly initialized CUDA")
    return state


def _atomic_json(path: Path, obj: Mapping, *, replace=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name(path.name + ".partial")
    if temp.exists():
        raise FileExistsError(temp)
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if sha256_file(path) != hashlib.sha256(raw.encode("utf-8")).hexdigest():
        raise IOError("launch record readback hash mismatch")


def _validate_eval_contract_binding(evaluator, eval_contract: Mapping, cfg: Mapping,
                                    contract: Mapping) -> None:
    # driver.validate_config returns no _logical_sha256 field. The frozen
    # evaluator's own main() adds that value before calling this API.
    evaluator_cfg = {**cfg, "_logical_sha256": contract["config_sha256"]}
    evaluator.validate_evaluation_contract(
        eval_contract, evaluator_sha256=contract["evaluator_sha256"], config=evaluator_cfg,
        config_file_sha256=contract["config_file_sha256"],
        source_lock_sha256=contract["source_lock_sha256"],
        training_manifest_sha256=contract["training_manifest_sha256"],
        training_seal_sha256=contract["seal_sha256"],
        main_release_sha256=contract["main_release_sha256"],
        foundation_refs=eval_contract.get("foundation_refs"))


def _expected_context_tuples(seal: Mapping) -> set[tuple]:
    runs = seal.get("runs")
    if not isinstance(runs, list) or len(runs) != 9:
        raise ValueError("sealed grid must enumerate exactly nine final contexts")
    return {(row.get("seed"), row.get("arm"), row.get("final_checkpoint_sha256"),
             row.get("run_receipt_sha256")) for row in runs}


def _context_tuples(rows) -> set[tuple]:
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("result must enumerate exactly nine verified contexts")
    return {(row.get("seed"), row.get("arm"), row.get("checkpoint_sha256"),
             row.get("receipt_sha256")) for row in rows}


def _validate_preflight_result(preflight: Mapping, contract: Mapping, seal: Mapping,
                               evaluator, foundation_audit: Mapping) -> None:
    expected = {"schema": evaluator.SCHEMA, "status": "pass", "mode": "preflight",
                "training_config_sha256": contract["config_sha256"],
                "source_lock_sha256": contract["source_lock_sha256"],
                "evaluation_contract_sha256": contract["evaluation_contract_sha256"],
                "evaluator_sha256": contract["evaluator_sha256"],
                "training_manifest_sha256": contract["training_manifest_sha256"],
                "seal_sha256": contract["seal_sha256"],
                "main_release_sha256": contract["main_release_sha256"],
                "test_payload_read": False, "model_loaded": False}
    if any(preflight.get(key) != value for key, value in expected.items()):
        raise RuntimeError("official evaluator CPU preflight source/safety gates failed")
    if preflight.get("foundation_audit") != foundation_audit:
        raise RuntimeError("CPU preflight foundation audit mismatch")
    audit = preflight.get("checkpoint_audit")
    if not isinstance(audit, Mapping) or (audit.get("count"), audit.get("cpu_deserialized"),
            audit.get("model_loaded"), audit.get("cuda_initialized"), audit.get("optimizer_updates")) != (
                9, True, False, False, 0):
        raise RuntimeError("CPU preflight did not verify all nine checkpoints without model/CUDA/updates")
    if _context_tuples(audit.get("verified_runs")) != _expected_context_tuples(seal):
        raise RuntimeError("CPU preflight checkpoint/receipt set differs from sealed nine-run grid")


def _validate_terminal_result(result: Mapping, contract: Mapping, seal: Mapping,
                              preflight: Mapping, evaluator) -> None:
    if result.get("schema") != evaluator.SCHEMA or result.get("status") != "complete":
        raise RuntimeError("official evaluator result is not terminal/complete")
    required_flags = {"optimizer_updates": 0, "all_contexts_frozen": True,
                      "backbone_unchanged": True,
                      "raw_prompts_predictions_and_token_ids_saved": False}
    if any(result.get(key) != value for key, value in required_flags.items()):
        raise RuntimeError("official evaluator terminal safety flags are invalid")
    training = result.get("training")
    if not isinstance(training, Mapping):
        raise RuntimeError("terminal result lacks training/source bindings")
    expected_training = {
        "config_sha256": contract["config_sha256"],
        "source_lock_sha256": contract["source_lock_sha256"],
        "training_manifest_sha256": contract["training_manifest_sha256"],
        "seal_sha256": contract["seal_sha256"],
        "evaluation_contract_sha256": contract["evaluation_contract_sha256"],
        "evaluator_sha256": contract["evaluator_sha256"],
        "main_release_sha256": contract["main_release_sha256"],
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise RuntimeError("terminal result source/config/seal/release bindings differ from launch contract")
    if training.get("foundation_audit") != preflight.get("foundation_audit"):
        raise RuntimeError("terminal foundation audit differs from CPU-preflight result")
    if _context_tuples(training.get("verified_contexts")) != _expected_context_tuples(seal):
        raise RuntimeError("terminal contexts differ from the exact sealed nine-run grid")
    test = result.get("test_source")
    expected_test = {"source_revision": TEST_SOURCE_REVISION, "source_sha256": TEST_SOURCE_SHA256,
                     "source_bytes": TEST_SOURCE_BYTES, "raw_rows": 3080,
                     "classes": 77, "rows_per_class": 40}
    if not isinstance(test, Mapping) or any(test.get(key) != value for key, value in expected_test.items()):
        raise RuntimeError("terminal result does not match the fixed official test source/counts")


def _check_inputs(contract: Mapping, *, contract_path: Path, contract_sha256: str,
                  launcher_path: Path) -> dict:
    validate_contract_shape(contract, launcher_sha256=sha256_file(launcher_path))
    if sha256_file(contract_path) != contract_sha256:
        raise ValueError("launcher contract bytes changed")
    path_keys = ("project_root", "task_root", "model_dir", "training_pack", "publisher_train_csv_path", "config_path",
                 "training_manifest_path", "training_root", "seal_path", "main_release_path",
                 "foundation_root", "evaluation_contract_path")
    paths = {key: Path(contract[key]).expanduser().resolve(strict=True) for key in path_keys}
    if not paths["project_root"].is_dir() or not paths["task_root"].is_dir() or not paths["model_dir"].is_dir():
        raise FileNotFoundError("project, task, or model directory is unavailable")
    if (not paths["training_pack"].is_file() or not paths["config_path"].is_file() or
            not paths["publisher_train_csv_path"].is_file()):
        raise FileNotFoundError("pinned training pack or config is unavailable")
    if not paths["training_root"].is_dir() or not paths["foundation_root"].is_dir():
        raise FileNotFoundError("sealed main-run or foundation root is unavailable")
    if sha256_file(paths["training_pack"]) != contract["training_pack_sha256"]:
        raise ValueError("training pack file bytes differ from contract")
    publisher_train = paths["publisher_train_csv_path"]
    if (publisher_train.name != "train.csv" or publisher_train.parent.name != "raw" or
            publisher_train.parent.parent.name != "banking77-original-9d08145" or
            publisher_train.stat().st_size != TRAIN_SOURCE_BYTES or
            sha256_file(publisher_train) != TRAIN_SOURCE_SHA256):
        raise ValueError("publisher train CSV path/bytes differ from verified source pin")
    for field, path in (("config_file_sha256", paths["config_path"]),
                        ("training_manifest_sha256", paths["training_manifest_path"]),
                        ("seal_sha256", paths["seal_path"]),
                        ("main_release_sha256", paths["main_release_path"]),
                        ("evaluation_contract_sha256", paths["evaluation_contract_path"])):
        if sha256_file(path) != contract[field]:
            raise ValueError(f"bound artifact bytes differ: {field}")
    evaluator_path = paths["task_root"] / "tools" / EVALUATOR_KEY
    builder_path = paths["task_root"] / "tools" / "build_banking_test_contract_v10.py"
    if sha256_file(evaluator_path) != contract["evaluator_sha256"]:
        raise ValueError("official evaluator bytes differ from source-bound contract")
    if sha256_file(builder_path) != contract["contract_builder_sha256"]:
        raise ValueError("evaluation contract builder bytes differ from source-bound contract")
    if str(paths["training_root"]) == str(paths["training_pack"].parent):
        raise ValueError("training_root must identify the sealed main-output root, not the data-pack directory")

    output = Path(contract["output_dir"]).expanduser().resolve()
    preflight = Path(contract["preflight_output_dir"]).expanduser().resolve()
    record = Path(contract["launch_record_path"]).expanduser().resolve()
    test_csv = Path(contract["test_csv_path"]).expanduser().resolve()
    validate_test_csv_location(publisher_train, test_csv)
    targets = (output, preflight, record)
    for index, left in enumerate(targets):
        for right in targets[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("preflight, evaluation output, and launch record must be independent")
    if output.exists() or preflight.exists() or record.exists():
        raise FileExistsError("preflight, evaluation, and launcher-record targets must be fresh")
    protected = (paths["project_root"], paths["model_dir"], paths["training_pack"].parent,
                 paths["training_root"], paths["foundation_root"], test_csv.parent)
    for target in (output, preflight, record):
        if any(target == item or item in target.parents for item in protected):
            raise ValueError("outputs/record must be outside protected source/model/data/run trees")

    cfg = _read_json(paths["config_path"])
    if cfg.get("source_hashes") != contract["source_hashes"]:
        raise ValueError("launcher source graph differs from frozen config")
    return {**paths, "output_dir": output, "preflight_output_dir": preflight,
            "launch_record_path": record, "test_csv_path": test_csv, "evaluator_path": evaluator_path,
            "builder_path": builder_path, "config": cfg}


def launch(contract_path: str | Path, expected_contract_sha256: str) -> int:
    contract_path = Path(contract_path).expanduser().resolve(strict=True)
    launcher_path = Path(__file__).resolve(strict=True)
    if sha256_file(contract_path) != expected_contract_sha256:
        raise ValueError("launcher contract SHA-256 mismatch")
    contract = _read_json(contract_path)
    paths = _check_inputs(contract, contract_path=contract_path,
                          contract_sha256=expected_contract_sha256, launcher_path=launcher_path)
    task_tools = paths["task_root"] / "tools"
    if str(task_tools) not in sys.path:
        sys.path.insert(0, str(task_tools))
    import run_banking_cil_v10 as driver
    import evaluate_banking_test_v10 as evaluator

    cfg = driver.validate_config(paths["config"], phase="preflight")
    if driver._config_sha(cfg) != contract["config_sha256"]:
        raise ValueError("semantic config hash differs from contract")
    if (cfg.get("device") != contract["device"] or cfg.get("training_manifest_sha256") != contract["training_manifest_sha256"]):
        raise ValueError("device/data binding differs from the frozen config")
    source_hashes, source_lock, _manifest, _train, _dev = driver._source_prepare(
        paths["project_root"], paths["task_root"], paths["model_dir"], paths["training_pack"], cfg)
    if source_hashes != contract["source_hashes"] or source_lock != contract["source_lock_sha256"]:
        raise ValueError("source graph/lock differs from the pinned contract")
    driver._verify_model_assets(paths["model_dir"], cfg["model_assets_sha256"])

    eval_contract = _read_json(paths["evaluation_contract_path"])
    _validate_eval_contract_binding(evaluator, eval_contract, cfg, contract)

    allocator_before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if allocator_before not in (None, ALLOCATOR_CONF):
        raise ValueError("conflicting CUDA allocator config; refusing to override")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR_CONF
    os.environ["PYTHONUNBUFFERED"] = "1"

    record = {"schema": "banking-v10-test-memory-launch-record-v1", "status": "cpu_preflight",
              "contract_sha256": expected_contract_sha256, "launcher_sha256": sha256_file(launcher_path),
              "evaluator_sha256": contract["evaluator_sha256"], "training_root": str(paths["training_root"]),
              "training_pack": str(paths["training_pack"]), "preflight_output_dir": str(paths["preflight_output_dir"]),
              "training_pack_sha256": contract["training_pack_sha256"],
              "evaluation_output_dir": str(paths["output_dir"]), "test_csv_declared_path": str(paths["test_csv_path"]),
              "publisher_train_csv_path": str(paths["publisher_train_csv_path"]),
              "publisher_train_csv_sha256": TRAIN_SOURCE_SHA256,
              "official_test_expected_sha256": TEST_SOURCE_SHA256,
              "official_test_expected_bytes": TEST_SOURCE_BYTES,
              "allocator_conf": ALLOCATOR_CONF, "memory_fraction": MEMORY_FRACTION,
              "memory_fraction_callback_registered_before_cuda_init": False,
              "memory_fraction_callback_called": False, "official_test_payload_read_by_launcher": False,
              "cuda_initialized_by_launcher_before_cpu_preflight": False,
              "automatic_retries": 0, "started_unix": time.time()}
    _atomic_json(paths["launch_record_path"], record)
    callback = None
    torch = None
    try:
        preflight_args = evaluator_argv(contract, mode="preflight")
        rc = evaluator.main(preflight_args)
        if rc != 0:
            raise RuntimeError(f"official evaluator CPU preflight returned {rc}")
        preflight_path = paths["preflight_output_dir"] / "preflight.json"
        preflight = _read_json(preflight_path)
        seal = _read_json(paths["seal_path"])
        expected_foundation_audit = {
            "release_sha256": contract["main_release_sha256"],
            "foundations": eval_contract["foundation_refs"],
            "all_qualified": True,
            "all_checkpoints_cpu_deserialized": True,
        }
        _validate_preflight_result(preflight, contract, seal, evaluator, expected_foundation_audit)
        record["cpu_preflight_passed"] = True
        import torch
        if torch.__version__ != TORCH_VERSION:
            raise RuntimeError(f"PyTorch version mismatch: {torch.__version__}")
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized during CPU preflight")

        # Rehash all externally bound sources after CPU preflight and before
        # registering CUDA policy or handing control to evaluation.
        if (sha256_file(contract_path) != expected_contract_sha256 or
                sha256_file(paths["config_path"]) != contract["config_file_sha256"] or
                sha256_file(paths["training_pack"]) != contract["training_pack_sha256"] or
                sha256_file(paths["publisher_train_csv_path"]) != TRAIN_SOURCE_SHA256 or
                sha256_file(paths["training_manifest_path"]) != contract["training_manifest_sha256"] or
                sha256_file(paths["seal_path"]) != contract["seal_sha256"] or
                sha256_file(paths["main_release_path"]) != contract["main_release_sha256"] or
                sha256_file(paths["evaluation_contract_path"]) != contract["evaluation_contract_sha256"] or
                sha256_file(paths["evaluator_path"]) != contract["evaluator_sha256"] or
                sha256_file(paths["builder_path"]) != contract["contract_builder_sha256"]):
            raise ValueError("a source/contract/seal binding changed after CPU preflight")
        current_source_lock = driver._verify_source_lock(
            cfg, driver._source_hashes(paths["project_root"], paths["task_root"]))
        if current_source_lock != contract["source_lock_sha256"]:
            raise ValueError("source lock changed after CPU preflight")
        driver._verify_model_assets(paths["model_dir"], cfg["model_assets_sha256"])
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized while rechecking CPU-preflight inputs")
        callback = _install_memory_policy(torch, contract["device"])
        record.update(status="evaluating", preflight_sha256=sha256_file(preflight_path),
                      memory_fraction_callback_registered_before_cuda_init=callback["registered"],
                      memory_fraction_callback_called=False)
        _atomic_json(paths["launch_record_path"], record, replace=True)

        rc = evaluator.main(evaluator_argv(contract, mode="evaluate"))
        if rc != 0:
            raise RuntimeError(f"official evaluator returned {rc}")
        result_path = paths["output_dir"] / "evaluation.json"
        result = _read_json(result_path)
        _validate_terminal_result(result, contract, seal, preflight, evaluator)
        if not callback["called"] or callback["error"] is not None:
            raise RuntimeError("deferred memory-fraction callback did not apply cleanly")
        # Verify the full source/model/config/pack graph once more after scoring.
        if (sha256_file(contract_path) != expected_contract_sha256 or
                sha256_file(paths["config_path"]) != contract["config_file_sha256"] or
                sha256_file(paths["training_pack"]) != contract["training_pack_sha256"] or
                sha256_file(paths["publisher_train_csv_path"]) != TRAIN_SOURCE_SHA256 or
                sha256_file(paths["training_manifest_path"]) != contract["training_manifest_sha256"] or
                sha256_file(paths["seal_path"]) != contract["seal_sha256"] or
                sha256_file(paths["main_release_path"]) != contract["main_release_sha256"] or
                sha256_file(paths["evaluation_contract_path"]) != contract["evaluation_contract_sha256"] or
                sha256_file(paths["evaluator_path"]) != contract["evaluator_sha256"] or
                sha256_file(paths["builder_path"]) != contract["contract_builder_sha256"]):
            raise ValueError("a bound source/model/config/data/seal file changed during evaluation")
        post_source_lock = driver._verify_source_lock(
            cfg, driver._source_hashes(paths["project_root"], paths["task_root"]))
        if post_source_lock != contract["source_lock_sha256"]:
            raise ValueError("source lock changed during evaluation")
        driver._verify_model_assets(paths["model_dir"], cfg["model_assets_sha256"])
        record.update(status="complete", exit_code=0, result_path=str(result_path),
                      result_sha256=sha256_file(result_path),
                      memory_fraction_callback_called=callback["called"],
                      callback_error=callback["error"], cuda_initialized_at_exit=bool(torch.cuda.is_initialized()),
                      ended_unix=time.time())
        _atomic_json(paths["launch_record_path"], record, replace=True)
        return 0
    except BaseException as exc:
        record.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}",
                      ended_unix=time.time())
        if callback is not None:
            record.update(memory_fraction_callback_registered_before_cuda_init=bool(callback.get("registered")),
                          memory_fraction_callback_called=bool(callback.get("called")),
                          callback_error=callback.get("error"))
        if torch is not None:
            record["cuda_initialized_at_failure"] = bool(torch.cuda.is_initialized())
        _atomic_json(paths["launch_record_path"], record, replace=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args(argv)
    return launch(args.contract, args.contract_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
