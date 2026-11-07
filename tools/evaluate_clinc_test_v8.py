#!/usr/bin/env python3
"""Strictly gated final-checkpoint evaluation for the local controlled-routing CLINC ablation.

The access-plan mode verifies the completed main-grid seal and every foundation,
run receipt, and final checkpoint before returning a download plan. Evaluation
repeats that gate before opening the test parquet. This is a final-checkpoint
evaluation for the local controlled-routing ablation, not paper-faithful HOPE;
it never selects checkpoints, updates parameters, or uses task masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence


TEST_SOURCE = {
    "repo_id": "clinc/clinc_oos",
    "revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
    "path": "plus/test-00000-of-00001.parquet",
    "bytes": 135844,
    "sha256": "3e60e45b25bf86543aa5df8ba4fcc674114164e6184f0197690648c2908d0102",
    "metadata_sha256": "957dedb294471540ac0ca1571195d147e4922ce20a38678193e07fee06771149",
}
EXPECTED_ROWS = 4500
EXPECTED_RAW_ROWS = 5500
EXPECTED_PER_CLASS = 30
OOS_LABEL_ID = 42
MAX_FORWARD_CALLS = 40_000
MAX_WALL_SECONDS = 3600
RESULT_SCHEMA = "clinc-v8-final-test-evaluation-v1"
PLAN_SCHEMA = "clinc-v8-test-access-plan-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _same(left, right) -> bool:
    return canonical_sha(left) == canonical_sha(right)


def mutable_context_digest(state: dict, torch) -> str:
    """Match R2's mutable-context digest directly from a checkpoint state dict."""
    digest = hashlib.sha256()
    if set(state) != {"memory", "readout_a", "readout_b", "readout_fp32_accumulation",
                      "routing", "trainable"}:
        raise ValueError("checkpoint mutable-context state schema mismatch")
    for name, tensor in sorted(state["memory"].items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    for name in ("readout_a", "readout_b"):
        value = state[name].detach().contiguous().cpu()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    digest.update(canonical_sha({"routing": state["routing"],
        "trainable": state["trainable"],
        "readout_fp32_accumulation": state["readout_fp32_accumulation"]}).encode("ascii"))
    return digest.hexdigest()


def audit_sleep_exposure(receipt: dict, arm: str, seed: int | None = None) -> dict:
    """Derive Sleep text replay from actual event/work counters, not a flag."""
    events = receipt.get("sleep_event_index")
    adam = receipt.get("physical_sleep_adamw_steps")
    work = receipt.get("work")
    work_events = work.get("sleep_events") if isinstance(work, dict) else None
    if any(type(value) is not int or value < 0 for value in (events, adam, work_events)):
        raise ValueError("receipt lacks typed nonnegative Sleep event/Adam/work counters")
    if events != work_events:
        raise ValueError("receipt Sleep event index differs from cumulative work counter")
    if arm == "HOPE" and (events != 0 or adam != 0):
        raise ValueError("HOPE control unexpectedly records Sleep events or Adam steps")
    if arm == "HOPE_SLEEP_FIXEDMIX" and (events == 0 or adam == 0):
        raise ValueError("fixed-mix Sleep arm lacks actual Sleep events or Adam steps")
    exposure = receipt.get("data_exposure")
    declared = exposure.get("sleep_text_replay") if isinstance(exposure, dict) else None
    if type(declared) is not bool:
        raise ValueError("receipt lacks a typed sleep_text_replay metadata flag")
    actual = events > 0 and adam > 0
    return {"seed": seed, "arm": arm, "sleep_events": events, "physical_sleep_adamw_steps": adam,
        "work_sleep_events": work_events, "sleep_text_replay_derived_from_counters": actual,
        "receipt_sleep_text_replay_claim": declared,
        "receipt_flag_matches_counters": declared == actual}


def validate_seal_summary(seal: dict, *, expected_master_sha: str,
                          expected_manifest_sha: str, expected_source_hashes: dict,
                          expected_source_lock_sha: str, expected_grid: list[dict]) -> None:
    """Pure structural/binding gate; filesystem bytes are verified by the caller."""
    if seal.get("schema") != "clinc-cil-v8-main-six-seal" or seal.get("complete") is not True:
        raise ValueError("main-grid seal is missing, incomplete, or has an unsupported schema")
    if seal.get("test_data_included") is not False or seal.get("test_release_ready") is not True:
        raise ValueError("main-grid seal does not authorize isolated test evaluation")
    if seal.get("master_config_sha256") != expected_master_sha:
        raise ValueError("main-grid seal master-config hash mismatch")
    if seal.get("training_manifest_sha256") != expected_manifest_sha:
        raise ValueError("main-grid seal training-manifest hash mismatch")
    if seal.get("source_hashes") != expected_source_hashes:
        raise ValueError("main-grid seal public source hashes mismatch")
    if seal.get("source_lock_sha256") != expected_source_lock_sha:
        raise ValueError("main-grid seal source-lock digest mismatch")
    if seal.get("main_grid") != expected_grid:
        raise ValueError("main-grid seal seed/arm grid mismatch")
    if not isinstance(seal.get("verified_foundations"), list) or len(seal["verified_foundations"]) != 3:
        raise ValueError("main-grid seal lacks all three foundation byte receipts")
    if not isinstance(seal.get("verified_run_files"), list) or len(seal["verified_run_files"]) != 6:
        raise ValueError("main-grid seal lacks all six final-checkpoint/receipt byte bindings")
    if not isinstance(seal.get("runs"), list) or len(seal["runs"]) != 6:
        raise ValueError("main-grid seal lacks exactly six terminal run receipts")


def validate_evaluation_contract(contract: dict, *, evaluator_sha: str,
                                 master_sha: str, source_lock_sha: str) -> None:
    if contract.get("schema") != "clinc-v8-test-evaluation-contract-v1" or contract.get("status") != "frozen":
        raise ValueError("test evaluation contract is missing or not frozen")
    if contract.get("evaluator_sha256") != evaluator_sha:
        raise ValueError("evaluation contract evaluator source hash mismatch")
    if contract.get("training_master_sha256") != master_sha:
        raise ValueError("evaluation contract master config hash mismatch")
    if contract.get("source_lock_sha256") != source_lock_sha:
        raise ValueError("evaluation contract training source lock mismatch")
    if contract.get("test_source") != TEST_SOURCE:
        raise ValueError("evaluation contract test source pin mismatch")
    if contract.get("evaluation_policy") != (
        "score all 4500 in-scope test rows on the six sealed final checkpoints; "
        "free generation, no task mask, no updates, no checkpoint selection"):
        raise ValueError("evaluation policy differs from the frozen all-row endpoint protocol")


def compare_work_counters(checkpoint_work: dict, receipt_work: dict) -> None:
    """Discrete work must match; wall time may increase between CP and receipt."""
    def visit(left, right, prefix="work"):
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
                raise ValueError(f"{prefix} key structure differs between checkpoint and receipt")
            for key in left:
                visit(left[key], right[key], f"{prefix}.{key}")
            return
        if prefix.endswith(".active_wall_seconds"):
            if (type(left) not in (int, float) or type(right) not in (int, float) or
                not math.isfinite(float(left)) or not math.isfinite(float(right)) or
                left < 0 or right < 0 or float(left) > float(right) + 1e-6):
                raise ValueError("checkpoint/receipt active wall time is invalid or reversed")
            return
        if left != right:
            raise ValueError(f"{prefix} discrete counter differs between checkpoint and receipt")
    visit(checkpoint_work, receipt_work)


def validate_paired_foundation_start(run_receipt: dict, foundation_receipt: dict) -> None:
    initial = run_receipt.get("initial_mutable_sha256")
    foundation = foundation_receipt.get("foundation_context_sha256")
    if (not isinstance(initial, str) or len(initial) != 64 or
        not isinstance(foundation, str) or len(foundation) != 64 or initial != foundation):
        raise ValueError("continuation initial mutable state differs from its paired foundation")


def _load_full_runner(project_root: Path):
    tools_dir = Path(__file__).resolve().parent
    task_root = tools_dir.parent
    for value in (str(task_root), str(tools_dir)):
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    from tools import run_clinc_cil_v8_full as full
    if Path(full.__file__).resolve() != tools_dir / "run_clinc_cil_v8_full.py":
        raise RuntimeError("full runner imported from an unexpected path")
    return full


def verify_release_gate(*, project_root: str | Path, model_path: str | Path,
                        master_config: str | Path, training_pack: str | Path,
                        foundation_dir: str | Path, run_root: str | Path,
                        seal_path: str | Path,
                        evaluation_contract: str | Path) -> dict:
    """Verify exact source/config/seal/receipt/checkpoint bytes without test access."""
    project_root, model_path = Path(project_root).resolve(), Path(model_path).resolve()
    master_path, pack_path = Path(master_config).resolve(), Path(training_pack).resolve()
    foundation_root, run_root = Path(foundation_dir).resolve(), Path(run_root).resolve()
    seal_path = Path(seal_path).resolve()
    contract_path = Path(evaluation_contract).resolve()
    for path in (master_path, pack_path, seal_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    full = _load_full_runner(project_root)
    cfg = full.validate_master_config(_load_json(master_path))
    master_sha, manifest_sha = sha256_file(master_path), sha256_file(pack_path)
    if manifest_sha != cfg.get("training_manifest_sha256"):
        raise ValueError("training-manifest bytes differ from the frozen master config")
    source_hashes = full._source_hashes(project_root, Path(full.__file__).resolve().parent)
    source_lock_sha = full.canonical_sha(source_hashes)
    evaluator_sha = sha256_file(Path(__file__).resolve())
    contract = _load_json(contract_path)
    validate_evaluation_contract(contract, evaluator_sha=evaluator_sha,
        master_sha=master_sha, source_lock_sha=source_lock_sha)
    for relative, expected in cfg["model_assets_sha256"].items():
        asset = (model_path / relative).resolve()
        if model_path not in asset.parents or not asset.is_file() or sha256_file(asset) != expected:
            raise ValueError(f"locked model asset bytes mismatch: {relative}")

    seal = _load_json(seal_path)
    grid = [{"seed": seed, "arm": arm} for seed in full.SEEDS for arm in full.ARMS]
    validate_seal_summary(seal, expected_master_sha=master_sha,
        expected_manifest_sha=manifest_sha, expected_source_hashes=source_hashes,
        expected_source_lock_sha=source_lock_sha, expected_grid=grid)
    if Path(seal.get("master_config_path", "")).resolve() != master_path:
        raise ValueError("seal config path differs from the supplied frozen master")

    foundations, verified_foundations = full.load_and_verify_foundation_grid(
        foundation_root, master_sha=master_sha, manifest_sha=manifest_sha,
        source_lock_sha=source_lock_sha, model_revision=cfg["model_revision"],
        model_assets_sha256=cfg["model_assets_sha256"])
    if not _same(seal["verified_foundations"], verified_foundations):
        raise ValueError("seal foundation receipt/checkpoint byte list mismatch")

    recreated_runs, recreated_files = [], []
    sealed_by_pair = {}
    for row in seal["runs"]:
        key = (int(row.get("seed", -1)), str(row.get("arm", "")))
        if key in sealed_by_pair:
            raise ValueError(f"duplicate sealed run receipt: {key}")
        sealed_by_pair[key] = row
    if set(sealed_by_pair) != {(seed, arm) for seed in full.SEEDS for arm in full.ARMS}:
        raise ValueError("sealed run receipt grid is incomplete")
    foundations_by_seed = {int(key): value for key, value in foundations.items()}
    exposure_audits = []
    verified_receipts = []
    for seed in full.SEEDS:
        for arm in full.ARMS:
            run_dir = run_root / f"seed{seed}" / arm
            receipt_path = run_dir / "run-receipt.json"
            if not receipt_path.is_file():
                raise FileNotFoundError(f"missing terminal run receipt: {receipt_path}")
            receipt = _load_json(receipt_path)
            if receipt.get("seed") != seed or receipt.get("arm") != arm:
                raise ValueError("run receipt path and seed/arm disagree")
            if receipt.get("child_config_sha256") != full._child_config_sha(master_sha, "continuation", seed, arm):
                raise ValueError(f"seed {seed} {arm}: child config binding mismatch")
            foundation = foundations_by_seed[seed]
            foundation_receipt_path = foundation_root / f"seed{seed}" / "foundation-receipt.json"
            if receipt.get("foundation_checkpoint_sha256") != foundation["final_checkpoint_sha256"]:
                raise ValueError(f"seed {seed} {arm}: common foundation checkpoint mismatch")
            try:
                validate_paired_foundation_start(receipt, foundation)
            except ValueError as exc:
                raise ValueError(f"seed {seed} {arm}: {exc}") from exc
            if receipt.get("foundation_receipt_sha256") != sha256_file(foundation_receipt_path):
                raise ValueError(f"seed {seed} {arm}: common foundation receipt bytes mismatch")
            checkpoint_path = full._safe_child_path(run_dir, receipt.get("final_checkpoint", ""))
            checkpoint_sha = receipt.get("final_checkpoint_sha256")
            if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_sha:
                raise ValueError(f"seed {seed} {arm}: final checkpoint bytes mismatch")
            receipt_sha = sha256_file(receipt_path)
            row = dict(receipt)
            row["run_receipt_path"] = str(receipt_path.relative_to(run_root))
            row["run_receipt_sha256"] = receipt_sha
            if not _same(row, sealed_by_pair[(seed, arm)]):
                raise ValueError(f"seed {seed} {arm}: terminal receipt differs from sealed bytes")
            recreated_runs.append(row)
            verified_receipts.append({"seed": seed, "arm": arm,
                "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha, "receipt": receipt})
            recreated_files.append({"seed": seed, "arm": arm,
                "checkpoint_path": str(checkpoint_path.relative_to(run_root)),
                "checkpoint_sha256": checkpoint_sha,
                "run_receipt_path": str(receipt_path.relative_to(run_root)),
                "run_receipt_sha256": receipt_sha})
    full.validate_main_grid_receipts(recreated_runs, master_sha=master_sha,
        manifest_sha=manifest_sha, source_lock_sha=source_lock_sha,
        inner_cycles_per_event=cfg["inner_cycles_per_event"],
        objective_version=cfg["objective_version"])
    if not _same(seal["verified_run_files"], recreated_files):
        raise ValueError("seal final checkpoint/receipt byte list mismatch")

    # Deserialize only after all byte hashes were checked, still before any test read.
    import torch
    lifecycle_path = Path(full.__file__).resolve().parent / "clinc_lifecycle_v8.py"
    if not lifecycle_path.is_file():
        raise FileNotFoundError(lifecycle_path)
    if str(lifecycle_path.parent) not in sys.path:
        sys.path.insert(0, str(lifecycle_path.parent))
    import clinc_lifecycle_v8  # noqa: F401
    for seed in full.SEEDS:
        foundation_receipt = foundations[seed]
        cp_path = full._safe_child_path(foundation_root / f"seed{seed}", foundation_receipt["final_checkpoint"])
        expected = {"master_config_sha256": master_sha,
            "child_config_sha256": full._child_config_sha(master_sha, "foundation", seed, None),
            "training_manifest_sha256": manifest_sha, "source_lock_sha256": source_lock_sha,
            "seed": seed, "arm": None, "mode": "foundation", "foundation_checkpoint_sha256": None}
        payload = full._load_safe_checkpoint(cp_path, torch=torch, expected=expected)
        if (payload["cursor"]["tasks_completed"] != 1 or
            payload["cursor"]["task_id"] != 0 or payload["cursor"]["next_step"] != 189):
            raise ValueError(f"seed {seed}: foundation checkpoint is not the task-0 endpoint")
        if payload["wake"].ticks != foundation_receipt["wake_ticks"] or \
           payload["physical_wake_updates_run"] != foundation_receipt["wake_ticks"]:
            raise ValueError(f"seed {seed}: foundation wake window differs from its receipt")
        if payload["physical_sleep_adamw_steps"] != 0:
            raise ValueError(f"seed {seed}: foundation checkpoint contains Sleep Adam steps")
        compare_work_counters(payload["work_cumulative"], foundation_receipt["work"])
        if mutable_context_digest(payload["context"], torch) != foundation_receipt.get("foundation_context_sha256"):
            raise ValueError(f"seed {seed}: foundation context bytes differ from its terminal receipt")
        del payload
    for seed in full.SEEDS:
        for arm in full.ARMS:
            run_dir = run_root / f"seed{seed}" / arm
            receipt = sealed_by_pair[(seed, arm)]
            cp_path = full._safe_child_path(run_dir, receipt["final_checkpoint"])
            expected = {"master_config_sha256": master_sha,
                "child_config_sha256": full._child_config_sha(master_sha, "continuation", seed, arm),
                "training_manifest_sha256": manifest_sha, "source_lock_sha256": source_lock_sha,
                "seed": seed, "arm": arm, "mode": "continuation",
                "foundation_checkpoint_sha256": foundations[seed]["final_checkpoint_sha256"]}
            payload = full._load_safe_checkpoint(cp_path, torch=torch, expected=expected)
            cursor = payload["cursor"]
            if (cursor["tasks_completed"] != 10 or cursor["task_id"] != 9 or
                cursor["next_step"] != 189 or cursor["safe_point"] != "post_all_due_events"):
                raise ValueError(f"seed {seed} {arm}: final checkpoint is not task-10 endpoint")
            receipt = sealed_by_pair[(seed, arm)]
            foundation_receipt = foundations[seed]
            if payload["physical_wake_updates_run"] + foundation_receipt["wake_ticks"] != receipt.get("physical_wake_updates"):
                raise ValueError(f"seed {seed} {arm}: continuation plus foundation wake count differs from receipt")
            if (receipt.get("wake_ticks") != receipt.get("physical_wake_updates") or
                payload["wake"].ticks != receipt.get("wake_ticks")):
                raise ValueError(f"seed {seed} {arm}: global wake tick/window differs from final checkpoint")
            for payload_key, receipt_key in (("physical_sleep_adamw_steps", "physical_sleep_adamw_steps"),
                                              ("sleep_event_index", "sleep_event_index")):
                if not _same(payload[payload_key], sealed_by_pair[(seed, arm)].get(receipt_key)):
                    raise ValueError(f"seed {seed} {arm}: final checkpoint {payload_key} differs from receipt")
            compare_work_counters(payload["work_cumulative"], receipt["work"])
            exposure_audits.append(audit_sleep_exposure(sealed_by_pair[(seed, arm)], arm, seed))
            del payload
    return {"schema": "clinc-v8-release-gate-v1", "passed": True,
        "seal_sha256": sha256_file(seal_path), "master_config_sha256": master_sha,
        "training_manifest_sha256": manifest_sha, "source_hashes": source_hashes,
        "source_lock_sha256": source_lock_sha,
        "verified_foundation_count": 3, "verified_run_count": 6,
        "verified_checkpoint_count": 9, "training_exposure_audit": exposure_audits,
        "source_lock_sha256": source_lock_sha, "evaluator_source_sha256": evaluator_sha,
        "evaluation_contract_sha256": sha256_file(contract_path),
        "evaluation_contract_path": str(contract_path),
        "verified_foundations": verified_foundations,
        "verified_run_files": recreated_files,
        "verified_receipts": verified_receipts,
        "test_payload_read": False}


def assert_gate_inputs_unchanged(gate: dict, *, project_root: str | Path,
                                 model_path: str | Path, master_config: str | Path,
                                 training_pack: str | Path, seal_path: str | Path) -> None:
    """Recheck all gate inputs and checkpoint byte bindings immediately pre-test."""
    full = _load_full_runner(Path(project_root).resolve())
    master_path, pack_path, seal_file = map(Path, (master_config, training_pack, seal_path))
    source_hashes = full._source_hashes(Path(project_root).resolve(), Path(full.__file__).resolve().parent)
    if (sha256_file(master_path) != gate["master_config_sha256"] or
        sha256_file(pack_path) != gate["training_manifest_sha256"] or
        sha256_file(seal_file) != gate["seal_sha256"] or
        sha256_file(Path(__file__).resolve()) != gate["evaluator_source_sha256"] or
        source_hashes != gate["source_hashes"]):
        raise ValueError("a release-gate config/seal/source input changed before test access")
    contract_path = Path(gate["evaluation_contract_path"])
    if sha256_file(contract_path) != gate["evaluation_contract_sha256"]:
        raise ValueError("evaluation contract changed after release gate")
    cfg = full.validate_master_config(_load_json(master_path))
    validate_evaluation_contract(_load_json(contract_path),
        evaluator_sha=gate["evaluator_source_sha256"],
        master_sha=gate["master_config_sha256"], source_lock_sha=gate["source_lock_sha256"])
    model_root = Path(model_path).resolve()
    for relative, expected in cfg["model_assets_sha256"].items():
        asset = (model_root / relative).resolve()
        if model_root not in asset.parents or not asset.is_file() or sha256_file(asset) != expected:
            raise ValueError(f"model asset changed after release gate: {relative}")
    for item in gate["verified_foundations"]:
        if sha256_file(Path(item["receipt_path"])) != item["receipt_sha256"] or \
           sha256_file(Path(item["checkpoint_path"])) != item["checkpoint_sha256"]:
            raise ValueError("foundation receipt/checkpoint changed after release gate")
    for item in gate["verified_receipts"]:
        if sha256_file(Path(item["receipt_path"])) != item["receipt_sha256"] or \
           sha256_file(Path(item["checkpoint_path"])) != item["checkpoint_sha256"]:
            raise ValueError("run receipt/checkpoint changed after release gate")


def access_plan(gate: dict) -> dict:
    if gate.get("passed") is not True or gate.get("test_payload_read") is not False:
        raise ValueError("release gate did not pass before access plan")
    return {"schema": PLAN_SCHEMA, "gate_sha256": canonical_sha(gate),
        "evaluation_contract_sha256": gate["evaluation_contract_sha256"],
        "source": dict(TEST_SOURCE), "download_after_gate_only": True,
        "repo_id": TEST_SOURCE["repo_id"],
        "url": "https://huggingface.co/datasets/clinc/clinc_oos/resolve/"
               + TEST_SOURCE["revision"] + "/" + TEST_SOURCE["path"]}


def _validate_source_intent_names(intent_names, classes, *, oos_id: int) -> list[str]:
    from tools.clinc_cil_common_v8 import normalize_prediction
    if not isinstance(intent_names, list) or len(intent_names) != len(classes) + 1:
        raise ValueError("test ClassLabel metadata must contain exactly 151 names")
    if type(oos_id) is not int or not 0 <= oos_id < len(intent_names):
        raise ValueError("OOS label ID is outside the source ClassLabel map")
    if any(not isinstance(name, str) or not name for name in intent_names):
        raise ValueError("test ClassLabel names must be nonempty strings")
    normalized = [normalize_prediction(name) for name in intent_names]
    if len(set(normalized)) != len(normalized) or normalized[oos_id] != "oos":
        raise ValueError(f"OOS must be the unique declared raw ClassLabel id {oos_id}")
    class_names = {normalize_prediction(name) for name in classes}
    in_scope_names = {name for index, name in enumerate(normalized) if index != oos_id}
    if in_scope_names != class_names:
        raise ValueError("non-OOS source label map does not exactly match the canonical 150 labels")
    return normalized


def filter_intent_records(texts, intents, classes: tuple[str, ...], *, intent_names=None,
                          oos_id: int = OOS_LABEL_ID,
                          expected_total=EXPECTED_ROWS,
                          expected_per_class=EXPECTED_PER_CLASS,
                          expected_raw_rows: int | None = None) -> tuple[list[dict], dict]:
    """Map exactly the source ClassLabel domain; only numeric ID 42 is OOS."""
    if len(texts) != len(intents):
        raise ValueError("test text/intent column lengths differ")
    from tools.clinc_cil_common_v8 import normalize_prediction
    names = _validate_source_intent_names(intent_names, classes, oos_id=oos_id)
    class_to_id = {normalize_prediction(label): index for index, label in enumerate(classes)}
    rows, excluded_labels = [], set()
    for text, intent in zip(texts, intents):
        if not isinstance(text, str):
            raise ValueError("test text values must be strings")
        if isinstance(intent, int) and not isinstance(intent, bool):
            label_index = intent
            if not 0 <= label_index < len(names):
                raise ValueError("test intent ID is outside the source ClassLabel map")
        elif isinstance(intent, str):
            key = normalize_prediction(intent)
            if key not in names:
                raise ValueError(f"unknown test label is not the declared OOS id {oos_id}")
            label_index = names.index(key)
        else:
            raise ValueError("test intent encoding must be a source ClassLabel string or integer")
        key = names[label_index]
        if label_index == oos_id:
            excluded_labels.add(key)
            continue
        class_id = class_to_id.get(key)
        if class_id is None:
            raise ValueError(f"unrecognized non-OOS source label at id {label_index}")
        rows.append({"text": text, "intent_id": class_id})
    counts = [0] * len(classes)
    for row in rows:
        counts[row["intent_id"]] += 1
    if len(rows) != expected_total or any(value != expected_per_class for value in counts):
        raise ValueError(f"in-scope CLINC test rows must be exactly {expected_total} ({expected_per_class} per each of {len(classes)} labels)")
    if expected_raw_rows is not None and len(texts) != expected_raw_rows:
        raise ValueError(f"raw CLINC test split must contain exactly {expected_raw_rows} rows")
    if expected_raw_rows is not None and len(texts) - len(rows) != expected_raw_rows - expected_total:
        raise ValueError("raw OOS ID 42 count differs from the locked 1,000-row scope")
    return rows, {"all_rows": len(texts), "primary_rows": len(rows),
        "oos_excluded_rows": len(texts) - len(rows),
        "oos_raw_label_id": oos_id, "oos_distinct_labels": len(excluded_labels),
        "rows_per_class": expected_per_class,
        "classes_present": len(counts)}


def _intent_names_from_metadata(table) -> list[str] | None:
    metadata = table.schema.metadata or {}
    raw = metadata.get(b"huggingface")
    if not raw:
        return None
    try:
        info = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    features = info.get("features", info.get("info", {}).get("features", {}))
    feature = features.get("intent", {}) if isinstance(features, dict) else {}
    names = feature.get("names") if isinstance(feature, dict) else None
    if isinstance(names, dict):
        try:
            names = [names[str(index)] for index in range(len(names))]
        except KeyError:
            return None
    if isinstance(names, list) and names and all(isinstance(x, str) for x in names):
        return names
    return None


def _read_test_rows(test_path: Path, classes: tuple[str, ...]) -> tuple[list[dict], dict]:
    """Read only the locked parquet; caller must first pass verify_release_gate."""
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    size = test_path.stat().st_size
    digest = sha256_file(test_path)
    if size != TEST_SOURCE["bytes"] or digest != TEST_SOURCE["sha256"]:
        raise ValueError("test parquet does not match the locked source bytes")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the locked test parquet") from exc
    table = pq.read_table(test_path)
    if sha256_file(test_path) != digest:
        raise ValueError("test parquet changed while it was being read")
    columns = set(table.column_names)
    if not {"text", "intent"}.issubset(columns):
        raise ValueError(f"locked parquet lacks required text/intent columns: {sorted(columns)}")
    texts, intents = table["text"].to_pylist(), table["intent"].to_pylist()
    intent_names = _intent_names_from_metadata(table)
    if len(texts) != EXPECTED_RAW_ROWS:
        raise ValueError("raw CLINC test split must contain exactly 5,500 rows")
    rows, scope = filter_intent_records(texts, intents, classes,
        intent_names=intent_names, oos_id=OOS_LABEL_ID,
        expected_raw_rows=EXPECTED_RAW_ROWS)
    return rows, {"payload_bytes": size, "payload_sha256": digest,
        "repo_id": TEST_SOURCE["repo_id"], "source_revision": TEST_SOURCE["revision"],
        "source_path": TEST_SOURCE["path"], **scope}


def _text_digest(text: str) -> str:
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("_", " ").split())


def greedy_forward_upper_bound(*, rows: int, batch_size: int,
                               distinct_prefix_lengths: int, max_generation_steps: int,
                               runs: int = 6) -> int:
    if min(rows, batch_size, distinct_prefix_lengths, max_generation_steps, runs) < 1:
        raise ValueError("forward bound arguments must be positive")
    group_batches = math.ceil(rows / batch_size) + distinct_prefix_lengths
    return runs * max_generation_steps * group_batches


def evaluate(*, project_root: str | Path, model_path: str | Path,
             master_config: str | Path, training_pack: str | Path,
             foundation_dir: str | Path, run_root: str | Path,
             seal_path: str | Path, evaluation_contract: str | Path,
             test_path: str | Path,
             output_path: str | Path) -> dict:
    started = time.monotonic()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_path}")
    # This is deliberately the first operation that can lead toward test access.
    gate = verify_release_gate(project_root=project_root, model_path=model_path,
        master_config=master_config, training_pack=training_pack,
        foundation_dir=foundation_dir, run_root=run_root, seal_path=seal_path,
        evaluation_contract=evaluation_contract)
    assert_gate_inputs_unchanged(gate, project_root=project_root, model_path=model_path,
        master_config=master_config, training_pack=training_pack, seal_path=seal_path)

    full = _load_full_runner(Path(project_root).resolve())
    master_cfg = full.validate_master_config(_load_json(Path(master_config)))
    runtime_cfg = full._runtime_cfg(master_cfg, full.SEEDS[0], max_mode="continuation", arm=None)
    manifest = _load_json(Path(training_pack))
    from tools import clinc_cil_common_v8 as common
    from tools import run_clinc_cil_v8_r2 as r2
    view = common.make_training_view(manifest)
    import torch
    tools_dir = Path(full.__file__).resolve().parent
    for value in (str(Path(project_root).resolve() / "src"), str(tools_dir)):
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    torch, tokenizer, model, immutable_versions = r2._load_common(
        Path(project_root).resolve(), Path(model_path).resolve(), runtime_cfg["device"], runtime_cfg)
    if full._source_hashes(Path(project_root).resolve(), tools_dir) != gate["source_hashes"]:
        raise RuntimeError("public source bytes changed after the release gate")
    model.eval()
    model.requires_grad_(False)
    frozen_before = immutable_versions(model)
    immutable_versions_before = {name: int(value[1]) for name, value in frozen_before.items()}
    maximum_generation_steps = max(len(tokenizer.encode(label, add_special_tokens=False))
                                   for label in view.classes) + 1
    forward_upper_bound = greedy_forward_upper_bound(rows=EXPECTED_ROWS,
        batch_size=runtime_cfg["batch_size"], distinct_prefix_lengths=runtime_cfg["max_sequence_length"],
        max_generation_steps=maximum_generation_steps)
    if maximum_generation_steps > 8 or forward_upper_bound > MAX_FORWARD_CALLS:
        raise ValueError("fixed R2 greedy-forward bound exceeds the frozen test budget")
    eval_cfg = dict(runtime_cfg)
    eval_cfg.update({"max_forward_calls": MAX_FORWARD_CALLS,
                     "max_wall_seconds": MAX_WALL_SECONDS,
                     "max_update_steps": 0})
    budget = r2.Budget(eval_cfg)
    budget.started = started
    original_logits = model.logits
    import clinc_tail_runtime_v8 as tail
    original_tail = tail.tail_logits

    def counted_logits(tokens, mask=None):
        budget.check()
        if budget.forward_calls >= MAX_FORWARD_CALLS:
            raise RuntimeError("test evaluator full-forward cap reached")
        effective_mask = mask if mask is not None else torch.ones_like(tokens)
        budget.forward(effective_mask)
        return original_logits(tokens, mask)

    def counted_tail(tail_model, context, cache):
        budget.tail(cache, torch.is_grad_enabled() and any(p.requires_grad for p in context.parameters()))
        return original_tail(tail_model, context, cache)

    model.logits = counted_logits
    tail.tail_logits = counted_tail
    per_run = []
    progress_path = Path(output_path).with_name(Path(output_path).stem + ".progress.jsonl")
    failure_path = Path(output_path).with_name(Path(output_path).stem + ".failure.json")
    if progress_path.exists() or failure_path.exists():
        raise FileExistsError("refusing to reuse a prior test-evaluation progress/failure ledger")
    progress = progress_path.open("x", encoding="utf-8", newline="\n")
    phase = "test_setup"
    def record_progress(row):
        progress.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        progress.flush()
        import os
        os.fsync(progress.fileno())
    record_progress({"schema": "clinc-v8-test-evaluation-progress-v1",
        "evaluator_source_sha256": gate["evaluator_source_sha256"],
        "evaluation_contract_sha256": gate["evaluation_contract_sha256"],
        "master_config_sha256": gate["master_config_sha256"],
        "seal_sha256": gate["seal_sha256"], "status": "started"})
    try:
        phase = "test_payload_read"
        from clinc_tail_runtime_v8 import clone_mutable_context
        # Rebind every gate input after loading the immutable model and computing
        # the fixed-budget bound, immediately before the first test reader call.
        assert_gate_inputs_unchanged(gate, project_root=project_root,
            model_path=model_path, master_config=master_config,
            training_pack=training_pack, seal_path=seal_path)
        test_rows, test_meta = _read_test_rows(Path(test_path).resolve(), view.classes)
        train_rows = list(common.iter_role(view, "train", Path(training_pack).resolve().parent))
        train_texts = {_text_digest(row["text"]) for row in train_rows}
        duplicate_mask = [_text_digest(row["text"]) in train_texts for row in test_rows]
        for row, is_duplicate in zip(test_rows, duplicate_mask):
            row["normalized_train_duplicate_secondary"] = is_duplicate
        test_meta["normalized_train_duplicate_rows_secondary"] = sum(duplicate_mask)
        test_meta["normalized_train_duplicate_policy"] = "flag only; retain all 4,500 primary rows"
        for seed in full.SEEDS:
            for arm in full.ARMS:
                phase = f"checkpoint_{seed}_{arm}"
                if time.monotonic() - started > MAX_WALL_SECONDS:
                    raise TimeoutError("test evaluation wall cap exceeded")
                run_started = time.monotonic()
                forwards_before = budget.forward_calls
                binding = next(row for row in gate["verified_receipts"]
                               if row["seed"] == seed and row["arm"] == arm)
                receipt_path = Path(binding["receipt_path"])
                cp_path = Path(binding["checkpoint_path"])
                if (sha256_file(receipt_path) != binding["receipt_sha256"] or
                    sha256_file(cp_path) != binding["checkpoint_sha256"]):
                    raise ValueError(f"seed {seed} {arm}: receipt/checkpoint changed after gate")
                receipt = binding["receipt"]
                if receipt["final_checkpoint_sha256"] != binding["checkpoint_sha256"]:
                    raise ValueError(f"seed {seed} {arm}: gated receipt/checkpoint binding changed")
                expected = {"master_config_sha256": gate["master_config_sha256"],
                    "child_config_sha256": full._child_config_sha(
                        gate["master_config_sha256"], "continuation", seed, arm),
                    "training_manifest_sha256": gate["training_manifest_sha256"],
                    "source_lock_sha256": gate["source_lock_sha256"],
                    "seed": seed, "arm": arm, "mode": "continuation",
                    "foundation_checkpoint_sha256": receipt["foundation_checkpoint_sha256"]}
                payload = full._load_safe_checkpoint(cp_path, torch=torch, expected=expected)
                if (payload["cursor"]["tasks_completed"] != 10 or payload["cursor"]["task_id"] != 9 or
                    payload["cursor"]["next_step"] != 189 or payload["cursor"]["safe_point"] != "post_all_due_events"):
                    raise ValueError(f"seed {seed} {arm}: checkpoint is not the final endpoint")
                context = clone_mutable_context(model)
                context.load_state_dict(payload["context"], strict=True)
                context.memory.eval()
                for parameter in context.parameters():
                    parameter.requires_grad_(False)
                    parameter.grad = None
                before = r2._mutable_context_sha(context)
                with torch.no_grad():
                    predictions, gold_ids, _ = r2._greedy_predictions(
                        model, context, test_rows, tokenizer, view, eval_cfg, torch,
                        runtime_cfg["device"], budget)
                if r2._mutable_context_sha(context) != before:
                    raise RuntimeError("test scoring mutated final checkpoint context")
                if budget.backward_calls or budget.update_steps:
                    raise RuntimeError("test evaluator performed a backward or update")
                overall = common.score_predictions(predictions, gold_ids, view.classes)
                duplicate_indices = [i for i, duplicate in enumerate(duplicate_mask) if duplicate]
                unique_indices = [i for i, duplicate in enumerate(duplicate_mask) if not duplicate]
                duplicate_score = common.score_predictions(
                    [predictions[i] for i in duplicate_indices],
                    [gold_ids[i] for i in duplicate_indices], view.classes) if duplicate_indices else None
                unique_score = common.score_predictions(
                    [predictions[i] for i in unique_indices],
                    [gold_ids[i] for i in unique_indices], view.classes) if unique_indices else None
                order = view.class_orders[str(seed)]
                task_metrics = {}
                for task_id in range(10):
                    ids = set(order[task_id * 15:(task_id + 1) * 15])
                    chosen = [(prediction, gold) for prediction, gold in zip(predictions, gold_ids)
                              if gold in ids]
                    task_metrics[str(task_id)] = common.score_predictions(
                        [item[0] for item in chosen], [item[1] for item in chosen], view.classes)
                per_run.append({"seed": seed, "arm": arm,
                    "checkpoint_sha256": receipt["final_checkpoint_sha256"],
                    "run_receipt_sha256": binding["receipt_sha256"],
                    "primary": overall,
                    "normalized_train_duplicate_secondary": duplicate_score,
                    "normalized_train_unique_secondary": unique_score,
                    "task_metrics": task_metrics,
                    "training_exposure_audit": next(item for item in gate["training_exposure_audit"]
                        if item["arm"] == arm and item["seed"] == seed),
                    "full_backbone_forward_calls": budget.forward_calls - forwards_before,
                    "elapsed_seconds": time.monotonic() - run_started,
                    "prediction_sequence_sha256": canonical_sha(predictions),
                    "checkpoint_context_sha256": before})
                record_progress({"schema": "clinc-v8-test-evaluation-progress-v1",
                    "status": "checkpoint_complete", "seed": seed, "arm": arm,
                    "checkpoint_sha256": receipt["final_checkpoint_sha256"],
                    "run_receipt_sha256": binding["receipt_sha256"],
                    "elapsed_seconds": per_run[-1]["elapsed_seconds"],
                    "full_backbone_forward_calls": per_run[-1]["full_backbone_forward_calls"],
                    "optimizer_steps": budget.update_steps, "backward_calls": budget.backward_calls,
                    "work": budget.as_dict()})
                del payload, context, predictions, gold_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except Exception as exc:
        if not failure_path.exists():
            failure = {"schema": "clinc-v8-test-evaluation-failure-v1",
                "phase": phase, "exception_type": type(exc).__name__, "message": str(exc),
                "evaluator_source_sha256": gate["evaluator_source_sha256"],
                "evaluation_contract_sha256": gate["evaluation_contract_sha256"],
                "completed_checkpoints": [{"seed": row["seed"], "arm": row["arm"],
                    "checkpoint_sha256": row["checkpoint_sha256"]} for row in per_run],
                "elapsed_seconds": time.monotonic() - started,
                "work": budget.as_dict(), "optimizer_steps": budget.update_steps,
                "backward_calls": budget.backward_calls}
            full._atomic_json(failure_path, failure, refuse_existing=True)
        raise
    finally:
        model.logits = original_logits
        tail.tail_logits = original_tail
        progress.close()
    if immutable_versions(model) != frozen_before:
        raise RuntimeError("frozen model changed during test scoring")
    if budget.backward_calls or budget.update_steps:
        raise RuntimeError("read-only test evaluator performed training work")
    by_seed_arm = {(row["seed"], row["arm"]): row for row in per_run}
    summaries = {}
    for arm in full.ARMS:
        values = [by_seed_arm[(seed, arm)]["primary"]["accuracy"] for seed in full.SEEDS]
        summaries[arm] = {"mean_accuracy": statistics.mean(values),
            "sample_sd_accuracy": statistics.stdev(values), "per_seed_accuracy": values,
            "mean_invalid_predictions": statistics.mean([
                by_seed_arm[(seed, arm)]["primary"]["invalid_predictions"] for seed in full.SEEDS])}
    paired = [by_seed_arm[(seed, "HOPE_SLEEP_FIXEDMIX")]["primary"]["accuracy"] -
              by_seed_arm[(seed, "HOPE")]["primary"]["accuracy"] for seed in full.SEEDS]
    result = {"schema": RESULT_SCHEMA, "status": "complete", "test_data_read": True,
        "development_only": False, "test_data_source": test_meta,
        "evaluator_source_sha256": gate["evaluator_source_sha256"],
        "evaluation_contract_sha256": gate["evaluation_contract_sha256"],
        "evaluation_contract": _load_json(Path(evaluation_contract)),
        "training_source_hashes": gate["source_hashes"],
        "runtime_config": runtime_cfg,
        "immutable_versions_before": immutable_versions_before,
        "immutable_versions_unchanged": True,
        "greedy_max_generation_steps": maximum_generation_steps,
        "full_forward_upper_bound": forward_upper_bound,
        "release_gate": gate, "primary_rows_used": EXPECTED_ROWS,
        "candidate_constrained": False, "task_mask_used": False,
        "checkpoint_selection_after_test": False,
        "runs": per_run, "arm_summaries": summaries,
        "paired_fixedmix_minus_hope": {"per_seed_accuracy_difference": paired,
            "mean_accuracy_difference": statistics.mean(paired),
            "sample_sd_accuracy_difference": statistics.stdev(paired)},
        "work": budget.as_dict(), "optimizer_steps": budget.update_steps,
        "backward_calls": budget.backward_calls,
        "elapsed_seconds": time.monotonic() - started,
        "forward_cap": MAX_FORWARD_CALLS, "wall_cap_seconds": MAX_WALL_SECONDS,
        "interpretation": "One fixed test evaluation of the six sealed final checkpoints; no checkpoint selection or parameter updates."}
    try:
        full._atomic_json(output_path, result, refuse_existing=True)
    except Exception as exc:
        if not failure_path.exists():
            full._atomic_json(failure_path, {"schema": "clinc-v8-test-evaluation-failure-v1",
                "phase": "write_terminal_result", "exception_type": type(exc).__name__,
                "message": str(exc), "evaluator_source_sha256": gate["evaluator_source_sha256"],
                "evaluation_contract_sha256": gate["evaluation_contract_sha256"],
                "completed_checkpoints": [{"seed": row["seed"], "arm": row["arm"],
                    "checkpoint_sha256": row["checkpoint_sha256"]} for row in per_run],
                "elapsed_seconds": time.monotonic() - started,
                "work": budget.as_dict(), "optimizer_steps": budget.update_steps,
                "backward_calls": budget.backward_calls}, refuse_existing=True)
        raise
    with progress_path.open("a", encoding="utf-8", newline="\n") as progress:
        progress.write(json.dumps({"schema": "clinc-v8-test-evaluation-progress-v1",
            "status": "complete", "completed_checkpoints": len(per_run),
            "optimizer_steps": budget.update_steps, "backward_calls": budget.backward_calls,
            "work": budget.as_dict()}, sort_keys=True, allow_nan=False) + "\n")
        progress.flush()
        import os
        os.fsync(progress.fileno())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("access-plan", "evaluate"), required=True)
    for name in ("project-root", "model-path", "master-config", "training-pack",
                 "foundation-dir", "run-root", "seal-path", "evaluation-contract"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--test-path")
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)
    if args.mode == "access-plan":
        gate = verify_release_gate(project_root=args.project_root, model_path=args.model_path,
            master_config=args.master_config, training_pack=args.training_pack,
            foundation_dir=args.foundation_dir, run_root=args.run_root, seal_path=args.seal_path,
            evaluation_contract=args.evaluation_contract)
        print(json.dumps(access_plan(gate), sort_keys=True, indent=2))
        return
    if not args.test_path or not args.output_path:
        parser.error("evaluate mode requires --test-path and --output-path")
    result = evaluate(project_root=args.project_root, model_path=args.model_path,
        master_config=args.master_config, training_pack=args.training_pack,
        foundation_dir=args.foundation_dir, run_root=args.run_root,
        seal_path=args.seal_path, evaluation_contract=args.evaluation_contract,
        test_path=args.test_path, output_path=args.output_path)
    print(json.dumps({"status": result["status"], "output_path": args.output_path,
                      "full_backbone_forward_calls": result["work"]["full_backbone_forward_calls"]}, sort_keys=True))


if __name__ == "__main__":
    main()
