"""Bounded task-ID router control over sealed v5 adapter checkpoints.

This is an exploratory capacity/routing control, not a Sleep method. It fits
one fixed ridge group gate on existing training rows and reports validation
aggregates only. It never reads confirmation/test records or writes model
weights, predictions, features, token arrays, or raw record identifiers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import agnews_v3_common as c
from nested_sleep.lm import execution_state, immutable_versions

SEEDS = (17, 29, 43)
ARMS = (
    "warm_CMS_reference", "warm_anchorKD", "warm_anchorGold",
    "warm_anchorKD_sleepP16_latest", "warm_anchorKD_sleepP16_warmmix",
    "warm_self_sleepP16_latest", "warm_self_sleepP16_warmmix",
)
EM = c.EM
METRIC_KEYS = (
    EM, "marker_accuracy", "class_given_marker_accuracy", "candidate_accuracy",
    "marker_nll", "class_nll", "answer_nll_nats_per_token",
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return c.sha(path)


def _safe_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise AssertionError(f"checkpoint escapes its sealed directory: {relative}")
    return path


def _validate_sealed_inputs(records_path: Path, warm_dir: Path, v5_run_dir: Path):
    """Validate every provenance/endpoint lock before model construction."""
    pack = c.load_pack(records_path)
    warm_path = warm_dir / "transfer-result.json"
    protocol_path = v5_run_dir / "protocol.json"
    seal_path = v5_run_dir / "development-sealed.json"
    warm, protocol, seal = _json(warm_path), _json(protocol_path), _json(seal_path)

    assert warm.get("all3_qualified") is True and warm.get("confirmation_scored") is False
    assert len(warm.get("qualified_models", [])) == 3
    assert {row["seed"] for row in warm["qualified_models"]} == set(SEEDS)
    assert seal.get("schema") == "agnews-anchored-development-seal-v5"
    assert seal.get("complete") is True
    assert seal.get("confirmation_scored") is False and seal.get("no_holdout_input") is True
    assert seal.get("protocol_sha256") == _sha(protocol_path)
    assert protocol.get("schema") == "agnews-anchored-comparison-protocol-v5"
    assert protocol.get("confirmation_input") is False and protocol.get("confirmation_scored") is False
    assert protocol.get("record_pack_sha256") == _sha(records_path) == c.DATA_SHA
    assert protocol.get("warm_result_sha256") == _sha(warm_path)
    assert protocol.get("seeds") == list(SEEDS) and protocol.get("arms") == list(ARMS)
    assert protocol.get("primary_steps") == 448
    assert seal.get("source_sha256") == protocol.get("source_sha256")

    locked_sources = {
        "driver": c.ROOT / "tools" / "run_agnews_anchored_comparison_v5.py",
        "common": Path(c.__file__),
        "v4_boundary": c.ROOT / "tools" / "agnews_preupdate_sleep_v4.py",
        "wake_helper": c.ROOT / "tools" / "agnews_anchored_wake_v5.py",
        "v4_engine": Path(c.engine.__file__),
    }
    expected_source = protocol.get("source_sha256", {})
    assert set(expected_source) == set(locked_sources)
    assert all(expected_source[key] == _sha(path) for key, path in locked_sources.items())

    runs = seal.get("runs")
    assert isinstance(runs, list) and len(runs) == len(SEEDS) * len(ARMS)
    by_key = {}
    for row in runs:
        key = (row.get("seed"), row.get("arm"))
        assert key not in by_key and key[0] in SEEDS and key[1] in ARMS
        assert row.get("status") == "complete" and row.get("primary_updates") == 448
        assert row.get("final_fingerprint")
        by_key[key] = row
    assert set(by_key) == {(seed, arm) for seed in SEEDS for arm in ARMS}
    assert seal.get("actual_primary_updates") == 21 * 448

    warm_by_seed = {row["seed"]: row for row in warm["qualified_models"]}
    refs = {}
    for seed in SEEDS:
        warm_row = warm_by_seed[seed]
        warm_cp = _safe_member(warm_dir, warm_row["checkpoint"]["path"])
        assert _sha(warm_cp) == warm_row["checkpoint"]["sha256"]
        main_row = by_key[(seed, "warm_CMS_reference")]
        cp_rows = {row["primary_step"]: row for row in main_row.get("checkpoints", [])}
        assert 0 in cp_rows and 448 in cp_rows
        endpoint = _safe_member(v5_run_dir, cp_rows[448]["path"])
        assert _sha(endpoint) == cp_rows[448]["sha256"]
        assert cp_rows[448].get("adapter_fingerprint") == main_row.get("final_fingerprint")
        refs[seed] = {
            "warm_path": warm_cp,
            "warm_sha256": warm_row["checkpoint"]["sha256"],
            "warm_metadata": warm_row,
            "main_row": main_row,
            "new_path": endpoint,
            "new_sha256": cp_rows[448]["sha256"],
            "new_metadata": cp_rows[448],
        }
    return pack, warm, protocol, seal, by_key, refs


def fit_group_ridge(train_features, train_classes, regularization=1.0):
    """Dual ridge gate: fixed old=-1/new=+1, z-normalized by train only."""
    x = np.asarray(train_features, dtype=np.float64)
    classes = np.asarray(train_classes, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] != classes.shape[0] or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("expected a nonempty N×D feature matrix and N class labels")
    if not np.isfinite(x).all() or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("ridge inputs must be finite and regularization positive")
    if not set(np.unique(classes)).issubset({0, 1, 2, 3}) or not ({0, 1} <= set(classes) and {2, 3} <= set(classes)):
        raise ValueError("training gate requires examples from all four task classes")
    mean = x.mean(axis=0, dtype=np.float64)
    scale = np.maximum(x.std(axis=0, ddof=0, dtype=np.float64), 1e-6)
    z = (x - mean) / scale / math.sqrt(x.shape[1])
    targets = np.where(classes < 2, -1.0, 1.0)
    gram = z @ z.T
    alpha = np.linalg.solve(gram + float(regularization) * np.eye(len(x), dtype=np.float64), targets)
    weight = z.T @ alpha
    if not (np.isfinite(mean).all() and np.isfinite(scale).all() and np.isfinite(weight).all()):
        raise ValueError("nonfinite ridge fit")
    payload = np.concatenate((mean, scale, weight)).astype("<f8", copy=False).tobytes()
    return {
        "mean": mean, "scale": scale, "weight": weight, "regularization": float(regularization),
        "n_train": int(len(x)), "feature_dim": int(x.shape[1]),
        "weight_l2": float(np.linalg.norm(weight)), "alpha_l2": float(np.linalg.norm(alpha)),
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "normalizer_sha256": hashlib.sha256(np.concatenate((mean, scale)).astype("<f8", copy=False).tobytes()).hexdigest(),
    }


def predict_group_ridge(features, gate):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != gate["feature_dim"] or not np.isfinite(x).all():
        raise ValueError("feature matrix does not match fitted gate")
    z = (x - gate["mean"]) / gate["scale"] / math.sqrt(gate["feature_dim"])
    return z @ gate["weight"] > 0.0


def aggregate_rows(rows):
    """Aggregate transient per-row metrics; never emit row IDs or predictions."""
    groups = {"all": {0, 1, 2, 3}, "old": {0, 1}, "new": {2, 3},
              **{f"class{cls}": {cls} for cls in range(4)}}
    out = {}
    for name, classes in groups.items():
        subset = [row for row in rows if row["class_id"] in classes]
        if not subset:
            continue
        out[name] = {"examples": len(subset)}
        for key in METRIC_KEYS:
            out[name][key] = float(sum(float(row[key]) for row in subset) / len(subset))
    return out


def gate_confusion(classes, predicted_new):
    classes = np.asarray(classes, dtype=np.int64)
    predicted_new = np.asarray(predicted_new, dtype=bool)
    assert classes.shape == predicted_new.shape
    actual_new = classes >= 2
    return {
        "actual_old_pred_old": int(np.sum((~actual_new) & (~predicted_new))),
        "actual_old_pred_new": int(np.sum((~actual_new) & predicted_new)),
        "actual_new_pred_old": int(np.sum(actual_new & (~predicted_new))),
        "actual_new_pred_new": int(np.sum(actual_new & predicted_new)),
    }


def score_rows(model, records, pack, seed, stage, meter, capture_features=False, check=None):
    """Mirror c.score with a hook on the original final-MLP output."""
    meter.stage = stage
    data, features = [], []
    code = pack["code_maps"][str(seed)]
    labels4 = code["class_to_token_ids"]
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    captured = []

    def hook(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.detach())

    handle = None
    if capture_features:
        original = model.backbone.model.layers[-1].mlp.original
        handle = original.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for offset in range(0, len(records), 8):
                if check is not None:
                    check()
                batch = [{**row, "marker": code["marker_token_id"], "label": labels4[row["class_id"]]}
                         for row in records[offset:offset + 8]]
                ids, mask, lengths, markers, labels = c.engine.shared.collate(
                    batch, device, pack["tokenizer"]["pad_token_id"])
                ids, mask = c.engine.shared.append_markers(
                    ids, mask, lengths, markers, pack["tokenizer"]["pad_token_id"])
                captured.clear()
                logits = model.logits(ids, mask)
                indices = torch.arange(len(batch), device=device)
                first, second = logits[indices, lengths - 1].float(), logits[indices, lengths].float()
                marker_nll = F.cross_entropy(first, markers, reduction="none").cpu().tolist()
                class_nll = F.cross_entropy(second, labels, reduction="none").cpu().tolist()
                marker_ok = (first.argmax(-1) == markers).cpu().tolist()
                class_ok = (second.argmax(-1) == labels).cpu().tolist()
                candidate = second[:, labels4].argmax(-1).cpu().tolist()
                for row, mnll, cnll, mok, cok, pred in zip(batch, marker_nll, class_nll,
                                                            marker_ok, class_ok, candidate):
                    data.append({"class_id": int(row["class_id"]), EM: int(mok and cok),
                                 "marker_accuracy": int(mok), "class_given_marker_accuracy": int(cok),
                                 "candidate_accuracy": int(pred == row["class_id"]),
                                 "marker_nll": float(mnll), "class_nll": float(cnll),
                                 "answer_nll_nats_per_token": float((mnll + cnll) / 2)})
                if capture_features:
                    if len(captured) != 1:
                        raise AssertionError(f"expected one original-MLP hook call, got {len(captured)}")
                    raw = captured[0]
                    prefix_index = (lengths - 1).to(raw.device)
                    rows_index = torch.arange(len(batch), device=raw.device)
                    features.extend(raw[rows_index, prefix_index].float().cpu().numpy().astype(np.float32, copy=False))
    finally:
        if handle is not None:
            handle.remove()
    metrics = aggregate_rows(data)
    if capture_features and len(features) != len(records):
        raise AssertionError("feature hook did not return one vector per scored row")
    return metrics, data, np.asarray(features, dtype=np.float32) if capture_features else None


def _compare_reference(observed, expected, label):
    if set(observed) != set(expected):
        raise AssertionError(f"{label}: metric groups differ")
    for group in observed:
        if observed[group]["examples"] != expected[group]["examples"]:
            raise AssertionError(f"{label}/{group}: denominator differs")
        for key in METRIC_KEYS:
            if observed[group][key] != float(expected[group][key]):
                raise AssertionError(f"{label}/{group}/{key}: score differs from sealed validation")


def _readonly_branch_state(model):
    return {
        "mutable_fingerprint": c.fingerprint(model),
        "execution_state": execution_state(model),
        "immutable_versions": immutable_versions(model),
    }


def _assert_readonly_branch_state(model, before, label):
    after = _readonly_branch_state(model)
    if after != before:
        raise AssertionError(f"read-only branch mutated {label}: model/policy/immutable versions changed")


def _selected_rows(warm_rows, new_rows, choose_new):
    if len(warm_rows) != len(new_rows) or len(warm_rows) != len(choose_new):
        raise AssertionError("branch outputs and gate decisions must align")
    return [new if use_new else old for old, new, use_new in zip(warm_rows, new_rows, choose_new)]


def _write(path: Path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(args):
    if args.out_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.out_dir}")
    start = time.monotonic()
    pack, warm, main_protocol, seal, runs, refs = _validate_sealed_inputs(
        args.records, args.warm_dir, args.v5_run_dir)
    c.preflight(args.model_dir, pack)  # hashes only; no model construction before sealed-input checks
    args.out_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "agnews-task-router-control-v5",
        "status": "exploratory_capacity_routing_control",
        "records_sha256": _sha(args.records),
        "warm_result_sha256": _sha(args.warm_dir / "transfer-result.json"),
        "main_protocol_sha256": _sha(args.v5_run_dir / "protocol.json"),
        "main_seal_sha256": _sha(args.v5_run_dir / "development-sealed.json"),
        "main_source_sha256": main_protocol["source_sha256"],
        "locked_source_sha256": {
            "task_router_driver": _sha(Path(__file__).resolve()),
            "main_driver": _sha(c.ROOT / "tools" / "run_agnews_anchored_comparison_v5.py"),
            "common": _sha(Path(c.__file__)),
            "wake_helper": _sha(c.ROOT / "tools" / "agnews_anchored_wake_v5.py"),
            "v4_boundary": _sha(c.ROOT / "tools" / "agnews_preupdate_sleep_v4.py"),
            "engine": _sha(Path(c.engine.__file__)),
            "core": _sha(c.ROOT / "src" / "nested_sleep" / "lifecycle.py"),
        },
        "checkpoint_sha256_by_seed": {
            str(seed): {"warm": refs[seed]["warm_sha256"], "new_CMS_reference_448": refs[seed]["new_sha256"]}
            for seed in SEEDS
        },
        "model_assets": {
            "revision": c.engine.MODEL_REVISION,
            "weight_sha256": dict(c.engine.WEIGHT_SHA256),
            "tokenizer_file_sha256": {x["name"]: x["sha256"] for x in pack["tokenizer"]["file_inventory"]},
        },
        "environment": {"python": sys.version, "torch": str(torch.__version__),
                        "numpy": str(np.__version__), "platform": platform.platform(),
                        "cuda_build": torch.version.cuda},
        "driver_sha256": _sha(Path(__file__).resolve()),
        "seeds": list(SEEDS), "validation_rows": 512, "training_rows": 1024,
        "training_group_label_uses": 1024 * len(SEEDS),
        "ridge": {"solver": "dual float64", "targets": {"old": -1, "new": 1},
                  "regularization": 1.0, "normalizer": "training mean/std(ddof=0), std floor 1e-6, then divide sqrt(d)",
                  "decision": "new iff score > 0; no hyperparameter search"},
        "conditions": ["warm_alone", "new_CMS_reference_alone", "learned_feature_gate", "task_ID_diagnostic"],
        "feature_hook": "backbone.model.layers[-1].mlp.original output at input prefix position len(tokens)-1; before memory adapters",
        "task_ID_diagnostic_note": "group-level old/new task identity selects a branch; diagnostic selector, not a deployable method or guaranteed upper bound",
        "branch_cost_note": "Both complete model branches are evaluated for every row; the post-hoc selector does not share inference computation. Ridge is fitted once per seed on 1024 training group labels. The training gate fit/evaluation is exploratory and not a free performance gain.",
        "confirmation_input": False, "confirmation_scored": False,
        "weights_or_row_predictions_written": False,
        "estimated_forward_calls": 1152,
        "estimated_forward_input_tokens": None,
        "elapsed_limit_seconds": 900, "time_enforcement": "cooperative checks between batches and seed transitions",
        "no_optimizer_updates": True, "no_gradient_forwards": True,
        "clock_started_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write(args.out_dir / "plan.json", protocol)
    meter = c.engine.WorkMeter()
    meter_token = c.engine.ACTIVE_METER.set(meter)
    result_seeds = []

    def check_time():
        if time.monotonic() - start >= 900:
            raise TimeoutError("task-router control reached cooperative 900-second limit")

    try:
        for seed in SEEDS:
            check_time()
            model = c.model(args.model_dir, seed)
            check_time()
            warm_cp = torch.load(refs[seed]["warm_path"], map_location="cpu", weights_only=False)
            new_cp = torch.load(refs[seed]["new_path"], map_location="cpu", weights_only=False)
            train_rows = c.rows(pack, seed, role="train")
            validation_rows = c.rows(pack, seed, role="validation")

            c.restore(model, warm_cp)
            if c.fingerprint(model) != refs[seed]["warm_metadata"]["checkpoint"]["adapter_fingerprint"]:
                raise AssertionError(f"seed{seed}: restored warm adapter differs from qualified warm reference")
            warm_guard = _readonly_branch_state(model)
            warm_train, warm_train_rows, train_features = score_rows(
                model, train_rows, pack, seed, f"seed{seed}_warm_train", meter, capture_features=True,
                check=check_time)
            warm_val, warm_val_rows, val_features = score_rows(
                model, validation_rows, pack, seed, f"seed{seed}_warm_validation", meter, capture_features=True,
                check=check_time)
            warm_main_rows = {x["primary_step"]: x for x in refs[seed]["main_row"]["validation"]}
            _compare_reference(warm_val, warm_main_rows[0]["metrics"], f"seed{seed} warm step0")
            _assert_readonly_branch_state(model, warm_guard, f"seed{seed} warm branch")

            c.restore(model, new_cp)
            if c.fingerprint(model) != refs[seed]["new_metadata"]["adapter_fingerprint"]:
                raise AssertionError(f"seed{seed}: restored CMS endpoint differs from sealed adapter fingerprint")
            new_guard = _readonly_branch_state(model)
            new_train, new_train_rows, _ = score_rows(
                model, train_rows, pack, seed, f"seed{seed}_new_train", meter, check=check_time)
            new_val, new_val_rows, _ = score_rows(
                model, validation_rows, pack, seed, f"seed{seed}_new_validation", meter, check=check_time)
            _compare_reference(new_val, warm_main_rows[448]["metrics"], f"seed{seed} CMS reference step448")
            _assert_readonly_branch_state(model, new_guard, f"seed{seed} new CMS branch")

            train_classes = np.asarray([row["class_id"] for row in train_rows], dtype=np.int64)
            val_classes = np.asarray([row["class_id"] for row in validation_rows], dtype=np.int64)
            gate = fit_group_ridge(train_features, train_classes, regularization=1.0)
            train_new = predict_group_ridge(train_features, gate)
            val_new = predict_group_ridge(val_features, gate)
            task_train_new = train_classes >= 2
            task_val_new = val_classes >= 2

            train_conditions = {
                "warm_alone": warm_train_rows,
                "new_CMS_reference_alone": new_train_rows,
                "learned_feature_gate": _selected_rows(warm_train_rows, new_train_rows, train_new),
                "task_ID_diagnostic": _selected_rows(warm_train_rows, new_train_rows, task_train_new),
            }
            val_conditions = {
                "warm_alone": warm_val_rows,
                "new_CMS_reference_alone": new_val_rows,
                "learned_feature_gate": _selected_rows(warm_val_rows, new_val_rows, val_new),
                "task_ID_diagnostic": _selected_rows(warm_val_rows, new_val_rows, task_val_new),
            }
            result_seeds.append({
                "seed": seed,
                "warm_reference_metrics": {"train": warm_train, "validation": warm_val},
                "new_CMS_reference_metrics": {"train": new_train, "validation": new_val},
                "conditions": {
                    "train": {key: aggregate_rows(rows) for key, rows in train_conditions.items()},
                    "validation": {key: aggregate_rows(rows) for key, rows in val_conditions.items()},
                },
                "gate": {
                    "model_sha256": gate["model_sha256"],
                    "normalizer_sha256": gate["normalizer_sha256"],
                    "weight_l2": gate["weight_l2"], "alpha_l2": gate["alpha_l2"],
                    "feature_dim": gate["feature_dim"], "regularization": gate["regularization"],
                    "training_confusion": gate_confusion(train_classes, train_new),
                    "validation_confusion": gate_confusion(val_classes, val_new),
                "learned_selector_uses_validation_labels": False,
                "task_ID_diagnostic_is_guaranteed_upper_bound": False,
                },
                "branch_endpoint_sha256": {"warm": refs[seed]["warm_sha256"], "new_CMS_reference": refs[seed]["new_sha256"]},
                "checkpoint_metric_match": {"warm_step0": True, "new_CMS_step448": True},
            })
            del model, warm_cp, new_cp, train_features, val_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            check_time()
            print({"seed": seed, "stage": "complete", "forward_calls": sum(v["forward_calls"] for v in meter.counts.values())}, flush=True)

        assert len(result_seeds) == 3
        check_time()
        actual_calls = sum(v["forward_calls"] for v in meter.counts.values())
        assert actual_calls == 1152, f"expected 1152 full-model batch forwards, got {actual_calls}"
        assert all(v["gradient_forward_input_tokens"] == 0 for v in meter.counts.values())
        _write(args.out_dir / "result.json", {
            "schema": "agnews-task-router-control-result-v5", "complete": True,
            "source_sha256": _sha(Path(__file__).resolve()),
            "plan_sha256": _sha(args.out_dir / "plan.json"),
            "seeds": result_seeds, "work": meter.counts,
            "actual_forward_calls": actual_calls,
            "actual_optimizer_updates": 0, "actual_gradient_forward_calls": 0,
            "confirmation_scored": False, "raw_rows_predictions_features_written": False,
            "elapsed_seconds": time.monotonic() - start,
        })
        print("TASK_ROUTER_CONTROL_COMPLETE", flush=True)
    except BaseException as error:
        if not (args.out_dir / "failure.json").exists():
            _write(args.out_dir / "failure.json", {
                "schema": "agnews-task-router-control-failure-v5",
                "error_type": type(error).__name__, "error": str(error),
                "source_sha256": _sha(Path(__file__).resolve()),
                "plan_sha256": _sha(args.out_dir / "plan.json"),
                "elapsed_seconds": time.monotonic() - start,
                "completed_seed_aggregates": result_seeds,
                "work": meter.counts, "actual_optimizer_updates": 0,
                "confirmation_scored": False, "automatic_retry": False,
            })
        raise
    finally:
        c.engine.ACTIVE_METER.reset(meter_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("records", "warm-dir", "v5-run-dir", "model-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
