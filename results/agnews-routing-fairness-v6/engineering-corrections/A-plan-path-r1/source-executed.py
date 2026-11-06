"""C-only offline scoring of shared-backbone, hard-routed AG News contexts.

The initial phase compares the qualified warm adapter to the v5 CMS endpoint.
An optional continuation phase scores all six completed v6 endpoints against
the same warm context. No optimizer step is performed. Routed selectors receive
only the original final-MLP activation and prefix lengths; task labels are used
only for the explicitly diagnostic task-ID condition and metric aggregation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"tools"))
import agnews_v3_common as c
import probe_agnews_task_router_v5 as v5router
from late_context_router_v6 import FrozenRidgeGate, LateContextRouter, make_gate_mask
from nested_sleep.lm import execution_state, immutable_versions

SEEDS = (17, 29, 43)
CONDITIONS = (
    "warm_alone", "new_CMS_reference_alone", "correct_phase_ridge",
    "permuted_phase_ridge", "hash_random", "confidence_marker",
    "confidence_selfjoint", "task_ID_DIAGNOSTIC",
)
PHYSICAL_CONDITIONS = (
    "constant_old", "constant_new", "correct_phase_ridge",
    "permuted_phase_ridge", "hash_random", "uniform_context_mix",
)
METRICS = v5router.METRIC_KEYS
ELAPSED_LIMIT_SECONDS = 1800
PERMUTATION_SEED = "nl-sleep-v6-permuted-phase-labels:6329"
HASH_RANDOM_PREFIX = "nl-sleep-v6-random:"
EXPECTED_V5_ROUTER_RESULT_SHA256 = "1763a4b50e5e3fa75d3cb949d96d09092b8790f08c187e72304a6f9c6421190a"
EXPECTED_ROUTER_HELPER_SHA256 = "fa549672ed143d161930699de8a22e6cbe99404ae9f7b00ac30f75b1ed53d76d"


def sha(path: Path) -> str:
    return c.sha(Path(path))


def write_json(path: Path, value) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def phase_targets(phases):
    """Encode task phase only; class ids are deliberately not accepted here."""
    values = list(phases)
    if not values or set(values) != {"old", "new"}:
        raise ValueError("ridge fitting needs both old and new phase rows")
    return np.asarray([-1.0 if phase == "old" else 1.0 for phase in values], dtype=np.float64)


def fit_phase_ridge(features, phases, regularization: float = 1.0):
    """Fixed train-only dual ridge gate; no class labels or validation labels."""
    x = np.asarray(features, dtype=np.float64)
    y = phase_targets(phases)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("expected finite N×D features aligned to phase labels")
    if not np.isfinite(x).all() or not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("ridge data and positive regularization must be finite")
    mean = x.mean(axis=0, dtype=np.float64)
    scale = np.maximum(x.std(axis=0, ddof=0, dtype=np.float64), 1e-6)
    z = (x - mean) / scale / math.sqrt(x.shape[1])
    alpha = np.linalg.solve(z @ z.T + regularization * np.eye(len(x), dtype=np.float64), y)
    weight = z.T @ alpha
    if not (np.isfinite(mean).all() and np.isfinite(scale).all() and np.isfinite(weight).all()):
        raise ValueError("nonfinite ridge parameters")
    packed = np.concatenate((mean, scale, weight)).astype("<f8", copy=False).tobytes()
    norm = np.concatenate((mean, scale)).astype("<f8", copy=False).tobytes()
    return {"mean": mean, "scale": scale, "weight": weight, "bias": 0.0,
            "regularization": float(regularization), "n_train": int(len(x)),
            "feature_dim": int(x.shape[1]), "model_sha256": hashlib.sha256(packed).hexdigest(),
            "normalizer_sha256": hashlib.sha256(norm).hexdigest(),
            "weight_l2": float(np.linalg.norm(weight)), "alpha_l2": float(np.linalg.norm(alpha))}


def ridge_mask(features, gate):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != gate["feature_dim"] or not np.isfinite(x).all():
        raise ValueError("feature matrix does not match fixed gate")
    z = (x - gate["mean"]) / gate["scale"] / math.sqrt(gate["feature_dim"])
    scores = z @ gate["weight"] + gate["bias"]
    # Strictly positive selects new; exact ties (including a zero signal) select old.
    return scores > 0.0


def permuted_phase_targets(phases):
    targets = list(phase_targets(phases))
    random.Random(PERMUTATION_SEED).shuffle(targets)
    return np.asarray(targets,dtype=np.float64)


def permuted_phase_fit(features, phases):
    targets = permuted_phase_targets(phases)
    labels = ["old" if value < 0 else "new" for value in targets]
    return fit_phase_ridge(features, labels), hashlib.sha256(
        np.asarray(targets, dtype="<f8").tobytes()).hexdigest()


def hash_random_mask(records):
    out = []
    for row in records:
        token_text = ",".join(str(int(x)) for x in row["tokens"])
        bit = hashlib.sha256((HASH_RANDOM_PREFIX + token_text).encode("ascii")).digest()[0] & 1
        out.append(bool(bit))
    return np.asarray(out, dtype=bool)


def metric_rows_from_logits(logits, batch_rows, lengths, marker_ids, label_ids, all_label_ids):
    """Transient row metrics and selector confidences; never serializes logits."""
    idx = torch.arange(len(batch_rows), device=logits.device)
    first = logits[idx, lengths - 1].float()
    second = logits[idx, lengths].float()
    log_first, log_second = F.log_softmax(first, dim=-1), F.log_softmax(second, dim=-1)
    marker_logp = log_first.gather(1, marker_ids[:, None]).squeeze(1)
    selfjoint = marker_logp + log_second.max(dim=-1).values
    first_pred, second_pred = first.argmax(-1), second.argmax(-1)
    m_nll = F.cross_entropy(first, marker_ids, reduction="none")
    c_nll = F.cross_entropy(second, label_ids, reduction="none")
    code_ids = torch.as_tensor(all_label_ids, dtype=torch.long, device=logits.device)
    candidate = second[:, code_ids].argmax(-1)
    raw = []
    for i, row in enumerate(batch_rows):
        marker_ok = bool(first_pred[i] == marker_ids[i])
        class_ok = bool(second_pred[i] == label_ids[i])
        raw.append({"class_id": int(row["class_id"]), "phase": str(row["phase"]),
                    c.EM: int(marker_ok and class_ok), "marker_accuracy": int(marker_ok),
                    "class_given_marker_accuracy": int(class_ok),
                    "candidate_accuracy": int(candidate[i] == int(row["class_id"])),
                    "marker_nll": float(m_nll[i]), "class_nll": float(c_nll[i]),
                    "answer_nll_nats_per_token": float((m_nll[i] + c_nll[i]) / 2),
                    "marker_logp": float(marker_logp[i]), "selfjoint_logp": float(selfjoint[i])})
    return raw


def aggregate_row_metrics(rows):
    groups = {"all": {0, 1, 2, 3}, "old": {0, 1}, "new": {2, 3},
              **{f"class{i}": {i} for i in range(4)}}
    out = {}
    for group, classes in groups.items():
        subset = [r for r in rows if r["class_id"] in classes]
        if not subset:
            continue
        out[group] = {"examples": len(subset)}
        for key in METRICS:
            out[group][key] = float(sum(float(r[key]) for r in subset) / len(subset))
    return out


def select_row_metrics(old_rows, new_rows, choose_new):
    if len(old_rows) != len(new_rows) or len(old_rows) != len(choose_new):
        raise ValueError("row metric branches and selector mask differ in length")
    return [dict(new if use_new else old) for old, new, use_new in zip(old_rows, new_rows, choose_new)]


def compare_metrics(left, right, *, nll_tolerance=1e-5, label="metrics"):
    if set(left) != set(right):
        raise AssertionError(f"{label}: metric groups differ")
    for group in left:
        if left[group]["examples"] != right[group]["examples"]:
            raise AssertionError(f"{label}/{group}: denominator mismatch")
        if left[group][c.EM] != right[group][c.EM]:
            raise AssertionError(f"{label}/{group}: greedy EM mismatch")
        for key in ("marker_nll", "class_nll", "answer_nll_nats_per_token"):
            if abs(float(left[group][key]) - float(right[group][key])) > nll_tolerance:
                raise AssertionError(f"{label}/{group}/{key}: difference exceeds {nll_tolerance}")


def compare_exact_metrics(observed, expected, label="sealed metrics"):
    if set(observed)!=set(expected): raise AssertionError(f"{label}: metric groups differ")
    for group in observed:
        if observed[group].get("examples")!=expected[group].get("examples"):
            raise AssertionError(f"{label}/{group}: denominator mismatch")
        for key in METRICS:
            if float(observed[group][key])!=float(expected[group][key]):
                raise AssertionError(f"{label}/{group}/{key}: metric differs from locked score")


def score_full_branch(model, records, pack, seed, stage, meter, *, capture_features, check_time):
    meter.stage = stage
    code = pack["code_maps"][str(seed)]
    label_map = [int(x) for x in code["class_to_token_ids"]]
    device = next(model.parameters()).device
    model.eval().requires_grad_(False)
    before = {"fingerprint": c.fingerprint(model), "execution": execution_state(model),
              "immutable": immutable_versions(model),
              "rng": c.engine.shared.tensor_tree_hash(c.engine.capture_rng())}
    features, result = [], []
    captured = []
    handle = None
    if capture_features:
        module = model.backbone.model.layers[-1].mlp.original
        def hook(_module, _args, output):
            captured.append((output[0] if isinstance(output, tuple) else output).detach())
        handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for offset in range(0, len(records), 8):
                check_time()
                raw_batch = records[offset:offset + 8]
                batch = [{**r, "marker": int(code["marker_token_id"]),
                          "label": label_map[int(r["class_id"])]} for r in raw_batch]
                ids, mask, lengths, markers, labels = c.engine.shared.collate(
                    batch, device, pack["tokenizer"]["pad_token_id"])
                ids, mask = c.engine.shared.append_markers(
                    ids, mask, lengths, markers, pack["tokenizer"]["pad_token_id"])
                captured.clear()
                logits = model.logits(ids, mask)
                result.extend(metric_rows_from_logits(logits, batch, lengths, markers, labels, label_map))
                if capture_features:
                    if len(captured) != 1:
                        raise AssertionError(f"expected one original-MLP hook, received {len(captured)}")
                    act = captured[0]
                    pos = (lengths - 1).to(act.device)
                    rowidx = torch.arange(len(batch), device=act.device)
                    features.extend(act[rowidx, pos].float().cpu().numpy().astype(np.float32, copy=False))
    finally:
        if handle is not None:
            handle.remove()
    after = {"fingerprint": c.fingerprint(model), "execution": execution_state(model),
             "immutable": immutable_versions(model),
             "rng": c.engine.shared.tensor_tree_hash(c.engine.capture_rng())}
    if before != after:
        raise AssertionError(f"scoring mutated model/RNG state at {stage}")
    if len(result) != len(records) or capture_features and len(features) != len(records):
        raise AssertionError("full-branch scoring returned unexpected row/feature count")
    return result, np.asarray(features, dtype=np.float32) if capture_features else None


def _token_batch(pack, seed, batch_rows):
    code = pack["code_maps"][str(seed)]
    labels = [int(x) for x in code["class_to_token_ids"]]
    batch = [{**r, "marker": int(code["marker_token_id"]), "label": labels[int(r["class_id"])]}
             for r in batch_rows]
    device = "cuda"
    ids, mask, lengths, markers, target = c.engine.shared.collate(
        batch, device, pack["tokenizer"]["pad_token_id"])
    ids, mask = c.engine.shared.append_markers(ids, mask, lengths, markers, pack["tokenizer"]["pad_token_id"])
    return ids, mask, lengths, markers, target


class RecordingSelector:
    """Capture only the callback's bool decisions to verify live-mask parity."""
    def __init__(self, gate):
        self.gate=gate;self.calls=0;self._parts=[]
    def __call__(self, original_mlp_output, prefix_lengths):
        selected=self.gate(original_mlp_output,prefix_lengths)
        self._parts.append(torch.as_tensor(selected,dtype=torch.bool).detach().cpu().reshape(-1).numpy())
        self.calls+=1
        return selected
    @property
    def selected(self):
        return np.concatenate(self._parts) if self._parts else np.asarray([],dtype=bool)


def parity_fixture(model, old_cp, new_cp, records, pack, seed, meter):
    """Four forwards total on balanced TRAIN rows: two branches, both dedup modes."""
    batch_rows = []
    for cls in range(4):
        batch_rows.extend([r for r in records if int(r["class_id"]) == cls][:2])
    if len(batch_rows) != 8 or [sum(r["class_id"] == k for r in batch_rows) for k in range(4)] != [2]*4:
        raise AssertionError("parity fixture requires two train rows from every class")
    ids, mask, _, _, _ = _token_batch(pack, seed, batch_rows)
    model.eval().requires_grad_(False)
    meter.stage = f"seed{seed}_train_only_routing_parity"
    with torch.no_grad():
        c.restore(model, old_cp)
        old_logits = model.logits(ids, mask).detach()
        new_state = new_cp["model"]
        new_exec = new_cp["execution_state"]
        c.restore(model, new_cp)
        new_logits = model.logits(ids, mask).detach()
        c.restore(model, old_cp)
        if c.engine.shared.tensor_tree_hash(model.adapter_state_dict()) != c.engine.shared.tensor_tree_hash(old_cp["model"]):
            raise AssertionError("warm adapter bytes must be restored before helper construction")
        mask_new = torch.tensor([r["phase"] == "new" for r in batch_rows], dtype=torch.bool, device=ids.device)
        if execution_state(model) != old_cp["execution_state"]:
            raise AssertionError("warm policy must be restored before helper construction")
        selected = torch.where(mask_new[:, None, None], new_logits, old_logits)
        outcomes = {}
        for dedup in (False, True):
            router = LateContextRouter(model, new_state, new_exec, old_execution_state=execution_state(model),
                                       deduplicate_equal=dedup)
            meter.charge(ids)
            routed = router.logits(ids, mask_new, attention_mask=mask).detach()
            max_abs = float((routed.float() - selected.float()).abs().max().cpu())
            greedy_equal = bool(torch.equal(routed.argmax(-1), selected.argmax(-1)))
            if max_abs > 1e-5 or not greedy_equal:
                raise AssertionError(f"dedup={dedup} routed parity failed: max_abs={max_abs}, greedy={greedy_equal}")
            outcomes[str(dedup).lower()] = {"max_abs_logit_difference":max_abs,
                                             "all_token_greedy_equal":greedy_equal}
            del router
    return {"balanced_train_rows":8,"old_rows":4,"new_rows":4,
            "deduplicate_false_and_true":outcomes,"actual_forward_calls":4,"optimizer_updates":0}


def _phase_labels(rows):
    phases = [str(r["phase"]) for r in rows]
    for row, phase in zip(rows, phases):
        if phase not in ("old", "new"):
            raise ValueError("record phase must be old/new")
        if (int(row["class_id"]) < 2) != (phase == "old"):
            raise AssertionError("class id and phase disagree in scored train/validation roles")
    return phases


def _condition_masks(records, features, gate, perm_gate, validation=False):
    return {
        "correct_phase_ridge": ridge_mask(features, gate),
        "permuted_phase_ridge": ridge_mask(features, perm_gate),
        "hash_random": hash_random_mask(records),
        "task_ID_DIAGNOSTIC": np.asarray([r["phase"] == "new" for r in records], dtype=bool),
    }


def _confidence_masks(old_rows, new_rows):
    return {
        "confidence_marker": np.asarray([n["marker_logp"] > o["marker_logp"]
                                         for o, n in zip(old_rows, new_rows)], dtype=bool),
        "confidence_selfjoint": np.asarray([n["selfjoint_logp"] > o["selfjoint_logp"]
                                            for o, n in zip(old_rows, new_rows)], dtype=bool),
    }


def posthoc_conditions(old_rows, new_rows, records, features, gate, perm_gate):
    if not (len(old_rows) == len(new_rows) == len(records) == len(features)):
        raise ValueError("posthoc rows/features are not aligned")
    masks = _condition_masks(records, features, gate, perm_gate)
    masks.update(_confidence_masks(old_rows, new_rows))
    masks["warm_alone"] = np.zeros(len(records), dtype=bool)
    masks["new_CMS_reference_alone"] = np.ones(len(records), dtype=bool)
    output = {}
    for condition in CONDITIONS:
        chosen = select_row_metrics(old_rows, new_rows, masks[condition])
        output[condition] = {"metrics": aggregate_row_metrics(chosen),
                             "gate_fraction_new": float(np.mean(masks[condition]))}
    return output, masks


def _serialize_metric_rows(rows):
    # Private, article-free cache: class/phase and aggregate-able scores only.
    return [{k: (int(v) if k == "class_id" else v) for k, v in row.items()} for row in rows]


def _save_private_gate(path, correct, permuted, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
        correct_mean=correct["mean"], correct_scale=correct["scale"], correct_weight=correct["weight"],
        permuted_mean=permuted["mean"], permuted_scale=permuted["scale"], permuted_weight=permuted["weight"],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))


def _load_private_gate(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        def gate(prefix):
            mean=z[prefix+"_mean"].copy(); scale=z[prefix+"_scale"].copy(); weight=z[prefix+"_weight"].copy()
            packed=np.concatenate((mean,scale,weight)).astype("<f8",copy=False).tobytes()
            norm=np.concatenate((mean,scale)).astype("<f8",copy=False).tobytes()
            return {"mean":mean,"scale":scale,"weight":weight,"bias":0.0,"regularization":1.0,
                    "feature_dim":len(weight),"model_sha256":hashlib.sha256(packed).hexdigest(),
                    "normalizer_sha256":hashlib.sha256(norm).hexdigest()}
        return gate("correct"), gate("permuted"), meta


def _safe_path(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise AssertionError(f"sealed checkpoint path escapes input directory: {relative}")
    return path


def load_inputs(args):
    pack=c.load_pack(args.records)
    if {str(r["role"]) for r in pack["records"]}!={"train","validation"}:
        raise AssertionError("v6 may only open the frozen train/validation pack roles")
    warm_path=args.warm_dir/"transfer-result.json"
    v5_protocol_path=args.v5_run_dir/"protocol.json"
    v5_seal_path=args.v5_run_dir/"development-sealed.json"
    warm=c.read(warm_path); protocol=c.read(v5_protocol_path); seal=c.read(v5_seal_path)
    assert warm.get("all3_qualified") is True and warm.get("confirmation_scored") is False
    assert {x["seed"] for x in warm["qualified_models"]}==set(SEEDS)
    assert protocol.get("schema")=="agnews-anchored-comparison-protocol-v5"
    assert protocol.get("confirmation_input") is False and protocol.get("confirmation_scored") is False
    assert seal.get("schema")=="agnews-anchored-development-seal-v5" and seal.get("complete") is True
    assert seal.get("confirmation_scored") is False and seal.get("no_holdout_input") is True
    assert seal.get("protocol_sha256")==sha(v5_protocol_path)
    assert protocol.get("record_pack_sha256")==sha(args.records)==c.DATA_SHA
    assert protocol.get("warm_result_sha256")==sha(warm_path)
    assert protocol.get("seeds")==list(SEEDS) and protocol.get("primary_steps")==448
    assert seal.get("source_sha256")==protocol.get("source_sha256")
    expected={
        "driver":c.ROOT/"tools"/"run_agnews_anchored_comparison_v5.py",
        "common":Path(c.__file__),"v4_boundary":c.ROOT/"tools"/"agnews_preupdate_sleep_v4.py",
        "wake_helper":c.ROOT/"tools"/"agnews_anchored_wake_v5.py","v4_engine":Path(c.engine.__file__)}
    assert protocol["source_sha256"]=={k:sha(v) for k,v in expected.items()}
    rows=seal.get("runs",[])
    by={(r.get("seed"),r.get("arm")):r for r in rows}
    expected_runs={(seed,arm) for seed in SEEDS for arm in protocol.get("arms",[])}
    if (len(rows)!=21 or len(by)!=21 or len(protocol.get("arms",[]))!=7
            or set(by)!=expected_runs or seal.get("actual_primary_updates")!=21*448
            or any(r.get("status")!="complete" or r.get("primary_updates")!=448 for r in rows)):
        raise AssertionError("v5 seal must contain all 21 complete fixed runs")
    warm_by={x["seed"]:x for x in warm["qualified_models"]}
    refs={}
    for seed in SEEDS:
        row=by[(seed,"warm_CMS_reference")]
        cps={x["primary_step"]:x for x in row["checkpoints"]}
        assert 0 in cps and 448 in cps
        warm_cp=_safe_path(args.warm_dir,warm_by[seed]["checkpoint"]["path"])
        end_cp=_safe_path(args.v5_run_dir,cps[448]["path"])
        assert sha(warm_cp)==warm_by[seed]["checkpoint"]["sha256"]
        assert sha(end_cp)==cps[448]["sha256"]
        refs[seed]={"warm_path":warm_cp,"warm_meta":warm_by[seed]["checkpoint"],
                    "warm_sha":sha(warm_cp),"v5_path":end_cp,"v5_meta":cps[448],"v5_sha":sha(end_cp)}
    router_result=c.read(args.v5_router_result)
    assert router_result.get("schema")=="agnews-task-router-control-result-v5"
    assert router_result.get("complete") is True and router_result.get("confirmation_scored") is False
    assert sha(args.v5_router_result)==args.expected_v5_router_result_sha256
    return pack,warm,protocol,seal,refs,router_result


def source_manifest(args):
    router_module=__import__("late_context_router_v6")
    return {
        "driver_sha256":sha(Path(__file__).resolve()),
        "router_helper_sha256":sha(Path(router_module.__file__)),
        "common_sha256":sha(Path(c.__file__)),"engine_sha256":sha(Path(c.engine.__file__)),
        "lifecycle_sha256":sha(c.ROOT/"src"/"nested_sleep"/"lifecycle.py"),
        "v5_router_probe_sha256":sha(c.ROOT/"tools"/"probe_agnews_task_router_v5.py"),
        "records_sha256":sha(args.records),"warm_result_sha256":sha(args.warm_dir/"transfer-result.json"),
        "v5_protocol_sha256":sha(args.v5_run_dir/"protocol.json"),
        "v5_seal_sha256":sha(args.v5_run_dir/"development-sealed.json"),
        "v5_router_result_sha256":sha(args.v5_router_result),
        "model_weights_sha256":dict(c.engine.WEIGHT_SHA256),
    }


def validate_continuation(path, refs, base_source):
    seal=c.read(path)
    protocol_path=Path(path).parent/"protocol.json"
    protocol=c.read(protocol_path)
    assert seal.get("schema")=="agnews-fair-continuation-development-seal-v6"
    assert seal.get("complete") is True and seal.get("development_only") is True
    assert seal.get("confirmation_scored") is False and seal.get("no_holdout_input") is True
    assert seal.get("protocol_sha256")==sha(protocol_path)
    assert protocol.get("schema")=="agnews-fair-continuation-protocol-v6"
    assert protocol.get("source_sha256")==seal.get("source_sha256")
    assert protocol.get("confirmation_input") is False and protocol.get("confirmation_scored") is False
    assert protocol.get("seeds")==list(SEEDS)
    assert protocol.get("arms")==["aux_new_CE","aux_old_gold_CE"]
    assert protocol.get("start_step")==448 and protocol.get("end_step")==1024
    assert protocol.get("evaluation_steps")==[448,512,768,1024]
    assert protocol.get("planned_primary_updates")==2*3*576
    assert seal.get("actual_primary_updates")==2*3*576
    source=seal.get("source_sha256",{})
    source_map={"records_sha256":"records_sha256","warm_result_sha256":"warm_result_sha256",
                "v5_development_seal_sha256":"v5_seal_sha256","v5_protocol_sha256":"v5_protocol_sha256"}
    for actual,expected in source_map.items():
        if source.get(actual)!=base_source.get(expected):
            raise AssertionError(f"continuation seal {actual} does not match locked initial input")
    expected_b_sources={"driver_source_sha256":sha(c.ROOT/"tools"/"run_agnews_fair_continuation_v6.py"),
        "continuation_helper_sha256":sha(c.ROOT/"tools"/"agnews_fair_continuation_v6.py"),
        "common_sha256":base_source["common_sha256"],"engine_sha256":base_source["engine_sha256"]}
    if any(source.get(k)!=v for k,v in expected_b_sources.items()):
        raise AssertionError("continuation seal source lock differs from current frozen B implementation")
    runs=seal.get("runs",[])
    by={(r.get("seed"),r.get("arm")):r for r in runs}
    expected={(s,a) for s in SEEDS for a in ("aux_new_CE","aux_old_gold_CE")}
    assert len(by)==6 and set(by)==expected
    out=[]
    for seed in SEEDS:
        for arm in ("aux_new_CE","aux_old_gold_CE"):
            row=by[(seed,arm)]
            assert row.get("status")=="complete" and row.get("primary_updates")==576
            assert row.get("start_step")==448
            assert row.get("train_final",{}).get("all",{}).get("examples")==1024
            val_by_step={x["primary_step"]:x for x in row.get("validation",[])}
            assert set(val_by_step)=={448,512,768,1024}
            assert all(val_by_step[k]["metrics"]["all"]["examples"]==512 for k in val_by_step)
            cp=next(x for x in row.get("checkpoints",[]) if x.get("primary_step")==1024)
            cp_path=_safe_path(Path(path).parent,cp["path"])
            assert sha(cp_path)==cp["sha256"]
            out.append({"seed":seed,"arm":arm,"checkpoint_path":cp_path,
                        "checkpoint_sha256":cp["sha256"],"metadata":cp,
                        "train_final_metrics":row["train_final"],
                        "validation_final_metrics":val_by_step[1024]["metrics"],
                        "validation_trajectory":[{"primary_step":k,"metrics":val_by_step[k]["metrics"]}
                                                 for k in (448,512,768,1024)]})
    return seal,out


def _plan(args, current, continuation=None, initial=None):
    phase="continuation" if continuation else "initial"
    plan={"schema":"agnews-routing-fairness-protocol-v6","phase":phase,
        "seeds":list(SEEDS),"conditions":list(CONDITIONS),"physical_shared_conditions":list(PHYSICAL_CONDITIONS),
        "records_sha256":current["records_sha256"],"source_sha256":current,
        "warm_and_v5_checkpoints":current.get("checkpoint_sha256_by_seed"),
        "v5_router_gate_reference_sha256":args.v5_router_gate_sha256,
        "elapsed_limit_seconds":ELAPSED_LIMIT_SECONDS,"elapsed_enforcement":"cooperative between batches/conditions/seeds",
        "training_or_optimizer_updates":0,"confirmation_input":False,"confirmation_scored":False,
        "row_predictions_features_and_gate_weights_public":False,
        "hard_route_note":"one frozen backbone/final-MLP/original-head call per routed batch; both memory/readout contexts are computed then row-selected",
        "offline_note":"eight composite conditions are also derived from one old/new full-branch pair; learned confidence requires both full forwards",
        "confidence_selfjoint":"logP(known common format marker|prefix) + max-vocabulary logP(second token|prefix, known marker)",
        "selector_target":"train phase old/new only; class ids are used for consistency checks, task-ID diagnostic and scoring aggregation",
        "selector_supervision_note":"the correct gate adds 1,024 train-row old/new phase labels; the permuted gate reuses a fixed balanced permutation of those targets; this is additional task-identity supervision, not label-free routing",
        "unique_phase_labels":1024,"phase_label_uses_in_two_gate_fits":2048,
        "ridge_fit_count":0 if continuation else 2,
        "ridge_fit_definition":"fixed dual float64 lambda=1; training mean/std(ddof=0), std floor 1e-6, divide sqrt(d), score > 0 selects new and ties select old; phase labels only",
        "uniform_context_mix":"0.5 blend of old/new memory outputs and 0.5 blend of corresponding context-specific readout logits after one shared original head call; not a probability-average",
        "created_utc":datetime.now(timezone.utc).isoformat()}
    if continuation:
        plan["continuation_seal_sha256"]=sha(continuation)
        plan["initial_result_sha256"]=sha(initial/"result.json")
    return plan


def score_routed(model, router, records, pack, seed, stage, meter, *, condition, gate=None, masks=None,
                 expected_callback_mask=None,check_time):
    meter.stage=stage
    code=pack["code_maps"][str(seed)];labels=[int(x) for x in code["class_to_token_ids"]]
    device=next(model.parameters()).device;out=[];before=dict(router.operation_counts)
    state_before={"fingerprint":c.fingerprint(model),"execution":execution_state(model),
        "immutable":immutable_versions(model),"rng":c.engine.shared.tensor_tree_hash(c.engine.capture_rng())}
    gate_calls0=getattr(gate,"calls",0) if gate is not None else 0
    torch.cuda.reset_peak_memory_stats(device)
    start=time.monotonic()
    with torch.no_grad():
        for off in range(0,len(records),8):
            check_time();rr=records[off:off+8]
            batch=[{**r,"marker":int(code["marker_token_id"]),"label":labels[int(r["class_id"])]}for r in rr]
            ids,attention,lengths,markers,target=c.engine.shared.collate(batch,device,pack["tokenizer"]["pad_token_id"])
            ids,attention=c.engine.shared.append_markers(ids,attention,lengths,markers,pack["tokenizer"]["pad_token_id"])
            if condition=="uniform_context_mix":
                meter.charge(ids)
                logits=router.logits(ids,attention_mask=attention,mode="uniform_context_mix")
            elif condition in ("correct_phase_ridge","permuted_phase_ridge"):
                meter.charge(ids)
                logits=router.logits(ids,attention_mask=attention,selector=gate,
                                     prefix_lengths=lengths)
            else:
                batch_mask=torch.tensor(masks[off:off+len(rr)],dtype=torch.bool,device=device)
                meter.charge(ids)
                logits=router.logits(ids,batch_mask,attention_mask=attention)
            out.extend(metric_rows_from_logits(logits,batch,lengths,markers,target,labels))
    elapsed=time.monotonic()-start
    state_after={"fingerprint":c.fingerprint(model),"execution":execution_state(model),
        "immutable":immutable_versions(model),"rng":c.engine.shared.tensor_tree_hash(c.engine.capture_rng())}
    if state_before!=state_after: raise AssertionError(f"routed scoring mutated model/policy/RNG: {stage}")
    if gate is not None and gate.calls-gate_calls0 != math.ceil(len(records)/8):
        raise AssertionError("frozen selector callback was not invoked exactly once per batch")
    if expected_callback_mask is not None and not np.array_equal(gate.selected,np.asarray(expected_callback_mask,dtype=bool)):
        raise AssertionError(f"{condition}: live callback mask differs from offline full-branch feature mask")
    counts={k:router.operation_counts[k]-before.get(k,0) for k in router.operation_counts}
    metrics=aggregate_row_metrics(out)
    return out,metrics,{"wall_seconds":elapsed,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(device),"operation_counts":counts,
        "batches":math.ceil(len(records)/8),"full_forward_calls":math.ceil(len(records)/8),
        "unique_context_payload_bytes_excluding_shared_backbone":router.model_plus_context_unique_payload_bytes,
        "deduplicated_tensor_aliases":len(router.alias_map)}


def _stage_posthoc(record_rows, train_rows, val_rows):
    return {"train":aggregate_row_metrics(train_rows),"validation":aggregate_row_metrics(val_rows),
            "gate_fraction_new_train":record_rows["train"],"gate_fraction_new_validation":record_rows["validation"]}


def _cache_write(path, payload):
    write_json(path,payload)


def _load_initial_cache(initial, current):
    protocol=c.read(initial/"protocol.json");result=c.read(initial/"result.json")
    assert result.get("complete") is True and result.get("confirmation_scored") is False
    assert protocol.get("schema")=="agnews-routing-fairness-protocol-v6" and protocol.get("phase")=="initial"
    assert protocol.get("source_sha256")==current,"initial cache provenance differs from this continuation request"
    assert result.get("protocol_sha256")==sha(initial/"protocol.json")
    assert result.get("source_sha256")==current
    gates_path=initial/"private"/"gate.npz"
    assert sha(gates_path)==result["private_artifact_sha256"]["gate.npz"]
    correct,permuted,meta=_load_private_gate(gates_path)
    return protocol,result,correct,permuted,meta


def run(args):
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    start=time.monotonic()
    pack,warm,v5protocol,v5seal,refs,v5router_result=load_inputs(args)
    helper_sha=sha(c.ROOT/"tools"/"late_context_router_v6.py")
    if helper_sha!=EXPECTED_ROUTER_HELPER_SHA256:
        raise AssertionError("v6 helper differs from the reviewed frozen helper version")
    gate_hashes={x["gate"]["model_sha256"] for x in v5router_result["seeds"]}
    if len(gate_hashes)!=1: raise AssertionError("v5 fitted ridge differs by seed")
    args.v5_router_gate_sha256=next(iter(gate_hashes))
    current=source_manifest(args)
    current["tokenizer_file_sha256"]={x["name"]:x["sha256"]for x in pack["tokenizer"]["file_inventory"]}
    current["checkpoint_sha256_by_seed"]={str(s):{"warm":refs[s]["warm_sha"],"v5_cms_448":refs[s]["v5_sha"]} for s in SEEDS}
    continuation=None;continuation_rows=[];initial_cache=None
    if args.continuation_dir:
        if not args.initial_dir: raise ValueError("--initial-dir is required with --continuation-dir")
        continuation,continuation_rows=validate_continuation(args.continuation_dir/"development-sealed.json",refs,current)
        initial_cache=_load_initial_cache(args.initial_dir,current)
    c.preflight(args.model_dir,pack)
    plan=_plan(args,current,continuation=args.continuation_dir,initial=args.initial_dir)
    args.out_dir.mkdir(parents=True,exist_ok=False);(args.out_dir/"private").mkdir()
    write_json(args.out_dir/"protocol.json",plan);plan_sha=sha(args.out_dir/"protocol.json")
    meter=c.engine.WorkMeter();token=c.engine.ACTIVE_METER.set(meter)
    rows_out=[];parity=[];feature_hashes={};gate_models=None;gate_meta=None;private_paths=[]
    def check_time():
        if time.monotonic()-start>=ELAPSED_LIMIT_SECONDS: raise TimeoutError("v6 routing fairness phase exceeded 1800s")
    try:
        if initial_cache:
            initial_plan,initial_result,correct_gate,permuted_gate,gate_meta=initial_cache
            assert initial_plan["source_sha256"]["v5_router_result_sha256"]==current["v5_router_result_sha256"]
            assert gate_meta["records_sha256"]==current["records_sha256"]
            assert gate_meta["correct_gate_sha256"]==args.v5_router_gate_sha256
            gate_models={"correct":correct_gate,"permuted":permuted_gate}
        for seed in SEEDS:
            check_time()
            train=c.rows(pack,seed,role="train");val=c.rows(pack,seed,role="validation")
            _phase_labels(train);_phase_labels(val)
            warm_cp=torch.load(refs[seed]["warm_path"],map_location="cpu",weights_only=False)
            model=c.model(args.model_dir,seed);c.restore(model,warm_cp)
            warm_fp=c.fingerprint(model)
            if warm_fp!=refs[seed]["warm_meta"]["adapter_fingerprint"]:
                raise AssertionError(f"seed{seed}: warm checkpoint fingerprint mismatch")
            if continuation:
                candidates=[x for x in continuation_rows if x["seed"]==seed]
            else:
                v5cp=torch.load(refs[seed]["v5_path"],map_location="cpu",weights_only=False)
                candidates=[{"arm":"v5_CMS_reference_step448","checkpoint":v5cp,
                             "checkpoint_sha256":refs[seed]["v5_sha"],"metadata":refs[seed]["v5_meta"]}]

            # Development parity is train-only and precedes all validation scoring.
            if not continuation and seed==SEEDS[0]:
                parity.append(parity_fixture(model,warm_cp,candidates[0]["checkpoint"],train,pack,seed,meter))

            if continuation:
                row_cache_path=args.initial_dir/"private"/f"seed-{seed}-warm-row-cache.json"
                row_cache=c.read(row_cache_path)
                artifact_hashes=initial_cache[1]["private_artifact_sha256"]
                assert sha(row_cache_path)==artifact_hashes[row_cache_path.name]
                assert row_cache["records_sha256"]==current["records_sha256"]
                old_train=row_cache["train_rows"];old_val=row_cache["validation_rows"]
                fpath=args.initial_dir/"private"/f"seed-{seed}-features.npz"
                assert sha(fpath)==artifact_hashes[fpath.name]
                with np.load(fpath,allow_pickle=False) as z:
                    old_train_features=z["train"].copy();old_val_features=z["validation"].copy()
                feature_hashes[str(seed)]=row_cache["feature_sha256"]
            else:
                old_train,old_train_features=score_full_branch(model,train,pack,seed,
                    f"seed{seed}_warm_train",meter,capture_features=True,check_time=check_time)
                old_val,old_val_features=score_full_branch(model,val,pack,seed,
                    f"seed{seed}_warm_validation",meter,capture_features=True,check_time=check_time)
                ref_seed=next(x for x in v5router_result["seeds"] if x["seed"]==seed)
                compare_exact_metrics(aggregate_row_metrics(old_train),ref_seed["warm_reference_metrics"]["train"],
                                      f"seed{seed} warm train replay")
                compare_exact_metrics(aggregate_row_metrics(old_val),ref_seed["warm_reference_metrics"]["validation"],
                                      f"seed{seed} warm validation replay")
                feature_hashes[str(seed)]={"train_sha256":hashlib.sha256(old_train_features.tobytes()).hexdigest(),
                                           "validation_sha256":hashlib.sha256(old_val_features.tobytes()).hexdigest()}
                row_cache_path=args.out_dir/"private"/f"seed-{seed}-warm-row-cache.json"
                write_json(row_cache_path,{"schema":"agnews-v6-private-warm-row-cache","seed":seed,
                    "records_sha256":current["records_sha256"],"warm_checkpoint_sha256":refs[seed]["warm_sha"],
                    "train_rows":_serialize_metric_rows(old_train),"validation_rows":_serialize_metric_rows(old_val),
                    "feature_sha256":feature_hashes[str(seed)]})
                private_paths.append(row_cache_path)
                fpath=args.out_dir/"private"/f"seed-{seed}-features.npz"
                np.savez_compressed(fpath,train=old_train_features,validation=old_val_features)
                private_paths.append(fpath)
                if seed==SEEDS[0]:
                    correct_gate=fit_phase_ridge(old_train_features,[r["phase"] for r in train])
                    permuted_gate,permutation_sha=permuted_phase_fit(old_train_features,[r["phase"] for r in train])
                    if correct_gate["model_sha256"]!=args.v5_router_gate_sha256:
                        raise AssertionError("phase-only fit failed to reproduce the locked v5 ridge hash")
                    gate_models={"correct":correct_gate,"permuted":permuted_gate}
                    gate_meta={"schema":"agnews-v6-private-gates","records_sha256":current["records_sha256"],
                        "fit_rows":len(train),"fit_count":2,"correct_gate_sha256":correct_gate["model_sha256"],
                        "v5_gate_sha256":args.v5_router_gate_sha256,"permuted_gate_sha256":permuted_gate["model_sha256"],
                        "phase_target_sha256":hashlib.sha256(phase_targets([r["phase"]for r in train]).astype("<f8").tobytes()).hexdigest(),
                        "permuted_target_sha256":permutation_sha,"feature_sha256_seed17":feature_hashes[str(seed)]}
                    gate_path=args.out_dir/"private"/"gate.npz"
                    _save_private_gate(gate_path,correct_gate,permuted_gate,gate_meta);private_paths.append(gate_path)
                elif feature_hashes[str(seed)]!=feature_hashes[str(SEEDS[0])]:
                    raise AssertionError("original final-MLP feature hashes differ across seeds")
            if not gate_models:
                raise AssertionError("fixed gates must be available before any second-context scoring")

            if continuation:
                candidates=[{**x,"checkpoint":torch.load(x["checkpoint_path"],map_location="cpu",weights_only=False)}for x in candidates]
            for pair in candidates:
                check_time();newcp=pair["checkpoint"]
                print(json.dumps({"seed":seed,"context":pair["arm"],"status":"scoring_full_branches"}),flush=True)
                # Reuse one 3B backbone: score the new mutable context, then restore
                # the warm state before constructing a helper that shares warm storage.
                warm_before={"fingerprint":c.fingerprint(model),"execution":execution_state(model),
                    "immutable":immutable_versions(model),
                    "rng":c.engine.shared.tensor_tree_hash(c.engine.capture_rng())}
                if warm_before["fingerprint"]!=warm_fp or warm_before["execution"]!=warm_cp["execution_state"]:
                    raise AssertionError("warm context changed before sequential branch scoring")
                c.restore(model,newcp)
                new_train,_=score_full_branch(model,train,pack,seed,f"seed{seed}_{pair['arm']}_train",
                    meter,capture_features=False,check_time=check_time)
                new_val,_=score_full_branch(model,val,pack,seed,f"seed{seed}_{pair['arm']}_validation",
                    meter,capture_features=False,check_time=check_time)
                if continuation:
                    compare_exact_metrics(aggregate_row_metrics(new_train),pair["train_final_metrics"],
                                          f"seed{seed}/{pair['arm']} sealed train score")
                    compare_exact_metrics(aggregate_row_metrics(new_val),pair["validation_final_metrics"],
                                          f"seed{seed}/{pair['arm']} sealed validation score")
                else:
                    ref_seed=next(x for x in v5router_result["seeds"] if x["seed"]==seed)
                    compare_exact_metrics(aggregate_row_metrics(new_train),ref_seed["new_CMS_reference_metrics"]["train"],
                                          f"seed{seed} new CMS train replay")
                    compare_exact_metrics(aggregate_row_metrics(new_val),ref_seed["new_CMS_reference_metrics"]["validation"],
                                          f"seed{seed} new CMS validation replay")
                c.restore(model,warm_cp)
                warm_after={"fingerprint":c.fingerprint(model),"execution":execution_state(model),
                    "immutable":immutable_versions(model),
                    "rng":c.engine.shared.tensor_tree_hash(c.engine.capture_rng())}
                if warm_after!=warm_before:
                    raise AssertionError("warm model/policy/RNG did not restore exactly before route-helper construction")
                if c.fingerprint(model)!=warm_fp or execution_state(model)!=warm_cp["execution_state"]:
                    raise AssertionError("warm branch changed before constructing routed context")
                train_post,train_masks=posthoc_conditions(old_train,new_train,train,old_train_features,
                    gate_models["correct"],gate_models["permuted"])
                val_post,val_masks=posthoc_conditions(old_val,new_val,val,old_val_features,
                    gate_models["correct"],gate_models["permuted"])
                router=LateContextRouter(model,newcp["model"],newcp["execution_state"],
                    old_execution_state=warm_cp["execution_state"],deduplicate_equal=True)
                physical={}
                for split,records,oldrows,newrows,masks in (("train",train,old_train,new_train,train_masks),
                                                            ("validation",val,old_val,new_val,val_masks)):
                    for condition in PHYSICAL_CONDITIONS:
                        check_time();callback=None;fixed=None
                        if condition=="uniform_context_mix":
                            pass
                        elif condition=="constant_old": fixed=np.zeros(len(records),dtype=bool)
                        elif condition=="constant_new": fixed=np.ones(len(records),dtype=bool)
                        elif condition=="hash_random": fixed=hash_random_mask(records)
                        elif condition=="correct_phase_ridge":
                            g=gate_models["correct"];callback=RecordingSelector(FrozenRidgeGate(g["mean"],g["scale"],g["weight"]))
                        elif condition=="permuted_phase_ridge":
                            g=gate_models["permuted"];callback=RecordingSelector(FrozenRidgeGate(g["mean"],g["scale"],g["weight"]))
                        _,routed_metrics,cost=score_routed(model,router,records,pack,seed,
                            f"seed{seed}_{pair['arm']}_{condition}_{split}",meter,
                            condition=condition,gate=callback,masks=fixed,
                            expected_callback_mask=masks[condition] if callback is not None else None,
                            check_time=check_time)
                        if condition=="uniform_context_mix": expected=None
                        elif condition=="constant_old": expected=aggregate_row_metrics(oldrows)
                        elif condition=="constant_new": expected=aggregate_row_metrics(newrows)
                        else: expected=aggregate_row_metrics(select_row_metrics(oldrows,newrows,masks[condition]))
                        if expected is not None:
                            compare_metrics(routed_metrics,expected,label=f"{condition}/{split} shared-forward parity")
                        call_count=math.ceil(len(records)/8)
                        if cost["operation_counts"]["original_final_mlp_calls"]!=call_count or cost["operation_counts"]["original_lm_head_calls"]!=call_count:
                            raise AssertionError(f"{condition}/{split}: shared original modules not called exactly once/batch")
                        if cost["operation_counts"]["old_memory_branch_calls"]!=call_count or cost["operation_counts"]["new_memory_branch_calls"]!=call_count:
                            raise AssertionError(f"{condition}/{split}: both memory contexts were not evaluated once/batch")
                        if cost["operation_counts"]["old_readout_adapter_calls"]!=call_count or cost["operation_counts"]["new_readout_adapter_calls"]!=call_count:
                            raise AssertionError(f"{condition}/{split}: both context readouts not evaluated once/batch")
                        physical[f"{split}:{condition}"]={"metrics":routed_metrics,"cost":cost}
                        print(json.dumps({"seed":seed,"context":pair["arm"],"split":split,
                                          "condition":condition,"status":"complete",
                                          "seconds":cost["wall_seconds"]}),flush=True)
                output={"seed":seed,"new_context":pair["arm"],"checkpoint_sha256":pair.get("checkpoint_sha256"),
                    "posthoc":{k:{"train":train_post[k]["metrics"],"validation":val_post[k]["metrics"],
                        "gate_fraction_new_train":train_post[k]["gate_fraction_new"],
                        "gate_fraction_new_validation":val_post[k]["gate_fraction_new"]}for k in CONDITIONS},
                    "physical_shared":physical,"old_feature_sha256":feature_hashes[str(seed)],
                    "new_branch_train":aggregate_row_metrics(new_train),"new_branch_validation":aggregate_row_metrics(new_val),
                    "continuation_validation_trajectory":pair.get("validation_trajectory")}
                rows_out.append(output)
                del router
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        if not continuation and len(feature_hashes)!=3: raise AssertionError("three seeds of backbone feature hashes required")
        if not continuation and len(rows_out)!=3: raise AssertionError("expected initial warm/v5 pair for all seeds")
        if continuation and len(rows_out)!=6: raise AssertionError("expected all six continuation endpoints")
        check_time()
        private_hashes={p.name:sha(p) for p in private_paths}
        if continuation:
            private_hashes={}
        result={"schema":"agnews-routing-fairness-result-v6","phase":"continuation" if continuation else "initial",
            "complete":True,"protocol_sha256":plan_sha,"source_sha256":current,"seeds":list(SEEDS),
            "rows":rows_out,"parity_fixtures":parity,"feature_hashes_by_seed":feature_hashes,
            "gate_models_public_hashes":None if gate_meta is None else {k:gate_meta[k] for k in
                ("correct_gate_sha256","v5_gate_sha256","permuted_gate_sha256","fit_count")},
            "elapsed_seconds":time.monotonic()-start,"elapsed_limit_seconds":ELAPSED_LIMIT_SECONDS,
            "work":meter.counts,"actual_optimizer_updates":0,"confirmation_scored":False,"no_holdout_input":True,
            "private_artifact_sha256":private_hashes,"public_row_predictions_features_or_gate_weights_written":False}
        write_json(args.out_dir/"result.json",result)
    except BaseException as exc:
        write_json(args.out_dir/"failure.json",{"schema":"agnews-routing-fairness-failure-v6",
            "error_type":type(exc).__name__,"error":str(exc),"work":meter.counts,
            "completed_pairs":len(rows_out),"confirmation_scored":False,"elapsed_seconds":time.monotonic()-start})
        raise
    finally:
        c.engine.ACTIVE_METER.reset(token)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("records","warm-dir","v5-run-dir","model-dir","out-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--v5-router-result",type=Path,default=ROOT/"v5_final_review"/"router"/"result.json")
    p.add_argument("--expected-v5-router-result-sha256",default=EXPECTED_V5_ROUTER_RESULT_SHA256)
    p.add_argument("--continuation-dir",type=Path)
    p.add_argument("--initial-dir",type=Path)
    args=p.parse_args();run(args)


if __name__=="__main__":main()
