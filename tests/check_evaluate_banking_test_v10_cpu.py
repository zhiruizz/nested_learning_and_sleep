"""Small synthetic tests for the sealed Banking77 evaluator (no model/data)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import evaluate_banking_test_v10 as ev


H = "a" * 64


def _seal():
    runs = []
    for seed in ev.SEEDS:
        for arm in ev.ARMS:
            runs.append({
                "seed": seed, "arm": arm, "status": "complete", "tasks_completed": 7,
                "final_checkpoint_sha256": H, "run_receipt_sha256": H,
                "final_checkpoint": "checkpoints/final-task06.pt",
                "config_sha256": H, "source_lock_sha256": H,
                "training_manifest_sha256": H,
                "endpoint_metrics": {}, "work": {},
                "run_dir": f"seed{seed}/{arm}",
            })
    return {"schema": "banking-cil-v10-training-seal-v1", "complete": True,
            "config_sha256": H, "source_lock_sha256": H,
            "training_manifest_sha256": H, "test_data_included": False,
            "official_test_read": False, "runs": runs}


def test_seal_requires_exact_complete_grid_and_unique_directories():
    rows = ev.validate_training_seal(_seal(), config_sha256=H,
                                     source_lock_sha256=H,
                                     training_manifest_sha256=H)
    assert len(rows) == 9
    bad = _seal()
    bad["runs"][1]["run_dir"] = bad["runs"][0]["run_dir"]
    with pytest.raises(ValueError, match="own run directory"):
        ev.validate_training_seal(bad, config_sha256=H,
                                  source_lock_sha256=H,
                                  training_manifest_sha256=H)


def test_evaluation_contract_binds_source_model_seal_test_and_policy():
    cfg = {"_logical_sha256": H, "model_revision": "revision-x",
           "model_assets_sha256": {"weights.safetensors": H},
           "source_hashes": {"tools/driver.py": H}, "max_sequence_length": 128}
    contract = {
        "schema": ev.CONTRACT_SCHEMA, "evaluator_sha256": H,
        "training_config_sha256": H, "training_config_file_sha256": H,
        "source_lock_sha256": H, "source_hashes": cfg["source_hashes"],
        "model_revision": cfg["model_revision"],
        "model_assets_sha256": cfg["model_assets_sha256"],
        "training_manifest_sha256": H, "training_seal_sha256": H,
        "main_release_sha256": H,
        "foundation_refs": {str(seed): {"receipt_sha256": H, "checkpoint_sha256": H}
                            for seed in ev.SEEDS},
        "official_test": {"revision": ev.EXPECTED_TEST_REVISION,
                          "bytes": ev.EXPECTED_TEST_BYTES,
                          "sha256": ev.EXPECTED_TEST_SHA256,
                          "rows": 3080, "classes": 77, "rows_per_class": 40},
        "evaluation_policy": {
            "primary": "global_77_sum_log_likelihood",
            "secondary": ["global_77_length_mean_log_likelihood", "unconstrained_greedy_generation"],
            "include_greedy": True, "max_sequence_length": 128,
            "candidate_batch_size": 32, "max_new_tokens": 10,
            "primary_includes_every_official_row": True,
            "secondary_subset": "exclude_normalized_text_overlap_with_training_and_development",
            "secondary_overlap_roles": ["train", "development"],
            "secondary_overlap_rule": "exact_common_normalize_text_match_in_either_role",
            "max_backbone_forwards": ev.MAX_FORWARDS, "max_wall_seconds": 21600.0,
            "optimizer_updates": 0,
        },
        "training_manifest_excludes_test": True,
    }
    ev.validate_evaluation_contract(contract, evaluator_sha256=H, config=cfg,
                                    config_file_sha256=H, source_lock_sha256=H,
                                    training_manifest_sha256=H, training_seal_sha256=H,
                                    main_release_sha256=H,
                                    foundation_refs=contract["foundation_refs"])
    contract["evaluation_policy"]["include_greedy"] = False
    with pytest.raises(ValueError, match="policy differs"):
        ev.validate_evaluation_contract(contract, evaluator_sha256=H, config=cfg,
                                        config_file_sha256=H, source_lock_sha256=H,
                                        training_manifest_sha256=H, training_seal_sha256=H,
                                        main_release_sha256=H,
                                        foundation_refs=contract["foundation_refs"])
    bad = _seal()
    bad["runs"].pop()
    with pytest.raises(ValueError, match="exactly the nine"):
        ev.validate_training_seal(bad, config_sha256=H,
                                  source_lock_sha256=H,
                                  training_manifest_sha256=H)


def test_seal_and_artifact_verification_bind_receipt_and_terminal_checkpoint(tmp_path):
    seal = _seal()
    rows = []
    for index, row in enumerate(seal["runs"]):
        run_dir = tmp_path / row["run_dir"]
        (run_dir / "checkpoints").mkdir(parents=True)
        checkpoint = run_dir / "checkpoints" / "final-task06.pt"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        cp_sha = ev.sha256_file(checkpoint)
        receipt = {
            "schema": "banking-cil-v10-run-receipt", "status": "complete",
            "seed": row["seed"], "arm": row["arm"], "tasks_completed": 7,
            # The Windows producer writes native separators while the seal is POSIX-normalized.
            "final_checkpoint": r"checkpoints\final-task06.pt",
            "final_checkpoint_sha256": cp_sha, "config_sha256": H,
            "source_lock_sha256": H, "training_manifest_sha256": H,
            "foundation_checkpoint_sha256": H,
            "test_data_included": False,
            "model_immutable_versions_unchanged": True,
        }
        receipt_path = run_dir / "run-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        protocol = {
            "schema": "banking-cil-v10-protocol", "phase": "main",
            "seed": row["seed"], "arm": row["arm"], "config_sha256": H,
            "source_lock_sha256": H, "training_manifest_sha256": H,
            "test_data_included": False, "test_reader_available": False,
            "parent": {"foundation_receipt_sha256": H,
                       "foundation_checkpoint_sha256": H,
                       "release_artifact_sha256": H},
        }
        (run_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        row["final_checkpoint_sha256"] = cp_sha
        row["run_receipt_sha256"] = ev.sha256_file(receipt_path)
        row["run_dir"] = str(run_dir)
        rows.append(row)
    refs = {str(seed): {"receipt_sha256": H, "checkpoint_sha256": H}
            for seed in ev.SEEDS}
    verified = ev.verify_run_artifacts(rows, tmp_path, config_sha256=H,
                                       source_lock_sha256=H,
                                       training_manifest_sha256=H,
                                       foundation_refs=refs,
                                       main_release_sha256=H)
    assert len(verified) == 9
    # Mutation after seal verification is detected before deserialization.
    verified[0]["checkpoint_path"].write_bytes(b"tampered")

    class NeverLoad:
        def load(self, *args, **kwargs):
            raise AssertionError("changed checkpoint must be rejected before load")

    with pytest.raises(ValueError, match="changed after artifact verification"):
        ev.preflight_checkpoint_payloads(verified, torch=NeverLoad(),
                                         config_sha256=H, source_lock_sha256=H,
                                         training_manifest_sha256=H)


def test_terminal_checkpoint_metadata_accepts_only_safe_relative_spellings():
    assert ev._is_terminal_checkpoint_relative("checkpoints/final-task06.pt")
    assert ev._is_terminal_checkpoint_relative(r"checkpoints\final-task06.pt")
    for unsafe in (
        r"..\checkpoints\final-task06.pt",
        r"checkpoints\..\final-task06.pt",
        r"C:\runs\checkpoints\final-task06.pt",
        r"\server\share\final-task06.pt",
        "/checkpoints/final-task06.pt",
        "./checkpoints/final-task06.pt",
        "checkpoints//final-task06.pt",
        "checkpoints/final-task07.pt",
    ):
        assert not ev._is_terminal_checkpoint_relative(unsafe), unsafe


def test_context_loader_uses_real_clone_api_and_model_loader_immutable_callback(tmp_path):
    import sys
    from unittest.mock import patch

    import torch
    from torch import nn

    task_root = Path(__file__).resolve().parents[1]
    tools_root = str(task_root / "tools")
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    import clinc_tail_runtime_v8 as tail_runtime
    import run_banking_cil_v10 as driver
    import banking_cil_v10_routing as routing

    class ToyWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.memory = nn.ModuleList([nn.Linear(2, 2)])
            self.original = nn.Identity()

    class ToyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.ones(2, 2))
            self.b = nn.Parameter(torch.ones(2, 2))
            self.fp32_accumulation = False

        def forward(self, values):
            return values

    class ToyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = ToyWrapper()
            self.post_attention_layernorm = nn.Identity()

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.model = nn.Module()
            self.backbone.model.layers = nn.ModuleList([ToyLayer()])
            self.backbone.model.norm = nn.Identity()
            self.backbone.lm_head = ToyHead()

    model = ToyModel()
    # Call the actual public helper to build the producer-shaped context state.
    expected_context = tail_runtime.clone_mutable_context(model)
    payload = {"context": expected_context.state_dict()}
    immutable_fn = lambda loaded_model: {"model_id": id(loaded_model)}
    with patch.object(driver, "_load_model",
                      return_value=(torch, object(), model, immutable_fn)), \
         patch.object(routing, "install_additive_routes"):
        tokenizer, loaded_model, contexts, frozen, callback = ev._load_model_contexts(
            [{"seed": 17, "arm": "CMS_ONLY", "payload": payload}],
            project_root=tmp_path, model_dir=tmp_path, cfg={}, torch=torch)

    assert loaded_model is model
    assert tokenizer is not None
    assert callback is immutable_fn
    assert frozen == immutable_fn(model)
    assert set(contexts) == {"seed17:CMS_ONLY"}
    cloned = contexts["seed17:CMS_ONLY"]
    assert cloned is not expected_context
    assert list(cloned.named_parameters())
    assert all(not parameter.requires_grad for parameter in cloned.parameters())


def test_checkpoint_cursor_accepts_real_wake_snapshot_surface_and_rejects_wrong_run():
    record = {"seed": 17, "arm": "CMS_ONLY"}
    wake = SimpleNamespace(context_state={}, pending={}, clocks={}, token_counts={},
                           ticks=123, periods=(16, 64, 256, 1024, None),
                           base_lr=0.1, fast_lr=0.2, bank_states=[])
    payload = {"schema": "banking-cil-v10-checkpoint", "seed": 17,
               "arm": "CMS_ONLY", "task_id": 6, "step": 123,
               "config_sha256": H, "source_lock_sha256": H,
               "training_manifest_sha256": H, "context": {}, "wake": wake,
               "rng": {"python": None, "numpy": None, "torch_cpu": None, "torch_cuda": []}}
    ev.validate_checkpoint_payload(payload, record, config_sha256=H,
                                    source_lock_sha256=H,
                                    training_manifest_sha256=H)
    payload["seed"] = 29
    with pytest.raises(ValueError, match="seed"):
        ev.validate_checkpoint_payload(payload, record, config_sha256=H,
                                       source_lock_sha256=H,
                                       training_manifest_sha256=H)


def test_foundation_release_requires_three_qualified_receipt_and_checkpoint_bindings(tmp_path):
    import torch
    foundation_root = tmp_path / "foundations"
    rows = {}
    for seed in ev.SEEDS:
        run_dir = foundation_root / f"seed{seed}"
        run_dir.mkdir(parents=True)
        cp = run_dir / "foundation.pt"
        wake = SimpleNamespace(context_state={}, pending={}, clocks={}, token_counts={},
                               ticks=1, periods=(16, 64, 256, 1024, None),
                               base_lr=0.1, fast_lr=0.2, bank_states=[])
        torch.save({"schema": "banking-cil-v10-checkpoint", "seed": seed,
                    "arm": "FOUNDATION_CMS_ONLY", "task_id": 0, "step": 1,
                    "context": {}, "wake": wake,
                    "rng": {"python": None, "numpy": None, "torch_cpu": torch.tensor([1]),
                            "torch_cuda": None},
                    "config_sha256": H, "source_lock_sha256": H,
                    "training_manifest_sha256": H, "extra": {}}, cp)
        cp_sha = ev.sha256_file(cp)
        receipt = {"schema": "banking-cil-v10-foundation-receipt",
                   "status": "qualified", "seed": seed,
                   "source_lock_sha256": H, "config_sha256": H,
                   "training_manifest_sha256": H,
                   "checkpoint_sha256": cp_sha, "checkpoint": "foundation.pt"}
        receipt_path = run_dir / "foundation-receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        rows[str(seed)] = {"status": "qualified", "run_dir": str(run_dir),
                           "receipt_sha256": ev.sha256_file(receipt_path),
                           "checkpoint_sha256": cp_sha}
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps({"foundations": rows}), encoding="utf-8")

    class Driver:
        @staticmethod
        def _validate_foundation_release(path, cfg, config_sha, source_sha, manifest_sha,
                                         seed, foundation_dir):
            assert seed == 17 and foundation_dir == Path(rows["17"]["run_dir"])
            return {}, {}

        @staticmethod
        def _validate_foundation_checkpoint_cpu(path, expected_sha, *, seed, cfg,
                                                source_lock, manifest_sha):
            assert ev.sha256_file(path) == expected_sha
            return {"step": 1, "cpu_deserialized": True}

    audit = ev.verify_foundation_grid(release_path, foundation_root, driver=Driver,
                                      cfg={}, config_sha256=H, source_lock_sha256=H,
                                      training_manifest_sha256=H, torch=torch)
    assert audit["all_qualified"] and audit["all_checkpoints_cpu_deserialized"]
    assert audit["foundations"] == {
        key: {"receipt_sha256": value["receipt_sha256"],
              "checkpoint_sha256": value["checkpoint_sha256"]}
        for key, value in rows.items()}
    outside = tmp_path / "outside"
    outside.mkdir()
    rows["17"]["run_dir"] = str(outside)
    release_path.write_text(json.dumps({"foundations": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="escaped"):
        ev.verify_foundation_grid(release_path, foundation_root, driver=Driver,
                                  cfg={}, config_sha256=H, source_lock_sha256=H,
                                  training_manifest_sha256=H, torch=torch)


def test_synthetic_parser_keeps_all_rows_and_reports_train_overlap(tmp_path):
    classes = tuple(f"intent_{i:02d}" for i in range(77))
    labels = {name: i for i, name in enumerate(classes)}
    csv_path = tmp_path / "synthetic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("text", "category"))
        writer.writeheader()
        for i, name in enumerate(classes):
            writer.writerow({"text": f"utterance {i}", "category": name})
    raw = csv_path.read_bytes()
    class Common:
        @staticmethod
        def canonical_label(value):
            return value.strip().casefold().replace("_", " ")

        @staticmethod
        def normalize_text(value):
            return " ".join(value.casefold().split())

    # The synthetic map uses normalized spellings, mirroring manifest aliases.
    normalized = {Common.canonical_label(k): v for k, v in labels.items()}
    primary, secondary, audit = ev.parse_test_csv(
        csv_path, classes=classes, source_label_to_id=normalized, common=Common,
        train_rows=[{"text": "utterance 0"}], development_rows=[], expected_bytes=len(raw),
        expected_sha256=hashlib.sha256(raw).hexdigest(), expected_rows=77,
        expected_rows_per_class=1)
    assert len(primary) == 77
    assert len(secondary) == 76
    assert audit["normalized_train_overlap_rows"] == 1
    assert audit["normalized_development_overlap_rows"] == 0
    assert audit["training_rows_considered"] == 1
    assert audit["development_rows_considered"] == 0
    assert audit["train_development_rows_considered"] == 1
    assert audit["normalized_train_or_development_overlap_rows"] == 1
    assert audit["train_development_disjoint_secondary_rows"] == 76
    assert audit["secondary_reuses_primary_cohort"] is False
    assert audit["primary_includes_all_official_rows"] is True


def test_development_only_overlap_is_excluded_from_secondary(tmp_path):
    classes = tuple(f"intent_{i:02d}" for i in range(77))
    labels = {name: i for i, name in enumerate(classes)}
    csv_path = tmp_path / "synthetic-dev-overlap.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("text", "category"))
        writer.writeheader()
        for i, name in enumerate(classes):
            writer.writerow({"text": f"utterance {i}", "category": name})
    raw = csv_path.read_bytes()

    class Common:
        @staticmethod
        def canonical_label(value): return value.casefold().replace("_", " ")
        @staticmethod
        def normalize_text(value): return " ".join(value.casefold().split())

    normalized = {Common.canonical_label(k): v for k, v in labels.items()}
    primary, secondary, audit = ev.parse_test_csv(csv_path, classes=classes,
        source_label_to_id=normalized, common=Common, train_rows=[{"text": "not present"}],
        development_rows=[{"text": " UtTeRaNcE   1 "}], expected_bytes=len(raw),
        expected_sha256=hashlib.sha256(raw).hexdigest(), expected_rows=77,
        expected_rows_per_class=1)
    assert len(primary) == 77 and len(secondary) == 76
    assert audit["normalized_train_overlap_rows"] == 0
    assert audit["normalized_development_overlap_rows"] == 1
    assert audit["training_rows_considered"] == 1
    assert audit["development_rows_considered"] == 1
    assert audit["train_development_rows_considered"] == 2
    assert audit["normalized_train_or_development_overlap_rows"] == 1
    assert audit["train_development_disjoint_secondary_rows"] == 76


def test_mixed_role_overlap_counts_union_once_and_zero_overlap_reuses_primary(tmp_path):
    classes = tuple(f"intent_{i:02d}" for i in range(77))
    labels = {name: i for i, name in enumerate(classes)}
    csv_path = tmp_path / "synthetic-role-overlap.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("text", "category"))
        writer.writeheader()
        for i, name in enumerate(classes):
            writer.writerow({"text": f"utterance {i}", "category": name})
    raw = csv_path.read_bytes()

    class Common:
        @staticmethod
        def canonical_label(value): return value.casefold().replace("_", " ")
        @staticmethod
        def normalize_text(value): return " ".join(value.casefold().split())

    normalized = {Common.canonical_label(k): v for k, v in labels.items()}
    primary, secondary, audit = ev.parse_test_csv(csv_path, classes=classes,
        source_label_to_id=normalized, common=Common,
        train_rows=[{"text": "utterance 0"}, {"text": "utterance 2"}],
        development_rows=[{"text": "utterance 1"}, {"text": "utterance 2"}],
        expected_bytes=len(raw), expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_rows=77, expected_rows_per_class=1)
    assert len(secondary) == 74
    assert audit["normalized_train_overlap_rows"] == 2
    assert audit["normalized_development_overlap_rows"] == 2
    assert audit["normalized_train_or_development_overlap_rows"] == 3
    assert audit["secondary_reuses_primary_cohort"] is False

    primary, secondary, audit = ev.parse_test_csv(csv_path, classes=classes,
        source_label_to_id=normalized, common=Common, train_rows=[], development_rows=[],
        expected_bytes=len(raw), expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_rows=77, expected_rows_per_class=1)
    assert ev._same_ordered_cohort(primary, secondary) is True
    assert audit["secondary_reuses_primary_cohort"] is True


def test_parser_rejects_unknown_label_without_treating_it_as_other(tmp_path):
    classes = tuple(f"intent_{i:02d}" for i in range(77))
    labels = {name: i for i, name in enumerate(classes)}
    csv_path = tmp_path / "synthetic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("text", "category"))
        writer.writeheader()
        for i, name in enumerate(classes):
            writer.writerow({"text": f"utterance {i}", "category": "not_a_bank_label" if i == 0 else name})
    raw = csv_path.read_bytes()

    class Common:
        @staticmethod
        def canonical_label(value): return value.casefold().replace("_", " ")
        @staticmethod
        def normalize_text(value): return value.casefold()

    normalized = {Common.canonical_label(k): v for k, v in labels.items()}
    with pytest.raises(ValueError, match="outside the locked Banking77 vocabulary"):
        ev.parse_test_csv(csv_path, classes=classes, source_label_to_id=normalized,
                          common=Common, train_rows=[], development_rows=[], expected_bytes=len(raw),
                          expected_sha256=hashlib.sha256(raw).hexdigest(),
                          expected_rows=77, expected_rows_per_class=1)


def test_budget_observes_candidate_and_greedy_logits_masks_and_restores_wrapper():
    torch = pytest.importorskip("torch")

    class TinyModel:
        def logits(self, tokens, mask=None):
            return tokens

    model = TinyModel()
    original_impl = model.logits.__func__
    budget = ev.EvaluationBudget(max_forwards=4, max_wall_seconds=30)
    restore = budget.install_logits_observer(model)
    try:
        candidate_ids = torch.ones((2, 3), dtype=torch.long)
        candidate_mask = torch.ones_like(candidate_ids)
        greedy_ids = torch.ones((3, 3), dtype=torch.long)
        greedy_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.long)
        model.logits(candidate_ids, candidate_mask)
        model.logits(greedy_ids, greedy_mask)
        # Candidate and greedy result counters remain the source of the work
        # reconciliation; the hook independently measures both actual calls.
        budget.add({"candidate_backbone_forwards": 1, "candidate_tail_calls": 1,
                    "candidate_input_tokens": 6,
                    "greedy": {"backbone_forwards": 1, "tail_calls": 1}})
    finally:
        restore()
    assert model.logits.__func__ is original_impl
    summary = budget.summary()
    assert summary["backbone_forwards"] == summary["observed_backbone_forward_calls"] == 2
    assert summary["input_tokens"] == summary["candidate_input_tokens"] == 6
    assert summary["input_tokens_scope"] == "candidate_scoring_only_legacy_counter"
    assert summary["observed_backbone_valid_tokens"] == 12
    assert summary["observed_backbone_executed_tokens_including_padding"] == 15
    assert summary["observed_backbone_forward_shape_histogram"] == {"2x3": 1, "3x3": 1}


def test_budget_rejects_forward_counter_mismatch():
    torch = pytest.importorskip("torch")

    class TinyModel:
        def logits(self, tokens, mask=None):
            return tokens

    model = TinyModel()
    budget = ev.EvaluationBudget(max_forwards=4, max_wall_seconds=30)
    restore = budget.install_logits_observer(model)
    try:
        model.logits(torch.ones((1, 2), dtype=torch.long), torch.ones((1, 2), dtype=torch.long))
        budget.add({"candidate_backbone_forwards": 2, "candidate_input_tokens": 2})
    finally:
        restore()
    with pytest.raises(RuntimeError, match="observed complete-backbone forwards"):
        budget.summary()


def test_budget_reconciles_actual_candidate_and_greedy_capture_pipeline():
    torch = pytest.importorskip("torch")
    from tools import banking_cil_common_v10 as common
    from tools.banking_cil_v10_runtime import score_candidates

    class TinyTokenizer:
        def __init__(self):
            self.eos_token_id = 127
            self._ids = {}
            self._texts = {}

        def encode(self, text, add_special_tokens=False):
            if text not in self._ids:
                token = len(self._ids) + 1
                self._ids[text] = token
                self._texts[token] = text
            return [self._ids[text]]

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(self._texts.get(int(x), "other") for x in ids)

    classes = tuple(f"label_{i:02d}" for i in range(77))
    tokenizer = TinyTokenizer()
    # Allocate stable body IDs before the common tokenizer builds prompt frames.
    first_body = tokenizer.encode("first", add_special_tokens=False)[0]
    second_body = tokenizer.encode("second", add_special_tokens=False)[0]

    class TinyModel:
        def logits(self, tokens, mask=None):
            logits = torch.zeros((tokens.shape[0], tokens.shape[1], 128), dtype=torch.float32)
            if tokens.shape[1] == 3:
                # First greedy row ends immediately; second emits one token and
                # reaches EOS on the next capture, creating one padded position.
                logits[:, -1, 126] = 5.0
                logits[tokens[:, 1] == first_body, -1, 127] = 6.0
            elif tokens.shape[1] == 4:
                logits[:, -1, 127] = 6.0
            return logits

    class TinyTailRuntime:
        @staticmethod
        def capture(model, ids, mask, *, device=None):
            return model.logits(ids, mask)

        @staticmethod
        def tail_logits(model, context, cache):
            return cache

    model = TinyModel()
    budget = ev.EvaluationBudget(max_forwards=100, max_wall_seconds=30)
    restore = budget.install_logits_observer(model)
    rows = [{"text": "first", "intent_id": 0}, {"text": "second", "intent_id": 1}]
    try:
        result = score_candidates(model, {"tiny": object()}, rows, tokenizer, classes, common,
            TinyTailRuntime, torch=torch, device="cpu", max_sequence_length=128,
            batch_size=32, budget=budget, class_order=None, task_id=None,
            include_greedy=True, max_new_tokens=2)
        budget.add(result["work"])
    finally:
        restore()
    summary = budget.summary()
    assert result["work"]["candidate_backbone_forwards"] == 5
    assert result["work"]["greedy"]["backbone_forwards"] == 2
    assert summary["backbone_forwards"] == summary["observed_backbone_forward_calls"] == 7
    assert summary["candidate_input_tokens"] == 770
    assert summary["observed_backbone_executed_tokens_including_padding"] == 784
    assert summary["observed_backbone_valid_tokens"] == 783
    assert summary["observed_backbone_forward_shape_histogram"] == {
        "2x3": 1, "2x4": 1, "26x5": 1, "32x5": 4}
