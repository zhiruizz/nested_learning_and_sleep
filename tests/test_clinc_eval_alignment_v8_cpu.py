"""End-to-end CPU regression for CLINC prompt-length eval ordering."""
from __future__ import annotations

import importlib
import sys
import types

import torch

from tools import clinc_cil_common_v8 as common


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self, classes):
        self.id_to_label = {1000 + i: label for i, label in enumerate(classes)}
        self.label_to_id = {label: 1000 + i for i, label in enumerate(classes)}

    def encode(self, text, add_special_tokens=False):
        return [self.label_to_id[text]]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.id_to_label.get(int(token), "") for token in token_ids)


class FakeBudget:
    def check(self):
        return None


def test_grouped_greedy_predictions_keep_original_row_label_alignment(monkeypatch):
    # Lexically sorted canonical labels satisfy the real CLINC metric helper.
    classes = tuple(f"label{i:03d}" for i in range(150))
    tokenizer = FakeTokenizer(classes)
    labels_by_prompt = {10: 1000, 20: 1015}

    def tokenize_prompt(text, _tokenizer, _classes, _eos, max_length=None, **_kwargs):
        code = 10 if text == "long-first" else 20
        ids = [code, code, code] if code == 10 else [code]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "prompt_length": len(ids)}

    monkeypatch.setattr(common, "tokenize_prompt", tokenize_prompt)
    monkeypatch.setitem(sys.modules, "clinc_cil_common_v8", common)

    def capture(_model, ids, _mask, device=None):
        return types.SimpleNamespace(ids=ids.detach().clone())

    def tail_logits(_model, _context, cache):
        ids = cache.ids
        logits = torch.full((ids.shape[0], ids.shape[1], 1200), -100.0, dtype=torch.float32)
        for row in range(ids.shape[0]):
            last = int(ids[row, -1])
            next_token = torch.tensor(2 if last >= 1000 else labels_by_prompt[last])
            logits[row, -1, int(next_token)] = 100.0
        return logits

    fake_runtime = types.ModuleType("clinc_tail_runtime_v8")
    fake_runtime.capture = capture
    fake_runtime.tail_logits = tail_logits
    monkeypatch.setitem(sys.modules, "clinc_tail_runtime_v8", fake_runtime)

    r2 = importlib.import_module("tools.run_clinc_cil_v8_r2")
    r1 = importlib.import_module("tools.run_clinc_cil_v8")
    rows = [
        {"id": "a" * 64, "text": "long-first", "intent_id": 0},
        {"id": "b" * 64, "text": "short-second", "intent_id": 15},
    ]
    order = tuple(range(150))
    view = types.SimpleNamespace(classes=classes, class_orders={"17": order})
    cfg = {"max_sequence_length": 128, "batch_size": 8, "seed": 17}

    fixed = r2._evaluate(None, None, rows, order, 1, tokenizer, view, cfg, torch, "cpu", FakeBudget())
    assert fixed["metrics"]["seen"]["accuracy"] == 1.0
    assert fixed["metrics"]["old"]["accuracy"] == 1.0
    assert fixed["metrics"]["current"]["accuracy"] == 1.0
    assert fixed["metrics"]["old"]["count"] == 1
    assert fixed["metrics"]["current"]["count"] == 1

    # The archived R1 aggregation is expected to reproduce the diagnosed bug:
    # short prompts are scored first, then zipped to the original long-first rows.
    broken = r1._evaluate(None, None, rows, order, 1, tokenizer, view, cfg, torch, "cpu", FakeBudget())
    assert broken["metrics"]["seen"]["accuracy"] == 0.0
    assert broken["metrics"]["old"]["accuracy"] == 0.0
    assert broken["metrics"]["current"]["accuracy"] == 0.0
