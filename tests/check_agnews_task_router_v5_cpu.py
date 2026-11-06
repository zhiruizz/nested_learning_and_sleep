"""Small CPU-only tests for the task-router control's pure math/aggregation."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import probe_agnews_task_router_v5 as probe


class TaskRouterMathTests(unittest.TestCase):
    def test_fixed_dual_ridge_learns_only_train_group_boundary(self):
        train = np.asarray([
            [-2.0, 1.0], [-1.5, 0.7], [-1.0, 1.2], [-0.5, 0.8],
            [0.5, -0.8], [1.0, -1.2], [1.5, -0.7], [2.0, -1.0],
        ])
        classes = np.asarray([0, 1, 0, 1, 2, 3, 2, 3])
        gate = probe.fit_group_ridge(train, classes)
        predicted = probe.predict_group_ridge(train, gate)
        np.testing.assert_array_equal(predicted, classes >= 2)
        self.assertEqual(gate["regularization"], 1.0)
        self.assertEqual(gate["n_train"], 8)
        self.assertEqual(len(gate["model_sha256"]), 64)
        # The API accepts only fit inputs; held-out statistics cannot be mixed in.
        self.assertNotIn("validation", gate)

    def test_zero_signal_is_old_at_strict_positive_threshold(self):
        train = np.ones((8, 4), dtype=np.float64)
        classes = np.asarray([0, 1, 0, 1, 2, 3, 2, 3])
        gate = probe.fit_group_ridge(train, classes)
        pred = probe.predict_group_ridge(np.ones((3, 4)), gate)
        np.testing.assert_array_equal(pred, np.zeros(3, dtype=bool))
        self.assertEqual(gate["weight_l2"], 0.0)

    def test_aggregation_keeps_group_denominators_and_only_metric_fields(self):
        rows = []
        for cls in range(4):
            for n in range(2):
                rows.append({"class_id": cls, probe.EM: int(n == 0),
                             "marker_accuracy": 1, "class_given_marker_accuracy": int(n == 0),
                             "candidate_accuracy": int(n == 0), "marker_nll": 0.1 * cls,
                             "class_nll": 0.2 * n, "answer_nll_nats_per_token": 0.1 * cls + 0.1 * n})
        aggregate = probe.aggregate_rows(rows)
        self.assertEqual(aggregate["all"]["examples"], 8)
        self.assertEqual(aggregate["old"]["examples"], 4)
        self.assertEqual(aggregate["new"]["examples"], 4)
        self.assertEqual(aggregate["class3"]["examples"], 2)
        self.assertEqual(set(aggregate["all"]), {"examples", *probe.METRIC_KEYS})

    def test_task_id_oracle_confusion_is_group_only(self):
        classes = np.asarray([0, 1, 2, 3, 0, 3])
        predicted = classes >= 2
        self.assertEqual(probe.gate_confusion(classes, predicted), {
            "actual_old_pred_old": 3, "actual_old_pred_new": 0,
            "actual_new_pred_old": 0, "actual_new_pred_new": 3,
        })

    def test_feature_hook_does_not_add_forward_and_preserves_reference_metrics(self):
        class Original(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                return x

        class MockLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                original = Original()
                self.backbone = SimpleNamespace(model=SimpleNamespace(
                    layers=[SimpleNamespace(mlp=SimpleNamespace(original=original))]))
                self.original = original
                self.calls = 0

            def logits(self, ids, mask):
                self.calls += 1
                meter = probe.c.engine.ACTIVE_METER.get()
                if meter is not None:
                    meter.charge(ids)
                features = ids.float().unsqueeze(-1).expand(-1, -1, 4)
                hidden = self.original(features)
                vocab_weights = torch.arange(1, 17, device=ids.device, dtype=torch.float32)
                return hidden[..., :1] * vocab_weights

        pack = {
            "tokenizer": {"pad_token_id": 0},
            "code_maps": {"17": {"marker_token_id": 3, "class_to_token_ids": [4, 5, 6, 7]}},
        }
        rows = [
            {"tokens": [1, 2, 3], "class_id": 0},
            {"tokens": [1, 4], "class_id": 1},
            {"tokens": [1, 5, 6], "class_id": 2},
            {"tokens": [2, 4], "class_id": 3},
        ]

        def score(capture):
            model = MockLM()
            meter = probe.c.engine.WorkMeter()
            token = probe.c.engine.ACTIVE_METER.set(meter)
            try:
                metrics, per_row, features = probe.score_rows(
                    model, rows, pack, 17, "mock", meter, capture_features=capture)
            finally:
                probe.c.engine.ACTIVE_METER.reset(token)
            self.assertEqual(model.calls, 1)
            self.assertEqual(model.original.calls, 1)
            self.assertEqual(meter.counts["mock"]["forward_calls"], 1)
            self.assertEqual(meter.counts["mock"]["gradient_forward_input_tokens"], 0)
            return metrics, per_row, features

        plain_metrics, plain_rows, _ = score(False)
        hooked_metrics, hooked_rows, features = score(True)
        self.assertEqual(plain_metrics, hooked_metrics)
        self.assertEqual(plain_rows, hooked_rows)
        self.assertEqual(features.shape, (len(rows), 4))


if __name__ == "__main__":
    unittest.main()
