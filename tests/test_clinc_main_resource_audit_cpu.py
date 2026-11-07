import unittest

import torch

from tools import audit_clinc_main_resources_v8 as audit


class TestClincMainResourceAuditCpu(unittest.TestCase):
    def _context(self):
        memory = {
            "0.a": torch.zeros((2, 3)),
            "0.b": torch.zeros((3, 2)),
            "0.experts.slots.0.a": torch.zeros((2, 3)),
            "0.experts.slots.0.b": torch.zeros((3, 2)),
            "0.experts.slots.0.route": torch.zeros((3,)),
            "0.experts.slots.0.priority": torch.zeros(()),
            "0.experts.slots.0.active": torch.tensor(True),
            "0.experts.slots.0.generation": torch.tensor(1, dtype=torch.long),
        }
        trainable = {"memory." + key: key.endswith((".experts.slots.0.a", ".experts.slots.0.b"))
                     for key in memory if key.endswith((".a", ".b"))}
        trainable.update({"readout_a": False, "readout_b": False})
        return {"memory": memory, "readout_a": torch.zeros((1, 3)),
            "readout_b": torch.zeros((5, 1)), "readout_fp32_accumulation": True,
            "routing": [None], "trainable": trainable}

    def test_context_inventory_separates_params_buffers_and_trainable_scope(self):
        result = audit.context_inventory(self._context(), torch, levels=1, capacity=1)
        self.assertEqual(result["serialized_mutable_parameter_numel"], 32)
        self.assertEqual(result["serialized_trainable_parameter_numel"], 12)
        self.assertEqual(result["serialized_buffer_tensor_count"], 4)
        self.assertEqual(result["active_slots_by_level"], [1])
        self.assertEqual(result["fresh_receiver_slot_A_B_numel"], 12)
        self.assertEqual(result["context_tensor_payload_bytes"],
                         result["serialized_mutable_parameter_tensor_bytes"] +
                         result["serialized_mutable_buffer_tensor_bytes"])

    def test_work_histograms_must_reconcile_to_recorded_calls(self):
        # One cache can be consumed by multiple teacher/student tail contexts.
        work = {"full_backbone_forward_calls": 1, "tail_calls": 2,
            "tail_grad_enabled_calls": 1,
            "tail_no_grad_calls": 1,
            "forward_input_shape_histogram": {"1x2": 1},
            "tail_shape_histogram": {"1x2": 2},
            "tail_grad_shape_histogram": {"1x2": 1},
            "tail_no_grad_shape_histogram": {"1x2": 1}}
        audit.validate_work_histograms(work, label="fixture")
        work["tail_calls"] = 3
        with self.assertRaisesRegex(ValueError, "histogram sums disagree"):
            audit.validate_work_histograms(work, label="fixture")
        work["tail_calls"] = 2
        work["tail_shape_histogram"] = {"1x2": 1}
        with self.assertRaisesRegex(ValueError, "histogram sums disagree"):
            audit.validate_work_histograms(work, label="fixture")

    def test_checkpoint_receipt_wall_may_advance_but_discrete_work_must_match(self):
        audit.validate_checkpoint_receipt_work({"active_wall_seconds": 1.0, "calls": 4},
            {"active_wall_seconds": 1.2, "calls": 4}, label="fixture")
        with self.assertRaisesRegex(ValueError, "discrete work counters differ"):
            audit.validate_checkpoint_receipt_work({"active_wall_seconds": 1.0, "calls": 4},
                {"active_wall_seconds": 1.2, "calls": 5}, label="fixture")


if __name__ == "__main__":
    unittest.main()
