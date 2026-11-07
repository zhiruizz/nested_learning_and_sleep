"""CPU-only tests for class-count 77 scoring and generic event helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import banking_cil_common_v10 as common
import banking_cil_v10_runtime as runtime


CLASSES = tuple(f"class_{i:02d}" for i in range(77))


class ToyTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        # Stable byte-like encoding, avoiding specials in segmented text.
        return [ord(ch) + 3 for ch in str(text)]

    def decode(self, ids, skip_special_tokens=True):
        chars = []
        for token in ids:
            token = int(token)
            if 3 <= token <= 258:
                chars.append(chr(token - 3))
        return "".join(chars)


class FakeTailRuntime:
    def __init__(self):
        self.capture_calls = 0
        self.tail_calls = 0
        self.vocab = 512

    def capture(self, model, ids, mask, *, device=None):
        self.capture_calls += 1
        return SimpleNamespace(ids=ids.detach().clone())

    def tail_logits(self, model, context, cache):
        self.tail_calls += 1
        value = torch.zeros((*cache.ids.shape, self.vocab), device=cache.ids.device)
        if isinstance(context, dict) and "scale" in context:
            value[..., 0] = context["scale"]
        return value

    @staticmethod
    def causal_token_ce(logits, ids, mask):
        losses = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                   ids[:, 1:].reshape(-1), reduction="none")
        selected = mask[:, 1:].reshape(-1).bool()
        return losses[selected].mean(), int(selected.sum())

    @staticmethod
    def causal_token_kl(student, teacher, mask, *, temperature=1.0):
        loss = ((student[:, :-1] - teacher[:, :-1].detach()) ** 2).mean()
        return loss, int(mask[:, 1:].sum())


def fake_common():
    class Common:
        @staticmethod
        def tokenize_prompt(text, tokenizer, classes, eos_token_id, max_length=None):
            return {"input_ids": [11, 12], "prompt_length": 2}

        @staticmethod
        def tokenize_row(row, tokenizer, classes, eos_token_id, max_length=None):
            return {"input_ids": [11, 12, 50, eos_token_id],
                    "labels": [-100, -100, 50, eos_token_id], "prompt_length": 2}

    return Common


class Budget:
    def __init__(self): self.checks = self.backwards = self.updates = 0
    def check(self): self.checks += 1
    def backward(self, mask): self.backwards += int(mask.sum())
    def update(self): self.updates += 1


def test_teacher_forced_loglikelihood_excludes_prefix_and_includes_eos():
    ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.zeros((1, 4, 5))
    logits[0, 0, 1] = 4.0   # prefix target; ignored
    logits[0, 1, 2] = 1.0
    logits[0, 2, 3] = 2.0
    logits[0, 3, 4] = 3.0   # answer EOS prediction
    total, mean = runtime.teacher_forced_scores(logits[:, :-1], ids, [2], [2])
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    expected = lp[0, 1, 2] + lp[0, 2, 3]
    assert torch.allclose(total[0], expected)
    assert torch.allclose(mean[0], expected / 2)


def test_all_77_candidates_share_each_capture_across_contexts_and_no_task_mask():
    tokenizer, tail = ToyTokenizer(), FakeTailRuntime()
    rows = [{"id": "ex-1", "text": "hello", "intent_id": 0}]
    contexts = {"base": {}, "warm": {}}
    result = runtime.score_candidates(object(), contexts, rows, tokenizer, CLASSES,
                                      fake_common(), tail, torch=torch, device="cpu",
                                      batch_size=16, class_order=tuple(range(77)), task_id=3)
    assert result["candidate_sequences"] == 77
    assert result["candidate_correct_counts"]["base"]["candidate_count_per_example"] == 77
    assert result["work"]["candidate_backbone_forwards"] == 5
    assert result["work"]["candidate_tail_calls"] == 10
    assert tail.capture_calls == 5 and tail.tail_calls == 10
    assert result["candidate_metric_primary"] == "global_77_sum_log_likelihood"
    assert set(result["candidate_metrics"]["base"]["teacher_forced_sum"]) >= {"seen", "current", "unseen"}


def test_old_current_unseen_report_groups_without_candidate_task_mask():
    rows = [{"intent_id": 0}, {"intent_id": 11}, {"intent_id": 22}]
    ids = [0, 11, 22]
    metrics, gold = runtime._score_rows({}, ids, ids, [1.0, 2.0, 3.0], runtime._RowsWithTask(rows, 1),
                                        CLASSES, tuple(range(77)))
    groups = metrics["teacher_forced_sum"]
    assert (groups["old"]["count"], groups["current"]["count"], groups["unseen"]["count"]) == (1, 1, 1)
    assert groups["per_class"]["11"]["accuracy"] == 1.0
    assert gold["all77"]["mean_token_nll"] == 2.0


def test_global_77_requirement_and_rollout_eos_cap_fail_closed():
    with pytest.raises(ValueError, match="77 unique"):
        runtime.score_candidates(object(), {"ctx": {}}, [{"text": "x", "intent_id": 0}],
                                 ToyTokenizer(), CLASSES[:15], fake_common(), FakeTailRuntime(),
                                 torch=torch, device="cpu")
    tokenizer = ToyTokenizer()
    long_label = CLASSES[0]
    tokenizer.encode = lambda value, add_special_tokens=False: [8] * (9 if value == long_label else 1)
    with pytest.raises(ValueError, match="greedy cap"):
        runtime.score_candidates(object(), {"ctx": {}}, [{"text": "x", "intent_id": 0}],
                                 tokenizer, CLASSES, fake_common(), FakeTailRuntime(),
                                 torch=torch, device="cpu", include_greedy=True, max_new_tokens=9)


def test_zero_reward_lti_keeps_nonzero_kl_update():
    tokenizer, tail, budget = ToyTokenizer(), FakeTailRuntime(), Budget()
    student_scale = torch.tensor(1.0, requires_grad=True)
    contexts = {"teacher": {"scale": torch.tensor(0.0)},
                "student": {"scale": student_scale}}

    class Transaction:
        generator = torch.Generator(device="cpu").manual_seed(7)
        receiver, slot = 1, 0
        counters = {"zero_gradient_steps": 0}
        def teacher_context(self): return contexts["teacher"]
        def student_context(self): return contexts["student"]
        def __init__(self): self.optimized = []
        def optimize(self, loss, kind, tokens): self.optimized.append((float(loss.detach()), kind, tokens))

    class Sample:
        token_ids = torch.tensor([[11, 12, 40, 2]])
        def answer_mask(self): return torch.tensor([[False, False, True, True]])

    def sample_tokens(model, context, prompt, max_tokens, temperature, generator, eos, torch, device, budget):
        return Sample(), [40, eos]

    tx = Transaction()
    result = runtime.run_sleep_event(object(), tx, [("lti", {"id": "id-1", "text": "text", "intent_id": 0}, "current")],
        tokenizer, CLASSES, fake_common(), {"max_new_tokens": 10, "temperature": 1.0,
        "max_sequence_length": 128, "lti_edit_distance_z0": 2}, torch, "cpu", budget, tail,
        sample_tokens=sample_tokens, lti_reward=lambda *args, **kwargs: 0.0)
    assert result["zero_reward_steps"] == 1
    assert result["skipped_zero_objective_steps"] == 0
    assert result["per_step"][0]["optimizer_updated"] is True
    assert tx.optimized[0][1] == "lti" and tx.optimized[0][0] > 0
    assert budget.updates == 1


def test_exact_zero_reward_and_zero_kl_skips_optimizer_and_logs_objective_skip():
    tokenizer, tail, budget = ToyTokenizer(), FakeTailRuntime(), Budget()
    contexts = {"teacher": {"scale": torch.tensor(0.0)},
                "student": {"scale": torch.tensor(0.0, requires_grad=True)}}

    class Transaction:
        generator = torch.Generator(device="cpu").manual_seed(7)
        receiver, slot = 1, 0
        counters = {"zero_gradient_steps": 0}
        def teacher_context(self): return contexts["teacher"]
        def student_context(self): return contexts["student"]
        def __init__(self): self.optimized = []
        def optimize(self, loss, kind, tokens): self.optimized.append((loss, kind, tokens))

    class Sample:
        token_ids = torch.tensor([[11, 12, 40, 2]])
        def answer_mask(self): return torch.tensor([[False, False, True, True]])

    def sample_tokens(model, context, prompt, max_tokens, temperature, generator, eos, torch, device, budget):
        return Sample(), [40, eos]

    tx = Transaction()
    result = runtime.run_sleep_event(object(), tx, [("lti", {"id": "id-2", "text": "text", "intent_id": 0}, "current")],
        tokenizer, CLASSES, fake_common(), {"max_new_tokens": 10, "temperature": 1.0,
        "max_sequence_length": 128, "lti_edit_distance_z0": 2}, torch, "cpu", budget, tail,
        sample_tokens=sample_tokens, lti_reward=lambda *args, **kwargs: 0.0)
    assert result["skipped_zero_objective_steps"] == 1
    assert result["per_step"][0]["optimizer_updated"] is False
    assert tx.optimized == []
    assert budget.updates == 0


def test_eos_only_lti_rollout_is_logged_without_saving_text_and_keeps_kl_update():
    tokenizer, tail, budget = ToyTokenizer(), FakeTailRuntime(), Budget()
    student_scale = torch.tensor(1.0, requires_grad=True)
    contexts = {"teacher": {"scale": torch.tensor(0.0)},
                "student": {"scale": student_scale}}

    class Transaction:
        generator = torch.Generator(device="cpu").manual_seed(11)
        receiver, slot = 1, 0
        counters = {"zero_gradient_steps": 0}
        def __init__(self): self.optimized = []
        def teacher_context(self): return contexts["teacher"]
        def student_context(self): return contexts["student"]
        def optimize(self, loss, kind, tokens): self.optimized.append((float(loss.detach()), kind, tokens))

    class EosSample:
        token_ids = torch.tensor([[11, 12, tokenizer.eos_token_id]])
        def answer_mask(self): return torch.tensor([[False, False, True]])

    def eos_only(*args, **kwargs):
        return EosSample(), [tokenizer.eos_token_id]

    tx = Transaction()
    result = runtime.run_sleep_event(
        object(), tx, [("lti", {"id": "eos-only", "text": "text", "intent_id": 0}, "current")],
        tokenizer, CLASSES, fake_common(),
        {"max_new_tokens": 10, "temperature": 1.0, "max_sequence_length": 128,
         "lti_edit_distance_z0": 8}, torch, "cpu", budget, tail,
        sample_tokens=eos_only, lti_reward=runtime.local_lti_reward)
    step = result["per_step"][0]
    assert step["sample_owner"] == "teacher"
    assert step["sampled_answer_tokens_including_eos"] == 1
    assert step["sampled_first_token_eos"] is True
    assert step["decoded_sample_empty"] is True
    assert step["teacher_answer_empty"] is True
    assert step["student_full_answer_empty"] is True
    assert result["lti_answer_diagnostics"] == {
        "count": 1, "teacher_answer_empty_count": 1,
        "student_full_answer_empty_count": 1}
    assert result["sampled_rollouts_by_owner"]["teacher"]["first_token_eos_count"] == 1
    assert result["sampled_rollouts_by_owner"]["teacher"]["decoded_empty_count"] == 1
    assert tx.optimized and tx.optimized[0][0] > 0  # KL still trains when both decoded strings are empty.
    assert not any("text" in str(key).lower() or "token_id" in str(key).lower()
                   for key in step)


def test_gold_ce_step_includes_eos_and_uses_caller_update_callback():
    tokenizer, tail, budget = ToyTokenizer(), FakeTailRuntime(), Budget()
    context = {"scale": torch.tensor(1.0, requires_grad=True)}
    applied = []
    result = runtime.run_gold_replay_step(object(), context,
        {"id": "example", "text": "text", "intent_id": 0}, tokenizer, CLASSES,
        fake_common(), {"max_sequence_length": 128}, torch, "cpu", budget, tail,
        apply_loss=lambda loss, count: applied.append((loss, count)))
    assert result["token_count"] == 2  # label token and EOS
    assert len(applied) == 1 and applied[0][1] == 2
    assert budget.backwards == 2 and budget.updates == 1


def test_route_summary_keeps_aggregate_scalars_only():
    summary = runtime._safe_route_summary({"receiver_answer_coverage": 0.5,
                                           "answer_tokens": 4,
                                           "selected": [1, 1, 0, 0],
                                           "prompt_text": "do not serialize"})
    assert summary == {"receiver_answer_coverage": 0.5, "answer_tokens": 4}

