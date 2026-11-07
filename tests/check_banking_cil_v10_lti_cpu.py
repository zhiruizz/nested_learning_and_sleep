"""Banking LTI reward matches the published full-v8 Eq. 4 implementation."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import banking_cil_v10_runtime as banking
import run_clinc_cil_v8_full as clinc_full


CLASSES = tuple(f"bank label {index:02d}" for index in range(77))
NORMALIZER = SimpleNamespace(normalize_text=banking._normalized_label)


@pytest.mark.parametrize("left,right,z0", [
    ([], [], 1),
    ([], [4, 5], 2),
    ([1, 2, 3], [1, 9, 3], 1),
    ([1, 2], [7, 8, 9], 1),
    ([1, 2, 3, 4], [4, 3, 2, 1], 8),
])
def test_edit_similarity_matches_full_v8(left, right, z0):
    assert banking.paper_edit_similarity(left, right, z0=z0) == clinc_full.paper_edit_similarity(
        left, right, z0=z0)


def test_local_lti_reward_matches_full_v8_text_and_token_semantics():
    cases = [
        ("Bank_Label_03", "bank label 03", [5, 6], [5, 6], 2),
        ("bank label 03", "bank label 04", [5, 6], [5, 7], 1),
        ("not a canonical label", "not a canonical label", [5, 6], [8, 9], 1),
        ("", "", [], [], 1),
    ]
    for teacher, student, teacher_tokens, student_tokens, z0 in cases:
        expected = clinc_full.full_lti_reward(
            teacher, student, teacher_tokens=teacher_tokens,
            student_tokens=student_tokens, z0=z0, canonical_labels=CLASSES,
            r2=NORMALIZER)
        actual = banking.local_lti_reward(
            teacher, student, teacher_tokens=teacher_tokens,
            student_tokens=student_tokens, z0=z0, canonical_labels=CLASSES)
        assert actual == expected


def test_reward_rejects_invalid_weights_and_class_vocabulary():
    with pytest.raises(ValueError, match="weights"):
        banking.local_lti_reward("x", "x", teacher_tokens=[1], student_tokens=[1],
                                  z0=1, canonical_labels=CLASSES,
                                  semantic_weight=float("nan"))
    with pytest.raises(ValueError, match="77 unique canonical"):
        banking.local_lti_reward("x", "x", teacher_tokens=[1], student_tokens=[1],
                                  z0=1, canonical_labels=CLASSES[:-1])
