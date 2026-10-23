"""Distribution/reward kernels for Sleep; sampling and RL loops are not implied."""
import torch
from torch.nn import functional as F


def token_forward_kl(teacher_logits, student_logits, mask):
    if teacher_logits.shape != student_logits.shape or mask.shape != student_logits.shape[:-1]:
        raise ValueError('unaligned teacher/student/mask')
    if not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()) or not bool(mask.sum() > 0):
        raise ValueError('mask must have positive finite mass')
    log_p = F.log_softmax(teacher_logits.detach(), dim=-1)
    log_q = F.log_softmax(student_logits, dim=-1)
    per_token = (log_p.exp() * (log_p - log_q)).sum(-1)
    return (per_token * mask).sum() / mask.sum()


def imitation_policy_loss(sequence_log_prob, rewards, *, baseline=0.0):
    """Negative score-function reward estimator: minimize to maximize reward.

    Policy trajectories must be sampled outside this function; no gradient through
    rewards/sampling. This is a local oracle, not a claim that paper specifies REINFORCE.
    """
    if sequence_log_prob.shape != rewards.shape:
        raise ValueError('one detached reward per sampled trajectory')
    return -((rewards.detach() - baseline) * sequence_log_prob).mean()


def edit_reward(reference, candidate, *, raw_distance_threshold):
    """Sleep Eq 4, token sequences and a RAW edit-distance threshold."""
    if raw_distance_threshold < 0:
        raise ValueError('threshold must be nonnegative')
    row = list(range(len(candidate) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(candidate, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = current
    distance = row[-1]
    if distance > raw_distance_threshold:
        return 0.0
    return 1.0 - distance / max(len(reference), len(candidate), 1)
