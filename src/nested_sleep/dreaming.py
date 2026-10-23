"""Independent Dreaming selection/trial primitives, NOT complete model Dreaming.

Missing here: LM generation, real random-expert routing, LoRA SFT, ReSTEM policy
training. Callbacks let isolation/selection/reward be verified without these claims.
"""
from copy import deepcopy
import math
import random
import torch


def gradient_norm_score(loss, parameters):
    """Explicit scalarization CHOICE; autograd.grad never fills .grad buffers."""
    params = list(parameters)
    if not params:
        raise ValueError('gradient scoring requires parameters')
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return sum(float(g.detach().double().square().sum()) for g in grads if g is not None) ** 0.5


def select_dreams(scores, *, top_k, random_count, seed):
    if top_k < 0 or random_count < 0 or top_k + random_count > len(scores):
        raise ValueError('invalid selection budget')
    if not all(math.isfinite(s) for s in scores):
        raise ValueError('nonfinite score')
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    selected = order[:top_k]
    return selected + random.Random(seed).sample(order[top_k:], random_count)


def isolated_trials(base_model, dreams, adapt, score_training_task):
    """One deep copy per candidate from the SAME base; never share trial updates.

    Caller owns reproducible callback RNG and must provide a training/dev task,
    never the held-out final test set. Returns binary Eq 5 reward and measured gain.
    """
    baseline = float(score_training_task(deepcopy(base_model)))
    if not math.isfinite(baseline):
        raise ValueError('nonfinite baseline')
    results = []
    for dream in dreams:
        trial = deepcopy(base_model)
        adapt(trial, dream)
        score = float(score_training_task(trial))
        if not math.isfinite(score):
            raise ValueError('nonfinite trial score')
        results.append({'reward': int(score > baseline), 'gain': score - baseline})
    return results
