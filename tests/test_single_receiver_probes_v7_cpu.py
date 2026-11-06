import pytest
import torch
from torch import nn

from tools.single_receiver_probes_v7 import gradient_conflict, slow_output_lesion


class Bank(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(2)])


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.bank = Bank()
        self.route = nn.Parameter(torch.tensor([0.2, 0.8]))

    def banks(self):
        return [nn.Identity(), self.bank]


def test_lesion_masks_output_preserves_route_and_cleans_hook():
    model = Tiny()
    x = torch.ones(1, 2)
    before = model.bank.slots[0](x)
    route = model.route.detach().clone()
    weights = [p.detach().clone() for p in model.parameters()]
    with slow_output_lesion(model, [0]):
        assert torch.equal(model.bank.slots[0](x), torch.zeros_like(before))
        assert torch.equal(model.route, route)
    assert torch.equal(model.bank.slots[0](x), before)
    assert torch.equal(model.route, route)
    assert all(torch.equal(p, saved) for p, saved in zip(model.parameters(), weights))
    assert not model.bank.slots[0]._forward_hooks
    with pytest.raises(RuntimeError):
        with slow_output_lesion(model, [0]):
            raise RuntimeError("probe")
    assert not model.bank.slots[0]._forward_hooks


@pytest.mark.parametrize("old_coef,new_coef,expected", [(1., -1., -1.)])
def test_gradient_cosine_known_cases_and_restores_flags(old_coef, new_coef, expected):
    p = nn.Parameter(torch.tensor([1., 2.]), requires_grad=False)
    model = nn.Module()
    old = lambda: (p * old_coef).sum()
    new = lambda: (p * new_coef).sum()
    rng = torch.random.get_rng_state().clone()
    result = gradient_conflict(model, old, new, {"g": [("p", p)]})["g"]
    assert result["cosine"] == pytest.approx(expected)
    assert result["numel"] == 2
    assert p.requires_grad is False
    assert p.grad is None
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_zero_gradient_cosine_is_none():
    p = nn.Parameter(torch.tensor([2.]))
    result = gradient_conflict(None, lambda: (p * 0).sum(), lambda: (p * 3).sum(),
                               {"g": [("p", p)]})["g"]
    assert result["old_norm"] == 0
    assert result["cosine"] is None


def test_perpendicular_gradients_have_zero_cosine():
    p = nn.Parameter(torch.tensor([1., 1.]))
    result = gradient_conflict(None, lambda: p[0], lambda: p[1],
                               {"g": [("p", p)]})["g"]
    assert result["dot"] == 0
    assert result["cosine"] == pytest.approx(0.)
