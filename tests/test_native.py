import io
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from nested_sleep.native import (DeepSelfModifyingTitans, FunctionalCMS, NativeConfig,
                                 NativeHOPELM, deep_map, deep_partials, feature_dgd_step, tree_map)


@pytest.fixture(params=['cpu'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(29)
    return request.param


def test_deep_partials_against_independent_autograd(device):
    wi = torch.randn(2, 3, 4, dtype=torch.double, device=device, requires_grad=True)
    wo = torch.randn(2, 4, 3, dtype=torch.double, device=device, requires_grad=True)
    key, target = torch.randn(2, 4, dtype=torch.double, device=device), torch.randn(2, 4, dtype=torch.double, device=device)
    # Independent expression and autograd, not the production helper.
    y = key + torch.bmm(wo, F.silu(torch.bmm(wi, key[..., None]))).squeeze(-1)
    expected = torch.autograd.grad(0.5 * (y - target).square().sum(), (wi, wo), create_graph=True)
    actual, _ = deep_partials((wi, wo), key, target)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


def test_feature_dgd_numpy_finite_difference_oracle(device):
    # Finite-difference loss derivative and dense products in NumPy.
    rng = np.random.default_rng(43)
    wi, wo = rng.normal(size=(3, 2)) * .1, rng.normal(size=(2, 3)) * .1
    k, target = rng.normal(size=2), rng.normal(size=2)
    def loss(a, b):
        u = a @ k
        return .5 * np.square(k + b @ (u / (1 + np.exp(-u))) - target).sum()
    gradients = []
    for which, source in enumerate((wi, wo)):
        g = np.zeros_like(source)
        for index in np.ndindex(source.shape):
            plus, minus = source.copy(), source.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            g[index] = ((loss(plus, wo) - loss(minus, wo)) if which == 0 else
                        (loss(wi, plus) - loss(wi, minus))) / 2e-6
        gradients.append(g)
    h = (wi @ k) / (1 + np.exp(-(wi @ k)))
    expected = [w @ (.99 * np.eye(len(phi)) - .02 * np.outer(phi, phi)) - .02 * g
                for w, phi, g in zip((wi, wo), (k, h), gradients)]
    tensor = lambda x: torch.tensor(x, dtype=torch.double, device=device)
    weights = tuple(tensor(x)[None] for x in (wi, wo))
    actual = feature_dgd_step(weights, weights, tensor(k)[None], tensor(target)[None], tensor([.02]), tensor([.99]))
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a[0].detach().cpu().numpy(), e, rtol=1e-8, atol=1e-9)


def test_deep_update_outer_gradient(device):
    w = tuple(torch.randn(1, 2, 2, dtype=torch.double, device=device, requires_grad=True) * .05 for _ in range(2))
    k = torch.tensor([[.3, -.2]], dtype=torch.double, device=device)
    v = torch.tensor([[.1, .2]], dtype=torch.double, device=device)
    eta, alpha = torch.tensor([.01], device=device), torch.tensor([.99], device=device)
    assert torch.autograd.gradcheck(lambda a, b: torch.cat([x.flatten() for x in feature_dgd_step(
        (a, b), (a, b), k, v, eta, alpha)]), w, fast_mode=True)


def test_outer_gradient_through_self_generated_target(device):
    tensors = tuple((torch.randn(1, 2, 2, dtype=torch.double, device=device) * .05).requires_grad_()
                    for _ in range(4))
    x = torch.tensor([[.2, -.1]], dtype=torch.double, device=device)
    k = torch.tensor([[.3, .4]], dtype=torch.double, device=device)
    eta, alpha = torch.tensor([.01], device=device), torch.tensor([.99], device=device)
    def algorithm(wi, wo, vi, vo):
        generated_value = deep_map(x, (vi, vo))
        self_target = deep_map(generated_value, (wi, wo))
        updated = feature_dgd_step((wi, wo), (wi, wo), k, self_target, eta, alpha)
        return deep_map(x, updated)
    assert torch.autograd.gradcheck(algorithm, tensors, fast_mode=True)


def test_streaming_partial_chunks_and_checkpoint(device):
    cfg = NativeConfig(vocab_size=12, dim=4, hidden=5, projection_chunk=4, memory_chunk=2)
    model = NativeHOPELM(cfg).to(device).double()
    tokens = torch.randint(0, 12, (2, 9), device=device)
    with torch.no_grad():
        whole, final = model(tokens, stream_ids=['a', 'b'])
        a, state = model(tokens[:, :3], stream_ids=['a', 'b'])
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True, map_location=device)
        b, final2 = model(tokens[:, 3:], restored, stream_ids=['a', 'b'])
    torch.testing.assert_close(torch.cat([a, b], 1), whole, rtol=0, atol=0)
    for name in model.blocks[0].smt.boxes:
        for p, q in zip(final['blocks'][0]['smt']['weights'][name], final2['blocks'][0]['smt']['weights'][name]):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    with pytest.raises(ValueError, match='identity'):
        model(tokens[:, :1], state, stream_ids=['b', 'a'])


def test_future_tokens_do_not_change_prefix(device):
    model = NativeHOPELM(NativeConfig(vocab_size=12, dim=4, hidden=5)).to(device)
    a = torch.randint(0, 12, (1, 8), device=device)
    b = a.clone()
    b[:, 4:] = (b[:, 4:] + 1) % 12
    with torch.no_grad():
        x, _ = model(a)
        y, _ = model(b)
    torch.testing.assert_close(x[:, :4], y[:, :4], rtol=0, atol=0)


def test_padding_and_batch_state_are_isolated(device):
    model = NativeHOPELM(NativeConfig(vocab_size=12, dim=4, hidden=5)).to(device)
    a = torch.tensor([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]], device=device)
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool, device=device)
    with torch.no_grad():
        batched, state = model(a, valid=mask)
        single, _ = model(a[:1, 2:])
    torch.testing.assert_close(batched[:1, 2:], single, rtol=2e-5, atol=2e-6)
    assert state['blocks'][0]['smt']['count'].tolist() == [3, 5]


def test_cms_accumulation_and_meta_gradient(device):
    cfg = NativeConfig(dim=3, hidden=4, cms_periods=(1, 2), cms_lr=.01)
    cms = FunctionalCMS(cfg).to(device).double()
    state = cms.initial_state(1)
    x = torch.tensor([[.2, .3, -.1]], dtype=torch.double, device=device)
    old = state['weights'][1][0].clone()
    for event in range(2):
        y = cms(x, state)
        loss = (y - .1).square().sum()
        state = cms.adapt(loss, state, higher_order=True)
        if event == 0:
            torch.testing.assert_close(old, state['weights'][1][0], rtol=0, atol=0)
            assert state['events'] == [0, 1]
    assert state['events'] == [0, 0]
    assert not torch.equal(old, state['weights'][1][0])
    outer = cms(x, state).square().sum()
    gradients = torch.autograd.grad(outer, tuple(cms.parameters()))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_complete_native_lm_backward_and_episodic_adapt(device):
    model = NativeHOPELM(NativeConfig(vocab_size=12, dim=4, hidden=5, cms_periods=(1, 2))).to(device)
    tokens = torch.randint(0, 12, (1, 7), device=device)
    logits, state = model(tokens[:, :3])
    inner_loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:4].flatten())
    state = model.adapt_cms(inner_loss, state)
    logits2, _ = model(tokens[:, 3:6], state)
    loss = F.cross_entropy(logits2.flatten(0, 1), tokens[:, 4:7].flatten())
    loss.backward()
    for name in ['k', 'v', 'eta', 'alpha', 'memory']:
        grads = [p.grad for p in model.blocks[0].smt.maps[name].parameters()]
        assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert model.blocks[0].cms.levels[0].wi.grad.abs().sum() > 0
