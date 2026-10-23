from copy import deepcopy
import io

import pytest
import torch

from nested_sleep import (ConsolidationTransaction, GradientWindow,
                          LinearExpertChain, LinearSelfModifyingReference, dgd_linear_step)
from nested_sleep.consolidation import same_snapshot, snapshot
from nested_sleep.dreaming import gradient_norm_score, isolated_trials, select_dreams
from nested_sleep.objectives import edit_reward, imitation_policy_loss, token_forward_kl


@pytest.fixture(params=['cpu'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(17)
    return request.param


def test_dgd_against_autograd_objective(device):
    matrix = torch.randn(3, 2, dtype=torch.double, device=device)
    anchor = torch.randn_like(matrix).requires_grad_()
    key = torch.tensor([0.2, -0.7], dtype=torch.double, device=device)
    target = torch.tensor([0.3, 0.5, -0.1], dtype=torch.double, device=device)
    loss = 0.5 * (anchor @ key - target).square().sum()
    grad, = torch.autograd.grad(loss, (anchor,))
    # A dense explicit retention multiplication is independent of the kernel.
    retention = torch.tensor([[0.896, 0.014], [0.014, 0.851]], dtype=torch.double, device=device)
    expected = matrix @ retention - 0.1 * grad
    torch.testing.assert_close(dgd_linear_step(matrix, key, target, 0.1, 0.9, anchor=anchor), expected)


def test_dgd_outer_gradient_finite_difference(device):
    w = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)
    eta = torch.tensor(0.03, dtype=torch.double, device=device, requires_grad=True)
    k = torch.tensor([0.1, 0.4], dtype=torch.double, device=device)
    target = torch.tensor([0.2, -0.3], dtype=torch.double, device=device)
    assert torch.autograd.gradcheck(lambda m, e: dgd_linear_step(m, k, target, e, 0.98), (w, eta))


def test_dgd_extra_retention_at_zero_error(device):
    w = torch.eye(2, dtype=torch.double, device=device)
    k = torch.tensor([1., 0.], dtype=torch.double, device=device)
    torch.testing.assert_close(dgd_linear_step(w, k, w @ k, 0.2, 1.0),
                               torch.diag(torch.tensor([0.8, 1.], dtype=torch.double, device=device)))


def test_self_targets_use_old_state_and_read_before_write(device):
    model = LinearSelfModifyingReference(3).to(device=device, dtype=torch.double)
    x = torch.tensor([0.3, 0.4, -0.1], dtype=torch.double, device=device)
    state = model.initial_state()
    old = {k: v.clone() for k, v in state.items()}
    params = snapshot(model)
    output, new, event = model(x, state)
    q = torch.nn.functional.normalize(model.query @ x, dim=0)
    torch.testing.assert_close(output, old['memory'] @ q)
    for box in model.boxes:
        torch.testing.assert_close(event['targets'][box], old[box] @ (old['v'] @ x))
        assert not event['targets'][box].requires_grad
        torch.testing.assert_close(state[box], old[box])
        assert not torch.equal(new[box], old[box])
    assert same_snapshot(model, params)


def test_fresh_state_and_explicit_carry_are_distinct(device):
    model = LinearSelfModifyingReference(3).to(device)
    x = torch.ones(3, device=device)
    first, state, _ = model(x)
    fresh, _, _ = model(x)
    carry, _, _ = model(x, state)
    torch.testing.assert_close(first, fresh)
    assert not torch.allclose(first, carry)


def test_state_not_shared_across_interleaved_streams(device):
    model = LinearSelfModifyingReference(3).to(device)
    a, b = torch.randn(3, device=device), torch.randn(3, device=device)
    _, state_a, _ = model(a)
    frozen_a = {k: v.clone() for k, v in state_a.items()}
    model(b)
    for box in model.boxes:
        torch.testing.assert_close(state_a[box], frozen_a[box])
    y1, _, _ = model(a, state_a)
    y2, _, _ = model(a, frozen_a)
    torch.testing.assert_close(y1, y2)


def test_clock_accumulates_weighted_sum_without_averaging(device):
    parameter = torch.tensor([10.], device=device)
    clock = GradientWindow.create(parameter, 3)
    for value, rate in [(1., 0.1), (2., 0.2), (3., 0.3)]:
        clock = clock.accumulate(torch.tensor([value], device=device), rate)
    torch.testing.assert_close(clock.prospective(parameter), torch.tensor([8.6], device=device))
    torch.testing.assert_close(parameter, torch.tensor([10.], device=device))
    assert clock.consume().events == 0
    assert clock.consume().pending.abs().sum() == 0


def test_multiple_clock_boundaries(device):
    p = torch.ones(1, device=device)
    clocks = [GradientWindow.create(p, n) for n in [1, 2, 4]]
    observed = []
    for event in range(1, 9):
        due = []
        for i, clock in enumerate(clocks):
            clock = clock.accumulate(p, 0.01)
            if clock.due:
                due.append(i)
                clock = clock.consume()
            clocks[i] = clock
        observed.append(due)
    assert observed == [[0], [0, 1], [0], [0, 1, 2], [0], [0, 1], [0], [0, 1, 2]]


def test_clock_midwindow_checkpoint(device):
    p = torch.ones(2, device=device)
    clock = GradientWindow.create(p, 4).accumulate(2 * p, 0.1).accumulate(3 * p, 0.2)
    buffer = io.BytesIO()
    torch.save(clock.state_dict(), buffer)
    buffer.seek(0)
    restored = GradientWindow.from_state_dict(torch.load(buffer, weights_only=True, map_location=device))
    for gradient in [4 * p, 5 * p]:
        clock = clock.accumulate(gradient, 0.1)
        restored = restored.accumulate(gradient, 0.1)
    torch.testing.assert_close(restored.prospective(p), clock.prospective(p), rtol=0, atol=0)


def test_clock_rejects_double_consumption_and_overflow():
    p = torch.zeros(1)
    clock = GradientWindow.create(p, 2)
    with pytest.raises(RuntimeError):
        clock.prospective(p)
    with pytest.raises(RuntimeError):
        clock.consume()
    clock = clock.accumulate(p, 1).accumulate(p, 1)
    with pytest.raises(RuntimeError):
        clock.accumulate(p, 1)


def test_zero_expansion_preserves_initial_output(device):
    model = LinearExpertChain().to(device)
    x = torch.randn(3, 4, device=device)
    initial = model(x).detach()
    model.add_slot(1, 2, active=True)
    torch.testing.assert_close(model(x), initial, rtol=0, atol=0)


def make_transaction(device):
    live = LinearExpertChain().to(device)
    sender_slot = live.add_slot(0, 2, active=True)
    with torch.no_grad():
        sender_slot.b.fill_(0.2)
    live.add_slot(1, 2)  # dormant slot must never receive gradients or decay
    prospective = live.base[0].weight.detach() + 0.03
    return live, ConsolidationTransaction(live, 0, prospective)


def test_teacher_student_live_storage_and_sender_states(device):
    live, txn = make_transaction(device)
    assert bool(txn.teacher.slots[0][0].active)
    assert not bool(txn.student.slots[0][0].active)
    assert txn.student.slots[0][0].b.abs().sum() == 0
    for name, p in txn.teacher.named_parameters():
        assert p.data_ptr() != dict(txn.student.named_parameters())[name].data_ptr()
        assert p.data_ptr() != dict(live.named_parameters())[name].data_ptr()
        assert not p.requires_grad


def test_only_new_expert_updates_with_adamw(device):
    live, txn = make_transaction(device)
    teacher_before, live_before = snapshot(txn.teacher), snapshot(live)
    student_before = snapshot(txn.student)
    x = torch.randn(6, 4, device=device)
    with torch.no_grad():
        target = txn.teacher(x)
    opt = torch.optim.AdamW(txn.trainable_parameters(), lr=0.1, weight_decay=0.1)
    for _ in range(3):
        loss = (txn.student(x) - target).square().mean()
        loss.backward()
        for name, p in txn.student.named_parameters():
            if name not in txn.mutable:
                assert p.grad is None
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert same_snapshot(live, live_before)
    assert same_snapshot(txn.teacher, teacher_before)
    after = snapshot(txn.student)
    assert any(not torch.equal(student_before[k], after[k]) for k in txn.mutable)
    for k in after.keys() - txn.mutable:
        torch.testing.assert_close(after[k], student_before[k], rtol=0, atol=0)
    committed = txn.commit()
    assert not bool(committed.slots[0][0].active)
    assert bool(committed.slots[1][-1].active)
    assert same_snapshot(live, live_before)
    with pytest.raises(RuntimeError):
        txn.commit()


def test_abort_and_frozen_mutation_rejection(device):
    live, txn = make_transaction(device)
    before = snapshot(live)
    with torch.no_grad():
        txn.student.base[1].weight.add_(0.1)
    with pytest.raises(RuntimeError, match='frozen'):
        txn.commit()
    txn.abort()
    assert same_snapshot(live, before)


def test_multicycle_state_and_checkpoint_topology(device):
    live, txn = make_transaction(device)
    live = txn.commit()
    for _ in range(3):
        prospective = live.base[0].weight.detach() + 0.01
        txn = ConsolidationTransaction(live, 0, prospective)
        live = txn.commit()
    assert len(live.slots[1]) == 5  # dormant initial + four receiving cycles
    assert all(not bool(s.active) for s in live.slots[0])
    buffer = io.BytesIO()
    rng_state = torch.get_rng_state()
    torch.save({'topology': live.topology(), 'weights': live.state_dict(), 'rng': rng_state}, buffer)
    expected_random = torch.rand(4)
    buffer.seek(0)
    data = torch.load(buffer, weights_only=True, map_location='cpu')
    restored = LinearExpertChain.from_topology(data['topology']).to(device)
    restored.load_state_dict(data['weights'])
    torch.set_rng_state(data['rng'])
    torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
    x = torch.randn(2, 4, device=device)
    torch.testing.assert_close(restored(x), live(x), rtol=0, atol=0)


def test_kl_teacher_detached_and_future_masked(device):
    teacher = torch.randn(1, 3, 4, device=device, requires_grad=True)
    student = torch.randn(1, 3, 4, device=device, requires_grad=True)
    mask = torch.tensor([[1., 1., 0.]], device=device)
    loss = token_forward_kl(teacher, student, mask)
    loss.backward()
    assert teacher.grad is None
    assert student.grad[:, -1].abs().sum() == 0
    torch.testing.assert_close(token_forward_kl(teacher, teacher, mask), torch.zeros((), device=device))


def test_reward_loss_sign_and_stop_gradient(device):
    logit = torch.tensor(0., device=device, requires_grad=True)
    reward = torch.tensor([1.], device=device, requires_grad=True)
    loss = imitation_policy_loss(torch.nn.functional.logsigmoid(logit)[None], reward)
    loss.backward()
    assert logit.grad < 0  # gradient descent increases probability of rewarded token
    assert reward.grad is None


def test_raw_edit_threshold():
    assert edit_reward([1] * 20, [2] * 2 + [1] * 18, raw_distance_threshold=1) == 0
    assert edit_reward([1] * 20, [2] * 2 + [1] * 18, raw_distance_threshold=2) == pytest.approx(0.9)
    assert edit_reward([], [], raw_distance_threshold=0) == 1  # explicit empty-sequence convention


@pytest.mark.parametrize('mask', [torch.tensor([0.]), torch.tensor([float('inf')])])
def test_kl_rejects_undefined_mask_mass(mask):
    with pytest.raises(ValueError):
        token_forward_kl(torch.ones(1, 2), torch.ones(1, 2), mask)


def test_gradient_selection_and_independent_trials(device):
    base = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        base.weight.fill_(1.)
    score = gradient_norm_score(base(torch.ones(1, 1, device=device)).square().sum(), base.parameters())
    assert score == pytest.approx(2.)
    assert base.weight.grad is None
    indices = select_dreams([0.1, 0.8, 0.3, 0.5], top_k=1, random_count=2, seed=17)
    assert indices[0] == 1 and len(set(indices)) == 3
    assert indices == select_dreams([0.1, 0.8, 0.3, 0.5], top_k=1, random_count=2, seed=17)
    starts = []
    def adapt(model, delta):
        starts.append(model.weight.detach().item())
        with torch.no_grad():
            model.weight.add_(delta)
    results = isolated_trials(base, [1., -1.], adapt, lambda m: m.weight.detach().item())
    assert starts == [1., 1.]
    assert [r['reward'] for r in results] == [1, 0]
    assert base.weight.item() == 1.


@pytest.mark.parametrize('scores,k,b', [([float('nan')], 1, 0), ([1.], 1, 1), ([1.], -1, 0)])
def test_dream_selection_rejects_invalid_inputs(scores, k, b):
    with pytest.raises(ValueError):
        select_dreams(scores, top_k=k, random_count=b, seed=1)
