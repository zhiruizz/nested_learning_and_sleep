import pytest
import torch
from nested_sleep.parameter_drift import parameter_drift


def test_equal_norms_can_hide_a_change_and_baseline_is_untouched():
    baseline = torch.tensor([3., 4.], requires_grad=True)
    after = torch.tensor([4., 3.], requires_grad=True)
    row = parameter_drift({'w': baseline}, {'w': after})[0]
    assert row['baseline_l2'] == row['current_l2'] == 5
    assert row['delta_l2'] == pytest.approx(2 ** .5)
    assert row['relative_delta_l2'] == pytest.approx(2 ** .5 / 5)
    assert row['changed'] and baseline.grad is None and after.grad is None
    assert torch.equal(baseline.detach(), torch.tensor([3., 4.]))


def test_zero_baseline_is_explicitly_undefined():
    rows = parameter_drift({'a': torch.zeros(2), 'b': torch.zeros(1)},
                           {'a': torch.tensor([0., 2.]), 'b': torch.zeros(1)})
    assert rows[0]['delta_l2'] == 2 and rows[0]['relative_delta_l2'] is None
    assert rows[1]['delta_l2'] == 0 and rows[1]['relative_delta_l2'] is None
    assert not rows[1]['changed']


@pytest.mark.parametrize('before,after', [
    ({}, {}),
    ({'a': torch.ones(1)}, {'b': torch.ones(1)}),
    ({'a': torch.ones(1)}, {'a': torch.ones(2)}),
    ({'a': torch.ones(1)}, {'a': torch.ones(1, dtype=torch.float64)}),
    ({'a': torch.tensor([float('nan')])}, {'a': torch.ones(1)}),
    ({'a': torch.tensor([1])}, {'a': torch.tensor([1])}),
    ({'a': torch.tensor([])}, {'a': torch.tensor([])}),
])
def test_incomparable_inputs_rejected(before, after):
    with pytest.raises(ValueError):
        parameter_drift(before, after)
