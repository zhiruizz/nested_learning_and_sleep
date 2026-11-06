import torch

from tools.matmul_work_v7 import MatmulWork


def test_linear_forward_backward_counts_actual_mm_work_and_preserves_results():
    torch.manual_seed(31)
    x_ref = torch.randn(3, 4, requires_grad=True)
    w_ref = torch.randn(4, 5, requires_grad=True)
    x_count = x_ref.detach().clone().requires_grad_()
    w_count = w_ref.detach().clone().requires_grad_()

    expected = x_ref @ w_ref
    expected.sum().backward()
    mode = MatmulWork(stage="fit")
    with mode:
        actual = x_count @ w_count
        assert torch.equal(actual, expected)
        mode.phase = "backward"
        actual.sum().backward()

    assert torch.equal(x_count.grad, x_ref.grad)
    assert torch.equal(w_count.grad, w_ref.grad)
    snapshot = mode.state_dict()
    assert snapshot["fit"]["forward"]["flops"] == 2 * 3 * 4 * 5
    assert snapshot["fit"]["forward"]["calls"] == 1
    # Autograd computes dX and dW as two observed matrix multiplications.
    assert snapshot["fit"]["backward"]["flops"] == 2 * (2 * 3 * 4 * 5)
    assert snapshot["fit"]["backward"]["calls"] == 2


def test_bmm_and_stage_phase_buckets():
    mode = MatmulWork(stage="warmup")
    a = torch.ones(2, 3, 4)
    b = torch.ones(2, 4, 6)
    with mode:
        torch.bmm(a, b)
    mode.stage = "measure"
    mode.phase = "forward"
    with mode:
        torch.bmm(a, b)
    assert mode.totals["warmup"]["forward"]["flops"] == 2 * 2 * 3 * 4 * 6
    assert mode.totals["measure"]["forward"]["calls"] == 1
    snapshot = mode.state_dict()
    snapshot["warmup"]["forward"]["ops"]["bmm"]["flops"] = -1
    assert mode.totals["warmup"]["forward"]["ops"]["bmm"]["flops"] > 0


def test_supported_vector_operations_and_non_matmul_are_not_counted():
    mode = MatmulWork()
    with mode:
        torch.mv(torch.ones(3, 4), torch.ones(4))
        torch.dot(torch.ones(7), torch.ones(7))
        torch.addmm(torch.ones(3, 5), torch.ones(3, 4), torch.ones(4, 5))
        torch.ones(3).sum()
    rec = mode.totals["default"]["forward"]
    assert rec["flops"] == 2 * 3 * 4 + 2 * 7 + 2 * 3 * 4 * 5
    assert rec["calls"] == 3
    assert set(rec["ops"]) == {"mv", "dot", "addmm"}
