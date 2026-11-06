"""Count observed matrix multiplication work using PyTorch's dispatch mode.

The counter reports only the FLOPs of aten.mm/addmm/bmm/mv/dot operations.
It uses the conventional MAC=2 accounting and deliberately excludes all other
operations; its result is a lower bound on model work.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode


_SUPPORTED = {"mm", "addmm", "bmm", "mv", "dot"}


def _tensor_arg(args: tuple[Any, ...], kwargs: dict[str, Any], position: int, name: str) -> torch.Tensor:
    value = kwargs.get(name)
    if value is None and len(args) > position:
        value = args[position]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor argument {name!r}")
    return value


def _flops(op: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    if op == "mm":
        a, b = _tensor_arg(args, kwargs, 0, "self"), _tensor_arg(args, kwargs, 1, "mat2")
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("mm expects compatible rank-2 matrices")
        return 2 * int(a.shape[0]) * int(a.shape[1]) * int(b.shape[1])
    if op == "addmm":
        a, b = _tensor_arg(args, kwargs, 1, "mat1"), _tensor_arg(args, kwargs, 2, "mat2")
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("addmm expects compatible rank-2 matrices")
        return 2 * int(a.shape[0]) * int(a.shape[1]) * int(b.shape[1])
    if op == "bmm":
        a, b = _tensor_arg(args, kwargs, 0, "self"), _tensor_arg(args, kwargs, 1, "mat2")
        if (a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]
                or a.shape[2] != b.shape[1]):
            raise ValueError("bmm expects compatible rank-3 batches")
        return 2 * int(a.shape[0]) * int(a.shape[1]) * int(a.shape[2]) * int(b.shape[2])
    if op == "mv":
        a, b = _tensor_arg(args, kwargs, 0, "self"), _tensor_arg(args, kwargs, 1, "vec")
        if a.ndim != 2 or b.ndim != 1 or a.shape[1] != b.shape[0]:
            raise ValueError("mv expects a rank-2 matrix and compatible rank-1 vector")
        return 2 * int(a.shape[0]) * int(a.shape[1])
    a, b = _tensor_arg(args, kwargs, 0, "self"), _tensor_arg(args, kwargs, 1, "tensor")
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("dot expects equal-length rank-1 vectors")
    return 2 * int(a.shape[0])


class MatmulWork(TorchDispatchMode):
    """Accumulate actual aten matrix multiply calls by stage and phase.

    Set ``stage`` (for example, ``"train"`` or ``"eval"``) and ``phase``
    (``"forward"`` or ``"backward"``) before entering the mode. The public
    ``totals`` mapping has ``totals[stage][phase]`` records containing total
    ``flops``, total ``calls``, and per-operation ``ops`` records.
    """

    def __init__(self, stage: str = "default", phase: str = "forward") -> None:
        super().__init__()
        self.stage = str(stage)
        self.phase = str(phase)
        self.totals: dict[str, dict[str, dict[str, Any]]] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        name = str(func).split(".")[-2]
        # Execute the intercepted operation exactly once. Failed operations do
        # not contribute to the report.
        result = func(*args, **kwargs)
        if name not in _SUPPORTED:
            return result
        count = _flops(name, args, kwargs)
        stage_record = self.totals.setdefault(self.stage, {})
        phase_record = stage_record.setdefault(self.phase, {"flops": 0, "calls": 0, "ops": {}})
        phase_record["flops"] += count
        phase_record["calls"] += 1
        op_record = phase_record["ops"].setdefault(name, {"flops": 0, "calls": 0})
        op_record["flops"] += count
        op_record["calls"] += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        """Return a detached, serialization-friendly snapshot of the totals."""
        return {
            stage: {
                phase: {
                    "flops": int(record["flops"]),
                    "calls": int(record["calls"]),
                    "ops": {op: {"flops": int(stats["flops"]), "calls": int(stats["calls"])}
                            for op, stats in record["ops"].items()},
                }
                for phase, record in phases.items()
            }
            for stage, phases in self.totals.items()
        }


__all__ = ["MatmulWork"]
