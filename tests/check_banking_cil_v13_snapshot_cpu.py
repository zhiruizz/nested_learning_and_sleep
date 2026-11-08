from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from banking_cil_v13_batching import BatchableAdditiveSleepTransactionV13
try:
    from test_banking_cil_v10_lifecycle_cpu import TinyContext, make_tx
except ModuleNotFoundError as exc:
    if exc.name != 'test_banking_cil_v10_lifecycle_cpu':
        raise
    from check_banking_cil_v10_lifecycle_cpu import TinyContext, make_tx


def test_real_additive_from_snapshot_initializes_v13_batch_guard():
    live = TinyContext()
    original = make_tx(live)
    snapshot = original.snapshot()
    restored = BatchableAdditiveSleepTransactionV13.from_snapshot(live, snapshot)
    assert restored._v13_batch_active is False
    # A V13 snapshot is allowed immediately after restore, before first batch.
    second = restored.snapshot()
    assert second.schema == snapshot.schema
