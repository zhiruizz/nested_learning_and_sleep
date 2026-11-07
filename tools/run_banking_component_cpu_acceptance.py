"""Run one explicit Banking CPU component with the existing offline guard once.

Use a fresh ledger, receipt and temporary directory for each component. Public
Banking checks use ``check_`` names to remain outside the ordinary CPU suite;
the historical execution workspace used ``test_`` names. Both layouts select
the same file bytes. Neither layout changes the 256-update process limit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("v10", "v11"), required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--basetemp", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.ledger, args.receipt, args.basetemp):
        if path.exists() or not path.parent.is_dir():
            raise ValueError("Acceptance paths must be fresh children of existing directories")
    task_root = Path(__file__).resolve().parents[1]
    public_root = args.public_root.resolve(strict=True)
    paths = sorted((task_root / "tests").glob(f"*banking*{args.component}*.py"))
    if not paths:
        raise ValueError("No explicit Banking component checks found")
    guard_path = public_root / "tests" / "conftest.py"
    guard_sha = hashlib.sha256(guard_path.read_bytes()).hexdigest()
    os.environ["RELEASE_BUDGET_FILE"] = str(args.ledger.resolve())
    sys.path.insert(0, str(task_root))
    sys.path.insert(0, str(public_root / "src"))
    spec = importlib.util.spec_from_file_location("_banking_component_guards", guard_path)
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)
    import pytest
    # Explicit plugin loading plus --noconftest avoids loading the same guard
    # twice when task_root and public_root are the same checkout.
    code = pytest.main(["-q", "--noconftest", *map(str, paths),
                        "--basetemp", str(args.basetemp)], plugins=[guard])
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    if ledger.get("limit") != 256 or hashlib.sha256(guard_path.read_bytes()).hexdigest() != guard_sha:
        raise RuntimeError("The existing CPU guard or update limit changed")
    receipt = {
        "schema": "banking-component-cpu-acceptance-v1",
        "component": args.component,
        "status": "pass" if code == 0 else "failed",
        "pytest_exit_code": int(code),
        "test_sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "public_guard_sha256": guard_sha,
        "unchanged_attempt_limit": 256,
        "ledger": ledger,
        "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        "guard_loaded_once": True,
        "network_prohibited": True,
        "cuda_initialization_prohibited": True,
        "actual_model_loaded": False,
    }
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
