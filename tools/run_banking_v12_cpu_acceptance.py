"""Run explicit V12 CPU checks under the project's unchanged offline guard."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--basetemp", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.ledger, args.receipt, args.log, args.basetemp):
        if path.exists() or not path.parent.is_dir():
            raise ValueError("fresh acceptance paths with existing parent required")
    root = Path(__file__).resolve().parents[1]
    tests = sorted((root / "tests").glob("*banking*v12*.py"))
    if not tests:
        raise ValueError("no V12 checks found")
    guard_path = args.public_root.resolve(strict=True) / "tests/conftest.py"
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    guard_sha = sha(guard_path)
    if guard_sha != "c22f07489d40ed5c3b49c6e9c6790ff15f97cae3bbdd9d65e388b853ef2f4fd9":
        raise ValueError("existing project CPU guard changed")
    tests_before = {p.name: sha(p) for p in tests}
    os.environ["RELEASE_BUDGET_FILE"] = str(args.ledger.resolve())
    sys.path[:0] = [str(root), str(args.public_root.resolve() / "src")]
    output = io.StringIO()
    started = time.monotonic()
    error = None
    code = 2
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            spec = importlib.util.spec_from_file_location("_banking_v12_offline_guard", guard_path)
            guard = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = guard
            spec.loader.exec_module(guard)
            import pytest
            code = pytest.main(["-q", "--noconftest", *map(str, tests),
                               "--basetemp", str(args.basetemp)], plugins=[guard])
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    log_bytes = output.getvalue().encode("utf-8")
    with args.log.open("xb") as stream:
        stream.write(log_bytes)
    ledger = json.loads(args.ledger.read_text(encoding="utf-8")) if args.ledger.exists() else None
    tests_after = {p.name: sha(p) for p in tests}
    unchanged = tests_before == tests_after and sha(guard_path) == guard_sha
    if not unchanged or (ledger is not None and ledger.get("limit") != 256):
        error = {"type": "IntegrityError", "message": "test sources or unchanged guard budget changed"}
        code = 2
    if ledger is None:
        error = error or {"type": "MissingLedger", "message": "guard produced no acceptance ledger"}
        code = 2
    receipt = {
        "schema": "banking-v12-cpu-acceptance-v1", "status": "pass" if code == 0 else "failed",
        "pytest_exit_code": int(code), "test_source_hashes": tests_before,
        "test_sources_unchanged": tests_before == tests_after,
        "guard_sha256": guard_sha, "guard_unchanged": sha(guard_path) == guard_sha, "ledger": ledger,
        "ledger_sha256": sha(args.ledger) if ledger is not None else None, "log_sha256": sha(args.log),
        "elapsed_wall_seconds": time.monotonic() - started,
        "network_prohibited": True, "cuda_initialization_prohibited": True,
        "real_model_loaded": False, "setup_or_integrity_error": error,
    }
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(output.getvalue(), end="")
    print(json.dumps(receipt))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
