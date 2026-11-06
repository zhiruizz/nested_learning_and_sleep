#!/usr/bin/env python3
"""Verify the staged Git-index artifacts for the AG News routing v6 release.

This intentionally delegates numeric/schema checks to the staged, reviewed
stdlib analyzer. It reads Git blobs from the index and never loads models,
datasets, row-level examples, or private checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Any


RELEASE = "results/agnews-routing-fairness-v6"
ANALYZER = "tools/summarize_agnews_routing_fairness_v6.py"
EXPECTED_HASHES = {
    "tools/run_agnews_routing_fairness_v6.py": "7cbfe600399f992533fcd5c81586a3c1a4ecef7744c24fa7706108e94fcb4752",
    "tools/late_context_router_v6.py": "fa549672ed143d161930699de8a22e6cbe99404ae9f7b00ac30f75b1ed53d76d",
    "tools/run_agnews_fair_continuation_v6.py": "1bdea90f357e595fbaa1461d71106b75ee11c0df56fbcfc5e869a89f8c5fa1be",
    "tools/agnews_fair_continuation_v6.py": "442e4175b58b5103f2cde9d7047ff5ef807026de9059b0cadd6e5f4cd55488e2",
    "tools/qualify_agnews_fair_resume_v6.py": "a03e56f4cba104b9905c6da47641d38e28bb81c7224e4b3251f2f21bbf14effe",
    "tools/audit_agnews_context_storage_v6.py": "f1b76e89b5ac39ddd7d9aa3cd333e62a5130aec8fa8829fb263422bb5c47e19b",
    "tools/probe_agnews_task_router_v5.py": "6659b2a66a1f470661d820a8bace8ed5fe2adae851d8f9006fbd5064be5c9a96",
    "tools/agnews_anchored_wake_v5.py": "3a90e122be597314514036a0e1fbbb91befae5d7df0425919c786f86c11b34a6",
    "tools/agnews_v3_common.py": "6c2e4b49122a6a2682ef70acffe327c97e2167cc90627708cdb90341d779f3a0",
    "tools/run_agnews_retention_v2.py": "498c317d876d7e37fdead90ad81057a2f164d24fe7867e3d00bb6abca449216b",
    "src/nested_sleep/lifecycle.py": "1fd3164e705c79d30d9c5666c055ef64b027672e547b8374708263066275df02",
}
CONFIG_HASHES = {
    "agnews-routing-fairness-v6-r3.json": "126f0d8b1556234fcf763d18f77893e81f5ae6e11175f999aa5dea3f713cd6da",
    "agnews-fair-continuation-v6.json": "578c472b54d405cff8c03dd89d6eb831c81cd4a49b06d3d450d96518cf9f3a0e",
    "agnews-fair-resume-v6-r2.json": "8064b3befba382f4f53c57c512c055f2245fd756ea03981c61332e23d8065a82",
}
RELEASE_FILES = {
    f"{RELEASE}/initial/protocol.json", f"{RELEASE}/initial/result.json",
    f"{RELEASE}/continuation/protocol.json", f"{RELEASE}/continuation/result.json",
    f"{RELEASE}/training/protocol.json", f"{RELEASE}/training/development-sealed.json",
    f"{RELEASE}/qualification/plan.json", f"{RELEASE}/qualification/qualification.json",
    f"{RELEASE}/storage/audit.json",
    f"{RELEASE}/failures/resume-config-preflight-1/config.json",
    f"{RELEASE}/failures/resume-config-preflight-1/source-executed.py",
    f"{RELEASE}/failures/resume-config-preflight-1/failure.json",
    f"{RELEASE}/failures/initial-metric-replay-r2/account.json",
    f"{RELEASE}/failures/initial-metric-replay-r2/failure.json",
    f"{RELEASE}/failures/initial-metric-replay-r2/protocol.json",
    f"{RELEASE}/failures/initial-metric-replay-r2/execution-lock.json",
    f"{RELEASE}/failures/initial-metric-replay-r2/source-executed.py",
    f"{RELEASE}/engineering-corrections/A-plan-path-r1/correction.json",
    f"{RELEASE}/engineering-corrections/A-plan-path-r1/execution-lock.json",
    f"{RELEASE}/engineering-corrections/A-plan-path-r1/source-executed.py",
    f"{RELEASE}/analysis/analysis.json", f"{RELEASE}/analysis/work-ledger.json",
    f"{RELEASE}/analysis/initial-summary.csv", f"{RELEASE}/analysis/endpoint-summary.csv",
    f"{RELEASE}/analysis/endpoint-gates.csv", f"{RELEASE}/analysis/work-ledger.csv",
    f"{RELEASE}/analysis/initial-routing-validation-em.png",
    f"{RELEASE}/analysis/initial-routing-validation-em.svg",
    f"{RELEASE}/analysis/endpoint-routing-validation-em.png",
    f"{RELEASE}/analysis/endpoint-routing-validation-em.svg",
    f"{RELEASE}/validation.json",
}
OPTIONAL_RELEASE_FILES = {f"{RELEASE}/validation-extra.json"}
INPUT_REL = {
    "initial_protocol": f"{RELEASE}/initial/protocol.json",
    "initial_result": f"{RELEASE}/initial/result.json",
    "b_protocol": f"{RELEASE}/training/protocol.json",
    "b_seal": f"{RELEASE}/training/development-sealed.json",
    "end_protocol": f"{RELEASE}/continuation/protocol.json",
    "end_result": f"{RELEASE}/continuation/result.json",
    "storage": f"{RELEASE}/storage/audit.json",
    "qualification": f"{RELEASE}/qualification/qualification.json",
    "failure": f"{RELEASE}/failures/resume-config-preflight-1/failure.json",
    "initial_failure": f"{RELEASE}/failures/initial-metric-replay-r2/account.json",
    "initial_failure_detail": f"{RELEASE}/failures/initial-metric-replay-r2/failure.json",
}
OPTIONAL_FAILURE_METADATA: set[str] = set()
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".bin", ".npz", ".npy", ".arrow",
    ".parquet", ".jsonl", ".pkl", ".pickle",
}
PROTECTED_PREFIXES = (
    "src/nested_sleep/", "results/agnews-anchored-v5/",
    "results/agnews-mechanism-v5/", "results/agnews-preupdate-v4/",
)


def fail(message: str) -> None:
    raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(root: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", *args], cwd=root, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        fail(f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def index_blobs(root: Path) -> dict[str, tuple[str, int]]:
    raw = git(root, "ls-files", "--stage", "-z")
    out: dict[str, tuple[str, int]] = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        meta, name = item.split(b"\t", 1)
        mode, oid, stage = meta.decode("ascii").split()
        path = name.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        if stage != "0":
            fail(f"unmerged index entry: {path}")
        out[path] = (oid, int(mode, 8))
    return out


def blob(root: Path, entry: tuple[str, int]) -> bytes:
    return git(root, "cat-file", "blob", entry[0])


def parse_json(data: bytes, label: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid UTF-8 JSON in {label}: {exc}")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _check_tracked_source_graph(root: Path, entries: dict[str, tuple[str, int]]) -> dict[str, str]:
    changed = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    for raw in changed.split(b"\0"):
        if not raw:
            continue
        path = raw.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        low = path.lower()
        if "private" in PurePosixPath(low).parts or PurePosixPath(low).suffix in FORBIDDEN_SUFFIXES:
            fail(f"staged change contains a private/binary-like path: {path}")
    for rel, expected in EXPECTED_HASHES.items():
        if rel not in entries:
            fail(f"required staged source/config missing: {rel}")
        actual = sha(blob(root, entries[rel]))
        if actual != expected:
            fail(f"staged source/config hash mismatch for {rel}: {actual}")
    if ANALYZER not in entries:
        fail(f"required staged analyzer missing: {ANALYZER}")
    config_receipt: dict[str, str] = {}
    for name, expected in CONFIG_HASHES.items():
        candidates = [p for p in (name, f"configs/{name}") if p in entries]
        if len(candidates) != 1:
            fail(f"expected exactly one staged location for config {name}; got {candidates}")
        actual = sha(blob(root, entries[candidates[0]]))
        if actual != expected:
            fail(f"staged configuration hash mismatch for {candidates[0]}: {actual}")
        config_receipt[candidates[0]] = actual
    for rel in RELEASE_FILES:
        if rel not in entries:
            fail(f"required release artifact missing from index: {rel}")
    for rel in entries:
        if rel.startswith(RELEASE + "/") and rel not in RELEASE_FILES | OPTIONAL_RELEASE_FILES | OPTIONAL_FAILURE_METADATA:
            fail(f"unexpected file under release directory: {rel}")
        if rel.startswith(RELEASE + "/") and PurePosixPath(rel).suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"private/binary-like payload forbidden in release: {rel}")
        low = rel.lower()
        if "private" in PurePosixPath(low).parts or any(low.endswith(s) for s in FORBIDDEN_SUFFIXES):
            if rel.startswith(RELEASE + "/") or rel in EXPECTED_HASHES:
                fail(f"private/binary artifact forbidden: {rel}")
    # Existing protected scientific/runtime trees must remain identical to HEAD.
    for rel, (oid, _mode) in entries.items():
        if not rel.startswith(PROTECTED_PREFIXES):
            continue
        try:
            old = git(root, "show", f"HEAD:{rel}")
        except ValueError:
            continue
        if sha(old) != sha(blob(root, (oid, _mode))):
            fail(f"protected prior artifact changed in index: {rel}")
    return ({rel: sha(blob(root, entry)) for rel, entry in entries.items() if rel in EXPECTED_HASHES}
            | {ANALYZER: sha(blob(root, entries[ANALYZER]))} | config_receipt)


def _verify_protocol_source_bindings(root: Path, entries: dict[str, tuple[str, int]]) -> dict[str, str]:
    """Bind protocol source-digest fields to the exact staged public source blobs."""
    ip = parse_json(blob(root, entries[INPUT_REL["initial_protocol"]]), "initial protocol")
    ep = parse_json(blob(root, entries[INPUT_REL["end_protocol"]]), "endpoint protocol")
    bp = parse_json(blob(root, entries[INPUT_REL["b_protocol"]]), "B protocol")
    a = ip.get("source_sha256", {})
    if ep.get("source_sha256") != a:
        fail("A initial and endpoint protocol source graphs differ")
    a_map = {
        "driver_sha256": "tools/run_agnews_routing_fairness_v6.py",
        "common_sha256": "tools/agnews_v3_common.py",
        "engine_sha256": "tools/run_agnews_retention_v2.py",
        "lifecycle_sha256": "src/nested_sleep/lifecycle.py",
        "router_helper_sha256": "tools/late_context_router_v6.py",
        "v5_router_probe_sha256": "tools/probe_agnews_task_router_v5.py",
    }
    b = bp.get("source_sha256", {})
    b_map = {
        "driver_source_sha256": "tools/run_agnews_fair_continuation_v6.py",
        "continuation_helper_sha256": "tools/agnews_fair_continuation_v6.py",
        "common_sha256": "tools/agnews_v3_common.py",
        "engine_sha256": "tools/run_agnews_retention_v2.py",
        "v5_anchor_helper_sha256": "tools/agnews_anchored_wake_v5.py",
    }
    checked = {}
    for key, path in (a_map | b_map).items():
        if path in (a_map.values()) and key not in a:
            fail(f"A protocol source graph lacks {key}")
        if path in (b_map.values()) and key not in b:
            fail(f"B protocol source graph lacks {key}")
    for key, path in a_map.items():
        actual = sha(blob(root, entries[path]))
        if a.get(key) != actual:
            fail(f"A protocol source hash differs from staged source {path}")
        checked[f"A/{key}"] = actual
    for key, path in b_map.items():
        actual = sha(blob(root, entries[path]))
        if b.get(key) != actual:
            fail(f"B protocol source hash differs from staged source {path}")
        checked[f"B/{key}"] = actual
    return checked


def _write_temp(root: Path, dest: Path, entries: dict[str, tuple[str, int]], rel: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob(root, entries[rel]))
    return dest


def _load_staged_analyzer(root: Path, entries: dict[str, tuple[str, int]], temp: Path):
    analyzer_path = _write_temp(root, temp / "staged" / ANALYZER, entries, ANALYZER)
    spec = importlib.util.spec_from_file_location("_staged_agnews_v6_analyzer", analyzer_path)
    if spec is None or spec.loader is None:
        fail("could not load staged stdlib analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _recompute(root: Path, entries: dict[str, tuple[str, int]], temp: Path) -> dict[str, str]:
    analyzer = _load_staged_analyzer(root, entries, temp)
    paths = {}
    for key, rel in INPUT_REL.items():
        paths[key] = _write_temp(root, temp / "inputs" / rel, entries, rel)
    out = temp / "recomputed-analysis"
    out.mkdir()
    result = analyzer.aggregate_analysis(
        paths["initial_protocol"], paths["initial_result"], paths["b_protocol"],
        paths["b_seal"], paths["end_protocol"], paths["end_result"],
        paths["storage"], paths["qualification"], paths["failure"],
        paths["initial_failure"])
    (out / "analysis.json").write_bytes(canonical_json(result))
    (out / "work-ledger.json").write_bytes(canonical_json(result["work_ledger"]))
    analyzer.write_tables(out, result)
    checks: dict[str, str] = {}
    for name in ("analysis.json", "work-ledger.json", "initial-summary.csv",
                 "endpoint-summary.csv", "endpoint-gates.csv", "work-ledger.csv"):
        rel = f"{RELEASE}/analysis/{name}"
        expected = (out / name).read_bytes()
        actual = blob(root, entries[rel])
        if actual != expected:
            fail(f"staged analysis/table differs from rerun analyzer: {rel}")
        checks[rel] = sha(actual)
    return checks


def _verify_svg(svg: bytes, label: str) -> None:
    text = svg.decode('utf-8', errors='strict')
    if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
        fail(f'external declaration in SVG: {label}')
    tree = ET.fromstring(text)
    if tree.tag != '{http://www.w3.org/2000/svg}svg':
        fail(f'invalid SVG root: {label}')
    for node in tree.iter():
        if node.tag.rsplit('}', 1)[-1].lower() in {'script', 'foreignobject'}:
            fail(f'active content in SVG: {label}')
        for key, value in node.attrib.items():
            key = key.rsplit('}', 1)[-1].lower()
            if key.startswith('on') or (key == 'href' and not value.startswith('#')):
                fail(f'external or active attribute in SVG: {label}')
        for value in [*node.attrib.values(), node.text or '']:
            for target in re.findall(r'url\(([^)]*)\)', value, re.I):
                if not target.strip().strip('\"\x27').startswith('#'):
                    fail(f'external style resource in SVG: {label}')


def _verify_figures(root: Path, entries: dict[str, tuple[str, int]]) -> dict[str, str]:
    checks = {}
    for base in ("initial-routing-validation-em", "endpoint-routing-validation-em"):
        png_rel = f"{RELEASE}/analysis/{base}.png"
        svg_rel = f"{RELEASE}/analysis/{base}.svg"
        png, svg = blob(root, entries[png_rel]), blob(root, entries[svg_rel])
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            fail(f"invalid PNG signature: {png_rel}")
        _verify_svg(svg, svg_rel)
        checks[png_rel] = sha(png); checks[svg_rel] = sha(svg)
    return checks


def _verify_historical_failure(root: Path, entries: dict[str, tuple[str, int]]) -> dict[str, str]:
    base = f"{RELEASE}/failures/initial-metric-replay-r2"
    account = parse_json(blob(root, entries[f"{base}/account.json"]), "initial R2 account")
    protocol = parse_json(blob(root, entries[f"{base}/protocol.json"]), "initial R2 protocol")
    lock = parse_json(blob(root, entries[f"{base}/execution-lock.json"]), "initial R2 execution lock")
    driver = blob(root, entries[f"{base}/source-executed.py"])
    driver_hash = sha(driver)
    if account.get("source_sha256") != driver_hash:
        fail("preserved R2 failure account does not bind its executed driver bytes")
    if protocol.get("phase") != "initial" or protocol.get("source_sha256", {}).get("driver_sha256") != driver_hash:
        fail("preserved R2 protocol does not bind its executed driver bytes")
    if lock.get("phase") != "initial" or lock.get("execution_lock_revision") != 2:
        fail("preserved R2 execution-lock metadata has unexpected phase/revision")
    if lock.get("source_sha256", {}).get("driver_sha256") != driver_hash:
        fail("preserved R2 execution lock does not bind its executed driver bytes")
    if account.get("schema") != "agnews-v6-metric-replay-failure-account-v1":
        fail("preserved R2 failure account has an unexpected schema")
    return {f"{base}/{name}": sha(blob(root, entries[f"{base}/{name}"]))
            for name in ("account.json", "failure.json", "protocol.json", "execution-lock.json", "source-executed.py")}


def run_verification(root: Path) -> dict[str, Any]:
    root = root.resolve()
    entries = index_blobs(root)
    source_hashes = _check_tracked_source_graph(root, entries)
    protocol_source_hashes = _verify_protocol_source_bindings(root, entries)
    historical_failure_hashes = _verify_historical_failure(root, entries)
    with tempfile.TemporaryDirectory(prefix="agnews-v6-index-") as td:
        temp = Path(td)
        analysis_hashes = _recompute(root, entries, temp)
    figure_hashes = _verify_figures(root, entries)
    head = git(root, "rev-parse", "HEAD").decode("ascii").strip()
    staged_release = sorted(p for p in entries if p.startswith(RELEASE + "/"))
    return {
        "schema": "agnews-routing-fairness-v6-index-validation-v1",
        "passed": True,
        "verification_scope": "Git index blobs only; no model, raw dataset, private checkpoint, or row-level example loaded",
        "head_commit": head,
        "required_release_file_count": len(RELEASE_FILES),
        "staged_release_file_count": len(staged_release),
        "source_config_sha256": source_hashes,
        "protocol_source_sha256": protocol_source_hashes,
        "preserved_initial_r2_failure_sha256": historical_failure_hashes,
        "recomputed_analysis_sha256": analysis_hashes,
        "figure_sha256": figure_hashes,
        "protected_prior_trees_unchanged": True,
        "analyzer_rerun": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="optional receipt path outside the staged release")
    args = parser.parse_args(argv)
    receipt = run_verification(args.root)
    encoded = canonical_json(receipt)
    if args.output:
        args.output.write_bytes(encoded)
    sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
