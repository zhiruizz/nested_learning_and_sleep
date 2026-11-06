"""Pure metadata checks for the V6 staged-index release verifier."""
from __future__ import annotations

import importlib.util
import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_agnews_routing_fairness_v6_release",
    ROOT / "tools" / "verify_agnews_routing_fairness_v6_release.py")
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_svg_namespaces_are_metadata_but_active_resources_are_rejected():
    prefix = b'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
    verifier._verify_svg(prefix + b'<path id="p"/><use xlink:href="#p"/></svg>', 'valid')
    for content in (b'<script/>', b'<foreignObject/>', b'<use xlink:href="https://example.invalid/p"/>',
                    b'<path onclick="run()"/>', b'<path style="fill:url(https://example.invalid/p)"/>'):
        with pytest.raises(ValueError):
            verifier._verify_svg(prefix + content + b'</svg>', 'invalid')


def _with_blob_fixture(monkeypatch, fixtures: dict[str, bytes]):
    entries = {path: (path, 0o100644) for path in fixtures}
    monkeypatch.setattr(verifier, "blob", lambda _root, entry: fixtures[entry[0]])
    return entries


def test_actual_a_b_protocol_source_graph_binds_staged_sources(monkeypatch):
    fixtures = {
        verifier.INPUT_REL["initial_protocol"]:
            (ROOT / verifier.INPUT_REL["initial_protocol"]).read_bytes(),
        verifier.INPUT_REL["end_protocol"]:
            (ROOT / verifier.INPUT_REL["end_protocol"]).read_bytes(),
        verifier.INPUT_REL["b_protocol"]:
            (ROOT / verifier.INPUT_REL["b_protocol"]).read_bytes(),
    }
    for rel in verifier.EXPECTED_HASHES:
        fixtures[rel] = (ROOT / rel).read_bytes()
    entries = _with_blob_fixture(monkeypatch, fixtures)
    checked = verifier._verify_protocol_source_bindings(ROOT, entries)
    assert len(checked) == 11
    assert checked["A/driver_sha256"] == "7cbfe600399f992533fcd5c81586a3c1a4ecef7744c24fa7706108e94fcb4752"
    assert checked["B/driver_source_sha256"] == "1bdea90f357e595fbaa1461d71106b75ee11c0df56fbcfc5e869a89f8c5fa1be"


def test_archived_r2_failure_bundle_has_exact_source_provenance(monkeypatch):
    folder = ROOT / verifier.RELEASE / "failures" / "initial-metric-replay-r2"
    prefix = f"{verifier.RELEASE}/failures/initial-metric-replay-r2/"
    names = ("account.json", "failure.json", "protocol.json", "execution-lock.json", "source-executed.py")
    fixtures = {prefix + name: (folder / name).read_bytes() for name in names}
    entries = _with_blob_fixture(monkeypatch, fixtures)
    checked = verifier._verify_historical_failure(ROOT, entries)
    assert len(checked) == 5
    assert checked[prefix + "source-executed.py"] == "3fdcbc79c744aef8e5ac9f4fd0de83f763e8fe5da7811a756edc5f53e5ff96dd"
