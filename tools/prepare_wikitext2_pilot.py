"""Build fixed, article-disjoint WikiText-2 raw train windows for the LM pilot.

Requires PyArrow in the preprocessing interpreter. It writes local text caches;
the caller must keep those files outside Git (the public repo ignores outputs/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

SOURCE_SHA256 = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
SALT = "nl-sleep-wikitext2-article-split-v1"
TITLE = re.compile(r"^\s*=\s+([^=\n]+?)\s+=\s*$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def article_id(title: str) -> tuple[str, int]:
    normalized = " ".join(title.casefold().split())
    value = hashlib.sha256((SALT + "\0" + normalized).encode("utf-8")).hexdigest()
    return value, int(value[:8], 16) % 100


def self_test() -> None:
    assert TITLE.match(" = Valkyria Chronicles III = ")
    assert TITLE.match("= Article title =")
    assert not TITLE.match(" = = Gameplay = = ")
    assert not TITLE.match(" = = = Music = = = ")
    assert not TITLE.match(" = = = = Nested = = = = ")
    ids = [article_id(x)[0] for x in ("Example Article", " Second Article ")]
    assert len(ids) == len(set(ids))
    buckets = [article_id(x)[1] for x in ("Example Article", " Second Article ")]
    assert all(0 <= b < 100 for b in buckets)
    print("title-boundary self-test passed: only single-equals top-level headings are articles")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    actual = digest(args.parquet)
    if actual != SOURCE_SHA256:
        raise SystemExit(f"source parquet SHA256 mismatch: {actual}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install pyarrow in a separate preprocessing environment; do not modify nl-sleep.") from exc

    texts = pq.read_table(args.parquet, columns=["text"]).column("text").to_pylist()
    articles: OrderedDict[str, list[str]] = OrderedDict()
    title = None
    unassigned_nonempty = 0
    for line in texts:
        match = TITLE.match(line)
        if match:
            title = match.group(1).strip()
            articles.setdefault(title, [])
        elif title and line.strip():
            articles[title].append(line.strip())
        elif not title and line.strip():
            unassigned_nonempty += 1
    if unassigned_nonempty:
        raise RuntimeError(f"unassigned nonempty source rows before first article: {unassigned_nonempty}")

    roles = {"adaptation": [], "calibration": [], "evaluation": []}
    bounds = (("adaptation", 0, 60, 8), ("calibration", 60, 80, 8), ("evaluation", 80, 100, 16))
    eligible = []
    seen_article_ids = set()
    for article_title, paragraphs in articles.items():
        text = "\n".join(paragraphs)
        if len(text) < 700:
            continue
        ident, bucket = article_id(article_title)
        if ident in seen_article_ids:
            raise RuntimeError("duplicate normalized article title/id; merge or audit before splitting")
        seen_article_ids.add(ident)
        eligible.append((ident, bucket, article_title, text))
    split_article_counts = {
        "adaptation": sum(0 <= bucket < 60 for _, bucket, _, _ in eligible),
        "calibration": sum(60 <= bucket < 80 for _, bucket, _, _ in eligible),
        "evaluation": sum(80 <= bucket < 100 for _, bucket, _, _ in eligible),
    }
    if sum(split_article_counts.values()) != len(eligible):
        raise RuntimeError("article partition is not exhaustive and mutually exclusive")
    for role, low, high, count in bounds:
        candidates = [(ident, text) for ident, bucket, _title, text in eligible if low <= bucket < high]
        candidates.sort(key=lambda item: item[0])
        if len(candidates) < count:
            raise RuntimeError(f"not enough eligible articles in {role}: {len(candidates)} < {count}")
        roles[role] = [{"article_id_sha256": ident, "text": text[:12000]}
                       for ident, text in candidates[:count]]
    all_selected = [row["article_id_sha256"] for rows in roles.values() for row in rows]
    if len(all_selected) != len(set(all_selected)):
        raise RuntimeError("selected article IDs overlap between roles")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for role, rows in roles.items():
        (args.out_dir / f"{role}.json").write_text(json.dumps({"role": role, "examples": rows},
            ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    manifest = {
        "schema": "nl-sleep-wikitext2-role-cache-v1",
        "source": "Salesforce/wikitext / wikitext-2-raw-v1 / train",
        "source_revision_ref": "refs/convert/parquet",
        "source_revision_commit": "f3fd811233cf8abb79006d3a5f566c280bf47e1a",
        "source_file": args.parquet.name,
        "source_file_sha256": actual,
        "source_rows": len(texts),
        "top_level_article_count": len(articles),
        "eligible_article_count": len(eligible),
        "eligible_article_counts_by_role": split_article_counts,
        "unassigned_nonempty_rows_before_first_article": unassigned_nonempty,
        "nested_section_headings_are_content": True,
        "salt": SALT,
        "rule": "SHA256(salt + NUL + casefolded normalized title), first 8 hex digits modulo 100; [0,60) adaptation, [60,80) calibration, [80,100) evaluation.",
        "selected_article_ids": {role: [row["article_id_sha256"] for row in rows]
                                  for role, rows in roles.items()},
        "cache_sha256": {role: digest(args.out_dir / f"{role}.json") for role in roles},
        "text_is_local_only": True,
        "license_metadata": ["cc-by-sa-3.0", "gfdl"],
        "license_card_text": "CC BY-SA 4.0"
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("selected_article_ids", "cache_sha256")}, indent=2))


if __name__ == "__main__":
    main()
