"""Stat retained CLINC artifacts only; no dataset, model or checkpoint contents read."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def inventory(directory):
    sizes = Counter()
    counts = Counter()
    for path in directory.rglob("*"):
        if path.is_file() and not path.is_symlink():
            kind = path.suffix.lower() or "<no_suffix>"
            sizes[kind] += path.stat().st_size
            counts[kind] += 1
    return {"file_count": sum(counts.values()), "logical_file_bytes": sum(sizes.values()),
            "files_by_suffix": dict(sorted(counts.items())),
            "logical_bytes_by_suffix": dict(sorted(sizes.items()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", type=Path, required=True)
    p.add_argument("--dataset-cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    attempts = {path.name: inventory(path) for path in sorted(a.outputs_root.iterdir())
                if path.is_dir() and not path.is_symlink() and path.name.startswith("clinc-v8-")}
    data = {path.name: inventory(path) for path in sorted(a.dataset_cache_root.iterdir())
            if path.is_dir() and not path.is_symlink() and path.name.startswith("clinc150-v8-")}
    result = {"schema": "clinc-v8-retained-artifact-logical-storage-v1",
              "as_of_utc": datetime.now(timezone.utc).isoformat(),
              "output_attempts": attempts, "dataset_cache_directories": data,
              "all_output_logical_bytes": sum(row["logical_file_bytes"] for row in attempts.values()),
              "dataset_cache_logical_bytes": sum(row["logical_file_bytes"] for row in data.values()),
              "file_contents_read": False, "model_loaded": False, "optimizer_steps": 0,
              "limitations": ["Retained file lengths at this timestamp, not peak storage or filesystem allocated blocks.",
                              "Hard-linked content is not deduplicated; compression and sparse-file allocation are not measured.",
                              "Deleted temporary writes and diagnostic in-memory state are not recoverable from this inventory.",
                              "Only explicitly named CLINC output/cache prefixes are included; unrelated work is excluded."]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"attempt_directories": len(attempts), "data_cache_directories": len(data),
                      "output_logical_bytes": result["all_output_logical_bytes"],
                      "dataset_logical_bytes": result["dataset_cache_logical_bytes"]}))


if __name__ == "__main__":
    main()
