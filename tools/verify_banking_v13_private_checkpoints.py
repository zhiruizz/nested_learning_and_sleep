"""Read-only post-run byte audit; never deserialize model/checkpoint tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

CONFIG_SHA = "1c4aae0c13d6b80193fee548639e7adbf6fae4d5b2c13d3d21d5f72ba26f0f35"
STAGES = ("event1_before", "event1_after", "event2_before", "event2_after", "postwake15")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    runroot = a.run_root.resolve(strict=True)
    if a.receipt.exists() or not a.receipt.parent.is_dir():
        raise ValueError("fresh receipt with an existing parent required")
    if a.receipt.resolve().is_relative_to(runroot):
        raise ValueError("audit must not write into the scientific run tree")
    if sha(a.config) != CONFIG_SHA:
        raise ValueError("frozen configuration changed")
    cfg, records, checkpoints = load(a.config), [], []
    started = time.monotonic()
    qualification = None
    for seed in cfg["seeds"]:
        for arm in cfg["execution_order_per_seed"]:
            directory = runroot / f"seed-{seed}" / arm
            terminal = directory / "final-receipt.json"
            r, protocol = load(terminal), load(directory / "protocol.json")
            digest = sha(terminal)
            if (r["status"], r["seed"], r["arm"], r["config_sha256"]) != ("complete", seed, arm, CONFIG_SHA):
                raise ValueError("terminal identity mismatch")
            if r["source_hashes"] != cfg["source_hashes"] or r["test_data_read"] or r["official_test_read"] or r["P2_accessed"]:
                raise ValueError("terminal source or access-boundary mismatch")
            if protocol["status"] != "complete" or protocol["final_receipt_sha256"] != digest:
                raise ValueError("protocol/terminal digest mismatch")
            cp = r["checkpoint_manifest"]
            if len(cp) != 5 or tuple(x["stage"] for x in cp) != STAGES:
                raise ValueError("five ordered stage checkpoints required")
            for item in cp:
                rel = PurePosixPath(item["file"])
                if rel.is_absolute() or ".." in rel.parts or ":" in item["file"] or "\\" in item["file"]:
                    raise ValueError("unsafe checkpoint path")
                path = directory / Path(*rel.parts)
                if path.is_symlink() or not path.resolve(strict=True).is_relative_to(directory.resolve(strict=True)):
                    raise ValueError("checkpoint escapes its run")
                if not item["destination_deserialized_readback"]:
                    raise ValueError("execution-time destination readback not recorded")
                if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                    raise ValueError("post-run checkpoint byte mismatch")
                if load(path.with_suffix(".json")) != item:
                    raise ValueError("checkpoint sidecar differs from terminal record")
                checkpoints.append({"seed": seed, "arm": arm, **item, "postrun_byte_hash_match": True})
            q = r["restore_qualification"]
            if (seed, arm) == (17, "B32_HIGH"):
                if q["status"] != "pass" or load(directory / "qualification/restore-qualification.json") != q:
                    raise ValueError("qualification receipt mismatch")
                path = directory / "qualification/after32-private.pt"
                if path.is_symlink() or sha(path) != q["snapshot_file_sha256"]:
                    raise ValueError("qualification snapshot byte mismatch")
                qualification = {"snapshot_file_sha256": q["snapshot_file_sha256"],
                    "bytes": path.stat().st_size, "postrun_byte_hash_match": True}
            elif q is not None or (directory / "qualification").exists():
                raise ValueError("unexpected extra qualification")
            records.append({"seed": seed, "arm": arm, "terminal_sha256": digest,
                            "protocol_sha256": sha(directory / "protocol.json"), "checkpoints": 5})
    if len(records) != 18 or len(checkpoints) != 90 or qualification is None:
        raise ValueError("incomplete frozen study")
    receipt = {"schema": "banking-v13-postrun-checkpoint-byte-audit-v1", "status": "pass",
        "config_sha256": CONFIG_SHA, "audit_source_sha256": sha(Path(__file__)),
        "runs": records, "checkpoints": checkpoints, "qualification_snapshot": qualification,
        "checkpoint_bytes_verified": sum(x["bytes"] for x in checkpoints) + qualification["bytes"],
        "tensor_deserialization_performed_by_this_audit": False,
        "scope": "postrun_bytes_and_sidecars; execution-time_readback_is_a_recorded_runner_assertion; not_new_restore_testing",
        "audit_wall_seconds": time.monotonic() - started, "scientific_outputs_modified": False,
        "test_data_read": False, "raw_rows_or_ids_emitted": False}
    with a.receipt.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(receipt, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": "pass", "runs": len(records), "checkpoints": len(checkpoints),
                      "bytes_verified": receipt["checkpoint_bytes_verified"], "receipt_sha256": sha(a.receipt)}))


if __name__ == "__main__":
    main()
