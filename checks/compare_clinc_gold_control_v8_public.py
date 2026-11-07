#!/usr/bin/env python3
"""Public, path-parameterized replay of the development Gold-control equality check.

This is a portable companion to the executed local check. It performs no model
forward, optimizer update, GPU initialization, test read, or artifact upload.
Run directories must be supplied explicitly; no local path is embedded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def safe_checkpoint(run_dir: Path, relative: str) -> Path:
    path=(run_dir/relative).resolve()
    if run_dir.resolve() not in path.parents:
        raise ValueError("checkpoint path escapes its run directory")
    return path


def compare_runs(run_a: Path, run_b: Path) -> dict:
    # Heavy imports remain inside the explicit command path.
    import torch
    root=Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    from tools.evaluate_clinc_controlled_boundaries_v8 import _value_sha
    from checks.check_clinc_real3b_resume_v8 import _hash_value
    records=[]
    for run in (run_a.resolve(),run_b.resolve()):
        seal=json.loads((run/"development-sealed.json").read_text(encoding="utf-8"))
        if seal.get("complete") is not True or seal.get("immutable_base_unchanged") is not True:
            raise ValueError("run lacks a complete immutable-base development seal")
        if seal.get("test_data_read",False) is not False or seal.get("test_data_included",False) is not False:
            raise ValueError("refusing any run marked as reading test data")
        receipt=next((row for row in seal.get("runs",[]) if row.get("arm")=="GOLD_CE_MIX"),None)
        if receipt is None:
            raise ValueError("Gold control receipt missing")
        checkpoint=safe_checkpoint(run,receipt["checkpoint_endpoint"]["path"])
        cp_sha=sha256_file(checkpoint)
        if cp_sha!=receipt["checkpoint_endpoint"]["sha256"]:
            raise ValueError("Gold checkpoint bytes do not match the seal")
        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
        records.append({"run_name":run.name,"checkpoint_sha256":cp_sha,
            "context_sha256":_value_sha(state["context"]),
            "wake_sha256":_hash_value(state["wake"]),
            "rng_sha256":_hash_value(state["rng"])})
        del state
    result={"schema":"clinc-v8-gold-control-equivalence-v1","complete":True,
        "records":records,
        "context_exact":records[0]["context_sha256"]==records[1]["context_sha256"],
        "wake_exact":records[0]["wake_sha256"]==records[1]["wake_sha256"],
        "rng_exact":records[0]["rng_sha256"]==records[1]["rng_sha256"],
        "model_loaded":False,"optimizer_steps":0,"test_data_read":False,
        "interpretation":"Same development Gold control reused across objective variants; not an independent experimental seed."}
    if torch.cuda.is_initialized():
        raise RuntimeError("comparison unexpectedly initialized CUDA")
    result["cuda_initialized"]=False
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a",type=Path,required=True)
    parser.add_argument("--run-b",type=Path,required=True)
    parser.add_argument("--output-json",type=Path,required=True)
    args=parser.parse_args(argv)
    result=compare_runs(args.run_a,args.run_b)
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"status":"complete","test_data_read":False,"optimizer_steps":0,
                      "output":args.output_json.name},sort_keys=True))

if __name__=="__main__":
    main()
