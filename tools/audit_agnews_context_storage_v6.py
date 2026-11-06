"""CPU-only storage/provenance audit; never scores articles or loads a model."""
from pathlib import Path
import hashlib
import json
import datetime
import argparse
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--v5-run-dir", type=Path, required=True)
parser.add_argument("--v5-router-result", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
args = parser.parse_args()
started = time.perf_counter()
RUN, OUT = args.v5_run_dir, args.out_dir
assert not OUT.exists()
seal = json.loads((RUN/"development-sealed.json").read_text(encoding="utf-8"))
router = json.loads(args.v5_router_result.read_text(encoding="utf-8"))
assert seal["complete"] is True

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def size(t):
    return t.numel()*t.element_size()
def digest(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

rows=[]
for seed in (17,29,43):
    run=next(r for r in seal["runs"] if r["seed"]==seed and r["arm"]=="warm_CMS_reference")
    cps={r["primary_step"]:r for r in run["checkpoints"]}
    paths=[(RUN/cps[step]["path"]).resolve() for step in (0,448)]
    assert all(p.is_relative_to(RUN.resolve()) for p in paths)
    assert all(sha(p)==cps[s]["sha256"] for s,p in zip((0,448),paths))
    warm,new=[torch.load(p,map_location="cpu",weights_only=False) for p in paths]
    states=[warm["model"],new["model"]]
    assert states[0].keys()==states[1].keys()
    tensors=[]
    unique_whole=set()
    whole_unique_bytes=0
    slotwise_units={}
    for name in states[0]:
        a,b=(state[name] for state in states)
        assert a.shape==b.shape and a.dtype==b.dtype
        tensors.append({"name":name,"shape":list(a.shape),"dtype":str(a.dtype),"bytes_per_context":size(a),
            "equal":torch.equal(a,b),"changed_elements":int(torch.count_nonzero(a!=b))})
        for t in (a,b):
            key=(name,tuple(t.shape),str(t.dtype),digest(t))
            if key not in unique_whole:
                unique_whole.add(key);whole_unique_bytes+=size(t)
            # Optional storage-only upper-bound compression unit: expert slots.
            # This is an audit, not the actual runtime representation.
            parts=list(t) if ".experts." in name and t.ndim>=2 and t.shape[0]==32 else [t]
            for i,part in enumerate(parts):
                unit=(name,i,tuple(part.shape),str(part.dtype),digest(part))
                slotwise_units[unit]=size(part)
    rows.append({"seed":seed,"checkpoint_file_bytes":dict(zip(("warm","new448"),(p.stat().st_size for p in paths))),
        "checkpoint_sha256":{"warm":cps[0]["sha256"],"new448":cps[448]["sha256"]},
        "tensor_bytes_per_context":sum(size(t) for t in states[0].values()),
        "two_context_tensor_bytes_actual_independent_copies":sum(size(t) for s in states for t in s.values()),
        "deduplicated_whole_tensor_storage_bytes":whole_unique_bytes,
        "deduplicated_expert_slot_storage_bytes_theoretical":sum(slotwise_units.values()),
        "distinct_tensor_count":sum(not t["equal"] for t in tensors),
        "total_tensor_count_per_context":len(tensors),"tensors":tensors,
        "warm_execution_policy":warm["execution_state"],"new_execution_policy":new["execution_state"],
        "warm_active_parameters":run["validation"][0].get("active_parameters"),
        "new_active_parameters":run["final_active_parameters"]})
    del warm,new,states
gates=[r["gate"] for r in router["seeds"]]
record={"schema":"agnews-routing-storage-audit-v6","rows":rows,
    "v5_gate_feature_dims":[g["feature_dim"] for g in gates],
    "v5_gate_model_sha256":[g["model_sha256"] for g in gates],
    "v5_gate_normalizer_sha256":[g["normalizer_sha256"] for g in gates],
    "all_three_gate_fits_identical":len({g["model_sha256"] for g in gates})==1,
    "minimal_gate_payload_bytes_float64_mean_scale_weight":gates[0]["feature_dim"]*3*8,
    "scoring_or_training":False,"CUDA_visible_devices":"-1",
    "source_sha256":sha(Path(__file__)),"main_seal_sha256":sha(RUN/"development-sealed.json"),
    "wall_seconds":time.perf_counter()-started,
    "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
OUT.mkdir(parents=True)
(OUT/"audit.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
print(json.dumps({"seeds":3,"gates_identical":record["all_three_gate_fits_identical"],"gate_bytes":record["minimal_gate_payload_bytes_float64_mean_scale_weight"],
    "state_summaries":[{k:r[k] for k in ("seed","tensor_bytes_per_context","two_context_tensor_bytes_actual_independent_copies","deduplicated_whole_tensor_storage_bytes","deduplicated_expert_slot_storage_bytes_theoretical","distinct_tensor_count")} for r in rows]}))
