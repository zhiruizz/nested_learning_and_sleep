"""Estimate covered GEMM + attention FLOPs from a sealed CLINC V8 ledger.

These are analytical operation estimates for listed dense matmuls and attention
products, not measured or complete model FLOPs. Excluded operators are stated
in every report. Frozen-backbone backward is not counted.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

FORMULA_VERSION="clinc-v8-gemm-attention-estimate-4"
EXCLUDED=["RMSNorm", "activation", "RoPE", "softmax", "router", "CE/KL", "sampling", "adds/copies", "optimizer", "I/O"]

def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def parse_config(obj):
    c=obj.get("text_config",obj)
    fields={"H":c.get("hidden_size"),"I":c.get("intermediate_size"),"L":c.get("num_hidden_layers"),"heads":c.get("num_attention_heads"),"kvheads":c.get("num_key_value_heads"),"Dh":None,"V":c.get("vocab_size")}
    if fields["heads"]: fields["Dh"]=fields["H"]//fields["heads"]
    if any(v is None for v in fields.values()): raise ValueError("model config missing required Llama dimensions")
    if fields["H"]%fields["heads"]: raise ValueError("hidden size not divisible by head count")
    return fields

def hist_flops(hist, fn):
    total=0
    for key,count in hist.items():
        b,s=map(int,key.lower().split("x"))
        if b < 1 or s < 1 or type(count) is not int or count < 0:
            raise ValueError("invalid recorded shape/count")
        total+=count*fn(b,s)
    return total

def estimate_arm(work, dims, run_cfg, *, arm=None):
    H,I,L,KH,Dh,V=(dims[k] for k in ("H","I","L","kvheads","Dh","V"))
    r=int(run_cfg["rank"]); levels=int(run_cfg["levels"]); head_r=int(run_cfg.get("head_lora_rank",2))
    # Frozen pilot configs predate the cost-only branch field. Infer the
    # intervention from the actual arm, not the technical source-arm selector.
    if arm is not None:
        inferred = 2 if str(arm).upper().endswith("MIX") else 1
    else:
        inferred = 2 if any(str(a).upper().endswith("MIX") for a in
                            run_cfg.get("target_arms",run_cfg.get("arms",[]))) else 1
    branches=run_cfg.get("expert_branches_per_level_upper_bound",inferred)
    if type(branches) is not int or branches not in (1,2):
        raise ValueError("expert branch upper bound must be 1 (natural) or 2 (controlled mix)")
    def base(b,s): return 4*b*s*H*r*levels
    def head_lora(b,s): return 2*b*s*head_r*(H+V)
    def lm_head(b,s): return 2*b*s*H*V
    def backbone(b,s): return 2*b*s*L*(2*H*H+2*H*(KH*Dh)+3*H*I)+4*b*L*s*s*H
    fwd=work["forward_input_shape_histogram"]
    tail=work["tail_shape_histogram"]
    grad=work["tail_grad_shape_histogram"]
    f_transformer=hist_flops(fwd,backbone)
    # capture executes the original LM head and zero-output adapter matrices
    # too. The independent context then recomputes the head in tail_logits.
    f_capture_head=hist_flops(fwd,lambda b,s: lm_head(b,s)+head_lora(b,s))
    f_capture_memory=hist_flops(fwd,base)
    f_back=f_transformer+f_capture_head+f_capture_memory
    f_tail=hist_flops(tail,lambda b,s: lm_head(b,s)+head_lora(b,s)+base(b,s))
    # Shapes do not specify active/routed banks. A controlled mix may execute
    # both its natural and its selected branch, even if both name one slot.
    f_expert_hi=hist_flops(tail,lambda b,s: branches*levels*4*b*s*H*r)
    # Gradient-enabled tail calls are treated as executed backward. Frozen head GEMMs have input-grad cost.
    b_head=hist_flops(grad,lambda b,s: lm_head(b,s)+head_lora(b,s))
    b_mut_hi=hist_flops(grad,lambda b,s: 2*(base(b,s)+branches*levels*4*b*s*H*r))
    covered=[f_back+f_tail+b_head,f_back+f_tail+f_expert_hi+b_head+b_mut_hi]
    return {"capture_transformer_gemm_attention":f_transformer,
            "capture_original_head_and_readout_lora":f_capture_head,
            "capture_zero_output_memory_matrices":f_capture_memory,
            "frozen_backbone_forward":f_back,"tail_forward_head_and_memory":f_tail,
            "tail_forward_expert_branches_interval":[0,f_expert_hi],
            "expert_branches_per_level_upper_bound":branches,
            "tail_backward_frozen_head_input_grad":b_head,"tail_backward_mutable_interval":[0,b_mut_hi],
            "covered_total_interval":covered,"covered_total_teraflops_interval":[v/1e12 for v in covered]}

def build_report(seal, config, config_hash, run_config, run_config_hash):
    dims=parse_config(config); arms={}
    runs=seal.get("runs")
    if not isinstance(runs,list): raise ValueError("seal.runs must be list")
    for run in runs:
        arm=run.get("arm") or run.get("arm_id")
        work=run.get("work_arm_delta")
        if not arm or not isinstance(work,dict): raise ValueError("run missing arm/work_arm_delta")
        key=f"seed{run.get('seed','unspecified')}/{arm}"
        if key in arms: raise ValueError("duplicate seed/arm work record")
        arms[key]=estimate_arm(work,dims,run_config,arm=arm)
    return {"schema":"clinc-v8-covered-flops-estimate-v4",
            "formula_version":FORMULA_VERSION,
            "classification":"analytical covered GEMM+attention estimate; not measured FLOPs and not complete FLOPs",
            "model_config_sha256":config_hash,"run_config_sha256":run_config_hash,
            "dimensions":dims,
            "mutable_dimensions":{"rank":run_config["rank"],"levels":run_config["levels"],
                                  "head_lora_rank":run_config.get("head_lora_rank",2)},
            "arms":arms,"excluded_operators":EXCLUDED,
            "notes":["FLOPs assume 2 per multiply-accumulate.",
                     "Padding is counted using recorded BxS shapes.",
                     "No frozen-backbone backward is included.",
                     "Capture executes its frozen zero-output memory and readout matrices; tail recomputes the LM head.",
                     "Per-bank expert forward upper bounds are one branch for natural routing or two for an explicitly configured controlled mix.",
                     "Mutable gradient GEMMs use lower bound 0 and upper bound 2x listed forward GEMMs.",
                     "Every gradient-enabled tail call is assumed to reach backward; failed/unused diagnostic graphs require separate accounting.",
                     "The interval bounds the covered sparse/mutable GEMMs only; omitted operators are outside both endpoints.",
                     "Generation prefix recapture is included only insofar as present in the forward shape histogram; no KV-cache savings are assumed."]}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seal",required=True,type=Path); p.add_argument("--model-config",required=True,type=Path); p.add_argument("--run-config",required=True,type=Path); p.add_argument("--out",required=True,type=Path); a=p.parse_args()
    if a.out.exists(): raise FileExistsError("refusing to overwrite estimate")
    raw=a.seal.read_bytes(); seal=json.loads(raw); cfgraw=a.model_config.read_bytes(); runraw=a.run_config.read_bytes(); report=build_report(seal,json.loads(cfgraw),hashlib.sha256(cfgraw).hexdigest(),json.loads(runraw),hashlib.sha256(runraw).hexdigest()); report["seal_sha256"]=hashlib.sha256(raw).hexdigest(); a.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__": main()
