"""Bounded real-Llama-3B v8 tail/runtime qualification (no dataset access).

This check uses only two hard-coded short prompts. It compares public
LlamaCMSLM full forwards with the v8 detached-tail path, including gradients
after a small deterministic adapter mutation, then checks mutable-context
snapshot/restore while the frozen backbone remains unchanged. It does not train
or create an optimizer. Maximum budget: 40 full sequence forwards, 0 Adam calls.

Example (run only after review):
  python checks/check_clinc_real3b_v8.py --project-root D:/my_paper/NL_Sleep_Public_Candidate \
      --model-path D:/my_paper/Nested_Learning_and_Sleep/.cache/models/llama32-3b \
      --device cuda:0 --output-dir outputs/clinc-v8-tail-qualification
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


MAX_FORWARDS = 40
RANK = 64
LEVELS = 5
TEST_CAPACITY = 2
PROMPTS = (
    "Please check the balance in my savings account.",
    "I would like to book a table for two tonight.",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _tree_digest(value: Any) -> str:
    """Stable digest for small adapter/context state only; never hashes backbone."""
    h = hashlib.sha256()

    def visit(item, prefix=""):
        if hasattr(item, "detach") and hasattr(item, "shape"):
            tensor = item.detach().contiguous().cpu()
            h.update(prefix.encode("utf-8"))
            h.update(str(tensor.dtype).encode("ascii"))
            h.update(str(tuple(tensor.shape)).encode("ascii"))
            h.update(tensor.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], prefix + "/" + str(key))
        elif isinstance(item, (list, tuple)):
            for i, child in enumerate(item):
                visit(child, prefix + f"/{i}")
        elif item is None or isinstance(item, (str, int, float, bool)):
            h.update((prefix + "=" + repr(item)).encode("utf-8"))
        else:
            raise TypeError(f"unsupported mutable-state value at {prefix}: {type(item)!r}")

    visit(value)
    return h.hexdigest()


def _add_source_paths(project_root: Path) -> tuple[Path, Path]:
    source = (project_root / "src").resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"public source directory not found: {source}")
    script_root = Path(__file__).resolve().parents[1]
    runtime_candidates = [project_root / "tools" / "clinc_tail_runtime_v8.py",
                          script_root / "tools" / "clinc_tail_runtime_v8.py"]
    runtime_path = next((p.resolve() for p in runtime_candidates if p.is_file()), None)
    if runtime_path is None:
        raise FileNotFoundError("clinc_tail_runtime_v8.py not found under project tools/ or task tools/")
    for item in (str(source), str(runtime_path.parent)):
        if item in sys.path:
            sys.path.remove(item)
        sys.path.insert(0, item)
    return source, runtime_path


def _max_abs(x, y) -> float:
    import torch
    if x.shape != y.shape:
        raise AssertionError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    return float((x.detach().float() - y.detach().float()).abs().max().item())


def _relative_l2(x, y) -> float:
    import torch
    xf, yf = x.detach().float(), y.detach().float()
    return float(torch.linalg.vector_norm(xf - yf) /
                 torch.linalg.vector_norm(xf).clamp_min(1e-20))


def _deterministic_mutation(model, seed: int = 82731) -> None:
    """Create a small nonzero mutable path without optimizer updates."""
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    layer = model.backbone.model.layers[-1].mlp
    with torch.no_grad():
        for i, level in enumerate(layer.memory):
            bank = level.experts
            bank.activate(0)
            bank.route_override = {"force": 0}
            slot = bank.slots[0]
            # Deterministic small updates make the parity check sensitive to
            # adapter execution while staying far from any training claim.
            slot.a.copy_(torch.randn(slot.a.shape, generator=generator).to(slot.a) * .004)
            slot.b.copy_(torch.randn(slot.b.shape, generator=generator).to(slot.b) * .002)
            level.b.copy_(torch.randn(level.b.shape, generator=generator).to(level.b) * .002)
        head = model.backbone.lm_head
        head.a.copy_(torch.randn(head.a.shape, generator=generator).to(head.a) * .002)
        head.b.copy_(torch.randn(head.b.shape, generator=generator).to(head.b) * .001)


def _set_trainable_adapters(model, torch):
    adapter_state_names = set(model.adapter_state_dict())
    names = {name for name, _parameter in model.named_parameters()
             if name in adapter_state_names}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != names:
        raise AssertionError("trainable adapter scope differs from adapter_state_dict")
    return names


def run(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model assets not found: {model_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    src_path, runtime_path = _add_source_paths(project_root)
    from nested_sleep.lm import LlamaCMSLM, immutable_versions, promote_mutable_to_fp32
    import clinc_tail_runtime_v8 as tail

    lm_path = Path(sys.modules["nested_sleep.lm"].__file__).resolve()
    if src_path not in lm_path.parents:
        raise RuntimeError(f"public-source binding failed; nested_sleep.lm loaded from {lm_path}")
    if Path(tail.__file__).resolve() != runtime_path:
        raise RuntimeError("v8 runtime module path mismatch")

    torch.manual_seed(1907008)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1907008)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(list(PROMPTS), return_tensors="pt", padding=True,
                        truncation=False, add_special_tokens=True)
    ids = encoded["input_ids"].to(args.device)
    mask = encoded["attention_mask"].to(args.device)
    if ids.shape[1] > 128:
        raise RuntimeError("manual fixture unexpectedly exceeds the 128-token bound")

    backbone = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager").eval()
    backbone.requires_grad_(False)
    model = LlamaCMSLM(backbone, rank=RANK, capacity=TEST_CAPACITY, levels=LEVELS)
    promote_mutable_to_fp32(model)
    model.eval()
    mutable_names = _set_trainable_adapters(model, torch)
    frozen_before = immutable_versions(model)
    forward_calls = 0
    original_logits_method = model.logits

    def counted_logits(tokens, attention_mask=None):
        nonlocal forward_calls
        forward_calls += 1
        if forward_calls > MAX_FORWARDS:
            raise RuntimeError(f"forward budget exceeded ({MAX_FORWARDS})")
        return original_logits_method(tokens, attention_mask)

    model.logits = counted_logits

    # Initial zero-output adapter path: compare full logits with fresh tail logits.
    baseline_context = tail.clone_mutable_context(model)
    baseline_cache = tail.capture(model, ids, mask)
    baseline_tail = tail.tail_logits(model, baseline_context, baseline_cache)
    baseline_full = baseline_cache.reference_logits
    baseline_max_abs = _max_abs(baseline_full, baseline_tail)

    # Apply deterministic small adapter values; this is a mutation/parity probe,
    # not an Adam step or a learning run.
    _deterministic_mutation(model)
    mutable_context = tail.clone_mutable_context(model)
    cache = tail.capture(model, ids, mask)
    full_logits = model.logits(ids, mask)
    tail_logits = tail.tail_logits(model, mutable_context, cache)
    mutated_max_abs = _max_abs(full_logits, tail_logits)

    # Compare gradients for the same fixed full-vocabulary scalar objective.
    objective_weights = torch.linspace(-0.25, 0.25, full_logits.numel(),
                                       device=full_logits.device, dtype=torch.float32).reshape_as(full_logits)
    full_loss = (full_logits.float() * objective_weights).sum()
    model_parameters = dict(model.named_parameters())
    full_params = [model_parameters[name] for name in sorted(mutable_names)]
    full_grads = torch.autograd.grad(full_loss, full_params, allow_unused=True)

    tail_objective = (tail_logits.float() * objective_weights).sum()
    context_by_name = dict(mutable_context.named_parameters())
    context_name_map = {}
    for name in sorted(mutable_names):
        if name.startswith(model.memory_prefix + "."):
            context_name_map[name] = "memory." + name[len(model.memory_prefix) + 1:]
        elif name.endswith(".lm_head.a"):
            context_name_map[name] = "readout_a"
        elif name.endswith(".lm_head.b"):
            context_name_map[name] = "readout_b"
        else:
            raise AssertionError(f"unrecognized mutable parameter name {name}")
    context_parameters = [context_by_name[context_name_map[name]] for name in sorted(mutable_names)]
    context_grads = torch.autograd.grad(tail_objective, context_parameters, allow_unused=True)
    if len(full_grads) != len(context_grads):
        raise AssertionError("full and tail mutable parameter lists differ in size")
    gradient_diffs = []
    for name, grad_full, grad_tail in zip(sorted(mutable_names), full_grads, context_grads):
        if (grad_full is None) != (grad_tail is None):
            raise AssertionError(f"gradient-usage mismatch for {name}")
        if grad_full is not None:
            gradient_diffs.append({"name": name, "max_abs": _max_abs(grad_full, grad_tail),
                                   "relative_l2": _relative_l2(grad_full, grad_tail)})
    parity_tolerance = 1e-4
    gradient_tolerance = 1e-4
    if baseline_max_abs > parity_tolerance or mutated_max_abs > parity_tolerance:
        raise AssertionError("full-vs-tail logits parity exceeded the fixed 1e-4 tolerance")
    if any(item["max_abs"] > gradient_tolerance or item["relative_l2"] > gradient_tolerance
           for item in gradient_diffs):
        raise AssertionError("full-vs-tail adapter gradient parity exceeded fixed 1e-4 tolerance")

    # Compact mutable context state roundtrip; confirm exact state digest and
    # exact tail output after restoring into a separately cloned context.
    state = mutable_context.state_dict()
    state_digest = _tree_digest(state)
    restored_context = tail.clone_mutable_context(model)
    restored_context.load_state_dict(state)
    restored_digest = _tree_digest(restored_context.state_dict())
    restored_logits = tail.tail_logits(model, restored_context, cache)
    restore_max_abs = _max_abs(tail_logits.detach(), restored_logits)

    frozen_after = immutable_versions(model)
    if frozen_before != frozen_after:
        raise AssertionError("frozen backbone parameter versions changed during the check")
    if forward_calls > MAX_FORWARDS:
        raise AssertionError("forward budget exceeded")

    result = {
        "schema": "clinc-real3b-tail-v8-check-v1",
        "complete": True,
        "model_path": str(model_path),
        "model_asset_files": len(list(model_path.rglob("*"))),
        "public_source_root": str(src_path),
        "nested_sleep_lm_path": str(lm_path),
        "nested_sleep_lm_sha256": _sha256(lm_path),
        "tail_runtime_path": str(runtime_path),
        "tail_runtime_sha256": _sha256(runtime_path),
        "architecture_scope": "LlamaCMSLM last MLP only; rank64, levels5, test capacity2; not native HOPE",
        "device": str(args.device),
        "dtype_backbone": "bfloat16",
        "dtype_mutables": "float32",
        "manual_prompt_count": len(PROMPTS),
        "sequence_lengths": mask.sum(-1).detach().cpu().tolist(),
        "max_sequence_tokens": int(ids.shape[1]),
        "full_sequence_forward_calls": forward_calls,
        "adam_calls": 0,
        "max_forward_budget": MAX_FORWARDS,
        "zero_adapter_full_tail_max_abs": baseline_max_abs,
        "mutated_adapter_full_tail_max_abs": mutated_max_abs,
        "logit_parity_tolerance": parity_tolerance,
        "mutable_gradient_comparisons": gradient_diffs,
        "gradient_parity_tolerance": gradient_tolerance,
        "mutable_context_state_sha256": state_digest,
        "restored_context_state_sha256": restored_digest,
        "context_state_roundtrip_exact": state_digest == restored_digest,
        "context_restore_tail_max_abs": restore_max_abs,
        "frozen_backbone_versions_unchanged": True,
        "elapsed_seconds": time.perf_counter() - started,
        "notes": [
            "Only hard-coded manual prompts were used; no CLINC records or labels were read.",
            "No optimizer was created and no learning step was run.",
            "The v8 capture retains a full reference-logit tensor; short prompts bound its memory use.",
            "This checks mutable-context state roundtrip, not SleepSession/Adam resume semantics.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "check-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                                     encoding="utf-8", newline="\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True, help="for example cuda:0; script never downloads assets")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError("refusing to overwrite qualification output directory")
    out.mkdir(parents=True)
    try:
        result = run(args)
    except Exception as exc:
        (out / "failure.json").write_text(json.dumps({"complete": False,
            "exception_type": type(exc).__name__, "error": str(exc),
            "max_forward_budget": MAX_FORWARDS, "adam_calls": 0}, indent=2) + "\n", encoding="utf-8")
        raise
    print(json.dumps({k: result[k] for k in (
        "complete", "device", "full_sequence_forward_calls", "adam_calls",
        "zero_adapter_full_tail_max_abs", "mutated_adapter_full_tail_max_abs",
        "context_state_roundtrip_exact", "context_restore_tail_max_abs", "elapsed_seconds")},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
