"""Exact late selection between old/new final-MLP memories and readout LoRAs.

The loaded LlamaCMSLM is the old/warm checkpoint. This helper reuses its
frozen final MLP memory and readout tensors as the old context, deep-copies
only the new context, and routes each batch row after one shared original MLP.
The frozen backbone, original MLP, and original LM head are never copied.

Gate decisions are frozen bools supplied per batch row. Fitting and scoring
the gate stay outside this module. A frozen ridge callback may consume the
shared original-MLP prefix feature in the same forward.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import MethodType
from typing import Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_GATE_MODES = {
    "fixed_correct_ridge",
    "fixed_shuffled_ridge",
    "constant_old",
    "constant_new",
    "deterministic_random",
}


def make_gate_mask(
    rows: int,
    mode: str,
    *,
    seed: int = 0,
    correct_mask: Optional[Tensor] = None,
    shuffled_mask: Optional[Tensor] = None,
    device: Optional[torch.device | str] = None,
) -> Tensor:
    """Build a fixed bool mask: True selects new context.

    Ridge decisions are precomputed by the caller. Random masks use a private
    CPU generator, independent of global Torch RNG state.
    """
    if rows < 0:
        raise ValueError("rows must be non-negative")
    if mode not in _GATE_MODES:
        raise ValueError(f"unknown gate mode: {mode}")
    if mode in ("fixed_correct_ridge", "fixed_shuffled_ridge"):
        source = correct_mask if mode == "fixed_correct_ridge" else shuffled_mask
        if source is None:
            raise ValueError(f"{mode} requires its precomputed mask")
        mask = torch.as_tensor(source, dtype=torch.bool).detach().reshape(-1)
        if mask.numel() != rows:
            raise ValueError(f"{mode} mask length {mask.numel()} != rows {rows}")
        return mask.clone().to(device=device)
    if mode == "constant_old":
        return torch.zeros(rows, dtype=torch.bool, device=device)
    if mode == "constant_new":
        return torch.ones(rows, dtype=torch.bool, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return (torch.rand(rows, generator=generator, device="cpu") < 0.5).to(device=device)


class FrozenRidgeGate:
    """CPU-float64 ridge decision using frozen standardization and weights.

    The callback reads the original final-MLP output at prefix_lengths[row]-1.
    It receives no labels and performs no backbone/model forward.
    """

    def __init__(self, mean, std, weights, bias=0.0, *, positive_selects_new=True):
        self.mean = torch.as_tensor(mean, dtype=torch.float64, device="cpu").reshape(-1).clone()
        self.std = torch.as_tensor(std, dtype=torch.float64, device="cpu").reshape(-1).clone()
        self.weights = torch.as_tensor(weights, dtype=torch.float64, device="cpu").reshape(-1).clone()
        self._bias = float(bias)
        if not (self.mean.shape == self.std.shape == self.weights.shape):
            raise ValueError("ridge mean/std/weights must have the same feature dimension")
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(self.std).all()
                    and torch.isfinite(self.weights).all()) or not torch.isfinite(torch.tensor(self._bias)):
            raise ValueError("ridge parameters must be finite")
        if bool((self.std <= 0).any()):
            raise ValueError("ridge standard deviations must be positive")
        self._positive_selects_new = bool(positive_selects_new)
        self.calls = 0
        self.last_features = None
        self._parameter_signature = self._signature()
        self._bias_snapshot = self._bias
        self._polarity_snapshot = self._positive_selects_new

    @property
    def bias(self):
        return self._bias

    @property
    def positive_selects_new(self):
        return self._positive_selects_new

    def _signature(self):
        return tuple((id(t), t._version) for t in (self.mean, self.std, self.weights))

    @torch.no_grad()
    def __call__(self, original_mlp_output: Tensor, prefix_lengths: Tensor) -> Tensor:
        if (self._signature() != self._parameter_signature
                or self._bias != self._bias_snapshot
                or self._positive_selects_new != self._polarity_snapshot):
            raise RuntimeError("frozen ridge parameters changed after construction")
        if original_mlp_output.ndim != 3:
            raise ValueError("ridge selector expects original-MLP output shaped [batch, sequence, hidden]")
        lengths = torch.as_tensor(prefix_lengths, dtype=torch.long).reshape(-1)
        batch, sequence, hidden = original_mlp_output.shape
        if lengths.numel() != batch or bool((lengths < 1).any()) or bool((lengths > sequence).any()):
            raise ValueError("prefix lengths must contain one valid length per batch row")
        if hidden != self.weights.numel():
            raise ValueError(f"ridge feature dimension {self.weights.numel()} != hidden size {hidden}")
        rows = torch.arange(batch, device=original_mlp_output.device)
        positions = lengths.to(original_mlp_output.device) - 1
        features = original_mlp_output[rows, positions].detach().to(device="cpu", dtype=torch.float64)
        scores = ((features - self.mean) / self.std / (hidden ** 0.5)) @ self.weights + self._bias
        selected = scores > 0
        self.last_features = features.clone()
        self.calls += 1
        return selected if self._positive_selects_new else ~selected


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module


def _row_where(mask: Tensor, old: Tensor, new: Tensor) -> Tensor:
    if old.shape != new.shape:
        raise ValueError(f"context output shapes differ: {old.shape} vs {new.shape}")
    if old.ndim < 2 or mask.ndim != 1 or mask.shape[0] != old.shape[0]:
        raise ValueError(f"gate mask must have one bool per batch row ({old.shape[0]})")
    view = (mask.shape[0],) + (1,) * (old.ndim - 1)
    return torch.where(mask.reshape(view), new, old)


def _apply_memory(memory: nn.ModuleList, base: Tensor) -> Tensor:
    result = base
    for level in memory:
        result = level(result)
    return result


class LateContextRouter:
    """Ephemeral router for a loaded LlamaCMSLM and a second adapter snapshot."""

    def __init__(
        self,
        model: nn.Module,
        new_adapter_state: Mapping[str, Tensor],
        new_execution_state: Mapping,
        *,
        old_execution_state: Optional[Mapping] = None,
        deduplicate_equal: bool = True,
    ):
        from nested_sleep.lm import LlamaCMSLM, execution_state

        if not isinstance(model, LlamaCMSLM):
            raise TypeError("requires nested_sleep.lm.LlamaCMSLM")
        self.model = model
        self.backbone = model.backbone
        layer_index = len(self.backbone.model.layers) - 1
        self.augmented_mlp = self.backbone.model.layers[layer_index].mlp
        if not hasattr(self.augmented_mlp, "original") or not hasattr(self.augmented_mlp, "memory"):
            raise TypeError("last layer MLP is not the expected AugmentedMLP")
        self.original_mlp = self.augmented_mlp.original
        self.readout_lora = self.backbone.lm_head
        if not all(hasattr(self.readout_lora, name) for name in ("original", "a", "b")):
            raise TypeError("backbone LM head is not ReadoutLoRA")
        self.original_head = self.readout_lora.original
        self.memory_prefix = model.memory_prefix
        self.old_execution_state = deepcopy(
            old_execution_state if old_execution_state is not None else execution_state(model)
        )
        self.new_execution_state = deepcopy(dict(new_execution_state))
        levels = len(self.augmented_mlp.memory)
        for name, state in (("old", self.old_execution_state), ("new", self.new_execution_state)):
            if len(state.get("routing", [])) != levels:
                raise ValueError(f"{name} execution state must have {levels} routing policies")
            if not isinstance(state.get("readout_fp32_accumulation"), bool):
                raise ValueError(f"{name} execution state lacks readout precision policy")

        # Inference reuses the loaded model's frozen old context; only the new
        # context is copied. The original final MLP and frozen LM head are shared.
        self.model.eval()
        self.old_memory = _freeze(self.augmented_mlp.memory)
        self.new_memory = _freeze(deepcopy(self.augmented_mlp.memory))
        self._load_new_memory(new_adapter_state)
        self._set_policies(self.old_memory, self.old_execution_state["routing"])
        self._set_policies(self.new_memory, self.new_execution_state["routing"])
        self.old_a = self.readout_lora.a.detach()
        self.old_b = self.readout_lora.b.detach()
        self.new_a = self._copy_parameter(
            new_adapter_state, "backbone.lm_head.a", self.readout_lora.a
        )
        self.new_b = self._copy_parameter(
            new_adapter_state, "backbone.lm_head.b", self.readout_lora.b
        )
        self.alias_map = self._deduplicate_equal_tensors() if deduplicate_equal else ()
        self.old_fp32 = self.old_execution_state["readout_fp32_accumulation"]
        self.new_fp32 = self.new_execution_state["readout_fp32_accumulation"]
        self.context_fingerprints = {
            "old": self._context_fingerprint(
                self.old_memory, self.old_a, self.old_b, self.old_execution_state
            ),
            "new": self._context_fingerprint(
                self.new_memory, self.new_a, self.new_b, self.new_execution_state
            ),
        }
        self._model_immutable_versions = self._immutable_model_versions()
        self._binding_ids = self._current_binding_ids()
        self._memory_versions = self._capture_versions()
        self.operation_counts = {
            "original_final_mlp_calls": 0,
            "old_memory_branch_calls": 0,
            "new_memory_branch_calls": 0,
            "original_lm_head_calls": 0,
            "old_readout_adapter_calls": 0,
            "new_readout_adapter_calls": 0,
        }
        self._active_mask: Optional[Tensor] = None
        self._active_selector = None
        self._active_prefix_lengths: Optional[Tensor] = None
        self._context_mode = "hard_route"
        self._active_forward = False
        self.operation_counts["selector_calls"] = 0
        self.operation_counts["uniform_mix_calls"] = 0

    @staticmethod
    def _copy_parameter(state, name, reference):
        value = state.get(name)
        if not isinstance(value, Tensor):
            raise TypeError(f"adapter entry {name!r} is missing or not a tensor")
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(f"adapter entry {name!r} has wrong shape")
        return value.detach().to(device=reference.device, dtype=reference.dtype).clone()

    def _load_new_memory(self, adapter_state):
        prefix = self.memory_prefix + "."
        state = {}
        for key, reference in self.new_memory.state_dict().items():
            full_key = prefix + key
            if full_key not in adapter_state:
                raise KeyError(f"new adapter state lacks {full_key}")
            value = adapter_state[full_key]
            if tuple(value.shape) != tuple(reference.shape):
                raise ValueError(f"new adapter entry {full_key!r} has wrong shape")
            state[key] = value.detach().to(device=reference.device, dtype=reference.dtype)
        self.new_memory.load_state_dict(state, strict=True)

    @staticmethod
    def _tensor_fingerprint_update(digest, name, tensor):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())

    @classmethod
    def _context_fingerprint(cls, memory, a, b, execution):
        digest = hashlib.sha256()
        for name, tensor in sorted(memory.state_dict().items()):
            cls._tensor_fingerprint_update(digest, "memory." + name, tensor)
        cls._tensor_fingerprint_update(digest, "head.a", a)
        cls._tensor_fingerprint_update(digest, "head.b", b)
        digest.update(json.dumps(execution, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return digest.hexdigest()

    def _deduplicate_equal_tensors(self):
        """Alias exact-equal new memory state to old, after loading is complete."""
        aliases = []
        old_state = self.old_memory.state_dict(keep_vars=True)
        new_state = self.new_memory.state_dict(keep_vars=True)
        for name in sorted(old_state):
            old, new = old_state[name], new_state[name]
            if (old.shape == new.shape and old.dtype == new.dtype
                    and torch.equal(old.detach(), new.detach())):
                parent_name, leaf = name.rsplit(".", 1) if "." in name else ("", name)
                parent = self.new_memory.get_submodule(parent_name) if parent_name else self.new_memory
                if leaf in parent._parameters:
                    parent._parameters[leaf] = old
                elif leaf in parent._buffers:
                    parent._buffers[leaf] = old
                else:
                    raise RuntimeError(f"unregistered state tensor {name}")
                aliases.append("memory." + name)
        for name in ("a", "b"):
            old = getattr(self, "old_" + name)
            new = getattr(self, "new_" + name)
            if old.shape == new.shape and old.dtype == new.dtype and torch.equal(old, new):
                setattr(self, "new_" + name, old)
                aliases.append("head." + name)
        return tuple(aliases)

    def _immutable_model_versions(self):
        mutable_names = set(self.model.policy_names())
        mutable_names.update(
            name for name, _ in self.model.named_parameters()
            if name.startswith(self.memory_prefix + ".")
        )
        current = dict(self.model.named_parameters())
        self._base_parameter_names = tuple(name for name in current if name not in mutable_names)
        trainable = [name for name in self._base_parameter_names if current[name].requires_grad]
        if trainable:
            raise ValueError("shared backbone/original MLP/head must be frozen: " + trainable[0])
        return self._immutable_model_versions_now()

    def _capture_versions(self):
        return {
            "old_memory": self._module_tensor_signature(self.old_memory),
            "new_memory": self._module_tensor_signature(self.new_memory),
            "heads": self._head_tensor_signature(),
            "module_ids": self._current_binding_ids(),
        }

    @staticmethod
    def _module_tensor_signature(module):
        entries = []
        for name, tensor in module.named_parameters():
            entries.append(("parameter", name, id(tensor), tensor._version, tensor.data_ptr()))
        for name, tensor in module.named_buffers():
            entries.append(("buffer", name, id(tensor), tensor._version, tensor.data_ptr()))
        return tuple(entries)

    def _head_tensor_signature(self):
        entries = []
        for name, tensor in (("old_a", self.old_a), ("old_b", self.old_b),
                             ("new_a", self.new_a), ("new_b", self.new_b)):
            entries.append((name, id(tensor), tensor._version, tensor.data_ptr()))
        return tuple(entries)

    def _current_binding_ids(self):
        layer_index = len(self.backbone.model.layers) - 1
        current_mlp = self.backbone.model.layers[layer_index].mlp
        current_head = self.backbone.lm_head
        return (
            id(self.model.backbone),
            id(self.backbone),
            id(current_mlp),
            id(current_mlp.original),
            id(current_mlp.memory),
            id(current_head),
            id(current_head.original),
            id(current_head.a),
            current_head.a.data_ptr(),
            id(current_head.b),
            current_head.b.data_ptr(),
        )

    @property
    def unique_context_tensor_bytes(self):
        """Storage bytes for distinct old/new memory and readout tensors."""
        tensors = []
        seen = set()
        for memory in (self.old_memory, self.new_memory):
            tensors.extend(memory.parameters())
            tensors.extend(memory.buffers())
        tensors.extend((self.old_a, self.old_b, self.new_a, self.new_b))
        total = 0
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        return total

    @property
    def model_plus_context_unique_payload_bytes(self):
        """Unique old/new adapter payload, excluding the frozen backbone."""
        return self.unique_context_tensor_bytes

    def _assert_unchanged(self):
        if self._model_immutable_versions != self._immutable_model_versions_now():
            raise RuntimeError("shared frozen backbone/MLP/head changed during routed evaluation")
        if self._current_binding_ids() != self._binding_ids:
            raise RuntimeError("shared model module or adapter binding was replaced")
        current = {
            "old_memory": self._module_tensor_signature(self.old_memory),
            "new_memory": self._module_tensor_signature(self.new_memory),
            "heads": self._head_tensor_signature(),
        }
        for scope in ("old_memory", "new_memory", "heads"):
            if current[scope] != self._memory_versions[scope]:
                raise RuntimeError(f"frozen routed {scope} tensor identity/version changed")

    def _immutable_model_versions_now(self):
        current = dict(self.model.named_parameters())
        missing = [name for name in self._base_parameter_names if name not in current]
        if missing:
            return (("missing", tuple(missing)),)
        return tuple((name, id(current[name]), current[name]._version)
                     for name in self._base_parameter_names)

    @staticmethod
    def _set_policies(memory, policies):
        for level, policy in zip(memory, policies):
            level.experts.route_override = deepcopy(policy)

    @torch.no_grad()
    def route_hidden(self, x: Tensor, new_context_mask: Optional[Tensor] = None) -> Tensor:
        """Run shared original MLP once, then select or uniformly mix contexts."""
        self._assert_unchanged()
        base = self.original_mlp(x)
        self.operation_counts["original_final_mlp_calls"] += 1
        if self._context_mode == "uniform_context_mix":
            mask = None
        elif self._active_selector is not None:
            mask = self._active_selector(base, self._active_prefix_lengths)
            self.operation_counts["selector_calls"] += 1
            mask = torch.as_tensor(mask, dtype=torch.bool, device=x.device).reshape(-1)
        else:
            if new_context_mask is None:
                raise ValueError("hard routing needs a mask or a frozen selector callback")
            mask = torch.as_tensor(new_context_mask, dtype=torch.bool, device=x.device).reshape(-1)
        if mask is not None and mask.numel() != x.shape[0]:
            raise ValueError(f"gate mask length {mask.numel()} != batch size {x.shape[0]}")
        self._active_mask = mask
        old = _apply_memory(self.old_memory, base)
        new = _apply_memory(self.new_memory, base)
        self.operation_counts["old_memory_branch_calls"] += 1
        self.operation_counts["new_memory_branch_calls"] += 1
        if self._context_mode == "uniform_context_mix":
            result = (old + new) * 0.5
            self.operation_counts["uniform_mix_calls"] += 1
        else:
            result = _row_where(mask, old, new)
        self._assert_unchanged()
        return result

    @torch.no_grad()
    def route_readout(self, hidden: Tensor, new_context_mask: Optional[Tensor] = None) -> Tensor:
        """Call shared frozen original head once and select/mix the two LoRAs."""
        self._assert_unchanged()
        base = self.original_head(hidden)
        self.operation_counts["original_lm_head_calls"] += 1
        old_update = F.linear(F.linear(hidden.to(self.old_a.dtype), self.old_a), self.old_b)
        new_update = F.linear(F.linear(hidden.to(self.new_a.dtype), self.new_a), self.new_b)
        self.operation_counts["old_readout_adapter_calls"] += 1
        self.operation_counts["new_readout_adapter_calls"] += 1
        old_logits = (base.float() + old_update.float()) if self.old_fp32 else (base.to(old_update.dtype) + old_update)
        new_logits = (base.float() + new_update.float()) if self.new_fp32 else (base.to(new_update.dtype) + new_update)
        if self._context_mode == "uniform_context_mix":
            result = (old_logits.float() + new_logits.float()) * 0.5
            self.operation_counts["uniform_mix_calls"] += 1
        else:
            mask = self._active_mask if new_context_mask is None else torch.as_tensor(
                new_context_mask, dtype=torch.bool, device=hidden.device
            ).reshape(-1)
            if mask is None or mask.numel() != hidden.shape[0]:
                raise ValueError("readout hard routing needs one active gate per batch row")
            view = (mask.shape[0],) + (1,) * (hidden.ndim - 1)
            result = torch.where(mask.reshape(view), new_logits, old_logits)
        self._assert_unchanged()
        return result

    @torch.no_grad()
    def logits(
        self,
        tokens: Tensor,
        new_context_mask: Optional[Tensor] = None,
        attention_mask=None,
        *,
        selector=None,
        prefix_lengths: Optional[Tensor] = None,
        mode: str = "hard_route",
    ) -> Tensor:
        """Temporarily patch one full forward; hard route or fixed half-mix."""
        self.model.eval()
        if torch.is_grad_enabled():
            raise RuntimeError("routed logits must run under torch.no_grad")
        if mode not in ("hard_route", "uniform_context_mix"):
            raise ValueError("mode must be hard_route or uniform_context_mix")
        if mode == "uniform_context_mix":
            if new_context_mask is not None or selector is not None:
                raise ValueError("uniform_context_mix does not take a gate mask or selector")
            mask = None
        elif selector is not None:
            if new_context_mask is not None or prefix_lengths is None:
                raise ValueError("callback routing needs prefix_lengths and no precomputed mask")
            if torch.as_tensor(prefix_lengths).numel() != tokens.shape[0]:
                raise ValueError("prefix_lengths must have one value per batch row")
            mask = None
        else:
            if new_context_mask is None:
                raise ValueError("hard_route requires a mask or selector")
            mask = torch.as_tensor(new_context_mask, dtype=torch.bool, device=tokens.device).reshape(-1)
            if mask.numel() != tokens.shape[0]:
                raise ValueError(f"gate mask length {mask.numel()} != batch size {tokens.shape[0]}")
        if self._active_forward:
            raise RuntimeError("router forward is not reentrant")
        self._active_forward = True
        self._active_mask = mask
        self._active_selector = selector
        self._active_prefix_lengths = None if prefix_lengths is None else torch.as_tensor(prefix_lengths)
        self._context_mode = mode
        modules = (self.augmented_mlp, self.readout_lora)
        prior = [(module, "forward" in module.__dict__, module.__dict__.get("forward")) for module in modules]
        old_bank_runtime = []
        for level in self.old_memory:
            bank = level.experts
            history = bank.audit_history
            old_bank_runtime.append((
                bank, bank.last_routes, bank.last_route_audit,
                history, len(history), bank.route_override, bank.dream,
                bank.generator, bank.audit_routes,
            ))

        def routed_mlp(_module, hidden):
            return self.route_hidden(hidden, self._active_mask)

        def routed_head(_module, hidden):
            return self.route_readout(hidden, self._active_mask)

        self.augmented_mlp.forward = MethodType(routed_mlp, self.augmented_mlp)
        self.readout_lora.forward = MethodType(routed_head, self.readout_lora)
        try:
            return self.backbone(
                input_ids=tokens,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
        finally:
            for module, was_instance, previous in prior:
                if was_instance:
                    module.forward = previous
                else:
                    del module.__dict__["forward"]
            for (bank, last_routes, last_audit, history, history_len,
                 route_override, dream, generator, audit_routes) in old_bank_runtime:
                del history[history_len:]
                bank.last_routes = last_routes
                bank.last_route_audit = last_audit
                bank.audit_history = history
                bank.route_override = route_override
                bank.dream = dream
                bank.generator = generator
                bank.audit_routes = audit_routes
            self._active_mask = None
            self._active_selector = None
            self._active_prefix_lengths = None
            self._context_mode = "hard_route"
            self._active_forward = False
