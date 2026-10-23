"""Unified executable LM APIs, preserving native vs Llama adapter identity."""
from dataclasses import asdict
from copy import deepcopy
import torch
from torch import nn
from .native import NativeConfig, NativeHOPELM
from .experts import LowRankMemoryLevel, SparseExperts
from .accounting import charge


class ReadoutLoRA(nn.Module):
    """Zero-output rank-two LoRA on the LM readout, used for isolated trials/policy."""
    def __init__(self, original, rank=2):
        super().__init__()
        self.original = original
        self.a = nn.Parameter(original.weight.new_empty(rank, original.in_features).normal_(std=.02))
        self.b = nn.Parameter(original.weight.new_zeros(original.out_features, rank))
        self.fp32_accumulation = True

    @property
    def weight(self):
        return self.original.weight

    def forward(self, x):
        base = self.original(x)
        update = torch.nn.functional.linear(torch.nn.functional.linear(x.to(self.a.dtype), self.a), self.b)
        # FP32 readout adapters must retain their small logit contribution.
        if self.fp32_accumulation:
            return base.float() + update.float()
        return base.to(update.dtype) + update


def execution_state(model):
    """Non-tensor forward policy; must accompany weights in a deployment checkpoint."""
    return {'routing': [deepcopy(b.route_override) for b in model.banks()],
            'readout_fp32_accumulation': bool(model.core.head.fp32_accumulation
                if isinstance(model, NativeResearchLM) else model.backbone.lm_head.fp32_accumulation)}


def load_execution_state(model, state):
    if len(state['routing']) != len(model.banks()):
        raise ValueError('routing bank count mismatch')
    for bank, policy in zip(model.banks(), state['routing']):
        bank.route_override = deepcopy(policy)
    head = model.core.head if isinstance(model, NativeResearchLM) else model.backbone.lm_head
    head.fp32_accumulation = state['readout_fp32_accumulation']


class SamplingMixin:
    def banks(self):
        return [self.get_submodule(d['bank']) for d in self.level_descriptors()]

    def set_dream_routing(self, enabled, generator=None):
        for bank in self.banks():
            bank.dream, bank.generator = enabled, generator

    def policy_parameters(self):
        head = self.core.head if isinstance(self, NativeResearchLM) else self.backbone.lm_head
        return [head.a, head.b]

    def policy_names(self):
        prefix = 'core.head' if isinstance(self, NativeResearchLM) else 'backbone.lm_head'
        return [prefix + '.a', prefix + '.b']

    @torch.no_grad()
    def sample(self, prefix, count, *, generator, dream=False, temperature=.8):
        self.set_dream_routing(dream, generator)
        generated, trace = prefix.clone(), []
        state = None
        try:
            for step in range(count):
                if isinstance(self, NativeResearchLM):
                    inputs = generated if step == 0 else generated[:, -1:]
                    charge(inputs)
                    logits, state = self.core(inputs, state)
                else:
                    logits = self.logits(generated)
                next_token = torch.multinomial(torch.softmax(logits[:, -1].float() / temperature, -1),
                                               1, generator=generator)
                generated = torch.cat([generated, next_token], -1)
                trace.append([[dict(r) for r in b.last_routes] for b in self.banks()])
        finally:
            self.set_dream_routing(False)
        return generated, trace


class NativeResearchLM(nn.Module, SamplingMixin):
    architecture = 'native_hope'
    def __init__(self, cfg=None, rank=4, capacity=8):
        super().__init__()
        self.cfg = cfg or NativeConfig(vocab_size=264)
        self.core = NativeHOPELM(self.cfg)
        self.core.head = ReadoutLoRA(self.core.head)
        self.rank, self.capacity = rank, capacity
        for block in self.core.blocks:
            block.cms.expert_banks = nn.ModuleList([SparseExperts(self.cfg.dim, rank, capacity)
                                                   for _ in self.cfg.cms_periods])

    def logits(self, tokens, mask=None):
        charge(tokens)
        for bank in self.banks():
            bank.audit_history = []
        return self.core(tokens, valid=mask)[0]

    def forward(self, tokens, mask=None):
        return self.logits(tokens, mask)

    def level_descriptors(self):
        return [{'base': [f'core.blocks.{b}.cms.levels.{i}.wi', f'core.blocks.{b}.cms.levels.{i}.wo'],
                 'bank': f'core.blocks.{b}.cms.expert_banks.{i}'}
                for b, block in enumerate(self.core.blocks) for i in range(len(block.cms.levels))]

    def export_config(self):
        return {'kind': 'native', 'cfg': asdict(self.cfg), 'rank': self.rank, 'capacity': self.capacity}


class AugmentedMLP(nn.Module):
    def __init__(self, original, dim, rank, capacity, levels):
        super().__init__()
        self.original = original
        self.memory = nn.ModuleList([LowRankMemoryLevel(dim, rank, capacity) for _ in range(levels)])

    def forward(self, x):
        y = self.original(x)
        for level in self.memory:
            y = level(y)
        return y


class LlamaCMSLM(nn.Module, SamplingMixin):
    """Load original HF weights first, then attach separately saved adapters.

    Only last-MLP output is extended in the pilot. This is CMS-inspired Track B,
    not a replacement of attention with native self-modifying Titans.
    """
    architecture = 'llama_cms'
    def __init__(self, backbone, rank=4, capacity=8, levels=2):
        super().__init__()
        self.backbone = backbone
        self.rank, self.capacity, self.levels = rank, capacity, levels
        index = len(backbone.model.layers) - 1
        original = backbone.model.layers[index].mlp
        self.memory_prefix = f'backbone.model.layers.{index}.mlp.memory'
        device, dtype = next(backbone.parameters()).device, next(backbone.parameters()).dtype
        backbone.model.layers[index].mlp = AugmentedMLP(original, backbone.config.hidden_size,
                                                       rank, capacity, levels).to(device=device, dtype=dtype)
        backbone.lm_head = ReadoutLoRA(backbone.lm_head)

    def logits(self, tokens, mask=None):
        # Do not materialize all prompt vocabulary logits when only the last
        # position is needed; general training/GKD still requests full sequence.
        charge(tokens)
        for bank in self.banks():
            bank.audit_history = []
        return self.backbone(input_ids=tokens, attention_mask=mask, use_cache=False).logits

    def forward(self, tokens, mask=None):
        return self.logits(tokens, mask)

    def level_descriptors(self):
        return [{'base': [f'{self.memory_prefix}.{i}.a', f'{self.memory_prefix}.{i}.b'],
                 'bank': f'{self.memory_prefix}.{i}.experts'} for i in range(self.levels)]

    def export_config(self):
        return {'kind': 'llama_cms', 'rank': self.rank, 'capacity': self.capacity, 'levels': self.levels,
                'base': 'USER_SUPPLIED_LOCAL_ASSETS', 'memory_prefix': self.memory_prefix}

    def adapter_state_dict(self):
        return {n: t.detach().cpu().clone() for n, t in self.state_dict().items()
                if n.startswith(self.memory_prefix) or n in self.policy_names()}


def restore_native(config):
    return NativeResearchLM(NativeConfig(**config['cfg']), config['rank'], config['capacity'])


def immutable_versions(model):
    if not isinstance(model, LlamaCMSLM):
        return {}
    mutable = set(model.adapter_state_dict())
    return {n: (id(p), p._version) for n, p in model.named_parameters() if n not in mutable}


def isolated_clone(model, *, share_backbone=False):
    """Share only explicitly immutable Llama parameters, copy all mutable scopes.

    Sharing is opt-in; callers must version-guard the frozen base. Independent
    modules/buffers and mutable parameters remain distinct. No optimizer may own
    the shared base. Native models always receive a complete independent copy.
    """
    if not share_backbone or not isinstance(model, LlamaCMSLM):
        return deepcopy(model)
    mutable = set(model.adapter_state_dict())
    memo = {}
    for name, p in model.named_parameters():
        if name not in mutable:
            if p.requires_grad:
                raise ValueError('shared immutable backbone contains a trainable parameter: ' + name)
            memo[id(p)] = p
    return deepcopy(model, memo)


def promote_mutable_to_fp32(model):
    """Promote trainable adapter tensors before constructing their optimizer.

    Preserve router buffers and immutable backbone arithmetic. This is an
    explicit precision intervention, not a change to the objective or routing.
    """
    names = set(model.adapter_state_dict()) if isinstance(model, LlamaCMSLM) else set(dict(model.named_parameters()))
    for name, p in list(model.named_parameters()):
        if name in names:
            parent, leaf = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), leaf, nn.Parameter(p.detach().float(), requires_grad=p.requires_grad))
    return model
