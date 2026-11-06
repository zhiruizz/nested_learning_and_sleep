"""CPU forward/gradient parity tests for final-block tail caching."""
import unittest

import torch
from torch import nn

from nested_sleep.lm import AugmentedMLP, ReadoutLoRA
from tools.agnews_tail_cache_v7 import capture, tail_logits, tail_logits_padded


class TinyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_residual = nn.Linear(dim, dim, bias=True)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = AugmentedMLP(nn.Linear(dim, dim, bias=True), dim, rank=2, capacity=3, levels=2)


class TinyLlamaBackbone(nn.Module):
    def __init__(self, vocab=19, dim=6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(dim)])
        self.model.norm = nn.LayerNorm(dim)
        self.lm_head = ReadoutLoRA(nn.Linear(dim, vocab, bias=False), rank=2)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embed_tokens(input_ids)
        layer = self.model.layers[-1]
        residual = hidden + self_attention(layer, hidden)
        normalized = layer.post_attention_layernorm(residual)
        y = layer.mlp(normalized)
        return type("Out", (), {"logits": self.lm_head(self.model.norm(residual + y))})()


def self_attention(layer, hidden):
    # Deterministic nontrivial residual path; this stands in for attention output.
    return torch.tanh(layer.attention_residual(hidden)) * 0.2


class TinyLlamaCMSLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyLlamaBackbone()

    def banks(self):
        # Capture's audit restoration is tested using the memory-level objects.
        return list(self.backbone.model.layers[-1].mlp.memory)

    def logits(self, ids, mask=None):
        for bank in self.banks():
            bank.audit_history = []
        return self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).logits


class TailCacheV7CpuTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(771)
        self.model = TinyLlamaCMSLM().eval()
        self.ids = torch.randint(0, 19, (3, 6))
        self.mask = torch.ones_like(self.ids)
        self.lengths = torch.tensor([2, 3, 4])

    def _hook_counts(self):
        layer=self.model.backbone.model.layers[-1]
        return (len(layer.post_attention_layernorm._forward_pre_hooks),
                len(layer.mlp.original._forward_hooks))

    def test_capture_tail_logits_match_full_two_positions(self):
        before={k:v.detach().clone() for k,v in self.model.state_dict().items()}
        audit=[]
        for bank in self.model.banks():
            bank.audit_history=["prior-audit"]
            audit.append(bank.audit_history)
        hooks_before=self._hook_counts()
        cache=capture(self.model,self.ids,self.mask,self.lengths,device="cpu")
        self.assertEqual(self._hook_counts(),hooks_before)
        for bank,old in zip(self.model.banks(),audit):
            self.assertIs(bank.audit_history,old)
        idx=torch.arange(len(self.lengths))
        expected=self.model.logits(self.ids,self.mask)[idx[:,None],torch.stack((self.lengths-1,self.lengths),dim=1)]
        actual=tail_logits(self.model,cache["y"],cache["residual"])
        torch.testing.assert_close(cache["logits"],expected,rtol=0,atol=0)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        self.assertTrue(all(not x.requires_grad for x in cache.values()))
        for key,value in self.model.state_dict().items():
            torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_full_and_cached_tail_gradients_match_for_mutable_parameters(self):
        layer=self.model.backbone.model.layers[-1]
        params=list(layer.mlp.memory.parameters())+[self.model.backbone.lm_head.a,
                                                     self.model.backbone.lm_head.b]
        full=self.model.logits(self.ids,self.mask)
        idx=torch.arange(len(self.lengths))
        pos=torch.stack((self.lengths-1,self.lengths),dim=1)
        full_selected=full[idx[:,None],pos]
        full_loss=(full_selected.square()).mean()
        full_grads=torch.autograd.grad(full_loss,params,allow_unused=True)

        cache=capture(self.model,self.ids,self.mask,self.lengths)
        cached=tail_logits(self.model,cache["y"],cache["residual"])
        cached_loss=(cached.square()).mean()
        cached_grads=torch.autograd.grad(cached_loss,params,allow_unused=True)
        padded=tail_logits_padded(self.model,cache["y"],cache["residual"],self.lengths,self.ids.shape[1])
        padded_grads=torch.autograd.grad(padded.square().mean(),params,allow_unused=True)
        torch.testing.assert_close(cached,full_selected.detach(),rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(padded,full_selected.detach(),rtol=1e-6,atol=1e-6)
        for actual,expected in zip(cached_grads,full_grads):
            self.assertEqual(actual is None,expected is None)
            if actual is not None:
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        for actual,expected in zip(padded_grads,full_grads):
            self.assertEqual(actual is None,expected is None)
            if actual is not None:
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_invalid_lengths_fail_without_leaking_hooks(self):
        before=self._hook_counts()
        for bad in (torch.tensor([0,2,3]),torch.tensor([2,3,6]),torch.tensor([1.5,2.,3.])):
            with self.assertRaises(ValueError):capture(self.model,self.ids,self.mask,bad)
            self.assertEqual(self._hook_counts(),before)

    def test_padded_tail_rejects_invalid_lengths_and_shape(self):
        cache=capture(self.model,self.ids,self.mask,self.lengths)
        for bad in (torch.tensor([0,2,3]),torch.tensor([2,3,6]),torch.tensor([1.5,2.,3.])):
            with self.assertRaises(ValueError):
                tail_logits_padded(self.model,cache["y"],cache["residual"],bad,self.ids.shape[1])
        with self.assertRaises(ValueError):
            tail_logits_padded(self.model,cache["y"],cache["residual"],self.lengths,4)
        with self.assertRaises(ValueError):
            tail_logits_padded(self.model,cache["y"][:,:1],cache["residual"][:,:1],self.lengths,6)

    def test_hooks_are_removed_when_forward_raises(self):
        layer=self.model.backbone.model.layers[-1]
        original=self.model.logits
        def fail(*_args,**_kwargs):raise RuntimeError("injected forward failure")
        self.model.logits=fail
        before=self._hook_counts()
        try:
            with self.assertRaisesRegex(RuntimeError,"injected"):
                capture(self.model,self.ids,self.mask,self.lengths)
            self.assertEqual(self._hook_counts(),before)
        finally:
            self.model.logits=original


if __name__=="__main__":
    unittest.main()
