"""Versioned opt-in Phase13 engineering executors, restricted fixed-token KS.

CACHE stores ONLY the frozen decoder-prefix residual and original last-MLP
output. CMS, residual addition, final norm and the complete readout stay live
under the caller's autograd mode. HOST packs independent diagnostic exports;
it does not change objective/reductions/optimizer/clip or protection locations.
This module never enables a candidate by default or edits historical drivers.
"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json,time,types
import struct
import torch
from .accounting import charge

VERSION='phase13-executor-v1'
DEFAULT_CACHE_BYTES=512*2**20

def bitwise_equal(a,b):
    """Strict tensor bytes, including signed zeros and equal NaN payloads."""
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(bitwise_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(bitwise_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,float) and struct.pack('!d',a)==struct.pack('!d',b)
    return type(a) is type(b) and a==b

class _PrefixReady(Exception):pass
class UnsupportedScope(RuntimeError):pass

def tensor_bytes(value):
    if value is None:return b'None'
    data=value.detach().cpu().contiguous()
    return repr((tuple(data.shape),str(data.dtype),str(value.device))).encode()+data.reshape(-1).view(torch.uint8).numpy().tobytes()

def packed_floats(values):
    """No reductions: export already computed independent scalar values."""
    if not values:return []
    assert all(v.numel()==1 and v.dtype==values[0].dtype and v.device==values[0].device for v in values)
    return torch.stack([v.detach().reshape(()) for v in values]).cpu().tolist()

def packed_gradients(parameters):
    """Immutable CPU snapshot, preserving None, shape, dtype and exact bits."""
    values=[p.grad for p in parameters];present=[v for v in values if v is not None]
    if not present:return [None for _ in values]
    assert all(v.dtype==present[0].dtype and v.device==present[0].device for v in present)
    flat=torch.cat([v.detach().reshape(-1) for v in present]).cpu();offset=0;result=[]
    for value in values:
        if value is None:result.append(None)
        else:
            result.append(flat[offset:offset+value.numel()].reshape(value.shape).clone());offset+=value.numel()
    return result

def host_optimize(session,loss,kind,tokens):
    """Original _optimize arithmetic and guard order, packed scalar transport.

    Norms are still each original .float().norm() operation; they are computed
    BEFORE clip, retained as scalar tensors, and exported with the original
    clip result before the original reject-zero decision/optimizer step.
    """
    if tokens<=0 or not loss.requires_grad:raise RuntimeError('no effective token/gradient path in '+kind)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite '+kind+' loss')
    if session.step_observer is not None:session.step_observer(session,'before',loss,kind,tokens)
    session.optimizer.zero_grad(set_to_none=True);loss.backward()
    parameters=(session.new_slot.a,session.new_slot.b)
    before=[p.detach().clone() for p in parameters]
    norms=[p.grad.detach().float().norm() if p.grad is not None else None for p in parameters]
    clip=torch.nn.utils.clip_grad_norm_(list(parameters),1.)
    exported=packed_floats([v for v in norms if v is not None]+[clip]);at=0;gradient=[]
    for value in norms:
        if value is None:gradient.append(None)
        else:gradient.append(float(exported[at]));at+=1
    grad_norm=float(exported[-1])
    if session.reject_zero_gradient and grad_norm==0. and abs(float(loss.detach()))>1e-6:
        session.optimizer.zero_grad(set_to_none=True);session.stage='failed'
        raise RuntimeError('nonzero objective but receiving expert gradient is zero; audit loss-position routing')
    session.optimizer.step()
    if session.step_observer is not None:session.step_observer(session,'after',loss,kind,tokens)
    session.optimizer.zero_grad(set_to_none=True)
    deltas=[(p.detach().float()-b.float()).norm() for p,b in zip(parameters,before)]
    result=packed_floats([loss.detach()]+deltas)
    session.records.append({'kind':kind,'loss':float(result[0]),'grad_norm':grad_norm,'tokens':tokens,
        'parameter_grad_norms':gradient,'parameter_delta_norms':[float(x) for x in result[1:]]})

@contextmanager
def host_reads(session):
    original=session._optimize
    session._optimize=types.MethodType(host_optimize,session)
    try:yield
    finally:session._optimize=original

def optimizer_diagnostics(optimizer,packed=False):
    states=list(optimizer.state.values())
    if not packed:
        return [{'step':float(v['step']),'first_moment_norm':float(v['exp_avg'].norm()),'second_moment_norm':float(v['exp_avg_sq'].norm())} for v in states]
    norms=[x for state in states for x in [state['exp_avg'].norm(),state['exp_avg_sq'].norm()]]
    values=packed_floats(norms)
    return [{'step':float(state['step']),'first_moment_norm':float(values[2*i]),'second_moment_norm':float(values[2*i+1])} for i,state in enumerate(states)]

def route_report_packed(model,*,position_slice=slice(None)):
    """Same original per-slot sum/mean reductions; independent exports packed."""
    rows=[]
    for i,bank in enumerate(model.banks()):
        audit=bank.last_route_audit
        if audit is None:rows.append({'bank':i,'active':[],'loss_positions':0});continue
        history=bank.audit_history or [audit]
        selected=torch.cat([r['selected'][:,None] if r['selected'].ndim==1 else r['selected'] for r in history],dim=1)[:,position_slice].reshape(-1)
        probabilities=torch.cat([r['probabilities'][:,None] if r['probabilities'].ndim==2 else r['probabilities'] for r in history],dim=1)[:,position_slice].reshape(-1,len(audit['active'])+1).float()
        counts=[(selected==slot).sum() for slot in audit['active']]
        count_values=torch.stack(counts).cpu().tolist() if counts else []
        means=[probabilities[:,j].mean() for j in range(len(audit['active'])+1)]
        values=packed_floats(means)
        row={'bank':i,'active':audit['active'],'loss_positions':selected.numel(),
            'selected_counts':{str(slot):int(count_values[j]) for j,slot in enumerate(audit['active'])},
            'mean_probabilities':{str(slot):float(values[j]) for j,slot in enumerate(audit['active'])},
            'zero_fallback_mean_probability':float(values[-1]),'score_abs_max':audit['score_abs_max']}
        row['executed_counts']=dict(row['selected_counts'])
        if audit.get('mixed_slot') is not None:
            row['executed_counts'][str(audit['mixed_slot'])]=selected.numel();row['mixed_fraction']=audit['mixed_fraction']
        if audit.get('included_slot') is not None:
            slot=str(audit['included_slot'])
            previous=torch.cat([r['old_selected'][:,None] if r['old_selected'].ndim==1 else r['old_selected'] for r in history],dim=1)[:,position_slice].reshape(-1)
            old_counts=[(previous==s).sum() for s in audit['active']]
            old_values=torch.stack(old_counts).cpu().tolist() if old_counts else []
            row['executed_counts']={str(s):int(old_values[j]) for j,s in enumerate(audit['active'])};row['executed_counts'][slot]=selected.numel();row['included_slot']=int(slot)
        rows.append(row)
    return rows

class FrozenFrontLRU:
    """Process-local bounded cache. Unsupported scopes explicitly use REF.

    Only fixed, integer input tokens plus optional attention mask are accepted
    by this model API. Custom position/cache kwargs are keyed by key_for_inputs
    for counterexamples, but rejected by logits before any mutable operation;
    original model.logits itself has no such public kwargs. No cross-model reuse.
    """
    def __init__(self,max_bytes=DEFAULT_CACHE_BYTES,code_hash=VERSION):
        self.max_bytes=int(max_bytes);self.code_hash=code_hash;self.entries=OrderedDict();self.bytes=0;self.bound={};self.epoch={};self.original={};self.namespace_counter=0
        self.stats={'logical_calls':0,'physical_prefix_executions':0,'hits':0,'misses':0,'evictions':0,'invalidations':0,'fallbacks':0,
                    'key_seconds':0.,'dependency_guard_seconds':0.,'build_seconds':0.,'hit_seconds':0.,'key_D2H_bytes':0,'H2D_bytes_by_cache':0,'loads':0,'peak_payload_bytes':0,'fallback_reasons':{}}
    def clear(self):self.entries.clear();self.bytes=0
    def _dependency_signature(self,m):
        if not hasattr(m,'memory_prefix') or m.architecture!='llama_cms':raise UnsupportedScope('only pinned TrackB LlamaCMS fixed-token KS')
        if torch.is_inference_mode_enabled():raise UnsupportedScope('inference_mode payload not accepted for later receiver autograd')
        excluded=(m.memory_prefix,'backbone.lm_head','backbone.model.norm')
        relevant=lambda n:not any(n.startswith(prefix) for prefix in excluded)
        parameters=[]
        for name,value in m.named_parameters():
            if relevant(name):
                if value.requires_grad:raise UnsupportedScope('prefix parameter trainable: '+name)
                parameters.append((name,id(value),value._version,str(value.dtype),str(value.device)))
        buffers=[(name,id(value),value._version,str(value.dtype),str(value.device)) for name,value in m.named_buffers() if relevant(name)]
        modes=[]
        for name,module in m.named_modules():
            if relevant(name):
                if module.training:raise UnsupportedScope('frozen prefix requires eval mode: '+name)
                if module._forward_hooks or module._forward_pre_hooks:raise UnsupportedScope('prefix external hook semantics require REF: '+name)
                modes.append((name,module.training))
        precision=(torch.is_autocast_enabled('cuda'),str(torch.get_autocast_dtype('cuda')),torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        return (tuple(parameters),tuple(buffers),tuple(modes),json.dumps(m.backbone.config.to_dict(),sort_keys=True,default=str),precision,self.code_hash)
    def dependency_signature(self,m):
        began=time.perf_counter()
        try:return self._dependency_signature(m)
        finally:self.stats['dependency_guard_seconds']+=time.perf_counter()-began
    def bind(self,model,original):
        signature=self.dependency_signature(model);self.clear();self.namespace_counter+=1
        self.bound[id(model)]=signature;self.epoch[id(model)]=self.namespace_counter;self.original[id(model)]=original
    def invalidate(self,model):
        self.clear();self.epoch[id(model)]=self.epoch.get(id(model),0)+1;self.stats['invalidations']+=1
    def key_for_inputs(self,model,tokens,mask=None,position_ids=None,cache_position=None):
        began=time.perf_counter();h=sha256(repr((id(model),self.epoch.get(id(model),0),self.bound.get(id(model)),self.code_hash)).encode())
        for value in [tokens,mask,position_ids,cache_position]:
            h.update(tensor_bytes(value))
            if value is not None and value.device.type!='cpu':self.stats['key_D2H_bytes']+=value.numel()*value.element_size()
        h.update(repr(('implicit_positions' if position_ids is None and cache_position is None else 'explicit_positions',tuple(tokens.shape),str(tokens.dtype),str(tokens.device))).encode())
        self.stats['key_seconds']+=time.perf_counter()-began;return h.hexdigest()
    def payload(self,m,tokens,mask,signature=None):
        signature=self.dependency_signature(m) if signature is None else signature
        if self.bound[id(m)] is None:self.bound[id(m)]=signature
        elif signature!=self.bound[id(m)]:self.invalidate(m);self.bound[id(m)]=signature
        key=self.key_for_inputs(m,tokens,mask)
        if key in self.entries:
            began=time.perf_counter();payload,size,versions=self.entries.pop(key)
            if tuple(v._version for v in payload)==versions:
                self.entries[key]=(payload,size,versions);self.stats['hits']+=1;self.stats['hit_seconds']+=time.perf_counter()-began;return payload
            self.bytes-=size;self.stats['invalidations']+=1
        self.stats['misses']+=1;self.stats['physical_prefix_executions']+=1
        layer=m.backbone.model.layers[-1];captured={}
        def residual(module,args):captured['residual']=args[0].detach();return None
        def ready(module,args,output):captured['mlp']=output.detach();raise _PrefixReady()
        handles=[layer.post_attention_layernorm.register_forward_pre_hook(residual),layer.mlp.original.register_forward_hook(ready)]
        began=time.perf_counter()
        try:
            with torch.no_grad():m.backbone.model(input_ids=tokens,attention_mask=mask,use_cache=False)
        except _PrefixReady:pass
        finally:
            for handle in handles:handle.remove()
        self.stats['build_seconds']+=time.perf_counter()-began
        if captured.keys()!={'residual','mlp'}:raise RuntimeError('frozen split did not reach both proven payload tensors')
        if self.dependency_signature(m)!=self.bound[id(m)]:raise RuntimeError('frozen prefix evolved during cache build')
        payload=(captured['residual'],captured['mlp']);size=sum(v.numel()*v.element_size() for v in payload)
        if size<=self.max_bytes:
            while self.entries and self.bytes+size>self.max_bytes:
                _,(_,removed,_) = self.entries.popitem(last=False);self.bytes-=removed;self.stats['evictions']+=1
            self.entries[key]=(payload,size,tuple(v._version for v in payload));self.bytes+=size;self.stats['peak_payload_bytes']=max(self.stats['peak_payload_bytes'],self.bytes)
        return payload
    def logits(self,m,tokens,mask=None,**kwargs):
        self.stats['logical_calls']+=1
        if kwargs:
            reason='custom position/cache kwargs outside original validated API'
            self.stats['fallbacks']+=1;self.stats['fallback_reasons'][reason]=self.stats['fallback_reasons'].get(reason,0)+1
            # Preserve the original public API's exact TypeError, not a new
            # candidate-specific exception category, before mutable updates.
            return self.original[id(m)](tokens,mask,**kwargs)
        try:
            if tokens.ndim!=2 or tokens.dtype!=torch.long or mask is not None and (mask.shape!=tokens.shape or mask.requires_grad):raise UnsupportedScope('input shape/dtype/mask/input-grad not validated')
            # Reset audit at the same logical boundary as the original logits.
            signature=self.dependency_signature(m)
        except UnsupportedScope as exc:
            reason=str(exc);self.stats['fallbacks']+=1;self.stats['fallback_reasons'][reason]=self.stats['fallback_reasons'].get(reason,0)+1
            self.invalidate(m);return self.original[id(m)](tokens,mask)
        charge(tokens)
        for bank in m.banks():bank.audit_history=[]
        residual,y=self.payload(m,tokens,mask,signature)
        for level in m.backbone.model.layers[-1].mlp.memory:y=level(y)
        hidden=m.backbone.model.norm(residual+y)
        return m.backbone.lm_head(hidden)

@contextmanager
def cached_front(model,cache):
    original=model.logits;cache.bind(model,original)
    def loaded(module,incompatible_keys):
        cache.invalidate(module);cache.bound[id(module)]=None;cache.stats['loads']+=1
    load_hook=model.register_load_state_dict_post_hook(loaded)
    model.logits=types.MethodType(lambda m,tokens,mask=None,**kwargs:cache.logits(m,tokens,mask,**kwargs),model)
    try:yield cache
    finally:
        model.logits=original;load_hook.remove();cache.clear()
        cache.original.pop(id(model),None);cache.bound.pop(id(model),None);cache.epoch.pop(id(model),None)

@contextmanager
def executor(session,name,cache=None):
    from contextlib import ExitStack
    if name not in ['REF','CACHE','HOST','BOTH']:raise ValueError(name)
    cache=cache or FrozenFrontLRU()
    with ExitStack() as stack:
        if name in ['CACHE','BOTH']:stack.enter_context(cached_front(session.student,cache))
        if name in ['HOST','BOTH']:stack.enter_context(host_reads(session))
        yield cache
