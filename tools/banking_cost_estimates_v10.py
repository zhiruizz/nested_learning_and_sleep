"""Offline partial algorithmic FLOP estimates from recorded forward shapes.

These are not measured complete FLOPs. Tail re-evaluation, backward, adapters,
normalization, nonlinearities, sampling and IO are deliberately outside this
partial count and remain covered by the separate measured work/time ledger.
"""
from __future__ import annotations

def frozen_backbone_matmul_estimate(work,model_config):
    h=int(model_config['hidden_size']);v=int(model_config['vocab_size'])
    layers=int(model_config['num_hidden_layers']);inter=int(model_config['intermediate_size'])
    heads=int(model_config['num_attention_heads']);kv=int(model_config['num_key_value_heads'])
    if min(h,v,layers,inter,heads,kv)<1 or h%heads:raise ValueError('Invalid model shape')
    # Q/O, K/V, three SwiGLU projections; output vocabulary projection is dense.
    head_dim=h//heads
    dense_weights=layers*(2*h*h+2*h*kv*head_dim+3*h*inter)+v*h
    hist=work.get('forward_input_shape_histogram',work.get('forward_shape_histogram'))
    if not isinstance(hist,dict)or not hist:raise ValueError('Recorded forward shape histogram required')
    positions=attention_pairs=calls=0
    for shape,count in hist.items():
        b,s=map(int,shape.split('x'))
        if min(b,s)<1 or type(count)is not int or count<1:raise ValueError('Invalid recorded shape/count')
        calls+=count;positions+=b*s*count;attention_pairs+=b*s*s*count
    expected=work.get('executed_input_tokens_including_padding',work.get('executed_input_tokens'))
    if expected is not None and expected!=positions:raise ValueError('Recorded shape/token totals disagree')
    dense=2*dense_weights*positions
    attention=4*layers*h*attention_pairs
    return {'schema':'banking-partial-matmul-flops-estimate-v10','measured_complete_flops':None,
      'method':'2*dense_projection_weights*executed_token_positions + 4*layers*hidden*sum(batch*sequence_length^2)',
      'dense_projection_weights':dense_weights,'recorded_full_backbone_forward_calls':calls,
      'executed_token_positions':positions,'dense_projection_flops_estimate':dense,
      'attention_qk_av_flops_estimate':attention,'partial_forward_flops_estimate':dense+attention,
      'includes_padding':True,'full_backbone_backward_passes':0,
      'scope':'Frozen backbone dense projections and attention QK/AV only; excludes mutable tail re-evaluation/backward, nonlinear ops, sampling and IO. Not total experiment FLOPs.'}
