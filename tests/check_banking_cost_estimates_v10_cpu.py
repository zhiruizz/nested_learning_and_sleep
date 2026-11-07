import pytest
from tools.banking_cost_estimates_v10 import frozen_backbone_matmul_estimate

CFG={'hidden_size':8,'vocab_size':16,'num_hidden_layers':2,'intermediate_size':12,'num_attention_heads':2,'num_key_value_heads':1}

def test_estimate_counts_executed_padding_and_no_backbone_backward():
    r=frozen_backbone_matmul_estimate({'forward_shape_histogram':{'2x3':4},'executed_input_tokens_including_padding':24},CFG)
    dense=2*(2*8*8+2*8*1*4+3*8*12)+16*8
    assert r['partial_forward_flops_estimate']==2*dense*24+4*2*8*(2*3*3*4)
    assert r['measured_complete_flops']is None and r['full_backbone_backward_passes']==0

def test_inconsistent_measured_shape_totals_rejected():
    with pytest.raises(ValueError,match='disagree'):
        frozen_backbone_matmul_estimate({'forward_shape_histogram':{'2x3':4},'executed_input_tokens_including_padding':23},CFG)
