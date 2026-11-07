from tools.estimate_clinc_v8_work import estimate_arm, EXCLUDED

DIMS={"H":4,"I":8,"L":1,"heads":2,"kvheads":1,"Dh":2,"V":10}
CFG={"rank":1,"levels":2,"head_lora_rank":2}


def test_hand_counted_capture_tail_and_frozen_head_backward():
    work={"forward_input_shape_histogram":{"1x2":1},
          "tail_shape_histogram":{"1x2":1},"tail_grad_shape_histogram":{"1x2":1}}
    out=estimate_arm(work,DIMS,CFG)
    # Transformer projection/MLP=576; dense QK/AV attention=64.
    assert out["capture_transformer_gemm_attention"] == 640
    # Original head=160; rank2 readout=112; two rank1 memories=64.
    assert out["capture_original_head_and_readout_lora"] == 272
    assert out["capture_zero_output_memory_matrices"] == 64
    assert out["frozen_backbone_forward"] == 976
    assert out["tail_forward_head_and_memory"] == 336
    assert out["tail_forward_expert_branches_interval"] == [0,64]
    assert out["tail_backward_frozen_head_input_grad"] == 272
    assert out["tail_backward_mutable_interval"] == [0,256]
    assert out["covered_total_interval"] == [1584,1904]


def test_padded_shapes_count_quadratic_attention_and_no_frozen_backbone_backward():
    def work(shape):
        return {"forward_input_shape_histogram":{shape:1},
                "tail_shape_histogram":{},"tail_grad_shape_histogram":{}}
    short=estimate_arm(work("2x2"),DIMS,CFG)
    padded=estimate_arm(work("2x4"),DIMS,CFG)
    assert short["frozen_backbone_forward"] == 1952
    assert padded["frozen_backbone_forward"] == 4160
    assert padded["frozen_backbone_forward"] == 2*short["frozen_backbone_forward"]+256
    assert padded["tail_backward_frozen_head_input_grad"] == 0
    assert padded["covered_total_interval"] == [4160,4160]
    assert "softmax" in EXCLUDED and "router" in EXCLUDED


def test_controlled_mix_bounds_extra_branch_even_if_it_names_natural_winner():
    work={"forward_input_shape_histogram":{"1x2":1},
          "tail_shape_histogram":{"1x2":1},"tail_grad_shape_histogram":{"1x2":1}}
    out=estimate_arm(work,DIMS,{**CFG,"expert_branches_per_level_upper_bound":2})
    assert out["tail_forward_expert_branches_interval"] == [0,128]
    assert out["tail_backward_mutable_interval"] == [0,384]
    assert out["covered_total_interval"] == [1584,2096]


def test_actual_arm_infers_controlled_cost_without_mutating_frozen_configuration():
    work={"forward_input_shape_histogram":{"1x2":1},
          "tail_shape_histogram":{"1x2":1},"tail_grad_shape_histogram":{"1x2":1}}
    cfg={**CFG,"target_arms":["SLEEP_MIX","GOLD_CE_MIX"],"arms":["HOPE"]}
    assert estimate_arm(work,DIMS,cfg)["expert_branches_per_level_upper_bound"] == 2
    assert estimate_arm(work,DIMS,cfg,arm="HOPE")["expert_branches_per_level_upper_bound"] == 1
    assert estimate_arm(work,DIMS,cfg,arm="HOPE_SLEEP_FIXEDMIX")["covered_total_interval"] == [1584,2096]
    assert "expert_branches_per_level_upper_bound" not in cfg
