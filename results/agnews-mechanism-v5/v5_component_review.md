# V5 component diagnostic review

Read-only review of the sealed development diagnostic. No model was loaded and no training or optimizer updates were performed.

Result SHA-256: abc46d14b3152f3016481eb6fb719610db61aee6657807862889ae195ecb39d5. Complete=True; development_only=True; confirmation_scored=False; 135 intervention states + 3 warm references; optimizer updates=0; all 135 old-receiver frozen guards pass; 15 warm-function controls match every warm reference metric group (EM exact, answer NLL < 1e-5). Each state uses 64 scoring forwards / 47,864 input tokens; gradient-input tokens are zero.

## Exact-match means across seeds 17, 29, 43

Values are mean (sample SD), n=3; old/new are full-vocabulary, two-token exact match.

| Checkpoint | Probe | Old EM | New EM |
|---|---|---:|---:|
| warm_CMS_no_sleep@16 | actual | 0.520833 (0.327341) | 0.110677 (0.191698) |
| warm_CMS_no_sleep@16 | mix_warm_receiver | 0.520833 (0.327341) | 0.110677 (0.191698) |
| warm_CMS_no_sleep@16 | force_warm_receiver | 0.584635 (0.289098) | 0.109375 (0.189443) |
| warm_CMS_no_sleep@16 | disable_fast_force_warm | 0.891927 (0.021514) | 0 (0) |
| warm_CMS_no_sleep@16 | restore_base0_force_warm | 0.799479 (0.124286) | 0.095052 (0.164635) |
| warm_CMS_no_sleep@16 | restore_base1_force_warm | 0.635417 (0.270473) | 0.111979 (0.193954) |
| warm_CMS_no_sleep@16 | restore_both_force_warm | 0.847656 (0.101035) | 0.066406 (0.115019) |
| warm_CMS_no_sleep@16 | restore_both_actual_route | 0.777344 (0.138823) | 0.070312 (0.121785) |
| warm_CMS_no_sleep@16 | warm_function_control | 0.919271 (0.019661) | 0 (0) |
| warm_CMS_no_sleep@112 | actual | 0.010417 (0.014789) | 0.5 (0.08119) |
| warm_CMS_no_sleep@112 | mix_warm_receiver | 0.010417 (0.014789) | 0.5 (0.08119) |
| warm_CMS_no_sleep@112 | force_warm_receiver | 0.186198 (0.319127) | 0.345052 (0.238388) |
| warm_CMS_no_sleep@112 | disable_fast_force_warm | 0.571615 (0.495607) | 0.136719 (0.236804) |
| warm_CMS_no_sleep@112 | restore_base0_force_warm | 0.598958 (0.508568) | 0.164062 (0.284165) |
| warm_CMS_no_sleep@112 | restore_base1_force_warm | 0.204427 (0.317353) | 0.348958 (0.238004) |
| warm_CMS_no_sleep@112 | restore_both_force_warm | 0.614583 (0.519242) | 0.164062 (0.284165) |
| warm_CMS_no_sleep@112 | restore_both_actual_route | 0.377604 (0.492441) | 0.309896 (0.269767) |
| warm_CMS_no_sleep@112 | warm_function_control | 0.919271 (0.019661) | 0 (0) |
| warm_pre_sleep_P4@16 | actual | 0.528646 (0.38934) | 0.10026 (0.173656) |
| warm_pre_sleep_P4@16 | mix_warm_receiver | 0.877604 (0.058767) | 0 (0) |
| warm_pre_sleep_P4@16 | force_warm_receiver | 0.640625 (0.346337) | 0 (0) |
| warm_pre_sleep_P4@16 | disable_fast_force_warm | 0.640625 (0.346337) | 0 (0) |
| warm_pre_sleep_P4@16 | restore_base0_force_warm | 0.897135 (0.014789) | 0 (0) |
| warm_pre_sleep_P4@16 | restore_base1_force_warm | 0.779948 (0.201857) | 0 (0) |
| warm_pre_sleep_P4@16 | restore_both_force_warm | 0.924479 (0.011934) | 0 (0) |
| warm_pre_sleep_P4@16 | restore_both_actual_route | 0.628906 (0.270633) | 0 (0) |
| warm_pre_sleep_P4@16 | warm_function_control | 0.919271 (0.019661) | 0 (0) |
| warm_pre_sleep_P4@112 | actual | 0 (0) | 0.326823 (0.283044) |
| warm_pre_sleep_P4@112 | mix_warm_receiver | 0.44401 (0.459717) | 0.143229 (0.24808) |
| warm_pre_sleep_P4@112 | force_warm_receiver | 0.450521 (0.453147) | 0.076823 (0.133061) |
| warm_pre_sleep_P4@112 | disable_fast_force_warm | 0.450521 (0.453147) | 0.076823 (0.133061) |
| warm_pre_sleep_P4@112 | restore_base0_force_warm | 0.885417 (0.025114) | 0.001302 (0.002255) |
| warm_pre_sleep_P4@112 | restore_base1_force_warm | 0.460938 (0.451489) | 0.005208 (0.009021) |
| warm_pre_sleep_P4@112 | restore_both_force_warm | 0.924479 (0.011934) | 0 (0) |
| warm_pre_sleep_P4@112 | restore_both_actual_route | 0.00651 (0.011276) | 0.328125 (0.284191) |
| warm_pre_sleep_P4@112 | warm_function_control | 0.919271 (0.019661) | 0 (0) |
| warm_post_sleep_P4@112 | actual | 0 (0) | 0.490885 (0.012557) |
| warm_post_sleep_P4@112 | mix_warm_receiver | 0.436198 (0.462925) | 0.299479 (0.263559) |
| warm_post_sleep_P4@112 | force_warm_receiver | 0.558594 (0.490992) | 0.165365 (0.28642) |
| warm_post_sleep_P4@112 | disable_fast_force_warm | 0.558594 (0.490992) | 0.165365 (0.28642) |
| warm_post_sleep_P4@112 | restore_base0_force_warm | 0.878906 (0.054688) | 0 (0) |
| warm_post_sleep_P4@112 | restore_base1_force_warm | 0.639323 (0.439628) | 0.13151 (0.227783) |
| warm_post_sleep_P4@112 | restore_both_force_warm | 0.924479 (0.011934) | 0 (0) |
| warm_post_sleep_P4@112 | restore_both_actual_route | 0.3125 (0.541266) | 0.3125 (0.271646) |
| warm_post_sleep_P4@112 | warm_function_control | 0.919271 (0.019661) | 0 (0) |

## Findings

The exact `warm_function_control` positive control matches the warm checkpoint metrics across all five checkpoint cases and all three seeds: mean old EM is 0.9193 (sample SD 0.0197; seed values 0.921875, 0.9375, 0.8984375) and new EM is 0. The separate `restore_both_force_warm` intervention at the two P4@112 checkpoints yields old EM 0.9245 (SD 0.0119), also with new EM 0; this is a distinct forced intervention, not the warm-function positive control. Neither result establishes joint old/new performance.

Under actual learned routing at P4@112, old EM is 0 in both pre- and post-teacher-order arms. New EM is 0.3268 (SD 0.2830) pre and 0.4909 (SD 0.0126) post. Restoring both bases but retaining actual routing gives old EM 0.0065 pre and 0.3125 post, so restoring weights alone is insufficient.

The pattern supports joint dependence on base context and routing/expert selection, but these are zero-training counterfactuals and do not identify a unique training-time cause. Single-base restorations and receiver controls are seed/checkpoint dependent. `force_warm_receiver` and `force_*` modify gate/null amplitudes; treat them as interventions, not natural policy behavior. Natural route telemetry at P4 checkpoints selects latest slot 4 at step 16 and slot 28 at step 112; warm receiver gates are near zero at step 112.

The planned 2×3 KD/no-KD × noSleep/P16-latest/P16-warmmix comparison plus matched gold-joint-CE is a sensible first factorial. If the study intends to attribute failure specifically to updated CMS bases, one matched base-write-freeze arm (or separate base0/base1 freezes) is the important missing control; otherwise that mechanism should remain a follow-up rather than be inferred from functional restoration alone.

No method-win or deployment claim is supported by this diagnostic. It is development-only and used no optimizer updates.

The machine-readable record includes the 45-cell mean/SD table and natural-route telemetry.
