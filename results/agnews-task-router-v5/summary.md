# Two-branch routing/capacity control

Exploratory n=3; fixed train-only ridge. Additional group labels and retained adapter state; not Sleep consolidation. Both branches were evaluated for every article. Task-ID selection uses evaluation group identity only as a diagnostic and is not a guaranteed upper bound.

| Condition | Old EM % mean ± SD | New EM % mean ± SD | All EM % mean ± SD | Both gates |
|---|---:|---:|---:|---:|
| warm_alone | 91.93 ± 1.97 | 0.00 ± 0.00 | 45.96 ± 0.98 | 0/3 |
| new_CMS_reference_alone | 0.00 ± 0.00 | 73.57 ± 1.13 | 36.78 ± 0.56 | 0/3 |
| learned_feature_gate | 88.67 ± 1.17 | 71.48 ± 1.03 | 80.08 ± 1.03 | 0/3 |
| task_ID_diagnostic | 91.93 ± 1.97 | 73.57 ± 1.13 | 82.75 ± 1.52 | 0/3 |

1,152 full-model batch forwards; zero LLM optimizer updates; three CPU ridge fits; wall seconds 80.329. 1,024 unique training group labels, reused in three fits (3,072 label uses).

| Seed | Learned gate validation group accuracy | Old→old | Old→new | New→old | New→new |
|---|---:|---:|---:|---:|---:|
| 17 | 96.29% | 244 | 12 | 7 | 249 |
| 29 | 96.29% | 244 | 12 | 7 | 249 |
| 43 | 96.29% | 244 | 12 | 7 | 249 |
