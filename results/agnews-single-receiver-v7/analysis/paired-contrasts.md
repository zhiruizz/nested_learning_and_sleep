# Per-seed paired contrasts

Signed endpoint deltas (right minus left); raw seed values only, no p-values or best-seed selection.

| Contrast | Scope | Seed | Old EM Δ | New EM Δ | All EM Δ | Old NLL Δ | New NLL Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| split_receiver minus current_receiver | receiver | 17 | +0.9258 | -0.1172 | +0.4043 | -5.2615 | +0.2489 |
| split_receiver minus current_receiver | receiver | 29 | +0.8789 | -0.0117 | +0.4336 | -3.5476 | +0.0221 |
| split_receiver minus current_receiver | receiver | 43 | +0.7109 | +0.0156 | +0.3633 | -3.7230 | -0.0481 |
| split_shared minus current_shared | shared | 17 | +0.9258 | -0.1602 | +0.3828 | -5.6968 | +0.3617 |
| split_shared minus current_shared | shared | 29 | +0.8477 | -0.0469 | +0.4004 | -3.6562 | +0.1476 |
| split_shared minus current_shared | shared | 43 | +0.8867 | +0.0039 | +0.4453 | -4.3832 | +0.0386 |
| split_receiver minus split_narrow_receiver | receiver | 17 | +0.2852 | -0.0898 | +0.0977 | -1.2072 | +0.1936 |
| split_receiver minus split_narrow_receiver | receiver | 29 | +0.2930 | -0.0156 | +0.1387 | -0.7206 | +0.0168 |
| split_receiver minus split_narrow_receiver | receiver | 43 | +0.0625 | +0.0039 | +0.0332 | -0.5085 | -0.0360 |
| split_shared minus split_narrow_shared | shared | 17 | +0.2227 | -0.1289 | +0.0469 | -1.0200 | +0.2288 |
| split_shared minus split_narrow_shared | shared | 29 | +0.1953 | +0.0352 | +0.1152 | -0.6400 | -0.0152 |
| split_shared minus split_narrow_shared | shared | 43 | +0.2734 | -0.0156 | +0.1289 | -0.7999 | +0.0187 |
| split_receiver minus gold_receiver | receiver | 17 | +0.0391 | -0.1250 | -0.0430 | -0.1864 | +0.1112 |
| split_receiver minus gold_receiver | receiver | 29 | -0.0273 | -0.1289 | -0.0781 | +0.0432 | +0.1564 |
| split_receiver minus gold_receiver | receiver | 43 | -0.2188 | +0.0195 | -0.0996 | +0.6684 | -0.1528 |
| split_shared minus gold_shared | shared | 17 | -0.0039 | -0.0508 | -0.0273 | -0.1129 | -0.4048 |
| split_shared minus gold_shared | shared | 29 | -0.0156 | -0.1523 | -0.0840 | -0.0728 | +0.1692 |
| split_shared minus gold_shared | shared | 43 | +0.1484 | -0.0078 | +0.0703 | -0.4235 | -0.0809 |
| gold_receiver minus new_only_receiver | receiver | 17 | +0.8867 | -0.0156 | +0.4355 | -6.6415 | -0.3127 |
| gold_receiver minus new_only_receiver | receiver | 29 | +0.9062 | -0.0742 | +0.4160 | -6.5078 | -0.1308 |
| gold_receiver minus new_only_receiver | receiver | 43 | +0.9297 | -0.0234 | +0.4531 | -6.9593 | +0.0271 |
| gold_shared minus new_only_shared | shared | 17 | +0.9297 | -0.1133 | +0.4082 | -8.4287 | +0.5049 |
| gold_shared minus new_only_shared | shared | 29 | +0.8633 | +0.0078 | +0.4355 | -7.5132 | -0.6061 |
| gold_shared minus new_only_shared | shared | 43 | +0.7383 | -0.0078 | +0.3652 | -7.7022 | -0.2749 |
