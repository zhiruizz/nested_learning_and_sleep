# Component probes

Full validation denominator: old/new = 256/256 (512 total), across seeds 17/29/43. Values are mean ± sample SD in percentage points; per-seed values follow.

| Case | Probe | Old EM mean ± SD | Old per seed (17/29/43) | New EM mean ± SD | New per seed (17/29/43) |
|---|---|---:|---|---:|---|
| warm_CMS_no_sleep@16 | actual | 52.08 ± 32.73 | 46.88%, 87.11%, 22.27% | 11.07 ± 19.17 | 0.00%, 0.00%, 33.20% |
| warm_CMS_no_sleep@16 | disable_fast_force_warm | 89.19 ± 2.15 | 90.23%, 86.72%, 90.62% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_CMS_no_sleep@16 | force_warm_receiver | 58.46 ± 28.91 | 58.98%, 87.11%, 29.30% | 10.94 ± 18.94 | 0.00%, 0.00%, 32.81% |
| warm_CMS_no_sleep@16 | mix_warm_receiver | 52.08 ± 32.73 | 46.88%, 87.11%, 22.27% | 11.07 ± 19.17 | 0.00%, 0.00%, 33.20% |
| warm_CMS_no_sleep@16 | restore_base0_force_warm | 79.95 ± 12.43 | 86.33%, 87.89%, 65.62% | 9.51 ± 16.46 | 0.00%, 0.00%, 28.52% |
| warm_CMS_no_sleep@16 | restore_base1_force_warm | 63.54 ± 27.05 | 58.98%, 92.58%, 39.06% | 11.20 ± 19.40 | 0.00%, 0.00%, 33.59% |
| warm_CMS_no_sleep@16 | restore_both_actual_route | 77.73 ± 13.88 | 69.14%, 93.75%, 70.31% | 7.03 ± 12.18 | 0.00%, 0.00%, 21.09% |
| warm_CMS_no_sleep@16 | restore_both_force_warm | 84.77 ± 10.10 | 86.72%, 93.75%, 73.83% | 6.64 ± 11.50 | 0.00%, 0.00%, 19.92% |
| warm_CMS_no_sleep@16 | warm_function_control | 91.93 ± 1.97 | 92.19%, 93.75%, 89.84% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_CMS_no_sleep@112 | actual | 1.04 ± 1.48 | 2.73%, 0.39%, 0.00% | 50.00 ± 8.12 | 54.69%, 40.62%, 54.69% |
| warm_CMS_no_sleep@112 | disable_fast_force_warm | 57.16 ± 49.56 | 91.80%, 0.39%, 79.30% | 13.67 ± 23.68 | 0.00%, 41.02%, 0.00% |
| warm_CMS_no_sleep@112 | force_warm_receiver | 18.62 ± 31.91 | 55.47%, 0.39%, 0.00% | 34.51 ± 23.84 | 8.20%, 40.62%, 54.69% |
| warm_CMS_no_sleep@112 | mix_warm_receiver | 1.04 ± 1.48 | 2.73%, 0.39%, 0.00% | 50.00 ± 8.12 | 54.69%, 40.62%, 54.69% |
| warm_CMS_no_sleep@112 | restore_base0_force_warm | 59.90 ± 50.86 | 89.06%, 89.45%, 1.17% | 16.41 ± 28.42 | 0.00%, 0.00%, 49.22% |
| warm_CMS_no_sleep@112 | restore_base1_force_warm | 20.44 ± 31.74 | 57.03%, 3.91%, 0.39% | 34.90 ± 23.80 | 7.42%, 48.05%, 49.22% |
| warm_CMS_no_sleep@112 | restore_both_actual_route | 37.76 ± 49.24 | 18.36%, 93.75%, 1.17% | 30.99 ± 26.98 | 43.75%, 0.00%, 49.22% |
| warm_CMS_no_sleep@112 | restore_both_force_warm | 61.46 ± 51.92 | 89.06%, 93.75%, 1.56% | 16.41 ± 28.42 | 0.00%, 0.00%, 49.22% |
| warm_CMS_no_sleep@112 | warm_function_control | 91.93 ± 1.97 | 92.19%, 93.75%, 89.84% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_post_sleep_P4@112 | actual | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% | 49.09 ± 1.26 | 50.00%, 49.61%, 47.66% |
| warm_post_sleep_P4@112 | disable_fast_force_warm | 55.86 ± 49.10 | 92.19%, 0.00%, 75.39% | 16.54 ± 28.64 | 0.00%, 49.61%, 0.00% |
| warm_post_sleep_P4@112 | force_warm_receiver | 55.86 ± 49.10 | 92.19%, 0.00%, 75.39% | 16.54 ± 28.64 | 0.00%, 49.61%, 0.00% |
| warm_post_sleep_P4@112 | mix_warm_receiver | 43.62 ± 46.29 | 92.19%, 0.00%, 38.67% | 29.95 ± 26.36 | 0.00%, 49.61%, 40.23% |
| warm_post_sleep_P4@112 | restore_base0_force_warm | 87.89 ± 5.47 | 91.80%, 81.64%, 90.23% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_post_sleep_P4@112 | restore_base1_force_warm | 63.93 ± 43.96 | 92.19%, 13.28%, 86.33% | 13.15 ± 22.78 | 0.00%, 39.45%, 0.00% |
| warm_post_sleep_P4@112 | restore_both_actual_route | 31.25 ± 54.13 | 0.00%, 93.75%, 0.00% | 31.25 ± 27.16 | 49.22%, 0.00%, 44.53% |
| warm_post_sleep_P4@112 | restore_both_force_warm | 92.45 ± 1.19 | 91.41%, 93.75%, 92.19% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_post_sleep_P4@112 | warm_function_control | 91.93 ± 1.97 | 92.19%, 93.75%, 89.84% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_pre_sleep_P4@16 | actual | 52.86 ± 38.93 | 47.66%, 94.14%, 16.80% | 10.03 ± 17.37 | 0.00%, 0.00%, 30.08% |
| warm_pre_sleep_P4@16 | disable_fast_force_warm | 64.06 ± 34.63 | 89.45%, 24.61%, 78.12% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_pre_sleep_P4@16 | force_warm_receiver | 64.06 ± 34.63 | 89.45%, 24.61%, 78.12% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_pre_sleep_P4@16 | mix_warm_receiver | 87.76 ± 5.88 | 88.28%, 93.36%, 81.64% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_pre_sleep_P4@16 | restore_base0_force_warm | 89.71 ± 1.48 | 91.41%, 89.06%, 88.67% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_pre_sleep_P4@16 | restore_base1_force_warm | 77.99 ± 20.19 | 89.45%, 54.69%, 89.84% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_pre_sleep_P4@16 | restore_both_actual_route | 62.89 ± 27.06 | 47.27%, 94.14%, 47.27% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_pre_sleep_P4@16 | restore_both_force_warm | 92.45 ± 1.19 | 91.41%, 93.75%, 92.19% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_pre_sleep_P4@16 | warm_function_control | 91.93 ± 1.97 | 92.19%, 93.75%, 89.84% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_pre_sleep_P4@112 | actual | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% | 32.68 ± 28.30 | 49.22%, 0.00%, 48.83% |
| warm_pre_sleep_P4@112 | disable_fast_force_warm | 45.05 ± 45.31 | 90.62%, 0.00%, 44.53% | 7.68 ± 13.31 | 0.00%, 0.00%, 23.05% |
| warm_pre_sleep_P4@112 | force_warm_receiver | 45.05 ± 45.31 | 90.62%, 0.00%, 44.53% | 7.68 ± 13.31 | 0.00%, 0.00%, 23.05% |
| warm_pre_sleep_P4@112 | mix_warm_receiver | 44.40 ± 45.97 | 91.80%, 0.00%, 41.41% | 14.32 ± 24.81 | 0.00%, 0.00%, 42.97% |
| warm_pre_sleep_P4@112 | restore_base0_force_warm | 88.54 ± 2.51 | 91.41%, 86.72%, 87.50% | 0.13 ± 0.23 | 0.00%, 0.39%, 0.00% |
| warm_pre_sleep_P4@112 | restore_base1_force_warm | 46.09 ± 45.15 | 90.23%, 0.00%, 48.05% | 0.52 ± 0.90 | 0.00%, 0.00%, 1.56% |
| warm_pre_sleep_P4@112 | restore_both_actual_route | 0.65 ± 1.13 | 0.00%, 0.00%, 1.95% | 32.81 ± 28.42 | 48.83%, 0.00%, 49.61% |
| warm_pre_sleep_P4@112 | restore_both_force_warm | 92.45 ± 1.19 | 91.41%, 93.75%, 92.19% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |
| warm_pre_sleep_P4@112 | warm_function_control | 91.93 ± 1.97 | 92.19%, 93.75%, 89.84% | 0.00 ± 0.00 | 0.00%, 0.00%, 0.00% |

All entries are fixed probes, not candidate selection. Route forcing changes both selected expert and effective amplitude; it is not a pure router-only intervention.
