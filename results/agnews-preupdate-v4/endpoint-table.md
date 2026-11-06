# AGNews pre-update v4: fixed endpoint table

Development/validation only. Fixed step 112; mean and sample SD across three seeds. No p-values or checkpoint selection.

| Arm | Old EM | New EM | All EM | Old drop from own start | Old NLL | New NLL | All NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| foundation_CMS_reference | 0.26% ± 0.23 pp | 52.60% ± 9.26 pp | 26.43% ± 4.52 pp | 92.19% ± 1.70 pp | 3.4019 ± 0.1864 | 0.4932 ± 0.1327 | 1.9475 ± 0.0383 |
| warm_CMS_no_sleep | 1.04% ± 1.48 pp | 50.00% ± 8.12 pp | 25.52% ± 4.39 pp | 90.89% ± 2.15 pp | 4.1317 ± 1.1536 | 0.4679 ± 0.0412 | 2.2998 ± 0.5863 |
| warm_pre_sleep_P64 | 0.13% ± 0.23 pp | 44.79% ± 2.15 pp | 22.46% ± 1.01 pp | 91.80% ± 1.79 pp | 4.4744 ± 0.9231 | 0.4704 ± 0.0233 | 2.4724 ± 0.4550 |
| warm_pre_sleep_P4 | 0.00% ± 0.00 pp | 32.68% ± 28.30 pp | 16.34% ± 14.15 pp | 91.93% ± 1.97 pp | 15.7108 ± 15.9565 | 17.7934 ± 29.5193 | 16.7521 ± 22.7346 |
| warm_post_sleep_P4 | 0.00% ± 0.00 pp | 49.09% ± 1.26 pp | 24.54% ± 0.63 pp | 91.93% ± 1.97 pp | 7.6793 ± 3.0702 | 1.3863 ± 0.5376 | 4.5328 ± 1.7448 |
| warm_extra_gold_replay_P4 | 77.73% ± 12.59 pp | 40.76% ± 10.61 pp | 59.24% ± 11.59 pp | 14.19% ± 14.10 pp | 0.3551 ± 0.1765 | 0.6864 ± 0.1468 | 0.5207 ± 0.1456 |
