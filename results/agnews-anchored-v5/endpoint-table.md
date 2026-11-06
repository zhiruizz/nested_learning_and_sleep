# Endpoint metrics at step 448

EM and accuracy are percentages; NLL is nats per answer token. Entries are mean ± sample SD over the three paired seeds. Gates retain all seed outcomes.

| Arm | Old EM | New EM | All EM | Old NLL | New NLL | Old pass | New pass | Both pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CMS reference | 0.0 ± 0.0 | 73.6 ± 1.1 | 36.8 ± 0.6 | 6.213 ± 2.627 | 0.334 ± 0.041 | 0/3 | 0/3 | 0/3 |
| Anchor KD | 75.1 ± 8.4 | 56.2 ± 4.1 | 65.7 ± 3.0 | 0.507 ± 0.257 | 0.490 ± 0.083 | 1/3 | 0/3 | 0/3 |
| Anchor Gold | 78.5 ± 8.8 | 46.5 ± 10.2 | 62.5 ± 3.7 | 0.384 ± 0.131 | 0.587 ± 0.161 | 1/3 | 0/3 | 0/3 |
| KD Sleep latest | 52.7 ± 19.8 | 33.6 ± 16.3 | 43.2 ± 5.7 | 0.743 ± 0.245 | 0.899 ± 0.194 | 0/3 | 0/3 | 0/3 |
| KD Sleep warmmix | 68.2 ± 13.5 | 39.5 ± 2.3 | 53.8 ± 6.5 | 0.508 ± 0.145 | 0.682 ± 0.135 | 0/3 | 0/3 | 0/3 |
| Self Sleep latest | 0.0 ± 0.0 | 63.2 ± 13.7 | 31.6 ± 6.9 | 4.439 ± 0.824 | 1.015 ± 0.990 | 0/3 | 0/3 | 0/3 |
| Self Sleep warmmix | 0.0 ± 0.0 | 66.7 ± 13.8 | 33.3 ± 6.9 | 5.155 ± 1.398 | 0.400 ± 0.155 | 0/3 | 0/3 | 0/3 |
