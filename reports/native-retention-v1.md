# Native symbol retention pilot v1

Completed bounded synthetic CPU experiment. This is not a paper benchmark or a direct replication of upstream HOPE scores.

Protocol: [native-retention-protocol.md](../docs/native-retention-protocol.md). Recompute with `python tools/summarize_native_retention.py --results results/native-retention-v1`.

Three fixed seeds; old A and one-token copy foundation, then new B adaptation. Here SFT denotes supervised per-batch updates on the same restricted parameter scope as CMS, not full-model fine-tuning. SFT/CMS share starting weights and batches. Candidate chance is 25%. All tasks have 512 held-out synthetic prompts per seed.

| Seed | State | A accuracy | Copy accuracy | B accuracy | A NLL | Copy NLL | B NLL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 17 | random | 0.00% | 0.00% | 0.00% | 3.740722 | 3.948116 | 3.757083 |
| 17 | foundation | 100.00% | 100.00% | 0.00% | 0.001967 | 0.002474 | 7.897772 |
| 17 | frozen | 100.00% | 100.00% | 0.00% | 0.001967 | 0.002474 | 7.897772 |
| 17 | sft | 0.00% | 25.00% | 99.80% | 2.520335 | 1.610567 | 0.177842 |
| 17 | cms | 0.00% | 25.00% | 100.00% | 2.542198 | 1.641414 | 0.186811 |
| 29 | random | 25.00% | 1.17% | 0.00% | 3.511553 | 3.884703 | 3.503753 |
| 29 | foundation | 100.00% | 100.00% | 0.20% | 0.002214 | 0.002703 | 7.645006 |
| 29 | frozen | 100.00% | 100.00% | 0.20% | 0.002214 | 0.002703 | 7.645006 |
| 29 | sft | 0.00% | 2.73% | 99.61% | 2.502562 | 1.880688 | 0.153937 |
| 29 | cms | 0.00% | 2.54% | 99.61% | 2.535492 | 1.916947 | 0.152074 |
| 43 | random | 0.00% | 4.49% | 25.00% | 4.419639 | 3.888232 | 3.645315 |
| 43 | foundation | 100.00% | 100.00% | 0.00% | 0.002005 | 0.002369 | 7.776361 |
| 43 | frozen | 100.00% | 100.00% | 0.00% | 0.002005 | 0.002369 | 7.776361 |
| 43 | sft | 0.00% | 6.64% | 100.00% | 3.035338 | 2.390056 | 0.083086 |
| 43 | cms | 0.00% | 5.08% | 100.00% | 3.077807 | 2.433019 | 0.082294 |

## Acquisition and retention

| Seed | Old A foundation >=80% | Copy foundation >=80% |
|---:|---|---|
| 17 | YES | YES |
| 29 | YES | YES |
| 43 | YES | YES |

Mean endpoints (descriptive, all three seeds):

| Arm | A accuracy | Copy accuracy | B accuracy | A change vs foundation (pp) | B change vs foundation (pp) |
|---|---:|---:|---:|---:|---:|
| frozen | 100.00% | 100.00% | 0.07% | +0.00 | +0.00 |
| sft | 0.00% | 11.46% | 99.80% | -100.00 | +99.74 |
| cms | 0.00% | 10.87% | 99.87% | -100.00 | +99.80 |

All three seeds acquired both old A and copy at100%; their subsequent accuracy loss is forgetting of acquired tasks in this experiment. SFT/CMS differences are within this budget and schedule; three seeds do not support a general superiority claim.

## Validation and cost

- Stream/full-forward fixture error: 0.0; period-one CMS matches SFT exactly.
- All adaptation starts and B batch hashes match; mutable scopes match; no frozen parameter changed (runner asserts).
- Adaptation parameter scope: 1600 / 7542 parameters. SFT base writes128/128, CMS32/8; both have128 online expert updates. Pending counts end at0/0.
- Main attempts/completed: 2304/2304; correctness fixture reserved/completed: 2/2.
- Elapsed runner time: 64.610s on CPU, one thread. Main supervised answer tokens: 73728; fixture: 16.
- Frozen metrics reuse the identical foundation state; no duplicate model forward is charged.

| Stage | Forward calls | Input tokens | Gradient-enabled input tokens |
|---|---:|---:|---:|
| adapt_cms | 384 | 49152 | 49152 |
| adapt_sft | 384 | 49152 | 49152 |
| correctness | 3 | 96 | 64 |
| evaluation | 432 | 110592 | 0 |
| foundation | 1536 | 196608 | 196608 |

The stream fixture additionally executes32 core input tokens outside WorkMeter, without gradients. Token counts are compute proxies, not FLOPs or GPU busy time. Baseline/evaluation and foundation are included in the ledger. No GPU job, model download or external dataset was used.

## Limits

One tiny native variant; synthetic nuisance split; short copy can use convolution history. The B domain marker makes old/new mappings jointly representable in principle, but the restricted adaptation scope need not access such a solution. No Sleep/GKD/Dreaming arm, long-context test, natural-language perplexity, factual benchmark or pretrained 3B result is established. Parameter drift locates changes; it does not identify causal learning mechanisms. All seeds, intermediate evaluation points32/128 and negative results remain in metrics.json. No weights or per-example dataset records are distributed.

Published JSON uses UTF-8/LF line endings; numerical values are unchanged from the Windows runner output retained locally. config_sha256 in provenance identifies the preregistered configs/native-retention-v1.json source, not the pretty-printed result copy. Source commit identifies the run code before outcomes; this report and aggregate archive were added afterwards.
