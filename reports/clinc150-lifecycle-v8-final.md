# CLINC-150 lifecycle V8: final results

**Status:** all six training runs, clean-development evaluation, and the sealed final-test evaluation are complete. The final-test aggregate is bound to the six-run seal and frozen evaluation contract.

## Main finding

On the ten-task clean-development trajectory, the `HOPE_SLEEP_FIXEDMIX` local ablation performed substantially worse than the paired `HOPE` arm on current-task and all-seen accuracy. At task 9, current-task accuracy was **24.69% ± 5.48%** for fixed-mix and **85.53% ± 6.83%** for HOPE; all-seen accuracy was **11.11% ± 1.08%** and **17.67% ± 1.27%**, respectively. Earlier-class accuracy was similarly low in both arms (9.60% ± 1.18% fixed-mix; 10.13% ± 0.84% HOPE), but that metric is not a pure forgetting measure because only task 0 had a learning gate.

These are descriptive means and sample SDs across three paired **class-order seeds**. All seeds share the same Llama initialization seed, so they are not three independent model initializations. The results do not support a positive Sleep benefit in this local configuration.

| Arm | Earlier classes, task 9 | Current task, task 9 | All seen, task 9 |
|---|---:|---:|---:|
| HOPE | 10.13% ± 0.84% | 85.53% ± 6.83% | 17.67% ± 1.27% |
| HOPE_SLEEP_FIXEDMIX | 9.60% ± 1.18% | 24.69% ± 5.48% | 11.11% ± 1.08% |
| Paired fixed-mix minus HOPE | −0.53 ± 1.61 pp | −60.84 ± 9.22 pp | −6.56 ± 2.32 pp |

Per-seed task-9 accuracies (HOPE → fixed-mix) were: earlier classes **9.23→9.42%, 10.90→8.53%, 10.27→10.86%**; current task **84.00→31.00%, 93.00→22.00%, 79.60→21.07%**; all seen **16.72→11.58%, 19.12→9.88%, 17.18→11.88%**. The large current-task and all-seen gaps have the same direction across all three class orders. The earlier-class differences do not.

Invalid free-generation outputs were frequent, particularly for fixed-mix. Across the three seeds at task 9, invalid predictions were 30.8–57.9% of HOPE's earlier-class examples and 71.2–76.6% of fixed-mix's earlier-class examples. On current-task examples the ranges were 4.3–10.0% for HOPE and 60.5–69.0% for fixed-mix. Invalid outputs count as incorrect; generation was not restricted to candidate labels.

## Protocol and scope

The frozen experiment used Llama 3.2 3B, revision `48d84dd14886f01964f72e7f742a404dd86f9fb6`, with a local CMS last-MLP augmentation: rank 64, five levels, capacity 8, and periods `[16, 64, 256, 1024, None]`. The dataset pack contains 15,000 training rows and 3,000 development rows. Three normalized train/development duplicates were excluded, leaving 2,997 clean development rows. There are 150 intents in ten class-incremental tasks of 15 intents each; each task used 1,500 training presentations, batch size 8, and 188 wake updates, including the final short batch.

The paired grid was seeds 17, 29, and 43 by arms `HOPE` and `HOPE_SLEEP_FIXEDMIX`. Each seed's task-0 foundation was trained once, passed its predeclared 0.5 clean-development acquisition gate, and was shared by both arms. Task 0 mean accuracy was 61.78% ± 9.32% across class orders (per seed: 72.33%, 54.67%, 58.33%). Tasks 1–9 had **no acquisition threshold**. Therefore the task-9 “earlier classes” score mixes forgetting with any earlier classes that might not have been learned at their own task boundary; it should not be called a pure retention rate. Current-task development accuracy helps interpret the trajectory but does not establish that every earlier class was acquired.

Training used one pass through each task's scheduled rows. The 1,692 continuation wake updates follow the shared 188-step foundation, for 1,880 global wake ticks per run. HOPE had no Sleep events. Fixed-mix had 141 events and 4,230 physical Sleep AdamW updates per run (20 GKD and 10 LTI optimizer updates per event); across its three runs that is 423 events and 12,690 Sleep updates. Its work counter was 5,922 per run: 1,692 continuation wake updates plus 4,230 Sleep updates. The fixed-mix pool replayed four class-balanced examples per class from the training pack. The run receipt marks Sleep replay availability for both arms, but actual exposure is identified from event and optimizer counters; HOPE had zero Sleep exposure.

`HOPE_SLEEP_FIXEDMIX` is a controlled 0.5 branch-mix **local ablation**, not native or paper-faithful HOPE Sleep. It adds replay and substantial optimization compute, so this comparison does not isolate a single Sleep mechanism and is not compute-matched. Both arms share the foundation and primary new-task wake schedule, but only fixed-mix performs the extra Sleep work.

## Alignment with the Sleep paper

The Sleep v2 paper reports class-incremental experiments on CLINC, Banking/Banking77, and DBpedia using Llama-3B and Llama-3-8B. This report covers only CLINC with a locally pinned Llama 3.2 3B checkpoint; the paper does not identify this exact checkpoint, so model identity is not an exact replication. The paper describes five MLP blocks with dimension 64 and reports fixed training settings for its own runs. Our last-MLP CMS augmentation, rank/capacity configuration, teacher/student path, local fixed-mix Sleep transition, and canonical semantic-agreement/edit proxy differ from that full method. This is a bounded local mechanism study, not a paper reproduction or a comparison to the paper's reported scores. See the [Sleep v2 paper](https://arxiv.org/abs/2606.03979).

Our data pack has 15,000 training and 3,000 development rows (2,997 clean development rows after the frozen overlap rule). It is not the paper's reported 10,000/750 InCA allocation, and the paper does not establish that these exact data files and split conventions match ours. Only CLINC with the 3B model was run; Banking/DBpedia with 8B were not run. The periods `[16, 64, 256, 1024, None]` are this protocol's choices, not copies of the paper's `[512, 2k, 8k]` long-context memory ablation; the unit/interpretation of those paper values is not established here, and they are not validated classification optima. We consulted the [upstream snapshot at commit `599244e`](https://github.com/obekt/HOPE-nested-learning/commit/599244e) as a reference; it had no LICENSE file, and this project implemented its method ideas independently rather than copying its code. The upstream results are not quantitatively comparable to this local experiment.

The natural-route Sleep diagnostic was kept separate and failed closed before the main comparison: at wake tick 64, the receiver's scalar objective was finite but had no gradient, and none of the eight answer positions selected the receiver. The existing gradient guard stopped the event; this failure was not silently relabeled as the fixed-mix arm. That attempt used 64 wake ticks, 9 completed Sleep AdamW updates, 222 full forwards, and 23.75 seconds. Earlier corrected k=1 and k=10 development transfer probes were negative for Sleep; the frozen protocol selected k=10 to expand transfer coverage, not to claim improvement.

## Compute and storage

The shared three foundations used 89.6 active seconds in total and are counted once. Continuation active time summed to **1,761.8 seconds** for HOPE and **5,231.0 seconds** for fixed-mix (about 2.97×); full-backbone forward calls were 35,871 and 102,777. These are measured workload counters and elapsed active time, not a speed comparison under matched compute.

A separate source-bound resource audit reported 6.43 GB of model-cache files and 1.356 GB across the nine audited endpoint checkpoints. A serialized mutable endpoint context contained 17,957,376 parameters plus 123,000 buffers (72,075,704 bytes, about 68.74 MiB); its trainable endpoint inventory was 1,966,080 parameters. The fresh Sleep receiver domain contains 393,216 parameters. These are checkpoint/source-derived storage estimates, not measured runtime VRAM. Beyond the live context, the audit estimated 144,151,408 bytes for two cloned contexts, 1,572,864 bytes for receiver gradients, and 3,145,728 bytes for Adam moments: 148,870,000 bytes (141.97 MiB) of additional state. Three contexts together account for 216,227,112 bytes (206.21 MiB), excluding those gradient and Adam tensors. These estimates exclude allocator fragmentation and temporary activations and should not be read as total peak memory.

An analytical estimate covered selected GEMM and attention terms only: **126,777.97–126,948.79 TFLOPs** across the unique grid (foundations counted once), comprising 55,115.31–55,165.95 TFLOPs for HOPE continuations, 69,762.81–69,879.68 for fixed-mix, and 1,899.85–1,903.16 for shared foundations. This is not measured or complete FLOPs: normalization, activation, RoPE, softmax, routing, losses, sampling, optimizer arithmetic, I/O, and other operators are excluded. The estimate also omits capture-time expert and original-MLP GEMMs that are not represented in its listed capture terms. No efficiency or speedup claim follows from this interval.

A retained-storage inventory taken at **2026-11-07 10:03:53 UTC** counted logical file lengths, not allocated disk blocks or deduplicated storage. The main six-run output directory contained 25,705,658,359 bytes across 171 checkpoints (checkpoint file lengths totaled 25,687,257,305 bytes). Across all 23 CLINC output-attempt directories, including retained pilots, failures, and diagnostic/replay runs, the total was 69,039,554,650 bytes; the 22 non-main directories accounted for 43,333,896,291 bytes. Three CLINC dataset-cache directories accounted for 4,626,402 bytes. The cache and output figures overlap with neither measured runtime VRAM nor a peak-disk estimate; hard links, allocated blocks, deleted temporary writes, and unrelated directories were not counted.

## Held-out test evaluation

The seal-bound evaluator scored all 4,500 test rows for each of the six final checkpoints, globally across all 150 intents, with no task mask or checkpoint selection. HOPE reached **17.83% ± 0.87%** accuracy and fixed-mix reached **10.76% ± 1.11%**. The paired fixed-mix minus HOPE difference was **−7.07 ± 1.95 percentage points** across the three class-order seeds.

| Seed | HOPE correct / 4,500 | Fixed-mix correct / 4,500 | Paired difference |
|---:|---:|---:|---:|
| 17 | 763 (16.96%) | 523 (11.62%) | −5.33 pp |
| 29 | 841 (18.69%) | 428 (9.51%) | −9.18 pp |
| 43 | 803 (17.84%) | 501 (11.13%) | −6.71 pp |

Invalid free-generation rates were high: HOPE produced 2,339, 1,353, and 1,845 invalid outputs by seed; fixed-mix produced 3,455, 3,320, and 3,182. Invalid outputs count as incorrect. The test result reinforces the local fixed-mix negative result, but these three class-order seeds share one model initialization and are descriptive, not independent replications. This is final-checkpoint performance only, not an official-test per-class forgetting curve or a wholly unseen-domain result; the test set shares the benchmark intent inventory. Evaluation cost was **715.063 seconds**, 14,770 full-model forwards, zero optimizer updates, and zero backward calls.

The first post-hoc aggregate attempt stopped on a schema mismatch (`primary.total` versus an assumed `primary.count`); it did not change or rerun the evaluator. The evaluation artifact remained bound by SHA-256 and the corrected count-only aggregation completed successfully afterward.

## Provenance and artifacts

- Frozen master config SHA-256: `ec8a9bfadde15a6b0bed202f1ad5eb621805e50b92151a6747827c0f177bc32f`
- Main six-run seal SHA-256: `13cb53aa440d1ec00d00b3059dc0fbb3c96e42405a984340a71b31fc56c73268`
- Source lock SHA-256: `0de1aadd3328089807f872a60b1b297a671bfb99ceac137d04faa08e2da2eb5a`
- Training-pack manifest SHA-256: `21b3651a3dce8534fa7363d7f31674c3d86dda735b0d896a60d03ffb055c9123`
- Clean-development cohort metadata SHA-256: `de8a7b7ba4f9af2d91ab0849b622daf0dcdfd507d64cfab54c0937a44b061780`
- Development aggregate JSON SHA-256: `1986b2ce7840fb53bc4f9d575cd2a6d4d16b25e5b6d70899064176f50102823c`
- Raw final-test evaluation artifact SHA-256: `e67bd2c7195c924a6ebf482af6b167550efb12c6b7f412e27a02a4ed72b49204`
- Final-test evaluator source SHA-256: `402ac9bc733354edf2fd52542eeb55aa5187faffaced16d2b620a0e9ba56501c`
- Frozen final-test evaluation contract SHA-256: `a278944e333894ddc3d2f4faa5adaadae3d6eaf2c5002133f1c88d97e8939c1a`
- Final aggregate JSON SHA-256: `31610594a4a0b5cdcc641a0b40dc7a72c019c2e606683a5a8d712360efe99000`

The aggregate generator re-derived cohort counts from the verified training pack with the frozen full-run input reader and checked them against the count-only sidecar. It used no model loading and no test payload. The report and charts are staged at [development aggregate](../results/clinc-v8-main/summary.md), [PNG trajectory](../results/clinc-v8-main/clinc-v8-main-six-development.png), and [SVG trajectory](../results/clinc-v8-main/clinc-v8-main-six-development.svg).
