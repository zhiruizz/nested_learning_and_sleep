# CLINC V8 main-six aggregate

Status: **complete_training_and_test**. Test status: **complete_all4500_global_unmasked**.

This is a paired, three-class-order-seed comparison using one fixed Llama initialization. The three seeds are not independent random initializations; statistics are descriptive means and sample SDs only.

## Development task-end accuracy

Rows show mean accuracy ± sample SD across the three paired class-order seeds. `old`, `current`, and `seen` are intent-scope metrics; denominators can vary slightly after the frozen duplicate-exclusion rule.

| Task | Arm | Earlier classes | Current classes | All seen |
|---:|---|---:|---:|---:|
| 0 | HOPE | n/a | 61.78% ± 9.32% | 61.78% ± 9.32% |
| 0 | HOPE_SLEEP_FIXEDMIX | n/a | 61.78% ± 9.32% | 61.78% ± 9.32% |
| 1 | HOPE | 6.11% ± 6.53% | 76.64% ± 6.99% | 41.36% ± 1.13% |
| 1 | HOPE_SLEEP_FIXEDMIX | 4.11% ± 3.79% | 40.38% ± 5.37% | 22.23% ± 1.69% |
| 2 | HOPE | 11.84% ± 1.85% | 81.83% ± 5.01% | 35.13% ± 2.60% |
| 2 | HOPE_SLEEP_FIXEDMIX | 14.34% ± 2.41% | 30.10% ± 9.17% | 19.58% ± 3.64% |
| 3 | HOPE | 12.17% ± 2.56% | 81.56% ± 7.67% | 29.53% ± 1.48% |
| 3 | HOPE_SLEEP_FIXEDMIX | 13.95% ± 2.68% | 36.33% ± 4.91% | 19.55% ± 2.34% |
| 4 | HOPE | 6.37% ± 2.63% | 79.78% ± 4.17% | 21.06% ± 2.29% |
| 4 | HOPE_SLEEP_FIXEDMIX | 11.99% ± 3.80% | 28.33% ± 17.23% | 15.26% ± 1.75% |
| 5 | HOPE | 5.16% ± 2.99% | 77.33% ± 6.84% | 17.20% ± 3.51% |
| 5 | HOPE_SLEEP_FIXEDMIX | 6.87% ± 2.49% | 26.56% ± 10.34% | 10.16% ± 1.55% |
| 6 | HOPE | 8.78% ± 3.55% | 86.85% ± 5.71% | 19.92% ± 3.68% |
| 6 | HOPE_SLEEP_FIXEDMIX | 7.25% ± 1.32% | 32.85% ± 1.56% | 10.90% ± 1.25% |
| 7 | HOPE | 6.36% ± 1.59% | 88.99% ± 4.15% | 16.68% ± 1.90% |
| 7 | HOPE_SLEEP_FIXEDMIX | 7.52% ± 0.73% | 29.15% ± 6.69% | 10.22% ± 1.26% |
| 8 | HOPE | 5.57% ± 2.63% | 78.65% ± 12.40% | 13.69% ± 3.25% |
| 8 | HOPE_SLEEP_FIXEDMIX | 7.81% ± 2.02% | 27.25% ± 20.68% | 9.97% ± 2.51% |
| 9 | HOPE | 10.13% ± 0.84% | 85.53% ± 6.83% | 17.67% ± 1.27% |
| 9 | HOPE_SLEEP_FIXEDMIX | 9.60% ± 1.18% | 24.69% ± 5.48% | 11.11% ± 1.08% |

## Final development endpoint (task 9)

Earlier-class accuracy is descriptive and combines forgetting with any earlier classes not learned at their own task boundary; only task 0 had an acquisition gate.

| Arm | Scope | Mean accuracy ± sample SD | Per-seed accuracy |
|---|---|---:|---|
| HOPE | old | 10.13% ± 0.84% | 9.23%, 10.90%, 10.27% |
| HOPE | current | 85.53% ± 6.83% | 84.00%, 93.00%, 79.60% |
| HOPE | seen | 17.67% ± 1.27% | 16.72%, 19.12%, 17.18% |
| HOPE_SLEEP_FIXEDMIX | old | 9.60% ± 1.18% | 9.42%, 8.53%, 10.86% |
| HOPE_SLEEP_FIXEDMIX | current | 24.69% ± 5.48% | 31.00%, 22.00%, 21.07% |
| HOPE_SLEEP_FIXEDMIX | seen | 11.11% ± 1.08% | 11.58%, 9.88%, 11.88% |

### Paired fixed-mix minus HOPE differences

| Scope | Mean percentage-point difference ± sample SD | Per-seed differences (pp) |
|---|---:|---|
| old | -0.53 ± 1.61 | +0.19, -2.37, +0.59 |
| current | -60.84 ± 9.22 | -53.00, -71.00, -58.53 |
| seen | -6.56 ± 2.32 | -5.14, -9.24, -5.31 |

## Work accounting

The common task-0 foundations are shared across both arms and counted once. Per-run global wake ticks include this shared 188-update foundation; continuation-only wake updates are shown separately.

- Shared foundation: 1150 forwards and 564 update-counter steps, counted once across seeds.
- HOPE: 5076 continuation wake updates; 0 Sleep events; 0 physical Sleep AdamW steps; 35871 full-backbone forwards; 1761.8 seconds summed across its three continuations.
- HOPE_SLEEP_FIXEDMIX: 5076 continuation wake updates; 423 Sleep events; 12690 physical Sleep AdamW steps; 102777 full-backbone forwards; 5231.0 seconds summed across its three continuations.

Sleep exposure is derived from actual event and physical AdamW counters. The receipt's `sleep_text_replay` flag describes pool availability; it is not treated as actual exposure for HOPE, which has zero Sleep events and zero Sleep AdamW steps.

## Interpretation and limitations

- HOPE and HOPE_SLEEP_FIXEDMIX use the same per-seed task-0 foundation; its cost is shared and counted once.
- Each continuation has 1,692 wake updates after the shared 188-update foundation; receipt global wake cursor is 1,880.
- HOPE_SLEEP_FIXEDMIX is a controlled fixed-mix local adaptation, not native/paper-faithful HOPE Sleep. Its additional class-balanced metadata replay pool draws four examples per class from the 15,000-row train pack; the continuation primary schedule contains 13,500 rows after the shared task-0 foundation. Replay exposure is inferred from actual events and optimizer counters, not the receipt availability flag.
- The two arms share their qualified foundation and primary new-task wake budget. The fixed-mix arm additionally incurs Sleep replay and optimizer work and applies a 0.5 branch mix; this is a local mechanism ablation, not a paper-faithful HOPE reproduction or a compute-matched causal estimate.
- The fixed-mix receipt may set sleep_text_replay=true for both arms as pool availability metadata; HOPE has zero Sleep events and zero Sleep AdamW steps, so actual Sleep text exposure is zero for HOPE.
- All three seeds share the fixed model initialization seed; they vary class order and are paired runs, not independent initializations. Report means/sample SD descriptively only; no inferential test is warranted.
- Development endpoint groups are old/current/seen at each task boundary. At task 9 these represent 135/15/150 intents; duplicated train-development rows were excluded per the frozen protocol, so example denominators can be slightly below nominal counts.
- Only the shared task-0 foundation has the configured >=0.5 acquisition gate; tasks 1-9 have no learning gate. Final earlier-class accuracy therefore combines forgetting with any earlier class that may not have been learned at its task boundary, and is not a pure retention rate. Current-task development accuracy provides context but does not certify earlier acquisition.
- The official test evaluation, when run, scores only the six final checkpoints globally. It does not provide initial test scores or an official-test per-class forgetting curve.
- The test split is held out from this training workflow but shares the benchmark's intent inventory; it is not a wholly unseen domain/task. Final test evaluation, when available, uses all 4,500 rows globally without task masks or checkpoint selection.

## Final held-out test (all rows, no task mask)

| Arm | Mean accuracy ± sample SD over three paired class-order seeds |
|---|---:|
| HOPE | 17.83% ± 0.87% |
| HOPE_SLEEP_FIXEDMIX | 10.76% ± 1.11% |

Paired fixed-mix minus HOPE: -7.07 ± 1.95 percentage points (descriptive, n=3).
