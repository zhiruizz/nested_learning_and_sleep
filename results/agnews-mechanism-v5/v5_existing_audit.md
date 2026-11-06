# Existing v4 aggregate audit for v5 probe design

Aggregate-only audit of the published Phase B results. It covers all 3 seeds, all 6 arms, and all 5 fixed checkpoints (18 runs, 90 evaluations). No model, GPU, raw records, prompts, labels, or confirmation scoring were used. The collapse interval is bracketed only by recorded checkpoints; no interpolation.

## Seed 29 warm_pre_sleep_P4

At step 64, marker accuracy is 99.61%, all EM 25.00%, and new EM 0%; at step 112, marker accuracy and old/new/all EM are all 0%. New answer NLL reaches 51.8794 and all answer NLL 43.0017. Therefore the aggregate supports a marker-level failure by step 112, but localizes its onset only to the 64-to-112 interval.

Event logs show 672/672 positive-gradient updates and zero zero-objective updates in this run. This does not prove gradients were useful; it does argue against a simple all-zero-gradient explanation.

| Checkpoint | Old EM | New EM | All EM | All marker accuracy | All answer NLL | New answer NLL | All marker NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 50.00% | 0.00% | 25.00% | 99.61% | 6.4241 | 7.8395 | 0.028458 |
| 112 | 0.00% | 0.00% | 0.00% | 0.00% | 43.0017 | 51.8794 | 46.943420 |

Across the five non-replay arms, step-112 old EM is at most 2.73% per seed; the replay arm retains 63.28% to 86.33% old EM. New-task EM generally appears after steps 16/32, but the seed29 pre4 run stays at 0% through all points. Marker accuracy remains high in most runs; the step-112 seed29 pre4 marker failure is an outlier, not a general pattern. The JSON also contains n=3 mean/sample-SD summaries for each arm/checkpoint.

## All-seed, all-arm checkpoint records

`v5_existing_audit.json` preserves all 90 rows and per-class EM, candidate accuracy, class-given-marker accuracy, marker accuracy, answer NLL, class NLL, and marker NLL. Its milestone section lists exact checkpoint series for each seed/arm and first observed threshold crossings; the fixed checkpoints are 0, 16, 32, 64, and 112.

## Event outliers and receiver visibility

Across pre4, seed29 has no zero-objective events; seed43 has 4 zero-objective updates at boundaries 28 and 32. Across post4, seed29 has 46 zero-objective updates, including 12 at boundary108. These events report no zero-objective-with-zero-gradient updates; every event has 24 positive-gradient updates. The outlier timing is not aligned with seed29 pre4's marker collapse.

For pre/post P4, step-0 adapter fingerprints match the selected warm checkpoint. Event metadata shows receiver level 1 and receiver slots progressing 1 through 28, with active slots extending to 28 before the final commit. Teacher order is `pre_fast` for pre4 and `post_fast` for post4. Event teacher fingerprints are distinct and do not match the stored warm teacher fingerprint. The first event's initial-student fingerprint is taken after primary updates, so the event log does not directly identify the warm receiver state at initialization.

Warm control chain (all-example EM, reset-without-KD train / transferred warm-student validation / receiver-disabled validation): seed 17 0.39% / 92.19% / 0.00%; seed 29 93.55% / 93.75% / 93.75%; seed 43 49.61% / 89.84% / 49.61%. This seed variation argues against a universal receiver-localized explanation.

## Planned class/pool coverage

The locked schedule uses 24-example Sleep pools with 6 planned draws per class. P4 runs have 28 event pools per seed; P64 has one. The extra-gold-replay path schedules 12 four-example batches at each P4 event, cycling the 24-example pool twice: 336 replay batches / 1,344 planned example draws per seed, or 336 per class. These are plan-derived counts; the aggregate event logs do not expose realized sampled class IDs.

## Data-driven hypotheses and next controls

1. Isolate teacher ordering first. At step 112 seed29 pre4 has marker/new EM 0, whereas post4 has 98.83% marker accuracy and 49.61% new EM. This is a useful matched contrast, but the three-seed result is not causal evidence because ordering is bundled with fast-update timing.
2. Add a no-update validation branch after every four-step boundary from 64 through 112; report marker and conditional class metrics separately. This will localize whether marker failure precedes class failure without changing training state.
3. Log warm-checkpoint, event-start, teacher, receiver-slot, source-bank, and reset/update provenance explicitly. Existing event metadata shows slot progression but not a direct semantic link from the event receiver to the warm checkpoint.
4. Repeat receiver-disabled and reset-without-KD controls for all seeds with classwise validation. At the chosen warm budget, receiver-disabled old/all EM is 0%, 93.75%, and 49.61% for seeds 17/29/43, so receiver dependence is clearly seed-specific.

## Limits

All observations are validation/development aggregates. The five checkpoints cannot pinpoint the exact event at which the seed29 failure began. No causal claim, significance test, or confirmation/generalization claim is made. The planned pool counts come from the frozen driver design, not actual label inspection.

Input aggregate SHA-256 values are recorded in the JSON. No D: files were changed.
