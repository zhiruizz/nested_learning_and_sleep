# Supplementary fixed-trajectory training-fit diagnostic

This input-selection plan was written while seed 17 was still fitting its
receiver, before any V12 development accuracy result was read. The primary
three-seed training recipe remains unchanged. This supplement measures whether
the receiver approximates its teacher on an identical small trajectory cohort;
it does not create a held-out or confirmatory result.

Use exactly the first 64 records of each seed's already-generated, private
teacher training-trajectory stream, without inspecting or filtering their
contents or scores. Frozen old/current alternation gives 32 records per role.
These requests may be used in receiver training. Labels, new trajectories and
gold targets do not enter the KL calculation. Do not present the result as
validation performance or an unbiased generalization estimate.

At the teacher, CMS-only/no-reset context, reset/untrained student and all seven
predeclared receiver checkpoints, compute the same full-vocabulary teacher-to-
student KL at T=1 on the same answer suffixes. Average tokens within each
request, then requests; keep old/current/all denominators explicit. Reuse each
fresh frozen-backbone capture across the contexts. No optimizer or generation
update is permitted. Report numerical self-KL tolerance rather than silently
clipping negative roundoff to zero.

Reconstructed baseline contexts must match the state hashes from the frozen
training receipt; otherwise abort the supplement and report the missing
evidence. Verify every checkpoint and trajectory-file hash, protect the shared
backbone, and compare context states before and after scoring. Never export raw
tokens, text, labels or individual predictions. Keep the supplement's source
lock, input manifest, counters and wall time separate from main experiment cost.

The frozen input plan is [the JSON plan](../configs/banking_v12_fixed_teacher_fit_eval_plan.json).
Its source and actual checkpoint manifest will be pinned only after all runs
are complete and the read-only evaluator passes review. Training batches and
their changing request composition remain a separately named streaming-loss
diagnostic; their final losses cannot substitute for this fixed-cohort metric.

The private rollout is one serialized 3,200-record payload per seed. The
evaluator deserializes that complete training payload and then selects and
scores exactly its first 64 records. The completed output distinguishes
`records_scored: 64` from `records_deserialized_from_private_payload: 3200`.
The initial evaluator's ambiguous `records_read` wording was corrected during
output repair. No other trajectory contributes to the KL calculation,
selection or public output.
