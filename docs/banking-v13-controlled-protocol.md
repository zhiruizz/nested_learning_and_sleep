# V13: batched local Knowledge Seeding across two natural boundaries

This is a development-only study designed before any V13 outcome is scored.
It extends V12's fixed-teacher, one-boundary diagnostic to both GKD sampling
policies, local LTI, two consecutive consolidation events, and subsequent wake
updates. It is not native HOPE, a full seven-task benchmark, or a numerical
reproduction of the Sleep paper.

## Evidence and choices

[The primary-source audit](sleep-paper-v13-primary-source-audit.md) distinguishes
the manuscript's general Table 5 (effective batch 32, LR 5e-6, Sleep 100 steps,
LoRA rank 64 / alpha 128) from undisclosed per-classification-event settings.
No author-linked executable recipe was found in the inspected sources.
We test an explicit local interpretation rather than presenting inferred
replay counts or clock units as source facts.

The model, dataset and qualified foundations are the same pinned V10 assets.
No new model or dataset is required. Existing official test outcomes were
already viewed in V10 and are not inputs to V13. All tuning and analysis use
training/development data. No sealed P2 data or Ubuntu experiment is accessed.

## Frozen design candidate

The complete machine-readable recipe begins as
[this draft](../configs/banking_v13_batched_lifecycle_draft.json); a separate
source-bound frozen config must identify actual launches. Seeds 17/29/43 share
initialization 1907008 and vary class/data order. Each of six arms restores the
same seed's qualified task-0 foundation and untouched pending gradient windows.
Those foundations were trained with wake batch 8; changing task-1 wake batching
does not retroactively make them batch-32 foundations.

Wake effective batch is 32 in every arm, using consecutive full batches from
the concatenated deterministic task-1 epoch-0 and epoch-1 streams. Batches can
cross an epoch boundary; none drops examples or silently becomes partial.
Fast/base rates remain 0.1/0.001 and periods remain 16/64/256/1024/held-fixed,
measured in wake minibatches. These numerical choices are local assumptions.
The study does not isolate a causal wake-batch-8 versus wake-batch-32 effect.

The actual foundation cursors are 348/326/322. The two level-0 boundaries occur
at 352/368, 336/352, and 336/352 respectively; postwake endpoints are 383/367/367.
The first prefixes are 4/10/14 ticks, then 16 ticks between the boundaries,
then 15 further wake ticks. No other level is due within these windows.
CPU replay of the actual restored clocks must verify these facts; unexpected
due levels abort rather than silently disappearing from the schedule.

| Arm | Sleep effective batch | LR | Requests/event | Update opportunities/event |
|---|---:|---:|---:|---:|
| CMS_ONLY | none | none | 0 | 0 |
| B1_HIGH | 1 | 1e-3 | 960 | 960 |
| B1_LOW | 1 | 5e-6 | 960 | 960 |
| B32_HIGH | 32 | 1e-3 | 960 | 30 |
| B32_LOW | 32 | 5e-6 | 960 | 30 |
| B32_LOW_DOSE | 32 | 5e-6 | 3,200 | 100 |

There are 18 complete seeded trajectories and two events per Sleep trajectory.
Planned Sleep exposures total 42,240 requests; up to 12,480 physical AdamW
updates occur if every opportunity makes a nonzero update. Actual updates,
skips, backward calls and generated tokens must be reported separately.
Wake adds 726 updates / 23,232 labeled request exposures over all arms.
These counts exclude qualification and failed engineering attempts.

## Meaning of a batch

Each event's requests follow GKD,GKD,LTI. The common selector constructs 3,200
positions per event and every arm uses its prescribed prefix, preventing the
second event's prompts from shifting with arm budget. Old/current training
roles and known class IDs balance the selector; this is additional metadata
access, not evidence that the paper prescribed the same replay selection.

GKD alternates teacher- and student-policy generation. LTI samples a teacher
sequence, a random prefix, and a student continuation. Generation has no
gradient. Within an effective batch the student parameters stay fixed;
individual token-mean objectives are backpropagated with weight 1/batch-size,
then one global clip and AdamW step is made. A batch of 32 is not 32 sequential
AdamW steps. Actual GPU tensor microbatching of one request is disclosed.

Local loss/reward formulas remain the V10 assembly, including its canonical
label-equivalence proxy, normalized edit reward, and 0.5 KL + 0.5 reward-CE LTI
term. This is not the paper's unspecified frozen semantic reward model.
The 3,200-request stream has 2,134 GKD and 1,066 LTI positions, including a
terminal two-GKD suffix. Its 100 opportunities are a local per-event Table-5
interpretation. LoRA alpha 128 is not inserted as an unexplained multiplier
into nonlinear CMS experts; that parameterization difference remains explicit.

## Comparisons and measurements

Primary contrasts are the request-matched Sleep batch×LR cells, within each
seed at every predeclared stage. On-policy outputs, actual token counts, Adam
steps and FLOPs are not matched by equal prompt counts. The dose contrast
changes both exposure and update budget. CMS has neither expansion nor Sleep
data/compute and remains a reference, not a compute-matched causal control.

At each boundary, compare the protected pre-update teacher with the committed
post-Sleep student on the same 110 old and 110 current development examples.
Evaluate the postwake-15 endpoint too. Primary accuracy ranks all 77 labels by
summed likelihood including EOS, without a task mask; token-mean scoring, gold
token NLL and unconstrained greedy are separate readouts. Candidate captures
can be shared between before/after contexts. Keep every seed, stage, failure
and mean/sample SD; do not select an endpoint to replace the primary result.

Teacher/live/protected student state, routing, optimizer and RNG guards are
required. Real-disk transaction restoration must be checked against the
original continuing object before expanding the main grid. It is a narrow
transaction qualification, not whole-job resume. Snapshots are only allowed at
completed effective-batch boundaries. Persist each evaluation stage before
writing the final aggregate, so serialization repair need not rerun scoring.

Output paths are fresh private project directories. Checkpoint tensors, raw
requests, labels, generated text and token trajectories are never published.
Progress contains counts and phase names only. Source/config/model/data pins,
CPU guard checks, actual clock preflight and independent review precede CUDA.
There are no automatic retries. Per-process memory policy remains 0.8 with the
existing allocator settings; no system settings or unrelated jobs are changed.
The draft ceilings (14,400 seconds, 200,000 backbone forwards and 8,000 update
attempts) apply to each run. Study cost is the sum of actual runs and separately
recorded qualification/engineering work, not one of those per-run ceilings.
