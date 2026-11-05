# Native symbol retention pilot v1

Frozen before execution, 2026-11-05 UTC. New synthetic namespace; no historical
Banking77 records, teacher banks, stopped queues or confirmation sets are read.
All three seeds (17/29/43), including failures, are reported. No budget or
hyperparameter changes are made in response to endpoint scores.

## Question and scope

After a common foundation, does the project's restricted multi-frequency CMS
update preserve old symbol mapping and short context copying while acquiring a
new mapping, compared with sequential updates on the identical parameter scope?
Frozen is a zero-update control. This borrows the old/new/copy diagnostics from
[obekt HOPE at 599244e](https://github.com/obekt/HOPE-nested-learning/commit/599244e).
Its 86M-tokenizer LM/Wikipedia PPL and factual metrics are quantitatively
incomparable with this tiny, synthetic, last-token classification experiment.
Reported `exp(NLL)` is a conditional answer-token perplexity, never corpus PPL.

## Fixed design

Use the released `NativeResearchLM`: one native deep SMT block, two CMS levels,
dim16/hidden24, rank2/capacity2. Activate one level-0 expert before foundation.
All inputs contain four tokens and a single target from values32..35:

- Old A: `[noise1, noise2, 40, 36+key] -> 32+key`.
- New B: `[noise1, noise2, 41, 36+key] -> 32+(key+1)%4`.
- Copy: `[42, value, noise1, noise2] -> value`.

Noise IDs0..15 are balanced nuisances. Training samples uniformly from even-sum
noise pairs; evaluation exhausts all 128 odd-sum pairs times four keys/values
(512 distinct prompts/task). Train/evaluation prompts are disjoint, individual
noise tokens occur in both. Copy is one value recalled across two distractors;
it can be solved with the convolution's short history. It does not establish
long-context or associative recall. No task uses padding or sample-to-sample
state carry. State persists only across the four tokens of each prompt.

Foundation: 512 AdamW steps, lr.003, weight_decay0, batch32, alternating old A
and copy (256 batches each), all model parameters eligible. Then fork exactly
the same foundation into Frozen/SFT/CMS. Adaptation: 128 batches of B, plain
SGD through existing CMSController, base/expert lr.03, clip1. SFT passes
`sequential=True`; CMS uses periods4/16 and LR-weighted gradient sums (not
averages). Both update only two CMS bases plus the active level-0 expert.
Embedding, SMT, normalization, readout and inactive experts stay frozen.
This is the project's persistent outer controller; functional `adapt_cms` is
not invoked. It is not identical to upstream's tiered AdamW or paper CMS.
No Sleep/GKD/Dreaming arm is claimed.

The two adaptation arms see exactly the same seed-specific B batches and token
budget. Base-write counts intentionally differ; learning rates and parameter
scope are matched. Evaluation uses a fresh state, eval mode and no gradients
at random initialization, foundation, and adaptation32/128. Frozen is measured
at128 with unchanged foundation weights. No model selection or early stopping.

## Outcomes, checks and cost

Report per-seed/task full-vocabulary greedy accuracy, four-value candidate
accuracy, full-vocabulary answer NLL, conditional answer PPL, old-task accuracy
loss and NLL increase, new-task improvement, parameter deltas, and raw aggregate
counts. Candidate chance=.25; uniform full-vocabulary reference NLL=ln43.
Acquisition is established for a task only if its foundation full-vocabulary
accuracy >=.8. Apply all arms regardless, but call subsequent differences
retention only for acquired tasks. Otherwise report failed acquisition.
No significance claim from three seeds; publish each and descriptive means.

Before the main run, check stream/full-forward equivalence and sequential vs
period-one CMS equivalence on a separate synthetic fixture. Failures stop the
run. Record config/source hashes, dependency versions, model/scope size,
attempts charged before updates, forward/input/gradient tokens, elapsed time,
and completed/failed state. CPU/one thread; cap2304 main update attempts plus
two separate correctness-fixture attempts, and1200s for the runner.
No automatic retry/resume/extension; retain partial outputs if interrupted.

New code is MIT and new documentation/results compilation CC BY4.0 under the
public repository's existing asset-specific license scope. Synthetic aggregate
metrics only are published; model snapshots and datasets are not bundled.
