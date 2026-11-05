# AG News class-incremental retention v2 - frozen preregistration

**Status at preregistration: source/config/evaluator locked; real-3B four-cycle DEV recovery passed; no main v2 training or official-test decoding has occurred. The user authorized execution and publication.**

## Scope and version boundary

The v1 run failed closed during seed 17 `CMS+Sleep` event 2 because the new receiver had zero gradient at the two loss positions. Publish and retain that result as a v1 failure (release reference `54b11c6`; preceding preregistration `b9d6033`). Its completed arms and partial Sleep updates are reported separately and are never combined with v2. V2 starts all five arms fresh for all three seeds.

This is an exploratory class-incremental Llama study and a local Sleep-inspired engineering variant. It is not a faithful reproduction of the Sleep paper, not a test of natural receiver routing, and not a contamination-free benchmark.

## Fixed data and model

Use only the frozen AG News train-role pack with SHA-256 `7b5169daadcf7d4ccaefc9d0bb79f1a4eef92fc275fae0c3a7d593e06b7e9e59`, from `fancyzhx/ag_news` revision `eb185aade064a813bc0b7f42de02595523103ca4`. Old classes are World (0) and Sports (1); new classes are Business (2) and Sci/Tech (3). Preserve the pack's precomputed schedules, seed-specific code maps and role splits. Main seeds are 17, 29 and 43; development seed 1907 is not part of the main results. Model: local ModelScope Llama 3.2 3B; frozen BF16 backbone, FP32 mutable last-MLP CMS-inspired scope, rank 16, capacity 8 per bank, two levels, periods `[4,16]`. Use base/expert learning rates `(0.005,0.01)` selected by the completed DEV protocol. Runner, helper, config, model/tokenizer asset, evaluator and schedule hashes are pinned in [the source lock](../results/agnews-retention-v2-preregistration.json). Asset source is ModelScope `LLM-Research/Llama-3.2-3B`, weights revision `48d84dd14886f01964f72e7f742a404dd86f9fb6`; auxiliary files are pinned individually, not claimed to share that revision. The queried ModelScope dataset mirror returned 404, so pinned official Hugging Face Parquet bytes were used.

Answers are two tokens (shared marker and seed-mapped single-token class code), with no EOS target. Use full-vocabulary greedy two-token exact match and NLL, plus marker accuracy and four-code candidate accuracy. Report each class and old/new groups. Do not constrain greedy prediction to the four class tokens.

## Foundation and five fresh arms

For each seed, train one 512-update, batch-size-4 foundation on old-class train prompts. Fork every arm from that exact foundation checkpoint with identical initialization, execution policy, optimizer initialization, RNG state and primary data schedule. Run all five arms for all three seeds:

1. **Frozen:** no new-task updates.
2. **Per-step adapter SFT (same mutable scope, frozen backbone):** 256 new-task updates with controller periods `[1,1]`.
3. **CMS:** 256 new-task updates with periods `[4,16]`.
4. **CMS+Sleep-v2:** the same 256 primary new-task updates and CMS periods, with Sleep after primary steps 64, 128, 192 and 256.
5. **CMS+extra-replay:** the same CMS primary schedule, plus 12 batch-size-4 labeled replay updates at each of those four boundaries (48 additional SGD updates per seed).

At a Sleep boundary, defer both due proposals on a candidate clone; apply slow level 1 first; construct the teacher from that slow-updated candidate; pass fast level 0 prospectively to the student; commit the candidate and consume each due proposal exactly once only after all engineering gates pass. The live model/controller must remain unchanged until commit. A failed gate or exception stops the entire main block. There are no automatic retries, seed replacements, v1 checkpoint reuse, or resumption from a partial v2 checkpoint.

Each Sleep transaction uses 16 GKD and 8 LTI AdamW updates at LR `0.003`, gradient clip `1.0`, `dream_tokens=2`, and zero Dream updates. The fixed 24-prompt train-only pool is 50/50 old/new. Keep the strict finite, scope/frozen-state, immutable-backbone and positive receiver-gradient gates. A zero receiver gradient remains a hard failure.

### V2 routing intervention

V2 applies an explicit persistent receiver mix of `0.5` in the student at the receiver level: the student's natural routed output is blended with the new receiver branch at fraction 0.5. The teacher keeps its pre-transaction execution policy. After commit, the mixed route remains active for subsequent training, later Sleep transactions as the starting teacher policy, checkpoints, and final evaluation. This changes both optimization and deployed inference policy; it is a controlled-route Sleep variant, not evidence for natural routing or a one-factor Sleep effect.

Record natural top-1 receiver token coverage and executed mixed-route coverage separately, with slot IDs and per-event gradient norms. Do not interpret forced branch coverage or positive gradients as proof of useful retention. Capacity is eight slots and four events; verify each event uses the expected free slot without overwrite. Serialize and restore the non-tensor routing policy for live, teacher and student execution state. CPU tests must show teacher policy is preserved, student mix policy survives restore, candidate failure leaves live model/controller/policy unchanged, and successful commit carries the policy into the deployed student.

## Replay arm and compute accounting

For each event, replay cycles that event's fixed 24-prompt pool twice to form 12 batch-size-4 labeled updates; one record from each class appears per batch. It uses only the frozen training pool and does not advance the primary-data cursor. Replay updates do advance the same CMSController clocks, so due-event phase and pending counts in this arm differ from CMS and CMS+Sleep. Report those actual clock counts and do not describe its schedule as CMS-identical.

The 12 batches were selected as a measured near-compute proxy for one Sleep event, not exact FLOP matching. Replay uses ground-truth labels, while Sleep uses teacher/student distillation objectives; prompt exposure, gradient updates, labels and controller-clock behavior are not identical. Report per-arm measured forward/backward token counters, update counts, wall time and CMS counters. Do not attribute an arm difference solely to the Sleep mechanism.

## Checkpoints, stopping and test sealing

Use a fixed 3,600-second hard cap for the complete 15-arm block, with no extension. Planned work is 4,752 SGD updates (1,536 foundation plus 3,072 primary adaptation plus 144 replay) and 288 Sleep-Adam updates. Checkpoint each foundation and each arm at primary cursor 0, 64, 128, 192 and 256 (Frozen at 0 and 256). Boundary checkpoints bind mutable adapter tensors, routing/readout execution policy, controller pending gradient sums/counts, all four RNG streams and data/config/source/schedule cursors/hashes. Boundaries are outside Sleep and do not contain in-flight teacher/Adam; the separate real-3B fixture verifies that in-flight lifecycle recovery. SGD pending base gradients use sums, not means; supervised clipping norm is 1. The CLI does not resume; any main failure leaves the block unsealed and ends the block.

Complete all three foundations and all 15 seed-arm runs before creating or scoring the fixed official-test sample. Only after the training seal, select 128 examples per class using the frozen v1 test salt `nl-sleep-agnews-officialtest-v1`; exclude exact normalized identities and tokenized prompt hashes matching every train and calibration role. No test decode, score or outcome may affect training, Sleep commits, checkpoints, seed retention or selection. If any arm fails or the hard cap expires, do not unseal the test set.

After all runs are sealed, report main calibration as descriptive acquisition information only; it must not filter seeds or arms. Report all seeds and all five arms, failures included. Interpret retention only for seeds that had acquired the old task at foundation; do not replace a failed seed or call an unacquired ability retained. With three seeds, present per-seed values and descriptive mean/dispersion without population-level claims.

## Development gates and limits

AG News is public and may overlap Llama pretraining. The experiment uses a deduplicated, hash-selected subset, caps article inputs at 96 tokens, and uses arbitrary code labels. The result therefore tests adaptation on this fixed setup, not unbiased performance on fresh news or full AG News. The mutable scope is a rank-16 last-MLP adapter and active slots differ by arm; disclose parameter counts and the persistent routing intervention. The reward is an exact-output proxy and the method omits Dream updates and other parts of the paper's Sleep procedure.

The [published v1 failure](../reports/agnews-retention-v1-failure.md) is retained. [Real-3B train-only route forensics](../results/agnews-receiver-routing-forensics-v1.json) showed that the new slot was selected at nine body positions but neither of two loss positions; a fixed mixed branch obtained nonzero gradient without modifying live state. The [four-cycle DEV fixture](../results/agnews-controlled-mix-dev-v2.json) completed 256 new-wake steps, four full Sleep events, exact wake127→128 restore under a mixed policy, and two independent in-flight Sleep restores. Its 96 actual Sleep updates plus two ghost restore updates are separate development cost. No DEV accuracy was used to tune mix strength, add steps or select a main endpoint. Positive gradients demonstrate reachability, not retention benefit.

Official-test selection/evaluation is source-pinned, gated by the complete 15-run seal and checkpoint hashes, and uses exclusive hash receipts plus exclusion/uniqueness/class-count/prompt-boundary checks. The same 512 articles are used across seeds, not 1,536 independent test examples. NLL is teacher-forced; fixed-length greedy EM equals marker correctness AND class-token correctness conditional on the gold marker for this causal model.

Papers: [Sleep v2](https://arxiv.org/abs/2606.03979v2), [Nested Learning v1](https://arxiv.org/abs/2512.24695v1); upstream diagnostic inspiration: [HOPE commit 599244e](https://github.com/obekt/HOPE-nested-learning/tree/599244e65fedf26f4db9844beb631c3dbca41188). No upstream code was copied. Model/data/checkpoints remain local; repository licensing does not extend to the third-party assets.
