# AG News v4: competent transfer and update-order development experiment

This is a development experiment using previously inspected train-derived validation roles. It is not independent confirmation, an official AG News benchmark, or a reproduction of the complete Sleep or HOPE architecture. The earlier v1–v3 failures remain in the repository.

This narrative was written while phase B was running. The executable configuration and phase-specific source locks, with their recorded timestamps and hashes, are the earlier frozen records; this document is an explanatory description, not a retrospectively dated preregistration.

## Question and boundaries

The preceding [v3 diagnosis](../reports/agnews-sleep-forensics-v3.md) found that the old-task ability was already mostly absent when the first teacher was constructed. This follow-up first asks whether a competent old teacher can be transferred through the reset/receiver operation. Only after all three seeds pass the transfer gate does it compare consolidation schedules and teacher order during new-task learning.

The old teacher is the qualified v3 foundation saved before its new-task adaptation, not a reconstruction from a later checkpoint. The originals are hash-verified and copied byte-for-byte into a private durable teacher bank before phase B. Copy time is distinct from the original snapshot time. Phase A uses the original teacher for its complete distillation trajectory. Phase B self-teachers are constructed from each arm's current state; the protected old checkpoint is not repeatedly substituted for them.

Seeds are 17, 29 and 43. The local Llama 3.2-3B backbone, tokenizer, original task mapping, data pack, v3 foundation checkpoints and core library are pinned by SHA-256. Model weights came from ModelScope `LLM-Research/Llama-3.2-3B`, revision `48d84dd14886f01964f72e7f742a404dd86f9fb6`. The dataset is `fancyzhx/ag_news`, revision `eb185aade064a813bc0b7f42de02595523103ca4`; its original acquisition used the official Hugging Face source after ModelScope was unavailable. No new weights or data are downloaded for this experiment.

The existing private pack has 256 training and 128 validation articles per class. Classes 0/1 are the old task and 2/3 the new task. Validation contains 256 old and 256 new articles. A shared marker followed by an arbitrary class code forms the two-token answer. Exact match is unrestricted full-vocabulary greedy prediction of both tokens, with the unchanged v3 scoring code. Candidate accuracy and answer NLL are secondary metrics, not substitutes for exact match.

## Phase A: transfer a competent teacher before new-task updates

The prospective fast-base proposal is its existing absolute weight tensor, so this boundary makes no new-data base update. The student resets the fast expert bank, expands one slow receiver, and trains with the persistent 0.5 receiver mixture. The two-level last-MLP extension uses rank 16 and capacity 32 per bank. Base matrices do not consume expert slots. The qualified state has active expert counts `[0, 1]` after transfer.

The finite, prewritten common budget ladder is 24, 96, 384 and 1,536 Adam updates per seed. Each later stage resumes the same session, including its optimizer and local generator; it is not a fresh restart or a sum of independent budgets. Every 24-update cycle uses 16 GKD updates, alternating eight teacher and eight student-generated trajectories, and eight LTI updates. Learning rate is 0.003, generation length two, with no Dream updates. Pools use training prefixes only and are balanced over the old classes. Their fixed schedule is source-pinned.

At every ladder stage, all seeds are evaluated on old train and old validation. A seed passes only if:

- Old train EM is at least 80%.
- Old validation EM is at least 75%, with each old class at least 70%.
- Validation EM is at most five percentage points below its immutable teacher, both overall and separately for each old class.

Phase B requires all three seeds to pass at the same first qualifying common stage. Earlier failing stages are retained and reported. No seed replacement, per-seed best-checkpoint selection or weakened threshold is allowed. Training and validation inform this budget choice, so the resulting state remains development-selected.

Read-only controls include the unchanged competent teacher, reset without KD, and each stage's trained student with its new receiver disabled. They are functional dependency probes; they do not identify a uniquely localized memory. The seed-17 stage-24 recovery probe restores the saved session twice, executes one identical GKD update in each branch, compares mutable model/optimizer/policy/generator/record state, and verifies the saved snapshot is unchanged. Its two additional Adam updates are charged separately. No probe changes the main trajectory.

The phase-A cooperative wall limit is 3,600 seconds, checked between operations. It is not a forced kernel deadline. A failure prevents phase B. Mutable session state, model/controller state, execution policies, RNG states and source/data hashes are checkpointed privately; neither pretrained weights nor generated token sequences are redistributed.

## Phase B: fixed-endpoint comparison

Before any phase-B updates, a separate source lock binds the driver, generalized boundary helper, configuration, completed phase-A result and teacher-bank manifest. It also fixes endpoints, paired contrasts and the development-only interpretation.

Each seed has six arms, all with 112 primary new-task minibatches using the same original v3 schedule prefix, batch size four, base/expert learning rates 0.02/0.04, and fast/slow CMS periods 4/16:

| Arm | Initial state | Additional operation |
|---|---|---|
| `foundation_CMS_reference` | Original competent foundation | No transfer or later Sleep; reference with a different initial representation |
| `warm_CMS_no_sleep` | Qualified transferred state | Matched no-Sleep control |
| `warm_pre_sleep_P64` | Same qualified state | One 24-Adam event at primary step 64, teacher before the fast base update |
| `warm_pre_sleep_P4` | Same qualified state | A 24-Adam event before every fast-base commit, 28 events |
| `warm_post_sleep_P4` | Same qualified state | Same event/pool/slot/Adam budgets as pre-P4, teacher after the fast-base update |
| `warm_extra_gold_replay_P4` | Same qualified state | Every four primary steps, 12 additional four-example gold-label minibatches from the same 24-prefix pool, traversed twice |

All five warm arms must have exactly identical step-0 validation metrics and initial mutable/policy state for a seed. The foundation reference is not described as an isolated teacher-transfer effect: warm transfer changes representation and incurs additional optimization.

At a Sleep boundary, the primary gradient is first accumulated with the due proposals deferred. When the slow level is also due, its absolute-weight proposal is applied to a private clone before teacher construction in both order arms. A pre-update teacher retains the current fast base, while its student receives the prospective fast proposal. The post-update control first applies that same fast proposal to its teacher clone and then passes the resulting identity proposal to its student. These are absolute target weights, not additive deltas. The first pre/post students must have identical mutable fingerprints. Later states can diverge because the teachers differ.

Here "pre-update" means before the fast **base** write. Active fast experts have already received the current minibatch's online SGD update before the boundary transaction; the teacher is not a snapshot from before that minibatch or from the task switch. This preserves the existing local wake rule and limits what the order contrast isolates.

Only the fast accumulator is consumed at fast-only boundaries. Pending slow sums/counts remain bit-identical until the 16-step boundary and are then consumed exactly once. Transactional checks require the original live model/controller/policy to remain unchanged, and committing a trained student must preserve its mutable fingerprint.

Each event uses 24 training prefixes balanced six per class, 16 GKD plus eight LTI updates, learning rate 0.003, two-token generation and the same persistent 0.5 mixture. Prefix selection uses class metadata for balance; gold answers do not enter Sleep optimization. The replay control explicitly uses gold labels and advances its own CMS clocks. These controls do not have equal FLOPs or equal label exposure.

Pre-P4 and post-P4 allocate one receiver per event and finish with active expert counts `[0, 29]`: the warm receiver plus 28 new slots. Capacity is checked, and no slot is overwritten. Period 64 versus period 4 changes cadence, optimization cost and capacity together; it is not a pure timing ablation.

Validation and private recovery checkpoints occur at primary steps 0, 16, 32, 64 and 112, giving 90 evaluated states across 18 runs. All fixed endpoints are reported. There is no metric-based early stopping or endpoint selection. The phase-B cooperative wall limit is 5,400 seconds. Numerical or guard failure stops the entry point without automatic retry; a failed event is not silently restarted. Infrastructure recovery, if necessary, requires explicit evidence/source binding and a separate cost receipt.

## Outcomes and cost accounting

Primary endpoints are old/new/all EM at step 112, old EM drop from each arm's own initial state, and old/new answer NLL. Report per-seed results, means and sample SD across three seeds, plus these fixed paired differences:

- Pre-P4 minus post-P4: the teacher-order policy with matched event, slot and Adam budgets.
- Pre-P4 minus warm no-Sleep: the complete added Sleep package.
- Pre-P4 minus pre-P64: the cadence/compute/capacity package.
- Warm no-Sleep minus foundation reference: the complete boundary-transfer package.
- Extra gold replay minus warm no-Sleep: additional labeled exposure and SGD.

No p-values or independent-confirmation claims are planned. Seeds are training trajectories on the same validation articles, not three independent datasets. A high old score with poor new-task learning is not a successful adaptation/retention result.

Phase A can use at most 4,608 main Adam updates plus the two restore-probe updates; these are shared preparation, counted once in actual experiment totals. A warm method's deployment path separately incurs its own 1,536-update transfer cost when that stage is selected. Phase B has 2,016 primary SGD updates, 1,008 extra replay SGD updates and 4,104 Sleep Adam updates. P4 uses 672 Sleep Adam updates per run; P64 uses 24. Model forward calls, input-token proxies, gradient-forward input tokens and wall time are retained. These proxies are not FLOPs, energy or GPU busy time; wall time includes scoring and checkpointing. CPU fixture updates are separate from GPU experiment updates.

## Isolation, limitations and reproduction

The original official test and v3 within-train holdout were already inspected in previous studies. Neither is scored or used by the training/evaluation entry points here. The CPU-only reservation utility does read the old v3 prompt pack to collect record IDs and stored token-prefix hashes for exclusion; it does not load its labels or run a model. A new 1,024-article within-AG-News-train candidate set is reserved with disjoint record IDs and token-prefix hashes relative to the v3 train/validation/holdout roles. Its prompt and label files are private and remain unscored. Only reservation metadata and hashes are published. It is neither an external test nor a guarantee against model-pretraining exposure. Future use would require freezing a method and confirmation plan before opening it.

No P2/Phase13 sealed confirmation data are used. This is independent of the blocked Ubuntu formal experiment. No Ubuntu action is needed for this study.

The local method remains a partial implementation: two-level rank-16 last-MLP adapters, fixed random routing, explicit mixture, two-token class coding and exact-output LTI reward. It does not implement the paper's complete hierarchy, semantic reward model, long-horizon evaluation or full HOPE architecture. The [Sleep paper §§3.2–3.3](https://arxiv.org/html/2606.03979v2#S3.S2) motivates consolidation before a sender base update; the implementation and controlled comparisons here are independent engineering experiments. No upstream HOPE source is copied.

The report links aggregate JSON, all fixed validation states, failures, hashes, cost and figures. The standard-library analyzer recomputes tables from published aggregates without models or data; `--plots` additionally requires Matplotlib. GPU training requires the legal local model assets, exact private role pack and qualified foundation snapshots referenced by hashes. Those assets are not bundled, so publication supports offline aggregate reanalysis rather than a claim of complete asset-free training reproducibility.

The ordinary suite remains `python -m pytest -q`. Run the seven generalized-boundary checks separately with `python -m pytest -q tests/check_agnews_preupdate_v4_cpu.py`, and the existing v3 observer-equivalence check with `python -m pytest -q tests/check_agnews_sleep_forensics_v3_cpu.py`. These explicit `check_` files retain `tests/conftest.py`'s network/CUDA prohibition and 256-update process guard. They use 233 and 64 toy update attempts, respectively; combining them with the ordinary suite would exhaust that unchanged guard. They are intentionally outside default `test_*.py` collection, not unexecuted checks. The observer check's code is unchanged apart from its filename. The initial combined-suite attempt reached the 256-update guard; that failure ledger is preserved separately rather than increasing the limit. On the experiment's CUDA wheel, the single pre-existing CPU-only-wheel assertion is deselected; a CPU-wheel environment remains needed to certify that packaging property.
