# AG News v5: interference diagnosis and anchored-wake development comparison

This is a development study following the negative v4 result. All method choices use previously inspected training/validation roles. The reserved 1,024-article split remains unscored; neither its prompts nor labels are inputs here. P2/Phase13 data and the separately blocked Ubuntu experiment are untouched. This narrative describes the executable plans; their actual timestamps and source hashes are the frozen records, not a retrospective preregistration.

## Evidence used to choose the comparison

The read-only component study evaluates 135 interventions plus three warm references, on the full existing 512-article validation role. It starts from each seed's existing v4 no-Sleep step16/112, pre-P4 step16/112 and post-P4 step112 checkpoints. No optimizer updates occur. The original warm receiver's tensors remain unchanged. Restoring bases and selecting that receiver can recover old accuracy, while restoring either component alone is inconsistent across seeds. This establishes functional recoverability under interventions, not a successful deployed old/new learner or a unique training-time cause.

The route audit counts the two answer positions per example. Under the actual P4 policy, natural selection favors the latest slot (4 at step16, 28 at step112), alongside the explicit latest-slot 0.5 mixture. `force` overrides its gate to one and changes amplitude; it is not a pure identity switch. Natural gates contain a null option, even with one active expert. The first probe incorrectly assumed `force0` reconstructed the original warm mixture; it stopped after nine scored states, with zero optimization. Its source, failure, plan and all metered work are retained. The corrected positive control restores both bases, the warm active topology and original execution policy; all 15 controls match warm-reference EM exactly and answer NLL within 1e-5. A separate `mix0` intervention keeps the mixture coefficient at 0.5.

The first-16-step interference study adds 48 ordinary wake updates, 66 diagnostic old/new gradient forward/backward passes, fixed 128-article validation observations and balanced 24-prompt training candidate checks. It exactly reproduces all three v4 step16 model/controller/policy/RNG states. The validation subset is the first32/class (64 old,64 new), distinct from the full512 scoring above. At the warm start, fixed old/new train-batch CE gradients oppose one another in both base groups (cosines about -0.43 to -0.89). These are local, labeled diagnostics, not a proof of a global conflict at every update. A preliminary diagnostic stopped before any wake update because of a counter-closure bug; its 128-example observation and four gradient passes are retained and charged separately. No failed seed was replaced.

Together these observations motivate two interventions: constrain changing wake parameters toward a competent old teacher, and explicitly preserve a warm-receiver contribution during wake. They do not justify claiming that receiver weights were erased, or that restoring an old function without learning the new task is sufficient.

## Fixed factorial and controls

Seeds17/29/43 use the original v4 qualified transferred checkpoints. Every arm starts from exactly the same model, routing policy and RNG within a seed. The backbone/tokenizer/role pack and rank16, capacity32 last-MLP extension are unchanged; weights were originally obtained from ModelScope. No new model or dataset is downloaded.

The fixed design has seven arms: teacher-anchored wake off/on crossed with no Sleep, P16 Sleep retaining the latest mixture, and P16 Sleep switching wake mixture to original slot0; plus a joint old-gold-CE control without Sleep. Each arm has 448 primary new-task B4 updates from the same v3 schedule prefix, base/expert learning rates0.02/0.04, CMS periods4/16 and gradient clip1. Sleep occurs every16 primary updates, giving28 events and29 active slow slots including the warm receiver. Base matrices occupy no expert slot. The P16 choice permits a longer adaptation budget within the unchanged capacity; comparisons with the older P4/112-step study are not isolated cadence ablations.

All auxiliary methods use the same fixed old-training B4 schedule (two examples per old class), chosen by a per-seed/per-step deterministic generator. The private pack contains256 training and128 validation articles per class. Schedule hashes/seeds, rather than raw token packs or article text, are published.

### Anchored wake

New-task CE is the unchanged mean of shared-marker CE and class-token CE. The old auxiliary term has fixed weight1; its weight is not searched. Both gradients accumulate before one global clip and one `CMSController.collect`, so a primary step advances each CMS clock once. Extra old-context backward work is charged separately rather than called another primary step.

For soft KD, the immutable teacher is a shared-backbone clone of that seed's qualified warm checkpoint. Its raw old prefix supplies a greedy first answer token, then full-vocabulary teacher distributions at the first and second answer positions under that generated context. The student receives the same context and minimizes mean KL(teacher||student) over the two positions, temperature1. Old marker/class answer fields do not enter this objective. However, class metadata is used to balance the replay reservoir: this is not a wholly unlabeled or replay-free method. It adds two teacher forwards and one old-context student forward/backward per primary step, plus the retained teacher adapter state.

The gold control replaces old KL with old marker/class CE, while keeping the same old prefixes, primary schedule, single collect and clipping. It adds direct old-answer supervision and one old-context student forward/backward; it has no corresponding teacher forwards. It therefore does not have equal FLOPs to soft KD. Its outcome cannot establish that Sleep is effective.

### Sleep and wake routing

All Sleep arms use the existing 24-update recipe:16 GKD (eight teacher, eight on-policy student) and eight LTI, two generated tokens, learning rate0.003, no Dream. Each event's training-prefix pool is balanced6/class, with8 old/8 new GKD and4 old/4 new LTI prompts. Every method shares that seed/boundary's pool and event seed. Gold targets are not used in Sleep; they appear only in wake CE, the explicit gold control and diagnostics.

The transaction collects the primary gradient with both due proposals deferred, applies the slow proposal to a private candidate, clones its current self-teacher, then gives the student the prospective fast base and resets fast experts while adding a receiver. The frozen warm teacher is used by anchored wake; it is not substituted for the current Sleep self-teacher. Original slow receiver weights stay frozen during ordinary wake and during training of newer Sleep slots.

Latest-policy arms keep the committed student's newest-slot 0.5 mixture. Warm-mixture arms switch the wake policy after commit to `{mix:0,fraction:0.5}`. Their Sleep optimization still guarantees a 0.5 contribution from the new receiver. This deliberately changes training-to-wake coefficients; it is a controlled policy intervention with possible adaptation costs, not a silent routing bug fix. Both the training and post-commit policies are recorded. Later trajectories and self-teachers may consequently diverge.

## Correctness, endpoints and costs

A real3B prerequisite runs14 anchored updates, saves pending CMS counts `[2,14]`, then independently restores twice and crosses steps15/16 with Sleep and the warm wake policy. Model, controller, execution policy and global RNG hashes must agree exactly, and the saved snapshot must remain unchanged. This costs18 wake updates and48 Sleep Adam updates, without any performance scoring. The helper and qualification-source/result hashes are required by the main configuration.

Fixed validation/private-checkpoint steps are0/16/64/112/256/448. Every arm reports the full512 validation role; there are21 runs and126 validation states. The reference arm additionally must exactly reproduce v4's model/controller/policy/RNG and full validation metrics at16/112. At448, all1,024 training records are scored as well (512 old,512 new). No metric-based early stop, best checkpoint, seed replacement or heldout tuning is permitted.

Primary endpoints are old/new/all full-vocabulary two-token greedy EM at448, old drop from own start, per-class scores and answer NLL. Means/sample SD and matched-seed differences are descriptive, with no p-values. The pre-existing acquisition thresholds are applied to each task separately: train EM≥.80, validation EM≥.75, and each task class validation EM≥.70. Report every pass/fail; a high old score with poor new learning is not joint success. All smaller-budget failures and collapsed trajectories remain visible.

Predefined comparisons cover KD versus no KD within each policy, each Sleep policy versus its matched no-Sleep wake objective, warm versus latest policy under each wake objective, and Gold versus KD/reference without Sleep. The factorial can assess the combined engineering interventions; it does not isolate every paper mechanism or establish external confirmation.

The main block plans9,408 primary wake updates,8,064 Sleep Adam updates,4,032 KD old-context steps and1,344 gold old-context steps. Those draw16,128 and5,376 old prefixes respectively. They add auxiliary backward passes, teacher inference, replay storage and sometimes new expert capacity. Metered forward calls/input-token proxies, gradient-forward tokens and wall time are reported per arm; they are not FLOPs, energy or GPU busy time. Diagnostic failures, successful probes and recovery qualification are separate additional costs, counted once rather than repeated into each arm.

The main cooperative limit is7,200 seconds, checked between operations; it cannot interrupt a running kernel. Numerical/guard failure stops the block and publishes partial results, without retrying a failing method or replacing its seed. Incomplete Sleep events have their unknown0–24-update bounds recorded. Any infrastructure-only continuation would require exact source/checkpoint/controller/RNG binding and a separate duplicate-work ledger; none is assumed automatically.

## Scope and publication

This independently written variant is inspired by the Sleep paper's consolidation ideas, but anchored wake is an added functional-replay regularizer. The two-level fixed-router last-MLP architecture, arbitrary two-token class codes, exact-output LTI reward and short AG News stream do not reproduce the complete Sleep/HOPE architecture or benchmark. No upstream HOPE source is copied.

The report preserves all seeds, negative results, code/config hashes, fixed endpoints, extra supervision/compute, correction history and limitations. Only code, aggregate metrics, manifests and figures are published. Articles, token arrays, teacher-generated sequences, weights and private checkpoints remain local. Offline aggregate reanalysis is supported; exact training reproduction additionally requires the legally held local model, private role pack and original qualified snapshots. Any future confirmatory use of the reserved within-train data requires a method and evaluation plan frozen before opening it, and remains distinct from an external or pretraining-independent test.

## Completed execution

The frozen comparison finished all21 runs without training retry. [Full results, extra routing/capacity control and limitations](../reports/agnews-anchored-v5.md) report every fixed endpoint; no joint old/new gate passed. The separate task-router control was specified after main completion and is exploratory, not part of the original seven-arm frozen protocol.
