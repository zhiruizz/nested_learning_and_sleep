 # Partial implementation and diagnostic observations

 The repository separates reference CMS/Sleep mechanisms, an assumed native model,
 and a frozen-backbone last-MLP Llama CMS-inspired extension. No full HOPE architecture
 or paper benchmark reproduction is claimed. This release adds packaging and
 independent CPU checks, not scientific training, seed selection or gate changes.

 In the original five-arm pilot, native final mean accuracy was25% in every arm
 (four trajectories each). Llama arm means were frozen38.89%, sequential/CMS41.67%,
 and KS/KS+Dream16.67% (one trajectory each). These exploratory outcomes did not
 establish adequate native acquisition or an independent Dreaming gain. A zero
 receiver-gradient cycle and lack of positive real Dreaming reward remain limitations.

An earlier native version's20 trajectories are retained in the private history as
superseded diagnostics:2/32 Sleep prefixes included left padding. The later complete
20-trajectory rerun used the unchanged protocol; no best seed was selected. Release
preparation did not rerun either version. In the later online18-path comparison
(six events x three methods), two CMS+KS paths stopped at tick31/56 with failures;
those failures are not successes or silently removed completed endpoints.

 Transition diagnostics separate prospective CMS-base writes (U), sender reset (R)
 and receiver activation (E). Natural routing and diagnostic inclusion are measured
 separately. An appended receiver can be accessible during distillation yet absent
 at an answer position; a gate-zero intervention is not reset. The selected complete
 transition/routing tables accompany this report without raw examples.

 The fixed-bank reset experiment has two teacher events, four trained paths and
 four scoring steps, all retained. Event83 with reset changed new dev0/32 to18/32,
 while old dev58/64 fell to54/64; its teacher had old38/64 and new29/32. Event97
 with reset recovered old38/64 to60/64, still below teacher62/64, and new stayed0/32,
 also0/32 in its teacher. These are local recoveries with tradeoffs, not joint
 lossless retention or new-task acquisition. R0 event83 stayed old56/64,new13/32.

 The later fixed-trajectory batch retained116 registered conditions:108 new256-step
 paths,4 reused paths and4 structurally unavailable first-position conditions.
 These arise from six saved teacher events, not116 independent samples. No universal
 old/new solution was obtained; finite optimization does not establish impossibility.
 Only this bounded aggregate inventory is included, not the complete per-condition
 archive, so its training outcomes cannot be independently reconstructed here.

 Under fixed tokens/targets/weights and receiver-only updates, absence of a dependency
 at position i gives P_phi(i)=P_phi0(i). Nonnegativity on the remaining positions
 yields L(phi)>=sum(i in certified U) w_i KL(Q_i||P_phi0_i). This is conditional
 graph reasoning, not a zero-gradient criterion, accuracy floor or global capacity
 theorem. Caller-asserted premises in the CPU helper must be proved for each model.
 The synthetic counterexample distinguishes unreachable outputs from reachable
 parameters with zero A-gradient at B=0, and rejects changed-token premises.

 Phase14B's six fixed-endpoint teacher preparations all failed the complete gate.
 Car seed311 matched its majority baseline; both Balance histories never predicted
 the minority C class. Seed/order effects were confounded. Conditional structure
 checks found empty inaccessible sets in four histories,48/96 positions for Car311,
 and24/96 for Balance401 (ALL4). Empty-set invariance is vacuous. Car311's saved
 answer-next FP32 estimate was -2.528190217e-9; normalized same-logit FP64 estimate
 was1.101112269e-9. These rounding-scale numbers do not establish a practically
 important strict positive error floor. Full distributions are not redistributed;
 these structure measurements are historical descriptions, not public recomputations.

 Phase13 CACHE preserved the variable CMS/readout graph while caching the frozen
 prefix and original last-MLP output. Ten numerical/state cases per executor and
512 reference steps were historically checked. On one repeated small bank, all
 three CACHE/REF full-path pairs improved, with reciprocal median ratio about3.398x.
 Timing includes cold builds, fit/dev checks and checkpoint/flush; loading is separate.
 HOST numerical checks passed but its timing adoption rule failed. BOTH had only
 two complete pairs, with one interrupted path and unknown in-flight outcome.
 Earlier KV and logits_to_keep candidates failed; none is adopted here. Public
 CACHE execution is NOT_RUN and REF remains the reliable entry default.

 The original Llama stays frozen; base refers to CMS base. True64, simultaneous
 expiration and terminal lifecycle are unresolved. Reused dev, teacher banks and
 shared model ancestry restrict independence. Unavailable snapshots and banks mean
 historical exact replay is not possible from this package. Failures concern this
 implementation/configuration, not the validity of the original papers.
