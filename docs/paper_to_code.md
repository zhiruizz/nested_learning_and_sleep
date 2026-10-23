# Paper / code / test / result mapping

Scientific reference versions: NL arXiv:2512.24695v1 and Sleep arXiv:2606.03979v2.
This is a mechanism index, not a claim of complete fidelity.

| Mechanism | Paper / category | Code | Tests / evidence | Boundary |
|---|---|---|---|---|
| Multirate CMS | NL §7–8, Eqs 70/71/73; specified concept | clocks.GradientWindow; optim.CMSController; native.FunctionalCMS | test_reference clock tests; test_native CMS meta-gradient | exact nonoverlapping C-event windows and minibatch clock are choices; SGD controller is a local variant |
| Deep self-modification | NL §8.1, Eqs 84–96; partial interpretation | native.deep_partials, feature_dgd_step, DeepSelfModifyingTitans | independent autograd, NumPy finite differences, outer gradchecks | feature-wise retention, query/gates, norm and chunk rules are assumptions |
| External fast state | NL §8; implemented | native.NativeHOPELM | streaming, padding, stream identity, causal prefix tests | serial small model; no full pretraining recipe or efficient dual kernel |
| Low-rank expansion | Sleep §3.2; implemented local choice | experts.SparseExperts; lm.AugmentedMLP | reset lifecycle and routing counterexamples | fixed random content router plus priority; not uniquely paper-specified |
| Llama adaptation | extra implementation assumption | lm.LlamaCMSLM | static CLI/parameter tests only this release | only last MLP output, two residual CMS levels; not native HOPE; NOT_RUN |
| Teacher/student/reset | Sleep §3.3; partial implementation | consolidation.ConsolidationTransaction; lifecycle.SleepSession | storage isolation, frozen mutation rejection, receiver-only update | reset retains A, zeros B and deactivates; commit is an in-memory operation |
| KS distribution kernels | Sleep §3.3; implemented, efficacy unverified | objectives.token_forward_kl; SleepSession.teacher_forced_step/seed_step | detached teacher, mask tests; fixed synthetic sequence update | live teacher/student sampling code retained but NOT_RUN; historical fixed banks are a separate diagnostic |
| LTI / Dreaming | Sleep §3.3–3.4; partial | objectives, dreaming, lifecycle | reward direction, isolated trials in reference tests | exact-output semantic proxy, readout-only trial LoRA; no release sampling or full policy validation |
| Reachability / conditional KL floor | additional diagnostic, not paper method | repairability.Assumptions/classify_position/bounds | premise rejection and reachable-zero-gradient counterexamples | fixed tokens/targets/weights, receiver-only domain, no downstream mixing; caller assertions are not an automatic graph proof |
| CACHE | additional engineering optimization | executor_v13.FrozenFrontLRU; execution.select_executor | import/static and missing-acceptance rejection; historical timing CSV | REF default, public accelerated execution NOT_RUN; no transferable private acceptance |
| Slowest/terminal lifecycle | missing/unverified | SleepSession rejects terminal sender | terminal rejection test | true 64 boundary and simultaneous-due resolution unresolved |

Current CPU checks establish selected numerical and state behavior. They do not
establish architecture equivalence, benchmark reproduction, successful long-horizon
continual learning, or independently replay the historical experiments.
