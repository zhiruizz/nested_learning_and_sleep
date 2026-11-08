# Sleep V13 primary-source audit

**Scope.** Read-only check of the v2 paper, its 26-page PDF/appendix, the authors’ publication listing, and the official Google Research post. No training, datasets, model downloads, or code execution. The paper is Ali Behrouz, Farnoosh Hashemi, Adel Javanmard, and Vahab Mirrokni, “Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories,” arXiv:2606.03979v2 (10 July 2026). This is the Sleep/Knowledge-Seeding paper, not the separately titled “Language Models Need Sleep” by Lee et al. (arXiv:2605.26099).

## What the primary sources establish

| Topic | Evidence in the paper | What it does and does not establish |
|---|---|---|
| Wake/Sleep scheduling | Section 2.2 (PDF p. 5) defines a module’s frequency as updates per unit time and takes one update of the slowest module as the time unit. Section 3.2 (p. 7) says consolidation occurs at chunk-length multiples; Section 3.3 (pp. 7–8) says Sleep is triggered when past steps are divisible by the receiving block’s chunk length and runs before the sender’s scheduled base update. Figure 7 (p. 22) illustrates 1K→5K→10K levels. | Sleep is a boundary in a hierarchy of update clocks, not a stated wall-clock interval. The paper does not identify a request batch size or say that a “step” always equals one user example, one token, or one optimizer step across the experiments. |
| Wake batch and sleep batch | Appendix B.4, Table 5 (PDF p. 26), lists effective batch size 32 for GRPO, SFT, and Sleep, with training steps 500, 100, and 100 respectively. | These are the table’s experiment-level GRPO/SFT/Sleep settings. The table does not say that every Knowledge-Seeding event uses 32 replay requests, nor does it define separate wake and Sleep microbatch sizes. The CLINC/Banking class-incremental subsection does not give task-specific batches. |
| Replay examples / optimizer update | Section 3.3 (PDF pp. 8–9) constructs a dataset by sampling from the teacher; LTI randomly samples a prefix from each teacher-generated sequence and asks the student to complete it. | No fixed number of teacher samples, replay requests per optimizer update, LTI prefixes per update, accumulation rule, or optimizer-step count per Sleep boundary is stated for Knowledge Seeding. “Replay” here should not be translated into a guessed fixed request count. |
| LR, steps, rank, LoRA alpha | Appendix B.4, Table 5 (PDF p. 26): LR `5×10⁻⁶`; effective batch 32; Sleep steps 100 (GRPO 500, SFT 100); LoRA rank 64; LoRA alpha 128. | These are the published Table 5 values. The paper does not identify a separate LR/step/rank/alpha recipe for each continual-classification task or for each Sleep event. LoRA alpha=128 is a LoRA setting; it is not the `α` coefficient in Eq. 5. |
| Additional memory structure | The opening paragraph of Appendix B (PDF p. 25) describes five additional MLP blocks with dimension 64 and states that active parameter count stays at the base-model count. | Our existing five memory levels and rank-64 experts share two numerical choices, but this does not establish identical MLP placement, routing, parameterization, or active parameter count. In particular, a rank-64 bottleneck is not automatically the paper's fully specified architecture; no executable author recipe was found to resolve this. |
| GKD mixture / on-policy policy | Section 3.3, Eq. 1 (PDF p. 8) uses teacher-generated samples and a student-generated on-policy term; `λ∈[0,1]` controls the on-policy fraction. The text says sampling is not backpropagated through and only expanded parameters are updated. | The source establishes a mixture of teacher/off-policy and student/on-policy sequences, but the inspected paper/appendix does not provide a numeric `λ` for the class-incremental experiments or a per-boundary schedule. It does not disclose rollout count or generation batch. |
| LTI reward and combined objective | Section 3.3, Eqs. 3–5 (PDF p. 9): a frozen reward model gives binary semantic reward; token similarity uses Levenshtein distance normalized by the maximum sequence length and is zero when distance exceeds threshold `z₀`; `γ` mixes semantic and absolute similarity. Eq. 5 combines reward and divergence with `α∈[0,1]`. | The objective structure is explicit. The values of `γ`, `z₀`, and objective `α`, the reward-model identity, divergence choice/temperature, and exact optimizer configuration for this component are not established by the inspected paper text/table. Do not substitute Table 5’s LoRA alpha for objective α. |
| Classification task | Section 4.1 (PDF p. 11) evaluates the memory-consolidation phase (without Dreaming) for class-incremental CLINC, Banking, and DBpedia; names Llama-3B and Llama3-8B backbones; compares ICL, EWC, InCA, and Hope. Appendix B.1 (p. 25) describes CLINC150 as 150 in-scope intents across 10 domains and 23.7K total queries (22.5K in-scope, 1.2K OOS); Banking77 as 77 intents. | The task family and benchmark sizes are explicit. Exact class order, task partition/count, example selection, prompt/label serialization, OOS treatment in the plotted CLINC result, per-task steps, and decoding/accuracy aggregation are not specified in these sections. Section 4.1 calls the ICL baseline the same continual pre-training process without Sleep, but gives no full recipe here. |

## Source and code provenance

- The arXiv v2 HTML has Sections 2.2, 3.2–3.3, 4.1 and Appendix B.1/B.4. Its linked PDF is the same v2, 26 pages; Table 5 is PDF p. 26 and Figure 7 is PDF p. 22. [arXiv v2 HTML](https://arxiv.org/html/2606.03979v2) · [arXiv v2 PDF](https://arxiv.org/pdf/2606.03979v2).
- The authors’ [publication page](https://alibehrouz.com/publications/) links this paper to arXiv, but lists no code link for it. The [Google Research post](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/) is by Ali Behrouz and Vahab Mirrokni and adds a high-level explanation, but exposes no GitHub link or executable recipe. In the inspected primary author/Google pages and paper, I found no author-linked implementation repository or immutable code commit for this paper. This is a bounded search finding, not proof that no private or later code exists.
- Search results surfaced [`sangyun884/llm-sleep`](https://github.com/sangyun884/llm-sleep), whose README says it is for **“Do Language Models Need Sleep?”** This is the distinct Lee–McLeish–Goldstein–Fanti paper (arXiv:2605.26099) on offline recurrent fast-weight consolidation, not the Behrouz et al. Sleep/Knowledge-Seeding paper. It is third-party relative to the paper at issue and must not be used to infer that paper’s batch, rank, reward, or replay settings.

## Practical replication boundary

For a close implementation, the primary source supports the structural ordering (fast sender update is pending; consolidate before applying it; expand and train the new lower-frequency parameters; then apply the sender update and reset its transient low-rank additions), the frequency-clock interpretation, the GKD mixture form, and the LTI reward form. It does **not** pin enough per-task operational values to reconstruct exact class-incremental wake/Sleep request batches, replay counts, task order, objective coefficients, or optimizer steps per Sleep event. A local implementation that selects those values must document them as implementation choices, not as paper-specified facts. Table 5 can be reported as a published training configuration whose task scope is unspecified, but should not be relabeled as the class-incremental replay batch recipe.

Appendix B.4 is headed “Additional Details.” Table 5 lists configurations by
GRPO, SFT and Sleep, without naming a dataset or saying the values cover all
experiments. It follows B.3's ARC discussion and precedes B.5's efficiency
discussion of AIME-24, AIME-25 and HMMT-25; this placement does not establish
an explicit mapping to those reasoning tasks or to continual classification.

Equation 4 compares the **raw** Levenshtein distance with `z0`; when it passes,
the similarity is `1 - distance / max(sequence lengths)`. The local V10/V13
edit component has this ordering. Its `z0=8` remains a local numerical choice,
and the canonical-label semantic component remains a proxy for the unspecified
frozen reward model. This formula check does not justify changing an already
frozen experiment or treating its reward as the author's model.

The cited [InCA publication](https://aclanthology.org/2025.coling-main.487.pdf),
Section 5/Table 1, reports a seven-task Banking baseline averaged over three
task splits. Its fine-tuning baselines use batch 8, Adam, LR `2e-5` and at most
10 epochs per task, while InCA itself uses an external learner without that
training. Those are different methods' settings, not missing Sleep settings
that can be filled in automatically. Neither that citation nor Sleep's general
reference to original benchmarks establishes the exact Sleep class order,
prompt format or per-boundary replay/update schedule. The two-boundary V13
development diagnostic is not the full-task final-accuracy endpoint of that
benchmark.
