# CLINC150 V8: lifecycle and evaluation protocol

This is a local, benchmark-adapted continual-learning study. It is not a faithful
reproduction of the Sleep paper's HOPE architecture or its published accuracies.
Development attempts, including invalid metrics and failed transactions, remain
in the evidence ledger. No accuracy is inferred from an engineering qualification.

## Assets and isolation

The backbone is **Llama-3.2-3B base**, not Instruct, acquired from ModelScope
`LLM-Research/Llama-3.2-3B` at revision
`48d84dd14886f01964f72e7f742a404dd86f9fb6`. Its original parameters are frozen;
the model's license and acceptable-use policy apply separately from this code.
Weights are local and are not redistributed.

The data source is [clinc/clinc_oos, plus, pinned revision](https://huggingface.co/datasets/clinc/clinc_oos/tree/155b9c710419136e17307b80d0a13e68cd46b4ec/plus).
The [earlier dataset loader](https://huggingface.co/datasets/clinc/clinc_oos/blob/5f6919aa8ae58804944da8f795dde95c7cfcc39a/clinc_oos.py)
maps `plus` to `data_oos_plus.json.gz` from the
[original CLINC repository](https://github.com/clinc/oos-eval). Plus is the OOS+
variant, not the original default configuration. OOS rows are excluded here.
The pinned source identifies CC BY 3.0; its license is not replaced by this
repository's documentation license.

Only train and validation shards have been acquired for development:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| train | 312096 | `30188119cf9f86fc9db27e1c22442d091cb5cb0913c9496f945fe11e7a02a28f` |
| validation | 77789 | `fbd545b46c611c4a7ba4b48cae6c7f09bb5b59f33ff56206ad1cd366c85cdfaa` |
| test, metadata only | 135844 | `3e60e45b25bf86543aa5df8ba4fcc674114164e6184f0197690648c2908d0102` |

After excluding 250/100 OOS rows, the training/development pack has 15000/3000
rows: 150 classes, 100/20 rows per class. Three development rows overlap training
after the locked normalization (NFKC, case-folding and whitespace/underscore
normalization). Clean development excludes these rows, leaving 2997. Task 0 and
task 1 of seed 17 each have 300 clean development rows.

ModelScope's two suggested `c41/deepopen-clinc150-*-v1` resources are trained
classification models, not raw dataset releases. Their cards describe CLINC
supervision; they are not used as initializers or teachers. Recorded read-only
ModelScope dataset queries returned no matches for the queried CLINC names.
This is evidence about those queries, not proof that no mirror exists anywhere.

No P2/Phase confirmation data is involved. The CLINC test shard remains unopened
until a complete, source-bound final training seal exists. Published aggregate
statistics and historic label annotations on public cards were visible during
source verification; this public benchmark is not claimed to be wholly unseen.
Repeatedly inspected development results are exploratory.

## Classification and task definition

Class names are canonicalized and sorted once. Seeds 17, 29 and 43 independently
permute the 150 class IDs; ten disjoint tasks each introduce 15 classes. There
are no task IDs, current-task candidate masks or label-dependent prompt prefixes
at inference. Tokenization reserves the global maximum label length before
truncating the request text, so the prompt cannot reveal the gold label length.
Training uses teacher-forced full-vocabulary cross-entropy on the complete label
and EOS. Evaluation greedily generates up to eight tokens and maps the result to
the fixed global 150-label vocabulary. Invalid generations count as errors.

Report accuracy, per-class macro accuracy, denominators, invalid generations,
and separate old/current/seen cohorts. During development, evaluate only already
introduced classes. A task-end seen accuracy is not final all-150-class accuracy
until task 9. A first-task clean development accuracy of at least 0.50 is the
declared acquisition qualification for interpreting subsequent retention.

The full official in-scope split is not InCA's reported 10000/750 subsample.
The Sleep paper does not disclose enough sample IDs, checkpoint details or
classification update schedules to equate these local choices with its run.

## Implemented mechanism

Five rank-64 residual memory modules augment the last MLP of the frozen Llama
backbone. This is the existing CMS-inspired adapter, not a native HOPE FFN.
Each level has a fixed allocation of eight low-rank expert slots. The rank-2
readout is frozen. Mutable matrices use FP32; the backbone uses BF16.

Development periods are **16, 64, 256, 1024 minibatches, terminal unclocked**.
These are explicit local assumptions. They are not a transcription of the
paper's 512/2k/8k long-context ablation, whose units and applicability to CLINC
cannot be established from the text. All minibatch gradients are clipped at
global norm 1. A fast expert takes SGD updates at LR 0.1; each nonterminal base
accumulates LR-weighted gradients (LR 0.001) and applies its absolute prospective
weights only at its boundary. Pending windows cross task boundaries intact.

A Sleep transaction freezes the true pre-base-update teacher, constructs the
prospective student, resets only the sender level's active experts (redraw A,
zero B), and activates a fresh slot in the next level. Only that slot's A/B are
trainable. Simultaneously due levels are processed from fast to slow; proposals
are reacquired after every commit, so a slower teacher includes earlier accepted
commits. Terminal memory has no own update clock.

The initial budget is two full-vocabulary GKD updates and one LTI update per
event, AdamW LR 0.001, weight decay 0, norm clip 1, sampling temperature 0.8.
LTI uses a declared local teacher/student label-text reward proxy and edit
distance, without the paper's frozen semantic reward model. No Dreaming is used.
Inputs are bounded old/current replay requests (four per seen class). Class
metadata selects balanced replay; absence of gold targets from the Sleep loss
does not make the complete replay-selection system label-free.

Natural routing can fail to execute the fresh trainable receiver at all. A
strict loss/gradient guard rejects such a transaction; no dummy zero-valued
parameter term is added. The separate **controlled-mix** ablation sets
`0.5 * natural_output + 0.5 * fresh_expert_output`, during training and deployment.
It ensures a receiver path and can execute an extra expert branch per affected
bank. It is explicitly a local intervention, not evidence that the original
paper uses this policy or that active computation is unchanged.

A gold-CE controlled arm uses the same capacity, route policy and number of
receiver optimizer updates. It consumes replay labels and is update/capacity
matched, **not** FLOP-, data-supervision- or token-matched to sampled distillation.
Wall time, forward shapes, padding, backward calls, physical optimizer calls,
resident/stored parameters and analytical covered operation bounds are separate
reported costs. Operation estimates exclude operators listed in each receipt;
they are not measurements of complete hardware FLOPs.

The frozen R1/R2 and two-task controlled probe retain the historical edit proxy
`max(0, 1 - token_distance / z0)`. Paper Equation 4 instead divides by the maximum
sequence length when the distance is within `z0`, and returns zero outside that
threshold. This is an additional objective difference, not a harmless naming
change. The proposed full-run implementation must correct that explicit formula
in a separately versioned objective before freezing production; a proxy run
cannot retrospectively be relabeled as that corrected protocol. Canonical-label
equivalence still substitutes for the paper's frozen semantic reward model and
must remain disclosed even after correcting the edit term.

## Correctness and checkpoint requirements

Full-model and independently recaptured-tail logits and gradients must agree.
Autoregressive suffixes require a fresh backbone capture at every prefix;
stale KV states or one cached prompt state are not used for generated suffixes.
Sleep checkpoints own teacher/student buffers, nonempty Adam moments, pending
clocks/gradients, route policies and all relevant RNG streams. Disk roundtrip
and three-step continuation must match an uninterrupted branch exactly before
running the corresponding mechanism on the dataset.

Checkpoints and data stay outside Git. Every run has immutable configuration,
source hashes and model/data provenance, bounded budgets, a fresh output
directory and an explicit terminal or failure record. Rejected update attempts
are distinguished from completed physical optimizer calls. Engineering replay
steps are counted as compute, never as independent experimental seeds.

The R1 evaluation implementation incorrectly associated length-grouped outputs
with original-order labels. **All R1 reported accuracies are invalid.** R2
restores prediction order and has a regression fixture that exposes the R1
failure. A competent R1 HOPE checkpoint may be reused only after corrected,
source-bound development rescoring; that reuse does not validate the old metrics.

## Relation to the papers

The [Sleep paper](https://arxiv.org/html/2606.03979v2) motivates upward
consolidation, parameter expansion, and CLINC/BANKING/DBpedia accuracy studies.
Its continual-learning results use Llama 3B/8B; 512/2k/8k is a separate
long-context frequency ablation. Architecture, optimizer schedules, data sample
IDs, routing, reward implementation and omitted Dreaming limit correspondence.
The [InCA paper](https://aclanthology.org/2025.coling-main.487.pdf) informs task
protocol comparisons; it is not evidence that Sleep used every InCA choice.
This repository separates mechanisms that are implemented and verified from
paper results that it has not reproduced.
