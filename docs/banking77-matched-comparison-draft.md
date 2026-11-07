# Banking77 comparison preparation (not a frozen experiment)

Status on 2026-11-07: source metadata audited; no Banking77 train/test files
have been acquired. No Banking77 model run or accuracy result is claimed.

## Source decision and acquisition gate

The preference is a verifiable ModelScope source. The
[SPACE-format candidate](https://www.modelscope.cn/datasets/modelscope/banking77/summary)
advertises 8,632 train / 1,544 validation / 3,084 test rows. This differs from the
publisher's 10,003 train / 3,080 test rows by 177 total rows; its provenance,
revision, conversion manifest and true payload size are not established. Its
2.04 KB page display is not a dataset-download estimate. The newer
[MTEB candidate](https://modelscope.cn/datasets/mteb/banking77) is a stronger hub
lead, but its exact revision and file inventory remain unavailable. Separately,
the [MTEB HF metadata snapshot](https://huggingface.co/datasets/mteb/banking77/tree/18072d2685ea682290f7b8924d94c62acc19c0b2)
reports 9,993 train / 3,076 test rows, rather than PolyAI's original counts.
Its processed variant and card license metadata do not establish that ModelScope
hosts the same bytes or change the original dataset's attribution requirements.
Matching a dataset name or search-index total does not verify a mirror.

The reference source is
[PolyAI's official repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data),
with [publisher-named metadata on Hugging Face](https://huggingface.co/datasets/PolyAI/banking77/tree/96a234bd25c04939c4a79213eb764ae90e4d0d81).
It has 77 intents, no native validation split, and CC BY 4.0 licensing.
Metadata reports 839,073 train CSV bytes and 239,961 test CSV bytes
(1,079,034 total); the corresponding parsed copy is estimated at 919,050 bytes.
These are metadata-reported sizes/checksums, not independent verification of
downloaded publisher bytes. Reserve 25 MB for raw files, derived JSONL, manifests,
and temporary/cache overhead. This excludes model weights and checkpoints.

If Hugging Face data acquisition is needed, the parent must first relay the
dataset name, exact links and capacity to the user. This document and metadata
lookups do not fulfill that relay. No alternative host should be used to evade
the requested notification. Prefer train-only acquisition first; keep the
official test payload unavailable to tuning/training until the complete
predeclared main grid is sealed.

## Proposed matching rules

These are project assumptions to freeze after source verification and CLINC
development diagnostics, not settings recovered from the Sleep paper.

- Use the same legally acquired Llama-3.2-3B base assets for every arm and record
  all weight/tokenizer/config hashes. Keep the same prompt, label target,
  normalization, initialization, task order and primary wake presentations.
- A seven-task, eleven-class-per-task design is an InCA-like candidate: 77 is
  divisible by seven. The Sleep paper does not establish this split or its IDs.
  Pair class-order seeds 17/29/43; distinguish these from independent adapter
  initializations. Do not silently rename this a ten-task reproduction.
- Derive validation from publisher training only. Freeze a normalized-text
  group-aware, class-stratified split before model scoring; quarantine conflicting
  duplicate labels and record any removed/grouped counts. Choose the split policy
  from training metadata, not test comparisons. No native validation is claimed.
- Preserve source label-to-ID mapping. Version case/Unicode/underscore handling
  and punctuation consistently; verify that normalized canonical labels remain
  unique. Do not reinterpret the publisher's punctuation-bearing intent names as
  an evaluation bug.
- The local tokenizer metadata audit found longest Banking77 label length nine,
  requiring ten tokens with EOS. One label exceeds the old eight-token Sleep
  rollout cap and four exceed it including EOS. Derive a dataset-specific cap
  of at least longest-label-plus-EOS before training; an unchanged cap of eight
  cannot represent every complete target. This is reachability, not an accuracy
  experiment. The same audit found all CLINC labels fit in seven tokens with EOS.
- Qualify all shared foundations under an explicit acquisition gate and verify
  exact disk recovery, RNG/pending-window preservation and nonzero receiver
  gradients before main continuations. Failures remain results; no automatic
  retry, seed substitution or post-test checkpoint selection.

## Controls and measurements to freeze

Primary comparison: identical local no-Sleep CMS versus consolidation, with the
same primary supervised wake budget and shared per-seed foundation. Natural
routing must pass its receiver/gradient qualification. A persistent fixed mix
is a separately named local intervention; failure of natural routing does not
authorize relabeling the fixed-mix arm as native Hope.

A supervised old-text replay control can test forgetting under the same replay
pool and primary wake schedule. It uses additional gold targets and must be
marked as extra supervision. Match and disclose replay samples/answer-token
exposure and actual work; equal optimizer-call counts alone are not equal FLOPs.
Do not infer a causal Sleep advantage from unmatched additional training.

Report per-task acquisition and final global 77-class accuracy without task-ID
masks. Free generation and all-label likelihood ranking are different inference
protocols: choose a primary before the main grid; if both are retained, report
both with their extra label-dictionary access and inference costs. Report
old/current/seen denominators, invalid generations, per-class/macro accuracy for
this imbalanced corpus, and final fixed-checkpoint results for all seeds.
Final old-class accuracy alone is not a forgetting rate unless prior acquisition
has been established. Preserve negative results and every operational retry.

Exact learning rates, epochs, update periods, receiver policy, replay size and
training-work bounds remain unfrozen. They must be justified with train/development
evidence, including the CLINC readout diagnosis, rather than copied from unrelated
reasoning or long-context settings. Record physical updates, full-backbone/tail
calls, token counts, elapsed time, parameter/storage costs and partial-FLOP exclusions.

## Scope

[Sleep v2](https://arxiv.org/html/2606.03979v2) and
[InCA](https://aclanthology.org/2025.coling-main.487.pdf) motivate this preparation,
but neither source establishes that the local model, architecture, split, prompt,
update budget or scoring protocol matches the Sleep classification runs. A new
Banking experiment remains exploratory until its own protocol and held-out
evaluation are frozen; exposed CLINC test results are not fresh confirmation.
