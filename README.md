# NL/Sleep Reproduction Lab

An unofficial, partial implementation and diagnostic study of Nested Learning and
Language Models Need Sleep. This repository is not an official release and does
not reproduce all architectures or reported benchmarks.

This release contains small reference mechanisms, an explicitly
assumed native HOPE variant, a last-MLP Llama CMS-inspired extension, and historical
diagnostic tables. It makes no institutional endorsement or maintenance promise.

Python 3.11 is the tested interpreter. Create a fresh virtual environment, then:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6 pytest==8.4.2 setuptools==80.9.0 wheel==0.45.1
python -m pip install . --no-build-isolation --no-deps
python -m pytest -q
python examples/cms_state.py
python examples/reset_activate.py
python examples/structural_counterexamples.py
python tools/reanalyze.py --results results --output outputs/historical_summary.md
```

Dependency installation requires a wheel source or a prepared local wheelhouse.
All commands after installation run offline on CPU. Tests use synthetic tensors,
block network connections and GPU initialization, and use no pretrained model or
downloaded dataset. Examples never train an LLM. The test suite includes a bounded
tiny receiver update; its outcome is a mechanism check, not a historical result.

| Evidence level | What a reader can do |
|---|---|
| CPU_MECHANISM_DEMO | Run clocks/state isolation, reset/activate, fixed-route counterexamples and numerical tests |
| BUNDLED_RESULTS_REANALYSIS | Recompute historical summaries from the included aggregate CSVs |
| USER_SUPPLIED_ASSETS_REQUIRED | Build an adapter around a legally acquired local Llama backbone using the library; inspect metadata with the CLI |
| HISTORICAL_ONLY | Read past model results whose exact snapshots, teacher banks and input roles are not redistributed |

The initial native pilot did not establish adequate acquisition (all five group
means 25%, four trajectories per group). All six later teacher histories failed
the complete acquisition gate. Fixed-trajectory distillation sometimes recovered
local errors while changing the old/new tradeoff. Teacher agreement is not task
correctness, and a teacher's unlearned capability cannot be described as retained.
[Historical tables and their denominators](results/README.md) include failures and
missing items; synthetic examples never stand in for the historical 3B numbers.

CACHE's historical approximately 3.40x result applies to one small repeated fixed
bank. Public selection defaults to REF; absent transferable acceptance or an unknown
scope never grants CACHE. The low-level engineering implementation is retained for
inspection and independent validation. Llama/GPU execution was not run in the original release; the later bounded pilots below have separate evidence and limitations.

Read the [mechanism mapping](docs/paper_to_code.md), [reproduction status](docs/reproduction_status.md),
[technical report](reports/technical_report.md), [reproducibility instructions](docs/reproducibility.md),
and [limitations](docs/limitations.md). The true period-64 boundary, simultaneous-due
policy and terminal/slowest-layer lifecycle are unresolved. These observations concern
this implementation and its configurations; they do not invalidate the original papers.

Papers: [Nested Learning v1](https://arxiv.org/abs/2512.24695v1) and
[Language Models Need Sleep v2](https://arxiv.org/abs/2606.03979v2).
See [asset-specific licensing](LICENSE.md) and [attribution](THIRD_PARTY_NOTICES.md).
Project attribution: **Zhirui Zhu (zhiruizz)**. New code, tests, examples and tools
use MIT; new documentation, reports and the results compilation use CC BY 4.0.
Existing copyright notices and third-party attribution are preserved; the license
scope does not extend to model weights, datasets or other third-party assets.

## Bounded native retention pilot

[Three-seed synthetic retention pilot](reports/native-retention-v1.md): old mapping and short copying were acquired at100% in each seed; after new-mapping adaptation both SFT and CMS lost the old mapping, with mean copying11.46%/10.87%. This tiny restricted-scope test did not show protection from the multi-frequency schedule. It is not directly comparable with upstream natural-language scores. Protocol, all seed metrics, cost and offline reanalysis are included.

## Llama 3.2-3B LM pilot

[Frozen WikiText train-window protocol](docs/wikitext2-lm-pilot-v1.md) and [aggregate results](results/wikitext2-lm-pilot-v1-summary.json): the two-seed fixed-step run completed all SFT/CMS/partial GKD-LTI arms. Score shifts were tiny and mixed across seeds, with no evidence of a robust method advantage. This is not a WikiText benchmark reproduction; source text and model weights remain local.

## AG News retention study

The [frozen preregistration](docs/agnews-retention-v1-protocol.md) defines a real Llama 3.2-3B, three-seed, five-arm class-incremental study. [Development feasibility and real-3B recovery checks](results/agnews-dev-v1-summary.json) passed before launch. An earlier [synthetic lookup feasibility attempt](docs/x870e-synthetic-dev-v2.md) failed acquisition and was stopped; its negative result and engineering failures are retained. The [main v1 block stopped at a zero-receiver-gradient safety gate](reports/agnews-retention-v1-failure.md) during the second Sleep event; official test remained unopened and no retention scores are reported. This is a project-specific partial implementation, not a paper reproduction.

## Controlled receiver routing follow-up

The [v2 preregistration](docs/agnews-retention-v2-protocol.md) introduces an explicit persistent 0.5 receiver mixture and checkpointed execution policies. The [complete three-seed/five-arm results](reports/agnews-retention-v2.md) are now available: all runs completed, but 0/3 foundations met the fixed acquisition gate. SFT/CMS/Sleep old-task EM ended at 0%; this does not establish a qualified retention effect. V1 and evaluation interruptions remain in the evidence ledger. This is an engineering variant of partial Sleep, with no claim of a paper reproduction.

## Qualified fresh-role follow-up

[AG News v3 report](reports/agnews-qualified-v3.md): all three old foundations passed frozen train/validation gates, followed by all15 fixed-endpoint runs and a fresh1024-article within-train holdout. CMS+Sleep endpoint old/new EM is0.00%/77.86%; the full five-arm tradeoff, all seeds, costs, failures and operational recovery amendment are published. This is a bounded partial implementation, not external confirmation or full-paper reproduction.

## Diagnose the missing retention

[Post-v3 exploratory forensics](reports/agnews-sleep-forensics-v3.md) exactly reproduces all three first 64 Sleep states and finds that old validation EM falls from 92.45% to 0.39% before Sleep; its teacher scores 0.26%. All 864 literal greedy checks match the scorer. No core bug or retention repair was established; the next priority is competent-teacher transfer before destructive new-task updates. This is development evidence, not fresh confirmation.

## Competent transfer and frequent Sleep follow-up

[AG News v4 development report](reports/agnews-preupdate-v4.md): all three competent teachers pass transfer at a shared 1,536-update budget (old validation EM 91.93% ± 1.97 pp). All 18 subsequent runs and 90 fixed validation states completed. Both high-frequency teacher-order arms finish at 0% old EM; pre/post new EM is 32.68%/49.09%, while extra gold replay trades 77.73% old against 40.76% new with additional labeled compute. Earlier transfer failures, the collapsed seed, costs and CPU budget failure are retained. The new 1,024-article reserve remains unscored; this is development evidence, not confirmation or full-paper reproduction.

The ordinary CPU suite and the two heavier [explicit check entry points](docs/agnews-preupdate-v4-protocol.md#isolation-limitations-and-reproduction) use separate processes under the unchanged 256-update guard.

## V5 mechanism diagnosis and frozen follow-up

[Completed v5 comparison](reports/agnews-anchored-v5.md): all21 runs and126 fixed validation states finished; none passes the joint old/new gate. No-Sleep ends at0%/73.57% old/new EM; frozen-teacher wake KD gives75.13%/56.25%, while its tested Sleep additions are worse on both means. A separate two-context, training-supervised router reaches88.67%/71.48%, with extra stored states, labels and two-branch inference; this is not a Sleep benefit and also fails the joint gate. All seeds, failures, costs and source locks are retained; reserved evaluation data remain unscored.

## V6 routing supervision and matched replay controls

The completed [v6 development study](reports/agnews-routing-fairness-v6.md) separates preserved-context routing from phase-label supervision, verifies a shared-backbone inference path, and runs six loss-count-matched continuation trials. Correct supervised routing after new-only continuation scores old/new EM 88.67%/75.78% (joint gate 2/3); the permuted gate scores 55.60%/36.59%. Old-gold replay relearns from an already collapsed checkpoint and passes 1/3. These are development results with extra saved context or supervision, not an unlabeled Sleep result or independent confirmation. All seeds, costs and failed attempts are retained.
