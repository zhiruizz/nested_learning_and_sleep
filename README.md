# NL/Sleep Reproduction Lab

Ubuntu P2 update: four 40-step runs completed, but independent review found the original checkpoint protocol was not satisfied. The two-seed k4 result was unstable. See the [descriptive report](reports/ubuntu-p2-formal-descriptive-disposition.md) and [aggregate evidence](results/ubuntu-p2-descriptive/README.md); this is not a formal acceptance or confirmation claim.


Banking V10 completed nine local Llama-3.2-3B continuations across three paired class-order seeds. On all 3,080 official test rows, global 77-label sum-score accuracy was **14.21% ± 1.92%** for CMS, **22.45% ± 3.23%** for additive Sleep and **26.45% ± 1.80%** for supervised replay (mean and sample SD, one shared initialization). Sleep improved this ranking readout while worsening free generation and gold-token NLL; its main continuation wall time was about 4.96 times CMS. The [V10 report](reports/banking77-controls-v10.md) preserves every seed, the separate 3,073-row disjoint subset, old/new endpoint partitions, extra supervision and costs.

V11 compared two predeclared LTI reward controls with CMS, Sleep and supervised replay on the same 220 task-1 development requests per seed. Its [report](reports/banking77-reward-ablation-v11.md) gives all fifteen endpoints and work counters; these are development ablations, not new official-test confirmation. [Bundled aggregates and offline analysis](results/banking-v10-v11/README.md), [execution instructions](docs/banking-v10-v11-reproduction.md), and the [paper-alignment audit](docs/sleep-paper-alignment-v10-v11.md) document the local architecture, lifecycle, reward and optimizer gaps. Native HOPE and full manuscript accuracy are not established.

V9 completed a zero-update diagnosis on 60 fixed development rows: label-conditioned ranking reduces output-format loss, while old-label NLL still worsens after task 1 and fixed-mix Sleep remains weak on new classes. This is one class-order seed, not new test confirmation. The [V9 report](reports/clinc-readout-and-paper-alignment-v9.md) includes the full readouts, operational failures, costs and [reproduction tools](docs/clinc-readout-v9-reproduction.md). The [preparation draft](docs/banking77-matched-comparison-draft.md) records the earlier pre-acquisition stage; completed Banking results are reported above.

V8 completed all six Llama-3.2-3B / CLINC150 ten-task runs and evaluated their sealed final checkpoints globally on all 4,500 in-scope official test rows. Accuracy was **17.83% ± 0.87%** for the local `HOPE` baseline and **10.76% ± 1.11%** for `HOPE_SLEEP_FIXEDMIX` (mean and sample SD across three paired class-order seeds with one adapter initialization). This configuration showed no overall Sleep benefit. The [final report](reports/clinc150-lifecycle-v8-final.md) includes every seed, development trajectories, failed attempts, added replay/label metadata and resource costs.

The arms share their foundation and primary wake schedule; fixed-mix adds Sleep work and took about 2.97 times the baseline's continuation active time. This is a local CMS last-MLP ablation with a persistent 0.5 receiver mixture, not native HOPE or a faithful reproduction of the paper. [Aggregate results and offline reanalysis](results/clinc-v8-main/README.md), the [frozen protocol](docs/clinc-cil-v8-protocol.md), and [asset/recovery instructions](docs/clinc-v8-reproduction.md) explain the evidence and limits. The [earlier development report](reports/clinc150-lifecycle-v8-development.md) remains a dated record of diagnostic probes and negative results.

V7 completed 30 fixed one-context training runs and 12 wake-stress branches. Protected teachers and prompt diversity help retention, but no arm passes both task gates across all three seeds. See the [v7 report](reports/agnews-single-receiver-v7.md) for mixed/negative results, extra supervision, cost limits and the explicitly recovered main run.

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
