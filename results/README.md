# Historical numeric tables

All tables are BUNDLED_RESULTS_REANALYSIS inputs. The underlying experiments are
HISTORICAL_ONLY: exact adapters, role assignments, teacher banks, complete logits,
optimizer and RNG states are not bundled. These are project measurements, not
published-paper benchmarks. This export projects safe columns and preserves every
row in each selected table; it is not an export of the entire research archive.

| Table | Complete selected denominator / endpoint | Initial state and comparison | Failures / independence / version |
|---|---|---|---|
| initial_pilots.csv | 10 groups: native 5x4 trajectories, Llama 5x1; final three-task macro-average;24 test examples/task,48 unique overlapping examples | frozen, sequential SFT, CMS, KS, KS+Dream arms | native acquisition unestablished; zero-gradient/no-positive outcomes retained in report; historical Phase3 execution manifests and separate postprocessing lock |
| update_reset_activation.csv | 2 events x8 U/R/E combinations x2 task groups=32 rows; old64/new32 | each event's pre-intervention T0; U=prospective CMS-base write, R=sender reset, E=receiver activation | reused dev; no optimizer in this factorial; historical Phase11 transition analysis |
| routing_readout.csv | 6 states x2 read modes x2 task groups=24 rows; old64/new32 | natural vs explicit receiver-inclusion intervention at saved states | not paper routing; answer coverage numerator counted across dev96 inputs; all-position coverage and token-call counts have different denominators; Phase11 analysis |
| reset_distillation.csv | 2 events x2 resets x4 steps x2 groups=32 rows; 4 trained paths x256 attempts | U=E=1, T1 step0 vs T2 step256; fixed T0 targets; old64/new32 dev | 256 trajectories/event (128 old+128 new), 4 continuation positions, same reused dev; 2 teacher events, not32 independent samples; historical Phase11-D1 analysis |
| phase13_benchmark.csv | 4 executors x3 registered repeats=12 rows; 8 warmup+256 formal steps | fixed seed97/boundary8, sender reset, natural route, ALL4, REF paired by repeat | BOTH repeat0 interrupted: 8 warmup+180 completed formal+1 unknown=189 attempts; no complete time; 3 engineering repeats, one workload; Phase13 engineering analysis |
| phase14_acquisition.csv | 6 histories x3 scoring steps x2 roles=36 rows; fixed step1024 | fresh frozen Llama-3.2-3B; trainable CMS base and two slot0 pairs | full gate0/6; seed211/223,307/311,401/409; h1 ascending vs h2 descending feature hash, seed/order jointly changed; family role data reused; Phase14B analysis |

The phase14 train denominators are Iris90, Car192, Balance192 and calibration24
per family/history. Full gate requires train>=95%, calibration>=75%, both NLLs
reduced at least20% from step0, and actual parameter/output changes. No best-step
substitution occurred. Aggregate counts contain no raw records or labels. Data
source attribution: [Iris](https://archive.ics.uci.edu/dataset/53/iris),
[Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation),
[Balance Scale](https://archive.ics.uci.edu/dataset/12/balance+scale).
Original records and role files are not redistributed; their terms are separate.

Routing columns `candidate_correct` and `generated_correct` are retained original
metrics, not retokenized or rescored. `b` counts teacher-wrong/student-right;
`c` teacher-right/student-wrong. `answer_KL` uses the event teacher, whereas
`KL_vs_earlier_reference` explicitly uses an earlier reference. Different targets,
positions and initial states must not be pooled into one ranking. The old/new
denominators64/32 are examples, not CMS clock ticks.

For the initial pilot, native used seeds17/29 and two task orders per group;
Llama had one seed/order trajectory per group. Each trajectory used four exposures
across three overlapping tasks. Average accuracy is the final macro-average of
those three24-example task scores, not a micro-average over48 unique examples.
Training manifests were inspected separately from the later postprocessing
inventory. Precise private source maps are not bundled; do not infer confidence
intervals from these group means.

Historical execution and analysis versions are distinct. Phase13 runtime locks
distinguish the original math implementation from a receipt-read I/O repair;
Phase14B driver is locked before final analysis; saved-result exporters and
verification tools were added after execution. Exact internal source maps stay outside this package. The
release source is therefore never advertised as byte-identical historical training
code. The public files' own hashes identify this export only.

Recompute with `python tools/reanalyze.py --results results`. No model is invoked.
Missing full-path time remains empty; failure/unknown rows stay in the denominator.
