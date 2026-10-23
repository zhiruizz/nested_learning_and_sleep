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
inspection and independent validation. Llama/GPU execution is NOT_RUN in this release.

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
