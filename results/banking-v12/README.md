# Banking77 V12 aggregate evidence

This package preserves all three seeded class orders and all ten contexts from
the fixed-teacher, one-boundary optimizer diagnostic. These are development
results, not a full Sleep lifecycle or independent confirmation. The complete
interpretation is in [the V12 report](../../reports/banking77-optimizer-boundary-v12.md).

- [Primary aggregate](v12-boundary-aggregate.json) binds source/config/run hashes,
  all 100 mean/sample-SD metric cells, 80 paired contrasts and phase costs.
- [Seed-level metrics](v12-boundary-metrics.csv) contains 300 aggregate rows,
  not individual examples, labels or predictions.
- [Independent arithmetic review](independent-crosscheck-r1.json) recomputes
  the summaries from each final receipt's aggregate fields.
- [Figures and input bindings](figures/plot-manifest.json) preserve three seed
  points and raw sample SD, not confidence intervals.
- [Read-only fixed-training-trajectory fit](fixed-fit/fixed-teacher-fit-aggregate.json)
  compares identical 64-request teacher trajectories per seed, with no updates.
  It is training-fit evidence, not held-out accuracy.
- [Execution and engineering provenance](provenance/README.md) preserves source
  locks, startup failure, restore qualification and terminal reviews.
- [Final published-layout CPU acceptance](software-acceptance-public.json)
  identifies the exact checks, unchanged offline guard and toy-update budget.

Requests, token sequences, raw data, model weights and checkpoint tensors remain
private. Paths inside historical receipts identify execution-workspace records;
not every referenced private file is bundled. Source-lock identity does not
prove numerical paper reproduction.

Rebuild and validate the public figures offline from the numeric aggregate:

```sh
python tools/plot_banking_optimizer_boundary_v12.py --aggregate results/banking-v12/v12-boundary-aggregate.json --metrics-csv results/banking-v12/v12-boundary-metrics.csv --config configs/banking_v12_optimizer_boundary_frozen_b.json --analyzer-source tools/reanalyze_banking_optimizer_boundary_v12.py --out-dir outputs/FRESH-v12-figures
```

The output directory must be fresh. The plotter rechecks every primary cell
against the CSV and verifies the analyzer/config hashes. Re-running training,
the receipt-level analyzer or the fixed-fit evaluator requires legally acquired
private model/data/checkpoint inputs; see [the execution guide](../../docs/banking-v12-reproduction.md).

Core receiver work is 18,240 request exposures and 6,150 updates. The separate
seed-17 restore branch adds 32 request evaluations and one update. Matched
request counts, matched update counts and the 3,200-request dose extension are
different contrasts. Stored parameters, activated slots and executed FLOPs are
not interchangeable; unmeasured costs are not assumed to be zero.
