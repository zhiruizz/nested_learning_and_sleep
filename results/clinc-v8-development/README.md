# CLINC V8 development artifacts

These are one-class-order-seed, two-task development results and diagnostics,
not the final three-seed benchmark or a paper-faithful HOPE/Sleep reproduction.
The [report](../../reports/clinc150-lifecycle-v8-development.md) records both
negative transfer budgets, Gold CE's extra known-label targets and compute,
historical evaluation corrections, failures and limits.

All public JSONs are aggregate-only. `_public_capture` records each raw local
input's SHA and the path/text normalization; `public-capture.json` binds the
published source/config/report/figure bytes. It is a publication manifest,
not a raw training seal. Weights, dataset rows, token arrays and checkpoints
remain local. The attempt ledger is an explicitly dated in-progress snapshot.

From the repository root, these commands use only public aggregates:

```sh
python tools/validate_clinc_v8_public_bundle.py .
python tools/plot_clinc_v8_corrected_transfer_budgets.py --transfer results/clinc-v8-development/transfer-budget-analysis.json --development results/clinc-v8-development/development-summary.json --output-stem outputs/clinc-v8-public-reanalysis/corrected-transfer
```

The second command requires Matplotlib (observed version 3.11.2). Plotting is
separate from the model runtime requirements. See `release-validation.json`
for the actual CPU invocation limitations and guard counts. Fixed full-grid
training is in progress; official test access awaits its verified six-run seal.
