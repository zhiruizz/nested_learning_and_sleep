# CLINC V8 final aggregates

This directory contains the main-six summary, separate resource/storage audits, a safe attempt-cost ledger, and one fixed global CLINC V8 final evaluation. It is a controlled CLINC150 experiment, not a full reproduction of native or paper-faithful HOPE Sleep. The three runs per arm are paired class-order seeds with a shared initialization seed; they are not independent random initializations.

`summary.json` and `summary.md` report task-end old/current/seen development metrics, paired descriptive differences, work counters, resource/storage summaries, and the one global test result. Three per-seed task-0 foundations are shared between each seed's two arms and counted once each. The summary's top-level `test_payload_read: false` refers to the training-side aggregator: it derives development counts from the train/development pack and receipts without opening raw test rows. The separate final evaluation artifact reports `test_data_read: true` because that fixed evaluation did read the locked test split. The JSON keeps finite denominators, invalid-generation counts, per-class test totals/correct counts, work/source/checkpoint hashes, and a `_public_capture` reference. The curated training-seal capture has its own schema and `gate_reusable: false`; it is provenance only. The final evaluation JSON keeps the evaluator's original aggregate schema and `runs[].primary.total/correct/per_class` counts. Its `_public_capture.source_artifact_sha256_raw` is the hash of the exact unmodified source evaluation JSON and is checked against `summary.json`'s `final_test.input_artifact_sha256`.

The publication bundle does not include datasets, prompts, row identifiers, token IDs, generated answers, model weights, or private checkpoint payloads. Source artifact hashes are retained; local filesystem paths are normalized.

For offline aggregate reanalysis after downloading this source bundle, run:

```powershell
python tools/reanalyze_clinc_v8_main.py `
  --summary results/clinc-v8-main/summary.json `
  --test-evaluation results/clinc-v8-main/final-test-evaluation.json `
  --out-dir outputs/clinc-v8-main-reanalysis
```

The reanalysis recalculates global and per-class test count checks and paired three-seed summary statistics from the published aggregate JSON. It does not access a dataset, load a model, or perform training.
