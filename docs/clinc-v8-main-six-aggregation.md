# CLINC V8 main-six aggregate reporting

`tools/summarize_clinc_v8_main_six.py` produces the final aggregate report only after it verifies the frozen `clinc-cil-v8-main-six-seal`, all three shared foundations, and all six completed final checkpoints against their receipt and file hashes. It refuses partial, repeated, failed, mixed-source, or mispaired runs.

The command reads the master JSON, seal JSON, the verified training pack's train/development roles, clean-development cohort-count metadata, three foundation receipts, six run receipts, and checkpoint bytes for SHA-256 verification. The source-locked full runner validates the pack manifest and role file hashes, excludes normalized train/development duplicates, and re-derives per-seed/task row/class counts for comparison with the sidecar. It does not load weights, initialize CUDA, or read the test role. Outputs contain only aggregate metrics, counters, and opaque provenance hashes.

```powershell
python tools/summarize_clinc_v8_main_six.py `
  --seal outputs\clinc-v8-full-k10\main-six-seal.json `
  --master-config configs\clinc_cil_v8_full_master_frozen_k10.json `
  --foundation-root outputs\clinc-v8-full-k10\foundations `
  --run-root outputs\clinc-v8-full-k10\runs `
  --project-root . `
  --training-pack outputs\clinc-v8-full-k10\data\training-pack.json `
  --development-cohorts outputs\clinc-v8-preparation\expected-development-cohorts.json `
  --output-dir reports\clinc-v8-main-six
```

If a separately authorized final evaluation has already completed, its JSON aggregate can be attached without passing test data to this tool. It verifies the result against the six-run seal, evaluator source hash, and frozen evaluation contract, and reports only global all-4,500-row scores with no task mask.

```powershell
python tools/summarize_clinc_v8_main_six.py `
  --seal outputs\clinc-v8-full-k10\main-six-seal.json `
  --master-config configs\clinc_cil_v8_full_master_frozen_k10.json `
  --foundation-root outputs\clinc-v8-full-k10\foundations `
  --run-root outputs\clinc-v8-full-k10\runs `
  --project-root . `
  --training-pack outputs\clinc-v8-full-k10\data\training-pack.json `
  --development-cohorts outputs\clinc-v8-preparation\expected-development-cohorts.json `
  --output-dir reports\clinc-v8-main-six `
  --test-evaluation outputs\clinc-v8-final-test\evaluation.json `
  --test-contract configs\clinc_cil_v8_test_evaluation_frozen.json `
  --test-evaluator tools\evaluate_clinc_test_v8.py
```

The aggregate reports each task-end old/current/seen clean-development accuracy, mean and sample SD over three paired class-order seeds, paired descriptive fixed-mix-minus-HOPE differences, unique shared foundation work counted once, and per-arm continuation work. The seed values are not independent initialization draws. It also audits Sleep exposure from actual event and AdamW counters; a receipt's replay-availability flag is not treated as evidence of actual exposure. Charts are emitted as PNG and SVG when matplotlib is available.

The fixed-mix arm has 141 Sleep events and 4,230 physical Sleep AdamW steps per run; across its three runs this is 423 events and 12,690 steps. Its per-run update counter must equal 1,692 continuation wake updates plus those 4,230 Sleep steps (5,922); the shared 188-step foundation is counted separately once per seed. The train-derived clean-development cohort metadata binds exact per-seed/per-task row and class counts. The empty old-intent scope at task 0 remains structural `n/a` for accuracy and invalid predictions rather than being rewritten as a measured zero.

The final held-out result remains `test_pending` when no evaluation JSON is provided. No production summary or plot should be generated from partial runs or synthetic fixtures.
