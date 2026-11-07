# Reproducing the V9 development diagnosis

The result is a fixed-checkpoint readout diagnosis, not new model training or
an independent test confirmation. The accepted run is **d**. Historical attempts
a/b failed before scoring; c is quarantined because a source-freeze failure did
not stop its shell launch. Their work and failure-source hashes remain in the
operational ledger. Do not combine them as independent experimental repeats.

## Offline aggregate checks

These commands use only the published aggregate counts and do not load Torch,
model weights, dataset rows or checkpoints. Each output directory must be new.

```sh
python tools/reanalyze_clinc_readout_v9.py --input results/clinc-v9-readout/readout-result.json --out-dir outputs/clinc-v9-readout-reanalysis
```

The tool recomputes count/accuracy/delta relationships and validates NLL cohort
weighted consistency. It does not independently recompute model-level NLL from
logits. Repository checksums authenticate the delivered bytes; its optional
`--expected-source-sha` compares the exact provided artifact bytes rather than
pretending a curated file must retain the private raw artifact's hash.

With matplotlib/numpy installed, regenerate the figure separately:

```sh
python tools/plot_clinc_readout_v9.py --input results/clinc-v9-readout/readout-result.json --output-dir outputs/clinc-v9-readout-figures
```

The observed real-model environment is documented in
`requirements/clinc-v8-observed.txt`. No installation or download occurs in
these commands.

## Real-checkpoint evaluation (explicit local assets required)

First follow the [V8 asset instructions](clinc-v8-reproduction.md) to obtain the
legally usable model and train/development pack, and produce the corresponding
complete V8 grid. Dataset rows, weights and checkpoint payloads are not bundled.
The original model, data and source hashes are in the frozen V8 master and V9
records. A different model, source or training protocol is a separate experiment.

Generate a local runtime context map from your own paths. The builder verifies
the actual source lock, model assets, raw grid seal, foundation gate, receipts,
metrics and three checkpoint hashes. It does not require the original machine's
absolute paths, Git HEAD or raw seal SHA; regenerated local artifact metadata
and serialization timestamps may differ.

```sh
python tools/prepare_clinc_readout_contexts_v9.py --project-root . --master-config configs/clinc_cil_v8_full_master_frozen_k10.json --manifest LOCAL_TRAIN_DEV/pack-manifest.json --grid-root LOCAL_V8_GRID --model-path LOCAL_LLAMA32_3B --out outputs/clinc-v9/context-map.json
```

The shipped `context-bindings.json` is a path-normalized provenance capture,
**not** the runtime context-map input. Generate the latter with the builder.
The checkpoint cursor contract is task 0 `(0,189,1)` for the foundation and
task 1 `(1,189,2)` for the two continuations: `(task_id,next_step,tasks_completed)`.
The next-step cursor is not silently normalized to the following task.

Run the CPU preflight and check its successful exit before taking another action:

```sh
python tools/diagnose_clinc_label_readout_v9.py --mode checkpoint-preflight --project-root . --manifest LOCAL_TRAIN_DEV/pack-manifest.json --master-config configs/clinc_cil_v8_full_master_frozen_k10.json --contexts outputs/clinc-v9/context-map.json --out-dir outputs/clinc-v9/cpu-preflight
```

Freeze and review the intended source/context hashes before a real-model run.
Do not chain a freeze assertion and launch with a shell separator that continues
after failure. The accepted run used a separate successful freeze action and a
launcher that rechecked file hashes before creating its single child process.
Then, in a separate action after successful validation:

```sh
python tools/diagnose_clinc_label_readout_v9.py --mode evaluate --project-root . --model-path LOCAL_LLAMA32_3B --manifest LOCAL_TRAIN_DEV/pack-manifest.json --master-config configs/clinc_cil_v8_full_master_frozen_k10.json --contexts outputs/clinc-v9/context-map.json --device cuda:0 --out-dir outputs/clinc-v9/new-readout-run
```

The runtime accepts no test-path argument. It selects the fixed 60 clean
development rows without predictions, scores all 150 label candidates, and
checks adapter and frozen-backbone immutability. Model setup and validation are
included in its 1,800-second evaluate timer; the CLI's initial manifest parse is
outside that timer. The full-backbone cap is 10,000, updates are zero, and there
is no automatic retry, terminal overwrite or checkpoint selection. Retain any
failure directory and record additional setup cost before a reviewed recovery.

The primary V8 evaluation remains unchanged. Candidate ranking adds label-set
information and inference compute; its output cannot be relabeled as repaired
V8 test accuracy or a paper-faithful result.
