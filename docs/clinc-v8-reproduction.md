# Running and interpreting CLINC V8

This is a local Llama last-MLP CMS extension with a controlled routing ablation.
It is not native HOPE and does not reproduce all of Sleep's architecture, reward
model, data protocol or reported accuracy. The [development report](../reports/clinc150-lifecycle-v8-development.md)
contains negative results, corrections and additional supervision/compute costs.

## Assets and offline checks

The pinned model is ModelScope `LLM-Research/Llama-3.2-3B` at
`48d84dd14886f01964f72e7f742a404dd86f9fb6`, the **base** checkpoint. Model weights
are subject to their own license and are not bundled. The experiment's observed
environment is in [requirements/clinc-v8-observed.txt](../requirements/clinc-v8-observed.txt);
it is an environment record, not a complete portable wheel lock.

The dataset is [clinc/clinc_oos, plus](https://huggingface.co/datasets/clinc/clinc_oos/tree/155b9c710419136e17307b80d0a13e68cd46b4ec/plus),
revision `155b9c710419136e17307b80d0a13e68cd46b4ec`. The two ModelScope deepopen
resources suggested during acquisition are trained classifiers, not raw datasets;
neither was used as an initializer or teacher. The pinned CLINC loader and source
identify this dataset variant as CC BY 3.0. Observe the dataset's own terms.

Acquire only train and validation before training: 312096 and 77789 bytes. Verify
the exact hashes and class/OOS counts using `tools/acquire_clinc_v8.py`; raw data
and the generated pack remain local. The full split gives 15000 training rows,
3000 development rows, and three normalized train/development overlaps. Clean
development excludes those three rows. This is not InCA's subsampled protocol.

CPU tests use synthetic inputs and block network/GPU initialization through the
existing repository fixture. Run `python -m pytest -q`. The separate real-3B
checks under `checks/check_clinc_*` require legally obtained local model files;
they are opt-in GPU qualifications, not benchmark results. Use each command's
`--help` for its local-asset arguments. Disk replay checks include pending
gradients, optimizer state and CPU/CUDA/Python/NumPy RNG.

## Captured protocol and a fresh replication

`configs/clinc_cil_v8_full_master_frozen_k10.json` is the captured protocol for
three class-order seeds (17,29,43), a fixed adapter initialization seed, ten
15-class tasks, and paired `HOPE` / `HOPE_SLEEP_FIXEDMIX` continuations. It pins
the original asset/manifest hashes and source version. It was frozen before any
full-grid training. The timestamped pretraining-freeze record is a historical
decision record, not a live progress file.

Do not silently edit captured configurations to pass hash guards. A fresh
replication must prepare its own verified local pack and freeze a new master
with the corresponding manifest/source hashes. The Parquet hashes and row
conversion are fixed, but captured API-metadata hashes and Windows JSON newline
bytes can differ from a freshly generated manifest on another platform. Equal
benchmark content does not imply an equal manifest-file hash.

For each seed, first invoke `tools/run_clinc_cil_v8_full.py --mode foundation`
with `--master-config`, `--training-pack`, `--project-root`, `--model-path`,
`--seed` and a fresh `--output-dir`. Each foundation trains only task 0. All
three must meet the frozen clean-development acquisition threshold before
any continuation. Both arms of one seed must use the exact same foundation
checkpoint, pending windows and RNG state; shared foundation work is counted once.

Then run `--mode continuation` for each seed/arm with `--foundation-dir` pointing
to the three verified `seedXX` foundation directories. There are no automatic
retries or checkpoint selection. Manual recovery is restricted to a post-event
safe checkpoint under the same source/config/data/model bindings. A successful
terminal run cannot be resumed or overwritten. Full benchmark outcomes require
all ten task-end metrics and the actual terminal artifacts, not merely an active
process or a partial log.

`tools/run_clinc_v8_full_queue.py` is the immediate serial queue used on the local
host after inspecting seed17. It is not a scheduler and never restarts a failed
child. `tools/status_clinc_v8_full.py` reads scalar progress without evaluation.

## Test isolation and evidence limits

Do not acquire the official test shard until `--mode seal` has verified the
exact six-run grid, the three qualified foundations, source/master/data/model
bindings, and actual receipt/checkpoint byte hashes. Training views expose no
test role. The main endpoint is global 150-class accuracy from fixed final
checkpoints, without a task mask. A test result requires the separately reviewed
final evaluator and its completed receipts; the development report does not
stand in for that result.

The public benchmark and label metadata have been seen historically; pretraining
contamination is not ruled out. Three class-order seeds with one shared adapter
initialization do not measure uncertainty across model initializations. The
fixed 0.5 receiver mixture adds inference/training work, and class-balanced replay
uses known training-label metadata. Four prompts per class is a replay working
set; the implementation retains the full training rows/cache in local storage.
Gold CE controls add known labeled optimization targets and are not matched to
Sleep by information or FLOPs. Analytical work estimates cover only the listed
GEMM/attention operators and are not measured complete hardware FLOPs.

The captured full runner has one descriptive receipt-field defect:
`data_exposure.sleep_text_replay` is written as `true` for both arms. Actual
`HOPE` executes no replay loss or Sleep AdamW step; executed arm, event and
optimizer counters determine exposure. Original source-bound receipts remain
immutable, and derived reports must state this correction.
