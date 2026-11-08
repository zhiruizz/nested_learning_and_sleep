# Reproducing the V13 batch and learning-rate diagnostic

This release extends the [V12 fixed-teacher diagnostic](banking-v12-reproduction.md)
to two natural level-0 consolidation boundaries in the local GKD/LTI lifecycle.
Read the [frozen protocol](banking-v13-controlled-protocol.md),
[paper audit](sleep-paper-v13-primary-source-audit.md), and
[report](../reports/banking77-batched-lifecycle-v13.md) before interpreting it.
It is a development experiment, not a numerical reproduction of Sleep's
continual-classification benchmark.

## Assets and environment

Use the model, original Banking77 training pack, qualified task-0 foundation,
and release-artifact prerequisites described in the V10/V12 reproduction notes.
The frozen configuration records the asset byte hashes. Model source is
`LLM-Research/Llama-3.2-3B@48d84dd14886f01964f72e7f742a404dd86f9fb6`
on ModelScope. The existing Banking77 source is the previously authorized
original-data revision; no new dataset or official-test download is needed.
The local pack has 9,229 training rows and 770 development rows over 77 labels.

The execution environment uses Python 3.11, PyTorch `2.8.0+cu128`,
Transformers `4.57.1`, NumPy `2.2.6`, and an RTX 5090 D with 32 GB VRAM.
The process-local CUDA memory fraction is 0.8 and allocator setting is
`garbage_collection_threshold:0.7`. These are reproducibility settings, not
changes to the host driver or other GPU applications.

The exact private foundations and their checkpoint byte hashes are bound in
the run receipts. Checkpoints, model weights, datasets, raw requests, generated
text and token trajectories are deliberately absent from Git. To rerun training,
prepare licensed local assets and qualified foundations using the V10 procedure;
do not silently substitute arbitrary checkpoints for an exact-run comparison.
Newly generated foundations need their own honest qualification and asset lock.

## CPU checks

From the repository root, with fresh output paths and their parent directory
already created:

```bash
python tools/run_banking_v13_cpu_acceptance.py \
  --public-root . \
  --ledger /path/to/fresh/cpu-ledger.json \
  --receipt /path/to/fresh/cpu-receipt.json \
  --log /path/to/fresh/cpu.log \
  --basetemp /path/to/fresh/pytest-temp
```

The unchanged offline guard prohibits CUDA and networking, enforces the existing
256-attempt toy-optimizer ceiling, and records actual attempts. This exercises
batch averaging, zero-objective behavior, budget reservation, protection guards,
request cursors, transaction snapshots, launcher preflight, B1 parity with the
existing objective, and the aggregate/plot contract. It is not GPU qualification.

## Frozen GPU execution

The exact configuration SHA-256 is
`1c4aae0c13d6b80193fee548639e7adbf6fae4d5b2c13d3d21d5f72ba26f0f35`.
It pins eight execution dependencies in addition to the unchanged V10 source
graph and model/data locks. Each launch requires a fresh scientific output
directory and a separate fresh launcher record; their parent directories must
exist. The common foundation root contains `seed17`, `seed29`, and `seed43`.

```bash
python -u tools/launch_banking_sleep_lifecycle_v13.py \
  --config configs/banking_v13_batched_lifecycle_frozen_a.json \
  --expected-config-sha256 1c4aae0c13d6b80193fee548639e7adbf6fae4d5b2c13d3d21d5f72ba26f0f35 \
  --v10-config configs/banking_v10_frozen_k10.json \
  --project-root . \
  --model-path /path/to/licensed/llama32-3b \
  --training-pack /path/to/train-dev-v10/pack-manifest.json \
  --foundation-root /path/to/qualified/foundations \
  --release-artifact /path/to/qualified/main-release.json \
  --output-dir /path/to/fresh/run-root/seed-17/B32_HIGH \
  --launch-record /path/to/fresh/seed17-B32_HIGH-launch.json \
  --seed 17 --arm B32_HIGH
```

The first seed-17/B32_HIGH event writes a private snapshot after request 32,
continues the original object to request 64, then reloads the on-disk snapshot
and replays requests 32:64 in a separate transaction. Full snapshot content,
optimizer/RNG state and batch metrics must agree. The original post-64 global
RNG state is restored and the extra 32 requests / one optimizer update are
charged separately. A failure ends the run; there is no automatic retry or
whole-job resume claim. Review that qualification and the completed first
trajectory before executing the other frozen cells.

Every arm uses actual wake tensors of 32 requests. Effective Sleep batch 32
uses request microbatches of one with gradient accumulation, followed by one
clip and AdamW update; student parameters stay fixed within the effective
batch. Effective batch 1 uses one update opportunity per request. Five stage
states are saved with immediate destination deserialization/readback. All five
development evaluations occur after training has finished, so they cannot
change an in-flight event's request budget or numerical settings.

## Offline analysis

The analysis tool consumes only complete aggregate receipts in the
`seed-{seed}/{arm}/final-receipt.json` layout; it does not read checkpoints or
datasets. It requires all 18 frozen cells, validates denominators, prompt
digests and counter reconciliation, and reports all stages and all three seeds.
Use fresh output directories:

```bash
python tools/reanalyze_banking_sleep_lifecycle_v13.py \
  --run-root /path/to/complete/receipt-tree \
  --config configs/banking_v13_batched_lifecycle_frozen_a.json \
  --source-root . \
  --out-dir /path/to/fresh/analysis
python tools/plot_banking_sleep_lifecycle_v13.py \
  --aggregate /path/to/fresh/analysis/v13-lifecycle-aggregate.json \
  --metrics-csv /path/to/fresh/analysis/v13-lifecycle-metrics.csv \
  --out-dir /path/to/fresh/figures
```

Wall time includes model loading and checkpoint overhead; scalar forward,
backward, token and update counters are separately reconciled. These counters
do not measure hardware FLOPs or imply equivalent compute across arms. The
primary 2-by-2 grid matches replay prompt exposure, while the dose comparison
changes exposure and updates. CMS remains a cheaper reference. No arm is
selected using the official test set, and these development results must not
be described as an independent confirmation.

The measured wall time also includes the research runtime's state-protection
hashes, checks and transfers. It is not an optimized deployment-throughput
benchmark. Replay uses stored training queries as prompts and class/task
metadata for selection; only their response targets are self-generated.

The frozen batch helper's schedule-validation docstring mistakenly describes
3,200 requests as a complete three-request cycle. Execution deliberately allows
the two-GKD terminal suffix: the actual per-event method counts are 2,134 GKD
and 1,066 LTI. The source bytes remain preserved for their execution hashes;
the protocol and strict aggregate checks use these correct counts.

For a CPU-only parameter/storage census of the four selected seed-17/B32_HIGH
states, use your private checkpoint tree. This loads mutable checkpoint tensors
on CPU, verifies their source/config bindings, and emits shapes, counts and
hashes without values; it does not load the Llama backbone or read scores:

```bash
python tools/audit_banking_v13_parameter_storage.py \
  --public-source-root . \
  --asset-root /path/to/licensed/local/assets \
  --trajectory-root /path/to/private/run-root/seed-17/B32_HIGH \
  --output /path/to/fresh/parameter-audit.json
```
