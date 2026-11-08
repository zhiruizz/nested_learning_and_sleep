# Banking77 V12: fixed-teacher boundary diagnostic

V12 uses a fixed teacher and one real consolidation boundary to test receiver
batch aggregation, learning rate and update budget. It is a development-only
mechanism study. It does not implement the paper's full alternating-policy
GKD/LTI objective, learned semantic reward, repeated-boundary lifecycle or
native HOPE architecture. The prior official test has already been viewed and
is not an input to this study.

## Assets and source identity

Use the original Banking77 train/development pack and the three qualified V10
task-0 foundations, with seeds 17, 29 and 43. Model, dataset preparation, licenses
and the foundation recipe are described in [the V10/V11 asset guide](banking-v10-v11-reproduction.md).
The pack hash is `ed9f4b047768021cbf16a3fa042db12baf212038f89f33a00099f2e32b7300f3`;
it contains 9,229 training and 770 development examples. It has no test-data field.
The model is the same hash-pinned ModelScope Llama-3.2-3B base cache used in V10.
Private model/data files and checkpoint tensors are not distributed.

The B execution recipe is [the frozen config](../configs/banking_v12_optimizer_boundary_frozen_b.json),
SHA-256 `a5a01bed82b5f82314436a00061b40c475a1396f9fb3a28b861f340e05e01d25`.
It pins the new runner/batch/launcher/CPU-check sources, while the original V10
config pins the 24-file numerical dependency graph. All launches verify source,
model, pack and seed-specific foundation identities before using CUDA.
The seed-specific foundation windows are retained: the next level-0 write is
due after 4, 10 and 14 task-1 wake minibatches respectively.

The first A execution failed because it called `.eval()` on a `MutableContext`
dataclass. It performed four prefix wake updates and zero receiver updates or
teacher rollouts. B only corrects teacher-context mode/freezing through the
actual memory modules; scientific hyperparameters, data and budgets are unchanged.
Keep A's source/config/failure evidence separate from the B scientific results.

## CPU checks

The observed environment uses PyTorch 2.8.0+cu128, NumPy 2.2.6, pytest 8.4.2,
Transformers 4.57.1 and Matplotlib 3.11.2. Matplotlib is required for the V12
plot smoke test and standalone figures; the original minimal CPU installation
does not include it. CPU checks still prohibit CUDA initialization even when
the installed Torch wheel supports CUDA.

Banking CPU components use explicit `check_` filenames in the public checkout.
The V12 entry point discovers those files, loads the existing offline guard
once and preserves its 256-attempt cap. Use fresh paths beneath an existing
directory for each invocation:

```sh
python tools/run_banking_v12_cpu_acceptance.py --public-root . --ledger ../checks/v12-budget.json --receipt ../checks/v12-acceptance.json --log ../checks/v12-pytest.txt --basetemp ../checks/v12-temp
```

These are small CPU tests. They prohibit network and CUDA initialization. The
real-model checks and complete GPU runs have their own source-bound evidence.
Additional analysis/evaluation checks, if included in a release, are listed in
that release's acceptance receipt rather than silently summed with overlapping
historical suites.

## One seed per process

Keep a source checkout separate from the private asset/output trees. Create an
explicit, fresh experiment parent directory. The launcher rejects output paths
inside model, training-pack, foundation or source trees, overlapping launch
records, existing scientific outputs and stale configuration hashes.

Example PowerShell command for seed 17, using the historical private layout:

```powershell
$studyPython = 'python'
$studySource = (Get-Location).Path
$studyAssets = 'D:\my_paper\Nested_Learning_and_Sleep'
& $studyPython tools/launch_banking_optimizer_boundary_v12.py --config configs/banking_v12_optimizer_boundary_frozen_b.json --expected-config-sha256 a5a01bed82b5f82314436a00061b40c475a1396f9fb3a28b861f340e05e01d25 --v10-config configs/banking_v10_frozen_k10.json --project-root $studySource --model-path "$studyAssets/.cache/models/llama32-3b" --training-pack "$studyAssets/.cache/datasets/banking77-original-9d08145/train-dev-v10/pack-manifest.json" --foundation-root "$studyAssets/outputs/banking-v10-foundations-20261107-a/foundations" --output-dir "$studyAssets/outputs/FRESH-V12/seed17" --seed 17 --launch-record ../checks/FRESH-seed17-launch.json
```

Replace asset and output locations with verified local equivalents. The parent
directories must exist; each seed's scientific directory and launch record must
be fresh. The reviewed GPU process uses PyTorch `2.8.0+cu128`, deferred CUDA
initialization, allocator `garbage_collection_threshold:0.7` and a process-local
0.8 memory fraction. No global driver or security setting is changed.

Seed 17 first builds 3,200 fixed teacher trajectories from training requests.
Its low-rate batch-32 arm executes first. At 32 requests, it saves and reads a
real disk snapshot; the original object and a restored branch each continue
through requests 32–64. A mismatch aborts the run. A pass permits the same
process to continue its fixed remaining arms. Review that actual qualification
receipt, the complete seed-17 terminal artifacts and GPU idleness before
starting seeds 29 and 43 serially with the same frozen settings.

The copied foundation windows, class order, training request stream, teacher
trajectories and initial receiver state are shared within each seed. Requests
alternate old/current roles and cycle classes; selection uses class/task
metadata, but gold labels never become KL targets. Batch 1 uses 960 updates per
960 requests; batch 32 uses 30 updates. The low-rate batch-32 arm continues to
100 total updates and 3,200 requests without a rerun. Both batch-1 arms retain
their 30-request checkpoints as update-count comparisons with unequal exposure.

Tokenization reserves the longest label plus EOS. With the pinned tokenizer and
77-label dictionary this is ten tokens, equal to the frozen generation cap, so
the 128-token total sequence budget is respected. Teacher generation uses
exact-length groups with microbatch eight. Receiver effective batch 32 instead
means 32 per-request token-mean KL losses, accumulated before one clip/Adam step.

## Analysis, checkpoints and reporting

Checkpoints are only taken between complete effective batches. They preserve
teacher/student/receiver routes, Adam state and RNG. This qualifies transaction
restoration; the runner has no whole-job production resume CLI. Preserve failed
attempts instead of overwriting or automatically retrying them.

Primary evaluation covers the same 110 old and 110 current development examples
per seed, with all 77 label candidates and no task mask. Summed label likelihood,
length-mean scoring, greedy exact match and row-mean token NLL are distinct
readouts. A mean/sample SD over three orders shares one backbone initialization.
Keep exposure-matched, update-count-matched and extended-budget contrasts named
separately; none is automatically a measured FLOP match.

Core receiver work is 18,240 requests and 6,150 AdamW updates over three seeds.
Seed 17's restored validation branch adds 32 request evaluations and one update.
Shared prefix wake, teacher generation, endpoint inference, checkpoint I/O and
engineering failures have separate costs. The seed-17 original request-32–64
training batch is retained in `qualification.original_metrics`; the batch-log
aggregator must include it once in the core trajectory and count only the
additional restored branch as qualification overhead.

The [fixed-trajectory KL supplement](banking-v12-fixed-teacher-fit-evaluation-plan.md)
uses 64 already-generated training trajectories per seed, zero updates and its
own source/input/work record. It is a training-fit diagnostic, not held-out
validation. Publish only safe aggregate evidence and source/config/tests; keep
raw requests, targets, trajectory tokens, model/data files and checkpoints private.

The draft config is retained only for small CPU fixtures; frozen B identifies
the actual scientific launches. Published CPU checks use the check_ filenames;
the old executed fixture is separately archived with its before/after import
portability hashes. These filename changes alter no training arithmetic.
