# V12 launcher and CPU acceptance review

The corrected B source/config lock, actual three-seed CPU-only preflight, and combined guarded CPU acceptance have been checked. There is no remaining blocker to starting the fresh seed-17 B run. Its disk-restore qualification is a hard condition inside that same run: failure aborts it; success automatically continues the remaining seed-17 arms. Seeds 29 and 43 remain held until the root independently inspects the actual seed-17 qualification result.

This reviewer did not load a model, access a dataset, initialize CUDA, or execute an optimizer update. The earlier seed-17 A attempt did load the model and execute four task-1 prefix updates; its exact failure and work accounting are recorded below.

## Frozen B source hashes

| File | SHA-256 |
|---|---|
| `configs/banking_v12_optimizer_boundary_frozen_b.json` | `A5A01BED82B5F82314436A00061B40C475A1396F9FB3A28B861F340E05E01D25` |
| `tools/run_banking_optimizer_boundary_v12.py` | `638852B8A064EA388119C503C59C3C66B1F2678E3F71C41304072A27E691269F` |
| `tools/banking_cil_v12_batching.py` | `99E0163C333FE1FDEC1928BB2F932C79D88C7F0683B8D917883F2C79E8B0C829` |
| `tools/launch_banking_optimizer_boundary_v12.py` | `E45CF1FDAD02D335138394CFA28FC0A82779952C11C87BB0DDF6263BC8DE9874` |
| `tools/run_banking_v12_cpu_acceptance.py` | `9D0A9FC5BBCA4F953BBCD3D8E0E1BBA687A207A4E987C2F462B7E1E650AADD39` |
| `tests/test_banking_optimizer_boundary_v12_cpu.py` | `7F07E47BDE64A5416F9F8BF2A87C9BF265E6C8D42D239149579D1B89B2773857` |
| `tests/test_banking_cil_v12_batching_cpu.py` | `3E0DAAF5C9460ABD3C82251F4D0E82A570C06259FF0DBA9F4A82A812E3553D7E` |
| `tests/test_banking_v12_launcher_cpu.py` | `B2B9B4540ECCF1C94FFCEC4055711F8F20A9D308F9CE2A79D1779652E4ADB170` |
| `tests/test_banking_v12_runtime_interfaces_cpu.py` | `0B9C499EECE8AB253EE1A66509E69C12EA4644B49A6E0F741969E4C4DCEB4044` |

The B source/config lock index is `outputs/v12-prep/v12-source-config-freeze-b.json`, SHA-256 `922FCC3CCD4011ECE149D428422A421B0B2DED3D62D81D79CA745BF9EF3B283C`. It records the prior A failure as preserved and states `scientific_settings_unchanged=true`. Its source map matches the local execution-source bytes above.

## Review and checks

The launcher rejects overlapping output and launch-record paths, validates the config digest before importing the runner, writes a startup receipt before CPU preflight, checks CUDA remains uninitialized around runner import/preflight, and only applies the process-local memory policy after preflight passes. It persists failed status and records startup wall time. The runner rechecks its V12 source map and foundation receipt/checkpoint digests after preflight, verifies the V10 source lock, reads only train/development inputs, and has no test-data path.

The boundary is selected by the actual due clock. Teacher rollouts are generated once and reused across arms. Each request loss is backpropagated immediately; each microbatch performs one global clip and one AdamW step. The run checks common receiver initialization across arms. Seed-17 `B32_LOW` runs first, snapshots after 32 requests, reads the snapshot back from disk, and compares original/restored branches through requests 32–64. A mismatch aborts the run. After a pass, the same process continues `B32_LOW` and the remaining seed-17 arms; no external pause/resume boundary exists. The restore branch’s extra 32 requests and one AdamW step are recorded.

The correction fixes the A failure without treating the `MutableContext` dataclass as an `nn.Module`: it evaluates the owned memory modules and freezes context parameters. A CPU regression uses the real context type and checks child module modes, gradients, and unrelated modules. The scoring cohort remains 220 development examples (110 old, 110 current) using all 77 labels; this is a single-boundary development mechanism diagnostic, not a full Sleep or paper-faithful reproduction.

The actual preflight bundle `outputs/v12-prep/actual-all-three-cpu-preflight-b.json` passed. All per-seed records bind B config SHA `a5a01b...`, runner `638852...`, batch helper `99e016...`, V10 source lock `3fcc2e...`, and manifest `ed9f4b...`. Each reports 77 classes, 9,229 training rows, 770 development rows, `model_loaded=false`, `cuda_initialized=false`, `optimizer_updates=0`, and `test_data_read=false`.

The combined guarded acceptance `outputs/v12-prep/combined-cpu-acceptance-c.json` passed with the source tests unchanged, the original offline guard unchanged (`c22f0748...`), limit 256, zero CMS attempts, 13 toy optimizer attempts, and no setup/integrity error. I also ran all four V12 CPU test files directly: 30 passed in 3.43 seconds. The acceptance log SHA-256 is `3b8cd803d3929202760db3aa0a670b971afac02d5f6ad0b00cb4d697d5ff8585`.

**Conditional readiness:** start only the fresh seed-17 B run. The disk-restore qualification is an internal hard gate: if it fails, the run aborts and remains a failure record; if it passes, the process continues the remainder of seed 17 as already authorized. Do not start seeds 29/43 until the root has independently reviewed seed 17’s actual qualification receipt.

## Seed-17 attempt A disposition

The A launch failed before teacher rollouts or receiver Sleep/AdamW with `AttributeError: 'MutableContext' object has no attribute 'eval'`. Its preserved launcher record and `failure.json` agree on the error and A source/config/foundation bindings. A did perform four task-1 prefix wake updates: 4 full-backbone forwards, 4 gradient-enabled tail/backward calls, 1,039 valid input tokens (369 padded), 0 Sleep events, 0 sampled tokens, and 0 physical AdamW updates. No official test data was read and automatic retries were 0. A is an engineering startup failure, not a negative scientific result; do not describe it as zero training work. The B attempt uses a new config and fresh output path.
