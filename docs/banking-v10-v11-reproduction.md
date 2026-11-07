# Banking77 V10/V11: assets, checks and execution

The numerical V10 protocol is in [the frozen design](banking77-v10-preflight-protocol.md), and the distinction between the manuscript and this local implementation is in [the alignment audit](sleep-paper-alignment-v10-v11.md). Exact historical source bytes and aggregate outcomes have separate release records. Private data, pretrained weights and checkpoint tensors are not distributed.

## Offline CPU checks

The Banking checks are explicit components, outside ordinary `test_*.py` discovery. They retain the existing network/CUDA prohibition and the 256-update process limit. Run each component in a separate fresh process, with fresh output paths in an existing directory:

```sh
python tools/run_banking_component_cpu_acceptance.py --component v10 --public-root . --ledger outputs/bank-v10-budget.json --receipt outputs/bank-v10-checks.json --basetemp outputs/bank-v10-temp
python tools/run_banking_component_cpu_acceptance.py --component v11 --public-root . --ledger outputs/bank-v11-budget.json --receipt outputs/bank-v11-checks.json --basetemp outputs/bank-v11-temp
```

The acceptance entry point loads `tests/conftest.py` once explicitly, with pytest's implicit conftest discovery disabled. Published `check_` files preserve the executed workspace's test bytes; the release records identify the filename mapping. The ordinary suite and historical heavier components remain separate. CUDA-wheel checks do not certify the existing CPU-wheel-only packaging assertion.

## Local assets

The study reuses the existing Llama-3.2-3B **base** cache. Earlier acquisition/protocol metadata identify ModelScope `LLM-Research/Llama-3.2-3B`, revision `48d84dd14886f01964f72e7f742a404dd86f9fb6`. The V10 config pins ten asset hashes, and every real launch verifies them before using `local_files_only=True`. Banking did not download a new backbone. Model weights remain subject to their own Llama license; the project's code license does not cover them.

Banking77 comes from the [original publisher](https://github.com/PolyAI-LDN/task-specific-datasets/tree/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data), under its [CC BY 4.0 license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE). Attribution: Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders](https://aclanthology.org/2020.nlp4convai-1.5/) (2020). ModelScope mirror equivalence was not established; this does not establish that no mirror exists.

| Original CSV | Bytes | SHA-256 |
|---|---:|---|
| train.csv | 839073 | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| test.csv | 239961 | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

Download into a private asset directory, verify bytes and hashes, then prepare train/development with `prepare_banking_train_v10.py --source-train <train.csv> --out-dir <fresh-pack-dir> --receipt <fresh-receipt.json>`. The deterministic carveout yields 9,229 train and 770 development rows after excluding four same-label duplicates. The test file is reserved until the sealed training and pretest gates below; it is not needed for preparation, foundation learning, transfer probes or V11 design.

## Order of operations

All output directories must be fresh; a failed attempt is preserved rather than overwritten. Obtain `--help` from the named tools for their full path arguments. A relocated reproduction must regenerate location-bound contracts and source/data receipts; an absolute path in an archived execution artifact is not an authorization or a portable input path.

1. Use `prepare_banking_config_v10.py`, `audit_banking_tokens_v10.py` and `run_banking_cil_v10.py --phase preflight` to verify the computational source map, local model bytes and training pack. The historical config is `configs/banking_v10_frozen_k10.json`. Do not change its numerical settings while presenting a run as this experiment.
2. Run all three task-0 foundations (`--phase foundation`) and apply the fixed qualification gates. Run the development transfer probe and its separately recorded original-transaction/disk-replay qualification. A consolidation transaction replay is not a full-job training-resume guarantee; the main driver has no production `--resume` CLI.
3. Build the source-bound main release and all nine contracts. The process-local policy is allocator `garbage_collection_threshold:0.7` plus a deferred 0.8 memory fraction. The original general launcher's **single** `launch` mode is the executed main path. Its old `queue` mode has a raw-versus-semantic config-hash bug and is retained only as historical source. `launch_banking_checked_queue_v10.py` is a separately reviewed future queue implementation; it was not used to execute the historical nine-run grid.
4. Run the exact three-seed/three-arm grid serially. Inspect each terminal receipt before proceeding. The historical queue had two orchestration failures; completed valid runs were retained, and numerical code/settings were unchanged. There is no automatic scientific retry.
5. Seal all nine final runs with `run_banking_cil_v10.py --phase seal`, repeating `--run-dir` exactly nine times and supplying the actual main release. The seal checks receipts, parent lineage, all task endpoints and final checkpoint payloads on CPU.
6. Before official-test acquisition, freeze the V11 child config with `prepare_banking_v11_launch_materials.py freeze-config`, using that actual V10 seal. Build the official evaluation contract with `build_banking_test_contract_v10.py`; perform the real evaluator CPU preflight. These gates must succeed before obtaining or scoring the reserved test CSV.
7. Evaluate all nine sealed final contexts with `evaluate_banking_test_v10.py`, through the dedicated `launch_banking_test_memory_policy_v10.py` source-bound wrapper. Its main-output root and train/development-pack root are distinct. The raw 3,080-row population is primary; an exact normalized train/development-disjoint subset is separately reported. No endpoint is selected using test scores. The 600,000-forward/6-hour ceilings were fixed from a conservative loop bound before test access; they do not change the readout. The historical A launch failed before model loading or test-row parsing at an unused, nonexistent import. Its narrow auxiliary repair occurred after payload acquisition, and the B launch has fresh contracts and outputs; see the [launch chronology](banking-v10-test-memory-launcher.md).
8. After the completed official V10 evaluation metadata is available, build the six V11 contracts with `prepare_banking_v11_launch_materials.py build-contracts`. Start each using `launch_banking_memory_policy_v11.py --contract <path> --contract-sha256 <sha>`. Both new arms restart from their seed's foundation, not from a task-1 reference checkpoint. Their narrow contrast changes only the stated local reward contribution.
9. Build the actual fifteen-context reference with `build_banking_v11_dev_reference.py`, then run `evaluate_banking_lti_ablation_v11.py` through `launch_banking_v11_dev_evaluation.py` with a source-bound memory-policy contract. The wrapper performs CPU preflight before registering the same process-local policy and evaluating into a second fresh directory. Each seed uses all 220 development rows from its first two task blocks, with global 77-label scoring and separately named free-generation metrics. No official test payload is read for this follow-up.

Extra replay requests, class metadata, label dictionary, preallocated slots, optimizer work, checkpoint I/O and inference costs must remain visible in the reports. A three-order mean and sample SD with one shared initialization is not evidence from independent backbone initializations. Missing cost fields remain unknown, and partial matmul estimates are not complete measured FLOPs.

The frozen V11 runner's CPU preflight uses a legacy return-key alias: its `training_manifest_sha256` value is the train JSONL digest. The current single-run wrapper separately verifies the complete pack-manifest file digest and that train-file digest, then records them under distinct names. The first historical V11 launch stopped at this binding check before model loading/CUDA/updates; its fresh B replacement uses the repaired wrapper without modifying the frozen runner, numerical config or source graph. The runner's training protocol/checkpoint/receipt path already records the correct complete manifest-file digest. See the [memory-policy details](banking-v11-memory-policy-launch.md); do not disable either binding check to reproduce the experiment.

The historical official evaluator also runs greedy generation for every context and row. This reference path recomputes the full prefix per token and does not use a generation KV cache; it can take hours even though it performs zero updates. Its 6-hour budget covers the frozen evaluation workload. Candidate-only latency and an optimized decoder require separate measurements rather than dividing this joint diagnostic wall time by nine.
