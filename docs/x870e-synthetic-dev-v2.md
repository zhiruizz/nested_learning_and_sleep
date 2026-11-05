# X870E synthetic DEV v2: acquisition gate not met

This report documents a development-only feasibility attempt. It is not a main experiment, retention result, or paper reproduction.

## Outcome

Only synthetic DEV seed 907 was run. Each of three fixed SGD LR pairs received 512 foundation updates (batch size 4). In the valid third attempt, training-template marker accuracy reached 100%, while class-token accuracy remained at four-class chance (26.56%, 25%, 25%) and calibration marker accuracy was 0% for every candidate. Full-vocabulary two-token exact match was 26.56%, 25%, 25% on training and 0% on calibration. None met the predeclared gate of train EM >=0.80 and calibration EM >=0.50.

| Base / expert LR | Train full-vocab 2-token EM | Calibration full-vocab 2-token EM | Train marker accuracy | Calibration marker accuracy | Qualified |
|---:|---:|---:|---:|---:|:---:|
| 0.005 / 0.01 | 0.265625 | 0.000000 | 1.000000 | 0.000000 | No |
| 0.02 / 0.04 | 0.250000 | 0.000000 | 1.000000 | 0.000000 | No |
| 0.05 / 0.10 | 0.250000 | 0.000000 | 1.000000 | 0.000000 | No |

Class-token train accuracy was 0.265625, 0.25, and 0.25; calibration class-token accuracy was 0.1875, 0.25, and 0.25. The four-class chance level is 0.25. Train per-answer-token full-vocabulary NLL was 0.7178, 0.7542, and 0.8075 nats; calibration NLL was 2.3969, 2.2208, and 1.9322. NLL improved relative to the pre-foundation model, but class accuracy remained near chance.

The protocol stopped at the old-task foundation gate. No new-task wake, Sleep update, extra-replay calibration, main run, or held-out scoring occurred. Zero held-out rows were generated or accessed. The CPU lifecycle/transaction fixture passed its 64-event checks, including restore across event 15to16 and slow-then-fast handling at event 64; this only validates CPU transaction logic and is not evidence that a 3B Sleep update ran or passed.

## Task and model

The task was generated synthetic lookup facts with 64 old and 64 disjoint new entities, four balanced class labels per phase, 256 old training rows and 64 old calibration rows. Each fact had four training templates and one distinct calibration template. Training prompts used varied “Answer:”/“Code:” cues; calibration used “The code is”. The answer format was exactly two tokens (shared marker plus one class token), with no EOS target. Marker learning on seen templates alongside zero calibration marker accuracy makes cue/template sensitivity a material limitation. It does not by itself explain the near-chance class accuracy on training or establish why calibration failed.

The model was the cached ModelScope `LLM-Research/Llama-3.2-3B`, revision `48d84dd14886f01964f72e7f742a404dd86f9fb6`, BF16 frozen backbone and FP32 mutable last-MLP CMS parameters, rank 16, capacity 8, two levels, periods `[4,16]`. The run used one RTX 5090 D, PyTorch 2.8.0+cu128 and eager attention. This is a synthetic single-seed feasibility probe, not a natural-language benchmark or an acquired-old-task retention test.

## Attempts and accounting

1. **Initial attempt, failed before training:** zero updates. The adapter provenance hash called `.numpy()` directly on a BF16 tensor and failed with unsupported `ScalarType BFloat16`. Model shards had loaded, but model training had not started. Failure record: `outputs/x870e-synthetic-dev-v2/run-20261105T000000Z/attempt-failure.json`.
2. **Second attempt, engineering/logger failure:** all three candidates completed 512 foundation updates each and train/calibration scores were computed in memory, but qualification/result serialization raised `KeyError: 'full_vocab_greedy_exact_match'`; the scorer exported `full_vocab_two_token_greedy_exact_match`. No metrics were persisted and no gate result is justified for this attempt. Its `dev-results.json` status string `no_lr_pair_met_fixed_train_calibration_gate` is not a valid scientific negative and is superseded by `qualification-not-evaluated-due-logger-keyerror.json`.
3. **Corrected engineering repeat, valid scored attempt:** another 1,536 foundation updates across the same three LR pairs. This attempt persisted metrics; none qualified. Active elapsed time was 87.8267 s (CUDA event time by candidate 30.204, 28.511, and 28.977 s). Peak CUDA allocation was 7,026,405,376; 7,027,191,808; and 7,027,191,808 bytes.

Total foundation work across the second and third attempts was **3,072 updates**: 1,536 from the logger-failed attempt with no persisted score, plus 1,536 in the valid scored repeat. The first attempt had zero updates. Cumulative wall time from the fixed budget start was **1,030.0394 s** (17 min 10.04 s); the valid repeat itself took 87.8267 s. There was no endpoint extension.

Per candidate in the valid repeat: 512 training forward calls, 49,352 forward input-token positions (including padding), 49,352 gradient-forward input-token positions, and 46,265 supervised non-padding forward tokens. Scoring used 80 forward calls, 15,584 forward input-token positions, and zero gradient-forward tokens. Across both trained attempts, aggregate accounting records 3,552 forward calls, 389,616 padded input-token positions, 296,112 gradient-forward input-token positions, and 321,930 non-padding input-token positions. These are work-counter positions, not unique examples or FLOPs.

## Reproducibility pins and artifacts

- Runner `tools/run_x870e_synthetic_dev_v2.py`, SHA-256: `6c7bf0a079a052eb87458010c1b55e52db1d9daa6dce99e55708b2da98385948`.
- Config `configs/x870e-synthetic-dev-v2.json`, raw file SHA-256: `e182d66c1e1c84c29115db70c72971fd80674b2f7ea30bcc8a10dcd734435e01`; run's canonical config SHA-256: `a6a6b06e1e5fa6833f0463fb298b4767ef628e8d674fc836eca2196cd47d41e0`.
- Run-pinned nested source hashes are recorded in `outputs/x870e-synthetic-dev-v2/run-20261105T204000Z-engineering-repeat/dev-provenance.json` (including `lifecycle.py` `906e535a2915102ee986f59e4c9cb26945799c8a149a3cb5eb3a76c7abaa50b5`, `lm.py` `25c26fe79c4d84a16bdaf2d321fee570eb99dc146d041ea2fa0e406400bd227c`, and the remaining module hashes there).
- Cached model shards recorded in provenance: `model-00001-of-00002.safetensors` SHA-256 `584d8d3e3f82f7964955174dfe5e3b1cf117a9d859f022cfdf7fcb884856e002`; `model-00002-of-00002.safetensors` SHA-256 `4719a04514ec2f060240711b7c33ab21187cac730ecaba3040b7a0fd95a9cefb`.
- Valid output: `outputs/x870e-synthetic-dev-v2/run-20261105T204000Z-engineering-repeat/`; prior attempt records are under `run-20261105T000000Z/` and `run-20261105T202730Z-attempt2/`.

The runner used one synthetic DEV namespace (`DEV907`), not three seeds. Private foundation checkpoints were written under the output namespace; they are not part of this report draft or a release artifact. Main/test roles were not touched.
