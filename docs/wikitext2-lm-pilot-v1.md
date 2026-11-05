# Llama 3.2-3B WikiText train-window pilot

This is a preregistered, fixed-budget engineering pilot of language-model adaptation and a partial Sleep-style consolidation path. It is not a reproduction of either paper's reported benchmark results.

## Frozen protocol

The base model is the locally acquired ModelScope `LLM-Research/Llama-3.2-3B` at revision `48d84dd14886f01964f72e7f742a404dd86f9fb6`. Its local ModelScope manifest and Llama 3.2 Community License / acceptable-use policy are retained beside the private weight cache. The experiment publishes no weights, tokenizer files, or derivatives. The frozen backbone runs in BF16; trainable adapter/master parameters are promoted to FP32 before optimization.

The source is `Salesforce/wikitext`, `wikitext-2-raw-v1`, from the `train` Parquet shard at `refs/convert/parquet` commit `f3fd811233cf8abb79006d3a5f566c280bf47e1a`. Its SHA-256 is `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`. The source page's license metadata lists CC-BY-SA-3.0 and GFDL while the card text says CC BY-SA 4.0; this discrepancy is retained here. Source text and prepared snippets remain in the ignored local `outputs/` cache and are not published.

Articles are recognized only by a single-equals top-level title such as `= Article =`. Double- and triple-equals headings stay inside their article. The fixed partition hashes the case-folded, whitespace-normalized title with salt `nl-sleep-wikitext2-article-split-v1`, then assigns the first eight SHA-256 hex digits modulo 100: 0–59 adaptation, 60–79 calibration, and 80–99 evaluation. The parsed shard contained 620 top-level titles, 607 with at least 700 characters, and 347/134/126 eligible articles in the three roles. The selected eight adaptation, eight calibration, and sixteen evaluation article IDs are pairwise disjoint and recorded only as hashes. Calibration is reserved and unused in v1.

The model sees eight adaptation windows per seed and update arm, batch size 1, context length 96, and 95 next-token targets per example. Seeds are 17 and 29; the same eight windows are shuffled deterministically per seed. The arms are frozen, sequential SFT, CMS wake, and CMS wake followed by one GKD/LTI consolidation session. All wake arms share the same data, sequence length, and step count. There is no early stopping or extension based on results.

The architecture is the repository's `LlamaCMSLM`: two rank-4 residual memory adapters and sparse expert banks attached to the final MLP, with a frozen Llama backbone. Sequential SFT applies the controller's per-minibatch gradient update to both base memory scopes and the first active expert. Both scopes are due every SFT minibatch; proposals are applied in ascending level order (0 then 1) from the same pre-update gradient batch. CMS uses local periods `(4, 16)`. At step 4, level 0 is applied and consumed. At step 8, level 0 is the prospective sender update passed once to `SleepSession`; it is not applied beforehand. The level-1 base clock has accumulated eight of sixteen minibatches and remains pending, so this pilot does not test a slow-tier base write or its terminal lifecycle.

The consolidation arm reuses `nested_sleep.lifecycle.SleepSession` with a shared frozen backbone, sender 0, receiver 1, two autoregressive GKD steps and one LTI step, each configured for four generated tokens. It uses the local exact-output-plus-edit reward proxy in LTI. Dreaming is disabled. Candidate commit depends only on finite values, unchanged frozen backbone, and the session's allowed-mutable-scope checks; no validation or evaluation loss is used to accept/reject or select a candidate. This is a partial GKD/LTI transfer pilot, not a full implementation or reproduction of *Language Models Need Sleep*.

## Verification and evaluation boundary

Before the main run, a fixed fixture trains one four-example sender proposal, runs a GKD step, saves a compact `SleepSession` checkpoint, restores it twice, and requires the next GKD result to have a byte-identical mutable-state fingerprint. It also verifies the live sender remains unchanged. The fixture is engineering work and is reported separately from the main arms.

Training reads only the adaptation role. The evaluation-role JSON is first opened by the separate `evaluate` stage after every seed/arm has written its adapter checkpoint and `training-sealed.json`. It scores the sixteen article-disjoint windows once, reporting mean next-token cross-entropy and perplexity. This is a small, train-derived held-out probe, not official WikiText-2 validation/test perplexity and not comparable to upstream benchmark scores. It does not establish task acquisition, durable knowledge retention, or general continual-learning performance.

Every arm reports per-step losses, all counted model forward-input tokens by stage (including teacher/student sampling and LTI), total wall time, CUDA event interval, and peak allocated CUDA memory. The KS arm receives additional compute and is not compute-matched to wake-only arms. Failures and both seeds remain in the results. No run artifact contains source text or model weights.

## Reproduce locally

First obtain the exact source shard from the dataset repository and place it under an ignored `outputs/` directory. Prepare the local role cache with an interpreter that has PyArrow; this can be a separate preprocessing interpreter and does not require changing the model environment. Then run the train stage with the local ModelScope model directory and its existing asset manifest:

```powershell
python tools/prepare_wikitext2_pilot.py --parquet outputs/wikitext-lm-pilot-v1/data/train-00000-of-00001.parquet --out-dir outputs/wikitext-lm-pilot-v1/data/roles
python tools/run_wikitext2_lm_pilot.py --stage train --config configs/wikitext2-lm-pilot-v1.json --model-dir D:/path/to/llama32-3b --model-manifest D:/path/to/llama32-3b-manifest.json --data-dir outputs/wikitext-lm-pilot-v1/data/roles --out-dir outputs/wikitext-lm-pilot-v1/run
python tools/run_wikitext2_lm_pilot.py --stage evaluate --config configs/wikitext2-lm-pilot-v1.json --model-dir D:/path/to/llama32-3b --data-dir outputs/wikitext-lm-pilot-v1/data/roles --out-dir outputs/wikitext-lm-pilot-v1/run
```

The scripts refuse source-hash, role-count, role-overlap, and sealed-config mismatches. Dataset/model assets and checkpoints are local-only; only the code, protocol, and aggregate summaries belong in the repository.
