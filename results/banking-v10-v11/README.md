# Banking77: V10 controls and V11 reward ablation

These files contain aggregate project measurements and reproducible analysis code. They do not contain requests, generated answers, tokenizer IDs, pretrained weights, adapter tensors or optimizer states. This is a local Llama-3.2-3B last-MLP study; native HOPE and the full Sleep manuscript benchmarks are not reproduced.

The [V10 report](../../reports/banking77-controls-v10.md) gives the three-seed, three-arm comparison on all 3,080 publisher test rows and a separately reported 3,073-row train/development-disjoint subset. The [V11 report](../../reports/banking77-reward-ablation-v11.md) describes two additional reward controls and the matched five-arm task-1 development evaluation. Its 220 rows per seed are a distinct population, and its findings are development exploration. The three orders share one initialization; error bars are sample standard deviations, not confidence intervals.

`training-seal.json` and `v10-official-evaluation.json` retain their original bytes and hashes. The seal contains historical D-drive run-directory strings. Those strings are inert provenance: the public reanalyzer reads only the aggregate JSON explicitly supplied on its command line and never follows run or checkpoint paths. A fresh machine must regenerate location-bound execution contracts rather than reuse archived local paths.

Run from the repository root, using new output filenames:

```sh
python tools/reanalyze_banking_v10_v11.py --v10-seal results/banking-v10-v11/training-seal.json --v10-test results/banking-v10-v11/v10-official-evaluation.json --v11-evaluation results/banking-v10-v11/v11-development-evaluation.json --output outputs/banking-reanalysis.json --csv outputs/banking-reanalysis.csv
python tools/plot_banking_v10_v11.py --summary outputs/banking-reanalysis.json --output-dir outputs/banking-figures
```

The reanalysis itself uses Python's standard library. Plotting additionally requires Matplotlib; it performs no model or dataset access. Output JSON records the supplied file paths, so a relocated reanalysis need not be byte-identical to the archived summary. Its counts, scalar metrics and derived statistics should match. Omit `--v11-evaluation` for V10-only analysis. Existing output files are refused.

Primary accuracy ranks the global 77-label dictionary by summed conditional log probability including EOS. Token-mean ranking, unrestricted greedy exact match and per-example gold-token mean NLL remain separate readouts. Final old/current class partitions are cross-sectional endpoint measurements, not before/after forgetting deltas.

The accounting files separate shared foundations, main continuations, qualification work, failed attempts and joint evaluation. Missing counters are unknown, not zero. Reserve storage, logical activation, branch work and extra data/label metadata are distinct. Wall time is not measured FLOPs or isolated deployment latency. The operation ledger preserves failures and the dates/scopes of auxiliary repairs; software acceptance receipts are engineering evidence rather than evidence of scientific efficacy.

See [asset acquisition and execution](../../docs/banking-v10-v11-reproduction.md), the [frozen V10 protocol](../../docs/banking77-v10-preflight-protocol.md), and the [paper alignment audit](../../docs/sleep-paper-alignment-v10-v11.md). Banking77 attribution and its CC BY 4.0 license are recorded there. Model files remain governed by the Llama license and are not redistributed.
