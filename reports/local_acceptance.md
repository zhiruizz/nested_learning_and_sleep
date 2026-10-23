# Local CPU acceptance — 2026-10-23

Independent Python3.11 environment, PyTorch2.8.0+cpu, NumPy2.2.6, pytest8.4.2;
no system site packages. A new file-only temporary export was installed regularly,
with no editable install, PYTHONPATH or user site import. The resolved import was
the new environment's site-packages, not an existing research checkout. CUDA build
support was absent and CUDA remained uninitialized.

| Check | Actual outcome |
|---|---|
| Package build/install and all module imports | PASS |
| Numerical gradients, CMS timing/state, isolation and lifecycle tests | Final55 passed,0 failed,0 skipped |
| Clock/state, reset/activate and structural examples | All three executed successfully on CPU |
| Historical table reanalysis | Summary generated from all six histories and all12 registered timing rows |
| Llama missing-path rejection and CLI help | Correct refusal / help; metadata fixture returned NOT_RUN |
| Dependency consistency | pip check passed |
| Llama/GPU/sampling/scientific training | NOT_RUN |

First acceptance attempt:54 passed,1 failed,0 skipped. A new synthetic fixture
accidentally made both A and B gradients zero because input and A were orthogonal.
The fixture's A values were corrected to demonstrate a reachable nonzero B-gradient
with zero A-gradient at B=0; no inherited algorithm, historical result, seed or
scientific gate changed. The second fresh export passed all55 tests. Original
failure output is retained separately for local review.

Across both runs: 10 optimizer.step attempts plus
6 functional CMS adaptation attempts, total16/256;
no failed optimizer attempts or warmup attempts. Pure derivative-oracle evaluations
are not optimization trajectories. Examples and historical reanalysis use zero
optimizer attempts. CPU acceptance completed in under one minute, below60 minutes;
dependency download/install time is separate. No original environment was upgraded.

This acceptance checks the listed mechanisms. It does not verify historical3B
outcomes, a complete Sleep recipe, terminal lifecycle or GPU acceleration.
