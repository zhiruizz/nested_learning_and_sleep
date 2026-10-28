# Contributing

Contributions of all sizes are welcome: documentation fixes, installation reports,
paper-to-code questions, mechanism tests, independent experiments, and engineering
improvements. Positive, negative, and contradictory results are equally welcome.
You do not need to reproduce either paper in full before participating.

Issues and pull requests may be written in English or Chinese.
欢迎使用中文或英文提问、讨论和提交修改。

## Choose a starting point

- **Small documentation fixes:** submit a focused pull request directly; an issue
  and a model run are not prerequisites.
- **Runtime problems:** describe the commit, relevant environment, exact command,
  expected and observed behavior, and a minimal example where possible. Redact logs
  before sharing them.
- **Paper or design questions:** point to the paper version and section/equation,
  the relevant code, and the interpretation you would like to discuss. Experiment
  results are optional, not an entry requirement.
- **Independent experiments:** state the code/configuration, model and data
  provenance, budgets, metrics, comparators, seeds or orders, and failures. Identify
  unavailable assets and what another reader can actually rerun.

Start with the [paper-to-code mapping](docs/paper_to_code.md),
[reproduction status](docs/reproduction_status.md), and
[limitations](docs/limitations.md). For substantial changes to the architecture,
Llama integration, routing, or memory lifecycle, open an issue before investing in
an implementation. This is a request to coordinate scope, not a requirement for
permission to explore your own fork.

## Local checks

Use a fresh Python 3.11 environment and follow the pinned CPU installation steps in
[README.md](README.md). From the repository root, the existing offline checks are:

```sh
python -m pytest -q
python examples/cms_state.py
python examples/reset_activate.py
python examples/structural_counterexamples.py
python tools/reanalyze.py --results results --output outputs/historical_summary.md
```

The default checks use synthetic inputs, not downloaded datasets or pretrained
weights, and do not initialize a GPU. Keep optional model downloads and GPU
experiments separate from the default test suite. Document their requirements
without implying that they are part of the CPU acceptance run.

For code changes, add a focused regression test and report the checks you actually
ran, including failures and skips. For a documentation-only change, checking the
rendering and links is sufficient; do not claim to have rerun model tests. Passing
unit tests is not evidence of reproducing a paper's benchmark results.

## Keep research claims traceable

Identify whether a change implements a paper-specified mechanism, fills an
unspecified detail with an assumption, or introduces a new experimental variant.
Link the relevant paper version/section and explain the difference. A new variant
is welcome without being presented as the original algorithm.

Preserve historical results and their scope. Corrections should explain what
changed and why, with provenance for the corrected values; do not silently replace
negative results or treat missing runs as successes. New experiments can use their
own clearly documented protocols, but must not be described as if they ran under
an unchanged historical protocol. No particular positive outcome is required.

For performance changes, state the workload, hardware, timing scope, numerical
comparison, and any state or gradient differences. Distinguish synthetic examples,
bundled-results reanalysis, user-supplied-asset runs, and historical-only evidence.
The period-64/terminal lifecycle remains unresolved unless new evidence and an
explicit specification establish otherwise; do not silently choose a policy.

## Submitting a pull request

Keep the patch focused and describe its purpose, changed behavior (if any), checks,
and remaining limitations. Documentation-only and algorithmic changes should be
separable. For changes that affect tracked files, the maintainer can help refresh
`RELEASE_MANIFEST.json` and `SHA256SUMS.txt` before merge; these bookkeeping files
should not prevent a first contribution. Historical release commits remain the
reference for their original checksums.

Use only material you are entitled to share. Do not upload credentials, private
paths or records, restricted weights, raw private data, or private model states.
Link to authorized asset sources and explain missing assets instead. Preserve
applicable copyright notices and attribution; consult [LICENSE.md](LICENSE.md)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the existing asset-specific
license scopes. Do not imply endorsement by the original paper authors or their
institutions.

## Review expectations

This is an unofficial, partial research implementation. Discussion, review, and
merging depend on maintainer availability; no response time, acceptance of every
pull request, or full large-model reproduction support is promised. Disagreement
is welcome when it is specific and supported by code, references, or observations.
