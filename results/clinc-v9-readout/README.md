# CLINC V9 aggregate-only development diagnosis

`readout-result.json` is the curated accepted **d** result. It preserves the
original aggregate schema, source bindings, count denominators, work counters
and immutable-state checks. `_public_capture.source_artifact_sha256_raw` records
the exact private raw artifact's digest. File hashes of this curated release
are separately recorded in `public-capture.json` and `RELEASE_MANIFEST.json`.

`operational-costs.json` distinguishes pre-scoring technical failures a/b,
quarantined c and accepted d. C's scores were not used for selection and are not
redistributed here; only its work counters are included. The historical a/b
source text files reproduce the actual buggy source bytes for inspection and
are not normal execution entry points. They remain code under the repository's
code license. Unknown failed-setup duration stays null, not zero.

`existing-development-audit.json` is the earlier pre-V9-results audit snapshot,
not a current job-status receipt. Stochastic teacher sample counts are separate
from greedy accuracy and from the six-run main statistics. `context-bindings.json`
is provenance-only with normalized local paths; use the portable builder to
create a real local runtime map. The CPU preflight and builder-qualification
records contain no model/checkpoint payloads.

Recompute the public statistics without model or dataset access:

```sh
python tools/reanalyze_clinc_readout_v9.py --input results/clinc-v9-readout/readout-result.json --out-dir outputs/clinc-v9-public-reanalysis
```

See the [report](../../reports/clinc-readout-and-paper-alignment-v9.md),
[reproduction instructions](../../docs/clinc-readout-v9-reproduction.md), and
[Banking77 preparation](../../docs/banking77-matched-comparison-draft.md).
This is one seed on 60 development rows; it is not fresh confirmation. No
Banking77 train/test file, CLINC row, generated answer, token array, model weight
or private checkpoint is distributed in this directory.
