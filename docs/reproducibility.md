# Reproducibility and assets

Use a fresh environment without system site packages. Clear PYTHONPATH and
PYTHONHOME, set PYTHONNOUSERSITE=1, CUDA_VISIBLE_DEVICES=-1, HF_HUB_OFFLINE=1,
TRANSFORMERS_OFFLINE=1 and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. Install a wheel or
regular package, never an editable link to another checkout. Inspect
`python -c "import nested_sleep; print(nested_sleep.__file__)"` to confirm that
the installed package belongs to this environment. CPU tests deny socket
connections and CUDA lazy initialization, including test collection.

Installation is separate from offline acceptance: acquire dependencies first or
prepare a local wheelhouse. The tested dependency versions and lightweight SBOM
are under requirements/. A lock describes one CPU environment, not every platform.

Run the README examples and `python -m pytest -q`. All synthetic fixtures are
created locally. Reference tests perform three tiny AdamW attempts, the synthetic
fixed-sequence receiver check two more, and CMS adaptation tests perform three
functional adaptation calls. Charge-before-attempt accounting lives in
tests/conftest.py. Set RELEASE_BUDGET_FILE to a new local JSON path to persist
counts across runs. The release run uses a shared ledger capped at 256 and a
coordinator timeout of 60 minutes. No warmup or reference optimization is exempt.

`python tools/reanalyze.py --results results` reconstructs a historical summary.
Reanalysis reads CSVs only: it does not recreate training or recover missing weights.
The summary is a derived artifact; original CSV metric values are unchanged.

`python -m nested_sleep.llama_entry --model-path YOUR_LOCAL_MODEL_DIRECTORY`
checks metadata and arguments only. It rejects missing paths, never downloads or
searches disk, never loads weights, and returns NOT_RUN. The library LlamaCMSLM
accepts a caller-created backbone; the caller must freeze original parameters,
define the exact training scope and retain applicable model/data terms. This release
provides no complete scientific training driver and no approved CACHE runtime scope.

Historical exact reruns additionally require nonredistributed initial adapters,
teacher trajectory banks, source-role assignments, full vocabulary targets, RNG,
optimizer and lifecycle states. They are HISTORICAL_ONLY. File digests cannot
substitute for these assets. No old checkpoint, raw token sequence, dataset record
or model weight is included. Data readers and acquisition runners were excluded.
