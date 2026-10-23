# Public provenance

The numerical implementation is inherited from the local research project's
independent reference code under its existing MIT notice. Local source review
records state that comparative third-party implementations were read, not vendored.
No upstream research repository source tree or paper snapshot is included.

Inherited modules: __init__, accounting, clocks, consolidation, dreaming,
executor_v13, experts, lifecycle, lm, memory, native, objectives, optim.
Tests reference/native retain their original numerical/state assertions with a
CPU-only device fixture. Packaging changes replace stale documentation references
and change lm.export_config's hardcoded model identity to USER_SUPPLIED_LOCAL_ASSETS.
repairability retains only Assumptions, classify_position, bounds and diagnostic_check;
obsolete interface metadata and saved-evidence adapters are omitted. Its theorem
arithmetic is unchanged. The selector and metadata-only CLI are new wrappers.

Historical CSVs project reviewed aggregate columns, preserve all selected rows,
and exclude raw asset digests and local locations. New public file digests can
identify this candidate. They do not claim historical assets are available.
Execution versus analysis provenance uncertainty is described in results/README.md.

Exact internal locations, source revisions, source/destination hashes, exclusions
and edit diffs are held separately for local review, outside this repository.
