# Banking V12 main-study provenance

This package contains exact source/config-freeze metadata, actual CPU preflight,
seed-17 disk restore qualification and terminal reviews, and all three root
terminal reviews. It includes the actual failed main startup-A runner/test
preimages, plus the executed test preimage before the public filename-import
portability change. Main startup A ran four wake updates before failing at a
MutableContext.eval call; it generated no rollouts and did no receiver updates.

The source archive is verified against frozen A and its original archive note.
The later auxiliary fit evaluator is a distinct program, with separate failure
and source history under ../fixed-fit. A corrected, unpublished packaging draft
is documented in packaging-correction.json; it changed no scientific execution
or metric. All exact byte bindings are listed in capture-manifest.json.

Freeze B references the historical 7f07 executed fixture; the published check_
fixture has the 83c110 import-portability change only. Both are identified in
public-test-portability-receipt.json. Source-lock identity alone is not paper
reproduction or formal scientific confirmation. Raw examples, labels, tokens,
trajectories, weights and checkpoints are not included.
