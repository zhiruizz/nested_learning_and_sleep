# Ubuntu P2 provenance and disclosure follow-up

See [the updated descriptive report](../../reports/ubuntu-p2-formal-descriptive-disposition.md).
Original-protocol acceptance remains rejected and the science aggregate is unchanged.

The model config establishes a Llama-family architecture, not an exact upstream
repository or immutable revision. A candidate historical contract describes
CLINC OOS-eval data and shares the formal train/dev payload digest, but its own
contract SHA differs from the formally bound P2 contract. Dataset release and
old/new mappings remain unresolved; do not infer contract equivalence.

The corrected incident records an additional training-schedule metadata exposure
without reproducing identifiers. Its exact event UTC timestamp is unknown;
record creation time is separate. It is distinct from the earlier development
endpoint exposure. Neither incident changes the completed computation, and the
sealed confirmation set was not read.

Public source metadata omits one private filesystem path. The full follow-up
review projection applies the independently checked r2 date/access-field
correction and record binding. Original/derived hashes and these transformations
are explicit in artifact-provenance.json. Private execution-workspace filenames
inside review records identify evidence; not all referenced records are bundled.
No raw stdout, row identifiers, text, labels, token sequences or weights are included.
