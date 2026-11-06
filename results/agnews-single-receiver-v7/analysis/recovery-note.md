# Recovery and cost accounting

The original process ended without terminal receipts; the cause is unknown. Its global execution-work ledger is missing and is not reconstructed.

- Logical scheduled updates across the 30 arms: 30,720.
- Known observed update attempts across original progress plus recovery: 31,168.
- Additional unrecorded original attempts are bounded at 0 to 64; this is a count bound after the last persisted progress, not a step-index interval.
- The recovered receiver resumed from step 256 for 768 updates; 448 repeat updates reproduce work already observed through step 704. It has 1,472 observed student forward/backward passes but a 1,024-step logical schedule.
- Recovery run work: 5,926,035,451,666,432 recorded arithmetic units across 177,552 instrumented calls.
- Recovery qualification work: 5,960,223,031,296 recorded arithmetic units across 88 instrumented calls.
- These are the recovery meter's recorded operation counts, not total hardware FLOPs. Original arm-local counters are incomplete overlapping observations and must not be added to the recovery totals.
- Fixed preparation work is represented by 864 reconstructed forward calls; no FLOP estimate is assigned. Main local timing sums are not wall time.
- Qualification failures and successful qualification runs are separate cost entries in `work-ledger.json`; no confirmation/test data were scored.
