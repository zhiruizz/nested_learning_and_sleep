# P2 formal run: independent final disposition

**Decision: not accepted under the frozen protocol; descriptive reporting only.** The four runs completed, but the All+N checkpoint writer did not read the destination `state.bin` back before publication. Later matching hashes verify the current bytes; they cannot satisfy the original write-time requirement retroactively.

The terminal metadata records four completed 40-step runs (seeds 402/403, both named arms), systemd success, exit 0, cleanup complete, and no worker or monitor error. Recorded work is 4,608 FW, 4,608 BW, 160 logical updates, 180 physical Adam steps, and 11,399 DEV_FW (5,120 fixed probes plus 6,279 endpoint evaluations). These are completion and budget checks, not acceptance by themselves.

The metadata-only snapshot verified 40 checkpoint receipts: DSM paths, state sizes, metadata JSON hashes, and no anomalies. Post-hoc SHA checks cover all 20 All+N checkpoints (steps 1, 2, 4, 6, 8, 12, 16, 24, 32, 40 for both seeds): 700,659,302,400 bytes, every digest matching its receipt and sidecar, no retry. The two-file pilot took 243.336 seconds; the 18-file remainder took 2164.719 seconds. The 20 k4 payloads were not rehashed here; the frozen writer review records destination readback and `COMMITTED` checks.

The sanitized development endpoint summary is seed-sensitive. All+N immediately emitted EOS as the only generated token for all old/current endpoint examples in both seeds. k4 returned non-empty outputs and was correct on 220/298 old and 198/300 current examples for seed 402; in seed 403 it immediately emitted EOS for all old/current examples. This two-seed run does not establish a stable benefit. These are development endpoints, not independent confirmation.

A recorded inspection incident occurred after all four runs were complete: a truncated endpoint JSON prefix exposed some IDs, gold labels, predictions, and loss fields; the number of visible rows is unknown. The incident record says no values were used for tuning or selection. This does not alter the completed computation, but endpoint review is no longer blind. Do not treat these endpoints as confirmatory or reuse them as blind confirmation data. The sealed confirmation set was not read.

No remote terminal was modified, no new job was started, and no checkpoint/model tensor payload was loaded for this disposition. Historical metadata-preflight failures and the sandbox BrokenPipe attempt remain preserved in their own records.

Evidence hashes and exact post-hoc coverage are indexed in `independent-final-disposition-20261108-r1.json`. The original remote `formal_terminal` remains `FORMAL_RUN_COMPLETE_PENDING_INDEPENDENT_REVIEW`. The user authorized descriptive aggregate reporting and disclosure of the protocol deviation; I found no explicit prohibition on that descriptive reporting in the reviewed records. This disposition does not change the terminal or grant formal accepted/released status.
