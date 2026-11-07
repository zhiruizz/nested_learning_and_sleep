# Banking V11 serial orchestration

`tools/orchestrate_banking_v11_canonical_teacher.py` wires the frozen six-run
V11 task-1 grid without changing its training driver or child config. The
orchestrator reads only the path-map JSON and the completed V10 official-test
evaluation aggregate; it never opens the official test CSV or its rows.

Run `prepare` only after the V10 evaluation artifact reports the completed
schema and immutable, zero-update flags. It invokes the existing
`prepare_banking_v11_launch_materials.py build-contracts`, which performs the
full nine-run/test aggregate checks, source/config binding, and creates six
contracts. The command creates the configured empty V11 output root first;
any contract-preparation failure is recorded and launches no jobs.

Run `run` to execute the contracts in seed-major order, one fresh Python
process per job. Each result must have a complete V11 run receipt, fixed
task-1 checkpoint, matching launch record, and successful deferred-memory
callback. The first failure stops the queue. Existing complete jobs are
verified and skipped; partial or failed outputs are never restarted. A
serial status record prevents rerunning the queue automatically.

After all six jobs pass those checks, `prepare-dev-eval` invokes the separate
strict 15-context reference builder. It validates the fixed development
cohorts and CPU-deserializes checkpoint payloads, then writes the V11
development-memory contract after validating its schema and source pins. It
does not start development scoring; the reference and contract are left for
independent review before a separate launch command.

Use the tested Python environment explicitly, for example:

```powershell
conda activate nl-sleep
python tools/orchestrate_banking_v11_canonical_teacher.py prepare --paths banking_v10_v11_paths.json
python tools/orchestrate_banking_v11_canonical_teacher.py run --paths banking_v10_v11_paths.json
python tools/orchestrate_banking_v11_canonical_teacher.py prepare-dev-eval --paths banking_v10_v11_paths.json
```

The first command is gated on the real completed B evaluation. The second
starts GPU work after its resource and artifact checks. The third
performs a CPU-only reference/checkpoint preflight and does not launch the
development evaluator.

For a fresh replication, create the path map from
`configs/banking_v10_v11_paths.example.json` and replace every placeholder.
The published checkout can serve as both `task_root` and `project_root`.
Keep model assets, foundations, run outputs and preparation records outside
that source checkout: the memory-policy wrappers reject output paths inside
the protected `project_root`. The template uses forward slashes, accepted on
Windows as well as Linux. Create the preparation parent directories first. Generate
fresh V10/V11 configs, releases, and seals for that run; the frozen configs
currently in this workspace are historical references bound to a different
seal and must not be reused as if they belonged to the new replication.
