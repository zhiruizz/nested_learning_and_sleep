# Known limitations

Native deep feature-DGD retention, query/gating, normalization, chunk anchors and
objective conventions are explicit choices. Small-model acquisition failure is
retained, not corrected by release packaging. SMT fast-state-to-CMS consolidation,
full training recipes, efficient kernels and paper benchmarks remain incomplete.

The frozen Llama path adds two CMS-inspired residual low-rank levels only after
the last MLP. A CMS base update is distinct from an original backbone update.
Sender retention, zeroing its gate, excluding it from route selection, and reset
are distinct interventions. Reset preserves A, clears B and deactivates the slot;
it is not full parameter reinitialization. Untrained zero-output receivers can
still alter softmax gates. A reachable A can have zero gradient when B=0.

Low-level SleepSession is an in-memory research API. Its terminal sender is
rejected. CMSController permits simultaneous proposals but does not define their
scientific resolution order. True boundary64, slowest-layer reuse and terminal
consolidation are unverified. CPU guards do not implement these missing policies.

Exact-output LTI judging is a semantic proxy. Live GKD sampling, LTI/Dreaming,
arbitrary external reward-model resume and new scientific quality were NOT_RUN.
The historical fixed-bank interventions are not live on-policy reproduction.

Small repeatedly reused development sets, repeated trajectories, shared teacher
ancestry and multiple checkpoints are not independent samples. Local restoration
can coexist with old/new accuracy losses; probability imitation and correctness
are different objectives. A teacher that failed to acquire a task cannot establish
its preservation. The complete later acquisition gate passed 0/6 histories.

Structural floors apply only to fixed inputs, targets, weights, lifecycle,
precision and a receiver-only parameter domain with verified graph dependencies.
Unknown premises remain UNKNOWN; a zero derivative is insufficient proof.
Reachability does not prove rank capacity, optimizer success, accuracy, general
impossibility or future learning behavior. Signed FP32 roundoff is retained;
normalization from saved logits in FP64 would not be an FP64 model execution.

CACHE timing is scoped to a small repeated bank, not general training acceleration.
HOST passed numerical checks but failed the adoption timing rule. BOTH has an
incomplete performance pair, including one unknown in-flight outcome. Earlier KV
and logits_to_keep candidates failed and were not adopted. No GPU equivalence or
speed claim was revalidated in this release; unknown scope defaults/refuses safely.

Precise historical training cannot be independently repeated from this package.
Missing private state is disclosed rather than replaced with synthetic 3B numbers.
None of these limitations establishes that the original papers are invalid.
