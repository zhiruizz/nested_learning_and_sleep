"""Order-preserving adapter for mixed GKD/LTI microbatch transactions.

The objective builder is responsible for V10's tested sampling, prompt,
full-vocabulary KL and local LTI reward construction. It is called once per
selected request, in the frozen GKD,GKD,LTI order, with the transaction's
current teacher/student contexts and generator. Requests inside a microbatch
see the same pre-update student parameters; later microbatches see the committed
update. Thus this is true sampled-policy training with minibatch-stale policy
within a group, not the old per-objective Adam trajectory.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from banking_cil_v13_batching import (
    group_request_indices,
    optimize_mixed_request_batch,
    validate_request_schedule,
)


def sample_owner(kind: str, gkd_ordinal: int) -> str:
    if kind == "lti":
        return "teacher"
    if kind != "gkd":
        raise ValueError("kind must be gkd or lti")
    return "teacher" if gkd_ordinal % 2 == 0 else "student"


def run_batched_sleep_event(
    selected: Sequence[tuple[str, Any, str]],
    tx,
    *,
    microbatch_size: int,
    objective_builder: Callable[..., Mapping[str, Any]],
    budget=None,
    request_offset: int = 0,
    gkd_ordinal_start: int | None = None,
    on_batch_committed: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Run one complete selected request stream as mixed-objective batches.

    ``selected`` rows use the V10 ``(kind,row,role)`` shape. Builder signature:
    ``objective_builder(kind, row, role, owner, tx, request_index)``. It must
    return the batching helper mapping and must not backpropagate or update.
    ``request_offset`` resumes only at a microbatch boundary. Supply the
    number of earlier GKD opportunities in ``gkd_ordinal_start`` to preserve
    alternating GKD sample ownership. ``on_batch_committed(batch_index,
    next_request_offset, next_gkd_ordinal, batch_result)`` runs only after a
    completed update or explicit all-zero LTI skip, when transaction snapshots
    are safe. The returned output retains only builder metadata and aggregate
    counters; the caller controls public/private logging.
    """
    if not callable(objective_builder):
        raise TypeError("objective_builder must be callable")
    if type(microbatch_size) is not int or not 1 <= microbatch_size <= 32:
        raise ValueError("microbatch_size must be an integer in [1,32]")
    if on_batch_committed is not None and not callable(on_batch_committed):
        raise TypeError("on_batch_committed must be callable")
    kinds = [str(item[0]) for item in selected]
    # The frozen 960-request grid ends on a full cycle; the 3200-request
    # extended-dose grid has a documented two-GKD terminal suffix.
    opportunities = validate_request_schedule(kinds, complete_cycles=False)
    if type(request_offset) is not int or not 0 <= request_offset <= len(selected):
        raise ValueError("request_offset is outside the selected request stream")
    if request_offset % microbatch_size and request_offset != len(selected):
        raise ValueError("request_offset must be at a microbatch boundary")
    preceding_gkd = sum(1 for kind in kinds[:request_offset] if kind == "gkd")
    if gkd_ordinal_start is None:
        gkd_ordinal_start = preceding_gkd
    if type(gkd_ordinal_start) is not int or gkd_ordinal_start != preceding_gkd:
        raise ValueError("gkd_ordinal_start does not match the selected prefix")
    # Validate the complete schedule above, then preserve its phase when
    # slicing. A legal batch boundary (e.g. 32) need not be a three-request
    # GKD/GKD/LTI cycle boundary, so revalidating the suffix would be wrong.
    groups = [list(range(i, min(i + microbatch_size, len(selected))))
              for i in range(request_offset, len(selected), microbatch_size)]
    gkd_ordinal = gkd_ordinal_start
    batches, request_summaries = [], []
    for batch_index, indexes in enumerate(groups):
        callbacks = []
        for index in indexes:
            kind, row, role = selected[index]
            ordinal = gkd_ordinal
            if kind == "gkd":
                gkd_ordinal += 1

            def callback(kind=kind, row=row, role=role, owner=sample_owner(kind, ordinal), index=index):
                result = dict(objective_builder(kind, row, role, owner, tx, index))
                if result.get("kind") != kind:
                    raise ValueError("objective builder changed scheduled kind")
                if "metadata" in result:
                    request_summaries.append({"index": index, "kind": kind,
                                              "owner": owner, "metadata": result["metadata"]})
                return result

            callbacks.append(callback)
        batch = optimize_mixed_request_batch(tx, callbacks, budget=budget)
        batches.append(batch)
        if on_batch_committed is not None:
            try:
                on_batch_committed(batch_index, indexes[-1] + 1, gkd_ordinal, batch)
                tx._check()
            except BaseException:
                # Persistence hooks run after the update is committed. If they
                # fail or mutate guarded state, do not permit continuation from
                # an ambiguous checkpoint boundary.
                tx.counters["closed"] = True
                tx.counters["v13_failed"] = True
                raise
    if gkd_ordinal != opportunities["gkd"]:
        raise RuntimeError("GKD owner cursor mismatch")
    return {
        "opportunities": opportunities,
        "processed_opportunities": len(selected) - request_offset,
        "request_offset_start": request_offset,
        "request_offset_end": len(selected),
        "gkd_ordinal_start": gkd_ordinal_start,
        "gkd_ordinal_end": gkd_ordinal,
        "microbatch_size": microbatch_size,
        "microbatch_count": len(batches),
        "physical_adamw_steps": sum(int(x["physical_adamw_steps"]) for x in batches),
        "backward_calls": sum(int(x["backward_calls"]) for x in batches),
        "skipped_zero_gradient_batches": sum(bool(x["skipped_zero_gradient"]) for x in batches),
        "batches": batches,
        "request_summaries": request_summaries,
    }
