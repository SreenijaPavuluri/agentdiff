"""Run a harnessed agent k times per task, and fork recorded runs for replay.

A harness is anything with a ``version`` string and ``run(task_id, tape, seed) -> Outcome``.
All of the agent's model and tool calls must go through ``tape`` so they are recorded and
replayable. Infrastructure failures (exceptions escaping ``run``) are retried and, if they
persist, recorded with ``outcome.infra_error`` so statistics can exclude them.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from agentdiff.schema import ForkInfo, Outcome, Step, Trajectory
from agentdiff.tape import Budget, BudgetExceeded, Intervention, Pricing, ReplayDivergence, Tape, UnsafeReplay


class Harness(Protocol):
    version: str

    def run(self, task_id: str, tape: Tape, seed: int) -> Outcome: ...


def _execute(harness: Harness, task_id: str, tape: Tape, seed: int, retries: int) -> tuple[Outcome, Tape, float]:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        run_tape = tape if attempt == 0 else _fresh(tape)
        started = time.perf_counter()
        try:
            outcome = harness.run(task_id, run_tape, seed)
            return outcome, run_tape, (time.perf_counter() - started) * 1000
        except (BudgetExceeded, ReplayDivergence, UnsafeReplay):
            raise
        except Exception as exc:  # noqa: BLE001 - infrastructure failure, retried
            last_exc = exc
            time.sleep(min(2.0**attempt, 10.0) * 0.1)
    return (
        Outcome(passed=False, infra_error=f"{type(last_exc).__name__}: {last_exc}"[:500]),
        run_tape,
        0.0,
    )


def _fresh(tape: Tape) -> Tape:
    return Tape(source=tape.source, fork_step=tape.fork_step, intervention=tape.intervention,
                transplant=tape.transplant, pricing=tape.pricing, budget=tape.budget,
                store_requests=tape.store_requests)  # fmt: skip


def run_suite(
    harness: Harness,
    task_ids: Iterable[str],
    k: int = 8,
    *,
    concurrency: int = 4,
    retries: int = 2,
    budget: Budget | None = None,
    pricing: Pricing | None = None,
    seed0: int = 0,
    on_result: Callable[[Trajectory], None] | None = None,
) -> list[Trajectory]:
    """Run every task ``k`` times. Seeds are ``seed0 + j`` for the j-th repetition of a task."""
    jobs = [(tid, seed0 + j) for tid in task_ids for j in range(k)]
    out: list[Trajectory] = []
    stop = False

    def one(tid: str, seed: int) -> Trajectory | None:
        if stop:
            return None
        tape = Tape(pricing=pricing, budget=budget)
        outcome, used, wall = _execute(harness, tid, tape, seed, retries)
        return Trajectory(task_id=tid, run_id=uuid.uuid4().hex[:12], agent_version=harness.version, seed=seed,
                          steps=used.steps, outcome=outcome, wall_ms=wall)  # fmt: skip

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, tid, seed) for tid, seed in jobs]
        for fut in as_completed(futures):
            try:
                traj = fut.result()
            except BudgetExceeded:
                stop = True
                continue
            if traj is None:
                continue
            out.append(traj)
            if on_result:
                on_result(traj)
    out.sort(key=lambda t: (t.task_id, t.seed or 0))
    return out


def fork_run(
    harness: Harness,
    source: Trajectory,
    step: int,
    intervention: Intervention,
    seed: int,
    *,
    transplant: Step | None = None,
    budget: Budget | None = None,
    pricing: Pricing | None = None,
    retries: int = 1,
) -> Trajectory:
    """Replay ``source`` up to ``step``, intervene there, and run the rest live."""
    tape = Tape(source=source, fork_step=step, intervention=intervention, transplant=transplant,
                pricing=pricing, budget=budget)  # fmt: skip
    outcome, used, wall = _execute(harness, source.task_id, tape, seed, retries)
    return Trajectory(
        task_id=source.task_id, run_id=uuid.uuid4().hex[:12], agent_version=source.agent_version, seed=seed,
        steps=used.steps, outcome=outcome, wall_ms=wall,
        parent=ForkInfo(source_run=source.run_id, fork_step=step, intervention=intervention),
    )  # fmt: skip
