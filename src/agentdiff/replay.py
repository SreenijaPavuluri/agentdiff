"""Replay confirmation of divergence candidates.

For a candidate step s (taken from the divergence ranking) and a few failing base runs:

* ``resample`` arm: replay the recorded prefix, re-issue step s live, run the rest live;
* ``keep`` arm: replay the prefix *and* the recorded (failing) step s, run the rest live.

Delta = pass_rate(resample) - pass_rate(keep). If step s caused the failure, re-drawing it
should raise the pass rate relative to keeping it. The step is **confirmed causal** iff the
Newcombe 95% CI for Delta excludes zero (lower bound > 0). Candidates are tested in rank
order and testing stops at the first confirmed one, so the replay budget is spent on 1-3
steps rather than every step of the run.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel

from agentdiff.divergence import DivergenceResult, project
from agentdiff.runner import Harness, fork_run
from agentdiff.schema import Trajectory
from agentdiff.stats.intervals import newcombe_diff
from agentdiff.tape import Budget, BudgetExceeded, Pricing, ReplayDivergence, UnsafeReplay


class ArmResult(BaseModel):
    n: int = 0
    passed: int = 0
    replay_divergences: int = 0
    infra_errors: int = 0


class CandidateTest(BaseModel):
    label: str
    column: int
    fork_steps: dict[str, int]  # base run id -> step index forked at
    resample: ArmResult
    keep: ArmResult
    delta: float | None = None
    lo: float | None = None
    hi: float | None = None
    status: Literal["confirmed", "not-confirmed", "skipped-budget", "skipped-no-step", "unsafe"]
    cost_usd: float = 0.0
    wall_s: float = 0.0
    tokens: int = 0


class ReplayReport(BaseModel):
    task_id: str
    tests: list[CandidateTest]
    confirmed_label: str | None
    predicted_labels: list[str]  # confirmed first, then remaining divergence ranking
    total_cost_usd: float
    total_wall_s: float
    total_forks: int


def _fork_index(run: Trajectory, ref: Trajectory, kind: str, j: int) -> int | None:
    cols = project(run, ref)
    if kind == "step":
        s = cols.step[j]
        return s.index if s is not None else None
    return cols.ins[j][0].index if cols.ins[j] else None


def confirm(
    harness: Harness,
    div: DivergenceResult,
    failing: list[Trajectory],
    passing: list[Trajectory],
    *,
    n_forks: int = 8,
    max_base_runs: int = 2,
    top_k: int = 3,
    alpha: float = 0.05,
    budget: Budget | None = None,
    pricing: Pricing | None = None,
    seed0: int = 100_000,
    stop_at_first: bool = True,
) -> ReplayReport:
    ref = next(r for r in passing if r.run_id == div.reference_run)
    tests: list[CandidateTest] = []
    seen_labels: set[str] = set()
    seed = seed0
    for ci in div.ranked:
        col = div.columns[ci]
        label = col.fail_label
        if label is None or label in seen_labels:
            continue
        seen_labels.add(label)
        if len(tests) >= top_k:
            break
        bases: list[tuple[Trajectory, int]] = []
        for r in failing:
            idx = _fork_index(r, ref, col.kind, col.j)
            if idx is not None:
                bases.append((r, idx))
            if len(bases) >= max_base_runs:
                break
        test = CandidateTest(label=label, column=ci, fork_steps={r.run_id: i for r, i in bases},
                             resample=ArmResult(), keep=ArmResult(), status="skipped-no-step")  # fmt: skip
        if not bases:
            tests.append(test)
            continue
        if budget is not None and budget.max_usd is not None:
            est = 2 * n_forks * sum(sum(s.cost_usd for s in r.steps[i:]) for r, i in bases) / len(bases)
            if budget.spent_usd + est > budget.max_usd:
                test.status = "skipped-budget"
                tests.append(test)
                break
        started = time.perf_counter()
        spent0 = budget.spent_usd if budget else 0.0
        unsafe = False
        for arm_name in ("resample", "keep"):
            arm: ArmResult = getattr(test, arm_name)
            for f in range(n_forks):
                base, idx = bases[f % len(bases)]
                seed += 1
                try:
                    tr = fork_run(harness, base, idx, arm_name, seed, budget=budget, pricing=pricing)  # type: ignore[arg-type]
                except ReplayDivergence:
                    arm.replay_divergences += 1
                    continue
                except UnsafeReplay:
                    unsafe = True
                    break
                except BudgetExceeded:
                    break
                if tr.outcome.infra_error:
                    arm.infra_errors += 1
                    continue
                arm.n += 1
                arm.passed += int(tr.outcome.passed)
                test.tokens += sum(s.tokens_in + s.tokens_out for s in tr.steps if not s.meta.get("replayed"))
            if unsafe:
                break
        test.wall_s = time.perf_counter() - started
        test.cost_usd = (budget.spent_usd - spent0) if budget else 0.0
        if unsafe:
            test.status = "unsafe"
        elif test.resample.n and test.keep.n:
            d, lo, hi = newcombe_diff(test.resample.passed, test.resample.n, test.keep.passed, test.keep.n, alpha)
            test.delta, test.lo, test.hi = d, lo, hi
            test.status = "confirmed" if lo > 0 else "not-confirmed"
        else:
            test.status = "skipped-budget"
        tests.append(test)
        if test.status == "confirmed" and stop_at_first:
            break
    confirmed = next((t.label for t in tests if t.status == "confirmed"), None)
    predicted = ([confirmed] if confirmed else []) + [lab for lab in div.predicted_labels if lab != confirmed]
    return ReplayReport(
        task_id=div.task_id, tests=tests, confirmed_label=confirmed, predicted_labels=predicted[: max(3, top_k)],
        total_cost_usd=sum(t.cost_usd for t in tests), total_wall_s=sum(t.wall_s for t in tests),
        total_forks=sum(t.resample.n + t.keep.n for t in tests),
    )  # fmt: skip
