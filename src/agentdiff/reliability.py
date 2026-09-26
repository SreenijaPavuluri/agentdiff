"""Per-task and suite reliability: pass@1 / pass@k / pass^k with intervals, flaky ranking."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from agentdiff.schema import Trajectory, group_by_task
from agentdiff.stats.estimators import pass_at_k, pass_hat_k
from agentdiff.stats.intervals import per_task_interval, suite_bootstrap

Label = Literal["stable-pass", "stable-fail", "flaky"]


class Estimate(BaseModel):
    value: float
    lo: float
    hi: float


class TaskReliability(BaseModel):
    task_id: str
    n: int
    c: int
    infra_errors: int
    label: Label
    metrics: dict[str, Estimate]  # keys like "pass@1", "pass@4", "pass^4"
    gain_pass_hat_k: float = 0.0  # suite pass^k gain if this task always passed
    gain_pass1: float = 0.0


class SuiteReliability(BaseModel):
    agent_version: str
    n_tasks: int
    runs_per_task: int
    ks: list[int]
    metrics: dict[str, Estimate]
    tasks: list[TaskReliability]
    flaky_ranking: list[str]
    stable_fail: list[str]
    infra_errors: int
    ci_method: str = "per-task: Clopper-Pearson mapped through p^k; suite: two-stage bootstrap (B=5000)"


def classify(n: int, c: int) -> Label:
    if c == n:
        return "stable-pass"
    if c == 0:
        return "stable-fail"
    return "flaky"


def default_ks(n: int) -> list[int]:
    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= n]
    if n not in ks:
        ks.append(n)
    return ks


def analyze_reliability(trajs: list[Trajectory], ks: list[int] | None = None, alpha: float = 0.05,
                        seed: int = 0) -> SuiteReliability:  # fmt: skip
    groups = group_by_task(trajs)
    counts = []
    tasks: list[TaskReliability] = []
    infra_total = 0
    for tid, runs in groups.items():
        valid = [r for r in runs if r.outcome.infra_error is None]
        infra = len(runs) - len(valid)
        infra_total += infra
        n, c = len(valid), sum(r.outcome.passed for r in valid)
        if n == 0:
            continue
        counts.append((n, c))
        tasks.append(TaskReliability(task_id=tid, n=n, c=c, infra_errors=infra, label=classify(n, c), metrics={}))
    if not tasks:
        raise ValueError("no valid runs")
    n_arr = np.array([n for n, _ in counts])
    c_arr = np.array([c for _, c in counts])
    n_min = int(n_arr.min())
    ks = [k for k in (ks or default_ks(n_min)) if k <= n_min]
    k_rep = max(ks)
    t = len(tasks)

    suite: dict[str, Estimate] = {}
    for k in ks:
        for name, metric, fn in (("pass@", "pass@k", pass_at_k), ("pass^", "pass^k", pass_hat_k)):
            if k == 1 and name == "pass^":
                continue  # pass^1 == pass@1
            key = f"{name}{k}"
            m = "pass@1" if k == 1 else metric
            v, lo, hi = suite_bootstrap(n_arr, c_arr, m, k, alpha, seed=seed)
            suite[key] = Estimate(value=v, lo=lo, hi=hi)
            for tr in tasks:
                lo_t, hi_t = per_task_interval(m, tr.c, tr.n, k, alpha)
                tr.metrics[key] = Estimate(value=fn(tr.n, tr.c, k), lo=lo_t, hi=hi_t)
    for tr in tasks:
        tr.gain_pass_hat_k = (1 - pass_hat_k(tr.n, tr.c, k_rep)) / t
        tr.gain_pass1 = (1 - tr.c / tr.n) / t
    flaky = sorted((tr for tr in tasks if tr.label == "flaky"), key=lambda x: (-x.gain_pass_hat_k, x.task_id))
    return SuiteReliability(
        agent_version=trajs[0].agent_version, n_tasks=t, runs_per_task=n_min, ks=ks, metrics=suite, tasks=tasks,
        flaky_ranking=[x.task_id for x in flaky], stable_fail=[x.task_id for x in tasks if x.label == "stable-fail"],
        infra_errors=infra_total,
    )  # fmt: skip
