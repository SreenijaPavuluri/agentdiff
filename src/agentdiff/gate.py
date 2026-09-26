"""Regression gate: compare a candidate agent version against a baseline on the same tasks.

Decision rule (family-wise false-alarm rate <= alpha):

* overall: stratified exact test (``stats.tests.stratified_exact_test``), one-sided "worse",
  at level alpha/2;
* per task: one-sided Fisher exact test, Holm-adjusted across tasks, at level alpha/2;
* regression  <=>  overall p < alpha/2  OR  any task's Holm-adjusted p < alpha/2
  (and, if ``min_effect`` > 0, the overall bootstrap upper bound is below -min_effect or the
  flagged task's drop exceeds ``min_effect``).

Improvements are reported with two-sided p-values for information only.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentdiff.schema import Trajectory, group_by_task
from agentdiff.stats.tests import (
    fisher_greater,
    fisher_less,
    holm,
    paired_bootstrap_diff,
    stratified_exact_test,
)


class TaskComparison(BaseModel):
    task_id: str
    base_pass: int
    base_n: int
    cand_pass: int
    cand_n: int
    delta: float
    p_worse: float
    p_worse_holm: float
    p_better: float
    verdict: str  # "regressed" | "improved" | "no significant change"


class GateResult(BaseModel):
    baseline: str
    candidate: str
    n_tasks: int
    alpha: float
    overall_delta: float
    overall_ci: tuple[float, float]
    overall_p_worse: float
    overall_p_better: float
    overall_method: str
    tasks: list[TaskComparison]
    regressed_tasks: list[str]
    improved_tasks: list[str]
    regression: bool
    reason: str

    @property
    def exit_code(self) -> int:
        return 1 if self.regression else 0


def _counts(runs: list[Trajectory]) -> tuple[int, int]:
    valid = [r for r in runs if r.outcome.infra_error is None]
    return sum(r.outcome.passed for r in valid), len(valid)


def compare(baseline: list[Trajectory], candidate: list[Trajectory], alpha: float = 0.05,
            min_effect: float = 0.0, seed: int = 0) -> GateResult:  # fmt: skip
    gb, gc = group_by_task(baseline), group_by_task(candidate)
    shared = sorted(set(gb) & set(gc))
    rows = []
    for tid in shared:
        xb, nb = _counts(gb[tid])
        xc, nc = _counts(gc[tid])
        if nb and nc:
            rows.append((tid, xb, nb, xc, nc))
    if not rows:
        raise ValueError("baseline and candidate share no tasks with valid runs")
    tids, xb, nb, xc, nc = (list(v) for v in zip(*rows, strict=True))
    overall = stratified_exact_test(xc, nc, xb, nb, seed=seed)
    d, lo, hi = paired_bootstrap_diff(xc, nc, xb, nb, alpha=alpha, seed=seed)
    p_w = [fisher_less(c, n_c, b, n_b) for c, n_c, b, n_b in zip(xc, nc, xb, nb, strict=True)]
    p_b = [fisher_greater(c, n_c, b, n_b) for c, n_c, b, n_b in zip(xc, nc, xb, nb, strict=True)]
    p_w_holm = holm(p_w)
    p_b_holm = holm(p_b)
    half = alpha / 2
    tasks = []
    for i, tid in enumerate(tids):
        delta = xc[i] / nc[i] - xb[i] / nb[i]
        verdict = "no significant change"
        if p_w_holm[i] < half and -delta >= min_effect:
            verdict = "regressed"
        elif p_b_holm[i] < half:
            verdict = "improved"
        tasks.append(TaskComparison(task_id=tid, base_pass=xb[i], base_n=nb[i], cand_pass=xc[i], cand_n=nc[i],
                                    delta=delta, p_worse=p_w[i], p_worse_holm=p_w_holm[i], p_better=p_b[i],
                                    verdict=verdict))  # fmt: skip
    regressed = [t.task_id for t in tasks if t.verdict == "regressed"]
    improved = [t.task_id for t in tasks if t.verdict == "improved"]
    overall_reg = overall.p_less < half and (min_effect <= 0 or hi < -min_effect)
    reasons = []
    if overall_reg:
        reasons.append(f"overall pass rate dropped {d:+.3f} (stratified exact p={overall.p_less:.2g} < {half:g})")
    if regressed:
        reasons.append(f"{len(regressed)} task(s) regressed after Holm correction: {', '.join(regressed)}")
    return GateResult(
        baseline=baseline[0].agent_version, candidate=candidate[0].agent_version, n_tasks=len(tids), alpha=alpha,
        overall_delta=d, overall_ci=(lo, hi), overall_p_worse=overall.p_less, overall_p_better=overall.p_greater,
        overall_method=overall.method, tasks=tasks, regressed_tasks=regressed, improved_tasks=improved,
        regression=bool(overall_reg or regressed),
        reason="; ".join(reasons) or "no statistically significant regression",
    )  # fmt: skip
