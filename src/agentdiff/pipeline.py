"""High-level analyses that tie the pieces together (used by the CLI and experiments)."""

from __future__ import annotations

from agentdiff.cluster import cluster_divergences
from agentdiff.divergence import DivergenceResult, find_divergence
from agentdiff.gate import GateResult, compare
from agentdiff.reliability import analyze_reliability
from agentdiff.replay import ReplayReport, confirm
from agentdiff.report.html import AnalysisBundle
from agentdiff.runner import Harness
from agentdiff.schema import Trajectory, group_by_task
from agentdiff.tape import Budget, Pricing


def _split(runs: list[Trajectory]) -> tuple[list[Trajectory], list[Trajectory]]:
    valid = [r for r in runs if r.outcome.infra_error is None]
    return [r for r in valid if r.outcome.passed], [r for r in valid if not r.outcome.passed]


def flaky_divergences(trajs: list[Trajectory], min_pass: int = 2) -> list[DivergenceResult]:
    out = []
    for runs in group_by_task(trajs).values():
        passing, failing = _split(runs)
        if len(passing) >= min_pass and failing:
            out.append(find_divergence(passing, failing, mode="flaky"))
    return out


def regression_divergences(baseline: list[Trajectory], candidate: list[Trajectory], tasks: list[str] | None = None,
                           min_pass: int = 2) -> list[DivergenceResult]:  # fmt: skip
    gb, gc = group_by_task(baseline), group_by_task(candidate)
    out = []
    for tid in tasks if tasks is not None else sorted(set(gb) & set(gc)):
        passing, _ = _split(gb.get(tid, []))
        _, failing = _split(gc.get(tid, []))
        if len(passing) >= min_pass and failing:
            out.append(find_divergence(passing, failing, mode="regression"))
    return out


def replay_all(harness: Harness, trajs: list[Trajectory], divs: list[DivergenceResult], *, n_forks: int = 8,
               max_tasks: int = 3, budget: Budget | None = None, pricing: Pricing | None = None,
               reference_runs: list[Trajectory] | None = None) -> list[ReplayReport]:  # fmt: skip
    groups = group_by_task(trajs)
    ref_groups = group_by_task(reference_runs) if reference_runs is not None else groups
    reports = []
    for d in divs[:max_tasks]:
        passing, _ = _split(ref_groups.get(d.task_id, []))
        _, failing = _split(groups.get(d.task_id, []))
        if passing and failing:
            reports.append(confirm(harness, d, failing, passing, n_forks=n_forks, budget=budget, pricing=pricing))
    return reports


def analyze(trajs: list[Trajectory], title: str | None = None) -> AnalysisBundle:
    rel = analyze_reliability(trajs)
    divs = flaky_divergences(trajs)
    rank = {t: i for i, t in enumerate(rel.flaky_ranking)}
    divs.sort(key=lambda d: rank.get(d.task_id, 10**6))
    return AnalysisBundle(
        title=title or f"agentdiff report · {rel.agent_version}", reliability=rel, divergences=divs,
        failure_modes=cluster_divergences(divs), cost_usd=sum(t.cost_usd for t in trajs),
    )  # fmt: skip


def compare_bundle(baseline: list[Trajectory], candidate: list[Trajectory], alpha: float = 0.05,
                   min_effect: float = 0.0) -> tuple[GateResult, AnalysisBundle]:  # fmt: skip
    gate = compare(baseline, candidate, alpha=alpha, min_effect=min_effect)
    worse = [t.task_id for t in sorted(gate.tasks, key=lambda t: t.delta) if t.delta < 0]
    divs = regression_divergences(baseline, candidate, worse)
    bundle = AnalysisBundle(
        title=f"agentdiff compare · {gate.baseline} → {gate.candidate}", gate=gate,
        reliability=analyze_reliability(candidate), divergences=divs, failure_modes=cluster_divergences(divs),
        cost_usd=sum(t.cost_usd for t in baseline + candidate),
        notes=["Divergence in this report is computed in regression mode: baseline passing runs vs candidate failing runs."],
    )  # fmt: skip
    return gate, bundle
