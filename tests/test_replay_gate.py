from __future__ import annotations

import random

import pytest

from agentdiff.cluster import cluster_divergences
from agentdiff.demo.toy import ToyAgent
from agentdiff.divergence import find_divergence
from agentdiff.gate import compare
from agentdiff.reliability import analyze_reliability
from agentdiff.replay import confirm
from agentdiff.runner import fork_run, run_suite
from agentdiff.schema import Outcome, group_by_task
from agentdiff.tape import Budget, BudgetExceeded, LLMReply, Pricing, ReplayDivergence, Tape, UnsafeReplay


def test_keep_everything_replay_reproduces_the_recording() -> None:
    agent = ToyAgent(p_truncate=0.5)
    runs = run_suite(agent, ["t3_sorting", "t2_nullcheck"], k=6, concurrency=2)
    for r in runs:
        last = len(r.steps) - 1
        fork = fork_run(agent, r, last, "keep", seed=999)
        assert fork.outcome.passed == r.outcome.passed
        assert [s.decision for s in fork.steps] == [s.decision for s in r.steps]
        assert [s.observation for s in fork.steps] == [s.observation for s in r.steps]


class _SecretlyNondeterministic:
    version = "nd"

    def run(self, task_id: str, tape: Tape, seed: int) -> Outcome:
        tape.tool("x", "a", {"v": random.random()}, lambda: 1, replay_safe=True)  # unseeded input
        return Outcome(passed=True)


def test_replay_detects_nondeterminism_outside_the_tape() -> None:
    h = _SecretlyNondeterministic()
    [run] = run_suite(h, ["t"], k=1)
    with pytest.raises(ReplayDivergence):
        fork_run(h, run, 1, "keep", seed=1)


class _Unsafe:
    version = "u"

    def run(self, task_id: str, tape: Tape, seed: int) -> Outcome:
        tape.llm("n", {"q": 1}, lambda: LLMReply("hi"))
        tape.tool("n", "send_email", {"to": "x"}, lambda: "sent", replay_safe=False)
        return Outcome(passed=True)


def test_unsafe_tools_are_never_executed_live_in_a_fork() -> None:
    h = _Unsafe()
    [run] = run_suite(h, ["t"], k=1)
    with pytest.raises(UnsafeReplay):
        fork_run(h, run, 0, "resample", seed=1)
    fork = fork_run(h, run, 1, "keep", seed=1)  # recorded result is served, not re-sent
    assert fork.steps[1].meta.get("replayed")


def test_budget_is_a_hard_stop() -> None:
    class Paid:
        version = "p"

        def run(self, task_id: str, tape: Tape, seed: int) -> Outcome:
            tape.llm("n", {"s": seed}, lambda: LLMReply("x", model="m", tokens_in=1_000_000))
            return Outcome(passed=True)

    budget = Budget(max_usd=2.5)
    runs = run_suite(Paid(), ["a", "b"], k=5, concurrency=1, budget=budget, pricing=Pricing({"m": (1.0, 0.0)}))
    assert len(runs) == 3 and budget.spent_usd == pytest.approx(3.0)
    with pytest.raises(BudgetExceeded):
        budget.check()


def test_replay_confirms_the_planted_decision_cause() -> None:
    agent = ToyAgent()
    runs = run_suite(agent, ["t4_encoding"], k=12)
    passing = [r for r in runs if r.outcome.passed]
    failing = [r for r in runs if not r.outcome.passed]
    div = find_divergence(passing, failing)
    rep = confirm(agent, div, failing, passing, n_forks=10)
    assert rep.confirmed_label == "planner:llm:llm#1"
    t = rep.tests[0]
    assert t.keep.passed == 0 and t.lo is not None and t.lo > 0


def test_replay_does_not_confirm_a_downstream_consequence_as_first_cause() -> None:
    """Resampling the patch after reading the wrong file cannot fix the run."""
    agent = ToyAgent()
    runs = run_suite(agent, ["t4_encoding"], k=12)
    failing = [r for r in runs if not r.outcome.passed]
    passing = [r for r in runs if r.outcome.passed]
    div = find_divergence(passing, failing)
    rep = confirm(agent, div, failing, passing, n_forks=6, stop_at_first=False)
    by_label = {t.label: t for t in rep.tests}
    patch = by_label.get("coder:llm:llm#1")
    assert patch is None or patch.status == "not-confirmed"


def test_gate_flags_real_regression_and_passes_identical_versions() -> None:
    good = ToyAgent(version="v1")
    bad = ToyAgent(version="v2", p_wrong_file={t: 0.7 for t in good.p_wrong_file})
    base = run_suite(good, good.task_ids(), k=8)
    cand = run_suite(bad, good.task_ids(), k=8, seed0=500)
    res = compare(base, cand)
    assert res.regression and res.exit_code == 1 and res.overall_delta < 0
    same = run_suite(ToyAgent(version="v1b"), good.task_ids(), k=8, seed0=900)
    assert not compare(base, same).regression


def test_reliability_and_clustering_on_toy_suite() -> None:
    agent = ToyAgent(p_truncate=0.6)
    runs = run_suite(agent, agent.task_ids(), k=8)
    rel = analyze_reliability(runs)
    assert rel.n_tasks == 6 and "pass^8" in rel.metrics
    assert all(rel.metrics[k].lo <= rel.metrics[k].value <= rel.metrics[k].hi for k in rel.metrics)
    divs = []
    for runs_t in group_by_task(runs).values():
        p = [r for r in runs_t if r.outcome.passed]
        f = [r for r in runs_t if not r.outcome.passed]
        if len(p) >= 2 and f:
            divs.append(find_divergence(p, f))
    modes = cluster_divergences(divs)
    assert sum(len(m.members) for m in modes) == len(divs)
    labels = {m.step_labels[0] for m in modes}
    assert "planner:llm:llm" in labels


def test_transplant_confirms_a_deterministic_regression() -> None:
    """A version that always opens the wrong file: resampling cannot help, transplanting the good step can."""
    good = ToyAgent(version="good", p_wrong_file={"t4_encoding": 0.0})
    bad = ToyAgent(version="bad", p_wrong_file={"t4_encoding": 1.0})
    base = run_suite(good, ["t4_encoding"], k=6)
    cand = run_suite(bad, ["t4_encoding"], k=6, seed0=50)
    div = find_divergence(base, cand, mode="regression")
    assert div.predicted_labels[0] == "planner:llm:llm#1"
    res = confirm(bad, div, cand, base, n_forks=6, intervention="resample")
    assert res.confirmed_label is None  # re-drawing under the faulty policy never fixes it
    tr = confirm(bad, div, cand, base, n_forks=6, intervention="transplant")
    assert tr.confirmed_label == "planner:llm:llm#1" and tr.tests[0].resample.passed == 6
