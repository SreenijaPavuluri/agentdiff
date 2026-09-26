from __future__ import annotations

from agentdiff.align.nw import align
from agentdiff.align.similarity import content_sim, normalize_text, step_sim, value_sim
from agentdiff.divergence import find_divergence, perm_test_greater
from agentdiff.schema import Outcome, Step, Trajectory


def st(i: int, name: str, decision: object = None, obs: object = None, kind: str = "tool", node: str = "a") -> Step:
    return Step(index=i, kind=kind, node=node, name=name, decision=decision, observation=obs)  # type: ignore[arg-type]


def traj(steps: list[Step], passed: bool, rid: str) -> Trajectory:
    return Trajectory(task_id="t", run_id=rid, agent_version="v", steps=steps, outcome=Outcome(passed=passed))


def test_normalize_masks_volatile_tokens() -> None:
    a = normalize_text("run 3f2a9c1b7d4e5f60 at /tmp/abc123/x.py took 12.5 ms")
    b = normalize_text("run 99aa00bb11cc22dd at /tmp/zzz/x.py took 3.1 ms")
    assert a == b


def test_value_sim_structured() -> None:
    assert value_sim({"a": 1, "b": "x y"}, {"a": 1, "b": "x y"}) == 1.0
    assert value_sim({"a": 1, "b": "x"}, {"a": 2, "b": "x"}) == 0.5
    assert value_sim({"a": 1}, {"a": 1, "extra": 3}) == 0.5
    assert value_sim({"duration_ms": 1.0, "k": 1}, {"duration_ms": 9.0, "k": 1}) == 1.0  # ignored key


def test_incompatible_steps_never_align() -> None:
    a = [st(0, "x"), st(1, "y")]
    b = [st(0, "y"), st(1, "x")]
    pairs = align(a, b)
    matched = [(i, j) for i, j in pairs if i is not None and j is not None]
    assert len(matched) == 1
    for i, j in matched:
        assert a[i].name == b[j].name


def test_alignment_handles_insertion_and_substitution() -> None:
    a = [st(0, "list"), st(1, "read", {"p": "a.py"}), st(2, "test")]
    b = [st(0, "list"), st(1, "retry"), st(2, "read", {"p": "b.py"}), st(3, "test")]
    pairs = align(a, b)
    assert (0, 0) in pairs and (1, 2) in pairs and (2, 3) in pairs and (None, 1) in pairs
    assert step_sim(a[1], b[2]) < 1.0 and step_sim(a[1], b[2]) >= 0.5


def test_alignment_is_deterministic() -> None:
    a = [st(i, "x", {"v": i}) for i in range(5)]
    b = [st(i, "x", {"v": i % 2}) for i in range(4)]
    assert align(a, b) == align(a, b)


def test_llm_step_content_is_the_decision() -> None:
    x = st(0, "llm", "open a.py", kind="llm")
    y = st(0, "llm", "open b.py", kind="llm")
    assert content_sim(x, y) == 0.5  # observation must not dilute LLM decisions


def test_permutation_test_exact_small_case() -> None:
    # fail = [1,1], pass = [0,0]: only 1 of C(4,2)=6 splits is as extreme
    assert perm_test_greater(__import__("numpy").array([1.0, 1.0]), __import__("numpy").array([0.0, 0.0])) == 1 / 6


def _planted(n_pass: int, n_fail: int, cause_obs: bool = False) -> tuple[list[Trajectory], list[Trajectory]]:
    """Runs share steps 0-1; failing runs split at step 2 (decision or observation), then propagate."""
    passing, failing = [], []
    for r in range(n_pass):
        steps = [st(0, "list", {}, ["a", "b"]), st(1, "plan", "go", kind="llm", node="p"),
                 st(2, "read", {"p": "a"}, "full body text here"), st(3, "patch", f"fix {r % 2}", kind="llm", node="c"),
                 st(4, "test", {}, {"passed": True})]  # fmt: skip
        passing.append(traj(steps, True, f"p{r}"))
    for r in range(n_fail):
        read = st(2, "read", {"p": "a"}, "trunc") if cause_obs else st(2, "read", {"p": "zzz"}, "other file body qq")
        steps = [st(0, "list", {}, ["a", "b"]), st(1, "plan", "go", kind="llm", node="p"), read,
                 st(3, "patch", "guess", kind="llm", node="c"), st(4, "test", {}, {"passed": False})]  # fmt: skip
        failing.append(traj(steps, False, f"f{r}"))
    return passing, failing


def test_divergence_finds_planted_earliest_split_not_last_step() -> None:
    passing, failing = _planted(5, 4)
    d = find_divergence(passing, failing)
    assert d.confidence == "significant"
    assert d.predicted_labels[0] == "a:tool:read#1"
    assert d.columns[d.chosen].deviation == "decision"  # type: ignore[index]
    assert "a:tool:test#1" in d.predicted_labels  # the propagated consequence is ranked later


def test_divergence_labels_observation_causes() -> None:
    passing, failing = _planted(5, 4, cause_obs=True)
    d = find_divergence(passing, failing)
    assert d.predicted_labels[0] == "a:tool:read#1"
    assert d.columns[d.chosen].deviation == "observation"  # type: ignore[index]
    assert d.evidence is not None and "trunc" in d.evidence.observation_diff


def test_divergence_low_confidence_with_too_few_runs() -> None:
    passing, failing = _planted(2, 1)
    d = find_divergence(passing, failing)
    assert d.confidence == "low"


def test_extra_retry_steps_are_reported_as_insertions() -> None:
    passing, _ = _planted(4, 0)
    failing = []
    for r in range(4):
        steps = [st(0, "list", {}, ["a", "b"]), st(1, "plan", "go", kind="llm", node="p"),
                 st(2, "read", {"p": "a"}, "full body text here"), st(3, "patch", "fix 0", kind="llm", node="c"),
                 st(4, "test", {}, {"passed": True}), st(5, "patch", "again", kind="llm", node="c")]  # fmt: skip
        failing.append(traj(steps, False, f"f{r}"))
    d = find_divergence(passing, failing)
    assert d.columns[d.chosen].kind == "ins"  # type: ignore[index]
    assert d.predicted_labels[0] == "c:llm:patch#2"
