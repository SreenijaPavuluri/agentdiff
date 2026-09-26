"""End-to-end check against the real code-verify-agents graph with an offline scripted model.

Skipped unless the `codeverify` package is installed and its benchmark dataset is found
(set CODEVERIFY_DATASET, or keep the repo checked out next to this one).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("codeverify")

from codeverify.eval.dataset import load_dataset  # noqa: E402

from agentdiff.divergence import find_divergence  # noqa: E402
from agentdiff.integrations.codeverify import (  # noqa: E402
    CodeVerifyHarness,
    canonical_code,
    llm_view,
    scripted_model,
)
from agentdiff.runner import fork_run, run_suite  # noqa: E402

DS = Path(os.environ.get("CODEVERIFY_DATASET", Path(__file__).parents[2] / "code-verify-agents/benchmarks/dataset"))
pytestmark = pytest.mark.skipif(not DS.exists(), reason="code-verify-agents dataset not found")


def _harness(p: float) -> CodeVerifyHarness:
    items = {it.task.task_id: it for it in load_dataset(DS)}
    return CodeVerifyHarness(version="scripted", models={"scripted": scripted_model(items, p)}, default_model="scripted",
                             dataset_dir=DS, task_filter=["t01_sum_to"],
                             settings_overrides={"planner_mode": "llm", "inspector_llm": True, "max_patch_attempts": 1})


def test_views_normalise_llm_replies() -> None:
    assert canonical_code("x = 1   # c\n") == canonical_code("x=1\n")
    assert llm_view("planner", '```json\n{"steps": [{"tool": "run_tests"}]}\n```') == {"tools": ["run_tests"]}
    assert llm_view("patch_generator", "no code here") == {"patched_code": "<no code>"}


def test_codeverify_runs_record_replay_and_diverge_at_the_patch() -> None:
    h = _harness(0.5)
    runs = run_suite(h, h.task_ids(), k=10, concurrency=4)
    labels = runs[0].labels()
    assert labels[:4] == ["planner:llm:llm#1", "inspector:tool:run_tests#1", "inspector:llm:llm#1",
                          "patch_generator:llm:llm#1"]
    r = runs[0]
    fork = fork_run(h, r, len(r.steps) - 1, "keep", seed=1)
    assert fork.outcome.passed == r.outcome.passed
    passing = [x for x in runs if x.outcome.passed]
    failing = [x for x in runs if not x.outcome.passed]
    assert len(passing) >= 2 and failing
    d = find_divergence(passing, failing)
    assert d.predicted_labels[0] == "patch_generator:llm:llm#1"
