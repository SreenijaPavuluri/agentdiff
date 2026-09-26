"""Import the native telemetry of code-verify-agents (github.com/SreenijaPavuluri/code-verify-agents).

That repo writes one JSON event per line (``<tag>_events.jsonl``), each tagged with ``run_id``:
``tool_call`` (tool, node, success, latency_ms, error), ``llm_call`` (node, model, tokens,
repairs, success), ``verification`` (attempt, passed, tests_passed, tests_total) and
``run_complete`` (task_id, status, ...). Per-task grading lives in ``<tag>.jsonl``
(``TaskResult.resolved`` = visible + hidden tests pass).

The native events contain no prompts, responses or tool payloads, so imported trajectories
support reliability statistics and *structural* divergence (which steps ran, in what order,
with what success/error), but not content-level diffs or replay. For those, run the agent
through :mod:`agentdiff.integrations.codeverify`, which records full payloads.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentdiff.schema import Outcome, Step, Trajectory


def from_codeverify_events(events_path: str | Path, results_path: str | Path | None = None,
                           agent_version: str = "codeverify") -> list[Trajectory]:  # fmt: skip
    resolved: dict[str, bool] = {}
    if results_path:
        for line in Path(results_path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                resolved[r["task_id"]] = bool(r.get("resolved"))
    runs: dict[str, list[dict]] = {}
    for line in Path(events_path).read_text().splitlines():
        if line.strip():
            ev = json.loads(line)
            if ev.get("run_id"):
                runs.setdefault(ev["run_id"], []).append(ev)
    trajs = []
    for run_id, evs in runs.items():
        done = next((e for e in evs if e.get("event") == "run_complete"), None)
        if done is None:
            continue
        steps: list[Step] = []
        for e in evs:
            kind = e.get("event")
            if kind == "tool_call":
                steps.append(Step(index=len(steps), kind="tool", node=e.get("node"), name=e.get("tool", "tool"),
                                  observation={"success": e.get("success")}, error=e.get("error"),
                                  latency_ms=e.get("latency_ms"), replay_safe=True))  # fmt: skip
            elif kind == "llm_call":
                steps.append(Step(index=len(steps), kind="llm", node=e.get("node"), name="llm", model=e.get("model"),
                                  tokens_in=int(e.get("input_tokens") or 0), tokens_out=int(e.get("output_tokens") or 0),
                                  latency_ms=e.get("latency_ms"),
                                  decision={"repairs": e.get("repairs"), "success": e.get("success")}))  # fmt: skip
        task_id = done.get("task_id", "unknown")
        passed = resolved.get(task_id, done.get("status") == "resolved")
        trajs.append(Trajectory(task_id=task_id, run_id=run_id, agent_version=agent_version, steps=steps,
                                outcome=Outcome(passed=passed, source="rule" if task_id in resolved else "imported",
                                                detail={"status": done.get("status")}),
                                wall_ms=float(done.get("wall_ms") or 0)))  # fmt: skip
    return trajs
