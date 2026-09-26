"""Import LangGraph runs recorded with ``graph.stream(..., stream_mode="updates")``.

Each update is ``{node_name: state_update}``. Messages inside an update become steps:
an AI message is an ``llm`` step (decision = text + tool calls), a tool message is a ``tool``
step (decision = the matching tool call's arguments, observation = the tool output). Other
state updates become ``agent`` steps whose decision is the JSON-able update.

The adapter is duck-typed (``.type``, ``.content``, ``.tool_calls``, ``.tool_call_id``) and
also accepts messages already serialised to dicts, so LangGraph is not a dependency.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agentdiff.schema import Outcome, Step, Trajectory
from agentdiff.tape import to_jsonable


def _get(m: Any, key: str, default: Any = None) -> Any:
    return m.get(key, default) if isinstance(m, dict) else getattr(m, key, default)


def _is_message(m: Any) -> bool:
    return _get(m, "type") in {"ai", "tool", "human", "system", "AIMessage", "ToolMessage"}


def updates_to_steps(updates: Iterable[dict[str, Any]], include_state: bool = True) -> list[Step]:
    steps: list[Step] = []
    call_args: dict[str, Any] = {}
    for upd in updates:
        for node, delta in upd.items():
            if not isinstance(delta, dict):
                continue
            msgs = delta.get("messages")
            msgs = msgs if isinstance(msgs, list) else [msgs] if msgs is not None else []
            for m in (x for x in msgs if _is_message(x)):
                mtype = _get(m, "type")
                if mtype in ("ai", "AIMessage"):
                    tcs = [{"name": _get(tc, "name"), "args": _get(tc, "args")} for tc in (_get(m, "tool_calls") or [])]
                    for tc in _get(m, "tool_calls") or []:
                        call_args[str(_get(tc, "id"))] = {"name": _get(tc, "name"), "args": _get(tc, "args")}
                    meta = _get(m, "response_metadata") or {}
                    usage = _get(m, "usage_metadata") or {}
                    steps.append(Step(index=len(steps), kind="llm", node=node, name="llm",
                                      decision={"content": to_jsonable(_get(m, "content")), "tool_calls": tcs},
                                      model=meta.get("model_name") if isinstance(meta, dict) else None,
                                      tokens_in=int(_get(usage, "input_tokens", 0) or 0),
                                      tokens_out=int(_get(usage, "output_tokens", 0) or 0)))  # fmt: skip
                elif mtype in ("tool", "ToolMessage"):
                    call = call_args.get(str(_get(m, "tool_call_id")), {})
                    status = _get(m, "status")
                    steps.append(Step(index=len(steps), kind="tool", node=node,
                                      name=str(_get(m, "name") or call.get("name") or "tool"),
                                      decision=to_jsonable(call.get("args")),
                                      observation=to_jsonable(_get(m, "content")),
                                      error="tool error" if status == "error" else None))  # fmt: skip
            rest = {k: v for k, v in delta.items() if k != "messages"}
            if include_state and rest:
                steps.append(Step(index=len(steps), kind="agent", node=node, name=node, decision=to_jsonable(rest)))
    return steps


def from_langgraph_updates(updates: Iterable[dict[str, Any]], *, task_id: str, run_id: str, agent_version: str,
                           passed: bool, include_state: bool = True) -> Trajectory:  # fmt: skip
    return Trajectory(task_id=task_id, run_id=run_id, agent_version=agent_version,
                      steps=updates_to_steps(updates, include_state), outcome=Outcome(passed=passed, source="imported"))  # fmt: skip


def from_langgraph_jsonl(path: str | Path, agent_version: str = "unknown") -> list[Trajectory]:
    """Each line: ``{"task_id", "run_id", "passed", "updates": [...]}`` (updates JSON-serialised)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out.append(from_langgraph_updates(rec["updates"], task_id=rec["task_id"], run_id=rec["run_id"],
                                              agent_version=rec.get("agent_version", agent_version),
                                              passed=bool(rec["passed"])))  # fmt: skip
    return out
