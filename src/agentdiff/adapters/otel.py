"""Import OpenTelemetry GenAI spans (JSONL) as trajectories.

Accepted input, one JSON object per line, either
* a flat span: ``{"trace_id", "span_id", "parent_span_id", "name", "start_time_unix_nano",
  "attributes": {...} | [{"key", "value"}], "status": {...}}`` (camelCase keys also accepted), or
* an OTLP/JSON export batch: ``{"resourceSpans": [{"scopeSpans": [{"spans": [...]}]}]}``.

Mapping (semantic conventions for GenAI, status "Development"; older names tolerated):

=======================================  ==============================================
``gen_ai.operation.name``                chat / text_completion / generate_content -> llm,
                                         execute_tool -> tool, invoke_agent -> sets node
``gen_ai.tool.name``                     step name for tools
``gen_ai.tool.call.arguments`` (opt-in)  decision of a tool step
``gen_ai.tool.call.result`` (opt-in)     observation of a tool step
``gen_ai.output.messages`` (opt-in)      decision of an LLM step
``gen_ai.request.model`` / response      model
``gen_ai.usage.input/output_tokens``     tokens
``error.type`` or span status ERROR      error
``gen_ai.agent.name`` (nearest ancestor) node
=======================================  ==============================================

Task id and outcome are not part of the conventions. They are read from the attributes
``agentdiff.task_id`` / ``agentdiff.passed`` on any span of the trace (the root, typically),
or supplied via ``outcomes={trace_id: bool}`` and ``task_of={trace_id: task}``.
Without the opt-in content attributes, alignment is structural (names and order only), and
replay is impossible because tool results were never captured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentdiff.schema import Outcome, Step, Trajectory

LLM_OPS = {"chat", "text_completion", "generate_content", "completion"}


def _attrs(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    out: dict[str, Any] = {}
    for kv in raw or []:
        v = kv.get("value", {})
        if isinstance(v, dict):
            v = next(iter(v.values()), None) if v else None
        out[kv["key"]] = v
    return out


def _g(span: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in span:
            return span[k]
    return None


def _maybe_json(v: Any) -> Any:
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def iter_spans(path: str | Path) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "resourceSpans" in obj:
            for rs in obj["resourceSpans"]:
                for ss in rs.get("scopeSpans", []):
                    spans.extend(ss.get("spans", []))
        else:
            spans.append(obj)
    return spans


def from_otel_jsonl(
    path: str | Path,
    agent_version: str = "unknown",
    outcomes: dict[str, bool] | None = None,
    task_of: dict[str, str] | None = None,
) -> list[Trajectory]:
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for s in iter_spans(path):
        by_trace.setdefault(str(_g(s, "trace_id", "traceId")), []).append(s)
    trajs = []
    for trace_id, spans in by_trace.items():
        spans.sort(key=lambda s: int(_g(s, "start_time_unix_nano", "startTimeUnixNano") or 0))
        by_id = {str(_g(s, "span_id", "spanId")): s for s in spans}
        task_id, passed = (task_of or {}).get(trace_id), (outcomes or {}).get(trace_id)
        version = agent_version
        steps: list[Step] = []
        for s in spans:
            a = _attrs(s.get("attributes"))
            task_id = task_id or a.get("agentdiff.task_id")
            if passed is None and "agentdiff.passed" in a:
                passed = bool(a["agentdiff.passed"])
            version = a.get("agentdiff.agent_version", version)
            op = a.get("gen_ai.operation.name")
            if op not in LLM_OPS and op != "execute_tool":
                continue
            node = _agent_of(s, by_id)
            status = s.get("status") or {}
            err = a.get("error.type") or (status.get("message") or "error" if str(status.get("code")) in ("2", "STATUS_CODE_ERROR", "ERROR") else None)
            tokens_in = int(a.get("gen_ai.usage.input_tokens") or a.get("gen_ai.usage.prompt_tokens") or 0)
            tokens_out = int(a.get("gen_ai.usage.output_tokens") or a.get("gen_ai.usage.completion_tokens") or 0)
            start = int(_g(s, "start_time_unix_nano", "startTimeUnixNano") or 0)
            end = int(_g(s, "end_time_unix_nano", "endTimeUnixNano") or start)
            common = dict(index=len(steps), node=node, error=err, tokens_in=tokens_in, tokens_out=tokens_out,
                          latency_ms=(end - start) / 1e6, meta={"span_id": _g(s, "span_id", "spanId")})  # fmt: skip
            if op == "execute_tool":
                steps.append(Step(kind="tool", name=str(a.get("gen_ai.tool.name") or s.get("name")),
                                  decision=_maybe_json(a.get("gen_ai.tool.call.arguments")),
                                  observation=_maybe_json(a.get("gen_ai.tool.call.result")), **common))  # fmt: skip
            else:
                steps.append(Step(kind="llm", name="llm", decision=_maybe_json(a.get("gen_ai.output.messages")),
                                  model=a.get("gen_ai.response.model") or a.get("gen_ai.request.model"), **common))  # fmt: skip
        if task_id is None or passed is None:
            continue  # cannot use a trace without a task and a verifiable outcome
        trajs.append(Trajectory(task_id=str(task_id), run_id=trace_id, agent_version=str(version), steps=steps,
                                outcome=Outcome(passed=passed, source="imported")))  # fmt: skip
    return trajs


def _agent_of(span: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str | None:
    """Nearest span (itself first, then ancestors) that carries ``gen_ai.agent.name``."""
    cur: dict[str, Any] | None = span
    for _ in range(100):
        if cur is None:
            return None
        name = _attrs(cur.get("attributes")).get("gen_ai.agent.name")
        if name:
            return str(name)
        parent = _g(cur, "parent_span_id", "parentSpanId")
        cur = by_id.get(str(parent)) if parent else None
    return None
