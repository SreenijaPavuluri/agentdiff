from __future__ import annotations

import json
from pathlib import Path

from agentdiff.adapters.codeverify import from_codeverify_events
from agentdiff.adapters.langgraph import from_langgraph_updates
from agentdiff.adapters.otel import from_otel_jsonl

FIX = Path(__file__).parent / "fixtures"


def test_codeverify_native_events() -> None:
    trajs = from_codeverify_events(FIX / "codeverify_events.jsonl", FIX / "codeverify_results.jsonl")
    assert len(trajs) == 2
    t = trajs[0]
    assert [s.kind for s in t.steps] == ["tool", "tool", "tool", "llm", "tool", "tool", "tool"]
    assert t.labels()[0] == "inspector:tool:run_tests#1"
    assert t.labels()[3] == "patch_generator:llm:llm#1"
    assert t.outcome.passed and t.outcome.source == "rule"


def _span(trace, sid, parent, t, attrs, status=None):
    return {"trace_id": trace, "span_id": sid, "parent_span_id": parent, "start_time_unix_nano": t,
            "end_time_unix_nano": t + 2_000_000, "attributes": attrs, "status": status or {"code": "OK"}}


def test_otel_genai_spans(tmp_path: Path) -> None:
    spans = [
        _span("T1", "a", None, 1, {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "fixer",
                                   "agentdiff.task_id": "task-1", "agentdiff.passed": False}),
        _span("T1", "b", "a", 2, {"gen_ai.operation.name": "chat", "gen_ai.request.model": "m",
                                  "gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 3,
                                  "gen_ai.output.messages": json.dumps([{"role": "assistant", "parts": []}])}),
        _span("T1", "c", "a", 3, {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "read",
                                  "gen_ai.tool.call.arguments": "{\"path\": \"a.py\"}", "error.type": "Timeout"}),
    ]
    # the same trace as an OTLP batch with list-style attributes
    otlp = {"resourceSpans": [{"scopeSpans": [{"spans": [
        {"traceId": "T2", "spanId": "x", "startTimeUnixNano": "5",
         "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                        {"key": "gen_ai.tool.name", "value": {"stringValue": "read"}},
                        {"key": "agentdiff.task_id", "value": {"stringValue": "task-1"}},
                        {"key": "agentdiff.passed", "value": {"boolValue": True}}]}]}]}]}
    p = tmp_path / "spans.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in spans) + "\n" + json.dumps(otlp) + "\n")
    trajs = {t.run_id: t for t in from_otel_jsonl(p, agent_version="v1")}
    t1 = trajs["T1"]
    assert not t1.outcome.passed and t1.task_id == "task-1"
    assert [s.kind for s in t1.steps] == ["llm", "tool"]
    assert t1.steps[0].node == "fixer" and t1.steps[0].tokens_in == 10
    assert t1.steps[1].decision == {"path": "a.py"} and t1.steps[1].error == "Timeout"
    assert trajs["T2"].outcome.passed and trajs["T2"].steps[0].name == "read"


def test_langgraph_updates_with_dict_messages() -> None:
    updates = [
        {"agent": {"messages": [{"type": "ai", "content": "", "tool_calls": [
            {"id": "1", "name": "search", "args": {"q": "x"}}]}]}},
        {"tools": {"messages": [{"type": "tool", "tool_call_id": "1", "name": "search", "content": "result"}]}},
        {"agent": {"messages": [{"type": "ai", "content": "done", "tool_calls": []}], "attempts": 1}},
    ]
    t = from_langgraph_updates(updates, task_id="q", run_id="r", agent_version="v", passed=True)
    assert [s.kind for s in t.steps] == ["llm", "tool", "llm", "agent"]
    assert t.steps[1].decision == {"q": "x"} and t.steps[1].observation == "result"
    assert t.steps[3].decision == {"attempts": 1}
