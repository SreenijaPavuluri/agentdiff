"""Full-payload harness for code-verify-agents (github.com/SreenijaPavuluri/code-verify-agents).

The agent is used unmodified. agentdiff plugs into the two seams the repo already exposes:

* ``build_graph(llm=...)``: :class:`_TapedLLM` implements its ``LLMClient`` protocol
  (``complete(system, user)``) and routes every call through the :class:`~agentdiff.tape.Tape`;
* ``build_graph(tools=...)``: :class:`_TapedTools` subclasses its ``ToolRegistry`` and routes
  every ``invoke`` through the tape (all three tools are pure or sandboxed: replay-safe).

The graph node issuing an LLM call is recovered from the system prompt header
("You are the PLANNER / CODE INSPECTOR / PATCH GENERATOR"). Each LLM step gets a normalised
``view`` for comparison: the planned tool list (planner), the cited suspect lines (inspector),
or the AST-canonicalised patched module (patch generator), so formatting noise and free-text
rationales do not count as divergence.

Outcome rule (same as the repo's own benchmark): the final code must pass the visible *and*
hidden tests in a fresh sandbox.

:class:`Fault` is a hook bundle used by the fault-injection experiments; the default does
nothing.
"""

from __future__ import annotations

import ast
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeverify.config import Settings
from codeverify.eval.dataset import BenchmarkItem, load_dataset
from codeverify.graph import build_graph, graph_config
from codeverify.llm.base import LLMResponse
from codeverify.llm.structured import extract_code_block, extract_json
from codeverify.schemas import AstReport, LintReport, TestReport, ToolCallRecord
from codeverify.tools import SubprocessSandbox, ToolRegistry

from agentdiff.schema import Outcome
from agentdiff.tape import LLMReply, Tape

NODE_HEADERS = {
    "You are the PLANNER": "planner",
    "You are the CODE INSPECTOR": "inspector",
    "You are the PATCH GENERATOR": "patch_generator",
}
RESULT_TYPES: dict[str, type] = {"TestReport": TestReport, "AstReport": AstReport, "LintReport": LintReport}

Model = Callable[..., LLMReply]  # (system, user, *, temperature, seed) -> LLMReply


def node_of(system: str) -> str:
    for head, node in NODE_HEADERS.items():
        if system.startswith(head):
            return node
    return "unknown"


def canonical_code(code: str) -> str:
    try:
        return ast.unparse(ast.parse(code))
    except (SyntaxError, ValueError):
        return "<unparseable>\n" + code.strip()


def llm_view(node: str, text: str) -> Any:
    """Normalised projection of a raw LLM reply, used by the divergence finder."""
    try:
        data = extract_json(text)
    except ValueError:
        data = None
    if node == "planner":
        steps = data.get("steps") if isinstance(data, dict) else None
        tools = [s.get("tool") for s in steps if isinstance(s, dict)] if isinstance(steps, list) else None
        return {"tools": tools if tools else "<invalid plan>"}
    if node == "inspector":
        locs = data.get("suspect_locations") if isinstance(data, dict) else None
        lines = sorted({int(x["line"]) for x in locs if isinstance(x, dict) and str(x.get("line", "")).isdigit()}) if isinstance(locs, list) else None
        return {"suspect_lines": lines if lines is not None else "<invalid report>"}
    if node == "patch_generator":
        code = extract_code_block(text)
        if code is None and isinstance(data, dict):
            code = data.get("patched_code")
        return {"patched_code": canonical_code(code) if isinstance(code, str) else "<no code>"}
    return text


@dataclass
class Fault:
    """Fault-injection hooks. Every hook is a no-op by default."""

    id: str = "none"
    description: str = "no fault"
    true_cause: str | None = None
    intermittent: bool = False

    def patch_request(self, node: str, occurrence: int, system: str, user: str) -> tuple[str, str]:
        return system, user

    def model_for(self, node: str) -> str | None:
        return None

    def temperature_for(self, node: str) -> float | None:
        return None

    def patch_tool(self, node: str, name: str, occurrence: int, result: Any, rng: random.Random) -> Any:
        return result

    def patch_response(self, node: str, occurrence: int, text: str, rng: random.Random) -> str:
        return text


class _TapedLLM:
    def __init__(self, tape: Tape, models: dict[str, Model], default: str, fault: Fault, seed: int,
                 temperature: float):  # fmt: skip
        self.tape, self.models, self.default, self.fault = tape, models, default, fault
        self.seed, self.temperature = seed, temperature
        self.calls: Counter[str] = Counter()
        self.fired: list[int] = []
        self.rng = random.Random(seed)

    def complete(self, system: str, user: str) -> LLMResponse:
        node = node_of(system)
        self.calls[node] += 1
        occ = self.calls[node]
        new_system, new_user = self.fault.patch_request(node, occ, system, user)
        fired = (new_system, new_user) != (system, user)
        system, user = new_system, new_user
        model_name = self.fault.model_for(node) or self.default
        temp = self.fault.temperature_for(node)
        if fired or model_name != self.default or temp is not None:
            self.fired.append(len(self.tape.steps))
        temp = self.temperature if temp is None else temp
        call_seed = (self.seed * 1_000_003 + occ * 7919 + hash(node) % 997) % (2**31)
        request = {"system": system, "user": user, "model": model_name, "temperature": temp}

        index = len(self.tape.steps)

        def call() -> LLMReply:
            reply = self.models[model_name](system, user, temperature=temp, seed=call_seed)
            text = self.fault.patch_response(node, occ, reply.text, self.rng)
            if text != reply.text:
                self.fired.append(index)
                reply.text = text
            return reply

        reply = self.tape.llm(node, request, call, view=lambda text: llm_view(node, text))
        return LLMResponse(text=reply.text, model=reply.model, input_tokens=reply.tokens_in,
                           output_tokens=reply.tokens_out, latency_ms=reply.latency_ms)  # fmt: skip


class _TapedTools(ToolRegistry):
    def __init__(self, sandbox: SubprocessSandbox, tape: Tape, fault: Fault, rng: random.Random):
        super().__init__(sandbox)
        self.tape, self.fault, self.rng = tape, fault, rng
        self.calls: Counter[tuple[str, str]] = Counter()
        self.fired: list[int] = []

    def invoke(self, name: str, node: str, **kwargs: Any) -> tuple[Any, ToolCallRecord]:
        self.calls[(node, name)] += 1
        occ = self.calls[(node, name)]

        index = len(self.tape.steps)

        def call() -> tuple[Any, ToolCallRecord]:
            result, rec = ToolRegistry.invoke(self, name, node, **kwargs)
            patched = self.fault.patch_tool(node, name, occ, result, self.rng)
            if patched is not result:
                self.fired.append(index)
            return patched, rec

        def encode(value: tuple[Any, ToolCallRecord]) -> dict[str, Any]:
            result, rec = value
            return {"type": type(result).__name__ if result is not None else None,
                    "result": result.model_dump(mode="json") if result is not None else None,
                    "success": rec.success, "error": rec.error}  # fmt: skip

        def decode(obs: dict[str, Any]) -> tuple[Any, ToolCallRecord]:
            cls = RESULT_TYPES.get(obs.get("type") or "")
            result = cls.model_validate(obs["result"]) if cls and obs.get("result") is not None else None
            return result, ToolCallRecord(tool=name, node=node, success=bool(obs.get("success")), latency_ms=0.0,
                                          error=obs.get("error"))  # fmt: skip

        return self.tape.tool(node, name, dict(kwargs), call, replay_safe=True, encode=encode, decode=decode)


@dataclass
class CodeVerifyHarness:
    """``agentdiff`` harness around the unmodified code-verify-agents LangGraph agent."""

    version: str
    models: dict[str, Model]
    default_model: str
    dataset_dir: str | Path
    fault: Fault = field(default_factory=Fault)
    temperature: float = 0.7
    settings_overrides: dict[str, Any] = field(default_factory=lambda: {"planner_mode": "llm", "inspector_llm": True})
    task_filter: list[str] | None = None

    def __post_init__(self) -> None:
        self._items: dict[str, BenchmarkItem] = {
            it.task.task_id: it for it in load_dataset(Path(self.dataset_dir), self.task_filter)
        }
        self.settings = Settings(provider="scripted", model=self.default_model, **self.settings_overrides)

    def task_ids(self) -> list[str]:
        return list(self._items)

    def run(self, task_id: str, tape: Tape, seed: int) -> Outcome:
        item = self._items[task_id]
        env_rng = random.Random(f"env-{self.fault.id}-{seed}")
        llm = _TapedLLM(tape, self.models, self.default_model, self.fault, seed, self.temperature)
        llm.rng = random.Random(f"llm-env-{self.fault.id}-{seed}")
        tools = _TapedTools(SubprocessSandbox(timeout_s=self.settings.sandbox_timeout_s), tape, self.fault, env_rng)
        graph = build_graph(self.settings, llm=llm, tools=tools)
        try:
            state = graph.invoke({"task": item.task, "step_count": 0}, graph_config(self.settings, f"{task_id}-{seed}"))
        finally:
            for idx in sorted(set(llm.fired + tools.fired)):  # ground truth for fault-injection experiments
                if idx < len(tape.steps) and not tape.steps[idx].meta.get("replayed"):
                    tape.steps[idx].meta["fault_fired"] = self.fault.id
        code = state.get("final_code") if state.get("status") == "resolved" else None
        visible = hidden = False
        if code:
            sb = SubprocessSandbox(timeout_s=15, per_test_timeout_s=3)
            visible = sb.run_tests(code, item.task.test_code).all_passed
            hidden = visible and sb.run_tests(code, item.hidden_tests).all_passed
        return Outcome(passed=bool(visible and hidden), detail={
            "status": state.get("status"), "attempts": state.get("patch_attempts", 0),
            "visible": visible, "hidden": hidden, "termination": state.get("termination_reason", ""),
        })  # fmt: skip


def scripted_model(items: dict[str, BenchmarkItem], p_correct: float = 0.6) -> Model:
    """Offline stochastic stand-in for tests: returns the reference fix with probability p_correct."""
    fixes = {it.task.source_code.splitlines()[0]: it.reference_fix for it in items.values()}

    def model(system: str, user: str, *, temperature: float = 0.7, seed: int | None = None) -> LLMReply:
        rng = random.Random(seed)
        node = node_of(system)
        if node == "planner":
            text = '```json\n{"bug_hypothesis": "see tests", "steps": [{"tool": "run_tests", "purpose": "reproduce it"}], "risk_level": "low"}\n```'
        elif node == "inspector":
            text = '```json\n{"summary": "failing tests", "suspect_locations": [{"line": 1, "reason": "guess here"}]}\n```'
        else:
            fix = next((f for head, f in fixes.items() if head in user), "pass\n")
            code = fix if rng.random() < p_correct else "def broken():\n    return None\n"
            text = f'```json\n{{"root_cause": "scripted cause", "change_summary": "scripted change", "confidence": 0.5}}\n```\n```python\n{code}```'
        return LLMReply(text=text, model="scripted", tokens_in=len((system + user).split()), tokens_out=len(text.split()))

    return model
