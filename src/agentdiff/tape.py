"""Recording and replay of agent runs.

Every model call and tool call of a harnessed agent goes through a :class:`Tape`.

* In **record** mode the tape executes the call, stores the full request/response and appends a
  :class:`~agentdiff.schema.Step`.
* In **fork** mode the tape serves steps ``0..s-1`` from a recorded trajectory (checking that the
  agent issues the *same* requests, otherwise :class:`ReplayDivergence`), applies an intervention
  at step ``s`` and runs everything after ``s`` live.

Interventions at the fork step:

``resample``    re-issue the call live (fresh LLM sample / fresh tool execution)
``keep``        serve the recorded step (the control arm)
``transplant``  serve a step taken from another run (e.g. the passing reference)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from agentdiff.schema import Step, Trajectory, stable_hash

Intervention = Literal["resample", "keep", "transplant"]


class ReplayDivergence(RuntimeError):
    """The agent issued a different request than the recording while replaying a prefix."""


class UnsafeReplay(RuntimeError):
    """A tool that is not marked replay-safe would have to run live during a fork."""


class BudgetExceeded(RuntimeError):
    pass


class RecordedToolError(RuntimeError):
    """Re-raised during replay when the recorded tool call raised."""


@dataclass
class LLMReply:
    text: str
    model: str = "unknown"
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


@dataclass
class Pricing:
    """USD per 1M tokens, keyed by model id. Unknown models cost 0 (e.g. local models)."""

    table: dict[str, tuple[float, float]] = field(default_factory=dict)

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        p_in, p_out = self.table.get(model, (0.0, 0.0))
        return (tokens_in * p_in + tokens_out * p_out) / 1e6


class Budget:
    """Thread-safe hard cost cap shared by every run in a session."""

    def __init__(self, max_usd: float | None = None, max_llm_calls: int | None = None):
        self.max_usd = max_usd
        self.max_llm_calls = max_llm_calls
        self.spent_usd = 0.0
        self.llm_calls = 0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self.max_usd is not None and self.spent_usd >= self.max_usd:
                raise BudgetExceeded(f"cost budget ${self.max_usd:.2f} exhausted")
            if self.max_llm_calls is not None and self.llm_calls >= self.max_llm_calls:
                raise BudgetExceeded(f"LLM call budget {self.max_llm_calls} exhausted")

    def charge(self, usd: float) -> None:
        with self._lock:
            self.spent_usd += usd
            self.llm_calls += 1


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


class Tape:
    def __init__(
        self,
        *,
        source: Trajectory | None = None,
        fork_step: int | None = None,
        intervention: Intervention = "resample",
        transplant: Step | None = None,
        pricing: Pricing | None = None,
        budget: Budget | None = None,
        store_requests: bool = True,
    ):
        if (source is None) != (fork_step is None):
            raise ValueError("source and fork_step go together")
        if intervention == "transplant" and transplant is None:
            raise ValueError("transplant intervention needs a transplant step")
        self.source = source
        self.fork_step = fork_step
        self.intervention = intervention
        self.transplant = transplant
        self.pricing = pricing or Pricing()
        self.budget = budget
        self.store_requests = store_requests
        self.steps: list[Step] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _next_index(self) -> int:
        return len(self.steps)

    def _mode_for(self, idx: int) -> Literal["live", "replay", "transplant"]:
        if self.source is None or self.fork_step is None:
            return "live"
        if idx < self.fork_step:
            return "replay"
        if idx == self.fork_step:
            return {"resample": "live", "keep": "replay", "transplant": "transplant"}[self.intervention]  # type: ignore[return-value]
        return "live"

    def _recorded(self, idx: int, kind: str, node: str | None, name: str, input_ref: str) -> Step:
        assert self.source is not None
        if idx >= len(self.source.steps):
            raise ReplayDivergence(f"step {idx}: recording has only {len(self.source.steps)} steps")
        rec = self.source.steps[idx]
        if rec.signature != (node or "", kind, name):
            raise ReplayDivergence(f"step {idx}: expected {rec.signature}, agent issued {(node, kind, name)}")
        if rec.input_ref is not None and rec.input_ref != input_ref:
            raise ReplayDivergence(f"step {idx} ({name}): request differs from the recording")
        return rec

    def _append(self, step: Step) -> None:
        with self._lock:
            self.steps.append(step)

    # --------------------------------------------------------------------- LLM
    def llm(
        self,
        node: str | None,
        request: dict[str, Any],
        call: Callable[[], LLMReply],
        name: str = "llm",
        view: Callable[[str], Any] | None = None,
    ) -> LLMReply:
        idx = self._next_index()
        ref = stable_hash(request)
        mode = self._mode_for(idx)
        meta: dict[str, Any] = {"request": request} if self.store_requests else {}
        if mode == "replay":
            rec = self._recorded(idx, "llm", node, name, ref)
            reply = LLMReply(rec.decision, rec.model or "unknown", rec.tokens_in, rec.tokens_out, rec.latency_ms or 0.0)
            self._append(rec.model_copy(update={"index": idx, "cost_usd": 0.0, "meta": {**rec.meta, "replayed": True}}))
            return reply
        if mode == "transplant":
            t = self.transplant
            assert t is not None
            reply = LLMReply(t.decision, t.model or "unknown", t.tokens_in, t.tokens_out, t.latency_ms or 0.0)
            self._append(t.model_copy(update={"index": idx, "cost_usd": 0.0, "input_ref": ref,
                                              "meta": {**meta, "transplanted": True}}))  # fmt: skip
            return reply
        if self.budget:
            self.budget.check()
        started = time.perf_counter()
        error = None
        try:
            reply = call()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            self._append(Step(index=idx, kind="llm", node=node, name=name, input_ref=ref, error=error, meta=meta,
                              latency_ms=(time.perf_counter() - started) * 1000))  # fmt: skip
            raise
        cost = self.pricing.cost(reply.model, reply.tokens_in, reply.tokens_out)
        if self.budget:
            self.budget.charge(cost)
        self._append(
            Step(
                index=idx, kind="llm", node=node, name=name, decision=reply.text, input_ref=ref,
                view=view(reply.text) if view else None,
                model=reply.model, tokens_in=reply.tokens_in, tokens_out=reply.tokens_out,
                latency_ms=reply.latency_ms or (time.perf_counter() - started) * 1000, cost_usd=cost, meta=meta,
            )
        )  # fmt: skip
        return reply

    # -------------------------------------------------------------------- tool
    def tool(
        self,
        node: str | None,
        name: str,
        args: dict[str, Any],
        call: Callable[[], Any],
        *,
        replay_safe: bool = False,
        encode: Callable[[Any], Any] = to_jsonable,
        decode: Callable[[Any], Any] = lambda x: x,
    ) -> Any:
        idx = self._next_index()
        decision = to_jsonable(args)
        ref = stable_hash({"tool": name, "args": decision})
        mode = self._mode_for(idx)
        if mode in ("replay", "transplant"):
            rec = self._recorded(idx, "tool", node, name, ref) if mode == "replay" else self.transplant
            assert rec is not None
            self._append(rec.model_copy(update={"index": idx, "input_ref": ref,
                                                "meta": {**rec.meta, "replayed" if mode == "replay" else "transplanted": True}}))  # fmt: skip
            if rec.error is not None and rec.observation is None:
                raise RecordedToolError(rec.error)
            return decode(rec.observation)
        if self.source is not None and not replay_safe:
            raise UnsafeReplay(f"tool '{name}' is not replay-safe; refusing to execute it live during a fork")
        started = time.perf_counter()
        try:
            result = call()
        except Exception as exc:
            self._append(Step(index=idx, kind="tool", node=node, name=name, decision=decision, input_ref=ref,
                              error=f"{type(exc).__name__}: {exc}"[:500], replay_safe=replay_safe,
                              latency_ms=(time.perf_counter() - started) * 1000))  # fmt: skip
            raise
        obs = encode(result)
        err = obs.get("error") if isinstance(obs, dict) and isinstance(obs.get("error"), str) else None
        self._append(Step(index=idx, kind="tool", node=node, name=name, decision=decision, observation=obs,
                          input_ref=ref, error=err, replay_safe=replay_safe,
                          latency_ms=(time.perf_counter() - started) * 1000))  # fmt: skip
        return result
