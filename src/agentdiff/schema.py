"""Trajectory data model shared by adapters, statistics, alignment, replay and reports.

A *trajectory* is one run of one agent version on one task: an ordered list of
steps plus a verifiable outcome. Every step separates what the agent chose
(``decision``) from what the environment returned (``observation``); the
divergence finder relies on that split to say "the agent chose differently"
versus "the tool answered differently".
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

StepKind = Literal["llm", "tool", "agent", "other"]


def stable_hash(obj: Any) -> str:
    """Order-independent content hash for JSON-able values (used for replay request checks)."""
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Step(BaseModel):
    index: int
    kind: StepKind
    node: str | None = None
    name: str
    decision: Any = None
    observation: Any = None
    view: Any = None  # optional normalised projection of `decision` used for comparison
    input_ref: str | None = None
    error: str | None = None
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float | None = None
    cost_usd: float = 0.0
    replay_safe: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def signature(self) -> tuple[str, str, str]:
        """Steps are only comparable when (node, kind, name) agree."""
        return (self.node or "", self.kind, self.name)


class Outcome(BaseModel):
    passed: bool
    source: Literal["rule", "judge", "imported"] = "rule"
    detail: dict[str, Any] = Field(default_factory=dict)
    infra_error: str | None = None  # harness crashed: excluded from statistics by default


class ForkInfo(BaseModel):
    source_run: str
    fork_step: int
    intervention: Literal["resample", "keep", "transplant"]


class Trajectory(BaseModel):
    task_id: str
    run_id: str
    agent_version: str
    seed: int | None = None
    steps: list[Step] = Field(default_factory=list)
    outcome: Outcome
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    wall_ms: float = 0.0
    parent: ForkInfo | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def tokens(self) -> int:
        return sum(s.tokens_in + s.tokens_out for s in self.steps)

    def labels(self) -> list[str]:
        return step_labels(self.steps)


def step_labels(steps: list[Step]) -> list[str]:
    """Stable per-run step labels ``node:kind:name#occurrence`` (occurrence is 1-based)."""
    seen: Counter[tuple[str, str, str]] = Counter()
    out = []
    for s in steps:
        seen[s.signature] += 1
        node, kind, name = s.signature
        out.append(f"{node}:{kind}:{name}#{seen[s.signature]}")
    return out


# ---------------------------------------------------------------------------- IO


def write_jsonl(trajs: list[Trajectory], path: str | Path, append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w") as fh:
        for t in trajs:
            fh.write(t.model_dump_json() + "\n")


def read_jsonl(path: str | Path) -> list[Trajectory]:
    with Path(path).open() as fh:
        return [Trajectory.model_validate_json(line) for line in fh if line.strip()]


def group_by_task(trajs: list[Trajectory]) -> dict[str, list[Trajectory]]:
    out: dict[str, list[Trajectory]] = {}
    for t in trajs:
        out.setdefault(t.task_id, []).append(t)
    return dict(sorted(out.items()))
