"""Shared setup for the code-verify-agents experiments (local Ollama model, $0)."""

from __future__ import annotations

import os
from pathlib import Path

from agentdiff.integrations.codeverify import CodeVerifyHarness, Fault
from agentdiff.llm import ChatModel

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASET = Path(os.environ.get("CODEVERIFY_DATASET", ROOT.parent / "code-verify-agents/benchmarks/dataset"))
AGENT_MODEL = os.environ.get("AGENTDIFF_AGENT_MODEL", "qwen2.5-coder:1.5b")
WEAK_MODEL = os.environ.get("AGENTDIFF_WEAK_MODEL", "qwen2.5-coder:0.5b")
TEMPERATURE = 0.7
SETTINGS = {"planner_mode": "llm", "inspector_llm": True, "max_patch_attempts": 2, "max_graph_steps": 16}

_models: dict[str, ChatModel] = {}


def model(name: str) -> ChatModel:
    if name not in _models:
        _models[name] = ChatModel(model=name, provider="ollama", temperature=TEMPERATURE, max_tokens=1200)
    return _models[name]


def _call(name: str):  # noqa: ANN202
    m = model(name)
    return lambda system, user, *, temperature=TEMPERATURE, seed=None: m.complete(system, user, temperature=temperature, seed=seed)


def make_harness(version: str, fault: Fault | None = None, tasks: list[str] | None = None) -> CodeVerifyHarness:
    return CodeVerifyHarness(
        version=version, models={AGENT_MODEL: _call(AGENT_MODEL), WEAK_MODEL: _call(WEAK_MODEL)},
        default_model=AGENT_MODEL, dataset_dir=DATASET, fault=fault or Fault(), temperature=TEMPERATURE,
        settings_overrides=dict(SETTINGS), task_filter=tasks,
    )
