"""Step similarity.

``content_sim(a, b)`` in [0, 1] compares two *compatible* steps (same node, kind and name):

    content_sim = 0.5 * S(view_or_decision_a, view_or_decision_b) + 0.5 * S_obs(a, b)

(or just the decision term when neither step has an observation or error, e.g. LLM steps).

``step_sim(a, b)`` is the alignment score: 0 for incompatible steps, otherwise
``0.5 + 0.5 * content_sim`` so compatible steps always pair up in the alignment even when
their contents differ.

``S(x, y)`` flattens both JSON values into ``path -> leaf`` maps and averages a per-leaf
similarity over the union of paths (a path present on one side only scores 0):

* numbers / booleans / null: exact equality after rounding floats to 6 significant digits;
* strings: normalised (whitespace collapsed, volatile tokens masked) then compared with
  ``difflib.SequenceMatcher`` over lines (multi-line text) or word tokens (single line).

``S_obs`` is ``S`` on observations, except that it is 0 when exactly one step has an error.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from agentdiff.schema import Step

IGNORE_KEYS = frozenset({"duration_ms", "latency_ms", "elapsed_ms", "ts", "timestamp", "run_id", "call_id", "id"})

_MASKS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<uuid>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?\b"), "<ts>"),
    (re.compile(r"(/private)?/(tmp|var/folders)/[^\s\"']+"), "<tmp>"),
    (re.compile(r"\b0x[0-9a-f]{6,}\b", re.I), "<addr>"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "<hex>"),
    (re.compile(r"\b\d+\.\d+\s?(ms|s)\b"), "<dur>"),
]


def normalize_text(s: str) -> str:
    for pat, rep in _MASKS:
        s = pat.sub(rep, s)
    lines = [" ".join(line.split()) for line in s.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def flatten(obj: Any, prefix: str = "", ignore: frozenset[str] = IGNORE_KEYS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj:
            out[prefix or "."] = "{}"
        for k, v in obj.items():
            if k in ignore:
                continue
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k), ignore))
    elif isinstance(obj, list | tuple):
        if not obj:
            out[prefix or "."] = "[]"
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]", ignore))
    else:
        out[prefix or "."] = obj
    return out


@lru_cache(maxsize=200_000)
def text_sim(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if "\n" in a or "\n" in b:
        return SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False).ratio()
    return SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def leaf_sim(x: Any, y: Any) -> float:
    if isinstance(x, str) and isinstance(y, str):
        return text_sim(x, y)
    if isinstance(x, bool) or isinstance(y, bool) or x is None or y is None:
        return float(x == y)
    if isinstance(x, int | float) and isinstance(y, int | float):
        return float(f"{x:.6g}" == f"{y:.6g}")
    return float(str(x) == str(y))


def value_sim(x: Any, y: Any) -> float:
    if x is None and y is None:
        return 1.0
    if x is None or y is None:
        return 0.0
    if isinstance(x, str) and isinstance(y, str):
        return text_sim(x, y)
    fx, fy = flatten(x), flatten(y)
    keys = fx.keys() | fy.keys()
    if not keys:
        return 1.0
    return sum(leaf_sim(fx[k], fy[k]) if k in fx and k in fy else 0.0 for k in keys) / len(keys)


def decision_sim(a: Step, b: Step) -> float:
    da = a.view if a.view is not None else a.decision
    db = b.view if b.view is not None else b.decision
    return value_sim(da, db)


def observation_sim(a: Step, b: Step) -> float:
    if (a.error is None) != (b.error is None):
        return 0.0
    return value_sim(a.observation, b.observation)


def compatible(a: Step, b: Step) -> bool:
    return a.signature == b.signature


def content_sim(a: Step, b: Step) -> float:
    if not compatible(a, b):
        return 0.0
    if a.observation is None and b.observation is None and a.error is None and b.error is None:
        return decision_sim(a, b)  # e.g. LLM steps: the decision is the whole content
    return 0.5 * decision_sim(a, b) + 0.5 * observation_sim(a, b)


def step_sim(a: Step, b: Step) -> float:
    return 0.0 if not compatible(a, b) else 0.5 + 0.5 * content_sim(a, b)
