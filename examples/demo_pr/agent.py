"""Demo agent for the CI gate example.

The "model" is agentdiff's offline toy policy (no API key, runs in CI in seconds). To make
the demo behave like a prompt-sensitive agent, how often the planner opens the wrong file
depends on whether the system prompt tells it to read the traceback first. The pull request
in ``bad-prompt.patch`` deletes that instruction; the gate should catch it.

This is a simulation of a prompt regression, not a real LLM. The real-agent evidence for
agentdiff is in ``experiments/``.
"""

from __future__ import annotations

from pathlib import Path

from agentdiff.demo.toy import ToyAgent

PROMPT = Path(__file__).with_name("prompt.txt")


def make_harness(version: str) -> ToyAgent:
    prompt = PROMPT.read_text().lower()
    reads_traceback = "traceback" in prompt
    base = {"t1_offbyone": 0.05, "t2_nullcheck": 0.3, "t3_sorting": 0.05, "t4_encoding": 0.4, "t5_unit": 0.05, "t6_cache": 0.2}
    p_wrong = base if reads_traceback else {t: min(0.9, p + 0.4) for t, p in base.items()}
    return ToyAgent(version=version, p_wrong_file=p_wrong, p_truncate=0.2)
