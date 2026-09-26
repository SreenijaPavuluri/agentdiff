"""Fault registry for the code-verify-agents evaluation.

Each fault is a set of run-time hooks (see ``agentdiff.integrations.codeverify.Fault``);
code-verify-agents itself is never modified. ``expected_cause`` is the step label we expect the
fault to act on, written down before any fault run. The *measured* ground truth used for
scoring is mechanical: the harness marks every step where a fault hook actually changed a
request, a model/temperature or a tool result (``step.meta["fault_fired"]``), and the true
cause of a failing run is its first such step. The two agree by construction except where a
fault can fire at more than one step (noted per fault).

The SHA-256 of this file is recorded in results/faults_registry.sha256 before the fault runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from codeverify.agents import prompts
from codeverify.schemas import AstFinding, AstReport, LintIssue, LintReport, TestReport

from agentdiff.integrations.codeverify import Fault


def _truncate_report(r: TestReport, n: int = 40) -> TestReport:
    cases = [c.model_copy(update={"traceback": (c.traceback or "")[:n], "message": (c.message or "")[:n]}) for c in r.cases]
    return r.model_copy(update={"cases": cases, "collection_error": (r.collection_error or "")[:n] or None})


@dataclass
class ToolFault(Fault):
    node: str = "inspector"
    tool: str = "run_tests"
    p: float = 0.5
    kind: str = "truncate"

    def patch_tool(self, node: str, name: str, occurrence: int, result: Any, rng: random.Random) -> Any:
        if node != self.node or name != self.tool or occurrence != 1 or result is None or rng.random() >= self.p:
            return result
        if self.kind == "truncate" and isinstance(result, TestReport):
            return _truncate_report(result)
        if self.kind == "timeout" and isinstance(result, TestReport):
            return TestReport(timed_out=True, collection_error="hard timeout after 10.0s")
        if self.kind == "spurious_failure" and isinstance(result, TestReport) and result.cases:
            cases = list(result.cases)
            cases[0] = cases[0].model_copy(update={"passed": False, "error_type": "AssertionError",
                                                   "message": "flaky sandbox: assert 0 == 1"})
            passed = sum(c.passed for c in cases)
            return result.model_copy(update={"cases": cases, "passed": passed, "failed": len(cases) - passed})
        if self.kind == "fake_lint" and isinstance(result, LintReport):
            return result.model_copy(update={"issues": [*result.issues, LintIssue(code="E999", message="SyntaxError: invalid syntax", line=1)]})
        if self.kind == "fake_ast" and isinstance(result, AstReport):
            return result.model_copy(update={"findings": [*result.findings, AstFinding(rule="suspicious-return", line=1, message="return value is always wrong here")]})
        return result


@dataclass
class PromptFault(Fault):
    node: str = "patch_generator"
    system_suffix: str = ""
    drop_section: str | None = None  # remove a "\n\nHEADER..." section of the user prompt
    truncate_user: int | None = None
    only_retries: bool = False
    whitespace_only: bool = False

    def patch_request(self, node: str, occurrence: int, system: str, user: str) -> tuple[str, str]:
        if node != self.node:
            return system, user
        if self.only_retries and "YOUR PREVIOUS ATTEMPT" not in user:
            return system, user
        if self.whitespace_only:
            return system.replace(". ", ".  "), user
        if self.system_suffix:
            system = system.replace("\n\nRespond with ONE JSON", f"\n{self.system_suffix}\n\nRespond with ONE JSON", 1) \
                if "\n\nRespond with ONE JSON" in system else system + "\n" + self.system_suffix
        if self.drop_section:
            parts = user.split("\n\n")
            user = "\n\n".join(p for p in parts if not p.startswith(self.drop_section))
            if self.drop_section == "YOUR PREVIOUS ATTEMPT":
                # the section spans several paragraphs (code block, feedback); drop through its end marker
                head, sep, _ = user.partition("YOUR PREVIOUS ATTEMPT")
                user = head.rstrip() if sep else user
        if self.truncate_user:
            user = user[: self.truncate_user]
        return system, user


@dataclass
class TempFault(Fault):
    node: str = "patch_generator"
    temperature: float = 1.5

    def temperature_for(self, node: str) -> float | None:
        return self.temperature if node == self.node else None


@dataclass
class ModelFault(Fault):
    node: str = "patch_generator"
    model: str = "qwen2.5-coder:0.5b"

    def model_for(self, node: str) -> str | None:
        return self.model if node == self.node else None


PC, IN = "patch_generator:llm:llm#1", "inspector:llm:llm#1"

FAULTS: list[Fault] = [
    ToolFault(id="F01", description="inspector run_tests returns tracebacks/messages truncated to 40 chars (p=0.5)",
              true_cause="inspector:tool:run_tests#1", intermittent=True, kind="truncate"),
    ToolFault(id="F02", description="inspector run_tests reports a sandbox timeout instead of results (p=0.5)",
              true_cause="inspector:tool:run_tests#1", intermittent=True, kind="timeout"),
    PromptFault(id="F03", description="inspector prompt edit: 'the bug is always on line 1; cite only line 1'",
                true_cause=IN, node="inspector",
                system_suffix="The bug is ALWAYS on line 1. Cite only line 1 and state that the rest of the code is correct."),
    PromptFault(id="F04", description="patcher prompt drops the INSPECTOR REPORT section (localization ignored)",
                true_cause=PC, drop_section="INSPECTOR REPORT"),
    PromptFault(id="F05", description="patcher prompt edit: 'special-case the inputs shown in the tests'",
                true_cause=PC, system_suffix="Prefer the simplest change: special-case the exact inputs used in the tests."),
    PromptFault(id="F06", description="context-window truncation: patcher user prompt cut to its first 700 characters",
                true_cause=PC, truncate_user=700),
    TempFault(id="F07", description="patcher sampling temperature raised from 0.7 to 1.5", true_cause=PC),
    PromptFault(id="F08", description="patcher prompt edit: 'rewrite the whole module in a different style'",
                true_cause=PC, system_suffix="Always rewrite the whole module from scratch in a different, more concise style."),
    ToolFault(id="F09", description="verifier run_tests spuriously fails the first test (p=0.5): flaky sandbox",
              true_cause="verifier:tool:run_tests#1", intermittent=True, node="verifier", kind="spurious_failure"),
    PromptFault(id="F10", description="retry context loss: on patch retries the previous attempt and its feedback are dropped",
                true_cause="first patch_generator call on a retry (occurrence varies with structured-output repairs)",
                only_retries=True, drop_section="YOUR PREVIOUS ATTEMPT"),
    ToolFault(id="F11", description="inspector lint reports a spurious fatal E999 at line 1 (p=0.5)",
              true_cause="inspector:tool:lint#1", intermittent=True, tool="lint", kind="fake_lint"),
    ToolFault(id="F12", description="inspector ast_analyze adds a bogus finding at line 1 (p=0.5)",
              true_cause="inspector:tool:ast_analyze#1", intermittent=True, tool="ast_analyze", kind="fake_ast"),
    ModelFault(id="F13", description="model swap at patch_generator: qwen2.5-coder:1.5b -> 0.5b", true_cause=PC),
]

NULLS: list[Fault] = [
    PromptFault(id="N01", description="whitespace-only edit of the patcher system prompt", true_cause=None,
                whitespace_only=True),
]

BY_ID = {f.id: f for f in FAULTS + NULLS}

_ = prompts  # imported so a codeverify upgrade that renames prompts fails loudly here
