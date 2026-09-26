"""A tiny offline bug-fixing agent with *planted* failure causes (no API key needed).

Loop: ``list_files`` -> LLM picks a file -> ``read_file`` -> LLM writes a patch -> ``run_tests``.
The "LLM" is a seeded stochastic policy, so repeated runs are flaky in a controlled way:

* ``p_wrong_file[task]``: probability the planner opens the wrong file (a *decision* cause at
  ``planner:llm:llm#1``);
* ``p_truncate``: probability ``read_file`` returns a truncated file (an *observation* cause at
  ``coder:tool:read_file#1``), after which the patch is wrong;
* ``p_bad_patch``: residual probability of a wrong patch even with good context.

It exists for tests, the ``agentdiff demo`` command and the README screenshot. It is not
evidence that agentdiff works on real agents; see ``experiments/`` for that.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from agentdiff.schema import Outcome
from agentdiff.tape import LLMReply, Tape

TASKS: dict[str, dict[str, str]] = {
    "t1_offbyone": {"bug": "calc.py", "files": "calc.py utils.py io.py"},
    "t2_nullcheck": {"bug": "parse.py", "files": "parse.py calc.py fmt.py cli.py"},
    "t3_sorting": {"bug": "order.py", "files": "order.py io.py"},
    "t4_encoding": {"bug": "text.py", "files": "text.py io.py fmt.py utils.py net.py"},
    "t5_unit": {"bug": "units.py", "files": "units.py calc.py"},
    "t6_cache": {"bug": "cache.py", "files": "cache.py store.py net.py io.py"},
}


def file_body(task: str, name: str) -> str:
    lines = [f"# {name} for {task}"] + [f"def f{i}(x):\n    return x + {i}" for i in range(6)]
    if name == TASKS[task]["bug"]:
        lines.insert(4, "def buggy(xs):\n    return xs[1:]  # BUG: drops first element")
    return "\n".join(lines)


@dataclass
class ToyAgent:
    version: str = "v1"
    p_wrong_file: dict[str, float] = field(default_factory=lambda: {
        "t1_offbyone": 0.0, "t2_nullcheck": 0.35, "t3_sorting": 0.0, "t4_encoding": 0.5,
        "t5_unit": 0.0, "t6_cache": 0.2,
    })  # fmt: skip
    p_truncate: float = 0.0
    truncate_tasks: tuple[str, ...] = ("t3_sorting",)
    p_bad_patch: float = 0.03

    def task_ids(self) -> list[str]:
        return list(TASKS)

    def run(self, task_id: str, tape: Tape, seed: int) -> Outcome:
        spec = TASKS[task_id]
        model_rng = random.Random(f"model-{seed}")
        env_rng = random.Random(f"env-{seed}")
        files = tape.tool("coder", "list_files", {"task": task_id}, lambda: spec["files"].split(), replay_safe=True)

        def plan() -> LLMReply:
            wrong = [f for f in files if f != spec["bug"]]
            pick = model_rng.choice(wrong) if wrong and model_rng.random() < self.p_wrong_file.get(task_id, 0) else spec["bug"]
            return LLMReply(text=f"open {pick}", model="toy-model", tokens_in=120, tokens_out=6)

        reply = tape.llm("planner", {"prompt": f"which file has the bug? {files}"}, plan)
        path = reply.text.removeprefix("open ").strip()

        def read() -> str:
            body = file_body(task_id, path)
            if task_id in self.truncate_tasks and env_rng.random() < self.p_truncate:
                return body[:30]
            return body

        body = tape.tool("coder", "read_file", {"path": path}, read, replay_safe=True)

        def patch() -> LLMReply:
            sees_bug = "BUG" in body
            good = sees_bug and model_rng.random() >= self.p_bad_patch
            text = "return xs  # fixed" if good else f"return sorted(xs)  # guess in {path}"
            return LLMReply(text=text, model="toy-model", tokens_in=300, tokens_out=12)

        patch_reply = tape.llm("coder", {"prompt": f"fix:\n{body}"}, patch)
        result = tape.tool("coder", "run_tests", {"patch": patch_reply.text, "file": path},
                           lambda: {"passed": patch_reply.text.startswith("return xs") and path == spec["bug"]},
                           replay_safe=True)  # fmt: skip
        return Outcome(passed=bool(result["passed"]), detail={"file": path})
