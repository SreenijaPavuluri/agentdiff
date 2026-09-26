"""Smoke test: 2 tasks x k runs; prints pass/fail, steps and timing."""

from __future__ import annotations

import sys
import time

from common import RESULTS, make_harness

from agentdiff.runner import run_suite
from agentdiff.schema import write_jsonl

k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
tasks = sys.argv[2].split(",") if len(sys.argv) > 2 else ["t01_sum_to", "t05_fizzbuzz"]
h = make_harness("smoke", tasks=tasks)
t0 = time.time()
runs = run_suite(h, h.task_ids(), k=k, concurrency=1,
                 on_result=lambda r: print(r.task_id, r.outcome.passed, r.outcome.detail.get("status"),
                                           f"{r.wall_ms / 1000:.0f}s", len(r.steps), r.outcome.infra_error, flush=True))
print(f"total {time.time() - t0:.0f}s for {len(runs)} runs; tokens/run {sum(r.tokens for r in runs) / len(runs):.0f}")
write_jsonl(runs, RESULTS / "smoke" / f"smoke_k{k}.jsonl")
