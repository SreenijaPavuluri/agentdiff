"""Resumable run storage: every finished run is appended to a JSONL file immediately."""

from __future__ import annotations

import threading
from pathlib import Path

from agentdiff.runner import Harness, run_suite
from agentdiff.schema import Trajectory, read_jsonl, write_jsonl

_lock = threading.Lock()


def load(path: Path) -> list[Trajectory]:
    return read_jsonl(path) if path.exists() else []


def run_resumable(harness: Harness, tasks: list[str], k: int, path: Path, seed0: int = 0,
                  concurrency: int = 1) -> list[Trajectory]:  # fmt: skip
    """Run seeds seed0..seed0+k-1 per task, skipping (task, seed) pairs already in ``path``."""
    done = {(t.task_id, t.seed) for t in load(path) if t.outcome.infra_error is None}
    todo: dict[int, list[str]] = {}
    for tid in tasks:
        for j in range(k):
            if (tid, seed0 + j) not in done:
                todo.setdefault(seed0 + j, []).append(tid)

    def save(t: Trajectory) -> None:
        with _lock:
            write_jsonl([t], path, append=True)
        print(f"  {path.stem} {t.task_id} seed={t.seed} {'PASS' if t.outcome.passed else 'fail'} "
              f"{t.wall_ms / 1000:.0f}s{' INFRA:' + t.outcome.infra_error if t.outcome.infra_error else ''}", flush=True)

    for seed, tids in sorted(todo.items()):
        run_suite(harness, tids, k=1, seed0=seed, concurrency=concurrency, on_result=save)
    runs = [t for t in load(path) if t.task_id in tasks and seed0 <= (t.seed or 0) < seed0 + k]
    # keep the latest valid record per (task, seed)
    best: dict[tuple[str, int | None], Trajectory] = {}
    for t in runs:
        if t.outcome.infra_error is None or (t.task_id, t.seed) not in best:
            best[(t.task_id, t.seed)] = t
    return sorted(best.values(), key=lambda t: (t.task_id, t.seed or 0))
