"""Run every experiment block, resumably. Usage: python run_all.py [screen|baseline|faults|nulls|all]

Blocks (all runs append to results/runs/*.jsonl as they finish):
  screen    20 tasks x 4 runs (seeds 1000-1003), used ONLY to choose the task subset
  baseline  chosen tasks x 16 runs (seeds 0-15): baseline pool; seeds 0-7 are the gate baseline
  faults    each fault x chosen tasks x 8 runs (seeds 10000-10007)
  nulls     N01 (whitespace prompt edit) x chosen tasks x 8 runs (seeds 10000-10007)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

from common import RESULTS, make_harness
from faults import BY_ID, FAULTS, NULLS
from store import load, run_resumable

RUNS = RESULTS / "runs"
TASKS_FILE = RESULTS / "chosen_tasks.json"
N_TASKS = 10


def screen() -> None:
    h = make_harness("screen")
    runs = run_resumable(h, h.task_ids(), 4, RUNS / "screen.jsonl", seed0=1000)
    rate: dict[str, list[int]] = {}
    for r in runs:
        rate.setdefault(r.task_id, []).append(int(r.outcome.passed))
    # prefer tasks that are neither always-fail nor always-pass on the screen, then higher pass rate
    scored = sorted(rate, key=lambda t: (not (0 < sum(rate[t]) < len(rate[t])), -sum(rate[t]), t))
    chosen = sorted(scored[:N_TASKS])
    TASKS_FILE.write_text(json.dumps({"tasks": chosen, "screen": {t: f"{sum(v)}/{len(v)}" for t, v in sorted(rate.items())},
                                      "rule": "flaky on screen first (0<c<n), then by screen pass count; seeds 1000-1003 not reused"}, indent=1))
    print("chosen:", chosen)


def tasks() -> list[str]:
    return json.loads(TASKS_FILE.read_text())["tasks"]


def baseline() -> None:
    run_resumable(make_harness("baseline"), tasks(), 16, RUNS / "baseline.jsonl", seed0=0)


def faults(ids: list[str] | None = None) -> None:
    reg = RESULTS / "faults_registry.sha256"
    digest = hashlib.sha256((RESULTS.parent / "experiments/faults.py").read_bytes()).hexdigest()
    if not reg.exists():
        reg.write_text(f"{digest}  experiments/faults.py  (recorded {time.strftime('%Y-%m-%d %H:%M')})\n")
    for f in FAULTS:
        if ids and f.id not in ids:
            continue
        run_resumable(make_harness(f.id, fault=f), tasks(), 8, RUNS / f"{f.id}.jsonl", seed0=10_000)


def nulls() -> None:
    for f in NULLS:
        run_resumable(make_harness(f.id, fault=f), tasks(), 8, RUNS / f"{f.id}.jsonl", seed0=10_000)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    extra = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    t0 = time.time()
    if what in ("screen", "all") and not TASKS_FILE.exists():
        screen()
    if what in ("baseline", "all"):
        baseline()
    if what in ("nulls", "all"):
        nulls()
    if what in ("faults", "all"):
        faults(extra)
    print(f"done {what} in {(time.time() - t0) / 3600:.2f}h; runs on disk:",
          {p.stem: len(load(p)) for p in sorted(RUNS.glob('*.jsonl'))})
    _ = BY_ID
