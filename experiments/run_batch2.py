"""Batch 2 (added after batch 1; see README): wait for F13 to reach k=6, stop run_all, run F14 at k=6."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import time

from common import RESULTS, make_harness
from faults import BATCH2
from run_all import RUNS, tasks
from store import load, run_resumable

K = 6
while len(load(RUNS / "F13.jsonl")) < K * len(tasks()):
    time.sleep(20)
for pid in subprocess.run(["pgrep", "-f", "run_all.py"], capture_output=True, text=True).stdout.split():
    os.kill(int(pid), signal.SIGTERM)
print("F13 reached k=6; stopped run_all", flush=True)
digest = hashlib.sha256((RESULTS.parent / "experiments/faults.py").read_bytes()).hexdigest()
with (RESULTS / "faults_registry.sha256").open("a") as fh:
    fh.write(f"{digest}  experiments/faults.py with batch 2 (recorded {time.strftime('%Y-%m-%d %H:%M')})\n")
for f in BATCH2:
    run_resumable(make_harness(f.id, fault=f), tasks(), K, RUNS / f"{f.id}.jsonl", seed0=10_000)
print("batch 2 done", flush=True)
