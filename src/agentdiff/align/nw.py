"""Needleman-Wunsch global alignment of two step sequences.

Scoring: ``step_sim`` for compatible pairs (always >= 0.5), incompatible pairs are forbidden,
gap penalty 0. The result is the order-preserving pairing that maximises total similarity
(a similarity-weighted longest common subsequence). Ties prefer the diagonal move, then
consuming ``a`` before ``b``, so the output is deterministic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from agentdiff.align.similarity import step_sim
from agentdiff.schema import Step

Pair = tuple[int | None, int | None]


def align(a: list[Step], b: list[Step], sim: Callable[[Step, Step], float] = step_sim) -> list[Pair]:
    n, m = len(a), len(b)
    s = np.full((n, m), -np.inf)
    for i in range(n):
        for j in range(m):
            v = sim(a[i], b[j])
            if v > 0:
                s[i, j] = v
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = max(dp[i - 1, j - 1] + s[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
    pairs: list[Pair] = []
    i, j = n, m
    eps = 1e-12
    while i > 0 or j > 0:
        if i > 0 and j > 0 and np.isfinite(s[i - 1, j - 1]) and abs(dp[i, j] - (dp[i - 1, j - 1] + s[i - 1, j - 1])) < eps:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif j > 0 and (i == 0 or abs(dp[i, j] - dp[i, j - 1]) < eps):
            pairs.append((None, j - 1))
            j -= 1
        else:
            pairs.append((i - 1, None))
            i -= 1
    pairs.reverse()
    return pairs


def alignment_score(a: list[Step], b: list[Step]) -> float:
    return sum(step_sim(a[i], b[j]) for i, j in align(a, b) if i is not None and j is not None)
