"""Confidence intervals.

* :func:`clopper_pearson` - exact binomial interval; per-task pass^k / pass@k intervals are
  obtained by pushing its endpoints through the monotone maps p -> p^k and p -> 1-(1-p)^k,
  which preserves coverage exactly.
* :func:`wilson` - score interval, used as the building block of Newcombe's interval.
* :func:`newcombe_diff` - Newcombe (1998) hybrid score interval ("method 10") for the
  difference of two independent proportions; used to decide replay confirmation.
* :func:`suite_bootstrap` - two-stage bootstrap (tasks, then runs within task) for suite means.
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb, sqrt

import numpy as np
from scipy import stats


def clopper_pearson(c: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if c == 0 else float(stats.beta.ppf(alpha / 2, c, n - c + 1))
    hi = 1.0 if c == n else float(stats.beta.ppf(1 - alpha / 2, c + 1, n - c))
    return lo, hi


def wilson(c: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = c / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_diff(c1: int, n1: int, c2: int, n2: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and CI for p1 - p2 (independent samples), Newcombe method 10."""
    p1, p2 = c1 / n1, c2 / n2
    l1, u1 = wilson(c1, n1, alpha)
    l2, u2 = wilson(c2, n2, alpha)
    d = p1 - p2
    lo = d - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def per_task_interval(metric: str, c: int, n: int, k: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact-coverage interval for the population value of pass@1 / pass@k / pass^k on one task."""
    lo, hi = clopper_pearson(c, n, alpha)
    if metric == "pass@1":
        return lo, hi
    if metric == "pass^k":
        return lo**k, hi**k
    if metric == "pass@k":
        return 1 - (1 - lo) ** k, 1 - (1 - hi) ** k
    raise ValueError(metric)


def _vec_metric(metric: str, k: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Vectorised estimator over (n, c) arrays of equal shape, via lookup tables."""

    def f(n: np.ndarray, c: np.ndarray) -> np.ndarray:
        n_max = int(n.max())
        table = np.zeros((n_max + 1, n_max + 1))
        for nn in np.unique(n):
            nn = int(nn)
            for cc in range(nn + 1):
                if metric == "pass@1":
                    table[nn, cc] = cc / nn
                elif metric == "pass^k":
                    table[nn, cc] = comb(cc, k) / comb(nn, k)
                elif metric == "pass@k":
                    table[nn, cc] = 1 - comb(nn - cc, k) / comb(nn, k)
                else:
                    raise ValueError(metric)
        return table[n, c]

    return f


def suite_bootstrap(
    n: np.ndarray,
    c: np.ndarray,
    metric: str = "pass@1",
    k: int = 1,
    alpha: float = 0.05,
    b: int = 5000,
    seed: int = 0,
    two_stage: bool = True,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI for the suite mean of a per-task metric.

    Stage 1 resamples tasks with replacement; stage 2 resamples each task's n runs with
    replacement, which is exactly c* ~ Binomial(n, c/n). The two-stage version reflects both
    task sampling and finite runs per task (and is known to be mildly conservative).
    """
    n = np.asarray(n, dtype=int)
    c = np.asarray(c, dtype=int)
    f = _vec_metric(metric, k)
    point = float(f(n, c).mean())
    if len(n) == 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(b, len(n)))
    nb = n[idx]
    cb = rng.binomial(nb, c[idx] / nb) if two_stage else c[idx]
    means = f(nb, cb).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
