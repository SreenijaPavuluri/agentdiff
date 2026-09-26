"""Unbiased per-task estimators of pass@1, pass@k and pass^k from n runs with c passes.

* pass@k  = P(at least one of k runs passes) = 1 - (1-p)^k.
  Unbiased estimator 1 - C(n-c, k)/C(n, k)  (Chen et al. 2021, "Evaluating LLMs trained on code").
* pass^k  = P(all k runs pass) = p^k.
  Unbiased estimator C(c, k)/C(n, k)  (Yao et al. 2024, tau-bench).

Both are unbiased for any p under i.i.d. Bernoulli runs; ``tests/test_estimators.py``
checks this exactly by summing the Binomial(n, p) pmf.
"""

from __future__ import annotations

from math import comb

import numpy as np


def _check(n: int, c: int, k: int) -> None:
    if not (0 <= c <= n):
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")


def pass_at_k(n: int, c: int, k: int) -> float:
    _check(n, c, k)
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k (the "all k succeed" reliability metric)."""
    _check(n, c, k)
    return comb(c, k) / comb(n, k)


def pass_at_k_vec(n: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """Vectorised pass@k for arrays of (n, c); used inside the bootstrap."""
    return np.array([pass_at_k(int(ni), int(ci), k) for ni, ci in zip(n, c, strict=True)])


def pass_hat_k_vec(n: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    return np.array([pass_hat_k(int(ni), int(ci), k) for ni, ci in zip(n, c, strict=True)])


METRICS = {
    "pass@k": pass_at_k,
    "pass^k": pass_hat_k,
}
