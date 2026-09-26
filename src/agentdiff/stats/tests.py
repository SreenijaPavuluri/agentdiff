"""Hypothesis tests used by the regression gate and the divergence finder.

Stratified exact test (overall gate)
------------------------------------
For task i the baseline has n_b,i runs and the candidate n_c,i runs; s_i is the pooled number
of passes. Under H0 ("the candidate is the same agent") the run labels are exchangeable
*within* each task, so the number of candidate passes X_i is Hypergeometric(N_i, s_i, n_c,i),
independently across tasks. This is the conditional (exact) version of the
Cochran-Mantel-Haenszel test. When every task has the same (n_b, n_c), the gate statistic
D = mean_i(p_c,i - p_b,i) is an increasing affine function of sum_i X_i, so the exact
p-value is a convolution of hypergeometric pmfs. With n_b = n_c = 1 this reduces to the
exact McNemar test (only discordant tasks can move). Unequal run counts fall back to Monte
Carlo over the same null.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def fisher_less(x_c: int, n_c: int, x_b: int, n_b: int) -> float:
    """One-sided Fisher exact p-value that the candidate's pass rate is lower than the baseline's."""
    s, big_n = x_c + x_b, n_c + n_b
    return float(stats.hypergeom.cdf(x_c, big_n, s, n_c))


def fisher_greater(x_c: int, n_c: int, x_b: int, n_b: int) -> float:
    s, big_n = x_c + x_b, n_c + n_b
    return float(stats.hypergeom.sf(x_c - 1, big_n, s, n_c))


def fisher_two_sided(x_c: int, n_c: int, x_b: int, n_b: int) -> float:
    return float(stats.fisher_exact([[x_c, n_c - x_c], [x_b, n_b - x_b]]).pvalue)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


@dataclass
class StratifiedResult:
    statistic: float  # D = mean_i(p_c,i - p_b,i)
    p_less: float  # P(D* <= D) : evidence the candidate is worse
    p_greater: float  # P(D* >= D) : evidence the candidate is better
    method: str


def stratified_exact_test(
    x_c: list[int], n_c: list[int], x_b: list[int], n_b: list[int], mc_samples: int = 20000, seed: int = 0,
    force_mc: bool = False,
) -> StratifiedResult:
    x_c_a, n_c_a, x_b_a, n_b_a = (np.asarray(v, dtype=int) for v in (x_c, n_c, x_b, n_b))
    d_obs = float(np.mean(x_c_a / n_c_a - x_b_a / n_b_a))
    s = x_c_a + x_b_a
    big_n = n_c_a + n_b_a
    if not force_mc and len(set(n_c_a.tolist())) == 1 and len(set(n_b_a.tolist())) == 1:
        # exact: distribution of T = sum X_i via convolution
        pmf = np.array([1.0])
        for si, ni, nci in zip(s, big_n, n_c_a, strict=True):
            lo, hi = max(0, nci - (ni - si)), min(si, nci)
            support = np.arange(lo, hi + 1)
            p_i = np.zeros(hi + 1)
            p_i[support] = stats.hypergeom.pmf(support, ni, si, nci)
            pmf = np.convolve(pmf, p_i)
        t_obs = int(x_c_a.sum())
        eps = 1e-12
        p_less = float(min(1.0, pmf[: t_obs + 1].sum() + eps))
        p_greater = float(min(1.0, pmf[t_obs:].sum() + eps))
        return StratifiedResult(d_obs, p_less, p_greater, "exact (convolved hypergeometric)")
    rng = np.random.default_rng(seed)
    xs = np.stack(
        [rng.hypergeometric(si, ni - si, nci, size=mc_samples) for si, ni, nci in zip(s, big_n, n_c_a, strict=True)],
        axis=1,
    )
    d_null = (xs / n_c_a - (s - xs) / n_b_a).mean(axis=1)
    tol = 1e-12
    p_less = float((1 + np.sum(d_null <= d_obs + tol)) / (mc_samples + 1))
    p_greater = float((1 + np.sum(d_null >= d_obs - tol)) / (mc_samples + 1))
    return StratifiedResult(d_obs, p_less, p_greater, f"monte-carlo ({mc_samples} draws)")


def paired_bootstrap_diff(
    x_c: list[int], n_c: list[int], x_b: list[int], n_b: list[int], alpha: float = 0.05, b: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Two-stage paired bootstrap CI for D = mean_i(p_c,i - p_b,i); tasks are the pairing unit."""
    x_c_a, n_c_a, x_b_a, n_b_a = (np.asarray(v, dtype=int) for v in (x_c, n_c, x_b, n_b))
    d = float(np.mean(x_c_a / n_c_a - x_b_a / n_b_a))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x_c_a), size=(b, len(x_c_a)))
    pc = rng.binomial(n_c_a[idx], x_c_a[idx] / n_c_a[idx]) / n_c_a[idx]
    pb = rng.binomial(n_b_a[idx], x_b_a[idx] / n_b_a[idx]) / n_b_a[idx]
    diffs = (pc - pb).mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return d, float(lo), float(hi)
