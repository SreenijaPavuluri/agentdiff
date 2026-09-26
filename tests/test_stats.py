"""Statistics checked against analytic results and published reference values."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest
from scipy import stats

from agentdiff.stats.estimators import pass_at_k, pass_hat_k
from agentdiff.stats.intervals import (
    clopper_pearson,
    newcombe_diff,
    per_task_interval,
    suite_bootstrap,
    wilson,
)
from agentdiff.stats.tests import (
    fisher_less,
    fisher_two_sided,
    holm,
    paired_bootstrap_diff,
    stratified_exact_test,
)

# ------------------------------------------------------------------ estimators


@pytest.mark.parametrize("n,c,k", [(5, 2, 2), (8, 5, 3), (6, 6, 4), (7, 0, 2), (8, 1, 8)])
def test_pass_at_k_and_pass_hat_k_match_subset_enumeration(n: int, c: int, k: int) -> None:
    outcomes = [1] * c + [0] * (n - c)
    subsets = list(combinations(range(n), k))
    any_pass = np.mean([any(outcomes[i] for i in s) for s in subsets])
    all_pass = np.mean([all(outcomes[i] for i in s) for s in subsets])
    assert pass_at_k(n, c, k) == pytest.approx(any_pass)
    assert pass_hat_k(n, c, k) == pytest.approx(all_pass)


@pytest.mark.parametrize("p", [0.0, 0.13, 0.5, 0.77, 1.0])
@pytest.mark.parametrize("n,k", [(8, 1), (8, 3), (8, 8), (10, 4)])
def test_estimators_are_exactly_unbiased(p: float, n: int, k: int) -> None:
    pmf = [comb(n, c) * p**c * (1 - p) ** (n - c) for c in range(n + 1)]
    e_hat = sum(w * pass_hat_k(n, c, k) for c, w in enumerate(pmf))
    e_at = sum(w * pass_at_k(n, c, k) for c, w in enumerate(pmf))
    assert e_hat == pytest.approx(p**k, abs=1e-12)
    assert e_at == pytest.approx(1 - (1 - p) ** k, abs=1e-12)


def test_estimator_input_validation() -> None:
    with pytest.raises(ValueError):
        pass_hat_k(4, 5, 2)
    with pytest.raises(ValueError):
        pass_at_k(4, 2, 5)


# ------------------------------------------------------------------- intervals


def test_clopper_pearson_reference_values() -> None:
    assert clopper_pearson(0, 10) == pytest.approx((0.0, 0.308497), abs=1e-6)
    lo, hi = clopper_pearson(5, 10)
    assert (lo, hi) == pytest.approx((0.187086, 0.812914), abs=1e-6)


def test_wilson_reference_value() -> None:
    # Newcombe (1998) Table I, 81/263: Wilson 0.2553-0.3662
    assert wilson(81, 263) == pytest.approx((0.2553, 0.3662), abs=1e-4)


@pytest.mark.parametrize(
    "c1,n1,c2,n2,lo,hi",
    [  # Newcombe (1998) Table II, method 10
        (56, 70, 48, 80, 0.0524, 0.3339),
        (9, 10, 3, 10, 0.1705, 0.8090),
        (6, 7, 2, 7, 0.0582, 0.8062),
        (5, 56, 0, 29, -0.0381, 0.1926),
    ],
)
def test_newcombe_method10_published_values(c1: int, n1: int, c2: int, n2: int, lo: float, hi: float) -> None:
    _, l_, h_ = newcombe_diff(c1, n1, c2, n2)
    assert (l_, h_) == pytest.approx((lo, hi), abs=1e-4)


def test_per_task_interval_is_monotone_map_of_clopper_pearson() -> None:
    lo, hi = clopper_pearson(6, 8)
    assert per_task_interval("pass^k", 6, 8, 4) == pytest.approx((lo**4, hi**4))
    assert per_task_interval("pass@k", 6, 8, 4) == pytest.approx((1 - (1 - lo) ** 4, 1 - (1 - hi) ** 4))


def test_clopper_pearson_coverage_is_at_least_nominal() -> None:
    n = 8
    for p in (0.1, 0.35, 0.6, 0.9):
        cover = sum(stats.binom.pmf(c, n, p) for c in range(n + 1) if clopper_pearson(c, n)[0] <= p <= clopper_pearson(c, n)[1])
        assert cover >= 0.95 - 1e-9


def test_two_stage_bootstrap_coverage_by_simulation() -> None:
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.2, 0.95, size=400)  # the "population" of tasks
    truth = true_p.mean()
    hits, trials = 0, 150
    for t in range(trials):
        tasks = rng.choice(true_p, size=20, replace=True)
        n = np.full(20, 8)
        c = rng.binomial(8, tasks)
        _, lo, hi = suite_bootstrap(n, c, "pass@1", 1, b=1000, seed=t)
        hits += lo <= truth <= hi
    assert 0.88 <= hits / trials <= 1.0  # nominal 0.95, two-stage is mildly conservative


# ------------------------------------------------------------------------ tests


def test_fisher_matches_scipy() -> None:
    for xc, nc, xb, nb in [(2, 8, 7, 8), (0, 8, 8, 8), (5, 8, 5, 8), (3, 10, 6, 9)]:
        ref = stats.fisher_exact([[xc, nc - xc], [xb, nb - xb]], alternative="less").pvalue
        assert fisher_less(xc, nc, xb, nb) == pytest.approx(ref)
        assert fisher_two_sided(xc, nc, xb, nb) == pytest.approx(stats.fisher_exact([[xc, nc - xc], [xb, nb - xb]]).pvalue)


def test_holm_known_case() -> None:
    assert holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_stratified_exact_reduces_to_exact_mcnemar_with_one_run() -> None:
    # 10 tasks, 1 run each: discordant pairs b (base pass, cand fail) = 6, c (base fail, cand pass) = 1
    xb = [1] * 6 + [0] * 1 + [1] * 2 + [0] * 1
    xc = [0] * 6 + [1] * 1 + [1] * 2 + [0] * 1
    res = stratified_exact_test(xc, [1] * 10, xb, [1] * 10)
    mcnemar_one_sided = stats.binomtest(1, 7, 0.5, alternative="less").pvalue
    assert res.p_less == pytest.approx(mcnemar_one_sided, abs=1e-9)


def test_stratified_exact_matches_monte_carlo_path() -> None:
    xb, xc = [8, 7, 6, 8, 5], [6, 5, 6, 7, 2]
    exact = stratified_exact_test(xc, [8] * 5, xb, [8] * 5)
    mc = stratified_exact_test(xc, [8] * 5, xb, [8] * 5, mc_samples=40000, force_mc=True)
    assert exact.method.startswith("exact") and mc.method.startswith("monte")
    assert mc.p_less == pytest.approx(exact.p_less, abs=0.004)
    assert mc.p_greater == pytest.approx(exact.p_greater, abs=0.004)


def test_stratified_test_false_positive_rate_is_controlled() -> None:
    rng = np.random.default_rng(7)
    rejections = 0
    trials = 400
    for _ in range(trials):
        p = rng.uniform(0.3, 1.0, size=20)
        xb, xc = rng.binomial(8, p), rng.binomial(8, p)
        rejections += stratified_exact_test(xc.tolist(), [8] * 20, xb.tolist(), [8] * 20).p_less < 0.05
    assert rejections / trials <= 0.05 + 0.03  # exact conditional test is at most nominal (+MC slack)


def test_paired_bootstrap_ci_contains_estimate() -> None:
    d, lo, hi = paired_bootstrap_diff([6, 5, 8], [8, 8, 8], [8, 7, 8], [8, 8, 8])
    assert lo <= d <= hi and d == pytest.approx((6 + 5 + 8 - 8 - 7 - 8) / 24)
