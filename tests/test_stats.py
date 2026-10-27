"""Tests for dsrlstudy.stats."""
import warnings

import numpy as np
import pytest

from dsrlstudy.stats import (
    bootstrap_ci, paired_bootstrap_diff, pooled_wilson, summarize_runs, wilson_interval,
)

SEEDS = [0.5, 0.6, 0.7, 0.55, 0.65, 0.6]


def test_wilson_known_value():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.404, abs=1e-3)
    assert hi == pytest.approx(0.596, abs=1e-3)


def test_wilson_not_degenerate_at_extremes():
    lo0, hi0 = wilson_interval(0, 10)
    lo1, hi1 = wilson_interval(10, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    assert hi1 == pytest.approx(1.0) and 0.0 < lo1 < 1.0


@pytest.mark.parametrize("s,n,c", [(-1, 10, 0.95), (11, 10, 0.95), (0, 0, 0.95),
                                   (1, 10, 0.0), (1, 10, 1.0), (1.5, 10, 0.95)])
def test_wilson_invalid(s, n, c):
    with pytest.raises(ValueError):
        wilson_interval(s, n, c)


def test_wilson_coverage():
    rng = np.random.default_rng(123)
    trials, true_p, draws = 50, 0.3, 2000
    covered = 0
    for k in rng.binomial(trials, true_p, size=draws):
        lo, hi = wilson_interval(int(k), trials)
        covered += lo <= true_p <= hi
    assert 0.93 <= covered / draws <= 0.975


def test_bootstrap_contains_point_and_deterministic():
    p, lo, hi = bootstrap_ci(SEEDS, seed=1)
    assert lo <= p <= hi
    assert p == pytest.approx(np.mean(SEEDS))
    assert bootstrap_ci(SEEDS, seed=1) == (p, lo, hi)
    spread = [0.1, 0.9, 0.3, 0.5, 0.7, 0.2, 0.8]
    assert bootstrap_ci(spread, seed=1, n_resamples=20) != bootstrap_ci(spread, seed=2, n_resamples=20)


def test_bootstrap_custom_statistic():
    p, lo, hi = bootstrap_ci(SEEDS, statistic=lambda x: float(np.max(x)), n_resamples=500)
    assert p == 0.7 and lo <= hi


def test_bootstrap_seed_count_rules():
    with pytest.raises(ValueError):
        bootstrap_ci([0.5])
    with pytest.warns(UserWarning):
        bootstrap_ci([0.5, 0.6, 0.7])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bootstrap_ci(SEEDS)


def test_bootstrap_invalid_input():
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, float("nan"), 0.3, 0.4, 0.5])
    with pytest.raises(ValueError):
        bootstrap_ci(SEEDS, confidence=1.5)
    with pytest.raises(ValueError):
        bootstrap_ci(SEEDS, n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_ci(np.ones((3, 3)))


def test_paired_diff():
    b = [0.4, 0.5, 0.6, 0.45, 0.55, 0.5]
    a = [x + 0.1 for x in b]
    d, lo, hi = paired_bootstrap_diff(a, b)
    assert d == pytest.approx(0.1)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)  # constant pairing -> no spread
    assert paired_bootstrap_diff(SEEDS, b) == paired_bootstrap_diff(SEEDS, b)


def test_paired_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 2, 3, 4, 5], [1, 2, 3, 4])


def test_summarize_runs_includes_per_seed():
    rows = summarize_runs({"A": SEEDS, "B": [x - 0.1 for x in SEEDS]})
    assert [r["method"] for r in rows] == ["A", "B"]
    for r in rows:
        assert r["n_seeds"] == 6 and len(r["per_seed"]) == 6
        assert r["ci_low"] <= r["point"] <= r["ci_high"]
    assert rows[0]["per_seed"] == SEEDS
    with pytest.raises(ValueError):
        summarize_runs({})


def test_pooled_wilson():
    rate, lo, hi = pooled_wilson([(30, 50), (20, 50)])
    assert rate == 0.5
    assert (lo, hi) == wilson_interval(50, 100)
    with pytest.raises(ValueError):
        pooled_wilson([])
    with pytest.raises(ValueError):
        pooled_wilson([(5, 4)])
