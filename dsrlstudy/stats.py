"""Statistics helpers for seed-level reporting of RL results.

Design: the unit of replication is the *training seed*, not the episode.
Primary report = percentile bootstrap over per-seed values (``bootstrap_ci``),
always alongside the raw per-seed values (``summarize_runs``). Wilson
intervals are for binomial success counts; ``pooled_wilson`` is secondary.

Depends only on numpy and scipy.
"""
from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
from scipy import stats as scipy_stats

CONFIG = {
    # 0.95 is the conventional reporting level; keeps intervals comparable to the literature.
    "confidence": 0.95,
    # 10k resamples gives percentile endpoints with Monte-Carlo error well below
    # the sampling error from a handful of seeds, and runs in milliseconds.
    "n_resamples": 10_000,
    # Fixed default seed so reported CIs are exactly reproducible.
    "bootstrap_seed": 0,
    # Below 2 seeds a bootstrap is meaningless (zero spread); below 5 it is
    # badly anti-conservative, so we warn.
    "min_seeds_error": 2,
    "min_seeds_warn": 5,
}

Statistic = Callable[[np.ndarray], float]


def _check_confidence(confidence: float) -> None:
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def _check_resamples(n_resamples: int) -> None:
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError(f"n_resamples must be a positive int, got {n_resamples}")


def _as_values(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _check_n_seeds(n_seeds: int) -> None:
    if n_seeds < CONFIG["min_seeds_error"]:
        raise ValueError(f"need at least {CONFIG['min_seeds_error']} seeds, got {n_seeds}")
    if n_seeds < CONFIG["min_seeds_warn"]:
        warnings.warn(
            f"only {n_seeds} seeds (<{CONFIG['min_seeds_warn']}): bootstrap CI will be "
            "too narrow; treat as descriptive and show per-seed values.",
            stacklevel=3,
        )


def _apply_statistic(statistic: Statistic, samples: np.ndarray) -> np.ndarray:
    """Apply ``statistic`` to each row of a 2-D array (vectorised for mean/median)."""
    if statistic is np.mean:
        return samples.mean(axis=1)
    if statistic is np.median:
        return np.median(samples, axis=1)
    return np.array([statistic(row) for row in samples], dtype=float)


def wilson_interval(successes: int, trials: int, confidence: float = CONFIG["confidence"]) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike the Wald interval it is non-degenerate at 0/n and n/n.
    Raises ValueError unless 0 <= successes <= trials, trials >= 1, 0 < confidence < 1.
    """
    if int(successes) != successes or int(trials) != trials:
        raise ValueError("successes and trials must be integers")
    successes, trials = int(successes), int(trials)
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if not (0 <= successes <= trials):
        raise ValueError(f"need 0 <= successes <= trials, got {successes}/{trials}")
    _check_confidence(confidence)
    z = float(scipy_stats.norm.ppf(0.5 + confidence / 2.0))
    p_hat = successes / trials
    z2n = z * z / trials
    center = (p_hat + z2n / 2.0) / (1.0 + z2n)
    half = z * np.sqrt(p_hat * (1.0 - p_hat) / trials + z * z / (4.0 * trials**2)) / (1.0 + z2n)
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def bootstrap_ci(
    per_seed_values: Sequence[float],
    statistic: Statistic = np.mean,
    n_resamples: int = CONFIG["n_resamples"],
    confidence: float = CONFIG["confidence"],
    seed: int = CONFIG["bootstrap_seed"],
) -> tuple[float, float, float]:
    """Percentile bootstrap CI, resampling *seeds* (not episodes) with replacement.

    Returns (point, low, high) where point = statistic(per_seed_values).
    Raises ValueError for <2 seeds or non-finite input; warns for <5 seeds.
    """
    values = _as_values(per_seed_values, "per_seed_values")
    _check_confidence(confidence)
    _check_resamples(n_resamples)
    _check_n_seeds(len(values))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    stats_boot = _apply_statistic(statistic, values[idx])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(stats_boot, [alpha, 1.0 - alpha])
    return float(statistic(values)), float(low), float(high)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    statistic: Statistic = np.mean,
    n_resamples: int = CONFIG["n_resamples"],
    confidence: float = CONFIG["confidence"],
    seed: int = CONFIG["bootstrap_seed"],
) -> tuple[float, float, float]:
    """Paired bootstrap of statistic(a) - statistic(b); a[i], b[i] share seed i.

    The same resampled seed indices are used for both methods, preserving the
    pairing. Returns (diff, low, high). Raises if lengths differ or <2 seeds.
    """
    a_arr, b_arr = _as_values(a, "a"), _as_values(b, "b")
    if len(a_arr) != len(b_arr):
        raise ValueError(f"a and b must have equal length, got {len(a_arr)} and {len(b_arr)}")
    _check_confidence(confidence)
    _check_resamples(n_resamples)
    _check_n_seeds(len(a_arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a_arr), size=(n_resamples, len(a_arr)))
    diffs = _apply_statistic(statistic, a_arr[idx]) - _apply_statistic(statistic, b_arr[idx])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(diffs, [alpha, 1.0 - alpha])
    return float(statistic(a_arr) - statistic(b_arr)), float(low), float(high)


def summarize_runs(
    per_seed: dict[str, list[float]],
    statistic: Statistic = np.mean,
    n_resamples: int = CONFIG["n_resamples"],
    confidence: float = CONFIG["confidence"],
    seed: int = CONFIG["bootstrap_seed"],
) -> list[dict]:
    """One row per method: method, n_seeds, point, ci_low, ci_high, per_seed.

    Per-seed values are always included; never report mean +- std alone.
    """
    if not per_seed:
        raise ValueError("per_seed is empty")
    rows = []
    for method, values in per_seed.items():
        point, low, high = bootstrap_ci(values, statistic, n_resamples, confidence, seed)
        rows.append({
            "method": method,
            "n_seeds": len(values),
            "point": point,
            "ci_low": low,
            "ci_high": high,
            "per_seed": [float(v) for v in values],
        })
    return rows


def pooled_wilson(
    per_seed_successes: list[tuple[int, int]], confidence: float = CONFIG["confidence"]
) -> tuple[float, float, float]:
    """Wilson interval on success counts pooled across seeds -> (rate, low, high).

    NOTE: pooling treats all episodes as i.i.d. and ignores between-seed
    variance, so the interval is typically too narrow. The primary report is
    ``bootstrap_ci`` over per-seed success rates; use this only as a secondary
    number.
    """
    if not per_seed_successes:
        raise ValueError("per_seed_successes is empty")
    total_s = total_n = 0
    for pair in per_seed_successes:
        if len(pair) != 2:
            raise ValueError(f"each entry must be (successes, trials), got {pair!r}")
        s, n = pair
        wilson_interval(s, n, confidence)  # validates the pair
        total_s += int(s)
        total_n += int(n)
    low, high = wilson_interval(total_s, total_n, confidence)
    return total_s / total_n, low, high


if __name__ == "__main__":
    print("wilson 0/10   :", wilson_interval(0, 10))
    print("wilson 10/10  :", wilson_interval(10, 10))
    lo, hi = wilson_interval(50, 100)
    print("wilson 50/100 :", (lo, hi))
    assert abs(lo - 0.404) < 1e-3 and abs(hi - 0.596) < 1e-3
    assert wilson_interval(0, 10)[1] > 0 and wilson_interval(10, 10)[0] < 1
    seeds_a = [0.62, 0.55, 0.71, 0.60, 0.66, 0.58]
    seeds_b = [0.50, 0.49, 0.60, 0.52, 0.57, 0.48]
    print("bootstrap A   :", bootstrap_ci(seeds_a))
    print("paired A-B    :", paired_bootstrap_diff(seeds_a, seeds_b))
    for row in summarize_runs({"A": seeds_a, "B": seeds_b}):
        print(row)
    print("pooled wilson :", pooled_wilson([(30, 50), (25, 50), (35, 50)]))
    print("sanity checks passed")
