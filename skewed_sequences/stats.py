"""Distribution-free statistics for the replicate comparisons.

Every interval and p-value the paper reports comes from the three functions here, so no
normal or Student-t assumption enters the reported numbers:

* ``bootstrap_ci`` — bias-corrected and accelerated (BCa) bootstrap interval of the mean
  over the replicates (``N_BOOT`` resamples, fixed seed), used both for a single loss and
  for a vector of seed-paired differences.
* ``signflip_p`` — the exact two-sided sign-flip (permutation) test of a zero mean paired
  difference: every one of the ``2**n`` sign assignments of the differences is enumerated
  (1,024 at ``n = 10``), so the p-value is exact and its floor is ``2 / 2**n``.
* ``holm`` — Holm's step-down adjustment of a family of p-values.

The seed-pairing of the replicates (one ``random_state`` per run index shared by every
loss) is what makes the sign-flip test valid: under the null hypothesis the sign of each
paired difference is a coin toss.
"""

import warnings

import numpy as np
from scipy.stats import bootstrap as _bootstrap

N_BOOT = 10_000
BOOT_SEED = 20261016
# Enumerate every sign pattern up to this n (2**16 patterns); Monte Carlo beyond.
MAX_EXACT_N = 16
N_MONTE_CARLO = 100_000


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def bootstrap_ci(
    x, n_boot: int = N_BOOT, seed: int = BOOT_SEED, confidence_level: float = 0.95
) -> tuple[float, float]:
    """BCa bootstrap interval ``(low, high)`` of the mean of ``x``.

    Returns ``(nan, nan)`` for fewer than two values and ``(mean, mean)`` when every value
    is the same (the interval is then a point and the BCa acceleration is undefined). The
    resampling uses a fixed seed so the published numbers are reproducible.
    """
    x = _clean(x)
    if x.size < 2:
        return np.nan, np.nan
    if np.ptp(x) == 0:
        m = float(x.mean())
        return m, m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _bootstrap(
            (x,),
            np.mean,
            axis=-1,
            vectorized=True,
            confidence_level=confidence_level,
            n_resamples=n_boot,
            method="BCa",
            rng=np.random.default_rng(seed),
        )
        low, high = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if not (np.isfinite(low) and np.isfinite(high)):  # BCa undefined: percentile
            res = _bootstrap(
                (x,),
                np.mean,
                axis=-1,
                vectorized=True,
                confidence_level=confidence_level,
                n_resamples=n_boot,
                method="percentile",
                rng=np.random.default_rng(seed),
            )
            low, high = float(res.confidence_interval.low), float(res.confidence_interval.high)
    return low, high


def signflip_p(
    diff, max_exact_n: int = MAX_EXACT_N, n_mc: int = N_MONTE_CARLO, seed: int = BOOT_SEED
) -> float:
    """Two-sided sign-flip permutation p-value for a zero mean of the paired differences.

    Exact (all ``2**n`` sign patterns) for ``n <= max_exact_n``, Monte Carlo with a fixed
    seed beyond that. ``nan`` for fewer than two differences or when they are all zero.
    """
    d = _clean(diff)
    n = d.size
    if n < 2 or np.all(d == 0):
        return np.nan
    observed = abs(d.mean())
    if n <= max_exact_n:
        patterns = np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n, dtype=np.int64)
        signs = (patterns & 1) * 2 - 1  # (2**n, n) of +-1, every pattern once
    else:
        signs = np.random.default_rng(seed).choice((-1, 1), size=(n_mc, n))
    means = np.abs(signs @ d) / n
    # ">=" with a relative tolerance so the identity pattern always counts: the matrix
    # product reproduces ``observed`` only up to rounding, at any scale of ``d``.
    return float(np.mean(means >= observed * (1 - 1e-9)))


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as the input; NaN stays NaN).

    The family size is the number of finite p-values. Adjusted values are monotone in the
    raw ones and capped at 1.
    """
    p = np.asarray(pvals, dtype=float).ravel()
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    if m == 0:
        return out
    order = finite[np.argsort(p[finite], kind="stable")]
    adjusted = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adjusted)
    return out
