from loguru import logger
import numpy as np


def mean_abs_deviation(x: np.ndarray) -> float:
    return np.mean(np.abs(x - np.mean(x)))


def generate_n_sample(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x)
    sample_size = len(x)

    # Generate a (n, sample_size) matrix of resampled indices
    resampled_indices = np.random.randint(0, sample_size, size=(n, sample_size))
    resampled_samples = x[resampled_indices]

    # Sum across the n samples
    return np.sum(resampled_samples, axis=0)


def kappa(x: np.ndarray, n: int) -> float:
    """Single-shot kappa exponent: ``2 - log(n) / log(M_n / M_1)``.

    Thin wrapper over :func:`estimate_kappa_exponent` (the single source of the
    M_1 / M_n / K computation) that raises instead of returning NaN on degenerate
    (constant / non-increasing-MAD) inputs.
    """
    _, _, k_1n, m1, mn = estimate_kappa_exponent(np.asarray(x), n)
    if not np.isfinite(k_1n):
        raise ValueError(
            f"MAD of sample is degenerate (M_1={m1}, M_n={mn}); log ratio is undefined."
        )
    return k_1n


def estimate_kappa_exponent(X: np.ndarray, n: int):
    if n <= 1:
        raise ValueError("n must be greater than 1")

    S_1 = X
    S_n = generate_n_sample(X, n)

    M_1 = mean_abs_deviation(S_1)
    M_n = mean_abs_deviation(S_n)

    numerator = np.log(n)

    # Degenerate (constant / near-constant sub-series): a non-positive MAD makes
    # the log-ratio undefined, and M_n <= M_1 makes the denominator zero/negative
    # which turns K into +/-inf or finite garbage. Emit NaN for these instead so
    # they don't slip into the diagnostic CSV as real values (mirrors kappa()).
    if M_1 <= 0 or M_n <= 0 or M_n <= M_1:
        denominator = np.log(M_n / M_1) if (M_1 > 0 and M_n > 0) else np.nan
        return numerator, denominator, np.nan, M_1, M_n

    denominator = np.log(M_n / M_1)
    K_1n = 2 - (numerator / denominator)

    return numerator, denominator, K_1n, M_1, M_n


def compute_dispersion_scaling_series(X: np.ndarray, num_values: int = 100):
    metric_array = np.zeros((num_values, 5))
    # n=1 is mathematically undefined (division by zero in log ratio),
    # so the first row is always NaN.  Start computation from n=2.
    metric_array[0] = np.nan

    for i in range(1, num_values):
        try:
            values = estimate_kappa_exponent(X, i + 1)
            metric_array[i] = values
        except Exception as e:
            logger.warning(f"Failed to compute dispersion scaling at n={i + 1}: {e}")
            metric_array[i] = np.nan

    return metric_array


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)

    if std == 0:
        raise ValueError("Standard deviation is zero; skewness is undefined.")

    return np.mean((x - mean) ** 3) / (std**3)


def hill_estimator(x: np.ndarray, k: int | None = None) -> float:
    """Hill estimator of the tail index ``alpha`` from the upper-order statistics.

    Uses the ``k`` largest ``|x|`` values::

        gamma = (1/k) * sum_{i=1..k} ( log|x|_(i) - log|x|_(k+1) ),   alpha = 1/gamma

    where ``|x|_(1) >= ... >= |x|_(k+1)`` are the top order statistics. A *smaller*
    ``alpha`` means heavier tails (``alpha < 2`` implies infinite variance);
    ``alpha`` grows large for light/Gaussian tails. ``k`` defaults to
    ``floor(N/10)`` (a common rule of thumb); pass an explicit ``k`` to trade bias
    against variance.

    Used only for dataset characterization and the SGT ``q``-selection guideline
    (lower estimated ``alpha`` -> choose a smaller, more heavy-tail-tolerant
    ``q``); never inside the training loop.
    """
    a = np.sort(np.abs(np.asarray(x, dtype=float)))[::-1]
    a = a[a > 0]
    n = len(a)
    if n < 3:
        raise ValueError("need at least 3 nonzero values for the Hill estimator")
    if k is None:
        k = max(1, n // 10)
    if not (1 <= k < n):
        raise ValueError(f"k must satisfy 1 <= k < n_nonzero ({n}), got {k}")

    threshold = a[k]  # the (k+1)-th largest |x|
    if threshold <= 0:
        raise ValueError("the (k+1)-th order statistic is non-positive; alpha undefined")

    gamma = float(np.mean(np.log(a[:k]) - np.log(threshold)))
    if gamma <= 0:
        raise ValueError("non-positive Hill gamma; tail index undefined")

    return 1.0 / gamma


# ---------------------------------------------------------------------------
# SGT maximum-likelihood fit of the skew / tail parameters (A3 / A11 guidance)
# ---------------------------------------------------------------------------


def excess_kurtosis(x: np.ndarray) -> float:
    """Sample excess kurtosis (``ddof=1`` variance, matching :func:`skewness`)."""
    x = np.asarray(x, dtype=float)
    var = np.var(x, ddof=1)
    if var == 0:
        raise ValueError("Variance is zero; kurtosis is undefined.")
    return float(np.mean((x - np.mean(x)) ** 4) / var**2 - 3.0)


def sgt_min_q(p: float, margin: float = 1.05) -> float:
    """Smallest admissible ``q`` (q^p reparameterization): ``q**p > 2/p``, with a margin."""
    return margin * (2.0 / p) ** (1.0 / p)


def fit_sgt(
    x: np.ndarray,
    p: float = 2.0,
    sigma: float | None = None,
    fix_lambda: float | None = None,
    q_max: float = 50.0,
    max_samples: int = 100_000,
    seed: int = 0,
    extra_starts=(),
) -> dict:
    """Maximum-likelihood ``(mu, lambda, q)`` of the SGT on a 1-D sample.

    ``p`` and ``sigma`` are held FIXED — ``sigma`` defaults to the MAD-based robust
    scale of ``x``, which is exactly how ``train.get_loss_function`` scales the SGT
    loss (``sigma = residual_scale``). The fit therefore answers "which skew and
    tail parameters would the SGT *loss as trained* prefer for these residuals",
    not the unconstrained best-fitting SGT. Multi-start L-BFGS-B over
    ``(mu, lambda, log q)``; ``fix_lambda`` pins the skew (e.g. ``0.0`` for the
    symmetric reference fit). Large samples are subsampled deterministically.
    ``extra_starts`` adds start points ``(mu, lambda, log q)`` (or ``(mu, log q)`` when
    ``fix_lambda`` is set). The result flags ``lam_at_bound`` / ``q_at_bound``: a
    pinned estimate means the data are more skewed / heavier-tailed than the SGT can
    express at this fixed ``sigma`` (or, for ``q`` at ``q_max``, that the likelihood is
    flat in ``q`` — the Gaussian limit, where ``q`` is unidentified).
    """
    from scipy.optimize import minimize

    from skewed_sequences.data.synthetic.generate_data import SkewedGeneralizedT

    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 10:
        raise ValueError("need at least 10 finite values to fit the SGT")
    if len(x) > max_samples:
        x = np.random.default_rng(seed).choice(x, size=max_samples, replace=False)

    if sigma is None:
        sigma = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    if not sigma > 0:
        raise ValueError("sigma must be positive (degenerate sample: zero MAD)")

    q_lo = sgt_min_q(p)
    if q_max <= q_lo:
        raise ValueError(f"q_max={q_max} must exceed the validity floor {q_lo:.3f}")
    center = float(np.median(x))
    mu_bounds = (center - 5.0 * sigma, center + 5.0 * sigma)
    log_q_bounds = (np.log(q_lo), np.log(q_max))

    def nll(mu, lam, log_q):
        dist = SkewedGeneralizedT(mu=mu, sigma=sigma, lam=lam, p=p, q=float(np.exp(log_q)))
        value = -float(np.mean(dist.logpdf(x)))
        return value if np.isfinite(value) else 1e12

    q_starts = (2.0, 8.0)
    best = None
    if fix_lambda is None:

        def objective(theta):
            return nll(theta[0], theta[1], theta[2])

        bounds = [mu_bounds, (-0.95, 0.95), log_q_bounds]
        starts = [
            [center, lam0, np.log(np.clip(q0, q_lo, q_max))]
            for lam0 in (-0.4, 0.0, 0.4)
            for q0 in q_starts
        ] + [list(s) for s in extra_starts]
    else:

        def objective(theta):
            return nll(theta[0], fix_lambda, theta[1])

        bounds = [mu_bounds, log_q_bounds]
        starts = [[center, np.log(np.clip(q0, q_lo, q_max))] for q0 in q_starts] + [
            list(s) for s in extra_starts
        ]

    for x0 in starts:
        result = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result

    if best is None or not np.isfinite(best.fun) or best.fun >= 1e12:
        raise RuntimeError("SGT fit did not converge from any start point")
    if fix_lambda is None:
        mu, lam, log_q = best.x
    else:
        (mu, log_q), lam = best.x, fix_lambda
    q = float(np.exp(log_q))
    tol = 1e-6
    return {
        "mu": float(mu),
        "lam": float(lam),
        "q": q,
        "p": float(p),
        "sigma": float(sigma),
        "nll": float(best.fun),
        "n": int(len(x)),
        "lam_at_bound": bool(fix_lambda is None and abs(lam) >= 0.95 - tol),
        "q_at_bound": bool(q >= q_max * (1 - tol) or q <= q_lo * (1 + tol)),
    }


def sgt_increment_fit(data: np.ndarray, p: float = 2.0, max_samples: int = 100_000) -> dict:
    """Fit ``(lambda, q)`` to the one-step increments of ``(N, T, 1)`` data.

    The increments are what a one-step forecaster's residuals look like before it
    learns anything (persistence residuals), and their MAD-based scale is the
    ``residual_scale`` the training loss is built with — so the fitted ``lambda``
    is the data-driven skew setting for ``SGTLoss`` and the fitted ``q`` its tail
    setting. ``delta_nll = nll - nll_symmetric`` (nats / sample): negative means the
    skewed fit beats the ``lambda = 0`` fit, i.e. the data support an asymmetric loss
    at this horizon. Note that smoothing / aggregation can erase the marginal skew
    at the one-step horizon (the ``normal-skewed`` synthetic data: marginal skew
    0.18, increment skew 0.04), which is exactly what this diagnostic exposes.

    The fitted ``q`` is the tail setting *for the loss at this fixed sigma*, not a
    tail-index estimate of the data — compare against :func:`hill_estimator` for that.
    The symmetric fit is run first and seeds the skewed fit, so ``delta_nll <= 0`` by
    construction (up to optimizer tolerance).
    """
    increments = np.diff(np.asarray(data, dtype=float)[:, :, 0], axis=1).ravel()
    sigma = 1.4826 * float(np.median(np.abs(increments - np.median(increments))))
    if not sigma > 0:
        raise ValueError(
            "Increments have zero MAD (piecewise-constant data); the residual scale is "
            "degenerate and no SGT fit is meaningful."
        )
    symmetric = fit_sgt(increments, p=p, sigma=sigma, fix_lambda=0.0, max_samples=max_samples)
    full = fit_sgt(
        increments,
        p=p,
        sigma=sigma,
        max_samples=max_samples,
        extra_starts=[(symmetric["mu"], 0.0, np.log(symmetric["q"]))],
    )
    return {
        "p": float(p),
        "sigma": sigma,
        "n_increments": int(len(increments)),
        "increment_skewness": float(skewness(increments)),
        "increment_excess_kurtosis": excess_kurtosis(increments),
        "lam": full["lam"],
        "q": full["q"],
        "mu": full["mu"],
        "nll": full["nll"],
        "q_symmetric": symmetric["q"],
        "nll_symmetric": symmetric["nll"],
        "delta_nll": full["nll"] - symmetric["nll"],
        "lam_at_bound": full["lam_at_bound"],
        "q_at_bound": full["q_at_bound"],
    }
