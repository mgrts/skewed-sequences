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
