from pathlib import Path

from loguru import logger
import numpy as np
from scipy.special import beta
import typer

from skewed_sequences.config import PROCESSED_DATA_DIR, SEED, SEQUENCE_LENGTH

app = typer.Typer(pretty_exceptions_show_locals=False)


def safe_normalize(kernel):
    kernel_sum = np.sum(kernel)
    if kernel_sum == 0:
        raise ValueError("Kernel sum is zero, normalization failed.")
    return kernel / kernel_sum


def sine_kernel(size, period):
    x = np.linspace(0, 2 * np.pi * size / period, size)
    kernel = (np.sin(x) + 1) / 2
    return safe_normalize(kernel)


def cosine_kernel(size, period):
    x = np.linspace(0, 2 * np.pi * size / period, size)
    kernel = (np.cos(x) + 1) / 2
    return safe_normalize(kernel)


def gaussian_kernel(size, sigma):
    # The arange grid is only symmetric/centered for odd sizes; an even size
    # yields a half-sample phase shift (off-center kernel).
    assert size % 2 == 1, f"kernel_size must be odd for a centered kernel, got {size}"
    x = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return safe_normalize(kernel)


def combined_cosine_gaussian_kernel(size, sigma, period):
    cosine_k = cosine_kernel(size, period)
    gaussian_k = gaussian_kernel(size, sigma)
    return safe_normalize(cosine_k * gaussian_k)


def filter1d_with_kernel(data, kernel):
    # Renormalize by the partial kernel weight at each position so the
    # zero-padding implied by mode="same" does not pull the boundary samples
    # toward zero (a normalized kernel still loses mass where it overhangs the
    # signal edges). Interior weights are ~1, so only the edges are corrected.
    smoothed = np.convolve(data, kernel, mode="same")
    norm = np.convolve(np.ones_like(data), kernel, mode="same")
    return smoothed / norm


def smooth_sequence(sequence, smoothing_type, kernel_size, sigma, period):
    sequence = sequence.flatten()

    if smoothing_type == "gaussian":
        kernel = gaussian_kernel(kernel_size, sigma)
    elif smoothing_type == "sine":
        kernel = sine_kernel(kernel_size, period)
    elif smoothing_type == "cosine":
        kernel = cosine_kernel(kernel_size, period)
    elif smoothing_type == "combined_cosine_gaussian":
        kernel = combined_cosine_gaussian_kernel(kernel_size, sigma, period)
    else:
        raise ValueError(f"Unsupported smoothing type: {smoothing_type}")

    smoothed = filter1d_with_kernel(sequence, kernel)
    return smoothed.reshape(-1)


class SkewedGeneralizedT:
    """SGT distribution with q^p reparameterization.

    Every occurrence of the original tail-weight parameter is replaced by
    ``q ** p``, making the parameter behave more uniformly across *p*.
    """

    def __init__(self, mu=0.0, sigma=1.0, lam=0.0, p=2.0, q=2.0):
        assert -1 < lam < 1, "lam must be in (-1, 1)"
        assert sigma > 0, "sigma must be positive"
        assert p > 0 and q > 0, "p and q must be positive"

        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.p = p
        self.q = q
        self.qp = q**p  # q^p reparameterization

        # Compute beta functions using q^p
        B1 = beta(1.0 / p, self.qp)
        B2 = beta(2.0 / p, self.qp - 1.0 / p)
        B3 = beta(3.0 / p, self.qp - 2.0 / p)

        # Compute v and m (q^(-1/p) → q^(-1), q^(1/p) → q)
        self.v = q ** (-1.0) / np.sqrt((1 + 3 * lam**2) * (B3 / B1) - 4 * lam**2 * (B2 / B1) ** 2)
        self.m = lam * self.v * sigma * (2 * q * B2 / B1)

        # Normalizing constant (q^(1/p) → q)
        self.norm_const = p / (2 * self.v * sigma * q * B1)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = x - self.mu + self.m
        sgn_z = np.sign(z)
        denom = self.qp * (self.sigma * self.v) ** self.p * (1 + self.lam * sgn_z) ** self.p
        bracket = 1 + (np.abs(z) ** self.p) / denom
        return self.norm_const * bracket ** (-(1 / self.p + self.qp))

    def rvs(self, size=1):
        """Sample via a numerically-integrated inverse CDF.

        Inverse-transform sampling is robust for any ``(p, q, lam)``: unlike
        rejection sampling there is no proposal envelope to blow up. A symmetric
        Student-t proposal cannot envelope a heavy *skewed* tail (e.g. the
        ``normal-skewed`` config q=10, lam=0.9 needs an envelope ~1e40 and never
        terminates), so the previous rejection scheme is replaced here. Faithful
        to the grid resolution; always terminates.
        """
        # Centre the grid on the mode (z = 0 -> x = mu - m) and span it wide
        # enough to capture essentially all the mass (the tail beyond is clipped,
        # which only renormalises the CDF by a negligible amount).
        center = self.mu - self.m
        half_width = 1000.0 * self.sigma
        grid = np.linspace(center - half_width, center + half_width, 400_001)

        cdf = np.cumsum(self.pdf(grid))
        total = cdf[-1]
        if total <= 0:
            raise ValueError("SGT pdf integrated to zero over the sampling grid.")
        cdf = cdf / total  # monotone in [0, 1]

        u = np.random.uniform(0.0, 1.0, size=size)
        return np.interp(u, cdf, grid)

    def generate_sequences(self, n_sequences, sequence_length, n_features=1):
        total_samples = n_sequences * sequence_length * n_features
        samples = self.rvs(size=total_samples)
        return samples.reshape(n_sequences, sequence_length, n_features)


@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / "synthetic_dataset.npy",
    n_sequences: int = 10000,
    sequence_length: int = SEQUENCE_LENGTH,
    n_features: int = 1,
    mu: float = 0.0,
    sigma: float = 1.0,
    lam: float = 0.0,
    p: float = 2.0,
    q: float = 2.0,
    apply_smoothing: bool = True,
    smoothing_type: str = "combined_cosine_gaussian",
    kernel_size: int = 99,
    kernel_sigma: float = 10.0,
    period: float = 30.0,
    exp_transform: bool = False,
    exp_scale: float = 0.1,
    standardize: bool = True,
    seed: int = SEED,
):
    """Generate synthetic sequences from the SGT distribution.

    Args:
        exp_transform: If True, apply multiplicative (geometric) transform:
            data = exp(scale * data). Produces skewed, non-negative sequences
            analogous to geometric Brownian motion in finance.
        exp_scale: Scaling factor applied before exponentiation to prevent
            overflow. The data is standardized (zero-mean, unit-var) per
            sequence before scaling, so exp_scale directly controls the
            spread of the log-normal-like output.
    """
    np.random.seed(seed)
    logger.info("Generating synthetic sequences with SGT distribution...")
    logger.info(
        f"Params: mu={mu}, sigma={sigma}, lambda={lam}, p={p}, q={q}, smoothing={smoothing_type}"
    )

    sgt = SkewedGeneralizedT(mu=mu, sigma=sigma, lam=lam, p=p, q=q)
    raw = sgt.generate_sequences(n_sequences, sequence_length, n_features)

    dataset = np.zeros_like(raw)
    for i in range(n_sequences):
        for j in range(n_features):
            if apply_smoothing:
                dataset[i, :, j] = smooth_sequence(
                    raw[i, :, j], smoothing_type, kernel_size, kernel_sigma, period
                )
            else:
                dataset[i, :, j] = raw[i, :, j]

    if exp_transform:
        logger.info(f"Applying exponential (multiplicative) transform with scale={exp_scale}")
        for i in range(n_sequences):
            for j in range(n_features):
                seq = dataset[i, :, j]
                # Standardize before exp to control magnitude and prevent overflow
                std = np.std(seq)
                if std > 0:
                    seq = (seq - np.mean(seq)) / std
                else:
                    seq = seq - np.mean(seq)
                dataset[i, :, j] = np.exp(exp_scale * seq)
    elif standardize:
        # Per-sequence standardization (zero-mean, unit-variance), matching the
        # real-data loaders, so the loss operates on a unit residual scale
        # (sgt_loss_sigma=1.0, scaled robust thresholds). Diagnostic/plot callers
        # that want the raw generative distribution pass standardize=False.
        logger.info("Standardizing each sequence (zero-mean, unit-variance)")
        for i in range(n_sequences):
            for j in range(n_features):
                seq = dataset[i, :, j]
                std = np.std(seq)
                if std > 0:
                    dataset[i, :, j] = (seq - np.mean(seq)) / std
                else:
                    dataset[i, :, j] = seq - np.mean(seq)

    logger.info(f"Saving dataset to {output_path} with shape {dataset.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, dataset)

    logger.success("Dataset generation complete.")


if __name__ == "__main__":
    app()
