from pathlib import Path
import numpy as np
import typer
from loguru import logger
from scipy.special import beta as beta_func

from skewed_sequences.config import SEED, SEQUENCE_LENGTH, PROCESSED_DATA_DIR

app = typer.Typer(pretty_exceptions_show_locals=False)


def safe_normalize(kernel):
    kernel_sum = np.sum(kernel)
    if kernel_sum == 0:
        raise ValueError('Kernel sum is zero, normalization failed.')
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
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return safe_normalize(kernel)


def combined_cosine_gaussian_kernel(size, sigma, period):
    cosine_k = cosine_kernel(size, period)
    gaussian_k = gaussian_kernel(size, sigma)
    return safe_normalize(cosine_k * gaussian_k)


def filter1d_with_kernel(data, kernel):
    return np.convolve(data, kernel, mode='same')


def smooth_sequence(sequence, smoothing_type, kernel_size, sigma, period):
    sequence = sequence.flatten()

    if smoothing_type == 'gaussian':
        kernel = gaussian_kernel(kernel_size, sigma)
    elif smoothing_type == 'sine':
        kernel = sine_kernel(kernel_size, period)
    elif smoothing_type == 'cosine':
        kernel = cosine_kernel(kernel_size, period)
    elif smoothing_type == 'combined_cosine_gaussian':
        kernel = combined_cosine_gaussian_kernel(kernel_size, sigma, period)
    else:
        raise ValueError(f'Unsupported smoothing type: {smoothing_type}')

    smoothed = filter1d_with_kernel(sequence, kernel)
    return smoothed.reshape(-1)


def sgt_pdf(x, mu, sigma, lambda_, p, q):
    B1 = beta_func(1 / p, q)
    B2 = beta_func(2 / p, q - 1 / p)
    B3 = beta_func(3 / p, q - 2 / p)

    expr = (1 + 3 * lambda_**2) * (B3 / B1) - 4 * lambda_**2 * (B2**2) / (B1**2)
    if expr <= 1e-8:
        raise ValueError(f'Invalid parameters: sqrt(expr) ≤ 0 (expr = {expr:.6f})')

    v = q**(-1 / p) / np.sqrt(expr)
    m = 2 * lambda_ * v * sigma * q**(1 / p) * B2 / B1

    z = (x - mu + m) / (v * sigma)
    scale_term = p / (2 * v * sigma * q**(1 / p) * B1)
    denom = (np.abs(z) ** p / (q * (1 + lambda_ * np.sign(z)) ** p) + 1) ** (1 / p + q)
    return scale_term / denom


def sample_sgt(n, mu, sigma, lambda_, p, q):
    df = 2 * q
    oversample = n * 10
    proposal = np.random.standard_t(df, size=oversample) * sigma + mu

    pdf_vals = sgt_pdf(proposal, mu, sigma, lambda_, p, q)
    max_pdf = np.max(pdf_vals)
    u = np.random.uniform(0, max_pdf, size=oversample)

    accepted = proposal[u < pdf_vals]
    if len(accepted) < n:
        raise RuntimeError('Sampling failed. Try adjusting lambda, p, q or increase oversample rate.')

    return accepted[:n]


def generate_sgt_sequence(length: int, mu: float, sigma: float, lambda_: float, p: float, q: float) -> np.ndarray:
    assert -1 < lambda_ < 1, 'Lambda must be in (-1, 1)'
    assert p > 0 and q > 0, 'p and q must be positive'
    assert sigma > 0, 'sigma must be positive'
    return sample_sgt(length, mu, sigma, lambda_, p, q)


@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / 'synthetic_dataset.npy',
    n_sequences: int = 1000,
    sequence_length: int = SEQUENCE_LENGTH,
    n_features: int = 1,
    mu: float = 0.0,
    sigma: float = 1.0,
    lambda_: float = 0.99,
    p: float = 2.0,
    q: float = 2.0,
    smoothing_type: str = 'combined_cosine_gaussian',
    kernel_size: int = 99,
    kernel_sigma: float = 10.0,
    period: float = 30.0,
    seed: int = SEED,
):
    np.random.seed(seed)
    logger.info('Generating synthetic sequences with SGT distribution...')
    logger.info(f'Params: mu={mu}, sigma={sigma}, lambda={lambda_}, p={p}, q={q}, smoothing={smoothing_type}')

    dataset = []

    for _ in range(n_sequences):
        sequence = []
        for _ in range(n_features):
            raw = generate_sgt_sequence(sequence_length, mu, sigma, lambda_, p, q)
            smoothed = smooth_sequence(raw, smoothing_type, kernel_size, kernel_sigma, period)
            sequence.append(smoothed)
        sequence = np.stack(sequence, axis=-1)
        dataset.append(sequence)

    dataset = np.stack(dataset)
    
    logger.info(f'Saving dataset to {output_path} with shape {dataset.shape}')
    
    np.save(output_path, dataset)
    
    logger.success('Dataset generation complete.')


if __name__ == '__main__':
    app()
