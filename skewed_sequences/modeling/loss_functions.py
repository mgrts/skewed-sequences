import torch
import torch.nn as nn
from scipy.special import beta


class SGTLoss(nn.Module):
    """
    Scaled Generalized T-Distribution Loss (SGT Loss).

    Args:
        p (float): Tail heaviness parameter.
        q (float): Peakedness parameter.
        lambda_ (float): Skewness parameter.
        sigma (float): Scale parameter.
    """
    def __init__(self, p: float = 2.0, q: float = 2.0, lambda_: float = 0.0, sigma: float = 1.0) -> None:
        super().__init__()
        self.p = p
        self.q = q
        self.lambda_ = lambda_
        self.sigma = max(sigma, 1e-6)  # Clamp for numerical stability
        self.eps = 1e-6

    @staticmethod
    def beta_function(a: float, b: float) -> float:
        return beta(a, b)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the SGT loss.

        Args:
            y_pred (Tensor): Predicted output [B, ...]
            y_true (Tensor): Ground truth [B, ...]

        Returns:
            Tensor: Scalar loss
        """
        p, q, lambda_, sigma = self.p, self.q, self.lambda_, self.sigma

        B1 = self.beta_function(1 / p, q)
        B2 = self.beta_function(2 / p, q - 1 / p)
        B3 = self.beta_function(3 / p, q - 2 / p)

        # Scale parameter v
        numerator = q ** (-1 / p)
        denom_scalar = (1 + 3 * lambda_ ** 2) * (B3 / B1) - 4 * lambda_ ** 2 * (B2 ** 2) / (B1 ** 2)
        v = numerator / (denom_scalar ** 0.5)

        # Shift parameter m
        m = lambda_ * v * sigma * 2 * q ** (1 / p) * B2 / B1

        # Standardized residual
        z = (y_true - y_pred + m) / (sigma * v)

        # Final loss
        base = torch.abs(z) ** p / (q * (1 + lambda_ * torch.sign(z)) ** p)
        loss = torch.log(1 + base + self.eps).mean()

        return loss
