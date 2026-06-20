from scipy.special import beta
import torch
import torch.nn as nn


class SGTLoss(nn.Module):
    def __init__(self, p=2.0, q=2.0, lam=0.0, sigma=1.0, eps=1e-6):
        super().__init__()
        # SGT validity: q**p > 2/p is required for B3 = beta(3/p, q**p - 2/p) to
        # be defined. Outside this domain beta returns a negative value or +inf,
        # so v_denom = sqrt(...) silently becomes NaN and the loss is all-NaN/inf
        # with no exception. Fail fast at construction instead. (All shipped
        # TRAINING_CONFIGS pairs satisfy this.)
        assert (
            q**p > 2.0 / p
        ), f"SGT validity requires q**p > 2/p, got q**p={q ** p:.4f}, 2/p={2.0 / p:.4f}"
        self.p = p
        self.q = q
        self.lam = lam
        self.sigma = sigma  # sigma must be > 0
        self.eps = eps

    def forward(self, input, target):
        """Compute the SGT negative log-likelihood loss.

        Uses the q^p reparameterization: every occurrence of the original
        tail-weight parameter is replaced by ``q ** p``, which makes the
        parameter behave more uniformly across different *p* values.

        Args:
            input: Model predictions.
            target: Ground-truth targets.

        The residual follows the SGT PDF convention:
        ``z = target - input + m`` where *m* is the mean-correction term.
        """
        p = self.p
        q = self.q
        lam = self.lam
        sigma = self.sigma
        eps = self.eps

        device = input.device
        dtype = input.dtype

        qp = q**p  # q^p reparameterization

        B1 = torch.tensor(beta(1.0 / p, qp), dtype=dtype, device=device)
        B2 = torch.tensor(beta(2.0 / p, qp - 1.0 / p), dtype=dtype, device=device)
        B3 = torch.tensor(beta(3.0 / p, qp - 2.0 / p), dtype=dtype, device=device)

        v_numer = q ** (-1.0)
        v_denom = torch.sqrt((1 + 3 * lam**2) * (B3 / B1) - 4 * lam**2 * (B2 / B1) ** 2)
        v = v_numer / (v_denom + eps)

        sigma_t = torch.tensor(sigma, dtype=dtype, device=device)

        m = lam * v * sigma_t * (2 * q * B2 / B1)

        # Residual: target - prediction + m  (SGT PDF convention)
        diff = target - input + m
        scaled = torch.abs(diff / (sigma_t * v)) ** p
        skew_term = (1 + lam * torch.sign(diff)) ** p

        ratio = scaled / (qp * skew_term + eps)
        loss = (1.0 / p + qp) * torch.log(1 + ratio + eps)

        return loss.mean()


class CauchyLoss(nn.Module):
    def __init__(self, gamma=1.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        diffs = input - target
        loss = self.gamma * torch.log(1 + (diffs**2) / self.gamma)

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction="mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, input, target):
        diff = input - target
        abs_diff = torch.abs(diff)

        loss = torch.where(
            abs_diff <= self.delta, 0.5 * diff**2, self.delta * (abs_diff - 0.5 * self.delta)
        )

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss


class TukeyLoss(nn.Module):
    def __init__(self, c=4.685, reduction="mean"):
        super().__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, input, target):
        diff = input - target
        abs_diff = torch.abs(diff)

        mask = abs_diff <= self.c
        r = diff / self.c

        loss = torch.zeros_like(diff)
        loss[mask] = (self.c**2 / 6) * (1 - (1 - r[mask] ** 2) ** 3)
        loss[~mask] = self.c**2 / 6

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier (pseudo-Huber) loss: a smooth, everywhere-differentiable L1.

    ``L = sqrt(diff**2 + eps**2) - eps``. Quadratic for ``|diff| << eps`` (gradient
    ``diff/eps``) and linear/MAE-like with bounded unit gradient for
    ``|diff| >> eps``, with no kink at the origin (unlike Huber/MAE) and ``L(0)=0``
    so its logged value is comparable to the other baselines. The transition scale
    ``eps`` is set from the robust residual scale by the loss factory (mirroring
    Huber's ``delta`` / Tukey's ``c``) so it brackets the residual bulk instead of
    collapsing to MSE at this data scale.
    """

    def __init__(self, eps=1.0, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        diff = input - target
        # Subtract eps so L(0)=0 (the -eps offset is constant, gradient unchanged).
        loss = torch.sqrt(diff**2 + self.eps**2) - self.eps

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss
