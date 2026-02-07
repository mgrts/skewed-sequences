from scipy.special import beta
import torch
import torch.nn as nn


class SGTLoss(nn.Module):
    def __init__(self, p=2.0, q=2.0, lam=0.0, sigma=1.0, eps=1e-6):
        super().__init__()
        self.p = p
        self.q = q
        self.lam = lam
        self.sigma = sigma  # sigma must be > 0
        self.eps = eps

    def forward(self, y, y_pred):
        p = self.p
        q = self.q
        lam = self.lam
        sigma = self.sigma
        eps = self.eps

        device = y.device
        dtype = y.dtype

        B1 = torch.tensor(beta(1.0 / p, q), dtype=dtype, device=device)
        B2 = torch.tensor(beta(2.0 / p, q - 1.0 / p), dtype=dtype, device=device)
        B3 = torch.tensor(beta(3.0 / p, q - 2.0 / p), dtype=dtype, device=device)

        v_numer = q ** (-1.0 / p)
        v_denom = torch.sqrt((1 + 3 * lam**2) * (B3 / B1) - 4 * lam**2 * (B2 / B1) ** 2)
        v = v_numer / (v_denom + eps)

        sigma_t = torch.tensor(sigma, dtype=dtype, device=device)

        m = lam * v * sigma_t * (2 * (q ** (1.0 / p)) * B2 / B1)

        diff = y - y_pred + m
        scaled = torch.abs(diff / (sigma_t * v)) ** p
        skew_term = (1 + lam * torch.sign(diff)) ** p

        ratio = scaled / (q * skew_term + eps)
        # Exponent: (1/p + q) per SGT PDF formula, i.e. 1/p plus q
        loss = (1.0 / p + q) * torch.log(1 + ratio + eps)

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
