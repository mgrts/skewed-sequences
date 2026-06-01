"""Pin that the three SGT implementations stay numerically consistent.

The SGT math is (intentionally) reimplemented in three places with no shared
core (CLAUDE.md invariant #9):
  - modeling/loss_functions.py        (torch training loss)
  - data/synthetic/generate_data.py   (SkewedGeneralizedT.pdf, the generator)
  - visualization/visualize_losses.py (numpy plotting reimplementation)

A silent edit to one and not the others would not be caught by any other test.
These tests tie them together at the output level:
  SGTLoss(0, x) == -log( pdf(x) / norm_const )           (loss <-> generator)
  SGTLoss(0, x) == visualize_losses.sgt_loss(x)  (lam=0)  (loss <-> numpy viz)
"""

import numpy as np
import pytest
import torch

from skewed_sequences.data.synthetic.generate_data import SkewedGeneralizedT
from skewed_sequences.modeling.loss_functions import SGTLoss
from skewed_sequences.visualization.visualize_losses import sgt_loss as sgt_loss_np

# (p, q) pairs satisfying the validity domain q**p > 2/p.
PQ = [(2.0, 2.0), (2.0, 1.3), (1.5, 2.5), (1.0, 2.5), (2.0, 10.0), (1.5, 1.3)]
XS = [-2.0, -0.7, 0.3, 1.5]


def _sgtloss_value(crit, x):
    target = torch.tensor([[x]], dtype=torch.float64)
    return crit(torch.zeros_like(target), target).item()


@pytest.mark.parametrize("p,q", PQ)
@pytest.mark.parametrize("lam", [0.0, 0.3, -0.5])
def test_sgtloss_matches_generative_pdf(p, q, lam):
    sigma = 1.3
    sgt = SkewedGeneralizedT(mu=0.0, sigma=sigma, lam=lam, p=p, q=q)
    crit = SGTLoss(p=p, q=q, lam=lam, sigma=sigma, eps=1e-12)
    for x in XS:
        expected = -np.log(sgt.pdf(x) / sgt.norm_const)
        assert np.isclose(_sgtloss_value(crit, x), float(expected), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("p,q", PQ)
def test_sgtloss_matches_numpy_viz_loss_when_symmetric(p, q):
    s = 1.0
    crit = SGTLoss(p=p, q=q, lam=0.0, sigma=s, eps=1e-12)
    for x in XS:
        expected = float(sgt_loss_np(np.array([x]), p, q, s)[0])
        assert np.isclose(_sgtloss_value(crit, x), expected, rtol=1e-5, atol=1e-6)
