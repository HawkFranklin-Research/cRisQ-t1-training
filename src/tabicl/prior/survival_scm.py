from __future__ import annotations
import math, random, numpy as np, torch
from torch import nn, Tensor
from .utils import XSampler

class SurvivalSCM(nn.Module):
    """
    Generates synthetic (X, time, event) triples for curriculum-style
    survival pre-training.

    Difficulty knobs (annealed across Stage 1-3):
        • latent_dim
        • non_prop_hazard_prob      # crossing or time-varying effects
        • competing_risk_prob       # δ flipped to 0 for competing events
    """

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 30,
        latent_dim: int = 4,
        non_prop_hazard_prob: float = 0.0,
        competing_risk_prob: float = 0.0,
        censor_rate: float = 0.25,
        device: str = "cpu",
        **_
    ):
        super().__init__()
        self.seq_len, self.num_features = seq_len, num_features
        self.latent_dim = latent_dim
        self.non_prop_hazard_prob = non_prop_hazard_prob
        self.competing_risk_prob = competing_risk_prob
        self.censor_rate = censor_rate
        self.device = device

        self.x_sampler = XSampler(seq_len, num_features, sampling="mixed", device=device)
        self.beta = torch.randn(latent_dim, device=device) / math.sqrt(latent_dim)

    # --------------------------------------------------------------------- #
    # internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _weibull_time(self, eta: Tensor) -> Tensor:
        """Weibull PH inverse-CDF sampling with optional time-varying effects."""
        u = torch.rand_like(eta).clamp_(1e-6, 1 - 1e-6)
        k = torch.empty_like(eta).uniform_(1.0, 2.5)   # shape
        lam = torch.exp(eta)                           # scale

        t = (-torch.log(u) / lam).pow(1 / k)

        if random.random() < self.non_prop_hazard_prob:
            # flip effect after τ  (simple crossing-hazard example)
            tau = torch.rand_like(t) * 0.6 + 0.2
            flip = t > tau
            eta2 = -eta
            lam2 = torch.exp(eta2)
            t_alt = (-torch.log(u) / lam2).pow(1 / k)
            t = torch.where(flip, t_alt, t)

        return t

    # --------------------------------------------------------------------- #
    # forward                                                               #
    # --------------------------------------------------------------------- #
    def forward(self) -> tuple[Tensor, Tensor]:
        X = self.x_sampler.sample()                        # (T, H)
        Z = torch.randn(self.seq_len, self.latent_dim, device=self.device)
        lin_pred = (X[:, :self.latent_dim] * self.beta).sum(-1) + (Z * self.beta).sum(-1)

        time = self._weibull_time(lin_pred)
        event = torch.ones_like(time)

        if random.random() < self.competing_risk_prob:
            event_mask = torch.rand_like(event) < 0.3
            event[event_mask] = 0.0                        # competing event

        # random right-censoring
        c = torch.distributions.Exponential(1 / time.mean()).sample(time.shape).to(self.device)
        censored = c < time
        time = torch.where(censored, c, time)
        event = torch.where(censored, torch.zeros_like(event), event)

        y = torch.stack([time, event], dim=-1)             # (T, 2)
        return X.float(), y.float()
