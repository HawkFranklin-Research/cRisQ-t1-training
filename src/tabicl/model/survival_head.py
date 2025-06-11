from __future__ import annotations
import torch, torch.nn.functional as F
from torch import nn, Tensor


def cox_ph_nll(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    """
    Negative log partial-likelihood of the Cox PH model.

    Args
    ----
    risk  : (N,) real-valued scores
    time  : (N,) follow-up times
    event : (N,) 1 if observed, 0 if censored
    """
    order = torch.argsort(time, descending=True)
    log_cum_hazard = torch.logcumsumexp(risk[order], dim=0)
    return -(event[order] * (risk[order] - log_cum_hazard)).sum() / event.sum()


class RiskHead(nn.Module):
    """Simple linear projection to a scalar risk score."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x).squeeze(-1)
