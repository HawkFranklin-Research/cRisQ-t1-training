from __future__ import annotations

import torch
from torch import nn, Tensor
import numpy as np

from .reg2cls import standard_scaling, outlier_removing, permute_classes


class RegressionToSurvival(nn.Module):
    """Transforms a regression dataset into a survival format with time-to-event
    and event-indicator labels.

    It also processes features similarly to the original Reg2Cls module."""

    def __init__(self, hp: dict):
        super().__init__()
        self.hp = hp
        # Use a random censoring rate for each batch, sampled uniformly between
        # 10% and 50%
        self.censoring_rate = np.random.uniform(0.1, 0.5)

    def forward(self, X: Tensor, y: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Processes a single dataset (X, y) and converts it to a survival
        format.

        Parameters
        ----------
        X : Tensor
            Features of shape (T, H), where H is the number of features.
        y : Tensor
            Continuous targets of shape (T,). This will be treated as the true
            time-to-event.

        Returns
        -------
        tuple[Tensor, tuple[Tensor, Tensor]]
            A tuple containing:
            - Processed features of shape (T, max_features).
            - A tuple of (event_indicator, observed_time) for the survival task.
        """
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Input shapes mismatch or incorrect dims. X: {X.shape}, y: {y.shape}"
            )

        # 1. Process features
        processed_X = self._process_features(X)

        # 2. Generate survival targets from the continuous y
        event_indicator, observed_time = self._generate_survival_targets(y)

        return processed_X.float(), (event_indicator.float(), observed_time.float())

    def _process_features(self, X: Tensor) -> Tensor:
        """Process features through outlier removal, shuffling, scaling, and
        padding."""
        num_features = X.shape[1]
        max_features = self.hp["max_features"]

        X = outlier_removing(X, threshold=4)
        X = standard_scaling(X)

        if self.hp.get("permute_features", True):
            perm = torch.randperm(num_features, device=X.device)
            X = X[:, perm]

        if self.hp.get("scale_by_max_features", False):
            scaling_factor = num_features / max_features
            X = X / scaling_factor

        if num_features < max_features:
            X = torch.nn.functional.pad(
                X, (0, max_features - num_features), mode="constant", value=0.0
            )

        return X

    def _generate_survival_targets(self, y_true_time: Tensor) -> tuple[Tensor, Tensor]:
        """Generates event and time labels using random censoring."""
        # Standardize the true time to have a well-behaved distribution
        y_true_time = standard_scaling(y_true_time.unsqueeze(-1)).squeeze(-1)

        # Generate random censoring times from a normal distribution
        # The mean is chosen to achieve the desired censoring rate
        censoring_time = torch.normal(
            mean=np.quantile(y_true_time.cpu().numpy(), 1 - self.censoring_rate),
            std=torch.std(y_true_time) * 2,  # High variance for more randomness
            size=y_true_time.shape,
            device=y_true_time.device,
        )

        # Determine the event indicator and the observed time
        # Event is 1 if the true event time is before the censoring time (event observed)
        # Event is 0 if censoring happens first (event not observed)
        event_indicator = (y_true_time <= censoring_time).float()
        observed_time = torch.min(y_true_time, censoring_time)

        return event_indicator, observed_time
