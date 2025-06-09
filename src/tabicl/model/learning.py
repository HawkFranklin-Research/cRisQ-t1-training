from __future__ import annotations

import torch
from torch import nn, Tensor

from .layers import OneHotAndLinear
from .encoders import Encoder


class ICLearning(nn.Module):
    """Dataset-wise in-context learning with automatic hierarchical classification support.

    This module implements in-context learning that:
    1. Takes row representations and training labels as input
    2. Conditions the model on training examples
    3. Makes predictions for test examples based on learned patterns
    4. Automatically handles both small and large label spaces

    Parameters
    ----------
    max_classes : int
        Number of classes that the model supports natively. If the number of classes
        in the dataset exceeds this value, hierarchical classification is used.

    d_model : int
        Model dimension

    num_blocks : int
        Number of blocks used in the ICL encoder

    nhead : int
        Number of attention heads of the ICL encoder

    dim_feedforward : int
        Dimension of the feedforward network of the ICL encoder

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)
    """

    def __init__(
        self,
        max_classes: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.max_classes = max_classes
        self.norm_first = norm_first

        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        if self.norm_first:
            self.ln = nn.LayerNorm(d_model)

        # --- New multi-head setup for survival analysis ---
        # Encoder for the binary event label
        self.event_encoder = OneHotAndLinear(2, d_model)  # Event is binary (0 or 1)

        # A simple linear layer to project the continuous time value into the model dimension
        self.time_encoder = nn.Linear(1, d_model)

        # Classification head for predicting the event
        self.classification_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

        # Regression head for predicting the time-to-event
        self.regression_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

        # Keep the original decoder for compatibility if needed, but we won't use it for survival task
        self.original_decoder = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, max_classes))
        # --- End of new setup ---


    def _label_encoding(self, y: Tensor) -> Tensor:
        """Remapping target values to contiguous integers starting from 0."""

        unique_vals, _ = torch.unique(y, return_inverse=True)
        indices = unique_vals.argsort()
        return indices[torch.searchsorted(unique_vals, y)]

    def _icl_predictions(self, R: Tensor, y_train_survival: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        """In-context learning predictions for survival task."""

        y_event_train, y_time_train = y_train_survival
        train_size = y_event_train.shape[1]

        # Encode both event and time information into the training samples
        event_embedding = self.event_encoder(y_event_train.long())
        time_embedding = self.time_encoder(y_time_train.unsqueeze(-1))
        R[:, :train_size] = R[:, :train_size] + event_embedding + time_embedding

        # Process through the ICL transformer
        src = self.tf_icl(R, attn_mask=train_size)
        if self.norm_first:
            src = self.ln(src)

        # Get predictions from both heads
        test_src = src[:, train_size:]

        classification_logits = self.classification_head(test_src).squeeze(-1)
        regression_output = self.regression_head(test_src).squeeze(-1)

        return {"logits": classification_logits, "time": regression_output}




    def forward(
        self,
        R: Tensor,
        y_train: tuple[Tensor, Tensor],
    ) -> dict[str, Tensor]:
        """
        Simplified forward pass for survival pre-training.
        """

        # This is only used during training, so we call _icl_predictions directly.
        # The complex inference logic is not needed for pre-training.
        return self._icl_predictions(R, y_train)
