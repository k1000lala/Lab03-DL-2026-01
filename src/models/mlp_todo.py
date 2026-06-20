"""Multitask multilayer perceptron implemented with PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseMultiTaskModel


class MultiTaskMLP(BaseMultiTaskModel):
    """Learn shared fully connected features and predict gender and age."""

    def __init__(self, image_size: int = 224, dropout: float = 0.4) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en el intervalo [0, 1).")

        self.image_size = image_size
        self.dropout = dropout
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size * image_size, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.gender_head = nn.Linear(128, 2)
        self.age_head = nn.Linear(128, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.shared(images)
        gender_logits = self.gender_head(representation)
        age_predictions = self.age_head(representation).squeeze(1)
        return gender_logits, age_predictions
