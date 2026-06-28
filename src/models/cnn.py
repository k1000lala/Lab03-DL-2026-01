"""Red convolucional multitarea simple implementada con PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseMultiTaskModel


class MultiTaskCNN(BaseMultiTaskModel):
    """Aprende características faciales compartidas y predice género y edad.

    La capa de pooling adaptativo mantiene pequeña la parte totalmente conectada
    y permite que la misma arquitectura reciba imágenes cuadradas de distintos
    tamaños.
    """

    def __init__(self, dropout: float = 0.4) -> None:
        """Inicializa la arquitectura CNN multitarea.

        Args:
            dropout: Probabilidad de dropout aplicada tras la capa compartida.
                Debe estar en el intervalo [0, 1).
        """
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en el intervalo [0, 1).")

        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.gender_head = nn.Linear(256, 2)
        self.age_head = nn.Linear(256, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcula los logits de género y la predicción de edad de un lote de imágenes.

        Args:
            images: Tensor de imágenes de entrada con forma [B, 3, H, W].

        Returns:
            Tupla con los logits de género (forma [B, 2]) y las predicciones de
            edad (forma [B]).
        """
        features = self.features(images)
        representation = self.shared(features)
        gender_logits = self.gender_head(representation)
        age_predictions = self.age_head(representation).squeeze(1)
        return gender_logits, age_predictions
