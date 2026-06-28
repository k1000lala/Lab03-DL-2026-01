"""Perceptrón multicapa multitarea implementado con PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseMultiTaskModel


class MultiTaskMLP(BaseMultiTaskModel):
    """Aprende características totalmente conectadas compartidas y predice género y edad."""

    def __init__(self, image_size: int = 224, dropout: float = 0.4) -> None:
        """Inicializa la arquitectura MLP multitarea.

        Args:
            image_size: Alto y ancho (en píxeles) de las imágenes cuadradas de entrada.
            dropout: Probabilidad de dropout aplicada tras la primera capa oculta.
                Debe estar en el intervalo [0, 1).
        """
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
        """Calcula los logits de género y la predicción de edad de un lote de imágenes.

        Args:
            images: Tensor de imágenes de entrada con forma [B, 3, H, W].

        Returns:
            Tupla con los logits de género (forma [B, 2]) y las predicciones de
            edad (forma [B]).
        """
        representation = self.shared(images)
        gender_logits = self.gender_head(representation)
        age_predictions = self.age_head(representation).squeeze(1)
        return gender_logits, age_predictions
