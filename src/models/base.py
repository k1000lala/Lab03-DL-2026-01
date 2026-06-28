"""Clase base para modelos con salidas de género y edad."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseMultiTaskModel(nn.Module, ABC):
    """Interfaz común para todas las estrategias neuronales multitarea."""

    @abstractmethod
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve los logits de género [B, 2] y las predicciones de edad [B].

        Args:
            images: Tensor de imágenes de entrada con forma [B, 3, H, W].

        Returns:
            Tupla con los logits de género (forma [B, 2]) y las predicciones de
            edad (forma [B]).
        """

    def count_trainable_parameters(self) -> int:
        """Cuenta los parámetros del modelo con `requires_grad=True`.

        Returns:
            Número total de parámetros entrenables.
        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
