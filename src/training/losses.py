"""Funciones de pérdida para clasificación de género y regresión de edad simultáneas."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LossOutput:
    """Mantiene visible la pérdida de cada tarea para curvas e interpretación."""

    total: torch.Tensor
    gender: torch.Tensor
    age: torch.Tensor


class MultiTaskLoss(nn.Module):
    """Combina la pérdida de clasificación y de regresión con un peso configurable."""

    def __init__(self, lambda_age: float = 0.01) -> None:
        """Inicializa los criterios de pérdida y el peso de la pérdida de edad.

        Args:
            lambda_age: Peso aplicado a la pérdida de edad al combinarla con la
                pérdida de género. No puede ser negativo.

        Raises:
            ValueError: Si `lambda_age` es negativo.
        """
        super().__init__()
        if lambda_age < 0:
            raise ValueError("lambda_age no puede ser negativo.")
        self.lambda_age = lambda_age
        self.gender_criterion = nn.CrossEntropyLoss()
        self.age_criterion = nn.SmoothL1Loss()

    def forward(
        self,
        gender_logits: torch.Tensor,
        age_predictions: torch.Tensor,
        gender_targets: torch.Tensor,
        age_targets: torch.Tensor,
    ) -> LossOutput:
        """Calcula la pérdida combinada de género y edad para un lote.

        Args:
            gender_logits: Logits de género predichos, con forma [B, 2].
            age_predictions: Edades predichas, con forma [B].
            gender_targets: Etiquetas verdaderas de género, con forma [B].
            age_targets: Edades verdaderas, con forma [B].

        Returns:
            `LossOutput` con la pérdida total y las pérdidas individuales de
            género y edad.
        """
        gender_loss = self.gender_criterion(gender_logits, gender_targets)
        age_loss = self.age_criterion(age_predictions, age_targets)
        total_loss = gender_loss + self.lambda_age * age_loss
        return LossOutput(total=total_loss, gender=gender_loss, age=age_loss)
