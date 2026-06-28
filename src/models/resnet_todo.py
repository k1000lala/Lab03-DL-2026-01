"""Modelo multitarea de transferencia de aprendizaje basado en ResNet, implementado con PyTorch."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from src.models.base import BaseMultiTaskModel


class MultiTaskResNet(BaseMultiTaskModel):
    """Aprende características compartidas de ResNet18 y predice género y edad."""

    def __init__(self, trainable_blocks: int = 0, pretrained: bool = True) -> None:
        """Inicializa el backbone ResNet18 con un número configurable de bloques entrenables.

        Args:
            trainable_blocks: Cantidad de bloques residuales a descongelar, contando
                desde el más profundo (`layer4`) hacia el más superficial (`layer1`).
                Con 0, todo el backbone permanece congelado.
            pretrained: Si es True, carga los pesos preentrenados de ImageNet.

        Raises:
            ValueError: Si `trainable_blocks` es negativo.
        """
        super().__init__()
        if trainable_blocks < 0:
            raise ValueError("trainable_blocks debe ser >= 0.")

        self.trainable_blocks = trainable_blocks
        self.pretrained = pretrained

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        for parameter in backbone.parameters():
            parameter.requires_grad = False

        unfreezable = [backbone.layer4, backbone.layer3, backbone.layer2, backbone.layer1]
        for index in range(min(trainable_blocks, len(unfreezable))):
            for parameter in unfreezable[index].parameters():
                parameter.requires_grad = True

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.gender_head = nn.Linear(in_features, 2)
        self.age_head = nn.Linear(in_features, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcula los logits de género y la predicción de edad de un lote de imágenes.

        Args:
            images: Tensor de imágenes de entrada con forma [B, 3, H, W].

        Returns:
            Tupla con los logits de género (forma [B, 2]) y las predicciones de
            edad (forma [B]).
        """
        representation = self.backbone(images)
        gender_logits = self.gender_head(representation)
        age_predictions = self.age_head(representation).squeeze(1)
        return gender_logits, age_predictions
