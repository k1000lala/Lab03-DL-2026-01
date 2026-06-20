"""ResNet transfer learning multitask model implemented with PyTorch."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from src.models.base import BaseMultiTaskModel


class MultiTaskResNet(BaseMultiTaskModel):
    """Learn shared ResNet18 features and predict gender and age."""

    def __init__(self, trainable_blocks: int = 0, pretrained: bool = True) -> None:
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
        representation = self.backbone(images)
        gender_logits = self.gender_head(representation)
        age_predictions = self.age_head(representation).squeeze(1)
        return gender_logits, age_predictions
