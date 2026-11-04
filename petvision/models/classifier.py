"""A compact CNN classifier implemented for the course project."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SmallCNN(nn.Module):
    """Four convolutional stages followed by global average pooling."""

    def __init__(self, num_classes: int = 37, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.features = nn.Sequential(
            ConvBlock(3, channels[0]),
            nn.MaxPool2d(2),
            ConvBlock(channels[0], channels[1]),
            nn.MaxPool2d(2),
            ConvBlock(channels[1], channels[2]),
            nn.MaxPool2d(2),
            ConvBlock(channels[2], channels[2] * 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(channels[2] * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


class ResNetClassifier(nn.Module):
    """ResNet-18 or ResNet-50 with a task-specific classification head."""

    def __init__(
        self, num_classes: int, architecture: str, pretrained: bool = False
    ) -> None:
        super().__init__()
        from torchvision.models import (
            ResNet18_Weights,
            ResNet50_Weights,
            resnet18,
            resnet50,
        )

        builder, default_weights = {
            "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
            "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
        }[architecture]
        weights = default_weights if pretrained else None
        self.backbone = builder(weights=weights)
        self.pretrained_source = (
            f"torchvision:{weights.__class__.__name__}.{weights.name}"
            if weights is not None
            else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Linear(in_features, num_classes))
        self.backbone_frozen = False

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone_frozen = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "ResNetClassifier":
        super().train(mode)
        if mode and self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

class FeatureFusionHead(nn.Module):
    """Classify concatenated raw and guided ResNet features."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        raw_feature: torch.Tensor,
        guided_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(torch.cat((raw_feature, guided_feature), dim=1))
def build_classifier(
    num_classes: int, architecture: str = "small_cnn", pretrained: bool = False
) -> nn.Module:
    if architecture == "small_cnn":
        if pretrained:
            raise ValueError("Pretrained weights require resnet18 or resnet50")
        return SmallCNN(num_classes=num_classes)
    if architecture in ("resnet18", "resnet50"):
        return ResNetClassifier(num_classes, architecture, pretrained=pretrained)
    raise ValueError(f"Unknown classifier architecture: {architecture}")


def classifier_from_checkpoint(checkpoint: dict, device: torch.device) -> nn.Module:
    model = build_classifier(
        num_classes=len(checkpoint["class_names"]),
        architecture=checkpoint.get("architecture", "small_cnn"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model
