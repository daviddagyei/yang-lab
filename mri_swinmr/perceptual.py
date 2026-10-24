"""Frozen VGG-19 perceptual loss used by the published SwinMR objective."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


class VGG19PerceptualLoss(nn.Module):
    """Weighted L1 distances at the five official VGG feature endpoints."""

    def __init__(self) -> None:
        super().__init__()
        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        endpoints = (2, 7, 16, 25, 34)
        starts = (0, 3, 8, 17, 26)
        self.blocks = nn.ModuleList(
            nn.Sequential(*[features[index] for index in range(start, stop + 1)])
            for start, stop in zip(starts, endpoints)
        )
        self.feature_weights = (0.1, 0.1, 1.0, 1.0, 1.0)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "VGG19PerceptualLoss":
        return super().train(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("VGG inputs must be matching [B, 3, H, W] tensors")
        prediction_features = (prediction - self.mean) / self.std
        target_features = (target.detach() - self.mean) / self.std
        loss = prediction.new_zeros(())
        for weight, block in zip(self.feature_weights, self.blocks):
            prediction_features = block(prediction_features)
            target_features = block(target_features)
            loss = loss + weight * torch.nn.functional.l1_loss(
                prediction_features, target_features
            )
        return loss
