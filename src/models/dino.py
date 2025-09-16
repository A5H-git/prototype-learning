"""
Adapted from Meta's dinov2
https://github.com/facebookresearch/dinov2
"""

import torch
import torchvision.transforms as T

from torch import nn


def get_dino_transforms(resize_size: int = 224):
    to_tensor = T.ToTensor()
    resize = T.Resize((resize_size, resize_size), antialias=True)
    normalize = T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return T.Compose([to_tensor, resize, normalize])


class DinoV2(nn.Module):
    def __init__(
        self, out_dim, repo="facebookresearch/dinov2", model="dinov2_vitb14"
    ) -> None:
        super().__init__()
        self.backbone: nn.Module = torch.hub.load(repo_or_dir=repo, model=model)  # type: ignore

    def forward(self, x):
        h = self.backbone(x)
        return h
