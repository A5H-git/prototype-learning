"""
Code adpated and borrowed from:
SSL Contrastive Learning by Phillip Lippe @ UvA
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Any
from torchvision.models import resnet50
from torchvision import transforms as T
from torch import optim, nn


class ContrastiveTransform:
    def __init__(self, base_tranform: T.Compose, n_views: int) -> None:
        self.base_transform = base_tranform
        self.n_views = n_views

    def __call__(self, x) -> Any:
        return [self.base_transform(x) for _ in range(self.n_views)]


def get_contrastive_transforms():
    color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

    t = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size=224),  # check size - don't really need
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=9),
            T.ToTensor(),
            # T.Normalize((0.5,), (0.5,)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return t


class SimCLR(pl.LightningModule):
    def __init__(
        self, hidden_dim, out_dim, learning_rate, temperature, weight_decay, max_epochs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        backbone = resnet50(weights=None)
        feat_dims = backbone.fc.in_features

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        # MLP projectoion layer
        self.mlp = nn.Sequential(
            nn.Linear(feat_dims, hidden_dim, bias=False),  # Linear
            nn.BatchNorm1d(hidden_dim),  # check this
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        h = self.encoder(x)  # f ~ [B, feat_dims, 1, 1]
        h = torch.flatten(h, 1)  # [B, feat_dims]
        z = self.mlp(h)  # g ~ [B, out_dim]
        return z, h

    def _log_stats(self, loss, top_1, top_5, mean_pos, mode="train"):
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_acc_top1", top_1, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_acc_top5", top_5, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_mean_pos", mean_pos, on_epoch=True, prog_bar=True)

    def _rank_loss(self, logits: torch.Tensor, pos_idx: torch.Tensor):
        sim_argsort = logits.argsort(dim=-1, descending=True)
        pos_rank = (sim_argsort == pos_idx[:, None]).float().argmax(dim=1)

        top_1 = (pos_rank == 0).float().mean()
        top_5 = (pos_rank < 5).float().mean()
        mean_pos = 1.0 + pos_rank.float().mean()

        return top_1, top_5, mean_pos

    def training_step(self, batch, batch_idx):
        (x1, x2), *_ = batch

        z1, _ = self(x1)
        z2, _ = self(x2)

        loss, logits, pos_idx = self.nt_xnet_loss(z1, z2)
        top_1, top_5, mean_pos = self._rank_loss(logits, pos_idx)

        self._log_stats(loss, top_1, top_5, mean_pos, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        (x1, x2), *_ = batch

        z1, _ = self(x1)
        z2, _ = self(x2)

        loss, logits, pos_idx = self.nt_xnet_loss(z1, z2)
        top_1, top_5, mean_pos = self._rank_loss(logits, pos_idx)

        self._log_stats(loss, top_1, top_5, mean_pos, "val")

        return loss

    def nt_xnet_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        tau: float = self.hparams.get("temperature", 0.2)

        # Normalize each batch entry
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        # Compute cos similarity
        sim = z @ z.T

        N = z.shape[0]

        self_mask = torch.eye(N, dtype=torch.bool, device=sim.device)

        # Remove self similarities from consideration
        sim.masked_fill_(self_mask, float("-inf"))
        logits = sim / tau

        # Two methods for locating positive views
        # Positive Views = The corresponding view pair

        # 1. Phillip Lippe: One-hot-encoding of positions
        # pos_idx = self_mask.roll(shifts=N//2, dims=0)
        # nll = (-logits[pos_idx] + torch.logsumexp(logits, dim=-1)).mean()

        # 2. ChatGPT: Get the N//2-wrapped index
        pos_idx = (torch.arange(N, device=z.device) + (N // 2)) % N
        nll = F.cross_entropy(logits, pos_idx)

        return nll, logits, pos_idx

    def configure_optimizers(self):  # type: ignore
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.get("learning_rate", 1e-4),
            weight_decay=self.hparams.get("weight_decay", 1e-2),
        )

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.get("max_epochs", 500),
            eta_min=self.hparams.get("learning_rate", 1e-4) / 50,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
