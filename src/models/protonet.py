"""
Code adpated and borrowed from:
Meta-Learning by Phillip Lippe @ UvA
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial16/Meta_Learning.html
"""

import torch
import random
import numpy as np

from collections import defaultdict

import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F

import torchvision.transforms as T


def get_protonet_transforms(resize_size: int = 224):
    t = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(
                size=(resize_size, resize_size),
                antialias=True,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return t


# Following UvA
class FewShotBatchSampler(object):
    def __init__(
        self,
        dataset_targets,
        N_way,
        K_shot,
        shuffle: bool = True,
        shuffle_once: bool = False,
        include_query: bool = False,
    ) -> None:
        super().__init__()

        self.dataset_targets = dataset_targets
        self.include_query = include_query

        self.N_way = N_way
        self.K_shot = K_shot
        if self.include_query:
            self.K_shot *= 2

        self.shuffle = shuffle

        # Creates the episode: size = n_classes * n_images
        self.batch_size = self.N_way * self.K_shot

        self.unique_classes = torch.unique(self.dataset_targets).tolist()
        self.n_classes = len(self.unique_classes)

        # Group classes
        self.class_idxs: dict[int, torch.Tensor] = {}

        # number of K_shot batches for each class
        self.batches_per_class: dict[int, int] = {}

        for c in self.unique_classes:
            self.class_idxs[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.class_idxs[c].shape[0] // self.K_shot

        # Create class list
        self.n_iterations = sum(self.batches_per_class.values()) // self.N_way
        self.classes = [
            c for c in self.unique_classes for _ in range(self.batches_per_class[c])
        ]

        if shuffle_once or self.shuffle:
            self._shuffle()
        else:
            sort_idxs = [
                i + p * self.n_classes
                for i, c in enumerate(self.unique_classes)
                for p in range(self.batches_per_class[c])
            ]

            self.classes = np.array(self.classes)[np.argsort(sort_idxs)].tolist()

    def _shuffle(self):
        for c in self.unique_classes:
            perm = torch.randperm(self.class_idxs[c].shape[0])
            self.class_idxs[c] = self.class_idxs[c][perm]

        random.shuffle(self.classes)

    def __iter__(self):
        """Gets the support + query batches"""
        if self.shuffle:
            self._shuffle()

        start_idx = defaultdict(int)

        for i in range(self.n_iterations):
            # Select N classes
            class_batch = self.classes[i * self.N_way : (i + 1) * self.N_way]

            idx_batch = []

            # Add next K examples for each class
            for c in class_batch:
                idxs = self.class_idxs[c]
                idx_batch.extend(idxs[start_idx[c] : self.K_shot + start_idx[c]])

                start_idx[c] += self.K_shot  # Next batch

            if self.include_query:
                idx_batch = idx_batch[::2] + idx_batch[1::2]

            yield idx_batch

    def __len__(self):
        return self.n_iterations


def split_batch(imgs: torch.Tensor, targets: torch.Tensor):
    """Util tool for imaging"""
    support_imgs, query_imgs = imgs.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets


class ProtoNet(pl.LightningModule):
    def __init__(self, learning_rate: float, backbone: nn.Module):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.encoder = backbone

    def configure_optimizers(self):  # type: ignore
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.get("learning_rate", 1e-4)
        )

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[140, 180], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    @staticmethod
    def calculate_prototypes(x: torch.Tensor, y: torch.Tensor):
        # x: [N, dim]; y: [N, ]
        classes, _ = torch.unique(y).sort()

        protos = []
        for c in classes:
            p = x[torch.where(y == c)[0]].mean(dim=0)
            protos.append(p)

        protos = torch.stack(protos, dim=0)
        return protos, classes

    @staticmethod
    def classify_feats(
        prototypes: torch.Tensor,
        classes: torch.Tensor,
        feats: torch.Tensor,
        targets: torch.Tensor,
    ):
        # Euclidean Dist
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)

        log_prob = F.log_softmax(-dist, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)

        acc = (log_prob.argmax(dim=1) == labels).float().mean()

        return log_prob, labels, acc

    def forward(self, x):
        return self.encoder(x)

    def proto_loss(self, x: torch.Tensor, y: torch.Tensor):
        y = y.flatten()
        feats: torch.Tensor = self(x)

        supp_feats, query_feats, supp_tgts, query_tgts = split_batch(feats, y)
        proto, classes = ProtoNet.calculate_prototypes(supp_feats, supp_tgts)

        log_prob, targets, acc = ProtoNet.classify_feats(
            proto, classes, query_feats, query_tgts
        )

        loss = F.nll_loss(log_prob, targets)  # CE already uses softmax

        return loss, acc

    def _log_stats(self, loss, acc, mode="train"):
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, acc = self.proto_loss(x, y)
        self._log_stats(loss, acc, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, acc = self.proto_loss(x, y)
        self._log_stats(loss, acc, "val")
