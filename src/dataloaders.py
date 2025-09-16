import os
import torch
import numpy as np

from typing import Literal
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from pathlib import Path


class TMEData(Dataset):
    def __init__(self, root: str | Path, transform: T.Compose | None = None) -> None:
        super().__init__()

        root = Path(root)
        self.transform = transform

        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        self.idx_to_class = {i: c for i, c in enumerate(classes)}
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples = []
        for c in classes:
            class_dir = root / c

            for fname in sorted(os.listdir(class_dir)):
                fpath = class_dir / fname
                self.samples.append((fpath, self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, y = self.samples[index]

        with Image.open(path) as img:
            x = img.convert("RGB")

            if self.transform:
                x = self.transform(x)

        return x, y


class MNISTData(Dataset):
    # @TODO: figureout lazy loading

    def __init__(
        self,
        path: str,
        data_type: Literal["train", "test", "val"],
        transform: T.Compose | None = None,
        to_ignore: list[int] | None = None,
    ) -> None:

        self.transform = transform
        self.mask = to_ignore

        with np.load(path) as data:
            self.images = data[f"{data_type}_images"]
            self.labels = data[f"{data_type}_labels"]

        if self.mask:
            keep = ~np.isin(self.labels, self.mask)
            self.images = self.images[keep]
            self.labels = self.labels[keep]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]

        # Convert and transform
        x = Image.fromarray(x)
        y = torch.from_numpy(y).int()

        if self.transform:
            x = self.transform(x)

        return x, y
