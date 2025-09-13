import numpy as np
import torch
from typing import Literal
from torch.utils.data import Dataset
from torchvision import transforms as T


class MNISTData(Dataset):
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
        x = torch.from_numpy(x).float()

        if self.transform:
            x = self.transform(x)

        return x, y
