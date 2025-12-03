"""PyTorch dataset and transforms for WM811K wafer maps."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class WaferTransforms:
    """Factory for train/validation/test transforms."""

    def __init__(self, image_size: int = 64, num_channels: int = 1, augment: bool = True) -> None:
        self.image_size = image_size
        self.num_channels = num_channels
        self.augment = augment

    def _base_transforms(self) -> List[Callable]:
        ops: List[Callable] = [transforms.Resize((self.image_size, self.image_size))]
        if self.num_channels == 1:
            ops.append(transforms.Grayscale(num_output_channels=1))
        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(mean=[0.5] * self.num_channels, std=[0.5] * self.num_channels))
        return ops

    def train(self) -> transforms.Compose:
        ops = []
        if self.augment:
            ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15)])
        ops.extend(self._base_transforms())
        return transforms.Compose(ops)

    def eval(self) -> transforms.Compose:
        return transforms.Compose(self._base_transforms())


class WaferMapDataset(Dataset):
    """Dataset that loads wafer map images and labels.

    The dataset expects wafer maps stored under ``root``. A simple reference
    implementation is provided for NumPy arrays (``.npy``) and PNG images.
    Update :meth:`_load_item` to match your storage format if necessary.
    """

    def __init__(
        self,
        root: Path,
        indices: Optional[List[int]] = None,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        label_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.indices = indices
        self.label_map = label_map or {}

        self.images, self.labels = self._scan_files()
        if self.indices is not None:
            self.images = [self.images[i] for i in self.indices]
            self.labels = [self.labels[i] for i in self.indices]

    def _scan_files(self) -> Tuple[List[Path], List[int]]:
        # Expect structure root/{class_name}/*.{png,npy}
        paths: List[Path] = []
        labels: List[int] = []
        class_names = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not self.label_map:
            self.label_map = {name: idx for idx, name in enumerate(class_names)}
        for class_name in class_names:
            class_dir = self.root / class_name
            for file in class_dir.glob("*.*"):
                if file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".npy"}:
                    continue
                paths.append(file)
                labels.append(self.label_map[class_name])
        return paths, labels

    def _load_item(self, path: Path) -> Image.Image:
        if path.suffix.lower() == ".npy":
            array = np.load(path)
            if array.ndim == 2:
                array = np.expand_dims(array, axis=-1)
            image = Image.fromarray(array.astype(np.uint8))
        else:
            image = Image.open(path).convert("RGB")
        return image

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:  # type: ignore[override]
        path = self.images[idx]
        label = self.labels[idx]
        image = self._load_item(path)
        if self.transform:
            image = self.transform(image)
        return image, label
