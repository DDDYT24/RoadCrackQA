"""Tiles in memory and the torch Dataset with augmentation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from crackqa.data_qa import read_image
from crackqa.tiling import CRACK, cut

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_tiles(rows: list[dict], root: Path, size: int) -> np.ndarray:
    """(N, size, size, 3) uint8 RGB tiles; each source image is decoded once."""
    out = np.zeros((len(rows), size, size, 3), np.uint8)
    by_image: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_image[r["image"]].append(i)
    for image, idx in by_image.items():
        _, bgr = read_image(root / image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode {image}; run crackqa qa first")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for i in idx:
            out[i] = cut(rgb, rows[i]["y"], rows[i]["x"], size)
    return out


def labels_of(rows: list[dict]) -> np.ndarray:
    return np.array([r["label"] == CRACK for r in rows], dtype=np.int64)


def normalize(tiles: np.ndarray) -> torch.Tensor:
    """uint8 (N, H, W, 3) RGB -> float (N, 3, H, W), ImageNet normalized."""
    x = torch.from_numpy(tiles).permute(0, 3, 1, 2).float().div_(255)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def augment(x: torch.Tensor) -> torch.Tensor:
    """One (3, H, W) tile in [0, 1]: flips, 90 degree rotations, brightness and contrast.
    Draws from the global torch generator (seeded by seed_everything)."""
    if torch.rand(()) < 0.5:
        x = x.flip(2)
    if torch.rand(()) < 0.5:
        x = x.flip(1)
    x = torch.rot90(x, int(torch.randint(0, 4, ())), dims=(1, 2))
    contrast = 0.8 + 0.4 * torch.rand(())
    brightness = 0.2 * torch.rand(()) - 0.1
    mean = x.mean()
    return ((x - mean) * contrast + mean + brightness).clamp_(0, 1)


class TileDataset(Dataset):
    def __init__(self, tiles: np.ndarray, labels: np.ndarray, train: bool):
        self.tiles = tiles
        self.labels = labels
        self.train = train

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.tiles[i]).permute(2, 0, 1).float().div_(255)
        if self.train:
            x = augment(x)
        return (x - IMAGENET_MEAN) / IMAGENET_STD, int(self.labels[i])
