"""Synthetic CRACK500-style fixtures: dark lines drawn on noise textures, built in tmp_path."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from crackqa.config import Config, parse_config

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": "traincrop", "val": "valcrop", "test": "testcrop"}
H, W = 360, 640


def texture(seed: int, h: int = H, w: int = W) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(128, 40, (h // 8, w // 8)).astype(np.float32)
    base = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    base += rng.normal(0, 8, (h, w)).astype(np.float32)
    gray = np.clip(base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def crack_pair(seed: int, h: int = H, w: int = W) -> tuple[np.ndarray, np.ndarray]:
    """An image with a dark polyline and its binary mask (0 / 255)."""
    rng = np.random.default_rng(seed + 10_000)
    img = texture(seed, h, w)
    mask = np.zeros((h, w), np.uint8)
    pts = np.stack([np.linspace(0, w - 1, 6), rng.uniform(0, h - 1, 6)], axis=1).astype(np.int32)
    cv2.polylines(img, [pts], False, (40, 40, 40), 3)
    cv2.polylines(mask, [pts], False, 255, 3)
    return img, mask


def write(path: Path, arr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, arr)
    assert ok
    buf.tofile(str(path))
    return path


class Dataset:
    """A small clean dataset; tests then damage it on purpose."""

    def __init__(self, root: Path):
        self.root = root
        self.seed = 0

    def add(self, split: str, stem: str, mask: bool = True) -> tuple[Path, Path]:
        self.seed += 1
        img, m = crack_pair(self.seed)
        folder = self.root / SPLITS[split]
        img_path = write(folder / f"{stem}.jpg", img)
        mask_path = folder / f"{stem}.png"
        if mask:
            write(mask_path, m)
        return img_path, mask_path

    def config_dict(self, reports: Path) -> dict:
        """configs/default.yaml pointed at this dataset, with no excluded sources."""
        raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
        work = self.root.parent
        raw["paths"] = {
            "raw": str(self.root),
            "processed": str(work / "processed"),
            "outputs": str(work / "outputs"),
            "reports": str(reports),
        }
        raw["data"]["splits"] = dict(SPLITS)
        raw["data"]["exclude_sources"] = {}
        return raw

    def config(self, reports: Path) -> Config:
        return parse_config(self.config_dict(reports))

    def config_file(self, reports: Path) -> Path:
        path = self.root.parent / "config.yaml"
        path.write_text(yaml.safe_dump(self.config_dict(reports)), encoding="utf-8")
        return path


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = Dataset(tmp_path / "raw")
    ds.add("train", "20160222_081011_1_1")
    ds.add("train", "20160222_081011_1_361")
    ds.add("train", "20160222_081102_1_1")
    ds.add("val", "20160307_164211_1_1")
    ds.add("test", "20160328_151021_1_1")
    return ds


@pytest.fixture
def default_config_path() -> Path:
    return ROOT / "configs" / "default.yaml"
