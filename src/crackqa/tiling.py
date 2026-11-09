"""Tile grid, tile labels from masks, and the tile table for every split."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from crackqa.config import Config
from crackqa.data_qa import parse_source_id, read_image

CRACK, NO_CRACK, AMBIGUOUS = 1, 0, -1
TILE_COLUMNS = ["tile_id", "image", "mask", "y", "x", "source_id", "split", "label", "crack_px"]


def tile_origins(length: int, size: int, stride: int, drop_edge: bool) -> list[int]:
    """Start offsets along one axis. Without drop_edge, one more tile sits flush with the end."""
    if length < size:
        return []
    starts = list(range(0, length - size + 1, stride))
    if not drop_edge and starts[-1] + size < length:
        starts.append(length - size)
    return starts


def tile_grid(height: int, width: int, size: int, stride: int, drop_edge: bool) -> list[tuple]:
    """(y, x) of every tile's top-left corner, row by row."""
    ys = tile_origins(height, size, stride, drop_edge)
    xs = tile_origins(width, size, stride, drop_edge)
    return [(y, x) for y in ys for x in xs]


def tile_label(crack_px: int, crack_min_px: int) -> int:
    if crack_px == 0:
        return NO_CRACK
    return CRACK if crack_px >= crack_min_px else AMBIGUOUS


def cut(image: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    return image[y : y + size, x : x + size]


def _image_files(folder: Path, cfg: Config) -> list[Path]:
    suffix = cfg.data.mask_suffix.lower()
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in cfg.data.image_exts
        and not p.name.lower().endswith(suffix)
    )


def mask_path_for(image: Path, cfg: Config) -> Path:
    return image.with_name(image.stem + cfg.data.mask_suffix)


def build_tile_table(cfg: Config) -> list[dict]:
    """One row per tile of every image in every split folder (exclusions are applied later)."""
    root, t = cfg.paths.raw, cfg.tiling
    rows = []
    for split, folder in cfg.data.splits.items():
        for image in _image_files(root / folder, cfg):
            mask_file = mask_path_for(image, cfg)
            _, mask = read_image(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"missing or unreadable mask for {image}; run crackqa qa first")
            source = parse_source_id(image.name, cfg.data.source_id_regex)
            if source is None:
                raise ValueError(f"no source id in {image.name}; run crackqa qa first")
            rel = image.relative_to(root).as_posix()
            for y, x in tile_grid(*mask.shape[:2], t.tile_size, t.stride, t.drop_edge):
                crack_px = int(np.count_nonzero(cut(mask, y, x, t.tile_size)))
                rows.append(
                    {
                        "tile_id": f"{rel}@{y},{x}",
                        "image": rel,
                        "mask": mask_file.relative_to(root).as_posix(),
                        "y": y,
                        "x": x,
                        "source_id": source,
                        "split": split,
                        "label": tile_label(crack_px, cfg.labels.crack_min_px),
                        "crack_px": crack_px,
                    }
                )
    return rows


def write_rows(rows: list[dict], path: Path, columns: list[str] = TILE_COLUMNS) -> Path:
    """Deterministic CSV: fixed columns, LF line endings, rows in the given order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("y", "x", "label", "crack_px"):
            r[key] = int(r[key])
    return rows
