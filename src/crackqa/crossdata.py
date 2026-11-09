"""Score a second labelled dataset (CFD) with the frozen model, no retraining."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from crackqa.baseline import crack_map, load_val_threshold
from crackqa.config import Config
from crackqa.data_qa import read_image
from crackqa.metrics import summarize
from crackqa.models import load_model, predict
from crackqa.tiling import AMBIGUOUS, CRACK, cut, tile_grid, tile_label


def labelled_pairs(images: Path, masks: Path, mask_suffix: str) -> list[tuple[Path, Path]]:
    """Images that have a mask; images without one are skipped (CFD ships 37 extra photos)."""
    pairs = []
    for image in sorted(p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".png")):
        mask = masks / f"{image.stem}{mask_suffix}"
        if mask.exists():
            pairs.append((image, mask))
    return pairs


def score_dataset(cfg: Config) -> dict:
    c, t = cfg.cross_dataset, cfg.tiling
    model, meta = load_model(cfg.paths.outputs)
    labels, probs, base, image_hits = [], [], [], []
    for image, mask_file in labelled_pairs(c.images, c.masks, c.mask_suffix):
        _, bgr = read_image(image, cv2.IMREAD_COLOR)
        _, mask = read_image(mask_file, cv2.IMREAD_GRAYSCALE)
        coords = tile_grid(*mask.shape[:2], t.tile_size, t.stride, t.drop_edge)
        tile_labels = [
            tile_label(int(np.count_nonzero(cut(mask, y, x, t.tile_size))), cfg.labels.crack_min_px)
            for y, x in coords
        ]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        p = predict(model, np.stack([cut(rgb, y, x, t.tile_size) for y, x in coords]))
        cmap = crack_map(bgr, cfg.baseline)
        b = [cut(cmap, y, x, t.tile_size).mean() for y, x in coords]
        image_hits.append(bool(p.max() >= meta["threshold"]))
        for label, pi, bi in zip(tile_labels, p, b, strict=True):
            if label != AMBIGUOUS:
                labels.append(label == CRACK)
                probs.append(pi)
                base.append(bi)
    y = np.array(labels)
    return {
        "dataset": c.name,
        "images": len(image_hits),
        "tiles": int(len(y)),
        "crack": int(y.sum()),
        "resnet18": summarize(y, np.array(probs), meta["threshold"]),
        "opencv_baseline": summarize(y, np.array(base), load_val_threshold(cfg)),
        # every CFD image contains a crack, so only image level recall is defined
        "image_level_recall": float(np.mean(image_hits)),
        "model_version": meta["model_version"],
    }


def write_report(cfg: Config, report: dict) -> Path:
    path = cfg.paths.reports / f"cross_{report['dataset'].lower()}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path
