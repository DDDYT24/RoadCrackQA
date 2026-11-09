"""Classical OpenCV crack detector: CLAHE, blur, black-hat, percentile threshold, small
component removal."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from crackqa.config import BaselineConfig, Config
from crackqa.data_qa import read_image
from crackqa.metrics import best_threshold, summarize
from crackqa.split import manifest_path
from crackqa.tiling import AMBIGUOUS, CRACK, cut, read_rows


def crack_map(bgr: np.ndarray, cfg: BaselineConfig) -> np.ndarray:
    """Binary map (0 / 1) of pixels the classical pipeline calls crack."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid,) * 2)
    gray = clahe.apply(gray)
    if cfg.blur_kernel:
        gray = cv2.GaussianBlur(gray, (cfg.blur_kernel,) * 2, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.blackhat_kernel,) * 2)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)  # dark structures < kernel
    level = np.percentile(blackhat, cfg.threshold_percentile)
    binary = (blackhat > level).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_component_px
    return keep[labels]


def tile_scores(rows: list[dict], cfg: Config) -> np.ndarray:
    """Crack pixel ratio of the baseline map in each tile; the map is built per whole image."""
    size = cfg.tiling.tile_size
    by_image: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_image[r["image"]].append(i)
    scores = np.zeros(len(rows))
    for image, idx in by_image.items():
        _, bgr = read_image(cfg.paths.raw / image, cv2.IMREAD_COLOR)
        cmap = crack_map(bgr, cfg.baseline)
        for i in idx:
            scores[i] = cut(cmap, rows[i]["y"], rows[i]["x"], size).mean()
    return scores


def scored_rows(cfg: Config, split: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Labelled rows (ambiguous dropped), their 0/1 labels and baseline scores."""
    rows = [r for r in read_rows(manifest_path(cfg, split)) if r["label"] != AMBIGUOUS]
    labels = np.array([r["label"] == CRACK for r in rows])
    return rows, labels, tile_scores(rows, cfg)


def tune_on_val(cfg: Config) -> dict:
    """Pick the tile threshold on val only and write reports/baseline_val.json."""
    _, labels, scores = scored_rows(cfg, "val")
    threshold, _ = best_threshold(labels, scores)
    result = {"split": "val", "rule": "crack if baseline crack pixel ratio >= threshold"}
    result.update(summarize(labels, scores, threshold))
    path = cfg.paths.reports / "baseline_val.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def load_val_threshold(cfg: Config) -> float:
    path: Path = cfg.paths.reports / "baseline_val.json"
    return json.loads(path.read_text(encoding="utf-8"))["threshold"]
