"""Shade tiles by crack probability, or predicted crack pixels, on top of the photo."""

from __future__ import annotations

import cv2
import numpy as np

RED = np.array([0, 0, 255], np.float32)  # BGR


def overlay(
    bgr: np.ndarray,
    coords: list[tuple[int, int]],
    probs: np.ndarray,
    size: int,
    threshold: float,
    max_alpha: float = 0.6,
) -> np.ndarray:
    """Same size as the input: each tile tinted red in proportion to its probability, with a
    border on tiles at or above the threshold."""
    out = bgr.astype(np.float32)
    for (y, x), p in zip(coords, probs, strict=True):
        a = max_alpha * float(p)
        region = out[y : y + size, x : x + size]
        region[:] = region * (1 - a) + RED * a
    out = out.clip(0, 255).astype(np.uint8)
    thickness = max(1, round(min(bgr.shape[:2]) / 360))
    for (y, x), p in zip(coords, probs, strict=True):
        if p >= threshold:
            cv2.rectangle(out, (x, y), (x + size - 1, y + size - 1), (0, 0, 255), thickness)
    return out


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Same size as the input: predicted crack pixels tinted red."""
    out = bgr.astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + RED * alpha
    return out.clip(0, 255).astype(np.uint8)
