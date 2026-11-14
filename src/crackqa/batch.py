"""Folder of photos -> results.csv, overlay images (and masks) and summary.json.

Two modes: `tiles` scores 128 px tiles with the ResNet18 classifier in paths.tile_model;
`seg` predicts a crack mask per pixel with SmallUNet. With gate.enabled, every photo is first
checked by the road gate; a photo below its threshold gets status not_road and no scoring.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from fnmatch import fnmatch
from pathlib import Path

import cv2
import numpy as np

from crackqa.config import Config
from crackqa.data_qa import read_image
from crackqa.models import load_model, predict, predict_mask
from crackqa.overlay import overlay, overlay_mask
from crackqa.tiling import cut, tile_grid

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLUMNS = [
    "file",
    "width",
    "height",
    "n_tiles",
    "max_prob",
    "mean_prob",
    "crack_tile_ratio",
    "flagged",
    "road_prob",
    "status",
    "model_version",
]
SEG_COLUMNS = [
    "file",
    "width",
    "height",
    "crack_pixel_ratio",
    "road_prob",
    "status",
    "model_version",
]


def _logger(output: Path) -> logging.Logger:
    log = logging.getLogger("crackqa.batch")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "batch.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(handler)
    log.propagate = False
    return log


def _save_jpg(image: np.ndarray, path: Path) -> None:
    cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(path))


def score_image(model, bgr: np.ndarray, cfg: Config) -> tuple[list[tuple], np.ndarray]:
    t = cfg.tiling
    coords = tile_grid(*bgr.shape[:2], t.tile_size, t.stride, t.drop_edge)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tiles = np.stack([cut(rgb, y, x, t.tile_size) for y, x in coords]) if coords else []
    probs = predict(model, tiles) if coords else np.zeros(0)
    return coords, probs


class TileScorer:
    columns = COLUMNS

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model, meta = load_model(cfg.paths.tile_model)
        self.threshold, self.version = meta["threshold"], meta["model_version"]

    def score(self, bgr: np.ndarray, out: Path, name: str) -> dict | None:
        """Result fields, or None when the photo is smaller than one tile."""
        coords, probs = score_image(self.model, bgr, self.cfg)
        if not coords:
            return None
        _save_jpg(
            overlay(bgr, coords, probs, self.cfg.tiling.tile_size, self.threshold),
            out / "overlays" / f"{name}_overlay.jpg",
        )
        return {
            "n_tiles": len(coords),
            "max_prob": round(float(probs.max()), 6),
            "mean_prob": round(float(probs.mean()), 6),
            "crack_tile_ratio": round(float(np.mean(probs >= self.threshold)), 6),
            "flagged": int(probs.max() >= self.threshold),  # any tile at or above threshold
        }


class SegScorer:
    columns = SEG_COLUMNS

    def __init__(self, cfg: Config):
        from crackqa.segment import load_seg_model

        self.model, meta = load_seg_model(cfg)
        self.threshold, self.version = meta["threshold"], meta["model_version"]

    def score(self, bgr: np.ndarray, out: Path, name: str) -> dict | None:
        prob = predict_mask(self.model, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        mask = prob >= self.threshold
        (out / "masks").mkdir(exist_ok=True)
        cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tofile(
            str(out / "masks" / f"{name}_mask.png")
        )
        _save_jpg(overlay_mask(bgr, mask), out / "overlays" / f"{name}_overlay.jpg")
        return {"crack_pixel_ratio": round(float(mask.mean()), 6)}


def list_files(input_dir: Path, include: str) -> list[Path]:
    """Files under input_dir whose name matches the include pattern, ignoring case."""
    pattern = include.lower()
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and fnmatch(p.name.lower(), pattern)
    )


def run_batch(
    cfg: Config, input_dir: Path, output_dir: Path, include: str = "*", mode: str = "tiles"
) -> dict:
    scorer = SegScorer(cfg) if mode == "seg" else TileScorer(cfg)
    gate = None
    if cfg.gate.enabled:
        from crackqa.gate import Gate

        gate = Gate(cfg)
    (output_dir / "overlays").mkdir(parents=True, exist_ok=True)
    log = _logger(output_dir)

    started = time.perf_counter()
    rows = []
    for path in list_files(input_dir, include):
        rel = path.relative_to(input_dir).as_posix()
        row = dict.fromkeys(scorer.columns, "")
        row.update(file=rel, model_version=scorer.version)
        rows.append(row)
        if path.suffix.lower() not in IMAGE_EXTS:
            row["status"] = "skipped"
            log.info("skipped %s: not an image extension", rel)
            continue
        _, bgr = read_image(path, cv2.IMREAD_COLOR)
        if bgr is None:
            row["status"] = "corrupt"
            log.warning("corrupt %s: cannot be decoded", rel)
            continue
        row.update(width=bgr.shape[1], height=bgr.shape[0])
        road = gate.road_prob(bgr) if gate is not None else None
        if road is not None:
            row["road_prob"] = round(road, 6)
            if road < gate.threshold:
                row["status"] = "not_road"
                log.info("not_road %s: road probability %.3f below %.3f", rel, road, gate.threshold)
                continue
        name = Path(rel).with_suffix("").as_posix().replace("/", "__")
        fields = scorer.score(bgr, output_dir, name)
        if fields is None:
            row["status"] = "skipped"
            row["n_tiles"] = 0
            log.info("skipped %s: smaller than one tile", rel)
            continue
        row.update(fields, status="ok")
    runtime = time.perf_counter() - started

    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scorer.columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    statuses = [r["status"] for r in rows]
    ok = statuses.count("ok")
    summary = {
        "mode": mode,
        "files": len(rows),
        "ok": ok,
        "not_road": statuses.count("not_road"),
        "corrupt": statuses.count("corrupt"),
        "skipped": statuses.count("skipped"),
    }
    if mode == "seg":
        ratios = [r["crack_pixel_ratio"] for r in rows if r["status"] == "ok"]
        summary["mean_crack_pixel_ratio"] = round(float(np.mean(ratios)), 6) if ratios else None
    else:
        summary["flagged"] = sum(1 for r in rows if r.get("flagged") == 1)
        summary["tiles"] = sum(r["n_tiles"] for r in rows if r["status"] == "ok")
    if gate is not None:
        summary.update(gate_threshold=gate.threshold, gate_model_version=gate.version)
    summary.update(
        threshold=scorer.threshold,
        model_version=scorer.version,
        runtime_s=round(runtime, 2),
        images_per_s=round(ok / runtime, 3) if runtime > 0 else None,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for handler in log.handlers:
        handler.close()
    log.handlers.clear()
    return summary
