"""Folder of photos -> results.csv, overlay images and summary.json."""

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
from crackqa.models import load_model, predict
from crackqa.overlay import overlay
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


def score_image(model, bgr: np.ndarray, cfg: Config) -> tuple[list[tuple], np.ndarray]:
    t = cfg.tiling
    coords = tile_grid(*bgr.shape[:2], t.tile_size, t.stride, t.drop_edge)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tiles = np.stack([cut(rgb, y, x, t.tile_size) for y, x in coords]) if coords else []
    probs = predict(model, tiles) if coords else np.zeros(0)
    return coords, probs


def list_files(input_dir: Path, include: str) -> list[Path]:
    """Files under input_dir whose name matches the include pattern, ignoring case."""
    pattern = include.lower()
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and fnmatch(p.name.lower(), pattern)
    )


def run_batch(cfg: Config, input_dir: Path, output_dir: Path, include: str = "*") -> dict:
    model, meta = load_model(cfg.paths.outputs)
    threshold, version = meta["threshold"], meta["model_version"]
    (output_dir / "overlays").mkdir(parents=True, exist_ok=True)
    log = _logger(output_dir)

    started = time.perf_counter()
    rows = []
    for path in list_files(input_dir, include):
        rel = path.relative_to(input_dir).as_posix()
        row = dict.fromkeys(COLUMNS, "")
        row.update(file=rel, model_version=version)
        if path.suffix.lower() not in IMAGE_EXTS:
            row["status"] = "skipped"
            log.info("skipped %s: not an image extension", rel)
            rows.append(row)
            continue
        _, bgr = read_image(path, cv2.IMREAD_COLOR)
        if bgr is None:
            row["status"] = "corrupt"
            log.warning("corrupt %s: cannot be decoded", rel)
            rows.append(row)
            continue
        coords, probs = score_image(model, bgr, cfg)
        row.update(width=bgr.shape[1], height=bgr.shape[0], n_tiles=len(coords))
        if not coords:
            row["status"] = "skipped"
            log.info("skipped %s: smaller than one tile", rel)
            rows.append(row)
            continue
        row.update(
            max_prob=round(float(probs.max()), 6),
            mean_prob=round(float(probs.mean()), 6),
            crack_tile_ratio=round(float(np.mean(probs >= threshold)), 6),
            flagged=int(probs.max() >= threshold),  # any tile at or above the threshold
            status="ok",
        )
        shaded = overlay(bgr, coords, probs, cfg.tiling.tile_size, threshold)
        name = Path(rel).with_suffix("").as_posix().replace("/", "__")
        target = output_dir / "overlays" / f"{name}_overlay.jpg"
        cv2.imencode(".jpg", shaded, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(target))
        rows.append(row)
    runtime = time.perf_counter() - started

    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    statuses = [r["status"] for r in rows]
    ok = statuses.count("ok")
    summary = {
        "files": len(rows),
        "ok": ok,
        "corrupt": statuses.count("corrupt"),
        "skipped": statuses.count("skipped"),
        "flagged": sum(1 for r in rows if r["flagged"] == 1),
        "tiles": sum(r["n_tiles"] for r in rows if r["status"] == "ok"),
        "threshold": threshold,
        "model_version": version,
        "runtime_s": round(runtime, 2),
        "images_per_s": round(ok / runtime, 3) if runtime > 0 else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for handler in log.handlers:
        handler.close()
    log.handlers.clear()
    return summary
