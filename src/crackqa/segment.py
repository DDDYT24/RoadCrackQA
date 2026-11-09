"""SmallUNet crack segmentation: training on random crops, pixel metrics, locked test run."""

from __future__ import annotations

import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from crackqa.baseline import crack_map
from crackqa.config import Config, config_digest
from crackqa.data_qa import read_image
from crackqa.datasets import IMAGENET_MEAN, IMAGENET_STD
from crackqa.models import SmallUNet, file_digest, predict_mask
from crackqa.runinfo import git_commit, git_problems, hardware
from crackqa.seed import seed_everything, torch_generator
from crackqa.split import file_sha256, manifest_path, read_test_lock, test_lock_path
from crackqa.tiling import read_rows

BINS = 1000  # probability histogram resolution for pooled pixel metrics


class SegRefused(RuntimeError):
    """The final segmentation test run's preconditions do not hold."""


def split_images(cfg: Config, split: str) -> list[tuple[str, str]]:
    """(image, mask) pairs of a split, after exclusions (taken from the split manifest)."""
    pairs = {(r["image"], r["mask"]) for r in read_rows(manifest_path(cfg, split))}
    return sorted(pairs)


def load_pairs(cfg: Config, pairs: list[tuple[str, str]]) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for image, mask in pairs:
        _, bgr = read_image(cfg.paths.raw / image, cv2.IMREAD_COLOR)
        _, m = read_image(cfg.paths.raw / mask, cv2.IMREAD_GRAYSCALE)
        out.append((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), m > 0))
    return out


class CropDataset(Dataset):
    """Random square crops with matching masks; flips, 90 degree turns, brightness and
    contrast. Draws from the global torch generator."""

    def __init__(self, items: list[tuple[np.ndarray, np.ndarray]], size: int, repeats: int):
        self.items, self.size, self.repeats = items, size, repeats

    def __len__(self) -> int:
        return len(self.items) * self.repeats

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, mask = self.items[i % len(self.items)]
        s = self.size
        y = int(torch.randint(0, rgb.shape[0] - s + 1, ()))
        x = int(torch.randint(0, rgb.shape[1] - s + 1, ()))
        img = torch.from_numpy(rgb[y : y + s, x : x + s]).permute(2, 0, 1).float() / 255
        m = torch.from_numpy(mask[y : y + s, x : x + s]).float()[None]
        if torch.rand(()) < 0.5:
            img, m = img.flip(2), m.flip(2)
        if torch.rand(()) < 0.5:
            img, m = img.flip(1), m.flip(1)
        k = int(torch.randint(0, 4, ()))
        img, m = torch.rot90(img, k, (1, 2)), torch.rot90(m, k, (1, 2))
        mean = img.mean()
        img = (img - mean) * (0.8 + 0.4 * torch.rand(())) + mean + 0.2 * torch.rand(()) - 0.1
        img = img.clamp(0, 1)
        return (img - IMAGENET_MEAN) / IMAGENET_STD, m


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    total = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (total + eps)).mean()


def histograms(model: SmallUNet, items) -> tuple[np.ndarray, np.ndarray]:
    """Counts of crack and background pixels per probability bin, pooled over images."""
    pos, neg = np.zeros(BINS, np.int64), np.zeros(BINS, np.int64)
    for rgb, mask in items:
        b = np.minimum((predict_mask(model, rgb) * BINS).astype(np.int64), BINS - 1)
        pos += np.bincount(b[mask], minlength=BINS)
        neg += np.bincount(b[~mask], minlength=BINS)
    return pos, neg


def pixel_metrics(tp: float, fp: float, fn: float) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def best_pixel_threshold(pos: np.ndarray, neg: np.ndarray) -> tuple[float, dict]:
    """Threshold (a bin edge) with the best pooled pixel F1; predicted crack if p >= edge.
    Ties keep the highest threshold, as for tiles."""
    tp = np.cumsum(pos[::-1])[::-1]  # pixels with bin >= k
    fp = np.cumsum(neg[::-1])[::-1]
    fn = pos.sum() - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(BINS), where=(2 * tp + fp + fn) > 0)
    k = BINS - 1 - int(np.argmax(f1[::-1]))  # ties keep the highest threshold
    return k / BINS, pixel_metrics(tp[k], fp[k], fn[k])


def metrics_at(pos: np.ndarray, neg: np.ndarray, threshold: float) -> dict:
    k = int(round(threshold * BINS))
    tp, fp = pos[k:].sum(), neg[k:].sum()
    return pixel_metrics(tp, fp, pos.sum() - tp)


def model_paths(cfg: Config) -> tuple[Path, Path]:
    folder = cfg.paths.outputs / "models"
    return folder / "smallunet.pt", folder / "smallunet.json"


def train_seg(cfg: Config) -> dict:
    started = time.perf_counter()
    s = cfg.seg
    seed_everything(cfg.seed)
    train_items = load_pairs(cfg, split_images(cfg, "train"))
    val_items = load_pairs(cfg, split_images(cfg, "val"))
    model = SmallUNet(s.levels, s.base_channels)
    loader = DataLoader(
        CropDataset(train_items, s.crop_size, s.crops_per_image),
        batch_size=s.batch_size,
        shuffle=True,
        generator=torch_generator(cfg.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=s.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    best = {"f1": -1.0}
    history, stale = [], 0
    for epoch in range(1, s.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + s.dice_weight * dice_loss(logits, y)
            loss.backward()
            optimizer.step()
            total, n = total + loss.item() * len(x), n + len(x)
        threshold, val = best_pixel_threshold(*histograms(model, val_items))
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total / n, 5),
                "val_f1": round(val["f1"], 5),
                "val_iou": round(val["iou"], 5),
                "threshold": threshold,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        print(f"epoch {epoch}: loss {total / n:.4f}  val F1 {val['f1']:.4f}  IoU {val['iou']:.4f}")
        if val["f1"] > best["f1"]:
            best = {"f1": val["f1"], "epoch": epoch, "threshold": threshold, "val": val}
            best["weights"] = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= s.patience:
                break

    weights_path, meta_path = model_paths(cfg)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best.pop("weights"), weights_path)
    meta = {
        "arch": "smallunet",
        "levels": s.levels,
        "base_channels": s.base_channels,
        "parameters": sum(p.numel() for p in model.parameters()),
        "threshold": best["threshold"],
        "best_epoch": best["epoch"],
        "config_digest": config_digest(cfg),
        "git_commit": git_commit(),
        "model_version": f"smallunet-{file_digest(weights_path)[:12]}",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    report = {
        **meta,
        "val": best["val"],
        "history": history,
        "train_images": len(train_items),
        "val_images": len(val_items),
        "runtime_s": round(time.perf_counter() - started, 1),
        "hardware": hardware(),
    }
    (cfg.paths.reports / "seg_train_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def load_seg_model(cfg: Config) -> tuple[SmallUNet, dict]:
    weights_path, meta_path = model_paths(cfg)
    if not weights_path.exists():
        raise FileNotFoundError(f"no segmentation model at {weights_path.as_posix()}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = SmallUNet(meta["levels"], meta["base_channels"])
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.eval(), meta


def baseline_counts(cfg: Config, items) -> dict:
    tp = fp = fn = 0
    for rgb, mask in items:
        pred = crack_map(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cfg.baseline).astype(bool)
        tp += int(np.sum(pred & mask))
        fp += int(np.sum(pred & ~mask))
        fn += int(np.sum(~pred & mask))
    return pixel_metrics(tp, fp, fn)


def seg_results_path(cfg: Config) -> Path:
    return cfg.paths.reports / "seg_test_results.json"


def evaluate_seg(cfg: Config, split: str, final: bool, config_file: Path) -> dict:
    if (split == "test") != final:
        raise SegRefused("the test split runs only with --final, and --final only on test")
    model, meta = load_seg_model(cfg)
    if final:
        problems = []
        if seg_results_path(cfg).exists():
            problems.append("final segmentation results already exist; the test split runs once")
        if meta["config_digest"] != config_digest(cfg):
            problems.append("config changed since the segmentation model was trained")
        if file_sha256(manifest_path(cfg, "test")) != read_test_lock(cfg):
            problems.append("test manifest does not match the committed test lock")
        problems += git_problems([config_file, test_lock_path(cfg)])
        if problems:
            raise SegRefused("; ".join(problems))
    items = load_pairs(cfg, split_images(cfg, split))
    report = {
        "split": split,
        "images": len(items),
        "pixels": int(sum(m.size for _, m in items)),
        "crack_pixels": int(sum(m.sum() for _, m in items)),
        "smallunet": {
            "threshold": meta["threshold"],
            **metrics_at(*histograms(model, items), meta["threshold"]),
        },
        "opencv_baseline": baseline_counts(cfg, items),
        "model_version": meta["model_version"],
        "git_commit": git_commit(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    name = "seg_test_results.json" if final else f"seg_{split}_results.json"
    (cfg.paths.reports / name).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report
