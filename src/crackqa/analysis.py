"""Error analysis on saved predictions: bootstrap CI, confusion figure, worst tile grid."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from crackqa.config import Config  # noqa: E402
from crackqa.data_qa import read_image  # noqa: E402
from crackqa.metrics import confusion, prf  # noqa: E402


def read_predictions(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["prob"] = float(r["prob"])
        r["baseline_score"] = float(r["baseline_score"])
    return rows


def bootstrap_f1(
    rows: list[dict], score_key: str, threshold: float, n: int = 1000, seed: int = 0
) -> dict:
    """95% CI of F1, resampling source photos (with all their tiles) with replacement."""
    sources = sorted({r["source_id"] for r in rows})
    index = {s: i for i, s in enumerate(sources)}
    # per source: tp, fp, fn
    counts = np.zeros((len(sources), 3))
    for r in rows:
        pred, label = r[score_key] >= threshold, r["label"] == 1
        counts[index[r["source_id"]]] += [pred and label, pred and not label, label and not pred]
    rng = np.random.default_rng(seed)
    f1s = np.empty(n)
    for b in range(n):
        tp, fp, fn = counts[rng.integers(0, len(sources), len(sources))].sum(axis=0)
        f1s[b] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    tp, fp, fn = counts.sum(axis=0)
    return {
        "f1": prf(tp, fp, fn)["f1"],
        "ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "resamples": n,
        "sources": len(sources),
    }


def confusion_figure(rows: list[dict], methods: dict[str, tuple[str, float]], path: Path) -> None:
    """One normalized confusion matrix per method, counts in each cell."""
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.8))
    y = np.array([r["label"] for r in rows])
    for ax, (title, (key, thr)) in zip(np.atleast_1d(axes), methods.items(), strict=True):
        c = confusion(y, np.array([r[key] for r in rows]) >= thr)
        grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        share = grid / grid.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(grid):
            color = "white" if share[i, j] > 0.5 else "black"
            ax.text(j, i, f"{v}\n{share[i, j]:.0%}", ha="center", va="center", color=color)
        ax.set_xticks([0, 1], ["no crack", "crack"])
        ax.set_yticks([0, 1], ["no crack", "crack"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def worst_errors(rows: list[dict], threshold: float, n: int) -> tuple[list[dict], list[dict]]:
    """Most confident false positives and false negatives of the model."""
    fp = [r for r in rows if r["label"] == 0 and r["prob"] >= threshold]
    fn = [r for r in rows if r["label"] == 1 and r["prob"] < threshold]
    fp.sort(key=lambda r: -r["prob"])
    fn.sort(key=lambda r: r["prob"])
    return fp[:n], fn[:n]


def _tile(cfg: Config, tile_id: str) -> np.ndarray:
    image, coords = tile_id.rsplit("@", 1)
    y, x = (int(v) for v in coords.split(","))
    _, bgr = read_image(cfg.paths.raw / image, cv2.IMREAD_COLOR)
    s = cfg.tiling.tile_size
    return cv2.cvtColor(bgr[y : y + s, x : x + s], cv2.COLOR_BGR2RGB)


def error_grid(cfg: Config, fp: list[dict], fn: list[dict], path: Path) -> None:
    """Two rows of tiles: worst false positives on top, worst false negatives below."""
    cols = max(len(fp), len(fn), 1)
    fig, axes = plt.subplots(2, cols, figsize=(1.9 * cols, 4.4), squeeze=False)
    for row, (name, items) in enumerate((("false positive", fp), ("false negative", fn))):
        for col in range(cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(items):
                ax.imshow(_tile(cfg, items[col]["tile_id"]))
                ax.set_title(f"{chr(65 + row)}{col + 1}  p={items[col]['prob']:.2f}", fontsize=8)
        axes[row, 0].text(
            -0.15,
            0.5,
            name,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


CAUSES = (
    "shadow",
    "joint",
    "lane_paint",
    "oil_stain",
    "texture",
    "thin_crack",
    "tile_edge",  # the labelled crack only grazes the tile border
    "label",  # the mask itself looks wrong or misses a visible crack
    "other",
)


def _read_causes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["tile_id"]: r["cause"] for r in csv.DictReader(f)}


def run_analysis(cfg: Config) -> dict:
    """Bootstrap CIs, confusion figure, worst errors and image level recall on the saved test
    predictions. Causes are filled in by hand in reports/error_candidates.csv and kept on
    rerun."""
    reports = cfg.paths.reports
    results = json.loads((reports / "test_results.json").read_text(encoding="utf-8"))["results"]
    model_thr = results["resnet18"]["threshold"]
    base_thr = results["opencv_baseline"]["threshold"]
    rows = read_predictions(reports / "predictions_test.csv")
    # baseline scores are crack pixel counts / tile area but were saved to 6 decimals;
    # snap them back so scores equal to the threshold compare as in the final run
    area = cfg.tiling.tile_size**2
    for r in rows:
        r["baseline_score"] = round(r["baseline_score"] * area) / area
    for name, key, thr in (
        ("resnet18", "prob", model_thr),
        ("opencv_baseline", "baseline_score", base_thr),
    ):
        c = confusion([r["label"] for r in rows], [r[key] >= thr for r in rows])
        if any(c[k] != results[name][k] for k in c):
            raise ValueError(f"saved predictions do not reproduce the {name} test counts: {c}")
    n = cfg.analysis.bootstrap_resamples

    confusion_figure(
        rows,
        {"ResNet18": ("prob", model_thr), "OpenCV baseline": ("baseline_score", base_thr)},
        reports / "figures" / "confusion_test.png",
    )

    fp, fn = worst_errors(rows, model_thr, cfg.analysis.worst_errors)
    candidates = reports / "error_candidates.csv"
    causes = _read_causes(candidates)
    picked = [
        {
            "id": f"{'A' if kind == 'false_positive' else 'B'}{i + 1}",
            "tile_id": r["tile_id"],
            "kind": kind,
            "prob": round(r["prob"], 4),
            "cause": causes.get(r["tile_id"], ""),
        }
        for kind, items in (("false_positive", fp), ("false_negative", fn))
        for i, r in enumerate(items)
    ]
    with open(candidates, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(picked[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(picked)
    # tiles are dataset pixels: the grid stays local (see DATA.md)
    error_grid(cfg, fp, fn, cfg.paths.outputs / "figures" / "error_grid.png")

    by_image: dict[str, bool] = {}
    for r in rows:
        image = r["tile_id"].rsplit("@", 1)[0]
        by_image[image] = by_image.get(image, False) or r["prob"] >= model_thr

    cause_counts: dict[str, dict[str, int]] = {}
    for p in picked:
        if p["cause"]:
            cause_counts.setdefault(p["kind"], {}).setdefault(p["cause"], 0)
            cause_counts[p["kind"]][p["cause"]] += 1

    report = {
        "bootstrap": {
            "resnet18": bootstrap_f1(rows, "prob", model_thr, n, cfg.seed),
            "opencv_baseline": bootstrap_f1(rows, "baseline_score", base_thr, n, cfg.seed),
        },
        "image_level": {
            "rule": "a crop is flagged when any of its tiles reaches the threshold",
            "crops": len(by_image),
            "flagged": sum(by_image.values()),
            "recall": sum(by_image.values()) / len(by_image),
            "note": "every CRACK500 crop contains a crack, so only recall is defined",
        },
        "worst_errors": picked,
        "cause_counts": cause_counts,
    }
    (reports / "error_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report
