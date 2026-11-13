"""A trained tile classifier and the OpenCV baseline on the V2 val tiles, per dataset and
pooled. One row per scored model is kept in reports/ablation.json and reports/ablation.md.

Only val is scored here; the V2 test split is scored once, later, behind its lock.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crackqa.baseline import load_val_threshold, tile_scores
from crackqa.config import Config
from crackqa.datasets import labels_of, load_tiles
from crackqa.metrics import summarize
from crackqa.models import load_model, predict
from crackqa.split import file_sha256
from crackqa.tiling import AMBIGUOUS
from crackqa.v2data import manifest_v2_path, read_v2_rows

CHUNK = 4096  # tiles in memory at once
POOLED = "pooled"


def ablation_path(cfg: Config) -> Path:
    return cfg.paths.reports / "ablation.json"


def _summary(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    out = summarize(y, scores, threshold)
    negatives = out["fp"] + out["tn"]
    out["false_alarm_rate"] = round(out["fp"] / negatives, 4) if negatives else None
    if not out["positives"]:  # no crack tile: only the false alarm rate means anything
        out.update(precision=None, recall=None, f1=None)
    return out


def score_rows(cfg: Config, rows: list[dict], model) -> tuple[np.ndarray, np.ndarray]:
    """Model probability and baseline score per tile, a chunk of tiles at a time."""
    probs, base = [], []
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start : start + CHUNK]
        probs.append(predict(model, load_tiles(chunk, Path("."), cfg.tiling.tile_size)))
        base.append(tile_scores(chunk, cfg, root=Path(".")))
    if not probs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(probs), np.concatenate(base)


def threshold_at_recall(y: np.ndarray, scores: np.ndarray, recall: float) -> float | None:
    """Highest threshold at which at least `recall` of the crack tiles are flagged."""
    positives = np.sort(np.asarray(scores)[np.asarray(y, bool)])[::-1]
    if not len(positives):
        return None
    k = max(1, int(np.ceil(recall * len(positives) - 1e-9)))
    return float(positives[k - 1])


def _at(y: np.ndarray, scores: np.ndarray, threshold: float | None) -> dict:
    """Recall and false alarm rate at one threshold (None where undefined)."""
    if threshold is None:
        return {"recall": None, "false_alarm_rate": None}
    y = np.asarray(y, bool)
    flagged = np.asarray(scores) >= threshold
    pos, neg = int(y.sum()), int((~y).sum())
    return {
        "recall": round(int((flagged & y).sum()) / pos, 4) if pos else None,
        "false_alarm_rate": round(int((flagged & ~y).sum()) / neg, 4) if neg else None,
    }


def evaluate_val(cfg: Config, name: str, outputs: Path) -> dict:
    manifest = manifest_v2_path(cfg, "val")
    rows = [r for r in read_v2_rows(manifest) if r["label"] != AMBIGUOUS]
    model, meta = load_model(outputs)
    probs, base = score_rows(cfg, rows, model)
    y = labels_of(rows)
    datasets = np.array([r["dataset"] for r in rows])
    base_threshold = load_val_threshold(cfg)
    recall = cfg.v2_train.compare_recall
    matched = threshold_at_recall(y, probs, recall)
    results = {}
    for dataset in [POOLED, *sorted(set(datasets))]:
        idx = np.ones(len(rows), bool) if dataset == POOLED else datasets == dataset
        results[dataset] = {
            "model": _summary(y[idx], probs[idx], meta["threshold"]),
            "model_at_compare_recall": _at(y[idx], probs[idx], matched),
            "opencv_baseline": _summary(y[idx], base[idx], base_threshold),
        }
    row = {
        "name": name,
        "model_version": meta["model_version"],
        "threshold": meta["threshold"],
        "compare_recall": recall,
        "compare_threshold": matched,
        "split": "val",
        "label_rule": cfg.v2.label_rule,
        "centerline_min_px": cfg.v2.centerline_min_px,
        "val_manifest_sha256": file_sha256(manifest),
        "results": results,
    }
    table = [r for r in read_ablation(cfg) if r["name"] != name] + [row]
    write_ablation(cfg, table)
    return row


def read_ablation(cfg: Config) -> list[dict]:
    path = ablation_path(cfg)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def ablation_markdown(table: list[dict]) -> str:
    lines = [
        "# Ablation on V2 val",
        "",
        "Every row is scored on the V2 val tiles (ambiguous tiles left out), each model at its "
        "own threshold and the OpenCV baseline at its CRACK500 val threshold. False alarm rate: "
        "no-crack tiles flagged / all no-crack tiles. SDNET2018 val holds no crack tiles, so "
        "only its false alarm rate is defined. Val also chooses each model's epoch and "
        "threshold (V1: on CRACK500 val), so these numbers are optimistic; the V2 test split "
        "reports once.",
        "",
    ]
    lines += _summary_table(table)
    for row in table:
        lines += [
            f"## {row['name']}",
            "",
            f"Model `{row['model_version']}`, threshold {row['threshold']:.4f}; label rule "
            f"`{row['label_rule']}` ({row['centerline_min_px']} px); val manifest "
            f"`{row['val_manifest_sha256'][:12]}`.",
            "",
            "| Dataset | Tiles | Crack | Precision | Recall | F1 | PR-AUC | False alarm rate | "
            "Baseline F1 | Baseline false alarm rate |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for dataset, r in row["results"].items():
            m, b = r["model"], r["opencv_baseline"]
            cells = [
                m["n"],
                m["positives"],
                m["precision"],
                m["recall"],
                m["f1"],
                m["pr_auc"],
                m["false_alarm_rate"],
                b["f1"],
                b["false_alarm_rate"],
            ]
            lines.append(f"| {dataset} | " + " | ".join(_fmt(c) for c in cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _matched(result: dict) -> str:
    m = result.get("model_at_compare_recall", {"recall": None, "false_alarm_rate": None})
    return f"{_fmt(m['recall'])} / {_fmt(m['false_alarm_rate'])}"


def _summary_table(table: list[dict]) -> list[str]:
    """One line per model: pooled scores, then each dataset at the matched recall threshold."""
    if not table:
        return []
    datasets = [d for d in table[0]["results"] if d != POOLED]
    recall = table[-1]["compare_recall"]
    lines = [
        "## Summary",
        "",
        f"Columns after PR-AUC: recall / false alarm rate at the threshold where each model "
        f"flags {recall:.1%} of all crack tiles (V1's pooled recall), so false alarms are "
        "compared at equal recall.",
        "",
        "| Model | Pooled F1 | Pooled PR-AUC | Pooled | " + " | ".join(datasets) + " |",
        "|---|---|---|---|" + "---|" * len(datasets),
    ]
    for row in table:
        res = row["results"]
        pooled = res[POOLED]["model"]
        cells = [_fmt(pooled["f1"]), _fmt(pooled["pr_auc"]), _matched(res[POOLED])]
        cells += [_matched(res[d]) if d in res else "-" for d in datasets]
        lines.append(f"| {row['name']} | " + " | ".join(cells) + " |")
    return lines + [""]


def write_ablation(cfg: Config, table: list[dict]) -> None:
    path = ablation_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8", newline="\n")
    path.with_suffix(".md").write_text(ablation_markdown(table), encoding="utf-8", newline="\n")
