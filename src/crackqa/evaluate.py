"""Evaluate the model against the baselines. The test split runs once, behind a lock."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from crackqa.baseline import load_val_threshold, tile_scores
from crackqa.config import Config, config_digest
from crackqa.datasets import labels_of, load_tiles
from crackqa.metrics import summarize
from crackqa.models import load_model, predict
from crackqa.runinfo import git_commit, git_problems
from crackqa.split import file_sha256, manifest_path, read_test_lock, test_lock_path
from crackqa.train import labelled_rows


class FinalRunRefused(RuntimeError):
    """The final test run's preconditions do not hold."""


def results_path(cfg: Config) -> Path:
    return cfg.paths.reports / "test_results.json"


def final_problems(cfg: Config, config_file: Path, meta: dict) -> list[str]:
    problems = []
    if results_path(cfg).exists():
        problems.append("final test results already exist; the test split runs once")
    if meta.get("config_digest") != config_digest(cfg):
        problems.append("config changed since the model was trained")
    if not test_lock_path(cfg).exists():
        problems.append("no test lock; run crackqa split")
    elif file_sha256(manifest_path(cfg, "test")) != read_test_lock(cfg):
        problems.append("test manifest does not match the committed test lock")
    problems += git_problems([config_file, test_lock_path(cfg)])
    return problems


def compare(cfg: Config, split: str, model, meta: dict) -> tuple[dict, list[dict], dict]:
    """Model, OpenCV baseline and majority class on the same labelled tiles."""
    rows = labelled_rows(cfg, split)
    y = labels_of(rows)
    probs = predict(model, load_tiles(rows, cfg.paths.raw, cfg.tiling.tile_size))
    base = tile_scores(rows, cfg)
    results = {
        "resnet18": summarize(y, probs, meta["threshold"]),
        "opencv_baseline": summarize(y, base, load_val_threshold(cfg)),
        # the majority class is "no crack": it never flags a tile
        "majority_class": summarize(y, np.zeros(len(y)), 0.5),
    }
    preds = [
        {
            "tile_id": r["tile_id"],
            "source_id": r["source_id"],
            "label": int(label),
            "prob": float(p),  # full precision: scores equal to a threshold must stay equal
            "baseline_score": float(b),
        }
        for r, label, p, b in zip(rows, y, probs, base, strict=True)
    ]
    return results, preds, {"tiles": len(rows), "crack": int(y.sum())}


def markdown(report: dict) -> str:
    lines = [
        f"# Tile results: {report['split']}",
        "",
        f"{report['counts']['tiles']} labelled tiles, {report['counts']['crack']} crack "
        f"({report['counts']['crack'] / report['counts']['tiles']:.1%}); ambiguous tiles "
        "excluded. Thresholds were chosen on val.",
        "",
        "| Method | Threshold | Precision | Recall | F1 | PR-AUC | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report["results"].items():
        auc = "-" if name == "majority_class" or m["pr_auc"] is None else f"{m['pr_auc']:.3f}"
        lines.append(
            f"| {name} | {m['threshold']:.4g} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {auc} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )
    lines += ["", f"Model: `{report['model_version']}`; commit: `{report['git_commit']}`."]
    return "\n".join(lines) + "\n"


def _write(cfg: Config, stem: str, report: dict) -> None:
    out = cfg.paths.reports
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out / f"{stem}.md").write_text(markdown(report), encoding="utf-8", newline="\n")


def evaluate(cfg: Config, split: str, final: bool, config_file: Path) -> dict:
    if split == "test" and not final:
        raise FinalRunRefused("the test split only runs with --final")
    if split != "test" and final:
        raise FinalRunRefused("--final is only for the test split")
    model, meta = load_model(cfg.paths.outputs)
    if final:
        problems = final_problems(cfg, config_file, meta)
        if problems:
            raise FinalRunRefused("; ".join(problems))

    results, preds, counts = compare(cfg, split, model, meta)
    report = {
        "split": split,
        "counts": counts,
        "results": results,
        "model_version": meta["model_version"],
        "config_digest": config_digest(cfg),
        "git_commit": git_commit(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if final:
        report["test_lock"] = read_test_lock(cfg)
        _write(cfg, "test_results", report)
        path = cfg.paths.reports / "predictions_test.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(preds[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(preds)
    else:
        _write(cfg, f"{split}_results", report)
    return report
