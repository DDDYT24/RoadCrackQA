"""The single run on the locked V2 test split: V1 and the chosen V2 experiment on the same tiles.

Writes reports/test2_results.json and .md, reports/predictions_test2.csv and two figures.
Refused when results exist, when the config differs from the one the V2 model was trained
with, when the test manifest does not match reports/test2_lock.txt, when the config, lock or
V2 training report is not committed, or when a model file is not the one on record.
`rehearse` runs the same code on V2 val and writes only to v2_train.outputs/rehearsal/.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from crackqa.analysis import confusion_figure, error_grid, worst_errors
from crackqa.baseline import load_val_threshold, tile_scores
from crackqa.config import Config, config_digest
from crackqa.data_qa import read_image
from crackqa.datasets import labels_of, load_tiles
from crackqa.evaluate import FinalRunRefused
from crackqa.evaluate import results_path as v1_results_path
from crackqa.models import load_model, predict
from crackqa.runinfo import git_commit, git_problems
from crackqa.split import file_sha256
from crackqa.tiling import AMBIGUOUS, NO_CRACK
from crackqa.v2data import manifest_v2_path, read_v2_rows, test2_lock_path
from crackqa.v2eval import CHUNK, POOLED, _at, _fmt, _summary, threshold_at_recall
from crackqa.v2train import experiment_dir, report_path

# No pixels of these datasets are shown in figures (DATA.md: GAPS384 terms).
NO_FIGURE = ("GAPS384",)
METHODS = ("v1", "v2", "opencv_baseline")


def results_path(cfg: Config) -> Path:
    return cfg.paths.reports / "test2_results.json"


def read_lock(cfg: Config) -> str:
    return test2_lock_path(cfg).read_text(encoding="utf-8").split()[0]


def final_problems(
    cfg: Config, config_file: Path, experiment: str, v1_meta: dict, v2_meta: dict
) -> list[str]:
    problems = []
    if results_path(cfg).exists():
        problems.append("V2 test results already exist; the V2 test split runs once")
    if v2_meta.get("config_digest") != config_digest(cfg):
        problems.append("config changed since the V2 model was trained")
    train_report = report_path(cfg, experiment)
    if not train_report.exists():
        problems.append(f"no training report {train_report.as_posix()}")
    elif json.loads(train_report.read_text(encoding="utf-8"))["model_version"] != v2_meta.get(
        "model_version"
    ):
        problems.append("the V2 model file is not the one in its training report")
    v1_results = v1_results_path(cfg)
    if not v1_results.exists():
        problems.append("no V1 test results; the V1 model on record is unknown")
    elif json.loads(v1_results.read_text(encoding="utf-8"))["model_version"] != v1_meta.get(
        "model_version"
    ):
        problems.append("the V1 model file is not the one in reports/test_results.json")
    lock = test2_lock_path(cfg)
    if not lock.exists():
        problems.append("no V2 test lock; run crackqa v2-split")
    elif file_sha256(manifest_v2_path(cfg, "test")) != read_lock(cfg):
        problems.append("V2 test manifest does not match the committed test lock")
    problems += git_problems([config_file, lock, train_report])
    return problems


def score(cfg: Config, rows: list[dict], v1, v2) -> dict[str, np.ndarray]:
    """V1 and V2 probability and baseline score per tile, a chunk of tiles at a time."""
    out = {m: [] for m in METHODS}
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start : start + CHUNK]
        tiles = load_tiles(chunk, Path("."), cfg.tiling.tile_size)
        out["v1"].append(predict(v1, tiles))
        out["v2"].append(predict(v2, tiles))
        out["opencv_baseline"].append(tile_scores(chunk, cfg, root=Path(".")))
    return {m: np.concatenate(v) if v else np.zeros(0) for m, v in out.items()}


def _photo(r: dict) -> str:
    return f"{r['dataset']}/{r['source_id']}"


def bootstrap(rows: list[dict], flags: dict[str, np.ndarray], n: int, seed: int) -> dict[str, dict]:
    """95% CIs of F1, recall and false alarm rate for each method and of V2 minus V1.

    Source photos are resampled with all their tiles, within each dataset (so the dataset mix
    stays as in test), the same draw for every method."""
    photos = sorted({_photo(r) for r in rows})
    index = {p: i for i, p in enumerate(photos)}
    which = np.array([index[_photo(r)] for r in rows])
    y = labels_of(rows).astype(bool)
    counts = {}  # per photo: tp, fp, fn, tn
    for m, f in flags.items():
        c = np.zeros((len(photos), 4))
        for k, cell in enumerate((f & y, f & ~y, ~f & y, ~f & ~y)):
            c[:, k] = np.bincount(which, weights=cell, minlength=len(photos))
        counts[m] = c
    strata = {}
    for p, i in index.items():
        strata.setdefault(p.split("/", 1)[0], []).append(i)
    strata = [np.array(v) for v in strata.values()]

    def measures(c: np.ndarray) -> np.ndarray:
        tp, fp, fn, tn = c
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.array([2 * tp / (2 * tp + fp + fn), tp / (tp + fn), fp / (fp + tn)])

    rng = np.random.default_rng(seed)
    draws = {m: np.empty((n, 3)) for m in flags}
    for b in range(n):
        pick = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
        for m, c in counts.items():
            draws[m][b] = measures(c[pick].sum(axis=0))
    draws["v2_minus_v1"] = draws["v2"] - draws["v1"]
    names = ("f1", "recall", "false_alarm_rate")
    out = {}
    for m, d in draws.items():
        point = (
            measures(counts["v2"].sum(axis=0)) - measures(counts["v1"].sum(axis=0))
            if m == "v2_minus_v1"
            else measures(counts[m].sum(axis=0))
        )
        out[m] = {
            name: {
                "value": round(float(point[k]), 4),
                "ci95": [
                    round(float(np.nanpercentile(d[:, k], 2.5)), 4),
                    round(float(np.nanpercentile(d[:, k], 97.5)), 4),
                ],
            }
            for k, name in enumerate(names)
        }
    return {**out, "resamples": n, "photos": len(photos), "strata": "dataset"}


def photo_level(all_rows: list[dict], rows: list[dict], flags: dict[str, np.ndarray]) -> dict:
    """A source photo is flagged when any of its labelled tiles is. Crack-free photos are those
    with no crack and no ambiguous tile at all."""
    crack_free = {_photo(r) for r in all_rows} - {
        _photo(r) for r in all_rows if r["label"] != NO_CRACK
    }
    with_crack = {_photo(r) for r in rows if r["label"] == 1}
    out = {
        "rule": "a source photo is flagged when any of its labelled tiles reaches the threshold",
        "crack_free_photos": len(crack_free),
        "photos_with_crack": len(with_crack),
    }
    for m, f in flags.items():
        flagged = {_photo(r) for r, hit in zip(rows, f, strict=True) if hit}
        out[m] = {
            "crack_free_flagged": len(crack_free & flagged),
            "false_alarm_rate": round(len(crack_free & flagged) / len(crack_free), 4)
            if crack_free
            else None,
            "with_crack_flagged": len(with_crack & flagged),
            "recall": round(len(with_crack & flagged) / len(with_crack), 4) if with_crack else None,
        }
    return out


def _panel(cfg: Config, r: dict, label: str, context: int = 32) -> np.ndarray:
    """The tile boxed in yellow with `context` px around it and the labelled crack in green."""
    s, y, x = cfg.tiling.tile_size, r["y"], r["x"]
    _, bgr = read_image(Path(r["image"]), cv2.IMREAD_COLOR)
    pad = ((context, context), (context, context))
    view = np.pad(bgr, (*pad, (0, 0)))[y : y + s + 2 * context, x : x + s + 2 * context].copy()
    if r["mask"]:
        _, mask = read_image(Path(r["mask"]), cv2.IMREAD_GRAYSCALE)
        ds = cfg.extra_data.get(r["dataset"])
        crack = (mask == 0) if ds is not None and ds.invert_mask else (mask > 0)
        m = np.pad(crack, pad)[y : y + s + 2 * context, x : x + s + 2 * context]
        outline, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(view, outline, -1, (0, 220, 0), 1)
    cv2.rectangle(view, (context, context), (context + s - 1, context + s - 1), (0, 210, 255), 2)
    for color, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(view, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, width)
    return view


def compare(rows: list[dict], scores: dict, thresholds: dict) -> dict:
    y = labels_of(rows)
    datasets = np.array([r["dataset"] for r in rows])
    v1_recall = _at(y, scores["v1"], thresholds["v1"])["recall"]
    matched = threshold_at_recall(y, scores["v2"], v1_recall) if v1_recall else None
    out = {}
    for dataset in [POOLED, *sorted(set(datasets))]:
        idx = np.ones(len(rows), bool) if dataset == POOLED else datasets == dataset
        out[dataset] = {m: _summary(y[idx], scores[m][idx], thresholds[m]) for m in METHODS}
        out[dataset]["v2_at_v1_recall"] = _at(y[idx], scores["v2"][idx], matched)
    return {"results": out, "v2_threshold_at_v1_recall": matched}


def acceptance(pooled: dict) -> dict:
    v1, v2 = pooled["v1"], pooled["v2"]
    checks = {
        "lower_false_alarm_rate": v2["false_alarm_rate"] < v1["false_alarm_rate"],
        "equal_or_higher_recall": v2["recall"] >= v1["recall"],
        "higher_f1": v2["f1"] > v1["f1"],
    }
    return {**checks, "met": all(checks.values())}


def markdown(report: dict) -> str:
    res, boot, photo = report["results"], report["bootstrap"], report["photo_level"]
    acc = report["acceptance"]
    c = report["counts"]
    lines = [
        f"# V1 and V2 on the V2 {report['split']} split",
        "",
        f"{c['tiles']} labelled tiles ({c['crack']} crack) from {', '.join(c['datasets'])}; "
        "ambiguous tiles left out. Each model at its own val threshold (V1 "
        f"{report['thresholds']['v1']:.4f}, V2 {report['thresholds']['v2']:.4f}), the OpenCV "
        f"baseline at its CRACK500 val threshold. False alarm rate: no-crack tiles flagged / "
        "all no-crack tiles. SDNET2018 has no crack tiles, so only its false alarm rate is "
        "defined; it holds most of the no-crack tiles.",
        "",
        f"Acceptance (V2 lower false alarm rate, equal or higher recall, higher F1, pooled): "
        f"**{'met' if acc['met'] else 'not met'}** (false alarm rate "
        f"{'lower' if acc['lower_false_alarm_rate'] else 'not lower'}, recall "
        f"{'equal or higher' if acc['equal_or_higher_recall'] else 'lower'}, F1 "
        f"{'higher' if acc['higher_f1'] else 'not higher'}).",
        "",
        "| Dataset | Tiles | Crack | Method | Precision | Recall | F1 | PR-AUC | "
        "False alarm rate | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset, r in res.items():
        for m in METHODS:
            s = r[m]
            cells = [s["n"], s["positives"], m, s["precision"], s["recall"], s["f1"]]
            cells += [s["pr_auc"], s["false_alarm_rate"], s["tp"], s["fp"], s["fn"], s["tn"]]
            lines.append(f"| {dataset} | " + " | ".join(_fmt(v) for v in cells) + " |")
    matched = report["v2_threshold_at_v1_recall"]
    lines += [
        "",
        "## V2 at V1's recall",
        "",
        f"V2 at threshold {_fmt(matched)}, where its pooled recall reaches V1's. This threshold "
        f"is read off the {report['split']} tiles, so it is a comparison only, not an "
        "operating point.",
        "",
        "| Dataset | Recall | False alarm rate |",
        "|---|---|---|",
    ]
    for dataset, r in res.items():
        m = r["v2_at_v1_recall"]
        lines.append(f"| {dataset} | {_fmt(m['recall'])} | {_fmt(m['false_alarm_rate'])} |")
    lines += [
        "",
        "## 95% bootstrap intervals, pooled",
        "",
        f"{boot['resamples']} resamples of the {boot['photos']} source photos, within each "
        "dataset, the same draw for every method.",
        "",
        "| Method | F1 | Recall | False alarm rate |",
        "|---|---|---|---|",
    ]
    for m in (*METHODS, "v2_minus_v1"):
        b = boot[m]
        cells = [
            f"{b[k]['value']:.3f} ({b[k]['ci95'][0]:.3f} to {b[k]['ci95'][1]:.3f})"
            for k in ("f1", "recall", "false_alarm_rate")
        ]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Source photos",
        "",
        f"{photo['rule'].capitalize()}. Crack-free photos: {photo['crack_free_photos']}; photos "
        f"with a crack tile: {photo['photos_with_crack']}.",
        "",
        "| Method | Crack-free photos flagged | Photos with crack flagged |",
        "|---|---|---|",
    ]
    for m in METHODS:
        p = photo[m]
        lines.append(
            f"| {m} | {p['crack_free_flagged']} ({_fmt(p['false_alarm_rate'])}) | "
            f"{p['with_crack_flagged']} ({_fmt(p['recall'])}) |"
        )
    lines += [
        "",
        f"Figures: `{report['figures']['confusion']}`, `{report['figures']['errors']}` (V2's "
        f"most confident errors; no {', '.join(NO_FIGURE)} tiles shown).",
        "",
        f"V1 `{report['model_versions']['v1']}`, V2 `{report['model_versions']['v2']}` "
        f"({report['experiment']}); config `{report['config_digest'][:12]}`; "
        + (f"test lock `{report['test_lock'][:12]}`; " if report.get("test_lock") else "")
        + f"commit `{report['git_commit']}`.",
    ]
    return "\n".join(lines) + "\n"


def run_final(cfg: Config, experiment: str, config_file: Path, rehearse: bool = False) -> dict:
    v1, v1_meta = load_model(cfg.paths.outputs)
    v2, v2_meta = load_model(experiment_dir(cfg, experiment))
    split = "val" if rehearse else "test"
    if not rehearse:
        problems = final_problems(cfg, config_file, experiment, v1_meta, v2_meta)
        if problems:
            raise FinalRunRefused("; ".join(problems))
    out = cfg.v2_train.outputs / "rehearsal" if rehearse else cfg.paths.reports
    figures = out / "figures"

    all_rows = read_v2_rows(manifest_v2_path(cfg, split))
    rows = [r for r in all_rows if r["label"] != AMBIGUOUS]
    scores = score(cfg, rows, v1, v2)
    thresholds = {
        "v1": v1_meta["threshold"],
        "v2": v2_meta["threshold"],
        "opencv_baseline": load_val_threshold(cfg),
    }
    flags = {m: scores[m] >= thresholds[m] for m in METHODS}
    table = compare(rows, scores, thresholds)

    y = labels_of(rows)
    keyed = [
        {**r, "v1": float(a), "v2": float(b), "prob": float(b)}
        for r, a, b in zip(rows, scores["v1"], scores["v2"], strict=True)
    ]
    confusion_path = figures / f"confusion_{split}2.png"
    confusion_figure(
        keyed,
        {
            f"V1 ({split})": ("v1", thresholds["v1"]),
            f"V2 {experiment} ({split})": ("v2", thresholds["v2"]),
        },
        confusion_path,
    )
    shown = [r for r in keyed if r["dataset"] not in NO_FIGURE]
    fp, fn = worst_errors(shown, thresholds["v2"], cfg.analysis.worst_errors)
    errors_path = figures / f"error_grid_{split}2.jpg"
    error_grid(cfg, fp, fn, errors_path, panel=lambda r, label: _panel(cfg, r, label))

    report = {
        "split": split,
        "experiment": experiment,
        "counts": {
            "tiles": len(rows),
            "crack": int(y.sum()),
            "datasets": sorted({r["dataset"] for r in rows}),
        },
        "thresholds": thresholds,
        **table,
        "acceptance": acceptance(table["results"][POOLED]),
        "bootstrap": bootstrap(rows, flags, cfg.analysis.bootstrap_resamples, cfg.seed),
        "photo_level": photo_level(all_rows, rows, flags),
        "worst_errors": [
            {
                "id": f"{tag}{i + 1}",
                "tile_id": r["tile_id"],
                "dataset": r["dataset"],
                "prob": round(r["prob"], 4),
            }
            for tag, items in (("A", fp), ("B", fn))
            for i, r in enumerate(items)
        ],
        "figures": {
            "confusion": confusion_path.relative_to(out.parent).as_posix(),
            "errors": errors_path.relative_to(out.parent).as_posix(),
        },
        "model_versions": {"v1": v1_meta["model_version"], "v2": v2_meta["model_version"]},
        "config_digest": config_digest(cfg),
        "git_commit": git_commit(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if not rehearse:
        report["test_lock"] = read_lock(cfg)
    out.mkdir(parents=True, exist_ok=True)
    stem = "test2" if not rehearse else "val2_rehearsal"
    with open(out / f"predictions_{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["tile_id", "dataset", "source_id", "label", *METHODS])
        for k, r in enumerate(rows):
            # full precision: scores equal to a threshold must stay equal
            writer.writerow(
                [r["tile_id"], r["dataset"], r["source_id"], r["label"]]
                + [repr(float(scores[m][k])) for m in METHODS]
            )
    (out / f"{stem}_results.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    # the json is written last: its presence is what refuses a second run
    (out / f"{stem}_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report
