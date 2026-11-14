"""The "not a road" gate: a ResNet18 tile classifier, road or not road, and a photo threshold.

A photo's road score is the mean road probability of all its tiles. `gate-split` writes the
photo lists per split (gate.processed/photos_*.csv), reports/gate_split.md and
reports/gate_test_lock.txt. `gate-train` trains on tiles of the train photos, picks the epoch
on val tiles and the photo threshold on val photos (the highest score at which
road_pass_rate of the road photos pass), and writes gate.model/models/ and
reports/gate_train.json. `gate-final` scores the locked test photos once.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from crackqa.batch import IMAGE_EXTS, score_image
from crackqa.config import Config, config_digest
from crackqa.data_qa import read_image
from crackqa.datasets import TileDataset, labels_of, load_tiles
from crackqa.evaluate import FinalRunRefused
from crackqa.models import (
    build_classifier,
    freeze_backbone,
    load_model,
    save_model,
    unfreeze_layer4,
)
from crackqa.runinfo import git_commit, git_problems, hardware
from crackqa.seed import seed_everything, torch_generator
from crackqa.split import file_sha256
from crackqa.tiling import tile_grid
from crackqa.train import _run_stage, class_weights
from crackqa.v2data import manifest_v2_path, read_v2_rows

COLUMNS = ["photo", "dataset", "road", "split", "height", "width"]
FULL = "CRACK500 full photos"
SPLITS = ("train", "val", "test")


def photos_path(cfg: Config, split: str) -> Path:
    return cfg.gate.processed / f"photos_{split}.csv"


def lock_path(cfg: Config) -> Path:
    return cfg.paths.reports / "gate_test_lock.txt"


def train_report_path(cfg: Config) -> Path:
    return cfg.paths.reports / "gate_train.json"


def results_path(cfg: Config) -> Path:
    return cfg.paths.reports / "gate_test_results.json"


# --- photo lists ------------------------------------------------------------------------


def _photo(path: str, dataset: str, road: int, split: str) -> dict | None:
    """A photo row with its size, or None when it cannot be decoded."""
    _, img = read_image(Path(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    return {
        "photo": path,
        "dataset": dataset,
        "road": road,
        "split": split,
        "height": h,
        "width": w,
    }


def _images(folder: Path, exclude: tuple[str, ...] = ()) -> list[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.stem.lower().endswith("_mask")
        and p.relative_to(folder).parts[0] not in exclude
    )


def _cap(items: list, n: int, rng: np.random.Generator) -> list:
    if len(items) <= n:
        return items
    return [items[i] for i in sorted(rng.choice(len(items), size=n, replace=False))]


def build_photos(cfg: Config) -> tuple[list[dict], int]:
    """Every gate photo with its split, and how many could not be decoded."""
    g, rng = cfg.gate, np.random.default_rng(cfg.seed)
    candidates = []  # (path, dataset, road, split)
    for split in SPLITS:
        images = {}
        for r in read_v2_rows(manifest_v2_path(cfg, split)):
            images.setdefault(r["image"], r["dataset"])
        for dataset in sorted(set(images.values())):
            paths = sorted(p for p, d in images.items() if d == dataset)
            if split != "train":
                paths = _cap(paths, g.max_photos_per_dataset, rng)
            candidates += [(p, dataset, 1, split) for p in paths]
        if split in g.full_photos:
            candidates += [(p.as_posix(), FULL, 1, split) for p in _images(g.full_photos[split])]
    for name, source in g.non_road.items():
        files = _images(source.images, source.exclude_dirs)
        order = rng.permutation(len(files))
        n_train = round(g.split["train"] * len(files))
        n_val = round(g.split["val"] * len(files))
        for rank, i in enumerate(order):
            split = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
            candidates.append((files[i].as_posix(), name, 0, split))
    photos = [p for p in (_photo(*c) for c in candidates) if p is not None]
    return photos, len(candidates) - len(photos)


def read_photos(cfg: Config, split: str) -> list[dict]:
    with open(photos_path(cfg, split), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("road", "height", "width"):
            r[key] = int(r[key])
    return rows


def _split_markdown(photos: list[dict], unreadable: int, cfg: Config) -> str:
    counts = Counter((p["dataset"], p["road"], p["split"]) for p in photos)
    lines = [
        "# Gate photos",
        "",
        "Road photos keep their V2 split (val and test hold at most "
        f"{cfg.gate.max_photos_per_dataset} per dataset, chosen with the seed); CRACK500 full "
        "photos come from the official val and test photo folders; non-road photos are split "
        f"by file ({', '.join(f'{k} {v:.0%}' for k, v in cfg.gate.split.items())}). "
        f"Unreadable files left out: {unreadable}.",
        "",
        "| Dataset | Road | Train | Val | Test |",
        "|---|---|---|---|---|",
    ]
    for dataset, road in sorted({(d, r) for d, r, _ in counts}, key=lambda k: (-k[1], k[0])):
        cells = [str(counts.get((dataset, road, s), 0)) for s in SPLITS]
        lines.append(f"| {dataset} | {'yes' if road else 'no'} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def make_gate_split(cfg: Config) -> dict:
    """Write the photo lists, reports/gate_split.md and reports/gate_test_lock.txt."""
    photos, unreadable = build_photos(cfg)
    seen: dict[str, str] = {}
    for p in photos:
        if seen.setdefault(p["photo"], p["split"]) != p["split"]:
            raise AssertionError(f"{p['photo']} is in two splits")
    cfg.gate.processed.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = sorted((p for p in photos if p["split"] == split), key=lambda p: p["photo"])
        with open(photos_path(cfg, split), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports / "gate_split.md").write_text(
        _split_markdown(photos, unreadable, cfg), encoding="utf-8", newline="\n"
    )
    lock = file_sha256(photos_path(cfg, "test"))
    lock_path(cfg).write_text(f"{lock}  photos_test.csv (gate)\n", encoding="utf-8", newline="\n")
    return {"photos": Counter(p["split"] for p in photos), "unreadable": unreadable, "lock": lock}


# --- scoring ----------------------------------------------------------------------------


def photo_score(model, bgr: np.ndarray, cfg: Config) -> float | None:
    """Mean road probability over the photo's tiles; None when it is smaller than a tile."""
    coords, probs = score_image(model, bgr, cfg)
    return float(probs.mean()) if coords else None


def photo_scores(model, photos: list[dict], cfg: Config) -> np.ndarray:
    out = np.full(len(photos), np.nan)
    for i, p in enumerate(photos):
        _, bgr = read_image(Path(p["photo"]), cv2.IMREAD_COLOR)
        score = photo_score(model, bgr, cfg) if bgr is not None else None
        if score is not None:
            out[i] = score
    return out


def photo_threshold(road_scores: np.ndarray, pass_rate: float) -> float:
    """Highest threshold at which at least pass_rate of the road photos score at or above it."""
    s = np.sort(np.asarray(road_scores, float))
    k = int(math.floor((1 - pass_rate) * len(s) + 1e-9))
    return float(s[min(k, len(s) - 1)])


class Gate:
    """The trained gate as batch uses it."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model, meta = load_model(cfg.gate.model)
        self.threshold, self.version = meta["photo_threshold"], meta["model_version"]

    def road_prob(self, bgr: np.ndarray) -> float | None:
        return photo_score(self.model, bgr, self.cfg)


def _wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def rates(photos: list[dict], scores: np.ndarray, threshold: float) -> dict:
    """Share of photos judged road, per dataset and pooled by kind, with 95% Wilson CIs."""
    passed = scores >= threshold  # NaN (no tile) never passes
    road = np.array([p["road"] == 1 for p in photos], bool)
    datasets = np.array([p["dataset"] for p in photos])
    kind = {d: bool(road[datasets == d][0]) for d in set(datasets.tolist())}
    selections = [("road photos", True, road), ("non-road photos", False, ~road)]
    selections += [
        (d, kind[d], datasets == d) for d in sorted(kind, key=lambda d: (not kind[d], d))
    ]
    out = {}
    for key, is_road, idx in selections:
        n, k = int(idx.sum()), int(passed[idx].sum())
        out[key] = {
            "road": is_road,
            "photos": n,
            "passed": k,
            "pass_rate": round(k / n, 4) if n else None,
            "ci95": _wilson(k, n),
        }
    return out


# --- training ---------------------------------------------------------------------------


def sample_tiles(photos: list[dict], cfg: Config, cap: int, rng: np.random.Generator) -> list[dict]:
    """At most tiles_per_photo tiles per photo, cap / n tiles per dataset of each class."""
    t, out = cfg.tiling, []
    for road in (1, 0):
        group = [p for p in photos if p["road"] == road]
        datasets = sorted({p["dataset"] for p in group})
        for dataset in datasets:
            per, taken = cap // len(datasets), []
            items = [p for p in group if p["dataset"] == dataset]
            for i in rng.permutation(len(items)):
                p = items[i]
                coords = tile_grid(p["height"], p["width"], t.tile_size, t.stride, t.drop_edge)
                k = min(cfg.gate.tiles_per_photo, len(coords), per - len(taken))
                if k <= 0:
                    continue
                for j in sorted(rng.choice(len(coords), size=k, replace=False)):
                    y, x = coords[j]
                    taken.append(
                        {"image": p["photo"], "y": y, "x": x, "label": road, "dataset": dataset}
                    )
                if len(taken) >= per:
                    break
            out += taken
    return out


def train_gate(cfg: Config) -> dict:
    started = time.perf_counter()
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    g, t, size = cfg.gate, cfg.train, cfg.tiling.tile_size
    val_photos = read_photos(cfg, "val")
    rows = sample_tiles(read_photos(cfg, "train"), cfg, g.max_tiles_per_class, rng)
    val_rows = sample_tiles(val_photos, cfg, max(2, g.max_tiles_per_class // 3), rng)
    x_train, y_train = load_tiles(rows, Path("."), size), labels_of(rows)
    x_val, y_val = load_tiles(val_rows, Path("."), size), labels_of(val_rows)

    model = build_classifier(t.pretrained)
    loader = DataLoader(
        TileDataset(x_train, y_train, train=True),
        batch_size=t.batch_size,
        shuffle=True,
        generator=torch_generator(cfg.seed),
    )
    ctx = {
        "cfg": cfg,
        "loader": loader,
        "loss_fn": nn.CrossEntropyLoss(weight=class_weights(y_train)),
        "x_val": x_val,
        "y_val": y_val,
        "patience": g.patience,
    }
    state = {"best_f1": -1.0, "history": []}
    _run_stage("A", model, freeze_backbone(model), t.lr, g.epochs, ctx, state)
    if g.finetune_layer4_epochs:
        model.load_state_dict(state["weights"])
        _run_stage(
            "B", model, unfreeze_layer4(model), t.finetune_lr, g.finetune_layer4_epochs, ctx, state
        )
    model.load_state_dict(state["weights"])
    del x_train

    scores = photo_scores(model, val_photos, cfg)
    road = np.array([p["road"] == 1 for p in val_photos])
    threshold = photo_threshold(scores[road & ~np.isnan(scores)], g.road_pass_rate)
    meta = {
        "arch": "resnet18",
        "purpose": "road gate: tile probability of road",
        "photo_threshold": threshold,
        "tile_threshold": state["threshold"],
        "best": state["best"],
        "road_pass_rate": g.road_pass_rate,
        "config_digest": config_digest(cfg),
        "git_commit": git_commit(),
        "train_tiles": len(rows),
        "select_val_tiles": len(val_rows),
    }
    meta = save_model(model, meta, g.model)
    tiles = Counter(f"{r['dataset']} {'road' if r['label'] else 'not road'}" for r in rows)
    report = {
        **meta,
        "train_tiles_by_dataset": dict(sorted(tiles.items())),
        "val_photos": rates(val_photos, scores, threshold),
        "history": state["history"],
        "runtime_s": round(time.perf_counter() - started, 1),
        "hardware": hardware(),
    }
    path = train_report_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


# --- the single test run ----------------------------------------------------------------


def final_problems(cfg: Config, config_file: Path, meta: dict) -> list[str]:
    problems = []
    if results_path(cfg).exists():
        problems.append("gate test results already exist; the gate test runs once")
    if meta.get("config_digest") != config_digest(cfg):
        problems.append("config changed since the gate was trained")
    report = train_report_path(cfg)
    if not report.exists():
        problems.append("no gate training report; run crackqa gate-train")
    elif json.loads(report.read_text(encoding="utf-8"))["model_version"] != meta.get(
        "model_version"
    ):
        problems.append("the gate model file is not the one in reports/gate_train.json")
    lock = lock_path(cfg)
    if not lock.exists():
        problems.append("no gate test lock; run crackqa gate-split")
    elif file_sha256(photos_path(cfg, "test")) != lock.read_text(encoding="utf-8").split()[0]:
        problems.append("gate test photos do not match the committed lock")
    problems += git_problems([config_file, lock, report])
    return problems


def _group(p: dict) -> str:
    """Dataset, or DTD/<class> for a DTD photo."""
    return (
        f"{p['dataset']}/{Path(p['photo']).parent.name}" if p["dataset"] == "DTD" else p["dataset"]
    )


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def final_markdown(report: dict) -> str:
    r = report["rates"]
    lines = [
        f"# Road gate on the {report['split']} photos",
        "",
        f"A photo passes (is judged a road) when the mean road probability of its tiles reaches "
        f"{report['photo_threshold']:.4f}, the threshold chosen on val so that "
        f"{report['road_pass_rate']:.0%} of the val road photos pass. 95% Wilson intervals.",
        "",
        "| Photos | Road | Count | Passed | Pass rate | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for key, v in r.items():
        ci = "-" if v["ci95"] is None else f"{v['ci95'][0]:.3f} to {v['ci95'][1]:.3f}"
        lines.append(
            f"| {key} | {'yes' if v['road'] else 'no'} | {v['photos']} | {v['passed']} | "
            f"{_fmt(v['pass_rate'])} | {ci} |"
        )
    lines += ["", "Non-road groups with the most photos passed:", ""]
    lines += [f"- {k}: {n}" for k, n in report["non_road_passed_by_group"][:10]] or ["- none"]
    lines += ["", f"Road photos stopped ({len(report['road_stopped'])}):", ""]
    lines += [
        f"- {s['dataset']}: `{s['photo']}` ({s['score']:.3f})" for s in report["road_stopped"]
    ]
    lines += [
        "",
        f"Model `{report['model_version']}`; config `{report['config_digest'][:12]}`; "
        + (f"lock `{report['test_lock'][:12]}`; " if report.get("test_lock") else "")
        + f"commit `{report['git_commit']}`.",
    ]
    return "\n".join(lines) + "\n"


def gate_final(cfg: Config, config_file: Path, rehearse: bool = False) -> dict:
    model, meta = load_model(cfg.gate.model)
    split = "val" if rehearse else "test"
    if not rehearse:
        problems = final_problems(cfg, config_file, meta)
        if problems:
            raise FinalRunRefused("; ".join(problems))
    photos = read_photos(cfg, split)
    scores = photo_scores(model, photos, cfg)
    threshold = meta["photo_threshold"]
    passed = scores >= threshold
    stopped = [
        {"photo": p["photo"], "dataset": p["dataset"], "score": round(float(s), 4)}
        for p, s, ok in zip(photos, scores, passed, strict=True)
        if p["road"] == 1 and not ok
    ]
    groups = Counter(
        _group(p) for p, ok in zip(photos, passed, strict=True) if not p["road"] and ok
    )
    report = {
        "split": split,
        "photo_threshold": threshold,
        "road_pass_rate": meta["road_pass_rate"],
        "rates": rates(photos, scores, threshold),
        "non_road_passed_by_group": groups.most_common(),
        "road_stopped": sorted(stopped, key=lambda s: s["score"]),
        "model_version": meta["model_version"],
        "config_digest": config_digest(cfg),
        "git_commit": git_commit(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = cfg.gate.model / "rehearsal" if rehearse else cfg.paths.reports
    stem = "gate_val_rehearsal" if rehearse else "gate_test"
    if not rehearse:
        report["test_lock"] = lock_path(cfg).read_text(encoding="utf-8").split()[0]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{stem}_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["photo", "dataset", "road", "score"])
        for p, s in zip(photos, scores, strict=True):
            writer.writerow([p["photo"], p["dataset"], p["road"], repr(float(s))])
    (out / f"{stem}_results.md").write_text(final_markdown(report), encoding="utf-8", newline="\n")
    # the json is written last: its presence is what refuses a second run
    (out / f"{stem}_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report
