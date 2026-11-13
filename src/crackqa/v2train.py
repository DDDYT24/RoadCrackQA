"""Train one V2 experiment (configs: v2_train.experiments) on the V2 train tiles.

Epoch and threshold are chosen on V2 val, with at most select_val_max_per_dataset tiles per
dataset; test is never read. The model goes to v2_train.outputs/<name>/models/ and the run
report to reports/v2_train_<name>.json.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from crackqa.config import Config, config_digest
from crackqa.datasets import TileDataset, labels_of, load_tiles
from crackqa.metrics import summarize
from crackqa.models import (
    build_classifier,
    freeze_backbone,
    load_model,
    predict,
    save_model,
    unfreeze_layer4,
)
from crackqa.runinfo import git_commit, hardware
from crackqa.seed import seed_everything, torch_generator
from crackqa.tiling import AMBIGUOUS, NO_CRACK
from crackqa.train import _run_stage, cap_rows, class_weights
from crackqa.v2data import manifest_v2_path, read_v2_rows
from crackqa.v2eval import CHUNK

ROOT = Path(".")  # V2 image paths are relative to the run directory


def experiment_dir(cfg: Config, name: str) -> Path:
    return cfg.v2_train.outputs / name


def report_path(cfg: Config, name: str) -> Path:
    return cfg.paths.reports / f"v2_train_{name}.json"


def labelled_v2(cfg: Config, split: str) -> list[dict]:
    return [r for r in read_v2_rows(manifest_v2_path(cfg, split)) if r["label"] != AMBIGUOUS]


def select_val(rows: list[dict], cap: int, seed: int) -> list[dict]:
    """At most cap val tiles per dataset, sampled per class, for picking epoch and threshold."""
    out = []
    for dataset in sorted({r["dataset"] for r in rows}):
        out += cap_rows([r for r in rows if r["dataset"] == dataset], cap, seed)
    return out


def score(model, rows: list[dict], size: int) -> np.ndarray:
    parts = [
        predict(model, load_tiles(rows[i : i + CHUNK], ROOT, size))
        for i in range(0, len(rows), CHUNK)
    ]
    return np.concatenate(parts) if parts else np.zeros(0)


def train_rows(cfg: Config, name: str) -> tuple[list[dict], dict]:
    """Training tiles of one experiment, and where they came from."""
    e = cfg.v2_train.experiments[name]
    pool = labelled_v2(cfg, "train")
    rows = [r for r in pool if r["dataset"] in e.datasets]
    available = len(rows)
    if e.max_tiles:
        rows = cap_rows(rows, e.max_tiles, cfg.seed)
    info = {"from_datasets": available, "after_cap": len(rows), "random": 0, "mined": 0}
    if e.negatives_from:
        negatives = [r for r in pool if r["dataset"] == e.negatives_from and r["label"] == NO_CRACK]
        order = np.random.default_rng(cfg.seed).permutation(len(negatives))
        negatives = [negatives[i] for i in order]
        rows += negatives[: e.negatives_random]
        rest = negatives[e.negatives_random :]
        info.update(random=min(e.negatives_random, len(negatives)), negative_pool=len(negatives))
        if e.negatives_mined:
            miner, meta = load_model(experiment_dir(cfg, e.mined_with))
            scores = score(miner, rest, cfg.tiling.tile_size)
            top = np.argsort(-scores, kind="stable")[: e.negatives_mined]
            rows += [rest[i] for i in sorted(top)]
            info.update(
                mined=len(top),
                mined_with=meta["model_version"],
                mined_min_score=round(float(scores[top].min()), 4) if len(top) else None,
            )
    return rows, info


def train_v2(cfg: Config, name: str) -> dict:
    if name not in cfg.v2_train.experiments:
        raise KeyError(f"unknown experiment {name!r}; see v2_train.experiments")
    started = time.perf_counter()
    seed_everything(cfg.seed)
    t, v = cfg.train, cfg.v2_train
    e = v.experiments[name]
    size = cfg.tiling.tile_size

    rows, info = train_rows(cfg, name)
    val_rows = select_val(labelled_v2(cfg, "val"), v.select_val_max_per_dataset, cfg.seed)
    x_train, y_train = load_tiles(rows, ROOT, size), labels_of(rows)
    x_val, y_val = load_tiles(val_rows, ROOT, size), labels_of(val_rows)

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
        "patience": v.patience,
    }
    state = {"best_f1": -1.0, "history": []}
    _run_stage("A", model, freeze_backbone(model), t.lr, v.epochs, ctx, state)
    if e.finetune_layer4_epochs:
        model.load_state_dict(state["weights"])
        _run_stage(
            "B", model, unfreeze_layer4(model), t.finetune_lr, e.finetune_layer4_epochs, ctx, state
        )
    model.load_state_dict(state["weights"])
    del x_train

    tiles = Counter(f"{r['dataset']} {'crack' if r['label'] else 'no crack'}" for r in rows)
    meta = {
        "arch": "resnet18",
        "experiment": name,
        "threshold": state["threshold"],
        "best": state["best"],
        "config_digest": config_digest(cfg),
        "git_commit": git_commit(),
        "train_tiles": len(rows),
        "select_val_tiles": len(val_rows),
    }
    meta = save_model(model, meta, experiment_dir(cfg, name))
    report = {
        **meta,
        "train_tiles_by_dataset": dict(sorted(tiles.items())),
        "train_rows": info,
        "select_val": summarize(y_val, predict(model, x_val), state["threshold"]),
        "history": state["history"],
        "runtime_s": round(time.perf_counter() - started, 1),
        "hardware": hardware(),
    }
    path = report_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report
