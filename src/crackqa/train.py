"""Train the ResNet18 tile classifier on CPU, choosing epoch and threshold on val."""

from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from crackqa.config import Config, config_digest
from crackqa.datasets import TileDataset, labels_of, load_tiles
from crackqa.metrics import best_threshold, summarize
from crackqa.models import (
    build_classifier,
    freeze_backbone,
    predict,
    save_model,
    set_train_mode,
    unfreeze_layer4,
)
from crackqa.runinfo import git_commit, hardware
from crackqa.seed import seed_everything, torch_generator
from crackqa.split import manifest_path
from crackqa.tiling import AMBIGUOUS, read_rows


def labelled_rows(cfg: Config, split: str) -> list[dict]:
    return [r for r in read_rows(manifest_path(cfg, split)) if r["label"] != AMBIGUOUS]


def cap_rows(rows: list[dict], cap: int, seed: int) -> list[dict]:
    """At most cap rows, sampled per class so the class shares stay the same."""
    if len(rows) <= cap:
        return rows
    rng = np.random.default_rng(seed)
    labels = labels_of(rows)
    keep = []
    for value in (0, 1):
        idx = np.flatnonzero(labels == value)
        n = round(cap * len(idx) / len(rows))
        keep.extend(rng.choice(idx, size=n, replace=False).tolist())
    return [rows[i] for i in sorted(keep)]


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    return torch.tensor(len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32)


def train_epoch(model, loader, loss_fn, optimizer) -> float:
    set_train_mode(model)
    total, n = 0.0, 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def _run_stage(name, model, params, lr, epochs, ctx, state) -> None:
    """Train for up to `epochs`, keeping the best val F1 in `state` (early stopping)."""
    cfg = ctx["cfg"]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.train.weight_decay)
    stale = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        loss = train_epoch(model, ctx["loader"], ctx["loss_fn"], optimizer)
        probs = predict(model, ctx["x_val"])
        threshold, f1 = best_threshold(ctx["y_val"], probs)
        state["history"].append(
            {
                "stage": name,
                "epoch": epoch,
                "train_loss": round(loss, 5),
                "val_f1": round(f1, 5),
                "val_threshold": round(threshold, 5),
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        print(f"{name} epoch {epoch}: loss {loss:.4f}  val F1 {f1:.4f}  thr {threshold:.3f}")
        if f1 > state["best_f1"]:
            state.update(best_f1=f1, threshold=threshold, weights=copy.deepcopy(model.state_dict()))
            state["best"] = f"{name} epoch {epoch}"
            stale = 0
        else:
            stale += 1
            if stale >= cfg.train.patience:
                break


def train(cfg: Config) -> dict:
    started = time.perf_counter()
    seed_everything(cfg.seed)
    t = cfg.train
    size = cfg.tiling.tile_size

    train_all = labelled_rows(cfg, "train")
    train_rows = cap_rows(train_all, t.max_train_tiles, cfg.seed)
    val_rows = labelled_rows(cfg, "val")
    x_train, y_train = load_tiles(train_rows, cfg.paths.raw, size), labels_of(train_rows)
    x_val, y_val = load_tiles(val_rows, cfg.paths.raw, size), labels_of(val_rows)

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
    }
    state = {"best_f1": -1.0, "history": []}
    _run_stage("A", model, freeze_backbone(model), t.lr, t.epochs, ctx, state)
    if t.finetune_layer4_epochs:
        model.load_state_dict(state["weights"])
        _run_stage(
            "B", model, unfreeze_layer4(model), t.finetune_lr, t.finetune_layer4_epochs, ctx, state
        )

    model.load_state_dict(state["weights"])
    threshold = state["threshold"]
    val = summarize(y_val, predict(model, x_val), threshold)
    train_fit = summarize(y_train, predict(model, x_train), threshold)
    meta = {
        "arch": "resnet18",
        "pretrained": t.pretrained,
        "threshold": threshold,
        "best": state["best"],
        "config_digest": config_digest(cfg),
        "git_commit": git_commit(),
        "train_tiles": len(train_rows),
        "train_tiles_available": len(train_all),
        "val_tiles": len(val_rows),
    }
    meta = save_model(model, meta, cfg.paths.outputs)
    report = {
        **meta,
        "val": val,
        "train_fit": train_fit,
        "train_minus_val_f1": round(train_fit["f1"] - val["f1"], 4),
        "history": state["history"],
        "runtime_s": round(time.perf_counter() - started, 1),
        "hardware": hardware(),
    }
    path = cfg.paths.reports / "train_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report
