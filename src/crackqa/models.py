"""ResNet18 tile classifier and inference helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from crackqa.datasets import normalize


def build_classifier(pretrained: bool) -> nn.Module:
    """ResNet18 with a 2 class head. pretrained=False needs no download (tests, CI)."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def freeze_backbone(model: nn.Module) -> list[nn.Parameter]:
    """Stage A: only the head learns. Returns the trainable parameters."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")
    return [p for p in model.parameters() if p.requires_grad]


def unfreeze_layer4(model: nn.Module) -> list[nn.Parameter]:
    """Stage B: layer4 and the head learn."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(("fc.", "layer4."))
    return [p for p in model.parameters() if p.requires_grad]


def set_train_mode(model: nn.Module) -> None:
    """Train mode only where weights learn, so frozen BatchNorm keeps its ImageNet stats."""
    model.eval()
    if any(p.requires_grad for p in model.layer4.parameters()):
        model.layer4.train()


@torch.no_grad()
def predict(model: nn.Module, tiles: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Crack probability for each uint8 RGB tile."""
    model.eval()
    probs = []
    for start in range(0, len(tiles), batch_size):
        logits = model(normalize(tiles[start : start + batch_size]))
        probs.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(probs) if probs else np.zeros(0, np.float32)


def model_paths(outputs: Path) -> tuple[Path, Path]:
    folder = outputs / "models"
    return folder / "resnet18.pt", folder / "resnet18.json"


def save_model(model: nn.Module, meta: dict, outputs: Path) -> dict:
    weights, meta_path = model_paths(outputs)
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    meta = {**meta, "model_version": f"resnet18-{file_digest(weights)[:12]}"}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta


def load_model(outputs: Path) -> tuple[nn.Module, dict]:
    weights, meta_path = model_paths(outputs)
    if not weights.exists():
        raise FileNotFoundError(f"no trained model at {weights.as_posix()}; run crackqa train")
    model = build_classifier(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model, json.loads(meta_path.read_text(encoding="utf-8"))


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
