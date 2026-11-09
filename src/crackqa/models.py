"""ResNet18 tile classifier, SmallUNet segmenter and inference helpers."""

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


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """U-Net with `levels` down steps and `base` channels at the top (about 2M parameters
    for 4 levels and 16 channels). Input height and width must be multiples of 2**levels.
    Outputs one crack logit per pixel."""

    def __init__(self, levels: int = 4, base: int = 16):
        super().__init__()
        widths = [base * 2**i for i in range(levels + 1)]
        self.down = nn.ModuleList(
            [_block(3 if i == 0 else widths[i - 1], widths[i]) for i in range(levels)]
        )
        self.bottom = _block(widths[levels - 1], widths[levels])
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
                for i in reversed(range(levels))
            ]
        )
        self.merge = nn.ModuleList(
            [_block(2 * widths[i], widths[i]) for i in reversed(range(levels))]
        )
        self.head = nn.Conv2d(base, 1, 1)
        self.multiple = 2**levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.bottom(x)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips), strict=True):
            x = merge(torch.cat([up(x), skip], dim=1))
        return self.head(x)


@torch.no_grad()
def predict_mask(model: SmallUNet, rgb: np.ndarray) -> np.ndarray:
    """Crack probability per pixel for one uint8 RGB image of any size (reflect padded)."""
    model.eval()
    h, w = rgb.shape[:2]
    m = model.multiple
    ph, pw = (-h) % m, (-w) % m
    padded = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    logits = model(normalize(padded[None]))[0, 0, :h, :w]
    return torch.sigmoid(logits).numpy()
