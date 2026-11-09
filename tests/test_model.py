import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from crackqa.datasets import TileDataset, augment, normalize
from crackqa.models import build_classifier, freeze_backbone, predict, unfreeze_layer4
from crackqa.seed import seed_everything, torch_generator
from crackqa.train import cap_rows, class_weights, train_epoch

from conftest import crack_pair


def test_model_output_shapes():
    model = build_classifier(pretrained=False)  # no download in CI
    assert model(torch.randn(2, 3, 128, 128)).shape == (2, 2)
    probs = predict(model, np.zeros((5, 128, 128, 3), np.uint8))
    assert probs.shape == (5,) and np.all((probs >= 0) & (probs <= 1))


def test_freezing_selects_head_then_layer4():
    model = build_classifier(pretrained=False)
    head = freeze_backbone(model)
    assert sum(p.numel() for p in head) == 512 * 2 + 2
    assert len(unfreeze_layer4(model)) > len(head)


def _synthetic_tiles(n):
    tiles, labels = [], []
    for i in range(n):
        img, mask = crack_pair(i)
        tile = img[:128, :128]
        tiles.append(tile[..., ::-1])
        labels.append(int(mask[:128, :128].sum() > 0))
    return np.ascontiguousarray(np.stack(tiles)), np.array(labels)


def test_a_few_training_steps_reduce_the_loss():
    seed_everything(0)
    tiles, labels = _synthetic_tiles(16)
    model = build_classifier(pretrained=False)
    params = list(model.parameters())
    loader = DataLoader(
        TileDataset(tiles, labels, train=False), batch_size=16, generator=torch_generator(0)
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=1e-3)
    first = train_epoch(model, loader, loss_fn, optimizer)
    for _ in range(4):
        last = train_epoch(model, loader, loss_fn, optimizer)
    assert last < first


def test_augment_keeps_shape_and_range():
    seed_everything(1)
    x = torch.rand(3, 128, 128)
    y = augment(x)
    assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1


def test_normalize_layout():
    x = normalize(np.full((2, 8, 8, 3), 255, np.uint8))
    assert x.shape == (2, 3, 8, 8)


def test_cap_keeps_class_shares_and_is_seeded():
    rows = [{"label": 1}] * 30 + [{"label": 0}] * 70
    capped = cap_rows(rows, 50, seed=3)
    assert len(capped) == 50 and sum(r["label"] for r in capped) == 15
    assert cap_rows(rows, 200, seed=3) is rows


def test_class_weights_balance_the_classes():
    w = class_weights(np.array([0, 0, 0, 1]))
    assert float(w[1]) == pytest.approx(3 * float(w[0]))
