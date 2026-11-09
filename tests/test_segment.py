import csv
import subprocess

import numpy as np
import pytest
import torch

from crackqa.batch import SEG_COLUMNS, run_batch
from crackqa.cli import main
from crackqa.config import load_config, parse_config
from crackqa.models import SmallUNet, predict_mask
from crackqa.overlay import overlay_mask
from crackqa.segment import (
    BINS,
    SegRefused,
    best_pixel_threshold,
    dice_loss,
    evaluate_seg,
    example_figure,
    metrics_at,
    train_seg,
)

from conftest import crack_pair, write


def test_small_unet_size_and_shapes():
    model = SmallUNet(levels=4, base=16)
    assert 1.8e6 < sum(p.numel() for p in model.parameters()) < 2.1e6
    assert model(torch.randn(2, 3, 256, 256)).shape == (2, 1, 256, 256)
    prob = predict_mask(model, np.zeros((360, 640, 3), np.uint8))  # padded to 368 inside
    assert prob.shape == (360, 640) and prob.min() >= 0 and prob.max() <= 1


def test_dice_loss_bounds():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:5, 2:5] = 1
    perfect = (target * 2 - 1) * 20
    assert float(dice_loss(perfect, target)) < 0.05
    assert float(dice_loss(-perfect, target)) > 0.9


def test_pixel_threshold_from_histograms():
    pos, neg = np.zeros(BINS, np.int64), np.zeros(BINS, np.int64)
    pos[800] = 90  # crack pixels mostly at p = 0.8
    pos[100] = 10
    neg[50] = 1000  # background mostly at p = 0.05
    neg[850] = 5
    threshold, m = best_pixel_threshold(pos, neg)
    # keeping the p = 0.1 crack pixels wins: F1 200/205 against 180/195 above 0.1;
    # bins 51 to 100 tie and the highest edge is kept
    assert threshold == pytest.approx(0.1)
    assert m["precision"] == pytest.approx(100 / 105)
    assert m["recall"] == pytest.approx(1.0)
    assert m["iou"] == pytest.approx(100 / 105)
    assert metrics_at(pos, neg, threshold) == pytest.approx(m)


def _tiny(dataset, tmp_path):
    """Synthetic dataset, a tiny SmallUNet config, tiles and split done, config committed."""
    raw = dataset.config_dict(tmp_path / "reports")
    raw["seg"].update(levels=2, base_channels=4, crop_size=64, epochs=1, batch_size=4)
    config_file = tmp_path / "config.yaml"
    import yaml

    config_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    for command in ("tile", "split"):
        assert main([command, "--config", str(config_file)]) == 0
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run([*git, "add", "config.yaml", "reports/test_lock.txt"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "lock"], cwd=tmp_path, check=True)
    return config_file, parse_config(raw)


def test_seg_train_eval_and_final_guard(dataset, tmp_path):
    config_file, cfg = _tiny(dataset, tmp_path)
    report = train_seg(cfg)
    assert report["best_epoch"] == 1 and 0 <= report["threshold"] < 1

    val = evaluate_seg(cfg, "val", final=False, config_file=config_file)
    assert set(val) >= {"smallunet", "opencv_baseline"}
    with pytest.raises(SegRefused):
        evaluate_seg(cfg, "test", final=False, config_file=config_file)
    figure = tmp_path / "reports" / "figures" / "seg_examples.jpg"
    with pytest.raises(SegRefused, match="final segmentation test first"):
        example_figure(cfg, figure, n=2)

    final = evaluate_seg(load_config(config_file), "test", final=True, config_file=config_file)
    assert 0 <= final["smallunet"]["iou"] <= 1
    assert len(example_figure(cfg, figure, n=2)) == 1 and figure.exists()  # one test crop
    with pytest.raises(SegRefused, match="already exist"):
        evaluate_seg(cfg, "test", final=True, config_file=config_file)


def test_batch_seg_mode_writes_masks_and_overlays(dataset, tmp_path):
    config_file, cfg = _tiny(dataset, tmp_path)
    train_seg(cfg)
    photos = tmp_path / "photos"
    write(photos / "a.jpg", crack_pair(7)[0])
    (photos / "bad.jpg").write_bytes(b"")
    summary = run_batch(cfg, photos, tmp_path / "out", mode="seg")

    with open(tmp_path / "out" / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == SEG_COLUMNS
    assert {r["file"]: r["status"] for r in rows} == {"a.jpg": "ok", "bad.jpg": "corrupt"}
    assert (tmp_path / "out" / "masks" / "a_mask.png").exists()
    assert (tmp_path / "out" / "overlays" / "a_overlay.jpg").exists()
    assert summary["mode"] == "seg" and summary["ok"] == 1
    args = ["batch", "--config", str(config_file), "--mode", "seg"]
    assert main([*args, "--input", str(photos), "--output", str(tmp_path / "cli")]) == 0


def test_overlay_mask_keeps_size():
    img = crack_pair(8)[0]
    mask = np.zeros(img.shape[:2], bool)
    mask[10:20, 10:20] = True
    out = overlay_mask(img, mask)
    assert out.shape == img.shape
    assert not np.array_equal(out[10:20, 10:20], img[10:20, 10:20])
    np.testing.assert_array_equal(out[30:], img[30:])
