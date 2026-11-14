import csv
import json

import numpy as np
import pytest

from crackqa.batch import COLUMNS, run_batch
from crackqa.config import parse_config
from crackqa.models import build_classifier, save_model
from crackqa.overlay import overlay
from crackqa.seed import seed_everything

from conftest import crack_pair, write


@pytest.fixture
def batch_setup(dataset, tmp_path):
    cfg = dataset.config(tmp_path / "reports")
    seed_everything(0)
    save_model(build_classifier(pretrained=False), {"threshold": 0.5}, cfg.paths.outputs)
    photos = tmp_path / "photos"
    write(photos / "a.jpg", crack_pair(1)[0])
    write(photos / "sub" / "b.png", crack_pair(2)[0])
    (photos / "broken.jpg").write_bytes(b"")
    (photos / "notes.txt").write_text("not an image", encoding="utf-8")
    write(photos / "tiny.jpg", crack_pair(3, 64, 64)[0])
    return cfg, photos


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_batch_outputs(batch_setup, tmp_path):
    cfg, photos = batch_setup
    out = tmp_path / "out"
    summary = run_batch(cfg, photos, out)

    rows = _rows(out / "results.csv")
    assert list(rows[0]) == COLUMNS
    status = {r["file"]: r["status"] for r in rows}
    assert status == {
        "a.jpg": "ok",
        "broken.jpg": "corrupt",
        "notes.txt": "skipped",
        "sub/b.png": "ok",
        "tiny.jpg": "skipped",
    }
    ok = [r for r in rows if r["status"] == "ok"]
    assert all(int(r["n_tiles"]) == 10 and r["width"] == "640" for r in ok)
    assert sorted(p.name for p in (out / "overlays").iterdir()) == [
        "a_overlay.jpg",
        "sub__b_overlay.jpg",
    ]
    assert (summary["ok"], summary["corrupt"], summary["skipped"]) == (2, 1, 2)
    on_disk = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["images_per_s"] > 0 and on_disk["model_version"].startswith("resnet18-")
    assert "corrupt broken.jpg" in (out / "batch.log").read_text(encoding="utf-8")


def test_batch_is_deterministic(batch_setup, tmp_path):
    cfg, photos = batch_setup
    run_batch(cfg, photos, tmp_path / "one")
    run_batch(cfg, photos, tmp_path / "two")
    first = (tmp_path / "one" / "results.csv").read_bytes()
    assert first == (tmp_path / "two" / "results.csv").read_bytes()


def test_tile_mode_scores_with_the_tile_model(dataset, tmp_path, batch_setup):
    _, photos = batch_setup
    raw = dataset.config_dict(tmp_path / "reports")
    raw["paths"]["tile_model"] = str(tmp_path / "v2")
    cfg = parse_config(raw)
    meta = save_model(build_classifier(pretrained=False), {"threshold": 0.25}, cfg.paths.tile_model)
    summary = run_batch(cfg, photos, tmp_path / "out")
    assert summary["threshold"] == 0.25 and summary["model_version"] == meta["model_version"]


@pytest.mark.parametrize("threshold", [2.0, -1.0])
def test_gate_marks_not_road(dataset, tmp_path, batch_setup, threshold):
    _, photos = batch_setup
    raw = dataset.config_dict(tmp_path / "reports")
    raw["gate"]["enabled"] = True
    cfg = parse_config(raw)
    gate = save_model(
        build_classifier(pretrained=False), {"photo_threshold": threshold}, cfg.gate.model
    )
    out = tmp_path / "out"
    summary = run_batch(cfg, photos, out)
    rows = {r["file"]: r for r in _rows(out / "results.csv")}
    blocked = threshold > 1  # no road probability reaches 2.0, every one reaches -1.0
    assert rows["a.jpg"]["status"] == ("not_road" if blocked else "ok")
    assert 0 <= float(rows["a.jpg"]["road_prob"]) <= 1
    assert rows["broken.jpg"]["status"] == "corrupt"
    assert summary["not_road"] == (2 if blocked else 0)
    assert summary["gate_model_version"] == gate["model_version"]
    assert (out / "overlays" / "a_overlay.jpg").exists() != blocked


def test_overlay_keeps_the_input_size():
    img = crack_pair(4, 300, 500)[0]
    coords = [(0, 0), (0, 128), (128, 256)]
    out = overlay(img, coords, np.array([0.1, 0.7, 0.95]), 128, threshold=0.5)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert not np.array_equal(out[:128, 128:256], img[:128, 128:256])
    np.testing.assert_array_equal(out[256:, :], img[256:, :])


def test_include_pattern_ignores_case(batch_setup, tmp_path):
    cfg, photos = batch_setup
    write(photos / "c.JPG", crack_pair(5)[0])
    write(photos / "c_mask.png", crack_pair(5)[1])
    summary = run_batch(cfg, photos, tmp_path / "out", include="*.jpg")
    files = [r["file"] for r in _rows(tmp_path / "out" / "results.csv")]
    assert files == ["a.jpg", "broken.jpg", "c.JPG", "tiny.jpg"]
    assert summary["ok"] == 2
