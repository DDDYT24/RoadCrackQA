"""V2 tiles: centre line labels, groups, split, test lock, and scoring on V2 val."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from crackqa.cli import main
from crackqa.config import V2DatasetConfig, parse_config
from crackqa.models import build_classifier, save_model
from crackqa.split import LeakageError
from crackqa.tiling import AMBIGUOUS, CRACK, NO_CRACK
from crackqa.v2data import (
    UNKNOWN,
    assert_no_leak,
    assign_groups,
    build_rows,
    centerline,
    group_of,
    label_of,
    make_v2_split,
    manifest_v2_path,
    read_v2_rows,
)
from crackqa.v2eval import evaluate_val, read_ablation

from conftest import texture, write


def test_centerline_is_one_line_whatever_the_width():
    bar = np.zeros((60, 140), np.uint8)
    bar[20:40, 20:120] = 255  # 20 px wide, 100 px long
    assert 70 <= centerline(bar).sum() <= 100
    line = np.zeros((60, 140), np.uint8)
    cv2.line(line, (10, 30), (130, 30), 255, 1)
    assert centerline(line).sum() == line.astype(bool).sum()
    assert not centerline(np.zeros((8, 8), np.uint8)).any()


def test_label_rules(dataset, tmp_path):
    raw = dataset.config_dict(tmp_path / "reports")
    cfg = parse_config(raw)  # centerline, 32 px
    assert label_of(0, 0, cfg) == NO_CRACK
    assert label_of(400, 10, cfg) == AMBIGUOUS  # a wide mask grazing the border
    assert label_of(40, 40, cfg) == CRACK  # a thin crack crossing the tile
    assert label_of(UNKNOWN, UNKNOWN, cfg) == AMBIGUOUS
    raw["v2"]["label_rule"] = "pixels"
    pixels = parse_config(raw)
    assert label_of(400, 10, pixels) == CRACK and label_of(40, 40, pixels) == AMBIGUOUS


def test_groups_by_runs_of_numbers_and_by_blocks():
    ids = ["train_0001", "train_0002", "train_0004", "test_0001", "x"]
    assert group_of(ids, gap=1, block=0) == {
        "train_0001": "train_0001",
        "train_0002": "train_0001",
        "train_0004": "train_0004",
        "test_0001": "test_0001",
        "x": "x",
    }
    blocks = group_of([f"{n:03d}" for n in range(1, 10)], gap=0, block=4)
    assert blocks["001"] == blocks["004"] != blocks["005"] == blocks["008"] != blocks["009"]
    assert group_of(["5", "6"], gap=0, block=0) == {"5": "5", "6": "6"}


def test_assignment_is_seeded_and_sends_crack_free_groups_to_test():
    d = V2DatasetConfig(0.6, 0.2, 0.2, 1, 0, True)
    sizes = {f"g{i}": 10 for i in range(20)}
    a = assign_groups(sizes, {"g3", "g7"}, d, 42, "X")
    assert a == assign_groups(sizes, {"g3", "g7"}, d, 42, "X")
    assert a["g3"] == a["g7"] == "test"
    counts = {s: sum(v == s for v in a.values()) for s in ("train", "val", "test")}
    assert counts == {"train": 12, "val": 4, "test": 4}
    no_test = V2DatasetConfig(0.8, 0.2, 0.0, 0, 10, False)
    assert "test" not in assign_groups(sizes, set(), no_test, 42, "X").values()


def _extra_config(dataset, tmp_path: Path) -> dict:
    """The CRACK500-style fixture plus a mask dataset and a 256 px patch dataset."""
    masks_root = tmp_path / "roads"
    for i in range(1, 7):
        img = texture(500 + i, 256, 384)
        mask = np.zeros((256, 384), np.uint8)
        if i == 1:
            mask[:, 120:136] = 255  # wide crack just left of the x = 128 border
        else:
            cv2.line(mask, (0, 60), (383, 60), 255, 2)  # crosses every top row tile
        write(masks_root / "img" / f"{i:04d}.jpg", img)
        write(masks_root / "gt" / f"{i:04d}.png", mask)
    patches = tmp_path / "patches"
    for n in range(1, 13):  # photos with a crack patch and a no-crack patch
        write(patches / "CP" / f"{n:03d}-1.jpg", texture(600 + n, 256, 256))
        write(patches / "UP" / f"{n:03d}-2.jpg", texture(800 + n, 256, 256))
    for n in (13, 14):  # a photo with no crack patch at all
        write(patches / "UP" / f"{n:03d}-1.jpg", texture(600 + n, 256, 256))
        write(patches / "UP" / f"{n:03d}-2.jpg", texture(700 + n, 256, 256))
    raw = dataset.config_dict(tmp_path / "reports")
    common = {"image_exts": [".jpg"], "invert_mask": False}
    raw["extra_data"] = {
        "ROADS": {
            **common,
            "images": str(masks_root / "img"),
            "masks": str(masks_root / "gt"),
            "mask_name": [r"^(.*)$", r"\1.png"],
            "label_dirs": {},
            "source_id_regex": r"^(\d+)\.",
        },
        "PATCHES": {
            **common,
            "images": str(patches),
            "masks": None,
            "mask_name": [],
            "label_dirs": {"CP": "crack", "UP": "no_crack"},
            "source_id_regex": r"^(\d{3})-",
        },
    }
    split = {"train": 0.6, "val": 0.2, "test": 0.2}
    raw["v2"]["processed"] = str(tmp_path / "processed" / "v2")
    raw["v2"]["datasets"] = {
        "ROADS": {**split, "gap": 0, "block": 0, "crack_free_to_test": False},
        "PATCHES": {**split, "gap": 0, "block": 2, "crack_free_to_test": True},
    }
    return raw


def test_rows_labels_and_crack500_test_left_out(dataset, tmp_path):
    cfg = parse_config(_extra_config(dataset, tmp_path))
    rows, skipped = build_rows(cfg)
    assert {r["split"] for r in rows if r["dataset"] == "CRACK500"} == {"train", "val"}
    assert not any("testcrop" in r["image"] for r in rows)
    patch = [r for r in rows if r["dataset"] == "PATCHES"]
    assert len(patch) == 28 * 4
    assert {r["label"] for r in patch if "/UP/" in r["image"]} == {NO_CRACK}
    assert {r["label"] for r in patch if "/CP/" in r["image"]} == {AMBIGUOUS}
    grazed = next(r for r in rows if r["image"].endswith("0001.jpg") and r["x"] == 128)
    assert grazed["crack_px"] >= 50 and grazed["label"] == AMBIGUOUS  # mask only reaches over
    crossed = [r for r in rows if r["image"].endswith("0002.jpg") and r["y"] == 0]
    assert {r["label"] for r in crossed} == {CRACK}
    crack_free = {r["split"] for r in patch if r["source_id"] in ("013", "014")}
    assert crack_free == {"test"} and skipped == {"ROADS": 0, "PATCHES": 0}


def test_split_writes_manifests_and_a_stable_lock(dataset, tmp_path):
    cfg = parse_config(_extra_config(dataset, tmp_path))
    first = make_v2_split(cfg)
    assert make_v2_split(cfg)["test_lock"] == first["test_lock"]
    lock = (cfg.paths.reports / "test2_lock.txt").read_text(encoding="utf-8")
    assert lock.startswith(first["test_lock"])
    rows = [r for s in ("train", "val", "test") for r in read_v2_rows(manifest_v2_path(cfg, s))]
    assert_no_leak(rows)
    assert "PATCHES" in (cfg.paths.reports / "v2_balance.md").read_text(encoding="utf-8")


def test_leak_is_refused():
    rows = [
        {"dataset": "A", "source_id": "1", "group": "g", "split": "train"},
        {"dataset": "A", "source_id": "2", "group": "g", "split": "test"},
    ]
    with pytest.raises(LeakageError, match="group"):
        assert_no_leak(rows)


def test_v2_eval_keeps_one_row_per_name(dataset, tmp_path):
    raw = _extra_config(dataset, tmp_path)
    cfg = parse_config(raw)
    make_v2_split(cfg)
    save_model(build_classifier(pretrained=False), {"threshold": 0.5}, cfg.paths.outputs)
    (cfg.paths.reports / "baseline_val.json").write_text(json.dumps({"threshold": 0.01}))
    row = evaluate_val(cfg, "untrained", cfg.paths.outputs)
    assert row["results"]["pooled"]["model"]["n"] > 0
    patches = row["results"]["PATCHES"]["model"]  # no crack tile: only false alarms defined
    assert patches["f1"] is None and patches["false_alarm_rate"] is not None
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    assert main(["v2-eval", "--config", str(config_file), "--name", "untrained"]) == 0
    assert [r["name"] for r in read_ablation(cfg)] == ["untrained"]
    assert "## untrained" in (cfg.paths.reports / "ablation.md").read_text(encoding="utf-8")
