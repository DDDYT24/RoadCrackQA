"""The road gate: photo lists and lock, threshold rule, training, the single test run."""

from __future__ import annotations

import json
import subprocess

import cv2
import numpy as np
import pytest
from test_v2train import _raw

from crackqa.cli import main
from crackqa.config import parse_config
from crackqa.evaluate import FinalRunRefused
from crackqa.gate import (
    gate_final,
    lock_path,
    make_gate_split,
    photo_threshold,
    read_photos,
    results_path,
    train_gate,
)
from crackqa.v2data import make_v2_split

from conftest import crack_pair, write


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _flat(i: int) -> np.ndarray:
    """A smooth colour gradient with a disc: nothing like a road."""
    img = np.zeros((200, 300, 3), np.uint8)
    img[:] = np.linspace(0, 255, 300, dtype=np.uint8)[None, :, None]
    img[..., i % 3] = 40 * (i % 6)
    cv2.circle(img, (150, 100), 60, (255, 255 - 20 * i, 0), -1)
    return img


@pytest.fixture
def gate_setup(dataset, tmp_path):
    raw = _raw(dataset, tmp_path)
    nonroad, full = tmp_path / "nonroad", tmp_path / "full"
    for i in range(10):
        write(nonroad / "A" / f"a{i}.jpg", _flat(i))
    for cls in ("bumpy", "cracked"):
        for i in range(5):
            write(nonroad / "B" / cls / f"{cls}_{i}.jpg", _flat(10 + i))
    for split in ("val", "test"):
        for i in range(2):
            write(full / split / f"p{i}.jpg", crack_pair(900 + i, 300, 400)[0])
            write(full / split / f"p{i}_mask.png", crack_pair(900 + i, 300, 400)[1])
    raw["gate"].update(
        non_road={
            "A": {"images": str(nonroad / "A"), "exclude_dirs": []},
            "B": {"images": str(nonroad / "B"), "exclude_dirs": ["cracked"]},
        },
        full_photos={"val": str(full / "val"), "test": str(full / "test")},
        max_photos_per_dataset=5,
        tiles_per_photo=2,
        max_tiles_per_class=40,
        epochs=1,
        finetune_layer4_epochs=0,
        patience=1,
    )
    cfg = parse_config(raw)
    make_v2_split(cfg)
    return raw, cfg


def test_photo_threshold():
    scores = np.arange(1, 11) / 10
    assert photo_threshold(scores, 0.9) == 0.2  # 9 of 10 at or above
    assert photo_threshold(scores, 1.0) == 0.1
    assert photo_threshold(np.array([0.7]), 0.99) == 0.7


def test_split_lists_and_lock(gate_setup):
    _, cfg = gate_setup
    first = make_gate_split(cfg)
    assert make_gate_split(cfg)["lock"] == first["lock"]
    assert lock_path(cfg).read_text(encoding="utf-8").startswith(first["lock"])
    rows = {s: read_photos(cfg, s) for s in ("train", "val", "test")}
    every = [r for split in rows.values() for r in split]
    assert len({r["photo"] for r in every}) == len(every)  # no photo in two splits
    assert not any("cracked" in r["photo"] or r["photo"].endswith("_mask.png") for r in every)
    assert not any(r["dataset"] == "CRACK500 full photos" for r in rows["train"])
    for split in ("val", "test"):
        assert {r["road"] for r in rows[split]} == {0, 1}
        datasets = [r["dataset"] for r in rows[split] if r["road"]]
        assert max(datasets.count(d) for d in set(datasets)) <= 5


def test_train_then_final_once(gate_setup, tmp_path):
    raw, cfg = gate_setup
    make_gate_split(cfg)
    report = train_gate(cfg)
    assert 0 <= report["photo_threshold"] <= 1
    assert report["val_photos"]["road photos"]["photos"] > 0
    assert (cfg.gate.model / "models" / "resnet18.pt").exists()

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    assert main(["gate-final", "--config", str(config_file), "--rehearse"]) == 0
    assert not results_path(cfg).exists()
    with pytest.raises(FinalRunRefused, match="not committed"):
        gate_final(cfg, config_file)

    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "config.json", "reports/gate_test_lock.txt", "reports/gate_train.json")
    _git(tmp_path, "commit", "-q", "-m", "lock")
    result = gate_final(cfg, config_file)
    assert set(result["rates"]) >= {"road photos", "non-road photos", "A", "B"}
    assert result["test_lock"] and (cfg.paths.reports / "gate_test_results.md").exists()
    with pytest.raises(FinalRunRefused, match="already exist"):
        gate_final(cfg, config_file)
    assert main(["gate-final", "--config", str(config_file)]) == 3
