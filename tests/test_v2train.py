"""V2 experiments: training rows, mined negatives, the run report and matched recall."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_v2data import _extra_config

from crackqa.cli import main
from crackqa.config import ConfigError, parse_config
from crackqa.tiling import NO_CRACK
from crackqa.v2data import make_v2_split
from crackqa.v2eval import _at, read_ablation, threshold_at_recall
from crackqa.v2train import experiment_dir, report_path, train_rows, train_v2

EXPERIMENT = {
    "max_tiles": 0,
    "finetune_layer4_epochs": 1,
    "negatives_from": None,
    "negatives_random": 0,
    "negatives_mined": 0,
    "mined_with": None,
}


def _raw(dataset, tmp_path) -> dict:
    raw = _extra_config(dataset, tmp_path)
    raw["train"]["pretrained"] = False  # no weight download in tests
    raw["v2_train"] = {
        "outputs": str(tmp_path / "outputs" / "v2"),
        "epochs": 1,
        "patience": 1,
        "select_val_max_per_dataset": 20,
        "compare_recall": 0.5,
        "experiments": {
            "base": {**EXPERIMENT, "datasets": ["CRACK500", "ROADS"]},
            "negs": {
                **EXPERIMENT,
                "datasets": ["CRACK500"],
                "negatives_from": "PATCHES",
                "negatives_random": 3,
                "negatives_mined": 4,
                "mined_with": "base",
            },
        },
    }
    return raw


def test_matched_recall_threshold():
    y = np.array([1, 1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.4, 0.2, 0.85, 0.5, 0.1])
    t = threshold_at_recall(y, s, 0.5)
    assert t == 0.8
    assert _at(y, s, t) == {"recall": 0.5, "false_alarm_rate": round(1 / 3, 4)}
    assert threshold_at_recall(np.zeros(3), np.ones(3), 0.5) is None
    assert _at(np.zeros(2), np.ones(2), None) == {"recall": None, "false_alarm_rate": None}


def test_mined_with_must_name_another_experiment(dataset, tmp_path):
    raw = _raw(dataset, tmp_path)
    raw["v2_train"]["experiments"]["negs"]["mined_with"] = "negs"
    with pytest.raises(ConfigError, match="another experiment"):
        parse_config(raw)


def test_train_mine_and_score(dataset, tmp_path):
    raw = _raw(dataset, tmp_path)
    cfg = parse_config(raw)
    make_v2_split(cfg)
    (cfg.paths.reports / "baseline_val.json").write_text(json.dumps({"threshold": 0.01}))

    base = train_v2(cfg, "base")
    assert (experiment_dir(cfg, "base") / "models" / "resnet18.pt").exists()
    assert report_path(cfg, "base").exists() and base["history"][0]["stage"] == "A"
    assert set(base["train_tiles_by_dataset"]) <= {
        "CRACK500 crack",
        "CRACK500 no crack",
        "ROADS crack",
        "ROADS no crack",
    }

    rows, info = train_rows(cfg, "negs")
    patches = [r for r in rows if r["dataset"] == "PATCHES"]
    assert info["random"] == 3 and info["mined"] == 4 and len(patches) == 7
    assert {r["label"] for r in patches} == {NO_CRACK} and len({r["tile_id"] for r in patches}) == 7

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    assert main(["v2-train", "--config", str(config_file), "--experiment", "negs"]) == 0
    assert main(["v2-eval", "--config", str(config_file), "--experiment", "negs"]) == 0
    row = read_ablation(cfg)[-1]
    assert row["name"] == "negs" and row["compare_recall"] == 0.5
    pooled = row["results"]["pooled"]["model_at_compare_recall"]
    assert pooled["recall"] >= 0.5
    assert "| negs |" in (cfg.paths.reports / "ablation.md").read_text(encoding="utf-8")
    assert main(["v2-eval", "--config", str(config_file)]) == 2  # neither experiment nor name
