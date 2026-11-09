import json

import numpy as np

from crackqa.config import parse_config
from crackqa.crossdata import labelled_pairs, score_dataset, write_report
from crackqa.models import build_classifier, save_model

from conftest import crack_pair, write


def test_second_dataset_is_scored_without_retraining(dataset, tmp_path):
    images, masks = tmp_path / "cfd" / "img", tmp_path / "cfd" / "gt"
    for i in range(3):
        img, mask = crack_pair(40 + i, 320, 480)
        write(images / f"00{i}.jpg", img)
        write(masks / f"00{i}.png", mask)
    write(images / "301.jpg", crack_pair(50, 320, 480)[0])  # no ground truth: skipped
    assert len(labelled_pairs(images, masks, ".png")) == 3

    raw = dataset.config_dict(tmp_path / "reports")
    raw["cross_dataset"] = {
        "name": "CFD",
        "images": str(images),
        "masks": str(masks),
        "mask_suffix": ".png",
    }
    cfg = parse_config(raw)
    save_model(build_classifier(pretrained=False), {"threshold": 0.5}, cfg.paths.outputs)
    cfg.paths.reports.mkdir(parents=True)
    (cfg.paths.reports / "baseline_val.json").write_text(json.dumps({"threshold": 0.01}))

    report = score_dataset(cfg)
    assert report["images"] == 3
    assert 0 < report["tiles"] <= 3 * 6  # 480x320 gives 3 x 2 tiles
    assert 0 <= report["image_level_recall"] <= 1
    assert set(report) >= {"resnet18", "opencv_baseline"}
    assert write_report(cfg, report).name == "cross_cfd.json"
    assert np.isfinite(report["resnet18"]["f1"])
