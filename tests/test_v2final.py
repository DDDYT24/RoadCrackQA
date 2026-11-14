"""The single V2 test run: guards, one run only, report contents, rehearsal on val."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest
from test_v2train import _raw

from crackqa.cli import main
from crackqa.config import parse_config
from crackqa.evaluate import FinalRunRefused
from crackqa.models import build_classifier, save_model
from crackqa.v2data import make_v2_split, manifest_v2_path
from crackqa.v2final import bootstrap, results_path, run_final
from crackqa.v2train import train_v2


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def ready(dataset, tmp_path):
    """V2 split, a trained experiment, a V1 model on record; config, lock, report committed."""
    raw = _raw(dataset, tmp_path)
    cfg = parse_config(raw)
    make_v2_split(cfg)
    reports = cfg.paths.reports
    (reports / "baseline_val.json").write_text(json.dumps({"threshold": 0.01}))
    train_v2(cfg, "base")
    v1 = save_model(build_classifier(pretrained=False), {"threshold": 0.5}, cfg.paths.outputs)
    (reports / "test_results.json").write_text(json.dumps({"model_version": v1["model_version"]}))
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "config.json", "reports/test2_lock.txt", "reports/v2_train_base.json")
    _git(tmp_path, "commit", "-q", "-m", "lock")
    return cfg, config_file


def test_final_runs_once(ready):
    cfg, config_file = ready
    report = run_final(cfg, "base", config_file)
    assert report["split"] == "test" and report["test_lock"]
    assert set(report["results"]["pooled"]) == {"v1", "v2", "opencv_baseline", "v2_at_v1_recall"}
    assert report["photo_level"]["crack_free_photos"] >= 1  # the patch photos without a crack
    assert set(report["acceptance"]) >= {"met", "higher_f1"}
    reports = cfg.paths.reports
    for name in ("test2_results.md", "predictions_test2.csv"):
        assert (reports / name).exists()
    assert (reports / "figures" / "confusion_test2.png").exists()
    assert results_path(cfg).exists()
    with pytest.raises(FinalRunRefused, match="already exist"):
        run_final(cfg, "base", config_file)
    assert main(["v2-final", "--config", str(config_file), "--experiment", "base"]) == 3


def test_final_refuses_a_changed_manifest_or_model(ready):
    cfg, config_file = ready
    path = manifest_v2_path(cfg, "test")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    (cfg.paths.reports / "test_results.json").write_text(json.dumps({"model_version": "other"}))
    with pytest.raises(FinalRunRefused) as exc:
        run_final(cfg, "base", config_file)
    assert "does not match the committed test lock" in str(exc.value)
    assert "V1 model file is not the one" in str(exc.value)
    assert not results_path(cfg).exists()


def test_final_refuses_uncommitted_config(ready):
    cfg, config_file = ready
    config_file.write_text(config_file.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FinalRunRefused, match="uncommitted changes"):
        run_final(cfg, "base", config_file)


def test_rehearsal_writes_outside_reports(ready):
    cfg, config_file = ready
    assert (
        main(["v2-final", "--config", str(config_file), "--experiment", "base", "--rehearse"]) == 0
    )
    assert (cfg.v2_train.outputs / "rehearsal" / "val2_rehearsal_results.json").exists()
    assert not results_path(cfg).exists()


def test_bootstrap_is_paired_and_stratified():
    rows = [
        {"dataset": d, "source_id": str(i), "label": label}
        for d in ("A", "B")
        for i in range(6)
        for label in (0, 1)
    ]
    y = np.array([r["label"] for r in rows], bool)
    flags = {"v1": y.copy(), "v2": y.copy(), "opencv_baseline": ~y}
    flags["v1"][1] = False  # V1 misses one crack tile
    out = bootstrap(rows, flags, n=200, seed=0)
    assert out["photos"] == 12
    assert out["v2"]["f1"] == {"value": 1.0, "ci95": [1.0, 1.0]}
    diff = out["v2_minus_v1"]["recall"]
    assert diff["value"] == pytest.approx(1 / 12, abs=1e-4) and diff["ci95"][0] >= 0
