import json
import subprocess

import pytest
import yaml

from crackqa.cli import main
from crackqa.config import config_digest, load_config
from crackqa.evaluate import FinalRunRefused, evaluate
from crackqa.models import build_classifier, save_model
from crackqa.split import manifest_path


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def ready(dataset, tmp_path):
    """Tiles, split, baseline and a saved (untrained) model, with config and lock committed."""
    config_file = dataset.config_file(tmp_path / "reports")
    for command in ("tile", "split", "baseline"):
        assert main([command, "--config", str(config_file)]) == 0
    cfg = load_config(config_file)
    meta = {"threshold": 0.5, "config_digest": config_digest(cfg)}
    save_model(build_classifier(pretrained=False), meta, cfg.paths.outputs)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "config.yaml", "reports/test_lock.txt")
    _git(tmp_path, "commit", "-q", "-m", "lock")
    return config_file


def test_val_evaluation_compares_three_methods(ready):
    report = evaluate(load_config(ready), "val", final=False, config_file=ready)
    assert set(report["results"]) == {"resnet18", "opencv_baseline", "majority_class"}
    assert report["results"]["majority_class"]["tp"] == 0


def test_test_split_needs_final(ready):
    with pytest.raises(FinalRunRefused, match="--final"):
        evaluate(load_config(ready), "test", final=False, config_file=ready)


def test_final_runs_once_and_logs_the_commit(ready, tmp_path):
    cfg = load_config(ready)
    report = evaluate(cfg, "test", final=True, config_file=ready)
    assert report["git_commit"] and report["test_lock"]
    assert (tmp_path / "reports" / "predictions_test.csv").exists()
    with pytest.raises(FinalRunRefused, match="already exist"):
        evaluate(cfg, "test", final=True, config_file=ready)


def test_final_refuses_when_the_config_changed(ready):
    raw = yaml.safe_load(ready.read_text(encoding="utf-8"))
    raw["train"]["lr"] = 0.5
    ready.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(FinalRunRefused, match="config changed") as exc:
        evaluate(load_config(ready), "test", final=True, config_file=ready)
    assert "uncommitted changes" in str(exc.value)


def test_final_refuses_when_the_test_manifest_changed(ready):
    cfg = load_config(ready)
    path = manifest_path(cfg, "test")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FinalRunRefused, match="does not match the committed test lock"):
        evaluate(cfg, "test", final=True, config_file=ready)


def test_cli_reports_refusal_with_exit_code(ready):
    assert main(["eval", "--config", str(ready), "--split", "test"]) == 3


def test_val_report_files(ready, tmp_path):
    assert main(["eval", "--config", str(ready), "--split", "val"]) == 0
    report = json.loads((tmp_path / "reports" / "val_results.json").read_text(encoding="utf-8"))
    assert report["split"] == "val"


def test_analysis_reproduces_the_final_counts(ready, tmp_path):
    from crackqa.analysis import run_analysis

    cfg = load_config(ready)
    evaluate(cfg, "test", final=True, config_file=ready)
    report = run_analysis(cfg)
    assert report["image_level"]["crops"] > 0
    lo, hi = report["bootstrap"]["resnet18"]["ci95"]
    assert lo <= hi
    assert (tmp_path / "reports" / "figures" / "confusion_test.png").exists()
    assert (tmp_path / "reports" / "error_candidates.csv").exists()
    assert (tmp_path / "outputs" / "figures" / "error_grid.png").exists()
