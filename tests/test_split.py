import shutil

import pytest

from crackqa.cli import main
from crackqa.config import parse_config
from crackqa.split import (
    LeakageError,
    assert_no_source_overlap,
    file_sha256,
    make_split,
    manifest_path,
    read_test_lock,
)
from crackqa.tiling import build_tile_table, read_rows, write_rows


def _row(source, split):
    return {"source_id": source, "split": split}


def test_no_overlap_passes_and_injected_leak_fails():
    rows = [_row("a", "train"), _row("a", "train"), _row("b", "val"), _row("c", "test")]
    assert_no_source_overlap(rows)
    with pytest.raises(LeakageError, match="1 source ids"):
        assert_no_source_overlap(rows + [_row("a", "test")])


def _tiles(cfg):
    return write_rows(build_tile_table(cfg), cfg.paths.processed / "tiles.csv")


def test_split_on_dataset_writes_manifests_balance_and_lock(dataset, tmp_path):
    cfg = dataset.config(tmp_path / "reports")
    result = make_split(cfg, _tiles(cfg))

    train = read_rows(manifest_path(cfg, "train"))
    assert len(train) == 3 * 10  # three 640x360 crops, 10 tiles each
    assert {r["source_id"] for r in train} == {"20160222_081011", "20160222_081102"}
    assert sum(b["crack"] + b["no_crack"] + b["ambiguous"] for b in result["balance"].values())
    assert (tmp_path / "reports" / "balance.md").exists()
    assert read_test_lock(cfg) == file_sha256(manifest_path(cfg, "test"))


def test_split_refuses_a_leak_and_exclusion_fixes_it(dataset, tmp_path):
    for ext in (".jpg", ".png"):
        shutil.copy(
            dataset.root / "traincrop" / f"20160222_081011_1_1{ext}",
            dataset.root / "testcrop" / f"20160222_081011_1_1{ext}",
        )
    raw = dataset.config_dict(tmp_path / "reports")
    cfg = parse_config(raw)
    with pytest.raises(LeakageError):
        make_split(cfg, _tiles(cfg))

    raw["data"]["exclude_sources"] = {"test": ["20160222_081011"]}
    cfg = parse_config(raw)
    result = make_split(cfg, _tiles(cfg))
    assert result["balance"]["test"]["excluded_tiles"] == 10
    assert all(r["source_id"] != "20160222_081011" for r in read_rows(manifest_path(cfg, "test")))


def test_manifest_and_lock_are_deterministic(dataset, tmp_path):
    cfg = dataset.config(tmp_path / "reports")
    first = make_split(cfg, _tiles(cfg))["test_lock"]
    assert make_split(cfg, _tiles(cfg))["test_lock"] == first


def test_cli_tile_split_baseline_chain(dataset, tmp_path):
    config = str(dataset.config_file(tmp_path / "reports"))
    assert main(["tile", "--config", config]) == 0
    assert main(["split", "--config", config]) == 0
    assert main(["baseline", "--config", config]) == 0
    assert (tmp_path / "reports" / "baseline_val.json").exists()
    assert (tmp_path / "reports" / "test_lock.txt").exists()
