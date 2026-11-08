import json
import shutil

import cv2
import numpy as np

from crackqa.cli import main
from crackqa.data_qa import dhash, run_qa, write_reports

from conftest import crack_pair, write

PROBLEMS = [
    "missing_split_dirs",
    "corrupt",
    "missing_mask",
    "orphan_mask",
    "size_mismatch",
    "non_binary_masks",
    "exact_duplicate_groups",
    "near_duplicate_pairs",
    "source_id_unmatched",
    "source_id_cross_split_overlap",
]


def test_clean_dataset_has_no_findings(dataset, tmp_path):
    report = run_qa(dataset.config(tmp_path / "reports"))
    s = report["summary"]
    assert (s["images"], s["masks"]) == (5, 5)
    assert all(s[k] == 0 for k in PROBLEMS), s
    assert report["inventory"]["train"]["by_extension"] == {".jpg": 3, ".png": 3}
    assert report["size_histogram"]["train"] == {"640x360": 3}
    assert report["source_ids"]["per_split"]["train"] == {
        "unique": 2,
        "images_per_source_min": 1,
        "images_per_source_max": 2,
    }
    assert 0 < report["mask_balance"]["train"]["crack_pixel_ratio"] < 0.1


def test_flags_zero_byte_jpg_and_truncated_png(dataset, tmp_path):
    folder = dataset.root / "traindata"
    (folder / "20160222_090000_1_1.jpg").write_bytes(b"")
    png = write(folder / "20160222_090100_1_1.png", crack_pair(99)[0])
    data = png.read_bytes()
    png.write_bytes(data[: len(data) // 2])

    report = run_qa(dataset.config(tmp_path / "reports"))
    reasons = {c["file"]: c["reason"] for c in report["corrupt"]}
    assert reasons == {
        "traindata/20160222_090000_1_1.jpg": "zero_bytes",
        "traindata/20160222_090100_1_1.png": "unreadable",
    }


def test_finds_exact_and_near_duplicates(dataset, tmp_path):
    train = dataset.root / "traindata"
    # exact copy leaks into test
    shutil.copy(
        train / "20160222_081011_1_1.jpg", dataset.root / "testdata" / "20160222_081011_1_1.jpg"
    )
    shutil.copy(
        train / "20160222_081011_1_1_mask.png",
        dataset.root / "testdata" / "20160222_081011_1_1_mask.png",
    )
    # slightly brighter re-encode of another train image goes to val
    img = cv2.imread(str(train / "20160222_081102_1_1.jpg"))
    write(dataset.root / "valdata" / "20160222_081102_1_1.png", cv2.add(img, 3))
    shutil.copy(
        train / "20160222_081102_1_1_mask.png",
        dataset.root / "valdata" / "20160222_081102_1_1_mask.png",
    )

    report = run_qa(dataset.config(tmp_path / "reports"))
    exact = report["exact_duplicates"]
    assert len(exact) == 1 and exact[0]["cross_split"]
    assert sorted(exact[0]["files"]) == [
        "testdata/20160222_081011_1_1.jpg",
        "traindata/20160222_081011_1_1.jpg",
    ]
    near = report["near_duplicates"]
    assert len(near) == 1
    assert {near[0]["a"], near[0]["b"]} == {
        "traindata/20160222_081102_1_1.jpg",
        "valdata/20160222_081102_1_1.png",
    }
    assert near[0]["cross_split"] and near[0]["distance"] <= 4
    overlap = report["source_ids"]["cross_split_overlap"]
    assert overlap == {"test/train": ["20160222_081011"], "train/val": ["20160222_081102"]}


def test_reports_missing_mask_size_mismatch_and_non_binary_mask(dataset, tmp_path):
    folder = dataset.root / "traindata"
    dataset.add("train", "20160222_100000_1_1", mask=False)
    _, mask_path = dataset.add("val", "20160222_100100_1_1")
    write(mask_path, np.zeros((100, 100), np.uint8))
    _, mask_path = dataset.add("test", "20160222_100200_1_1")
    soft = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    soft[soft == 255] = 128
    soft[0, 0] = 255
    write(mask_path, soft)
    write(folder / "20160222_100300_1_1_mask.png", np.zeros((360, 640), np.uint8))

    report = run_qa(dataset.config(tmp_path / "reports"))
    pairing = report["pairing"]
    assert pairing["missing_mask"] == ["traindata/20160222_100000_1_1.jpg"]
    assert pairing["orphan_mask"] == ["traindata/20160222_100300_1_1_mask.png"]
    assert [m["image"] for m in pairing["size_mismatch"]] == ["valdata/20160222_100100_1_1.jpg"]
    assert pairing["size_mismatch"][0]["mask_wh"] == (100, 100)
    non_binary = report["non_binary_masks"]
    assert [m["file"] for m in non_binary] == ["testdata/20160222_100200_1_1_mask.png"]
    assert non_binary[0]["values_head"] == [0, 128, 255]


def test_missing_split_folder_is_reported(dataset, tmp_path):
    shutil.rmtree(dataset.root / "valdata")
    report = run_qa(dataset.config(tmp_path / "reports"))
    assert report["missing_split_dirs"] == ["val: valdata"]


def test_dhash_is_stable_under_small_changes_and_differs_across_textures():
    a = cv2.cvtColor(crack_pair(1)[0], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(crack_pair(2)[0], cv2.COLOR_BGR2GRAY)
    assert (dhash(a) ^ dhash(cv2.add(a, 5))).bit_count() <= 4
    assert (dhash(a) ^ dhash(b)).bit_count() > 4


def test_write_reports_and_cli(dataset, tmp_path):
    out = tmp_path / "reports"
    json_path, md_path = write_reports(run_qa(dataset.config(out)), out)
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["images"] == 5
    assert md_path.read_text(encoding="utf-8").startswith("# Data QA report")

    cli_out = tmp_path / "cli_reports"
    assert main(["qa", "--config", str(dataset.config_file(out)), "--out", str(cli_out)]) == 0
    assert (cli_out / "qa_report.json").exists() and (cli_out / "qa_report.md").exists()


def test_cli_fails_when_dataset_folder_is_missing(dataset, tmp_path):
    config = dataset.config_file(tmp_path / "reports")
    shutil.rmtree(dataset.root)
    assert main(["qa", "--config", str(config), "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
