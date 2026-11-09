import json
import shutil

import cv2
import numpy as np

from crackqa.cli import main
from crackqa.config import parse_config
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
    folder = dataset.root / "traincrop"
    (folder / "20160222_090000_1_1.jpg").write_bytes(b"")
    write(folder / "20160222_090000_1_1.png", crack_pair(98)[1])
    _, mask_path = dataset.add("train", "20160222_090100_1_1")
    data = mask_path.read_bytes()
    mask_path.write_bytes(data[: len(data) // 2])

    report = run_qa(dataset.config(tmp_path / "reports"))
    reasons = {c["file"]: c["reason"] for c in report["corrupt"]}
    assert reasons == {
        "traincrop/20160222_090000_1_1.jpg": "zero_bytes",
        "traincrop/20160222_090100_1_1.png": "unreadable",
    }


def _copy_pair(src_dir, stem, dst_dir, new_stem=None):
    for ext in (".jpg", ".png"):
        shutil.copy(src_dir / f"{stem}{ext}", dst_dir / f"{new_stem or stem}{ext}")


def test_finds_exact_and_near_duplicates(dataset, tmp_path):
    train = dataset.root / "traincrop"
    _copy_pair(train, "20160222_081011_1_1", dataset.root / "testcrop")  # exact copy into test
    img = cv2.imread(str(train / "20160222_081102_1_1.jpg"))  # brighter re-encode into val
    write(dataset.root / "valcrop" / "20160222_081102_1_1.jpg", cv2.add(img, 3))
    shutil.copy(train / "20160222_081102_1_1.png", dataset.root / "valcrop")

    report = run_qa(dataset.config(tmp_path / "reports"))
    exact = report["exact_duplicates"]
    assert len(exact) == 1 and exact[0]["cross_split"]
    assert sorted(exact[0]["files"]) == [
        "testcrop/20160222_081011_1_1.jpg",
        "traincrop/20160222_081011_1_1.jpg",
    ]
    near = report["near_duplicates"]
    assert len(near) == 1
    assert {near[0]["a"], near[0]["b"]} == {
        "traincrop/20160222_081102_1_1.jpg",
        "valcrop/20160222_081102_1_1.jpg",
    }
    assert near[0]["cross_split"] and near[0]["distance"] <= 4
    overlap = report["source_ids"]["cross_split_overlap"]
    assert overlap == {"test/train": ["20160222_081011"], "train/val": ["20160222_081102"]}


def test_finds_rotated_copy_saved_under_another_name(dataset, tmp_path):
    # CRACK500 has photos that reappear rotated, under the next second's timestamp
    train = dataset.root / "traincrop"
    img = cv2.imread(str(train / "20160222_081102_1_1.jpg"))
    write(dataset.root / "testcrop" / "20160222_081103_1_1.jpg", cv2.rotate(img, cv2.ROTATE_180))
    write(dataset.root / "testcrop" / "20160222_081103_1_1.png", np.zeros((360, 640), np.uint8))

    near = run_qa(dataset.config(tmp_path / "reports"))["near_duplicates"]
    assert len(near) == 1
    assert near[0]["transform"] == "rot180" and near[0]["cross_split"]


def test_exclusions_clear_cross_split_findings(dataset, tmp_path):
    train = dataset.root / "traincrop"
    _copy_pair(train, "20160222_081011_1_1", dataset.root / "testcrop")
    raw = dataset.config_dict(tmp_path / "reports")

    before = run_qa(parse_config(raw))["summary"]
    assert before["exact_duplicate_groups_cross_split_after_exclusions"] == 1
    assert before["source_id_cross_split_overlap_after_exclusions"] == 1

    raw["data"]["exclude_sources"] = {"test": ["20160222_081011"]}
    after = run_qa(parse_config(raw))["summary"]
    assert after["excluded_images"] == 1
    assert after["exact_duplicate_groups_cross_split_after_exclusions"] == 0
    assert after["near_duplicate_pairs_cross_split_after_exclusions"] == 0
    assert after["source_id_cross_split_overlap_after_exclusions"] == 0
    assert after["exact_duplicate_groups_cross_split"] == 1  # the raw finding stays visible


def test_reports_missing_mask_size_mismatch_and_non_binary_mask(dataset, tmp_path):
    folder = dataset.root / "traincrop"
    dataset.add("train", "20160222_100000_1_1", mask=False)
    _, mask_path = dataset.add("val", "20160222_100100_1_1")
    write(mask_path, np.zeros((100, 100), np.uint8))
    _, mask_path = dataset.add("test", "20160222_100200_1_1")
    soft = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    soft[soft == 255] = 128
    soft[0, 0] = 255
    write(mask_path, soft)
    write(folder / "20160222_100300_1_1.png", np.zeros((360, 640), np.uint8))

    report = run_qa(dataset.config(tmp_path / "reports"))
    pairing = report["pairing"]
    assert pairing["missing_mask"] == ["traincrop/20160222_100000_1_1.jpg"]
    assert pairing["orphan_mask"] == ["traincrop/20160222_100300_1_1.png"]
    assert [m["image"] for m in pairing["size_mismatch"]] == ["valcrop/20160222_100100_1_1.jpg"]
    assert pairing["size_mismatch"][0]["mask_wh"] == (100, 100)
    non_binary = report["non_binary_masks"]
    assert [m["file"] for m in non_binary] == ["testcrop/20160222_100200_1_1.png"]
    assert non_binary[0]["values_head"] == [0, 128, 255]


def test_missing_split_folder_is_reported(dataset, tmp_path):
    shutil.rmtree(dataset.root / "valcrop")
    report = run_qa(dataset.config(tmp_path / "reports"))
    assert report["missing_split_dirs"] == ["val: valcrop"]


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
