"""crackqa qa-extra: per dataset checks and duplicates against CRACK500, on synthetic files."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from crackqa.cli import main
from crackqa.config import parse_config
from crackqa.extra_qa import run_extra_qa

from conftest import crack_pair, texture, write


def entry(images: Path, regex: str, masks: Path | None = None, **kw) -> dict:
    return {
        "images": str(images),
        "image_exts": [".jpg"],
        "masks": str(masks) if masks else None,
        "mask_name": kw.get("mask_name", [r"^(.*)$", r"\1.png"] if masks else []),
        "invert_mask": kw.get("invert_mask", False),
        "label_dirs": kw.get("label_dirs", {}),
        "source_id_regex": regex,
    }


def config_with(dataset, tmp_path: Path, extra: dict):
    raw = dataset.config_dict(tmp_path / "reports")
    raw["extra_data"] = extra
    raw["v2"]["datasets"] = {}
    return parse_config(raw)


def test_inverted_renamed_masks_pair_and_count(dataset, tmp_path):
    root = tmp_path / "survey"
    for i, stem in enumerate(["01a", "01b", "02a"]):
        img, mask = crack_pair(100 + i)
        write(root / "img" / "SYS" / f"Im_GT_SYS_{stem}or.jpg", img)
        write(root / "gt" / "SYS" / f"GT_SYS_{stem}.png", 255 - mask)  # black crack on white
    write(root / "img" / "SYS" / "Im_GT_SYS_03aor.jpg", crack_pair(110)[0])  # no mask
    write(root / "gt" / "SYS" / "GT_SYS_09a.png", np.zeros((8, 8), np.uint8))  # no image
    (root / "img" / "SYS" / "Im_GT_SYS_04aor.jpg").write_bytes(b"")  # corrupt
    cfg = config_with(
        dataset,
        tmp_path,
        {
            "SURVEY": entry(
                root / "img",
                r"^Im_GT_([A-Z_]+?_\d+)[a-z]",
                root / "gt",
                mask_name=[r"^(\w+)/Im_(GT_\w+?)(?:or)?$", r"\1/\2.png"],
                invert_mask=True,
            )
        },
    )
    report = run_extra_qa(cfg)
    e, s = report["datasets"]["SURVEY"], report["summary"]["SURVEY"]
    assert s["images"] == 5 and s["corrupt"] == 1 and s["crack_images"] == 3
    assert e["pairing"]["missing_mask"] == ["SURVEY/SYS/Im_GT_SYS_03aor.jpg"]
    assert e["pairing"]["unpaired_mask_files"] == ["SYS/GT_SYS_09a.png"]
    assert 0 < e["crack_pixel_ratio"] < 0.05  # the thin line, not the white background
    assert s["sources"] == 4  # 01a and 01b are parts of one image; 03, 04 have no mask
    assert s["greyscale_images"] == 4


def test_folder_labels_and_sources_without_crack(dataset, tmp_path):
    root = tmp_path / "patches"
    write(root / "CP" / "7001-1.jpg", crack_pair(200)[0])
    write(root / "UP" / "7001-2.jpg", texture(201))
    write(root / "UP" / "7002-1.jpg", texture(202))
    write(root / "UP" / "7002-2.jpg", texture(203))
    write(root / "7003-1.jpg", texture(204))  # outside the label folders
    cfg = config_with(
        dataset,
        tmp_path,
        {"PATCHES": entry(root, r"^(\d+)-", label_dirs={"CP": "crack", "UP": "no_crack"})},
    )
    report = run_extra_qa(cfg)
    s = report["summary"]["PATCHES"]
    assert (s["crack_images"], s["no_crack_images"], s["unlabelled"]) == (1, 3, 1)
    assert s["sources"] == 3 and s["sources_without_crack"] == 1  # 7002 has no crack patch
    assert report["datasets"]["PATCHES"]["unlabelled"] == ["PATCHES/7003-1.jpg"]


def test_rotated_copy_of_a_crack500_crop_is_a_duplicate(dataset, tmp_path):
    crop = dataset.root / "traincrop" / "20160222_081011_1_1.jpg"
    rotated = np.ascontiguousarray(np.rot90(cv2.imdecode(np.fromfile(crop, np.uint8), 1)))
    root = tmp_path / "other"
    write(root / "img" / "0001.jpg", rotated)  # saved again as JPEG, so not byte identical
    write(root / "gt" / "0001.png", np.zeros(rotated.shape[:2], np.uint8))
    write(root / "img" / "0002.jpg", crack_pair(300)[0])
    write(root / "gt" / "0002.png", crack_pair(300)[1])
    cfg = config_with(dataset, tmp_path, {"OTHER": entry(root / "img", r"^(\d+)\.", root / "gt")})
    dup = run_extra_qa(cfg)["duplicates"]
    assert len(dup["cross_dataset"]) == 1
    pair = dup["cross_dataset"][0]
    assert pair["a"] == "CRACK500/traincrop/20160222_081011_1_1.jpg"
    assert pair["b"] == "OTHER/0001.jpg" and pair["ncc"] >= 0.9
    assert run_extra_qa(cfg)["summary"]["OTHER"]["duplicate_pairs_cross_dataset"] == 1


def test_texture_check_rejects_dhash_collisions(dataset, tmp_path):
    """Two photos with the same lighting gradient share every dHash bit, not their texture."""
    ramp = np.tile(np.linspace(30, 220, 640, dtype=np.float32), (360, 1))
    root = tmp_path / "ramps"
    for i in (1, 2):
        noise = np.random.default_rng(i).normal(0, 12, ramp.shape).astype(np.float32)
        write(root / f"{i:04d}.jpg", np.clip(ramp + noise, 0, 255).astype(np.uint8))
    cfg = config_with(
        dataset,
        tmp_path,
        {"RAMPS": entry(root, r"^(\d+)\.", label_dirs={"x": "crack"})},
    )
    dup = run_extra_qa(cfg)["duplicates"]
    assert dup["candidates"] >= 1
    assert dup["cross_dataset"] == [] and dup["within_dataset_other_source"] == []
    closest = dup["closest_unconfirmed"][0]
    assert {closest["a"], closest["b"]} == {"RAMPS/0001.jpg", "RAMPS/0002.jpg"}
    assert closest["distance"] == 0 and closest["ncc"] < 0.3


def test_missing_dataset_folder_is_reported(dataset, tmp_path):
    cfg = config_with(
        dataset, tmp_path, {"GONE": entry(tmp_path / "nowhere", r"^(\d+)", tmp_path / "nomask")}
    )
    report = run_extra_qa(cfg)
    assert report["summary"]["GONE"] == {"missing_folders": 2}


def test_cli_writes_both_reports(dataset, tmp_path):
    root = tmp_path / "small"
    img, mask = crack_pair(400)
    write(root / "img" / "0001.jpg", img)
    write(root / "gt" / "0001.png", mask)
    raw = dataset.config_dict(tmp_path / "reports")
    raw["extra_data"] = {"SMALL": entry(root / "img", r"^(\d+)\.", root / "gt")}
    raw["v2"]["datasets"] = {}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "out"
    assert main(["qa-extra", "--config", str(config_file), "--out", str(out)]) == 0
    md = (out / "qa_extra.md").read_text(encoding="utf-8")
    assert (out / "qa_extra.json").exists() and "## SMALL" in md
