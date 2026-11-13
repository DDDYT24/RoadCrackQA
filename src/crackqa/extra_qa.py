"""QA of the extra datasets: per dataset checks, and duplicates against every other dataset.

The CRACK500 crops are hashed as the reference; pairs inside CRACK500 are left to
`crackqa qa` (reports/qa_report.md).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crackqa.config import Config, ExtraDatasetConfig
from crackqa.data_qa import (
    TRANSFORMS,
    _corrupt,
    _duplicates,
    _Image,
    _is_binary,
    _md_list,
    dihedral_dhashes,
    parse_source_id,
    read_image,
)

REFERENCE = "CRACK500"
THUMB = 128  # px; one picture at any size lines up with itself once rotated or flipped back
CLOSEST = 20  # unconfirmed candidate pairs listed for review


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file())


def _hash(path: Path, dataset: str, rel: str) -> tuple[_Image, np.ndarray | None]:
    data, arr = read_image(path, cv2.IMREAD_COLOR)
    sha = hashlib.sha256(data).hexdigest()
    if arr is None:
        return _Image(dataset, rel, rel, sha, None, None), None
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    size = (arr.shape[1], arr.shape[0])
    return _Image(dataset, rel, rel, sha, size, dihedral_dhashes(gray)), arr


def _is_grey(arr: np.ndarray) -> bool:
    return bool(
        np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2])
    )


def _mask_label(
    ds: ExtraDatasetConfig, rel_in: str, img: _Image, entry: dict, used: set[str]
) -> str | None:
    """crack / no_crack from the image's mask, recording pairing problems in entry."""
    stem = Path(rel_in).with_suffix("").as_posix()
    mask_rel = re.sub(ds.mask_name[0], ds.mask_name[1], stem)
    mask_path = ds.masks / mask_rel
    pairing = entry["pairing"]
    if not mask_path.is_file():
        pairing["missing_mask"].append(img.rel)
        return None
    used.add(mask_rel)
    _, arr = read_image(mask_path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        entry["corrupt"].append(_corrupt(img.split, f"{img.split}/masks/{mask_rel}", mask_path))
        return None
    plane = arr if arr.ndim == 2 else arr.max(axis=2)
    values = [int(v) for v in np.unique(arr)]
    if not _is_binary(values):
        pairing["non_binary_masks"].append(
            {"file": mask_rel, "n_values": len(values), "values_head": values[:10]}
        )
    if img.size != (arr.shape[1], arr.shape[0]):
        pairing["size_mismatch"].append(
            {"image": img.rel, "image_wh": img.size, "mask_wh": (arr.shape[1], arr.shape[0])}
        )
    crack = int(np.count_nonzero(plane == 0 if ds.invert_mask else plane))
    entry["_crack_px"] += crack
    entry["_total_px"] += int(plane.size)
    return "crack" if crack else "no_crack"


def _scan_dataset(name: str, ds: ExtraDatasetConfig) -> tuple[dict, list[_Image], dict]:
    """Checks of one extra dataset; returns its entry, its images and their source ids."""
    entry: dict[str, Any] = {
        "images_dir": ds.images.as_posix(),
        "masks_dir": ds.masks.as_posix() if ds.masks is not None else None,
        "labels_from": "mask" if ds.masks is not None else "folder",
        "missing": [
            d.as_posix() for d in (ds.images, ds.masks) if d is not None and not d.is_dir()
        ],
    }
    if entry["missing"]:
        return entry, [], {}
    entry.update(
        images=0,
        by_extension={},
        other_files=[],
        corrupt=[],
        greyscale_images=0,
        size_histogram={},
        labels={},
        unlabelled=[],
        pairing={
            "missing_mask": [],
            "unpaired_mask_files": [],
            "size_mismatch": [],
            "non_binary_masks": [],
        },
        _crack_px=0,
        _total_px=0,
    )
    images: list[_Image] = []
    label_of: dict[str, str] = {}
    by_ext: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    used_masks: set[str] = set()

    for path in _files(ds.images):
        rel_in = path.relative_to(ds.images).as_posix()
        by_ext[path.suffix.lower() or "(none)"] += 1
        if path.suffix.lower() not in ds.image_exts:
            entry["other_files"].append(rel_in)
            continue
        img, arr = _hash(path, name, f"{name}/{rel_in}")
        images.append(img)
        if arr is None:
            entry["corrupt"].append(_corrupt(name, img.rel, path))
            continue
        sizes[f"{img.size[0]}x{img.size[1]}"] += 1
        entry["greyscale_images"] += _is_grey(arr)
        if ds.masks is not None:
            label = _mask_label(ds, rel_in, img, entry, used_masks)
        else:
            top = rel_in.split("/")[0] if "/" in rel_in else None
            label = ds.label_dirs.get(top) if top else None
            if label is None:
                entry["unlabelled"].append(img.rel)
        if label is not None:
            label_of[img.rel] = label

    if ds.masks is not None:
        entry["pairing"]["unpaired_mask_files"] = [
            rel
            for rel in (p.relative_to(ds.masks).as_posix() for p in _files(ds.masks))
            if rel not in used_masks
        ]
    crack_px, total_px = entry.pop("_crack_px"), entry.pop("_total_px")
    entry["crack_pixel_ratio"] = round(crack_px / total_px, 6) if total_px else None
    entry["images"] = len(images)
    entry["by_extension"] = dict(sorted(by_ext.items()))
    entry["size_histogram"] = dict(sizes.most_common())
    entry["labels"] = dict(sorted(Counter(label_of.values()).items()))

    source_of: dict[str, str] = {}
    per_source: Counter[str] = Counter()
    labels_by_source: dict[str, set[str]] = defaultdict(set)
    unmatched = []
    for img in images:
        sid = parse_source_id(Path(img.rel).name, ds.source_id_regex)
        if sid is None:
            unmatched.append(img.rel)
            continue
        source_of[img.rel] = sid
        per_source[sid] += 1
        if img.rel in label_of:
            labels_by_source[sid].add(label_of[img.rel])
    entry["source_ids"] = {
        "unique": len(per_source),
        "images_per_source_min": min(per_source.values(), default=0),
        "images_per_source_max": max(per_source.values(), default=0),
        "sources_with_crack": sum("crack" in s for s in labels_by_source.values()),
        "sources_without_crack": sum(s == {"no_crack"} for s in labels_by_source.values()),
        "unmatched": unmatched,
    }
    return entry, images, source_of


def _scan_reference(cfg: Config) -> tuple[dict, list[_Image], dict]:
    images: list[_Image] = []
    source_of: dict[str, str] = {}
    missing = []
    for folder in cfg.data.splits.values():
        directory = cfg.paths.raw / folder
        if not directory.is_dir():
            missing.append(directory.as_posix())
            continue
        for path in _files(directory):
            if path.suffix.lower() in cfg.data.image_exts:
                img, _ = _hash(path, REFERENCE, f"{REFERENCE}/{folder}/{path.name}")
                images.append(img)
                sid = parse_source_id(path.name, cfg.data.source_id_regex)
                if sid is not None:
                    source_of[img.rel] = sid
    return {"images": len(images), "missing_dirs": missing}, images, source_of


def _dataset_of(rel: str) -> str:
    return rel.split("/", 1)[0]


def _path_of(rel: str, cfg: Config) -> Path:
    dataset, rest = rel.split("/", 1)
    root = cfg.paths.raw if dataset == REFERENCE else cfg.extra_data[dataset].images
    return root / rest


def _detail(path: Path) -> np.ndarray:
    """Square grey thumbnail minus its blur: fine texture that only the same picture shares."""
    _, gray = read_image(path, cv2.IMREAD_GRAYSCALE)
    thumb = cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)
    detail = thumb - cv2.GaussianBlur(thumb, (0, 0), 3)
    return detail - detail.mean()


def _ncc(a: np.ndarray, b: np.ndarray, transform: str) -> float:
    """Correlation of a with b after the transform under which their dHashes matched."""
    i = TRANSFORMS.index(transform)
    b = np.rot90(b[:, ::-1] if i >= 4 else b, i % 4)
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den else 0.0


def _pairs(images: list[_Image], cfg: Config, source_of: dict[str, str]) -> dict:
    """Duplicate pairs, except those inside CRACK500: dHash candidates, confirmed by texture."""
    found: dict[str, list] = {"exact_duplicates": [], "near_duplicates": []}
    _duplicates(images, cfg.qa.near_dup_max_hamming, found)
    pairs = []
    for group in found["exact_duplicates"]:
        first, *rest = group["files"]
        pairs += [{"a": first, "b": b, "distance": 0, "transform": "identical"} for b in rest]
    pairs += [
        {k: d[k] for k in ("a", "b", "distance", "transform")} for d in found["near_duplicates"]
    ]

    details: dict[str, np.ndarray] = {}

    def detail(rel: str) -> np.ndarray:
        if rel not in details:
            details[rel] = _detail(_path_of(rel, cfg))
        return details[rel]

    out: dict[str, Any] = {
        "candidates": 0,
        "cross_dataset": [],
        "within_dataset_other_source": [],
        "same_source": 0,
    }
    unconfirmed = []
    for p in sorted(pairs, key=lambda p: (p["distance"], p["a"], p["b"])):
        da, db = _dataset_of(p["a"]), _dataset_of(p["b"])
        if da == db == REFERENCE:
            continue
        out["candidates"] += 1
        if p["transform"] == "identical":
            p["ncc"] = 1.0
        else:
            p["ncc"] = round(_ncc(detail(p["a"]), detail(p["b"]), p["transform"]), 4)
        if p["ncc"] < cfg.qa.confirm_min_ncc:
            unconfirmed.append(p)
            continue
        sa, sb = source_of.get(p["a"]), source_of.get(p["b"])
        if da != db:
            out["cross_dataset"].append(p)
        elif sa is not None and sa == sb:
            out["same_source"] += 1
        else:
            out["within_dataset_other_source"].append(p)
    out["closest_unconfirmed"] = sorted(unconfirmed, key=lambda p: -p["ncc"])[:CLOSEST]
    return out


def run_extra_qa(cfg: Config) -> dict[str, Any]:
    reference, all_images, source_of = _scan_reference(cfg)
    datasets = {}
    for name, ds in cfg.extra_data.items():
        entry, images, sources = _scan_dataset(name, ds)
        datasets[name] = entry
        all_images += images
        source_of.update(sources)

    duplicates = _pairs(all_images, cfg, source_of)
    summary = {}
    for name, e in datasets.items():
        if e["missing"]:
            summary[name] = {"missing_folders": len(e["missing"])}
            continue
        involved = [
            p
            for p in duplicates["cross_dataset"]
            if name in (_dataset_of(p["a"]), _dataset_of(p["b"]))
        ]
        summary[name] = {
            "images": e["images"],
            "corrupt": len(e["corrupt"]),
            "greyscale_images": e["greyscale_images"],
            "crack_images": e["labels"].get("crack", 0),
            "no_crack_images": e["labels"].get("no_crack", 0),
            "unlabelled": len(e["unlabelled"]),
            "missing_mask": len(e["pairing"]["missing_mask"]),
            "unpaired_mask_files": len(e["pairing"]["unpaired_mask_files"]),
            "size_mismatch": len(e["pairing"]["size_mismatch"]),
            "non_binary_masks": len(e["pairing"]["non_binary_masks"]),
            "sources": e["source_ids"]["unique"],
            "sources_without_crack": e["source_ids"]["sources_without_crack"],
            "source_id_unmatched": len(e["source_ids"]["unmatched"]),
            "duplicate_pairs_cross_dataset": len(involved),
            "duplicate_pairs_other_source": sum(
                _dataset_of(p["a"]) == name for p in duplicates["within_dataset_other_source"]
            ),
        }
    return {
        "settings": {
            "near_dup_max_hamming": cfg.qa.near_dup_max_hamming,
            "confirm_min_ncc": cfg.qa.confirm_min_ncc,
            "reference": f"{REFERENCE} crops of every split (data.splits under paths.raw)",
        },
        "reference": reference,
        "datasets": datasets,
        "duplicates": duplicates,
        "summary": summary,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Extra data QA report", ""]
    ref = report["reference"]
    lines += [
        f"Reference: {report['settings']['reference']}, {ref['images']} images.",
        "Pairs inside CRACK500 are in `qa_report.md`.",
        "",
    ]
    if ref["missing_dirs"]:
        lines += ["Missing reference folders:", ""]
        _md_list(lines, ref["missing_dirs"], lambda d: f"`{d}`")

    names = list(report["summary"])
    checks = list(dict.fromkeys(k for s in report["summary"].values() for k in s))
    lines += ["## Summary", "", "| Check | " + " | ".join(names) + " |"]
    lines.append("|---|" + "---|" * len(names))
    for key in checks:
        cells = [str(report["summary"][n].get(key, "-")) for n in names]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.append("")

    for name, e in report["datasets"].items():
        lines += [f"## {name}", ""]
        if e["missing"]:
            lines += ["Missing folders:", ""]
            _md_list(lines, e["missing"], lambda d: f"`{d}`")
            continue
        sizes = ", ".join(f"{k}: {v}" for k, v in e["size_histogram"].items()) or "none"
        exts = ", ".join(f"{k}: {v}" for k, v in e["by_extension"].items())
        s = e["source_ids"]
        lines += [
            f"- Images: `{e['images_dir']}`; labels from {e['labels_from']}"
            + (f" (`{e['masks_dir']}`)" if e["masks_dir"] else ""),
            f"- Files by extension: {exts}",
            f"- Sizes (width x height): {sizes}",
            f"- Labels: {e['labels']}; crack pixel ratio: {e['crack_pixel_ratio']}",
            f"- Source photos: {s['unique']} ({s['images_per_source_min']} to "
            f"{s['images_per_source_max']} images each); {s['sources_with_crack']} with a crack, "
            f"{s['sources_without_crack']} without",
            "",
        ]
        problems = {
            "Corrupt files": [f"`{c['file']}` ({c['reason']})" for c in e["corrupt"]],
            "Images without a mask": [f"`{f}`" for f in e["pairing"]["missing_mask"]],
            "Mask folder files no image uses": [
                f"`{f}`" for f in e["pairing"]["unpaired_mask_files"]
            ],
            "Size mismatches": [
                f"`{m['image']}` {m['image_wh']} vs mask {m['mask_wh']}"
                for m in e["pairing"]["size_mismatch"]
            ],
            "Non binary masks": [
                f"`{m['file']}` ({m['n_values']} values, first: {m['values_head']})"
                for m in e["pairing"]["non_binary_masks"]
            ],
            "Images outside the label folders": [f"`{f}`" for f in e["unlabelled"]],
            "Files the source id pattern does not match": [f"`{f}`" for f in s["unmatched"]],
            "Other files": [f"`{f}`" for f in e["other_files"]],
        }
        for title, items in problems.items():
            if items:
                lines += [f"{title}:", ""]
                _md_list(lines, items, str)

    dup = report["duplicates"]
    settings = report["settings"]

    def fmt(p: dict) -> str:
        return (
            f"`{p['a']}` ~ `{p['b']}` (distance {p['distance']}, {p['transform']}, "
            f"texture correlation {p['ncc']})"
        )

    lines += [
        "## Duplicates",
        "",
        f"Candidates: same SHA256, or dHash distance <= {settings['near_dup_max_hamming']} under "
        "any rotation or flip. A candidate is a duplicate when the fine texture of the two "
        f"aligned {THUMB} px thumbnails correlates at least {settings['confirm_min_ncc']}. "
        f"Candidates checked: {dup['candidates']}.",
        "",
        "Across datasets, CRACK500 included:",
        "",
    ]
    _md_list(lines, dup["cross_dataset"], fmt)
    lines += ["Inside one dataset, between different source photos:", ""]
    _md_list(lines, dup["within_dataset_other_source"], fmt)
    lines += [
        f"Inside one source photo (overlapping crops are expected): {dup['same_source']}.",
        "",
        f"Closest candidates that are not duplicates (up to {CLOSEST}, for review by eye):",
        "",
    ]
    _md_list(lines, dup["closest_unconfirmed"], fmt)
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "qa_extra.json"
    md_path = out_dir / "qa_extra.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(to_markdown(report), encoding="utf-8", newline="\n")
    return json_path, md_path
