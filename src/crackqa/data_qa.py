"""Data QA: inventory, corrupt files, duplicates, mask pairing and values, sizes, source ids."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crackqa.config import Config

MD_LIST_LIMIT = 50


def parse_source_id(filename: str, pattern: str) -> str | None:
    """Source photo id from a file name, or None when the pattern does not match."""
    m = re.search(pattern, filename)
    return m.group(1) if m else None


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash: sign of horizontal gradients on a 9x8 thumbnail."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def read_image(path: Path, flags: int) -> tuple[bytes, np.ndarray | None]:
    """Raw bytes and decoded array (None if empty or undecodable).

    Decodes from bytes rather than cv2.imread, which cannot open non-ASCII paths on Windows;
    the result is the same as cv2.imread returning None.
    """
    data = path.read_bytes()
    if not data:
        return data, None
    return data, cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)


@dataclass
class _Image:
    split: str
    rel: str
    key: str
    sha256: str
    size: tuple[int, int] | None  # (width, height)
    hash: int | None


@dataclass
class _Mask:
    split: str
    rel: str
    key: str
    size: tuple[int, int] | None
    values: list[int] | None
    crack_px: int
    total_px: int


def _is_binary(values: list[int]) -> bool:
    return set(values) <= {0, 255} or set(values) <= {0, 1}


def _scan_split(root: Path, split: str, folder: str, cfg: Config, report: dict) -> tuple:
    images: list[_Image] = []
    masks: list[_Mask] = []
    directory = root / folder
    if not directory.is_dir():
        report["missing_split_dirs"].append(f"{split}: {folder}")
        return images, masks

    suffix = cfg.data.mask_suffix.lower()
    by_ext: Counter[str] = Counter()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        name = path.name.lower()
        by_ext[path.suffix.lower() or "(none)"] += 1

        if name.endswith(suffix):
            key = path.relative_to(directory).as_posix()[: -len(suffix)]
            _, arr = read_image(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                report["corrupt"].append(_corrupt(split, rel, path))
                masks.append(_Mask(split, rel, key, None, None, 0, 0))
                continue
            plane = arr if arr.ndim == 2 else arr.max(axis=2)
            masks.append(
                _Mask(
                    split,
                    rel,
                    key,
                    (arr.shape[1], arr.shape[0]),
                    [int(v) for v in np.unique(arr)],
                    int(np.count_nonzero(plane)),
                    int(plane.size),
                )
            )
        elif path.suffix.lower() in cfg.data.image_exts:
            key = path.relative_to(directory).with_suffix("").as_posix()
            data, arr = read_image(path, cv2.IMREAD_COLOR)
            sha = hashlib.sha256(data).hexdigest()
            if arr is None:
                report["corrupt"].append(_corrupt(split, rel, path))
                images.append(_Image(split, rel, key, sha, None, None))
                continue
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            images.append(_Image(split, rel, key, sha, (arr.shape[1], arr.shape[0]), dhash(gray)))
        else:
            report["other_files"].append(rel)

    report["inventory"][split] = {
        "folder": folder,
        "images": len(images),
        "masks": len(masks),
        "by_extension": dict(sorted(by_ext.items())),
    }
    return images, masks


def _corrupt(split: str, rel: str, path: Path) -> dict:
    reason = "zero_bytes" if path.stat().st_size == 0 else "unreadable"
    return {"split": split, "file": rel, "reason": reason}


def _pairing(split: str, images: list[_Image], masks: list[_Mask], report: dict) -> None:
    mask_by_key = {m.key: m for m in masks}
    image_keys = {i.key for i in images}
    for img in images:
        mask = mask_by_key.get(img.key)
        if mask is None:
            report["pairing"]["missing_mask"].append(img.rel)
        elif img.size and mask.size and img.size != mask.size:
            report["pairing"]["size_mismatch"].append(
                {"image": img.rel, "mask": mask.rel, "image_wh": img.size, "mask_wh": mask.size}
            )
    for mask in masks:
        if mask.key not in image_keys:
            report["pairing"]["orphan_mask"].append(mask.rel)

    readable = [m for m in masks if m.values is not None]
    for m in readable:
        if not _is_binary(m.values):
            report["non_binary_masks"].append(
                {"file": m.rel, "n_values": len(m.values), "values_head": m.values[:10]}
            )
    crack = sum(m.crack_px for m in readable)
    total = sum(m.total_px for m in readable)
    report["mask_balance"][split] = {
        "crack_pixel_ratio": round(crack / total, 6) if total else None,
        "masks_without_crack": sum(1 for m in readable if m.crack_px == 0),
    }
    report["size_histogram"][split] = dict(
        Counter(f"{i.size[0]}x{i.size[1]}" for i in images if i.size).most_common()
    )


def _duplicates(images: list[_Image], max_hamming: int, report: dict) -> None:
    by_sha: dict[str, list[_Image]] = defaultdict(list)
    for img in images:
        by_sha[img.sha256].append(img)
    for sha, group in sorted(by_sha.items()):
        if len(group) > 1:
            report["exact_duplicates"].append(
                {
                    "sha256": sha,
                    "files": [i.rel for i in group],
                    "cross_split": len({i.split for i in group}) > 1,
                }
            )

    hashed = [i for i in images if i.hash is not None]
    if len(hashed) < 2:
        return
    hashes = np.array([i.hash for i in hashed], dtype=np.uint64)
    for a in range(len(hashed) - 1):
        dist = np.bitwise_count(hashes[a] ^ hashes[a + 1 :])
        for offset in np.flatnonzero(dist <= max_hamming):
            b = a + 1 + int(offset)
            ia, ib = hashed[a], hashed[b]
            if ia.sha256 == ib.sha256:
                continue  # already listed as an exact duplicate
            report["near_duplicates"].append(
                {
                    "a": ia.rel,
                    "b": ib.rel,
                    "distance": int(dist[offset]),
                    "cross_split": ia.split != ib.split,
                }
            )


def _source_ids(images: list[_Image], pattern: str, report: dict) -> None:
    ids: dict[str, Counter[str]] = defaultdict(Counter)
    for img in images:
        sid = parse_source_id(Path(img.rel).name, pattern)
        if sid is None:
            report["source_ids"]["unmatched"].append(img.rel)
        else:
            ids[img.split][sid] += 1
    per_split = {}
    for split, counts in ids.items():
        per_split[split] = {
            "unique": len(counts),
            "images_per_source_min": min(counts.values()),
            "images_per_source_max": max(counts.values()),
        }
    report["source_ids"]["per_split"] = per_split
    overlap = {}
    for s1, s2 in combinations(sorted(ids), 2):
        shared = sorted(set(ids[s1]) & set(ids[s2]))
        if shared:
            overlap[f"{s1}/{s2}"] = shared
    report["source_ids"]["cross_split_overlap"] = overlap


def run_qa(cfg: Config) -> dict[str, Any]:
    root = cfg.paths.raw
    report: dict[str, Any] = {
        "dataset_root": root.as_posix(),
        "settings": {
            "splits": cfg.data.splits,
            "mask_suffix": cfg.data.mask_suffix,
            "source_id_regex": cfg.data.source_id_regex,
            "near_dup_max_hamming": cfg.qa.near_dup_max_hamming,
        },
        "missing_split_dirs": [],
        "inventory": {},
        "other_files": [],
        "corrupt": [],
        "pairing": {"missing_mask": [], "orphan_mask": [], "size_mismatch": []},
        "non_binary_masks": [],
        "mask_balance": {},
        "size_histogram": {},
        "exact_duplicates": [],
        "near_duplicates": [],
        "source_ids": {"unmatched": []},
    }

    all_images: list[_Image] = []
    for split, folder in cfg.data.splits.items():
        images, masks = _scan_split(root, split, folder, cfg, report)
        _pairing(split, images, masks, report)
        all_images.extend(images)

    _duplicates(all_images, cfg.qa.near_dup_max_hamming, report)
    _source_ids(all_images, cfg.data.source_id_regex, report)

    pairing = report["pairing"]
    report["summary"] = {
        "images": len(all_images),
        "masks": sum(v["masks"] for v in report["inventory"].values()),
        "missing_split_dirs": len(report["missing_split_dirs"]),
        "corrupt": len(report["corrupt"]),
        "missing_mask": len(pairing["missing_mask"]),
        "orphan_mask": len(pairing["orphan_mask"]),
        "size_mismatch": len(pairing["size_mismatch"]),
        "non_binary_masks": len(report["non_binary_masks"]),
        "exact_duplicate_groups": len(report["exact_duplicates"]),
        "exact_duplicate_groups_cross_split": sum(
            d["cross_split"] for d in report["exact_duplicates"]
        ),
        "near_duplicate_pairs": len(report["near_duplicates"]),
        "near_duplicate_pairs_cross_split": sum(
            d["cross_split"] for d in report["near_duplicates"]
        ),
        "source_id_unmatched": len(report["source_ids"]["unmatched"]),
        "source_id_cross_split_overlap": sum(
            len(v) for v in report["source_ids"]["cross_split_overlap"].values()
        ),
    }
    return report


def _md_list(lines: list[str], items: list, fmt) -> None:
    if not items:
        lines.append("None.\n")
        return
    for item in items[:MD_LIST_LIMIT]:
        lines.append(f"- {fmt(item)}")
    if len(items) > MD_LIST_LIMIT:
        lines.append(f"- ... and {len(items) - MD_LIST_LIMIT} more (see the JSON report)")
    lines.append("")


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Data QA report", "", f"Dataset root: `{report['dataset_root']}`", ""]

    lines += ["## Summary", "", "| Check | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in report["summary"].items()]
    lines.append("")

    lines += ["## Inventory", "", "| Split | Folder | Images | Masks | By extension |"]
    lines.append("|---|---|---|---|---|")
    for split, inv in report["inventory"].items():
        exts = ", ".join(f"{k}: {v}" for k, v in inv["by_extension"].items())
        lines.append(f"| {split} | {inv['folder']} | {inv['images']} | {inv['masks']} | {exts} |")
    lines.append("")
    if report["missing_split_dirs"]:
        lines += ["Missing split folders:", ""]
        _md_list(lines, report["missing_split_dirs"], str)

    lines += ["## Image sizes (width x height)", ""]
    for split, hist in report["size_histogram"].items():
        sizes = ", ".join(f"{k}: {v}" for k, v in hist.items()) or "none"
        lines.append(f"- {split}: {sizes}")
    lines.append("")

    lines += ["## Mask balance", "", "| Split | Crack pixel ratio | Masks without crack |"]
    lines.append("|---|---|---|")
    for split, bal in report["mask_balance"].items():
        lines.append(f"| {split} | {bal['crack_pixel_ratio']} | {bal['masks_without_crack']} |")
    lines.append("")

    lines += ["## Source ids", "", "| Split | Unique sources | Images per source (min to max) |"]
    lines.append("|---|---|---|")
    for split, s in report["source_ids"]["per_split"].items():
        span = f"{s['images_per_source_min']} to {s['images_per_source_max']}"
        lines.append(f"| {split} | {s['unique']} | {span} |")
    lines += ["", "Cross split overlap:", ""]
    overlap = [
        f"{k}: {', '.join(v)}" for k, v in report["source_ids"]["cross_split_overlap"].items()
    ]
    _md_list(lines, overlap, str)
    lines += ["Files the source id pattern does not match:", ""]
    _md_list(lines, report["source_ids"]["unmatched"], lambda f: f"`{f}`")

    lines += ["## Corrupt files", ""]
    _md_list(lines, report["corrupt"], lambda c: f"`{c['file']}` ({c['reason']})")

    lines += ["## Mask pairing", "", "Images without a mask:", ""]
    _md_list(lines, report["pairing"]["missing_mask"], lambda f: f"`{f}`")
    lines += ["Masks without an image:", ""]
    _md_list(lines, report["pairing"]["orphan_mask"], lambda f: f"`{f}`")
    lines += ["Size mismatches:", ""]
    _md_list(
        lines,
        report["pairing"]["size_mismatch"],
        lambda m: f"`{m['image']}` {m['image_wh']} vs mask {m['mask_wh']}",
    )
    lines += ["Non binary masks:", ""]
    _md_list(
        lines,
        report["non_binary_masks"],
        lambda m: f"`{m['file']}` ({m['n_values']} values, first: {m['values_head']})",
    )

    lines += ["## Exact duplicates (SHA256)", ""]
    _md_list(
        lines,
        report["exact_duplicates"],
        lambda d: ", ".join(f"`{f}`" for f in d["files"]) + (" (cross split)" * d["cross_split"]),
    )
    lines += [
        f"## Near duplicates (dHash, distance <= {report['settings']['near_dup_max_hamming']})"
    ]
    lines.append("")
    near = sorted(report["near_duplicates"], key=lambda d: (not d["cross_split"], d["distance"]))
    _md_list(
        lines,
        near,
        lambda d: (
            f"`{d['a']}` ~ `{d['b']}` (distance {d['distance']})"
            + (" (cross split)" * d["cross_split"])
        ),
    )
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "qa_report.json"
    md_path = out_dir / "qa_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path
