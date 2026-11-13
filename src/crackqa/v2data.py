"""V2 tile table over CRACK500 and the extra datasets: labels, groups, split, test lock.

The V1 files (tiles.csv, manifest_*.csv, test_lock.txt) are not touched. Image and mask
paths in V2 rows are relative to the directory crackqa runs from, since they span datasets.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from crackqa.config import V2_SPLITS, Config, ExtraDatasetConfig, V2DatasetConfig
from crackqa.data_qa import is_excluded, parse_source_id, read_image
from crackqa.extra_qa import REFERENCE, mask_rel_for
from crackqa.split import LeakageError, file_sha256
from crackqa.tiling import (
    AMBIGUOUS,
    CRACK,
    NO_CRACK,
    _image_files,
    cut,
    mask_path_for,
    read_rows,
    tile_grid,
    tile_label,
    write_rows,
)

V2_COLUMNS = [
    "tile_id",
    "dataset",
    "image",
    "mask",
    "y",
    "x",
    "source_id",
    "group",
    "split",
    "label",
    "crack_px",
    "centerline_px",
]
UNKNOWN = -1  # crack_px of a tile from a patch labelled crack without a mask


def manifest_v2_path(cfg: Config, split: str) -> Path:
    return cfg.v2.processed / f"manifest_{split}.csv"


def test2_lock_path(cfg: Config) -> Path:
    return cfg.paths.reports / "test2_lock.txt"


def centerline(mask: np.ndarray) -> np.ndarray:
    """Centre line of a binary mask by Zhang-Suen thinning: a connected line one pixel wide,
    whatever the width of the drawn crack. Runs on the bounding box of the mask only."""
    m = mask > 0
    out = np.zeros(m.shape, bool)
    ys, xs = np.nonzero(m)
    if not len(ys):
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = np.pad(m[y0:y1, x0:x1].astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            # neighbours P2..P9, clockwise from the pixel above
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = sum(p.astype(np.int16) for p in ring[:8])
            turns = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                kept = (p2 & p4 & p6) | (p4 & p6 & p8)
            else:
                kept = (p2 & p4 & p8) | (p2 & p6 & p8)
            centre = img[1:-1, 1:-1]
            remove = (centre == 1) & (neighbours >= 2) & (neighbours <= 6) & (turns == 1)
            remove &= kept == 0
            if remove.any():
                centre[remove] = 0
                changed = True
    out[y0:y1, x0:x1] = img[1:-1, 1:-1].astype(bool)
    return out


def label_of(crack_px: int, centerline_px: int, cfg: Config) -> int:
    if crack_px == UNKNOWN:
        return AMBIGUOUS
    if crack_px == 0:
        return NO_CRACK
    if cfg.v2.label_rule == "pixels":
        return tile_label(crack_px, cfg.labels.crack_min_px)
    return CRACK if centerline_px >= cfg.v2.centerline_min_px else AMBIGUOUS


def _mask_tiles(
    dataset: str, image: Path, mask_file: Path, invert: bool, source: str, cfg: Config
) -> list[dict]:
    _, mask = read_image(mask_file, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"unreadable mask {mask_file}; run crackqa qa-extra first")
    crack = (mask == 0) if invert else (mask > 0)
    line = centerline(crack)
    t = cfg.tiling
    rows = []
    for y, x in tile_grid(*mask.shape[:2], t.tile_size, t.stride, t.drop_edge):
        rows.append(
            {
                "dataset": dataset,
                "image": image.as_posix(),
                "mask": mask_file.as_posix(),
                "y": y,
                "x": x,
                "source_id": source,
                "crack_px": int(np.count_nonzero(cut(crack, y, x, t.tile_size))),
                "centerline_px": int(np.count_nonzero(cut(line, y, x, t.tile_size))),
            }
        )
    return rows


def _crack500_rows(cfg: Config) -> list[dict]:
    """Official train and val crops, minus data.exclude_sources; the test crops stay out."""
    rows = []
    for split in ("train", "val"):
        for image in _image_files(cfg.paths.raw / cfg.data.splits[split], cfg):
            if is_excluded(cfg, split, image.name):
                continue
            source = parse_source_id(image.name, cfg.data.source_id_regex)
            if source is None:
                raise ValueError(f"no source id in {image.name}; run crackqa qa first")
            for r in _mask_tiles(REFERENCE, image, mask_path_for(image, cfg), False, source, cfg):
                rows.append({**r, "group": source, "split": split})
    return rows


def _extra_rows(name: str, ds: ExtraDatasetConfig, cfg: Config) -> tuple[list[dict], int]:
    """Tiles of one extra dataset, without group or split; also the images skipped (no mask)."""
    rows, skipped = [], 0
    t = cfg.tiling
    for path in sorted(p for p in ds.images.rglob("*") if p.is_file()):
        if path.suffix.lower() not in ds.image_exts:
            continue
        rel_in = path.relative_to(ds.images).as_posix()
        source = parse_source_id(path.name, ds.source_id_regex)
        if source is None:
            raise ValueError(f"no source id in {path.name}; run crackqa qa-extra first")
        if ds.masks is not None:
            mask_file = ds.masks / mask_rel_for(ds, rel_in)
            if not mask_file.is_file():
                skipped += 1
                continue
            rows += _mask_tiles(name, path, mask_file, ds.invert_mask, source, cfg)
            continue
        label = ds.label_dirs.get(rel_in.split("/")[0]) if "/" in rel_in else None
        if label is None:
            skipped += 1
            continue
        _, img = read_image(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"unreadable image {path}; run crackqa qa-extra first")
        crack_px = 0 if label == "no_crack" else UNKNOWN
        for y, x in tile_grid(*img.shape[:2], t.tile_size, t.stride, t.drop_edge):
            rows.append(
                {
                    "dataset": name,
                    "image": path.as_posix(),
                    "mask": "",
                    "y": y,
                    "x": x,
                    "source_id": source,
                    "crack_px": crack_px,
                    "centerline_px": crack_px,
                }
            )
    return rows, skipped


def group_of(sources: list[str], gap: int, block: int) -> dict[str, str]:
    """Group per source id: runs of consecutive numbers (gap), or fixed blocks of numbers."""
    parsed = {}
    for sid in sorted(set(sources)):
        m = re.fullmatch(r"(.*?)(\d+)", sid)
        parsed[sid] = (m.group(1), int(m.group(2))) if m else None
    out = {}
    if block:
        for sid, p in parsed.items():
            out[sid] = f"{p[0]}#{(p[1] - 1) // block}" if p else sid
        return out
    numbered = sorted((p[0], p[1], sid) for sid, p in parsed.items() if p)
    out.update({sid: sid for sid, p in parsed.items() if not p})
    start, prev = None, None
    for prefix, n, sid in numbered:
        if gap == 0 or prev is None or prefix != prev[0] or n - prev[1] > gap:
            start = sid
        out[sid] = start
        prev = (prefix, n)
    return out


def assign_groups(
    sizes: dict[str, int], crack_free: set[str], d: V2DatasetConfig, seed: int, name: str
) -> dict[str, str]:
    """Split per group: crack-free groups to test if asked, then each group in a seeded order
    to the split furthest below its share of tiles."""
    order = sorted(sizes)
    random.Random(f"{seed}:{name}").shuffle(order)
    shares = {"train": d.train, "val": d.val, "test": d.test}
    total = sum(sizes.values())
    done = dict.fromkeys(V2_SPLITS, 0)
    out = {}
    if d.crack_free_to_test:
        for g in order:
            if g in crack_free:
                out[g] = "test"
                done["test"] += sizes[g]
    for g in order:
        if g in out:
            continue
        split = max(
            (s for s in V2_SPLITS if shares[s] > 0), key=lambda s: shares[s] * total - done[s]
        )
        out[g] = split
        done[split] += sizes[g]
    return out


def assert_no_leak(rows: list[dict]) -> None:
    for key in ("source_id", "group"):
        splits: dict[tuple, set[str]] = defaultdict(set)
        for r in rows:
            splits[(r["dataset"], r[key])].add(r["split"])
        leaked = sorted(k for k, s in splits.items() if len(s) > 1)
        if leaked:
            raise LeakageError(f"{len(leaked)} {key}s in more than one split: {leaked[:10]}")


def build_rows(cfg: Config) -> tuple[list[dict], dict]:
    """Every V2 tile with its dataset, group, split and label; also images skipped per dataset."""
    rows = _crack500_rows(cfg)
    skipped = {}
    for name, d in cfg.v2.datasets.items():
        extra, skipped[name] = _extra_rows(name, cfg.extra_data[name], cfg)
        groups = group_of([r["source_id"] for r in extra], d.gap, d.block)
        sizes: Counter[str] = Counter()
        has_crack: set[str] = set()
        for r in extra:
            sizes[groups[r["source_id"]]] += 1
            if r["crack_px"] != 0:
                has_crack.add(r["source_id"])
        crack_free = {groups[s] for s in groups if s not in has_crack}
        split_of = assign_groups(dict(sizes), crack_free, d, cfg.seed, name)
        for r in extra:
            r["group"] = groups[r["source_id"]]
            r["split"] = split_of[r["group"]]
        rows += extra
    for r in rows:
        r["tile_id"] = f"{r['image']}@{r['y']},{r['x']}"
        r["label"] = label_of(r["crack_px"], r["centerline_px"], cfg)
    return rows, skipped


def balance(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for (dataset, split), group in sorted(_by(rows, ("dataset", "split")).items()):
        labels = Counter(r["label"] for r in group)
        out.setdefault(dataset, {})[split] = {
            "images": len({r["image"] for r in group}),
            "sources": len({r["source_id"] for r in group}),
            "groups": len({r["group"] for r in group}),
            "sources_without_crack": len(
                {r["source_id"] for r in group}
                - {r["source_id"] for r in group if r["crack_px"] != 0}
            ),
            "tiles": len(group),
            "crack": labels[CRACK],
            "no_crack": labels[NO_CRACK],
            "ambiguous": labels[AMBIGUOUS],
        }
    return out


def _by(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def balance_markdown(table: dict, skipped: dict, cfg: Config) -> str:
    v = cfg.v2
    rule = (
        f"crack: at least {v.centerline_min_px} pixels of the mask's centre line"
        if v.label_rule == "centerline"
        else f"crack: at least {cfg.labels.crack_min_px} mask pixels"
    )
    lines = [
        "# V2 tile balance",
        "",
        f"Tiles of {cfg.tiling.tile_size} px. Label rule `{v.label_rule}`: {rule}; no crack: no "
        "mask pixel; ambiguous: the rest, and every tile of a patch labelled crack without a "
        "mask. Ambiguous tiles are left out of training and metrics.",
        "",
        "| Dataset | Split | Images | Sources | Groups | Sources without crack | Tiles | Crack "
        "| No crack | Ambiguous |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset, splits in table.items():
        for split in V2_SPLITS:
            if split not in splits:
                continue
            b = splits[split]
            lines.append(
                f"| {dataset} | {split} | {b['images']} | {b['sources']} | {b['groups']} | "
                f"{b['sources_without_crack']} | {b['tiles']} | {b['crack']} | {b['no_crack']} | "
                f"{b['ambiguous']} |"
            )
    lines.append("")
    skips = [f"{k}: {n}" for k, n in skipped.items() if n]
    if skips:
        lines += ["Images skipped (no mask, or outside the label folders): " + ", ".join(skips)]
        lines.append("")
    return "\n".join(lines)


def make_v2_split(cfg: Config) -> dict:
    """Write the V2 manifests, reports/v2_balance.md and reports/test2_lock.txt."""
    rows, skipped = build_rows(cfg)
    assert_no_leak(rows)
    for split in V2_SPLITS:
        part = sorted((r for r in rows if r["split"] == split), key=lambda r: r["tile_id"])
        write_rows(part, manifest_v2_path(cfg, split), V2_COLUMNS)
    table = balance(rows)
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports / "v2_balance.md").write_text(
        balance_markdown(table, skipped, cfg), encoding="utf-8", newline="\n"
    )
    lock = file_sha256(manifest_v2_path(cfg, "test"))
    test2_lock_path(cfg).write_text(
        f"{lock}  {manifest_v2_path(cfg, 'test').name} (v2)\n", encoding="utf-8", newline="\n"
    )
    return {"balance": table, "skipped": skipped, "test_lock": lock}


def read_v2_rows(path: Path) -> list[dict]:
    rows = read_rows(path)
    for r in rows:
        r["centerline_px"] = int(r["centerline_px"])
    return rows
