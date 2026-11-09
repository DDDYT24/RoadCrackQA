"""Grouped split manifests: official split minus excluded sources, leakage check, test lock."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from crackqa.config import REQUIRED_SPLITS, Config
from crackqa.tiling import AMBIGUOUS, CRACK, NO_CRACK, read_rows, write_rows


class LeakageError(AssertionError):
    """A source photo has tiles in more than one split."""


def manifest_path(cfg: Config, split: str) -> Path:
    return cfg.paths.processed / f"manifest_{split}.csv"


def test_lock_path(cfg: Config) -> Path:
    return cfg.paths.reports / "test_lock.txt"


def assert_no_source_overlap(rows: list[dict]) -> None:
    splits_of: dict[str, set[str]] = {}
    for r in rows:
        splits_of.setdefault(r["source_id"], set()).add(r["split"])
    leaked = sorted(s for s, splits in splits_of.items() if len(splits) > 1)
    if leaked:
        raise LeakageError(f"{len(leaked)} source ids in more than one split: {leaked[:10]}")


def apply_exclusions(rows: list[dict], cfg: Config) -> tuple[list[dict], list[dict]]:
    """(kept, dropped) after removing data.exclude_sources from their split."""
    kept, dropped = [], []
    for r in rows:
        excluded = r["source_id"] in cfg.data.exclude_sources.get(r["split"], ())
        (dropped if excluded else kept).append(r)
    return kept, dropped


def balance(kept: list[dict], dropped: list[dict]) -> dict:
    out = {}
    for split in REQUIRED_SPLITS:
        labels = Counter(r["label"] for r in kept if r["split"] == split)
        crack, no_crack = labels[CRACK], labels[NO_CRACK]
        out[split] = {
            "images": len({r["image"] for r in kept if r["split"] == split}),
            "sources": len({r["source_id"] for r in kept if r["split"] == split}),
            "crack": crack,
            "no_crack": no_crack,
            "ambiguous": labels[AMBIGUOUS],
            "crack_share": round(crack / (crack + no_crack), 4) if crack + no_crack else None,
            "excluded_tiles": sum(1 for r in dropped if r["split"] == split),
        }
    return out


def balance_markdown(table: dict, cfg: Config) -> str:
    lines = [
        "# Tile class balance",
        "",
        f"Tiles of {cfg.tiling.tile_size} px, stride {cfg.tiling.stride}. Crack: at least "
        f"{cfg.labels.crack_min_px} crack pixels; no crack: 0; ambiguous: 1 to "
        f"{cfg.labels.crack_min_px - 1} (left out of training and metrics).",
        "",
        "| Split | Images | Sources | Crack | No crack | Ambiguous | Crack share | "
        "Excluded tiles |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split, b in table.items():
        lines.append(
            f"| {split} | {b['images']} | {b['sources']} | {b['crack']} | {b['no_crack']} | "
            f"{b['ambiguous']} | {b['crack_share']} | {b['excluded_tiles']} |"
        )
    return "\n".join(lines) + "\n"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_split(cfg: Config, tiles_csv: Path) -> dict:
    """Write one manifest per split, the balance report and the test lock."""
    kept, dropped = apply_exclusions(read_rows(tiles_csv), cfg)
    assert_no_source_overlap(kept)
    for split in REQUIRED_SPLITS:
        rows = sorted((r for r in kept if r["split"] == split), key=lambda r: r["tile_id"])
        write_rows(rows, manifest_path(cfg, split))

    table = balance(kept, dropped)
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports / "balance.md").write_text(
        balance_markdown(table, cfg), encoding="utf-8", newline="\n"
    )
    lock = file_sha256(manifest_path(cfg, "test"))
    test_lock_path(cfg).write_text(f"{lock}  manifest_test.csv\n", encoding="utf-8", newline="\n")
    return {"balance": table, "test_lock": lock}


def read_test_lock(cfg: Config) -> str:
    return test_lock_path(cfg).read_text(encoding="utf-8").split()[0]
