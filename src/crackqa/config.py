"""Load and validate the YAML config."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SPLITS = ("train", "val", "test")


class ConfigError(ValueError):
    """The config file is missing a key, has an unknown key, or holds a bad value."""


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    reports: Path


@dataclass(frozen=True)
class DataConfig:
    splits: dict[str, str]
    image_exts: tuple[str, ...]
    mask_suffix: str
    source_id_regex: str


@dataclass(frozen=True)
class QAConfig:
    near_dup_max_hamming: int


@dataclass(frozen=True)
class Config:
    seed: int
    paths: PathsConfig
    data: DataConfig
    qa: QAConfig


def _section(raw: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{name}: expected a mapping")
    missing = keys - raw.keys()
    unknown = raw.keys() - keys
    if missing:
        raise ConfigError(f"{name}: missing keys {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{name}: unknown keys {sorted(unknown)}")
    return raw


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name}: expected an integer, got {value!r}")
    return value


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name}: expected a non-empty string, got {value!r}")
    return value


def parse_config(raw: Any) -> Config:
    top = _section(raw, "config", {"seed", "paths", "data", "qa"})

    paths = _section(top["paths"], "paths", {"raw", "reports"})
    data = _section(top["data"], "data", {"splits", "image_exts", "mask_suffix", "source_id_regex"})
    qa = _section(top["qa"], "qa", {"near_dup_max_hamming"})

    splits = data["splits"]
    if not isinstance(splits, dict) or set(splits) != set(REQUIRED_SPLITS):
        raise ConfigError(f"data.splits: expected exactly the keys {list(REQUIRED_SPLITS)}")
    splits = {k: _str(v, f"data.splits.{k}") for k, v in splits.items()}

    exts = data["image_exts"]
    if not isinstance(exts, list) or not exts:
        raise ConfigError("data.image_exts: expected a non-empty list")
    exts = tuple(_str(e, "data.image_exts").lower() for e in exts)
    if not all(e.startswith(".") for e in exts):
        raise ConfigError("data.image_exts: each extension must start with '.'")

    pattern = _str(data["source_id_regex"], "data.source_id_regex")
    try:
        groups = re.compile(pattern).groups
    except re.error as exc:
        raise ConfigError(f"data.source_id_regex: {exc}") from exc
    if groups < 1:
        raise ConfigError("data.source_id_regex: needs a capture group for the source id")

    hamming = _int(qa["near_dup_max_hamming"], "qa.near_dup_max_hamming")
    if not 0 <= hamming <= 64:
        raise ConfigError("qa.near_dup_max_hamming: must be between 0 and 64")

    return Config(
        seed=_int(top["seed"], "seed"),
        paths=PathsConfig(
            raw=Path(_str(paths["raw"], "paths.raw")),
            reports=Path(_str(paths["reports"], "paths.reports")),
        ),
        data=DataConfig(
            splits=splits,
            image_exts=exts,
            mask_suffix=_str(data["mask_suffix"], "data.mask_suffix"),
            source_id_regex=pattern,
        ),
        qa=QAConfig(near_dup_max_hamming=hamming),
    )


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f))
