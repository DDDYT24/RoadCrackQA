"""Load and validate the YAML config.

Each section is a frozen dataclass. Keys and value types are checked from the type hints;
range checks live in each section's __post_init__.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SPLITS = ("train", "val", "test")


class ConfigError(ValueError):
    """The config file is missing a key, has an unknown key, or holds a bad value."""


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ConfigError(message)


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    processed: Path
    outputs: Path
    reports: Path


@dataclass(frozen=True)
class DataConfig:
    splits: dict[str, str]
    image_exts: tuple[str, ...]
    mask_suffix: str
    source_id_regex: str
    exclude_sources: dict[str, tuple[str, ...]]

    def __post_init__(self):
        _check(
            set(self.splits) == set(REQUIRED_SPLITS),
            f"data.splits: expected exactly the keys {list(REQUIRED_SPLITS)}",
        )
        object.__setattr__(self, "image_exts", tuple(e.lower() for e in self.image_exts))
        _check(
            all(e.startswith(".") for e in self.image_exts),
            "data.image_exts: each extension must start with '.'",
        )
        try:
            groups = re.compile(self.source_id_regex).groups
        except re.error as exc:
            raise ConfigError(f"data.source_id_regex: {exc}") from exc
        _check(groups >= 1, "data.source_id_regex: needs a capture group for the source id")
        _check(
            set(self.exclude_sources) <= set(REQUIRED_SPLITS),
            "data.exclude_sources: keys must be split names",
        )


@dataclass(frozen=True)
class QAConfig:
    near_dup_max_hamming: int

    def __post_init__(self):
        _check(
            0 <= self.near_dup_max_hamming <= 64,
            "qa.near_dup_max_hamming: must be between 0 and 64",
        )


@dataclass(frozen=True)
class TilingConfig:
    tile_size: int
    stride: int
    drop_edge: bool

    def __post_init__(self):
        _check(self.tile_size > 0 and self.stride > 0, "tiling: tile_size and stride must be > 0")


@dataclass(frozen=True)
class LabelsConfig:
    crack_min_px: int

    def __post_init__(self):
        _check(self.crack_min_px >= 1, "labels.crack_min_px: must be >= 1")


@dataclass(frozen=True)
class BaselineConfig:
    clahe_clip: float
    clahe_grid: int
    blur_kernel: int
    blackhat_kernel: int
    threshold_percentile: float
    min_component_px: int

    def __post_init__(self):
        _check(self.blackhat_kernel % 2 == 1, "baseline.blackhat_kernel: must be odd")
        _check(
            self.blur_kernel == 0 or self.blur_kernel % 2 == 1,
            "baseline.blur_kernel: must be 0 or odd",
        )
        _check(self.clahe_grid >= 1, "baseline.clahe_grid: must be >= 1")
        _check(
            0 < self.threshold_percentile < 100,
            "baseline.threshold_percentile: must be between 0 and 100",
        )


@dataclass(frozen=True)
class TrainConfig:
    pretrained: bool
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    max_train_tiles: int
    patience: int
    finetune_layer4_epochs: int
    finetune_lr: float

    def __post_init__(self):
        _check(self.epochs >= 1 and self.batch_size >= 1, "train: epochs and batch_size >= 1")
        _check(self.lr > 0 and self.finetune_lr > 0, "train: learning rates must be > 0")


@dataclass(frozen=True)
class Config:
    seed: int
    paths: PathsConfig
    data: DataConfig
    qa: QAConfig
    tiling: TilingConfig
    labels: LabelsConfig
    baseline: BaselineConfig
    train: TrainConfig


def _coerce(value: Any, tp: Any, name: str) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, name)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(value, inner, name)
    if tp is bool:
        _check(isinstance(value, bool), f"{name}: expected true or false, got {value!r}")
        return value
    if tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
        _check(ok, f"{name}: expected an integer, got {value!r}")
        return value
    if tp is float:
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        _check(ok, f"{name}: expected a number, got {value!r}")
        return float(value)
    if tp in (str, Path):
        ok = isinstance(value, str) and bool(value)
        _check(ok, f"{name}: expected a non-empty string, got {value!r}")
        return Path(value) if tp is Path else value
    if origin is tuple:
        _check(isinstance(value, list), f"{name}: expected a list")
        return tuple(_coerce(v, args[0], f"{name}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        _check(isinstance(value, dict), f"{name}: expected a mapping")
        return {str(k): _coerce(v, args[1], f"{name}.{k}") for k, v in value.items()}
    raise TypeError(f"unsupported config type {tp!r} for {name}")


def _build(cls: type, raw: Any, name: str) -> Any:
    _check(isinstance(raw, dict), f"{name}: expected a mapping")
    keys = {f.name for f in dataclasses.fields(cls)}
    missing, unknown = keys - raw.keys(), raw.keys() - keys
    _check(not missing, f"{name}: missing keys {sorted(missing)}")
    _check(not unknown, f"{name}: unknown keys {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    prefix = "" if name == "config" else f"{name}."
    return cls(**{k: _coerce(raw[k], hints[k], f"{prefix}{k}") for k in keys})


def parse_config(raw: Any) -> Config:
    return _build(Config, raw, "config")


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return parse_config(yaml.safe_load(f))


def config_digest(cfg: Config) -> str:
    """SHA256 of the parsed config, stable across key order and formatting."""
    as_dict = dataclasses.asdict(cfg)
    text = json.dumps(as_dict, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
