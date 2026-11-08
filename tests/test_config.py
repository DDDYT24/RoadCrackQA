import copy

import pytest
import yaml

from crackqa.config import ConfigError, load_config, parse_config


def test_default_config_loads(default_config_path):
    cfg = load_config(default_config_path)
    assert set(cfg.data.splits) == {"train", "val", "test"}
    assert cfg.qa.near_dup_max_hamming == 4
    assert ".jpg" in cfg.data.image_exts


@pytest.fixture
def raw(default_config_path):
    return yaml.safe_load(default_config_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.pop("qa"), "missing keys"),
        (lambda r: r.update(extra=1), "unknown keys"),
        (lambda r: r["data"]["splits"].pop("val"), "data.splits"),
        (lambda r: r["data"].update(source_id_regex=r"^\d{8}"), "capture group"),
        (lambda r: r["data"].update(source_id_regex="("), "source_id_regex"),
        (lambda r: r["data"].update(image_exts=["jpg"]), "start with"),
        (lambda r: r["qa"].update(near_dup_max_hamming=65), "between 0 and 64"),
        (lambda r: r.update(seed=True), "integer"),
        (lambda r: r["paths"].update(raw=""), "paths.raw"),
    ],
)
def test_invalid_config_rejected(raw, mutate, message):
    bad = copy.deepcopy(raw)
    mutate(bad)
    with pytest.raises(ConfigError, match=message):
        parse_config(bad)
