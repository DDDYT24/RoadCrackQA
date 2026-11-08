import pytest

from crackqa.config import load_config
from crackqa.data_qa import parse_source_id


@pytest.mark.parametrize(
    "name, expected",
    [
        ("20160222_081011.jpg", "20160222_081011"),  # full photo
        ("20160222_081011_1_361.jpg", "20160222_081011"),  # crop of that photo
        ("20160222_081011_1_361_mask.png", "20160222_081011"),  # its mask
        ("20160307_164211_641_1.jpg", "20160307_164211"),
        ("IMG_0001.jpg", None),
        ("x20160222_081011.jpg", None),
    ],
)
def test_parse_source_id_real_style(default_config_path, name, expected):
    pattern = load_config(default_config_path).data.source_id_regex
    assert parse_source_id(name, pattern) == expected


def test_crops_of_one_photo_share_id(default_config_path):
    pattern = load_config(default_config_path).data.source_id_regex
    ids = {
        parse_source_id(f"20160222_081011_{r}_{c}.jpg", pattern) for r in (1, 361) for c in (1, 641)
    }
    assert ids == {"20160222_081011"}
