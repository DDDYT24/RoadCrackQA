import pytest

from crackqa.config import load_config
from crackqa.data_qa import parse_source_id

# Names as they appear in the CRACK500 archive (confirmed on the real data)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("20160222_081011.jpg", "20160222_081011"),  # full photo
        ("20160222_081011_1281_721.jpg", "20160222_081011"),  # crop of that photo
        ("20160222_081011_1281_721.png", "20160222_081011"),  # its mask
        ("20160307_163627_mask.png", "20160307_163627"),  # full photo mask
        ("IMG_2941.jpg", "IMG_2941"),
        ("IMG_2941_1_969.jpg", "IMG_2941"),
        ("DSC_0001.jpg", None),
        ("x20160222_081011.jpg", None),
    ],
)
def test_parse_source_id_real_style(default_config_path, name, expected):
    pattern = load_config(default_config_path).data.source_id_regex
    assert parse_source_id(name, pattern) == expected


def test_crops_of_one_photo_share_id(default_config_path):
    pattern = load_config(default_config_path).data.source_id_regex
    names = [f"20160222_081011_{r}_{c}.jpg" for r in (1, 361, 721) for c in (1, 641, 1921)]
    assert {parse_source_id(n, pattern) for n in names} == {"20160222_081011"}
