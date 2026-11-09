import cv2
import numpy as np

from crackqa.baseline import crack_map
from crackqa.config import load_config
from crackqa.tiling import cut, tile_grid

from conftest import texture


def test_baseline_scores_a_drawn_crack_higher_than_plain_texture(default_config_path):
    cfg = load_config(default_config_path).baseline
    plain = texture(5)
    cracked = plain.copy()
    pts = np.array([[0, 60], [200, 120], [420, 80], [639, 150]], np.int32)
    cv2.polylines(cracked, [pts], False, (35, 35, 35), 20)  # CRACK500 cracks are wide

    truth = np.zeros(plain.shape[:2], np.uint8)
    cv2.polylines(truth, [pts], False, 1, 20)
    crack_tiles = [
        (y, x) for y, x in tile_grid(360, 640, 128, 128, True) if cut(truth, y, x, 128).any()
    ]

    on_crack = np.mean([cut(crack_map(cracked, cfg), y, x, 128).mean() for y, x in crack_tiles])
    on_plain = np.mean([cut(crack_map(plain, cfg), y, x, 128).mean() for y, x in crack_tiles])
    assert on_crack > 2 * on_plain


def test_crack_map_is_binary_and_same_size(default_config_path):
    m = crack_map(texture(3), load_config(default_config_path).baseline)
    assert m.shape == (360, 640) and set(np.unique(m)) <= {0, 1}
