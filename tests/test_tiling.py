import numpy as np
import pytest

from crackqa.tiling import AMBIGUOUS, CRACK, NO_CRACK, cut, tile_grid, tile_label, tile_origins


def test_tile_count_and_shape_for_640x360():
    image = np.zeros((360, 640, 3), np.uint8)
    grid = tile_grid(360, 640, 128, 128, drop_edge=True)
    assert len(grid) == 10  # 5 columns x 2 rows, the 104 px bottom strip is dropped
    assert all(cut(image, y, x, 128).shape == (128, 128, 3) for y, x in grid)
    assert max(y for y, _ in grid) + 128 <= 360


def test_keeping_edges_adds_a_flush_row():
    grid = tile_grid(360, 640, 128, 128, drop_edge=False)
    assert len(grid) == 15  # 640 divides evenly; one extra row flush with the bottom
    assert sorted({y for y, _ in grid}) == [0, 128, 232]


def test_image_smaller_than_a_tile_gives_no_tiles():
    assert tile_origins(100, 128, 128, drop_edge=False) == []


@pytest.mark.parametrize("crack_px, label", [(0, NO_CRACK), (10, AMBIGUOUS), (200, CRACK)])
def test_label_rule(crack_px, label):
    assert tile_label(crack_px, crack_min_px=50) == label


def test_label_rule_boundary():
    assert tile_label(49, 50) == AMBIGUOUS
    assert tile_label(50, 50) == CRACK


def test_tile_coordinates_rebuild_the_covered_region():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    rebuilt = np.zeros_like(image)
    for y, x in tile_grid(360, 640, 128, 128, drop_edge=True):
        rebuilt[y : y + 128, x : x + 128] = cut(image, y, x, 128)
    np.testing.assert_array_equal(rebuilt[:256], image[:256])
    assert not rebuilt[256:].any()
