import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from crackqa.seed import seed_everything, torch_generator


def _draw(seed):
    seed_everything(seed)
    return random.random(), np.random.rand(3), torch.rand(3)


def test_same_seed_same_values():
    a, b = _draw(7), _draw(7)
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


def test_different_seed_different_values():
    assert not torch.equal(_draw(7)[2], _draw(8)[2])


def _order(seed):
    loader = DataLoader(
        list(range(50)), batch_size=50, shuffle=True, generator=torch_generator(seed)
    )
    return next(iter(loader)).tolist()


def test_shuffle_order_reproducible():
    torch.manual_seed(123)  # global state must not leak into the loader's order
    first = _order(3)
    torch.manual_seed(999)
    assert _order(3) == first
    assert _order(4) != first
