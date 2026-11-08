"""Seeding for reproducible runs."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and every CUDA device)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_generator(seed: int) -> torch.Generator:
    """A generator to pass to DataLoader so the shuffle order does not depend on global state."""
    return torch.Generator().manual_seed(seed)
