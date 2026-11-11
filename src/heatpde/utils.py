"""Small shared helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return ``prefer`` if given, else the best available device (cuda > mps > cpu)."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (all devices) RNGs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG that third-party code may use
    torch.manual_seed(seed)
