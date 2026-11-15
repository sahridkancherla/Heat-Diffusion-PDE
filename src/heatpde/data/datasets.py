"""Sliding-window datasets on the (thermistor z, time) grid for FNO2d."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from heatpde.data.io import RunInfo
from heatpde.data.normalise import RunNormalisers, Scales
from heatpde.data.preprocess import ProcessedRun
from heatpde.geometry import THERMISTOR_X

Forcing = Literal["thermistor_0", "voltage", "current"]


class WindowDataset(Dataset):
    """Windows of ``window`` samples from each run.

    x: (C, 8, window) channels = [forcing, (initial profile), z, t], each broadcast over the grid.
    y: (1, 8, window) normalised temperatures.
    t is time since the window start divided by ``scales.time``, so runs with different dt agree.
    """

    def __init__(
        self,
        runs: Sequence[ProcessedRun],
        norm: RunNormalisers,
        scales: Scales,
        window: int,
        stride: int | None = None,
        forcing: Forcing = "thermistor_0",
        initial_condition: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.window = window
        self.initial_condition = initial_condition
        self.infos: list[RunInfo] = [r.info for r in runs]
        self.z = torch.as_tensor(scales.x(THERMISTOR_X), dtype=dtype)

        def tensor(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(a, dtype=dtype)

        self.fields, self.forcings, self.times = [], [], []
        self.index: list[tuple[int, int]] = []  # (run, start sample)
        for k, r in enumerate(runs):
            T = tensor(norm.T.encode(r.T).T)  # (8, M)
            self.fields.append(T)
            self.forcings.append(
                T[0]
                if forcing == "thermistor_0"
                else tensor(getattr(norm, forcing).encode(getattr(r, forcing)))
            )
            self.times.append(tensor(scales.t(r.t)))
            self.index += [(k, s) for s in range(0, len(r.t) - window + 1, stride or window)]

        if not self.index:
            raise ValueError(f"no run is at least {window} samples long")

    @property
    def in_channels(self) -> int:
        return 4 if self.initial_condition else 3

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, s = self.index[i]
        sl = slice(s, s + self.window)
        T, f, t = self.fields[k][:, sl], self.forcings[k][sl], self.times[k][sl]
        shape = T.shape

        channels = [f.expand(shape)]
        if self.initial_condition:
            channels.append(T[:, :1].expand(shape))
        channels += [self.z[:, None].expand(shape), (t - t[0]).expand(shape)]
        return torch.stack(channels), T[None]
