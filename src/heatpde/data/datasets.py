"""Sliding-window datasets on the (thermistor z, time) grid for FNO2d."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
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

    x: (C, 8, window) channels = [forcing, (initial profile), (α̂), z, t], broadcast over the grid.
    y: (1, 8, window) normalised temperatures.
    t is time since the window start divided by ``scales.time`` (or the run's period if None).
    α̂ = scales.alpha(run.info.alpha) needs runs whose RunInfo went through assign_materials.
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
        alpha_channel: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.window = window
        self.initial_condition = initial_condition
        self.alpha_channel = alpha_channel
        self.infos: list[RunInfo] = [r.info for r in runs]
        self.z = torch.as_tensor(scales.x(THERMISTOR_X), dtype=dtype)

        def tensor(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(a, dtype=dtype)

        self.fields, self.forcings, self.times, self.alphas = [], [], [], []
        self.index: list[tuple[int, int]] = []  # (run, start sample)
        for k, r in enumerate(runs):
            T = tensor(norm.T.encode(r.T).T)  # (8, M)
            self.fields.append(T)
            self.forcings.append(
                T[0]
                if forcing == "thermistor_0"
                else tensor(getattr(norm, forcing).encode(getattr(r, forcing)))
            )
            s = self.run_scales(scales, r)
            self.times.append(tensor(s.t(r.t)))
            if alpha_channel:
                if r.info.alpha is None or not np.isfinite(r.info.alpha):
                    raise ValueError(f"no measured alpha for {r.info.path.name}")
                self.alphas.append(float(s.alpha(r.info.alpha)))
            self.index += [(k, s) for s in range(0, len(r.t) - window + 1, stride or window)]

        if not self.index:
            raise ValueError(f"no run is at least {window} samples long")

    @staticmethod
    def run_scales(scales: Scales, run: ProcessedRun) -> Scales:
        return scales if scales.time is not None else replace(scales, time=run.period)

    @staticmethod
    def n_channels(initial_condition: bool = True, alpha_channel: bool = False) -> int:
        return 3 + initial_condition + alpha_channel

    @property
    def in_channels(self) -> int:
        return self.n_channels(self.initial_condition, self.alpha_channel)

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
        if self.alpha_channel:
            channels.append(torch.full(shape, self.alphas[k], dtype=T.dtype))
        channels += [self.z[:, None].expand(shape), (t - t[0]).expand(shape)]
        return torch.stack(channels), T[None]
