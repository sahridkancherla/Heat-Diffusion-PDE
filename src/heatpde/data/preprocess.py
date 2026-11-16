"""Trim runs to whole drive cycles and resample onto a uniform time grid."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

from heatpde.data.io import Run, RunInfo, load_run


@dataclass
class ProcessedRun:
    info: RunInfo
    period: float  # s, fitted from the drive voltage
    t: np.ndarray  # (M,) s, uniform; t = 0 at a rising zero crossing of the drive
    T: np.ndarray  # (M, 8) °C
    voltage: np.ndarray  # (M,)
    current: np.ndarray  # (M,)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def drive_cycles(t: np.ndarray, v: np.ndarray, nominal_period: float, tol: float = 0.02):
    """Rising zero crossings of the drive over the last stretch matching the nominal period.

    Returns (crossing times, fitted period).
    """
    i = np.flatnonzero((v[:-1] < 0) & (v[1:] >= 0))
    tc = t[i] - v[i] * (t[i + 1] - t[i]) / (v[i + 1] - v[i])

    keep = [tc[0]]
    for x in tc[1:]:
        if x - keep[-1] > 0.5 * nominal_period:
            keep.append(x)
    tc = np.array(keep)

    good = np.abs(np.diff(tc) / nominal_period - 1) < tol
    if not good[-1]:
        raise ValueError("drive at end of run does not match the nominal period")
    start = 0 if good.all() else len(good) - np.argmin(good[::-1])
    tc = tc[start:]
    period = np.polyfit(np.arange(len(tc)), tc, 1)[0]
    return tc, period


def preprocess_run(
    run: Run, dt: float = 0.1, points_per_period: int | None = None, skip_cycles: int = 0
) -> ProcessedRun:
    """Resample ``run`` between its first and last usable drive crossings.

    ``points_per_period`` overrides ``dt`` so every run has the same resolution in t / period.
    """
    voltage = run.data["voltage"].to_numpy()
    crossings, period = drive_cycles(run.t, voltage, run.info.period)
    crossings = crossings[skip_cycles:]
    if len(crossings) < 2:
        raise ValueError(f"no whole drive cycles left in {run.info.path.name}")

    if points_per_period:
        dt = period / points_per_period
        n = (len(crossings) - 1) * points_per_period
    else:
        n = int((crossings[-1] - crossings[0]) / dt)
    t = crossings[0] + dt * np.arange(n)

    def resample(y: np.ndarray) -> np.ndarray:
        return np.interp(t, run.t, y)

    return ProcessedRun(
        info=run.info,
        period=period,
        t=t - t[0],
        T=np.stack([resample(col) for col in run.temps.T], axis=1),
        voltage=resample(voltage),
        current=resample(run.data["current"].to_numpy()),
    )


def load_processed(infos: Iterable[RunInfo], **kwargs) -> list[ProcessedRun]:
    """Keeps the given RunInfo (e.g. material/alpha from assign_materials)."""
    return [replace(preprocess_run(load_run(i.path), **kwargs), info=i) for i in infos]
