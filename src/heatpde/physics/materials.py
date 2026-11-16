"""Assign each run's material from its measured diffusivity rather than its filename.

Filename labels are unreliable (the "Al 4.1" runs are the steel cylinder, and many runs are
unlabelled), but the Ångström α of each cylinder is reproducible to <0.5% across runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from heatpde.data.io import DATA_DIR, RunInfo, load_run
from heatpde.physics.angstrom import analyse_run

# Mean finite-rod α of each cylinder's runs [m²/s].
REFERENCE_ALPHA = {"steel": 1.07e-5, "brass": 3.24e-5, "aluminium": 7.36e-5}
TOLERANCE = 0.25  # max |ln(α / reference)| to accept a match
CACHE = DATA_DIR.parent / "outputs" / "cache" / "run_alpha.csv"


def classify(alpha: float | None) -> str | None:
    if alpha is None or not np.isfinite(alpha):
        return None
    name, ref = min(REFERENCE_ALPHA.items(), key=lambda kv: abs(np.log(alpha / kv[1])))
    return name if abs(np.log(alpha / ref)) <= TOLERANCE else None


def _measure(info: RunInfo) -> float:
    if not info.is_drive_run:
        return np.nan
    try:
        return analyse_run(load_run(info.path)).alpha
    except ValueError:
        return np.nan


def measure_alphas(infos: Iterable[RunInfo], cache: Path | None = CACHE) -> list[float]:
    """Ångström α per run, cached by (path, file size)."""
    infos = list(infos)
    keys = [(i.path.resolve().as_posix(), i.path.stat().st_size) for i in infos]

    known = {}
    if cache is not None and cache.exists():
        df = pd.read_csv(cache)
        known = {(p, s): a for p, s, a in zip(df.path, df["size"], df.alpha, strict=True)}

    alphas = [known[k] if k in known else _measure(i) for k, i in zip(keys, infos, strict=True)]

    if cache is not None and any(k not in known for k in keys):
        known.update(zip(keys, alphas, strict=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"path": p, "size": s, "alpha": a} for (p, s), a in known.items()]
        pd.DataFrame(rows).to_csv(cache, index=False)
    return alphas


def assign_materials(infos: Iterable[RunInfo], cache: Path | None = CACHE) -> list[RunInfo]:
    """Set ``alpha`` and ``material`` (from α; falls back to the filename label if unmatched)."""
    infos = list(infos)
    return [
        replace(i, alpha=float(a), material=classify(a) or i.label)
        for i, a in zip(infos, measure_alphas(infos, cache), strict=True)
    ]
