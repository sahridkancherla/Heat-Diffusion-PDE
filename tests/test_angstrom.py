from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heatpde.data.io import Run, RunInfo
from heatpde.geometry import ROD_LENGTH, THERMISTOR_X
from heatpde.physics.angstrom import analyse_run


def synthetic_run(alpha, h, period, duration=600.0, noise=1e-3, seed=0) -> Run:
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.06, 0.09, int(duration / 0.075)))
    omega = 2 * np.pi / period
    q = np.sqrt((h + 1j * omega) / alpha)
    profile = np.cosh(q * (ROD_LENGTH - THERMISTOR_X)) / np.cosh(q * ROD_LENGTH)
    temps = 30 + 0.01 * t[:, None] + np.real(profile * np.exp(1j * omega * t[:, None]))
    temps += noise * rng.standard_normal(temps.shape)

    df = pd.DataFrame(temps, columns=[f"thermistor_{i}" for i in range(len(THERMISTOR_X))])
    df.insert(0, "voltage", 7 * np.sin(omega * t))
    df.insert(0, "timestamp", t)
    info = RunInfo(path=Path("synthetic.csv"), voltage=7.0, period=period)
    return Run(info=info, started=None, polling_ms=None, comments="", data=df)


@pytest.mark.parametrize(("alpha", "period"), [(1e-5, 20.0), (3.3e-5, 30.0), (7e-5, 50.0)])
def test_recovers_alpha(alpha, period):
    r = analyse_run(synthetic_run(alpha, h=5e-4, period=period))
    assert r.used.all()
    assert r.period == pytest.approx(period, rel=1e-4)
    assert r.alpha == pytest.approx(alpha, rel=0.01)
