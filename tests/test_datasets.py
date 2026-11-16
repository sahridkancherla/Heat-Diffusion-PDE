from pathlib import Path

import numpy as np
import pytest
import torch

from heatpde.data import ProcessedRun, RunInfo, RunNormalisers, Scales, WindowDataset


def make_runs(lengths=(100, 50)):
    rng = np.random.default_rng(0)
    return [
        ProcessedRun(
            info=RunInfo(Path(f"{m}.csv"), 7.0, 10.0),
            period=10.0,
            t=0.1 * np.arange(m),
            T=30 + rng.standard_normal((m, 8)),
            voltage=rng.standard_normal(m),
            current=rng.standard_normal(m),
        )
        for m in lengths
    ]


def test_window_shapes_and_channels():
    runs = make_runs()
    norm = RunNormalisers.fit(runs)
    ds = WindowDataset(runs, norm, Scales(time=10.0), window=32, stride=16)
    assert len(ds) == 5 + 2  # (100-32)//16+1, (50-32)//16+1
    x, y = ds[1]
    assert x.shape == (ds.in_channels, 8, 32) and y.shape == (1, 8, 32)
    torch.testing.assert_close(x[0, 3], y[0, 0])  # thermistor_0 forcing, broadcast over z
    torch.testing.assert_close(x[1, :, 5], y[0, :, 0])  # initial profile, broadcast over t
    assert x[3, 0, 0] == 0 and x[3, 0, 1] == pytest.approx(0.01)  # dt / time scale
    expected = norm.T.encode(runs[0].T[16:48].T)
    np.testing.assert_allclose(y[0].numpy(), expected, rtol=1e-5)


def test_voltage_forcing_without_initial_condition():
    runs = make_runs()
    norm = RunNormalisers.fit(runs)
    ds = WindowDataset(
        runs, norm, Scales(time=1.0), window=50, forcing="voltage", initial_condition=False
    )
    x, _ = ds[0]
    assert x.shape[0] == ds.in_channels == 3
    np.testing.assert_allclose(x[0, 0].numpy(), norm.voltage.encode(runs[0].voltage[:50]), 1e-5)


def test_window_too_long():
    with pytest.raises(ValueError):
        WindowDataset(make_runs(), RunNormalisers.fit(make_runs()), Scales(1.0), window=101)


def test_alpha_channel():
    from dataclasses import replace

    runs = make_runs()
    runs = [
        replace(r, info=replace(r.info, alpha=a)) for r, a in zip(runs, [1e-5, 3e-5], strict=True)
    ]
    scales = Scales(time=60.0, length=0.041)
    ds = WindowDataset(runs, RunNormalisers.fit(runs), scales, window=50, alpha_channel=True)
    x, _ = ds[len(ds) - 1]
    assert ds.in_channels == WindowDataset.n_channels(True, True) == 5 == x.shape[0]
    assert torch.allclose(x[2], torch.full_like(x[2], scales.alpha(3e-5)))


def test_alpha_channel_requires_alpha():
    runs = make_runs()
    with pytest.raises(ValueError, match="alpha"):
        WindowDataset(runs, RunNormalisers.fit(runs), Scales(1.0), window=50, alpha_channel=True)


def test_period_time_scale():
    from dataclasses import replace

    runs = [
        replace(r, period=p, info=replace(r.info, alpha=3e-5))
        for r, p in zip(make_runs(), [5.0, 20.0], strict=True)
    ]
    ds = WindowDataset(
        runs,
        RunNormalisers.fit(runs),
        Scales(time=None, length=0.04),
        window=50,
        alpha_channel=True,
    )
    x0, _ = ds[0]
    x1, _ = ds[len(ds) - 1]
    assert x0[4, 0, 1] == pytest.approx(0.1 / 5.0) and x1[4, 0, 1] == pytest.approx(0.1 / 20.0)
    assert x0[2, 0, 0] == pytest.approx(3e-5 * 5.0 / 0.04**2)
    assert x1[2, 0, 0] == pytest.approx(3e-5 * 20.0 / 0.04**2)
