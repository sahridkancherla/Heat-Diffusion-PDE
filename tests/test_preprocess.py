from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heatpde.data import (
    Normaliser,
    Run,
    RunInfo,
    RunNormalisers,
    Scales,
    holdout_periods,
    preprocess_run,
    random_split,
)


def make_run(period=10.0, true_period=9.985, duration=200.0, seed=0) -> Run:
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.06, 0.09, int(duration / 0.075)))
    phase = 2 * np.pi * t / true_period
    df = pd.DataFrame({f"thermistor_{i}": 30 + np.sin(phase - i) for i in range(8)})
    df.insert(0, "current", 3 * np.sin(phase))
    df.insert(0, "voltage", 7 * np.sin(phase + 1.0))
    df.insert(0, "timestamp", t)
    info = RunInfo(path=Path("x.csv"), voltage=7.0, period=period)
    return Run(info=info, started=None, polling_ms=None, comments="", data=df)


def test_preprocess_uniform_whole_cycles():
    p = preprocess_run(make_run(), points_per_period=32, skip_cycles=2)
    assert p.period == pytest.approx(9.985, rel=1e-4)
    assert len(p.t) % 32 == 0
    np.testing.assert_allclose(np.diff(p.t), p.dt)
    assert p.T.shape == (len(p.t), 8)
    assert p.voltage[0] == pytest.approx(0.0, abs=0.05)
    assert p.voltage[8] == pytest.approx(7.0, rel=0.01)  # quarter period after rising crossing


def test_preprocess_fixed_dt():
    p = preprocess_run(make_run(), dt=0.1)
    assert p.dt == pytest.approx(0.1)
    assert p.t[-1] <= 200


def test_normalisers_roundtrip():
    runs = [preprocess_run(make_run(seed=s)) for s in range(2)]
    n = RunNormalisers.fit(runs)
    assert RunNormalisers.from_dict(n.to_dict()) == n
    x = n.T.encode(runs[0].T)
    assert abs(x.mean()) < 0.1
    np.testing.assert_allclose(n.T.decode(x), runs[0].T)
    assert Normaliser().encode(3.0) == 3.0


def test_scales():
    s = Scales(time=30.0, length=0.04)
    assert s.alpha_si(s.alpha(3e-5)) == pytest.approx(3e-5)
    assert s.alpha(3e-5) == pytest.approx(3e-5 * 30 / 0.04**2)


def test_splits_partition():
    infos = [RunInfo(Path(f"{p}.csv"), 7.0, p, m, "1") for p in (5, 10, 20) for m in ("a", "b")]
    s = holdout_periods(infos, test={5}, val={20})
    assert {i.period for i in s.test} == {5} and {i.period for i in s.val} == {20}
    r = random_split(infos, val_frac=1 / 6, test_frac=1 / 6)
    assert sorted(map(str, sum(r, []))) == sorted(map(str, infos))
    assert (len(r.train), len(r.val), len(r.test)) == (4, 1, 1)


def test_matches_filter():
    from heatpde.data.splits import matches, split_by

    infos = [RunInfo(Path(f"{p}{m}.csv"), 7.0, p, m) for p in (10, 30) for m in ("brass", "steel")]
    s = split_by(infos, matches({"material": ["brass"]}), matches({"period": [30]}))
    assert [(i.period, i.material) for i in s.test] == [(10, "brass"), (30, "brass")]
    assert [(i.period, i.material) for i in s.val] == [(30, "steel")]
    assert not any(matches({})(i) for i in infos)
