from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heatpde.data import RunInfo, parse_run_info
from heatpde.physics.materials import assign_materials, classify


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(1.06e-5, "steel"), (3.3e-5, "brass"), (7e-5, "aluminium"), (3e-4, None), (np.nan, None)],
)
def test_classify(alpha, expected):
    assert classify(alpha) == expected


def test_assign_materials_from_cache(tmp_path: Path):
    paths = [tmp_path / "7V 10s Al 4.1.csv", tmp_path / "7V 10s 2.1.csv", tmp_path / "7V 10s 9.csv"]
    for p in paths:
        p.write_text("x")
    cache = tmp_path / "alpha.csv"
    rows = zip(paths, [1.06e-5, 3.22e-5, 5e-4], strict=True)
    pd.DataFrame([{"path": p.resolve().as_posix(), "size": 1, "alpha": a} for p, a in rows]).to_csv(
        cache, index=False
    )

    steel, brass, unknown = assign_materials([parse_run_info(p) for p in paths], cache)
    assert (steel.material, steel.label) == ("steel", "aluminium")
    assert (brass.material, brass.label) == ("brass", None)
    assert unknown.material is None and unknown.alpha == 5e-4


def test_classify_skips_calibration(tmp_path: Path):
    p = tmp_path / "Noise.csv"
    p.write_text("x")
    (info,) = assign_materials([RunInfo(p)], cache=None)
    assert np.isnan(info.alpha) and info.material is None


def test_load_processed_keeps_info(monkeypatch):
    from dataclasses import replace

    import heatpde.data.preprocess as pre

    info = replace(parse_run_info("Steel/7V 10s 7.1 dark.csv"), material="x", alpha=1e-5)
    monkeypatch.setattr(pre, "load_run", lambda path: path)
    monkeypatch.setattr(pre, "preprocess_run", lambda run: pre.ProcessedRun(*(None,) * 6))
    (out,) = pre.load_processed([info])
    assert out.info is info
