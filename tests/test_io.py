from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from heatpde.data import DATA_DIR, list_runs, load_run, parse_run_info


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("7V 10s 2.1.csv", (7.0, 10.0, None, "2.1", None)),
        ("7V 20s 1.csv", (7.0, 20.0, None, "1", None)),
        ("7V 10s Al 4.1.csv", (7.0, 10.0, "aluminium", "4.1", None)),
        ("7V 15s 5.1 Light.csv", (7.0, 15.0, None, "5.1", "light")),
        ("Brass/7V 06s 3.1.csv", (7.0, 6.0, "brass", "3.1", None)),
        ("Steel/7V 10s 7.1 dark.csv", (7.0, 10.0, "steel", "7.1", "dark")),
    ],
)
def test_parse_run_info(path, expected):
    info = parse_run_info(path)
    assert (info.voltage, info.period, info.material, info.run_id, info.tag) == expected
    assert info.is_drive_run


@pytest.mark.parametrize("name", ["Noise.csv", "warmup.csv"])
def test_parse_calibration_files(name):
    info = parse_run_info(name)
    assert not info.is_drive_run
    assert info.material is None


def test_load_run_synthetic(tmp_path: Path):
    path = tmp_path / "Steel" / "7V 10s 7.1 dark.csv"
    path.parent.mkdir()
    path.write_text(
        "Date and Time: 2026-02-13 10:31:39\n"
        "Approximate Polling Time: 100 ms,,,,,,,,,,\n"  # trailing commas occur in real files
        "Comments: No comments provided\n"
        "timestamp/s,voltage/V,current/A,thermistor_0/C,thermistor_1/C\n"
        "0.0,1.0,0.5,30.0,29.0\n"
        "0.1,2.0,0.6,30.5,29.5\n"
    )
    run = load_run(path)

    assert run.started == datetime(2026, 2, 13, 10, 31, 39)
    assert run.polling_ms == 100.0
    assert run.comments == "No comments provided"
    assert run.info.material == "steel"
    assert list(run.data.columns) == [
        "timestamp",
        "voltage",
        "current",
        "thermistor_0",
        "thermistor_1",
    ]
    np.testing.assert_allclose(run.t, [0.0, 0.1])
    assert run.temps.shape == (2, 2)


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="raw data not present")
def test_all_real_files_load():
    infos = list_runs(include_calibration=True)
    assert infos, f"no CSVs found under {DATA_DIR}"
    for info in infos:
        run = load_run(info.path)
        assert run.temps.shape[1] == 8, info.path
        assert not run.data.isna().any().any(), info.path
        assert np.all(np.diff(run.t) > 0), f"timestamps not increasing in {info.path}"
