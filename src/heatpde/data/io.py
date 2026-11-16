"""Load thermistor recordings from the heat-pump rig.

Each CSV starts with three metadata lines, then a header row::

    Date and Time: 2026-02-13 10:31:39
    Approximate Polling Time: 100 ms
    Comments: No comments provided
    timestamp/s,voltage/V,current/A,thermistor_0/C,...,thermistor_7/C

Drive settings are encoded in the filename (``<amplitude>V <period>s [material] <run id> [tag]``)::

    7V 10s 2.1.csv            -> 7 V, 10 s period, run 2.1, material not stated
    7V 10s Al 4.1.csv         -> material "aluminium"
    Brass/7V 06s 3.1.csv      -> material taken from the parent folder
    Steel/7V 10s 7.1 dark.csv -> tag "dark"

Files that don't follow this pattern (``Noise.csv``, ``warmup.csv``) are still loadable with
:func:`load_run`; their drive fields are ``None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "Data"

N_METADATA_LINES = 3

# Material names as they appear in filenames / folder names -> canonical name.
MATERIAL_ALIASES = {
    "al": "aluminium",
    "aluminium": "aluminium",
    "aluminum": "aluminium",
    "brass": "brass",
    "steel": "steel",
    "copper": "copper",
    "cu": "copper",
}

_NUM = r"\d+(?:\.\d+)?"
_FILENAME_RE = re.compile(
    rf"^(?P<voltage>{_NUM})V\s+(?P<period>{_NUM})s"
    r"(?:\s+(?P<material>[A-Za-z]+))?"
    rf"\s+(?P<run_id>{_NUM})"
    r"(?:\s+(?P<tag>.+))?$"
)


@dataclass(frozen=True)
class RunInfo:
    """Experiment settings parsed from a recording's path."""

    path: Path
    voltage: float | None = None  # drive amplitude [V]
    period: float | None = None  # drive period [s]
    material: str | None = (
        None  # from the filename, or from α after physics.materials.assign_materials
    )
    run_id: str | None = None  # e.g. "2.1"; kept as a string so "2.10" != "2.1"
    tag: str | None = None  # trailing note, e.g. "light" / "dark"
    label: str | None = None  # material as written in the filename (may be wrong)
    alpha: float | None = None  # measured diffusivity [m²/s], set by assign_materials

    @property
    def is_drive_run(self) -> bool:
        """True if this is a sinusoidal-drive run (not a calibration file like Noise/warmup)."""
        return self.period is not None


@dataclass
class Run:
    """A single recording: metadata plus the time series."""

    info: RunInfo
    started: datetime | None
    polling_ms: float | None
    comments: str
    data: pd.DataFrame  # columns: timestamp, voltage, current, thermistor_0..thermistor_7

    @property
    def t(self) -> np.ndarray:
        """Sample times [s], shape (n_samples,). Sampling is irregular (~10 Hz)."""
        return self.data["timestamp"].to_numpy()

    @property
    def temps(self) -> np.ndarray:
        """Thermistor temperatures [°C], shape (n_samples, n_thermistors)."""
        return self.data[self.thermistor_columns].to_numpy()

    @property
    def thermistor_columns(self) -> list[str]:
        return [c for c in self.data.columns if c.startswith("thermistor_")]


def _canonical_material(name: str | None) -> str | None:
    if name is None:
        return None
    return MATERIAL_ALIASES.get(name.lower(), name.lower())


def parse_run_info(path: str | Path) -> RunInfo:
    """Parse drive settings from a recording's filename (and parent folder for material)."""
    path = Path(path)
    m = _FILENAME_RE.match(path.stem.strip())
    if m is None:
        return RunInfo(path=path)

    material = m["material"]
    if material is None and path.parent.name.lower() in MATERIAL_ALIASES:
        material = path.parent.name

    tag = m["tag"]
    material = _canonical_material(material)
    return RunInfo(
        path=path,
        voltage=float(m["voltage"]),
        period=float(m["period"]),
        material=material,
        run_id=m["run_id"],
        tag=tag.strip().lower() if tag else None,
        label=material,
    )


def _read_metadata(path: Path) -> dict[str, str]:
    meta = {}
    with path.open(encoding="utf-8") as f:
        for _ in range(N_METADATA_LINES):
            key, _, value = f.readline().strip().rstrip(",").partition(":")
            meta[key.strip()] = value.strip()
    return meta


def load_run(path: str | Path) -> Run:
    """Load one CSV recording."""
    path = Path(path)
    meta = _read_metadata(path)

    started = meta.get("Date and Time")
    polling = re.match(_NUM, meta.get("Approximate Polling Time", ""))

    df = pd.read_csv(path, skiprows=N_METADATA_LINES)
    df.columns = [c.split("/")[0].strip() for c in df.columns]

    return Run(
        info=parse_run_info(path),
        started=datetime.fromisoformat(started) if started else None,
        polling_ms=float(polling[0]) if polling else None,
        comments=meta.get("Comments", ""),
        data=df,
    )


def list_runs(root: str | Path = DATA_DIR, *, include_calibration: bool = False) -> list[RunInfo]:
    """Find all recordings under ``root``, sorted by (material, period, run id, tag).

    Calibration files whose names don't encode drive settings (Noise, warmup) are skipped unless
    ``include_calibration`` is set.
    """
    infos = [parse_run_info(p) for p in Path(root).rglob("*.csv")]
    if not include_calibration:
        infos = [i for i in infos if i.is_drive_run]
    return sorted(
        infos,
        key=lambda i: (i.material or "", i.period or 0.0, i.run_id or "", i.tag or "", i.path),
    )
