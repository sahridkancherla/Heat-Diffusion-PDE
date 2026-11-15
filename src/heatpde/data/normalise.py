"""Value normalisation and nondimensional scales.

Temperature uses a single scalar mean/std (not per thermistor), so the heat equation keeps its form
in normalised units.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from heatpde.data.preprocess import ProcessedRun
from heatpde.geometry import ROD_LENGTH


@dataclass(frozen=True)
class Normaliser:
    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray]) -> Normaliser:
        x = np.concatenate([np.ravel(a) for a in arrays])
        return cls(float(x.mean()), float(x.std()))

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


@dataclass(frozen=True)
class RunNormalisers:
    T: Normaliser
    voltage: Normaliser
    current: Normaliser

    @classmethod
    def fit(cls, runs: Sequence[ProcessedRun]) -> RunNormalisers:
        """Fit on training runs only."""
        return cls(
            T=Normaliser.fit(r.T for r in runs),
            voltage=Normaliser.fit(r.voltage for r in runs),
            current=Normaliser.fit(r.current for r in runs),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunNormalisers:
        return cls(**{k: Normaliser(**v) for k, v in d.items()})


@dataclass(frozen=True)
class Scales:
    """x̂ = x / length, t̂ = t / time. Dimensionless α̂ = α · time / length², ĥ = h · time."""

    time: float  # s, e.g. a fixed reference so α̂ is comparable across drive periods
    length: float = ROD_LENGTH  # m

    def x(self, x):
        return x / self.length

    def t(self, t):
        return t / self.time

    def alpha(self, alpha):
        return alpha * self.time / self.length**2

    def alpha_si(self, alpha_hat):
        return alpha_hat * self.length**2 / self.time

    def h(self, h):
        return h * self.time

    def h_si(self, h_hat):
        return h_hat / self.time
