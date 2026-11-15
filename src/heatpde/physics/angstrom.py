"""Thermal diffusivity from a sinusoidally driven rod (Ångström's method).

Each thermistor's steady-periodic response is reduced to a complex amplitude Â(x) at the drive
frequency ω, with T(x, t) ≈ mean + Re[Â(x) e^{iωt}]. Two estimates are made from Â(x):

* ``alpha_semi_infinite``: classic Ångström. Assumes an infinitely long rod,
  Â ∝ e^{-(k_a + i k_p) x}, giving α = ω / (2 k_a k_p) regardless of lateral heat loss.
* ``alpha`` and ``h``: finite rod of length L with insulated far end and lateral loss
  ∂T/∂t = α ∂²T/∂x² - h (T - T_amb), so Â(x) = B cosh(q (L - x)), q² = (h + iω) / α.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from heatpde.data.io import Run
from heatpde.data.preprocess import drive_cycles
from heatpde.geometry import ROD_LENGTH, THERMISTOR_X


@dataclass
class AngstromResult:
    period: float  # s, estimated from the drive voltage
    window: tuple[float, float]  # s, time span used
    amplitude: np.ndarray  # °C, per thermistor
    phase: np.ndarray  # rad, per thermistor, unwrapped (decreasing = lagging)
    snr: np.ndarray  # amplitude / standard error
    used: np.ndarray  # bool, thermistors used in the fits
    alpha_semi_infinite: float  # m²/s
    alpha: float  # m²/s
    h: float  # 1/s
    fit_rms: float  # rms of complex-log residuals of the finite-rod fit


def harmonic_fit(
    t: np.ndarray, temps: np.ndarray, period: float, n_harmonics: int = 2, poly_degree: int = 2
):
    """Fit T = poly(t) + Σ_k [a_k cos kωt + b_k sin kωt] per column of ``temps``.

    Returns the complex fundamental amplitudes Â = a_1 - i b_1 and their standard errors.
    """
    omega = 2 * np.pi / period
    s = (t - t.mean()) / np.ptp(t)
    cols = [s**d for d in range(poly_degree + 1)]
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(k * omega * t), np.sin(k * omega * t)]
    X = np.stack(cols, axis=1)

    coef, *_ = np.linalg.lstsq(X, temps, rcond=None)
    sigma2 = ((temps - X @ coef) ** 2).sum(0) / (len(t) - X.shape[1])
    cov = np.diag(np.linalg.inv(X.T @ X))

    j = poly_degree + 1
    amp = coef[j] - 1j * coef[j + 1]
    stderr = np.sqrt(sigma2 * (cov[j] + cov[j + 1]) / 2)
    return amp, stderr


def _log_amp(z: np.ndarray) -> np.ndarray:
    return np.log(np.abs(z)) + 1j * np.unwrap(np.angle(z))


def alpha_semi_infinite(x: np.ndarray, amp: np.ndarray, omega: float) -> float:
    la = _log_amp(amp)
    k_a = -np.polyfit(x, la.real, 1)[0]
    k_p = -np.polyfit(x, la.imag, 1)[0]
    return omega / (2 * k_a * k_p)


def fit_finite_rod(
    x: np.ndarray, amp: np.ndarray, omega: float, length: float, alpha0: float, h0: float = 1e-3
):
    """Fit α, h of Â(x) = B cosh(q (L - x)). Residuals are in log space (log-amplitude, phase)."""
    la = _log_amp(amp)

    def residual(p):
        alpha, h = np.exp(p)
        q = np.sqrt((h + 1j * omega) / alpha)
        r = la - _log_amp(np.cosh(q * (length - x)))
        r -= r.mean()  # optimal log B
        return np.concatenate([r.real, r.imag])

    bounds = (np.log([1e-7, 1e-7]), np.log([1e-3, 1.0]))
    p0 = np.clip(np.log([alpha0, h0]), bounds[0] + 1e-6, bounds[1] - 1e-6)
    sol = least_squares(residual, p0, bounds=bounds)
    alpha, h = np.exp(sol.x)
    return alpha, h, float(np.sqrt(np.mean(sol.fun**2)))


def analyse_run(
    run: Run,
    x: np.ndarray = THERMISTOR_X,
    length: float = ROD_LENGTH,
    skip_frac: float = 0.3,
    min_snr: float = 50.0,
) -> AngstromResult:
    """Ångström analysis of one run.

    Drops the first ``skip_frac`` of the usable drive (transient), fits whole cycles of the rest,
    and uses thermistors from the heat-pump end outwards until one falls below ``min_snr``.
    """
    t = run.t
    tc, period = drive_cycles(t, run.data["voltage"].to_numpy(), run.info.period)
    tc = tc[int(skip_frac * len(tc)) :]
    if len(tc) < 3:
        raise ValueError("fewer than 2 steady drive cycles")
    m = (t >= tc[0]) & (t < tc[-1])

    amp, stderr = harmonic_fit(t[m], run.temps[m], period)
    snr = np.abs(amp) / stderr
    used = np.cumprod(snr >= min_snr).astype(bool)
    if used.sum() < 3:
        raise ValueError(f"only {used.sum()} thermistors above min_snr={min_snr}")

    omega = 2 * np.pi / period
    a_si = alpha_semi_infinite(x[used], amp[used], omega)
    alpha, h, rms = fit_finite_rod(x[used], amp[used], omega, length, alpha0=a_si)

    return AngstromResult(
        period=period,
        window=(tc[0], tc[-1]),
        amplitude=np.abs(amp),
        phase=np.unwrap(np.angle(amp)),
        snr=snr,
        used=used,
        alpha_semi_infinite=a_si,
        alpha=alpha,
        h=h,
        fit_rms=rms,
    )
