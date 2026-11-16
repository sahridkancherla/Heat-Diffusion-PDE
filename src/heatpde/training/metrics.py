"""Evaluation metrics in physical units (°C, rad, m²/s)."""

from __future__ import annotations

import numpy as np

from heatpde.geometry import ROD_LENGTH, THERMISTOR_X
from heatpde.physics.angstrom import alpha_semi_infinite, fit_finite_rod, harmonic_fit


def rmse_per_sensor(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(M, 8) -> (8,)"""
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


def max_error_per_sensor(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(M, 8) -> (8,)"""
    return np.abs(pred - true).max(axis=0)


def rmse_vs_window_position(
    pred_windows: np.ndarray, true_windows: np.ndarray, skip_sensors: int = 0
) -> np.ndarray:
    """(N, 8, W) -> (W,). Rising error near either end of the window suggests FFT wrap-around."""
    err = (pred_windows - true_windows)[:, skip_sensors:]
    return np.sqrt(np.mean(err**2, axis=(0, 1)))


def drive_response(t: np.ndarray, T: np.ndarray, period: float, skip_frac: float = 0.3):
    """Complex amplitude per sensor at the drive frequency over whole steady cycles.

    Assumes t = 0 at a rising drive crossing (as from preprocess_run).
    """
    n = int(t[-1] // period)
    m = (t >= np.ceil(skip_frac * n) * period) & (t < n * period)
    return harmonic_fit(t[m], T[m], period)


def drive_frequency_errors(
    t: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    period: float,
    skip_frac: float = 0.3,
    min_snr: float = 50.0,
    skip_sensors: int = 0,
) -> dict:
    """Amplitude ratio and phase error per sensor, and Ångström α of measured vs predicted field.

    The first ``skip_sensors`` predicted rows (model inputs, e.g. the forcing thermistor) are
    replaced by the measured values so the α fit is not polluted by untrained outputs.
    """
    a_true, stderr = drive_response(t, true, period, skip_frac)
    a_pred, _ = drive_response(t, pred, period, skip_frac)
    a_pred[:skip_sensors] = a_true[:skip_sensors]
    used = np.cumprod(np.abs(a_true) / stderr >= min_snr).astype(bool)
    omega = 2 * np.pi / period

    def alpha(a: np.ndarray) -> float:
        if used.sum() < 3:
            return np.nan
        x, a = THERMISTOR_X[used], a[used]
        return fit_finite_rod(x, a, omega, ROD_LENGTH, alpha0=alpha_semi_infinite(x, a, omega))[0]

    return {
        "amp_ratio": np.abs(a_pred) / np.abs(a_true),
        "phase_err": np.angle(a_pred / a_true),  # rad, positive = prediction leads
        "used": used,
        "alpha_true": alpha(a_true),
        "alpha_pred": alpha(a_pred),
    }
