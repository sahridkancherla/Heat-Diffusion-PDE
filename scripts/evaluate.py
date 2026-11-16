"""Evaluate a checkpoint on the runs of one split.

The checkpoint's ``extra`` must contain ``config`` (the YAML dict) and ``norm``
(RunNormalisers.to_dict()).

Usage:
    uv run python scripts/evaluate.py outputs/fno_brass/best.pt [--split test] [--plots]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from heatpde.config import build_model, build_scales, build_split
from heatpde.data import RunNormalisers, load_processed
from heatpde.training.metrics import (
    drive_frequency_errors,
    max_error_per_sensor,
    rmse_per_sensor,
    rmse_vs_window_position,
)
from heatpde.training.predict import predict_run
from heatpde.training.trainer import load_checkpoint
from heatpde.utils import get_device

SURFACE, INK, GRID, PRED = "#fcfcfb", "#52514e", "#e4e3df", "#2a78d6"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 9,
    }
)


def plot_run(path, run, pred, n_cycles):
    m = run.t >= run.t[-1] - n_cycles * run.period
    fig, axes = plt.subplots(8, 1, figsize=(8, 11), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(run.t[m], run.T[m, k], color=INK, lw=1.5, label="measured")
        ax.plot(run.t[m], pred[m, k], color=PRED, lw=1.5, ls="--", label="predicted")
        ax.set_ylabel(f"T{k} (°C)")
    axes[0].legend(loc="upper right", ncols=2, frameon=False)
    axes[0].set_title(run.info.path.name, loc="left")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_window_error(path, rmse, subtitle):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(len(rmse)), rmse, color=PRED, lw=1.5)
    ax.set(xlabel="sample in window", ylabel="RMSE (°C)", ylim=(0, None))
    ax.set_title(f"Error vs position in window ({subtitle})", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--stride", type=int, help="window stride (default: window length)")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--plot-cycles", type=int, default=5, help="drive cycles shown per run")
    parser.add_argument("--device")
    args = parser.parse_args()

    plt.switch_backend("Agg")
    device = get_device(args.device)
    extra = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["extra"]
    cfg, norm = extra["config"], RunNormalisers.from_dict(extra["norm"])
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    infos = getattr(build_split(cfg), args.split)
    runs = load_processed(infos, **cfg["data"]["preprocess"])
    ds_kwargs = cfg["data"]["dataset"] | {"stride": args.stride}
    skip = cfg.get("loss", {}).get("skip_sensors", 0)

    out_dir = args.checkpoint.parent / f"eval_{args.split}"
    out_dir.mkdir(exist_ok=True)

    rows, sensor_rows, window_preds, window_trues = [], [], [], []
    for run in runs:
        if len(run.t) < ds_kwargs["window"]:
            print(f"skipping {run.info.path.name}: {len(run.t)} samples < window")
            continue
        pred, pw, tw = predict_run(model, run, norm, build_scales(cfg), device=device, **ds_kwargs)
        window_preds.append(pw)
        window_trues.append(tw)

        rmse = rmse_per_sensor(pred, run.T)
        freq = drive_frequency_errors(run.t, pred, run.T, run.period, skip_sensors=skip)
        s = slice(skip, None)
        ok = freq["used"] & (np.arange(8) >= skip)  # sensors where the drive signal is measurable
        rows.append(
            {
                "file": run.info.path.name,
                "material": run.info.material or "?",
                "period_s": run.info.period,
                "rmse_C": float(np.sqrt(np.mean(rmse[s] ** 2))),
                "max_err_C": float(max_error_per_sensor(pred, run.T)[s].max()),
                "amp_err_pct": float(100 * np.mean(np.abs(freq["amp_ratio"][ok] - 1))),
                "phase_err_rad": float(np.mean(np.abs(freq["phase_err"][ok]))),
                "n_sensors_ok": int(ok.sum()),
                "alpha_true": freq["alpha_true"],
                "alpha_pred": freq["alpha_pred"],
            }
        )
        sensor_rows.append(rmse)
        if args.plots:
            plot_run(out_dir / f"{run.info.path.stem}.png", run, pred, args.plot_cycles)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    window_rmse = rmse_vs_window_position(
        np.concatenate(window_preds), np.concatenate(window_trues), skip
    )

    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(df.to_string(index=False))
        print("\nRMSE per sensor (degC), mean over runs:")
        print(
            pd.Series(np.mean(sensor_rows, axis=0), index=[f"T{k}" for k in range(8)]).to_string()
        )
    w = len(window_rmse)
    print(
        f"\nWindow RMSE: first 10% {window_rmse[: w // 10].mean():.4f}, "
        f"middle {window_rmse[w // 3 : 2 * w // 3].mean():.4f}, "
        f"last 10% {window_rmse[-w // 10 :].mean():.4f} degC"
    )
    if args.plots:
        plot_window_error(out_dir / "window_error.png", window_rmse, f"{len(df)} runs")
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
