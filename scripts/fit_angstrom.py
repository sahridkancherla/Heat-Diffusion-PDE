"""Estimate thermal diffusivity for every run with Ångström's method.

Usage:
    uv run python scripts/fit_angstrom.py [--min-snr 50] [--skip-frac 0.3] [--csv out.csv]
"""

import argparse

import pandas as pd

from heatpde.data import list_runs, load_run
from heatpde.physics.angstrom import analyse_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-snr", type=float, default=50.0)
    parser.add_argument("--skip-frac", type=float, default=0.3)
    parser.add_argument("--csv", help="also write the table to this file")
    args = parser.parse_args()

    rows = []
    for info in list_runs():
        row = {
            "file": info.path.name,
            "material": info.material or "?",
            "run": info.run_id,
            "period_s": info.period,
        }
        try:
            r = analyse_run(load_run(info.path), skip_frac=args.skip_frac, min_snr=args.min_snr)
        except ValueError as e:
            rows.append(row | {"error": str(e)})
            continue
        rows.append(
            row
            | {
                "fit_period_s": round(r.period, 3),
                "cycles": round((r.window[1] - r.window[0]) / r.period),
                "n_used": int(r.used.sum()),
                "A0_C": round(float(r.amplitude[0]), 3),
                "alpha_semi_inf": r.alpha_semi_infinite,
                "alpha": r.alpha,
                "h_per_s": r.h,
                "fit_rms": round(r.fit_rms, 4),
            }
        )

    df = pd.DataFrame(rows)
    with pd.option_context(
        "display.max_rows", None, "display.width", 250, "display.float_format", "{:.3g}".format
    ):
        print(df.to_string(index=False))
        ok = df.dropna(subset=["alpha"])
        print("\nalpha by material (m^2/s):")
        print(ok.groupby("material")["alpha"].describe()[["count", "mean", "std", "min", "max"]])
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
