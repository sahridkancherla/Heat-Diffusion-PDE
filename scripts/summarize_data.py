"""Print an inventory of the recordings under Data/.

Usage:
    uv run python scripts/summarize_data.py [--root PATH]
"""

import argparse

import numpy as np
import pandas as pd

from heatpde.data import DATA_DIR, list_runs, load_run
from heatpde.physics.materials import assign_materials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DATA_DIR, help="data directory (default: %(default)s)")
    args = parser.parse_args()

    rows = []
    for info in assign_materials(list_runs(args.root, include_calibration=True)):
        run = load_run(info.path)
        dt = np.diff(run.t)
        rows.append(
            {
                "file": info.path.relative_to(args.root).as_posix(),
                "material": info.material or "?",
                "label": info.label or "",
                "alpha": info.alpha,
                "period_s": info.period,
                "run": info.run_id,
                "tag": info.tag or "",
                "samples": len(run.data),
                "duration_s": round(run.t[-1] - run.t[0], 1),
                "median_dt_ms": round(1000 * float(np.median(dt)), 1),
                "T_min": round(float(run.temps.min()), 2),
                "T_max": round(float(run.temps.max()), 2),
            }
        )

    df = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False))
    print(f"\n{len(df)} files")
    print(df.groupby("material")["period_s"].count().rename("runs").to_string())


if __name__ == "__main__":
    main()
