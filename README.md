# Heat Diffusion PDE

Fourier neural operators (FNO) and physics-informed neural operators (PINO) for the 1D heat
equation, and potentially advection-diffusion, trained on real thermistor data.

## Experiment

A heat pump at one end of a cylinder is driven with a sinusoidal voltage, which alternately heats
and cools it. Eight thermistors along the cylinder record the temperature wave as it travels and
decays. Runs vary the drive period (5–60 s) and the cylinder material (aluminium, brass, steel, and
a set of runs whose material isn't recorded in the filename).

```
Data/
├── 7V <period>s [Al] <run>[ light].csv   # aluminium ("Al") and unlabelled runs
├── Brass/7V <period>s 3.1.csv
├── Steel/7V <period>s 7.1 dark.csv
├── Noise.csv                              # sensor noise, no drive
└── warmup.csv                             # warm-up transient
```

Each CSV has three metadata lines (start time, polling interval, comments), then the columns
`timestamp/s, voltage/V, current/A, thermistor_0/C … thermistor_7/C`. The header says
"100 ms" polling, but samples are irregular and the median spacing is about 76 ms (~13 Hz).

```python
from heatpde.data import list_runs, load_run

infos = list_runs()  # parsed drive settings for every run
steel = [i for i in infos if i.material == "steel"]
run = load_run(steel[0].path)
run.t, run.temps, run.data["voltage"]  # (N,), (N, 8), (N,)
```

To print an inventory of all the files, run `uv run python scripts/summarize_data.py`.

## Setup

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                   # creates .venv, installs torch (CUDA 13.0) and dev tools
uv run pytest
```

With pip:

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt                    # includes -e . and the CUDA torch index
```

`requirements.txt` is generated from `uv.lock`. After you change dependencies, regenerate it with:

```bash
uv export --all-groups --no-hashes --emit-index-url -o requirements.txt
```

PyTorch comes from the `cu130` index on Windows and Linux; the NVIDIA driver must support
CUDA ≥ 13.0. If yours doesn't, change the index URL in `pyproject.toml` (for example to `cu128`).

## Layout

```
src/heatpde/
├── data/       # CSV loading (io.py), torch Datasets
├── models/     # FNO / PINO architectures
├── physics/    # PDE residuals for physics-informed losses
├── training/   # training loops, losses, metrics
└── utils.py    # device selection, seeding
scripts/        # command-line entry points
notebooks/      # exploration
tests/
outputs/        # checkpoints, logs, figures (git-ignored)
```
