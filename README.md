# Heat Diffusion PDE

A Fourier neural operator (FNO), written from scratch in PyTorch, that learns how heat waves
travel along metal rods from real temperature measurements.

The data was collected for a thermal waves lab practical; this project is a side experiment to
see how well an FNO can learn the physics directly from it.

## Experiment

A heat pump at one end of a 41 mm cylinder is driven with a sinusoidal voltage, alternately heating
and cooling it. Eight thermistors, 5 mm apart (at 3–38 mm), record the temperature wave as it
travels along the rod and decays. Runs vary the drive period (5–60 s) and the cylinder material.

```
Data/                                      # git-ignored
├── 7V <period>s [Al] <run>[ light].csv
├── Brass/7V <period>s 3.1.csv
├── Steel/7V <period>s 7.1 dark.csv
├── Noise.csv                              # sensor noise, no drive
└── warmup.csv                             # warm-up transient
```

Each CSV has three metadata lines, then `timestamp/s, voltage/V, current/A,
thermistor_0/C … thermistor_7/C`, sampled irregularly at about 13 Hz.

**Materials are identified from the data, not the filenames.** Many files are unlabelled, and the
"Al" runs are actually the steel cylinder. Each run's thermal diffusivity α is measured with
Ångström's method (fitting amplitude decay and phase lag along the rod with a finite-rod model). This
splits the 58 drive runs into three tight groups, reproducible to within 0.5%:

| material | α (m²/s) | runs |
|---|---|---|
| steel | 1.07 × 10⁻⁵ | 21 |
| brass | 3.24 × 10⁻⁵ | 24 |
| aluminium | 7.36 × 10⁻⁵ | 13 |

## Approach

- **Task.** Given the temperature at the heat-pump end (thermistor 0) over a window of 4 drive
  cycles, plus the starting temperature profile, predict the temperature at all other thermistors.
- **Preprocessing.** Runs are trimmed to whole drive cycles (the drive period is measured from the
  voltage signal) and resampled to 64 points per period.
- **Model.** A 2D FNO over the thermistor × time grid, with about 100k parameters. Padding in both
  directions stops the FFT treating the rod's ends, or the window's start and end, as neighbours.
- **Dimensionless inputs.** Time is scaled by each run's drive period, and the measured diffusivity
  is passed in as α̂ = α·P/L². That one number sets how quickly the wave decays along the rod, so
  different materials and periods share a single parameter range.
- **Per-window rescaling.** Each window is shifted and scaled by its starting temperature and its
  forcing amplitude. The heat equation is linear, so this changes nothing physically, but it stops
  long-period, large-amplitude runs from dominating the loss.
- **Splits are by whole run.** Validation is the 30 s runs; test is the 6, 55 and 60 s runs; all
  other periods (5–50 s) are used for training.

## Results

Best model: [configs/fno_all_wnorm.yaml](configs/fno_all_wnorm.yaml).

**Mean RMSE of 0.0058 °C** across the 11 held-out runs (6 validation and 5 test), averaged over
thermistors 1–7. This leaves out one run's start-up transient. For comparison, the temperature
waves swing by up to about ±3 °C.

## Usage

```bash
uv run python scripts/summarize_data.py                       # inventory of all recordings
uv run python scripts/fit_angstrom.py                         # Ångström diffusivity per run
uv run python scripts/train.py configs/fno_all_wnorm.yaml     # train
uv run python scripts/evaluate.py outputs/fno_all_wnorm/best.pt --plots   # metrics + plots
```

`evaluate.py` reports, per run, the RMSE and maximum error in °C, the amplitude and phase error at
the drive frequency, and the diffusivity recovered from the predicted field. Use `--split val`
when comparing configurations, and keep the test set for final results.

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

`requirements.txt` is generated from `uv.lock`. After you change dependencies, regenerate it with
`uv export --all-groups --no-hashes --emit-index-url -o requirements.txt`.

PyTorch comes from the `cu130` index on Windows and Linux, so the NVIDIA driver must support CUDA
13.0 or later. If yours doesn't, change the index URL in `pyproject.toml` (for example to `cu128`).

## Layout

```
src/heatpde/
├── data/        # CSV loading, preprocessing, normalisation, splits, windowed Dataset
├── models/      # FNO2d (fno.py), CNN baseline (baselines.py)
├── physics/     # Ångström diffusivity fit, material assignment from α
├── training/    # training loop, loss, metrics, whole-run prediction
├── config.py    # builds splits, datasets and models from a YAML config
├── geometry.py  # rod length and thermistor positions
└── utils.py     # device selection, seeding
configs/         # one YAML per experiment
scripts/         # train, evaluate, data inventory, Ångström fit
tests/
outputs/         # checkpoints, histories, plots (git-ignored)
```
