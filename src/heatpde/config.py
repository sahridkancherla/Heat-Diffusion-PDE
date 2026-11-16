"""YAML experiment configs and the builders shared by scripts/train.py and scripts/evaluate.py."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch.nn as nn
import yaml

from heatpde.data import (
    RunInfo,
    RunNormalisers,
    Scales,
    Split,
    WindowDataset,
    holdout_materials,
    holdout_periods,
    list_runs,
    random_split,
    split_by,
)
from heatpde.data.preprocess import ProcessedRun
from heatpde.data.splits import matches
from heatpde.models.baselines import CNN2d
from heatpde.models.fno import FNO2d
from heatpde.physics.materials import assign_materials
from heatpde.training.trainer import TrainConfig

MODELS: dict[str, type[nn.Module]] = {"fno2d": FNO2d, "cnn2d": CNN2d}


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_runs(cfg: dict) -> list[RunInfo]:
    """Drive runs with material assigned from measured α, filtered by data.materials."""
    materials = cfg["data"].get("materials")
    infos = assign_materials(list_runs())
    return infos if materials is None else [i for i in infos if i.material in materials]


def build_split(cfg: dict) -> Split:
    s, infos = cfg["data"]["split"], select_runs(cfg)
    match s["type"]:
        case "holdout_periods":
            return holdout_periods(infos, s["test"], s.get("val", ()))
        case "holdout_materials":
            return holdout_materials(infos, s["test"], s.get("val", ()))
        case "filter":
            return split_by(infos, matches(s["test"]), matches(s.get("val", {})))
        case "random":
            return random_split(
                infos, s.get("val_frac", 0.15), s.get("test_frac", 0.15), cfg.get("seed", 0)
            )
    raise ValueError(f"unknown split type {s['type']!r}")


def build_scales(cfg: dict) -> Scales:
    """data.time_scale: seconds, or "period" to scale each run by its own drive period."""
    t = cfg["data"]["time_scale"]
    return Scales(time=None if t == "period" else float(t))


def build_dataset(
    cfg: dict, runs: list[ProcessedRun], norm: RunNormalisers, **overrides
) -> WindowDataset:
    return WindowDataset(runs, norm, build_scales(cfg), **(cfg["data"]["dataset"] | overrides))


def build_model(cfg: dict) -> nn.Module:
    ds = cfg["data"]["dataset"]
    in_c = WindowDataset.n_channels(
        ds.get("initial_condition", True), ds.get("alpha_channel", False)
    )
    return MODELS[cfg["model"]["name"]](in_c=in_c, **cfg["model"]["kwargs"])


def train_config(cfg: dict) -> TrainConfig:
    names = {f.name for f in fields(TrainConfig)}
    tc = TrainConfig(**{k: v for k, v in cfg["train"].items() if k in names})
    tc.out_dir = Path(tc.out_dir)
    return tc
