"""Train a model from a YAML config.

Usage:
    uv run python scripts/train.py configs/fno_brass.yaml [--epochs N] [--out-dir DIR] [--device D]
"""

import argparse
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from heatpde.config import build_dataset, build_model, build_split, load_config, train_config
from heatpde.data import RunNormalisers, load_processed
from heatpde.training.losses import field_mse
from heatpde.training.trainer import evaluate, fit
from heatpde.utils import seed_everything


class Persistence(torch.nn.Module):
    """Predicts the window's initial profile at every time step (input channel 1)."""

    def forward(self, x):
        return x[:, 1:2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument("--device")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key in ("epochs", "out_dir", "device"):
        if getattr(args, key) is not None:
            cfg["train"][key] = getattr(args, key)

    torch.backends.cuda.matmul.allow_tf32 = True
    seed_everything(cfg["seed"])

    split = build_split(cfg)
    for name, infos in split._asdict().items():
        print(f"{name:5s} ({len(infos)}): {', '.join(i.path.name for i in infos)}")
    if not split.train or not split.val:
        raise ValueError("train and val splits must both be non-empty; check data.split")

    pre = cfg["data"]["preprocess"]
    train_runs = load_processed(split.train, **pre)
    val_runs = load_processed(split.val, **pre)
    norm = RunNormalisers.fit(train_runs)

    window = cfg["data"]["dataset"]["window"]
    train_ds = build_dataset(cfg, train_runs, norm)
    val_ds = build_dataset(cfg, val_runs, norm, stride=window)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    val_loader = DataLoader(val_ds, batch_size=64)
    print(f"windows: train {len(train_ds)}, val {len(val_ds)}")

    model = build_model(cfg)
    loss_fn = partial(field_mse, skip_sensors=cfg["loss"]["skip_sensors"])
    print(f"model: {cfg['model']['name']}, {sum(p.numel() for p in model.parameters()):,} params")
    if train_ds.initial_condition:
        print(f"persistence val loss: {evaluate(Persistence(), val_loader, loss_fn, 'cpu'):.4e}")

    tc = train_config(cfg)
    tc.out_dir.mkdir(parents=True, exist_ok=True)
    with open(tc.out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    history = fit(
        model, train_loader, val_loader, loss_fn, tc, {"config": cfg, "norm": norm.to_dict()}
    )

    best = min(history, key=lambda h: h["val_loss"])
    print(f"\nbest val loss {best['val_loss']:.4e} at epoch {best['epoch']}")
    print(f"next: uv run python scripts/evaluate.py {(tc.out_dir / 'best.pt').as_posix()} --plots")


if __name__ == "__main__":
    main()
