"Training loop, with AdamW and cosine schedule."

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from ..utils import get_device


@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    patience: int | None = None
    min_delta: float = 0.0  # absolute, in loss units
    out_dir: Path = Path("outputs/run")
    device: str | None = None


class EarlyStopper:
    def __init__(self, cfg: TrainConfig):
        self.patience = cfg.patience
        self.min_delta = cfg.min_delta
        self.counter = 0
        self.min_val_loss = float("inf")

    def early_stop(self, val_loss):
        if self.patience is None:
            return False
        if val_loss < self.min_val_loss - self.min_delta:
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, loss_fn, optimiser, device, grad_clip):
    model.train()
    total = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimiser.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimiser.step()

        total += loss.item() * len(inputs)

    return total / len(loader.dataset)


@torch.no_grad
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        total += loss.item() * len(inputs)

    return total / len(loader.dataset)


def save_checkpoint(path, model, optimiser, scheduler, epoch, val_loss, extra: dict):
    """``extra`` must hold only plain types (dict/list/str/number/None) for weights_only loading."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "extra": extra,
        },
        path,
    )


def load_checkpoint(path, model, optimiser=None, scheduler=None, device="cpu") -> dict:
    """Load weights (optionally optimiser/scheduler state) in place; returns the checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimiser is not None:
        optimiser.load_state_dict(ckpt["optimiser"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def fit(model, train_loader, val_loader, loss_fn, cfg: TrainConfig, extra: dict):
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimiser, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2
    )

    device = get_device(cfg.device)
    model.to(device)
    early_stopper = EarlyStopper(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimiser, device, grad_clip=cfg.grad_clip
        )
        val_loss = evaluate(model, val_loader, loss_fn, device)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        pd.DataFrame(history).to_csv(cfg.out_dir / "history.csv", index=False)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            save_checkpoint(
                cfg.out_dir / "best.pt", model, optimiser, scheduler, epoch, val_loss, extra
            )

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            print(
                f"epoch {epoch:4d}  train {train_loss:.4e}  val {val_loss:.4e}  "
                f"best {best_val:.4e}  lr {lr:.2e}{'  *' if improved else ''}"
            )

        if early_stopper.early_stop(val_loss):
            print(
                f"early stop at epoch {epoch}: no improvement > {cfg.min_delta:g} "
                f"in {cfg.patience} epochs"
            )
            break

    return history
