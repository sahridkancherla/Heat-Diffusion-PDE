"""Run a trained model over a whole recording."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from heatpde.data import RunNormalisers, Scales, WindowDataset
from heatpde.data.preprocess import ProcessedRun


@torch.no_grad()
def predict_run(
    model: torch.nn.Module,
    run: ProcessedRun,
    norm: RunNormalisers,
    scales: Scales,
    window: int,
    stride: int | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    **dataset_kwargs,
):
    """Predict every window of ``run`` and average overlaps back onto the full run.

    Each window starts from the measured initial profile (no autoregressive rollout).
    Returns (stitched (M, 8), window predictions (N, 8, W), window targets (N, 8, W)), all in °C.
    """
    ds = WindowDataset(
        [run], norm, scales, window=window, stride=stride or window, **dataset_kwargs
    )
    tail = (0, len(run.t) - window)
    if ds.index[-1] != tail:
        ds.index.append(tail)

    model.eval()
    preds, targets = [], []
    for x, y in DataLoader(ds, batch_size=batch_size):
        preds.append(model(x.to(device))[:, 0].cpu())
        targets.append(y[:, 0])
    pred = norm.T.decode(torch.cat(preds).numpy())
    true = norm.T.decode(torch.cat(targets).numpy())

    total = np.zeros((pred.shape[1], len(run.t)))
    count = np.zeros(len(run.t))
    for (_, s), p in zip(ds.index, pred, strict=True):
        total[:, s : s + window] += p
        count[s : s + window] += 1
    return (total / count).T, pred, true
