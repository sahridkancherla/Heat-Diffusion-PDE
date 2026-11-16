"""Baselines with the same interface as FNO2d: (B, in_c, 8, W) -> (B, out_c, 8, W)."""

import torch.nn as nn
import torch.nn.functional as F


class CNN2d(nn.Module):
    """Residual stack of dilated convolutions over (z, t).

    Kernel (3, kernel_t) with time dilations ``dilations``; the time receptive field is
    1 + (kernel_t - 1) * sum(dilations) samples (253 for the defaults). ``causal`` pads time on the
    left only, so outputs depend on past inputs only.
    """

    def __init__(
        self,
        in_c,
        out_c=1,
        width=32,
        kernel_t=5,
        dilations=(1, 2, 4, 8, 16, 32),
        causal=False,
    ):
        super().__init__()
        self.causal = causal
        self.pads = [(kernel_t - 1) * d for d in dilations]
        self.lift = nn.Conv2d(in_c, width, 1)
        self.convs = nn.ModuleList(
            nn.Conv2d(width, width, (3, kernel_t), dilation=(1, d)) for d in dilations
        )
        self.project = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_c, 1))

    def forward(self, x):
        x = self.lift(x)
        for conv, p in zip(self.convs, self.pads, strict=True):
            pad_t = (p, 0) if self.causal else (p // 2, p - p // 2)
            x = x + F.gelu(conv(F.pad(x, (*pad_t, 1, 1))))
        return self.project(x)
