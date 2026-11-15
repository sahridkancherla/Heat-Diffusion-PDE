"Module for spectral conv and FNO"

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_c, out_c, k_z, k_t):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k_z = k_z
        self.k_t = k_t

        scale = 1 / (in_c * out_c)
        shape = (in_c, out_c, k_z, k_t)
        self.w_pos = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    def forward(self, x): # b, c, z, t
        z, t = x.shape[-2:]
        if 2 * self.k_z > z or self.k_t > t // 2 + 1:
            raise ValueError(f"Modes ({self.k_z}, {self.k_t}) do not fit a ({z}, {t}) grid.")
        xf = torch.fft.rfft2(x) # b, c, z, t//2 + 1
        out = torch.zeros(x.shape[0], self.out_c, z, t//2 +1, dtype=torch.cfloat, device=x.device)
        mul = lambda a, w: torch.einsum('bczt,cozt->bozt', a, w)
        out[..., :self.k_z, :self.k_t] = mul(xf[..., :self.k_z, :self.k_t], self.w_pos)
        out[..., -self.k_z:, :self.k_t] = mul(xf[..., -self.k_z:, :self.k_t], self.w_neg)

        return torch.fft.irfft2(out, s=(z,t))

class FNO2d(nn.Module):
    def __init__(self, k_z, k_t, width=16, in_c=3, out_c=1, num_layers=4, pad_t_frac=4, pad_z=4):
        super().__init__()
        self.pad_t_frac = pad_t_frac
        self.pad_z = pad_z
        self.k_z = k_z
        self.k_t = k_t
        self.width = width
        self.num_layers = num_layers

        self.lift = nn.Conv2d(in_c, self.width, 1)

        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.k_z, self.k_t) for _ in range(self.num_layers)
        ])

        self.w_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1) for _ in range(self.num_layers)
        ])

        self.project = nn.Sequential(
            nn.Conv2d(self.width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_c, 1)
        )

    def forward(self, x):
        z, t = x.shape[-2:]
        x = self.lift(x)
        x = F.pad(x, (0, int(self.pad_t_frac * t), 0, self.pad_z))

        for n in range(self.num_layers):
            x = F.gelu(self.fourier_layers[n](x) + self.w_layers[n](x))

        x = x[..., :z, :t]
        x = self.project(x)
        return x