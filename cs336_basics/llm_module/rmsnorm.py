import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # mean of squares
        rms = x.pow(2).mean(dim=(-1,), keepdim=True).add(self.eps).sqrt()
        y = x / rms

        if self.weight is not None:
            y = y * self.weight  # broadcast over leading dims

        return y