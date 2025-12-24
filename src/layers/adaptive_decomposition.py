"""
Adaptive decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, "B T C -> B C T")
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2), mode='replicate')
        x = self.avg(x)
        x = rearrange(x, "B C T -> B T C")
        return x

class LearnableClassicalDecomposition(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        hidden_dim: int,
        dropout: float,
        moving_avg_kernel_size: int,
        moving_avg_stride: int,
    ):
        super().__init__()

        self.moving_avg = MovingAverage(moving_avg_kernel_size, moving_avg_stride)

        self.map_trend = nn.Linear(sequence_length, sequence_length)
        self.map_season = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, sequence_length),
        )

    def forward(self, x):
        B, T, C = x.shape
        # --- Trend ---
        trend = self.moving_avg(x)  
        trend = rearrange(trend, "B T C -> (B C) T")
        trend = self.map_trend(trend)
        trend = rearrange(trend, "(B C) T -> B T C", C=C)

        # --- Season ---
        season = x - trend
        season = rearrange(season, "B T C -> (B C) T")
        season = self.map_season(season)
        season = rearrange(season, "(B C) T -> B T C", C=C)

        # --- Residual ---
        residual = x - trend - season

        return trend, season, residual







