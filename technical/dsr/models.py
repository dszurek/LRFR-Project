"""Lightweight color DSR model definition and loading helpers.

This module factors out the super-resolution network used during training so that it
can be reused for inference inside portable pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms

__all__ = ["ResidualBlock", "DSRColor", "load_dsr_model", "DSRConfig"]


@dataclass
class DSRConfig:
    """Configuration controlling the DSR model layout."""

    in_channels: int = 3
    base_channels: int = 64
    residual_blocks: int = 8
    upsample_factors: tuple[int, ...] = (2, 5)
    output_size: tuple[int, int] = (160, 160)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        return self.prelu(x + residual)


class DSRColor(nn.Module):
    """Color super-resolution network used for DSR."""

    def __init__(self, config: Optional[DSRConfig] = None) -> None:
        super().__init__()
        self.config = config or DSRConfig()
        c = self.config

        self.conv_in = nn.Conv2d(
            c.in_channels, c.base_channels, kernel_size=3, stride=1, padding=1
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(c.base_channels) for _ in range(c.residual_blocks)]
        )
        self.upsamplers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        c.base_channels,
                        c.base_channels * (factor**2),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.PixelShuffle(factor),
                    nn.PReLU(),
                )
                for factor in c.upsample_factors
            ]
        )
        self.conv_out = nn.Conv2d(
            c.base_channels, c.in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv_in(x)
        x = self.residual_blocks(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        x = self.conv_out(x)
        return transforms.functional.resize(x, self.config.output_size, antialias=True)


def _load_checkpoint(weights_path: Path | str, device: torch.device) -> dict:
    checkpoint = torch.load(Path(weights_path), map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format at {weights_path}")


def load_dsr_model(
    weights_path: Path | str,
    device: torch.device | str = "cpu",
    config: Optional[DSRConfig] = None,
    strict: bool = False,
) -> DSRColor:
    """Instantiate and load the DSR model on the requested device."""

    device = torch.device(device)
    model = DSRColor(config=config)
    state_dict = _load_checkpoint(weights_path, device)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing and not strict:
        print(f"[DSR] Warning - missing keys when loading checkpoint: {missing}")
    if unexpected and not strict:
        print(f"[DSR] Warning - unexpected keys in checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model
