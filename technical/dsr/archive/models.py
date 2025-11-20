"""Lightweight color DSR model definition and loading helpers.

This module factors out the super-resolution network used during training so that it
can be reused for inference inside portable pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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


def _load_checkpoint(
    weights_path: Path | str, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict]:
    """Return the checkpoint state dict along with raw metadata."""

    checkpoint = torch.load(Path(weights_path), map_location=device, weights_only=False)

    state_dict: Optional[dict[str, torch.Tensor]] = None
    metadata: dict = {}

    if isinstance(checkpoint, dict):
        # Preserve auxiliary entries (e.g. config, epoch)
        metadata = checkpoint
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value  # type: ignore[assignment]
                break
        if state_dict is None:
            # Heuristic: assume tensor-only dict is already a state dict
            tensor_items = {
                k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)
            }
            if tensor_items:
                state_dict = tensor_items
    elif hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
        metadata = {
            "config": getattr(checkpoint, "config", None),
        }
    else:
        raise ValueError(f"Unsupported checkpoint format at {weights_path}")

    if state_dict is None:
        raise ValueError(f"No state_dict found inside checkpoint {weights_path}")

    return state_dict, metadata


def _strip_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("model.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _is_hybrid_dsr(state: dict[str, torch.Tensor]) -> bool:
    """Detect if checkpoint is a Hybrid DSR model (has transformer blocks)."""
    return any(key.startswith("transformer_blocks.") for key in state)


def _infer_dsr_config(state: dict[str, torch.Tensor]) -> DSRConfig:
    """Infer DSRConfig fields from checkpoint tensor shapes."""

    base_channels = 64
    conv_in = state.get("conv_in.weight")
    if isinstance(conv_in, torch.Tensor):
        base_channels = conv_in.shape[0]

    residual_block_indices = set()
    for key in state:
        if key.startswith("residual_blocks"):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                residual_block_indices.add(int(parts[1]))

    residual_blocks = max(residual_block_indices) + 1 if residual_block_indices else 8

    return DSRConfig(base_channels=base_channels, residual_blocks=residual_blocks)


def load_dsr_model(
    weights_path: Path | str,
    device: torch.device | str = "cpu",
    config: Optional[DSRConfig] = None,
    strict: bool = False,
) -> Union[DSRColor, "HybridDSR"]:
    """Instantiate and load the DSR model on the requested device.
    
    Automatically detects whether the checkpoint is a basic DSR or Hybrid DSR
    and loads the appropriate architecture.
    """

    device = torch.device(device)
    state_dict, metadata = _load_checkpoint(weights_path, device)

    state_dict = _strip_prefixes(state_dict)

    # Detect if this is a hybrid DSR model
    if _is_hybrid_dsr(state_dict):
        # Import here to avoid circular dependency
        from .hybrid_model import HybridDSR, HybridDSRConfig
        
        # Try to get config from metadata
        hybrid_config = None
        if isinstance(metadata, dict) and "config" in metadata:
            meta_config = metadata["config"]
            if isinstance(meta_config, dict):
                # Try to extract vlr_size/input_size from training config
                vlr_size = meta_config.get("vlr_size") or meta_config.get("input_size")
                
                if vlr_size:
                    # Use the factory function to get correct architecture for this resolution
                    from .hybrid_model import create_hybrid_dsr
                    model = create_hybrid_dsr(vlr_size, target_size=112)
                    hybrid_config = model.config
                else:
                    # Try to construct HybridDSRConfig directly from metadata
                    try:
                        hybrid_config = HybridDSRConfig(**meta_config)
                    except Exception:
                        hybrid_config = None
        
        if hybrid_config is None:
            # Fall back to inference from state dict (may not get input_size right)
            from .hybrid_model import infer_hybrid_config
            hybrid_config = infer_hybrid_config(state_dict)
            print("[Hybrid DSR] Warning: Could not determine input_size from metadata, using inference")
        
        # Create model if we don't have it yet
        if 'model' not in locals():
            model = HybridDSR(hybrid_config)
        
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        
        if missing and not strict:
            print(f"[Hybrid DSR] Warning - missing keys: {missing[:5]}..." if len(missing) > 5 else f"[Hybrid DSR] Warning - missing keys: {missing}")
        if unexpected and not strict:
            print(f"[Hybrid DSR] Warning - unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"[Hybrid DSR] Warning - unexpected keys: {unexpected}")
        
        model.to(device)
        model.eval()
        return model
    
    # Load basic DSR model
    effective_config = config
    if effective_config is None:
        meta_config = None
        if isinstance(metadata, dict):
            meta_config = metadata.get("config")
        if meta_config and hasattr(DSRConfig, "__dataclass_fields__"):
            allowed = set(DSRConfig.__dataclass_fields__.keys())
            # Only accept metadata if it actually contains DSRConfig fields.
            filtered = {k: v for k, v in meta_config.items() if k in allowed}
            if filtered:
                try:
                    effective_config = DSRConfig(**filtered)
                except Exception:
                    effective_config = None
        if effective_config is None:
            effective_config = _infer_dsr_config(state_dict)

    model = DSRColor(config=effective_config)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing and not strict:
        print(f"[DSR] Warning - missing keys when loading checkpoint: {missing}")
    if unexpected and not strict:
        print(f"[DSR] Warning - unexpected keys in checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model
