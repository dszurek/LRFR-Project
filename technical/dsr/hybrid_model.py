"""Hybrid Transformer-CNN Super-Resolution Model.

Implements a lightweight hybrid architecture combining:
- Max-Feature-Map (MFM) activation for efficient feature selection
- Transformer attention blocks for global context
- Residual CNN blocks for local feature extraction
- Identity-aware loss functions (cosine + magnitude)

Based on concepts from:
- MFM from LightCNN-29v2 (Wu et al.)
- CMU-Net architecture (Yulianto et al. 2022)
- Vision Transformer attention mechanisms
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HybridDSRConfig:
    """Configuration for Hybrid Transformer-CNN DSR model."""
    
    # Input/Output dimensions
    input_size: int = 16  # VLR input size (16x16, 24x24, or 32x32)
    output_size: int = 112  # HR output size for EdgeFace
    base_channels: int = 64  # Base channel count (kept low for efficiency)
    
    # Architecture components
    num_residual_blocks: int = 6  # Number of MFM residual blocks
    num_transformer_blocks: int = 2  # Number of transformer attention blocks
    num_heads: int = 4  # Multi-head attention heads
    transformer_dim: int = 256  # Transformer hidden dimension
    mlp_ratio: float = 2.0  # MLP expansion ratio in transformer
    
    # MFM settings
    use_mfm: bool = True  # Use Max-Feature-Map activation
    
    # Dropout for regularization
    dropout: float = 0.1
    
    # Total parameters target: < 2M for efficiency
    

class MaxFeatureMap(nn.Module):
    """Max-Feature-Map activation function.
    
    Performs feature selection by splitting channels into two groups
    and taking the maximum value at each spatial location.
    This acts as a competitive relationship between feature maps,
    suppressing noise while enhancing meaningful features.
    
    From LightCNN-29v2 (Wu et al. 2018)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MFM activation.
        
        Args:
            x: Input tensor (B, C, H, W) where C must be even
            
        Returns:
            Output tensor (B, C//2, H, W)
        """
        if x.size(1) % 2 != 0:
            raise ValueError(f"MFM requires even number of channels, got {x.size(1)}")
        
        # Split channels into two groups and take max
        x1, x2 = torch.chunk(x, 2, dim=1)
        return torch.max(x1, x2)


class MFMConvBlock(nn.Module):
    """Convolutional block with MFM activation and skip connection.
    
    Structure:
    - Conv2d (outputs 2x channels for MFM)
    - BatchNorm
    - MFM activation (reduces back to original channels)
    - Skip connection
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_skip: bool = True
    ):
        super().__init__()
        
        self.use_skip = use_skip and (in_channels == out_channels) and (stride == 1)
        
        # Convolution outputs 2x channels for MFM
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.mfm = MaxFeatureMap()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.mfm(out)
        
        if self.use_skip:
            out = out + identity
        
        return out


class ResidualMFMBlock(nn.Module):
    """Residual block with two MFM conv layers.
    
    Based on ResBlock from LightCNN-29v2 and CMU-Net paper.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv1 = MFMConvBlock(channels, channels, use_skip=False)
        self.conv2 = MFMConvBlock(channels, channels, use_skip=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity


class TransformerAttentionBlock(nn.Module):
    """Lightweight transformer attention block for global context.
    
    Uses efficient windowed attention to capture long-range dependencies
    while maintaining low computational cost.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer attention.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence: (B, H*W, C)
        x_seq = x.flatten(2).transpose(1, 2)
        
        # Self-attention with residual
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out
        
        # MLP with residual
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        
        # Reshape back to spatial: (B, C, H, W)
        return x_seq.transpose(1, 2).reshape(B, C, H, W)


class PixelShuffle2x(nn.Module):
    """Efficient 2x upsampling using pixel shuffle (sub-pixel convolution)."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Conv outputs 4x channels for 2x2 pixel shuffle
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 4,
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.act = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        return x


class HybridDSR(nn.Module):
    """Hybrid Transformer-CNN Super-Resolution Network.
    
    Architecture stages:
    1. Shallow feature extraction (single conv)
    2. Deep feature extraction (MFM residual blocks)
    3. Global context modeling (transformer attention)
    4. Progressive upsampling (pixel shuffle)
    5. Final reconstruction (conv)
    
    Key innovations from CMU-Net paper:
    - MFM activation for efficient feature selection
    - Identity-aware training (cosine + magnitude loss)
    - Lightweight design (< 2M parameters)
    """
    
    def __init__(self, config: HybridDSRConfig):
        super().__init__()
        
        self.config = config
        
        # Calculate upsampling factor
        self.scale_factor = config.output_size // config.input_size
        # Note: 24→112 and 32→112 will use integer scale_factor then interpolate to exact size
        self.exact_scale = config.output_size / config.input_size
        if self.scale_factor not in [2, 3, 4, 7, 8]:
            raise ValueError(f"Unsupported scale factor: {self.scale_factor}")
        
        # Stage 1: Shallow feature extraction
        self.shallow_feat = nn.Conv2d(
            3,  # RGB input
            config.base_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Stage 2: Deep feature extraction with MFM residual blocks
        res_blocks = []
        for _ in range(config.num_residual_blocks):
            res_blocks.append(ResidualMFMBlock(config.base_channels))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Fusion conv after residual blocks
        self.fusion = nn.Conv2d(
            config.base_channels,
            config.base_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Stage 3: Transformer attention for global context
        # Only apply if resolution is reasonable (not too small)
        self.use_transformer = config.input_size >= 16
        if self.use_transformer:
            # Project to transformer dimension
            self.to_transformer = nn.Conv2d(
                config.base_channels,
                config.transformer_dim,
                kernel_size=1,
                bias=True
            )
            
            # Transformer blocks
            transformer_blocks = []
            for _ in range(config.num_transformer_blocks):
                transformer_blocks.append(
                    TransformerAttentionBlock(
                        config.transformer_dim,
                        config.num_heads,
                        config.mlp_ratio,
                        config.dropout
                    )
                )
            self.transformer_blocks = nn.Sequential(*transformer_blocks)
            
            # Project back from transformer
            self.from_transformer = nn.Conv2d(
                config.transformer_dim,
                config.base_channels,
                kernel_size=1,
                bias=True
            )
        
        # Stage 4: Progressive upsampling
        upsample_blocks = []
        current_channels = config.base_channels
        
        # Handle different scale factors
        if self.scale_factor == 2:
            # 2x: Single pixel shuffle
            upsample_blocks.append(PixelShuffle2x(current_channels, current_channels // 2))
            current_channels = current_channels // 2
        elif self.scale_factor == 3:
            # 3x (32→96, then bilinear to 112): Single 4x upsample then downsample
            # 32 → 128 (via 2x → 2x) → bilinear to 112
            upsample_blocks.append(PixelShuffle2x(current_channels, current_channels // 2))
            current_channels = current_channels // 2
            upsample_blocks.append(PixelShuffle2x(current_channels, 32))
            current_channels = 32
        elif self.scale_factor == 4:
            # 4x: Two 2x upsamples
            upsample_blocks.append(PixelShuffle2x(current_channels, current_channels // 2))
            current_channels = current_channels // 2
            upsample_blocks.append(PixelShuffle2x(current_channels, 32))
            current_channels = 32
        elif self.scale_factor == 7:
            # 7x (16→112): 2x → 2x → bilinear to 112
            # 16 → 32 → 64 → 112
            upsample_blocks.append(PixelShuffle2x(current_channels, current_channels // 2))
            current_channels = current_channels // 2
            upsample_blocks.append(PixelShuffle2x(current_channels, 48))
            current_channels = 48
        elif self.scale_factor == 8:
            # 8x: Three 2x upsamples
            upsample_blocks.append(PixelShuffle2x(current_channels, current_channels // 2))
            current_channels = current_channels // 2
            upsample_blocks.append(PixelShuffle2x(current_channels, 48))
            current_channels = 48
            upsample_blocks.append(PixelShuffle2x(current_channels, 32))
            current_channels = 32
        
        self.upsample = nn.Sequential(*upsample_blocks)
        
        # Stage 5: Final reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(current_channels, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)  # RGB output
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input LR image (B, 3, H, W) in [0, 1] range
            
        Returns:
            Output SR image (B, 3, H*scale, W*scale) in [0, 1] range
        """
        # Stage 1: Shallow features
        feat_shallow = self.shallow_feat(x)
        
        # Stage 2: Deep features with residual learning
        feat_deep = self.res_blocks(feat_shallow)
        feat_deep = self.fusion(feat_deep)
        feat_deep = feat_deep + feat_shallow  # Global residual
        
        # Stage 3: Global context with transformer
        if self.use_transformer:
            feat_global = self.to_transformer(feat_deep)
            feat_global = self.transformer_blocks(feat_global)
            feat_global = self.from_transformer(feat_global)
            feat_deep = feat_deep + feat_global  # Combine local + global
        
        # Stage 4: Progressive upsampling
        feat_up = self.upsample(feat_deep)
        
        # Handle non-power-of-2 scales with final resize
        current_size = self.config.input_size * (2 ** len(self.upsample))
        if current_size != self.config.output_size:
            # Bilinear interpolation to target size
            feat_up = F.interpolate(
                feat_up,
                size=(self.config.output_size, self.config.output_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Stage 5: Final reconstruction
        out = self.reconstruction(feat_up)
        
        # Clamp to valid range
        out = torch.clamp(out, 0.0, 1.0)
        
        return out
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def infer_hybrid_config(state_dict: dict[str, torch.Tensor]) -> HybridDSRConfig:
    """Infer HybridDSRConfig from a state dict.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Inferred HybridDSRConfig
    """
    # Infer base_channels from shallow_feat
    base_channels = 64
    if "shallow_feat.weight" in state_dict:
        base_channels = state_dict["shallow_feat.weight"].shape[0]
    
    # Count residual blocks
    res_block_indices = set()
    for key in state_dict:
        if key.startswith("res_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                res_block_indices.add(int(parts[1]))
    num_residual_blocks = max(res_block_indices) + 1 if res_block_indices else 6
    
    # Count transformer blocks
    transformer_block_indices = set()
    for key in state_dict:
        if key.startswith("transformer_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                transformer_block_indices.add(int(parts[1]))
    num_transformer_blocks = max(transformer_block_indices) + 1 if transformer_block_indices else 2
    
    # Infer transformer_dim from to_transformer projection
    transformer_dim = 256
    if "to_transformer.weight" in state_dict:
        transformer_dim = state_dict["to_transformer.weight"].shape[0]
    
    # Infer num_heads from attention projection weights
    num_heads = 4
    if "transformer_blocks.0.attn.in_proj_weight" in state_dict:
        # in_proj_weight has shape (3 * transformer_dim, transformer_dim)
        # num_heads = transformer_dim // head_dim, typically head_dim = 64
        head_dim = 64
        num_heads = transformer_dim // head_dim
    
    # Infer input_size - this is trickier, default to 16
    # Could be improved by checking metadata if available
    input_size = 16
    
    # Create config with inferred values
    config = HybridDSRConfig(
        input_size=input_size,
        output_size=112,
        base_channels=base_channels,
        num_residual_blocks=num_residual_blocks,
        num_transformer_blocks=num_transformer_blocks,
        num_heads=num_heads,
        transformer_dim=transformer_dim,
        mlp_ratio=2.0,
        dropout=0.1
    )
    
    return config


def create_hybrid_dsr(vlr_size: int, target_size: int = 112) -> HybridDSR:
    """Factory function to create optimized hybrid DSR model.
    
    Args:
        vlr_size: Input VLR size (16, 24, or 32)
        target_size: Output HR size (default 112 for EdgeFace)
        
    Returns:
        Configured HybridDSR model
    """
    # Adjust architecture based on input resolution - OPTIMIZED FOR A100 (Target ~4.5M - 5M params)
    if vlr_size <= 16:
        # 16x16 -> 112x112 (7x upscale)
        # Est: ~4.8M params
        config = HybridDSRConfig(
            input_size=vlr_size,
            output_size=target_size,
            base_channels=96,         # Reduced from 128
            num_residual_blocks=8,    # Reduced from 12
            num_transformer_blocks=4, # Kept at 4
            num_heads=8,
            transformer_dim=256,      # Reduced from 384
            mlp_ratio=2.0,
            dropout=0.1
        )
    elif vlr_size <= 24:
        # 24x24 -> 112x112 (~4.67x upscale)
        # Est: ~4.8M params
        config = HybridDSRConfig(
            input_size=24,
            output_size=target_size,
            base_channels=112,        # Reduced from 144
            num_residual_blocks=6,    # Reduced from 10
            num_transformer_blocks=4, # Kept at 4
            num_heads=8,
            transformer_dim=256,      # Reduced from 384
            mlp_ratio=2.0,
            dropout=0.1
        )
    else:
        # 32x32 -> 112x112 (3.5x upscale)
        # Est: ~5.1M params
        config = HybridDSRConfig(
            input_size=vlr_size,
            output_size=target_size,
            base_channels=120,        # Reduced from 160
            num_residual_blocks=5,    # Reduced from 8
            num_transformer_blocks=3, # Reduced from 6
            num_heads=8,
            transformer_dim=320,      # Reduced from 448
            mlp_ratio=2.0,
            dropout=0.1
        )
    
    model = HybridDSR(config)
    print(f"Created Hybrid DSR model for {vlr_size}x{vlr_size} -> {target_size}x{target_size}")
    print(f"Parameters: {model.count_parameters():,}")
    
    return model



# ==============================================================================
# Model Loading Utilities
# ==============================================================================

def _load_checkpoint(
    weights_path: str | Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict]:
    """Return the checkpoint state dict along with raw metadata."""
    from pathlib import Path
    
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    state_dict: Optional[dict[str, torch.Tensor]] = None
    metadata: dict = {}

    if isinstance(checkpoint, dict):
        # Preserve auxiliary entries (e.g. config, epoch)
        metadata = checkpoint
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
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


def load_dsr_model(
    weights_path: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = False,
) -> HybridDSR:
    """Instantiate and load the Hybrid DSR model on the requested device.
    
    Args:
        weights_path: Path to model checkpoint
        device: Device to load model on
        strict: Whether to enforce strict state dict matching
        
    Returns:
        Loaded HybridDSR model
    """
    device = torch.device(device)
    state_dict, metadata = _load_checkpoint(weights_path, device)
    state_dict = _strip_prefixes(state_dict)

    # Try to get config from metadata
    hybrid_config = None
    if isinstance(metadata, dict) and "config" in metadata:
        meta_config = metadata["config"]
        if isinstance(meta_config, dict):
            # Try to extract vlr_size/input_size from training config
            vlr_size = meta_config.get("vlr_size") or meta_config.get("input_size")
            
            if vlr_size:
                # Use the factory function to get correct architecture for this resolution
                model = create_hybrid_dsr(vlr_size, target_size=112)
                hybrid_config = model.config
            else:
                # Try to construct HybridDSRConfig directly from metadata
                try:
                    hybrid_config = HybridDSRConfig(**meta_config)
                except Exception:
                    hybrid_config = None
    
    if hybrid_config is None:
        # Fall back to inference from state dict
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


if __name__ == "__main__":
    # Test model creation and forward pass
    for vlr_size in [16, 24, 32]:
        print(f"\n{'='*60}")
        model = create_hybrid_dsr(vlr_size)
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 3, vlr_size, vlr_size)
        
        with torch.no_grad():
            y = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
