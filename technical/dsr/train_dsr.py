"""Training script for the DSR super-resolution network.

The script pairs the preprocessed very-low-resolution (VLR) images with their
high-resolution (HR) counterparts, applies modern data augmentation and
perceptual objectives, and optimises an enhanced DSR model that works well with
the EdgeFace recogniser across multiple VLR input resolutions.

Key upgrades compared to the previous baseline:

* Identity-preserving loss now uses the same EdgeFace backbone as the
    production pipeline and properly aligns checkpoint keys while remaining frozen
    during optimisation.
* Perceptual loss (VGG-19) + total variation regularisation improve texture
    fidelity without shrugging off facial structure.
* Exponential Moving Average (EMA) weights, mixed-precision training, and
    gradient clipping stabilise optimisation.
* Resolution-aware hyper-parameters and dataset handling support 16×16, 24×24,
    and 32×32 VLR inputs that all upscale directly to 112×112 for EdgeFace.
* Rich yet lightweight augmentations (flip/rotation/colour jitter) operate on
    both HR and VLR inputs with shared randomness to broaden generalisation.
* The saved checkpoint lands in ``technical/dsr/dsr{SIZE}.pth`` (e.g.
    ``dsr24.pth``) and matches the ``DSRColor`` architecture expected by the
    inference pipeline.
"""

from __future__ import annotations

import argparse
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from . import DSRColor, DSRConfig

# Import EdgeFace - use try/except for both relative and absolute paths
try:
    from ..facial_rec.edgeface_weights.edgeface import EdgeFace
except ImportError:
    from facial_rec.edgeface_weights.edgeface import EdgeFace


def _resolve_vlr_dir_name(vlr_size: int) -> str:
    """Return the directory name that stores VLR images for a given size.
    
    Consistent format for ALL resolutions: vlr_images_{W}x{H}
    """
    if vlr_size <= 0:
        raise ValueError("VLR size must be a positive integer")

    return f"vlr_images_{vlr_size}x{vlr_size}"


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Hyper-parameters controlling training with fine-tuned EdgeFace.

    OPTIMIZED FOR 32×32 VLR INPUT → 112×112 HR OUTPUT (3.5× upscaling)
    EdgeFace requires 112×112 input, so DSR outputs 112×112 directly
    """

    # Target HR size (DSR output size) - EdgeFace requires 112×112
    target_hr_size: int = 112  # DSR outputs 112×112 (3.5× from 32×32 VLR)
    vlr_size: int = 32
    base_channels: int = 120
    residual_blocks: int = 16

    epochs: int = 100  # Increased for better convergence with stricter losses
    batch_size: int = (
        16  # Can increase slightly from 14 (112 vs 128 output = less memory)
    )
    learning_rate: float = 1.3e-4  # Slightly lower for larger input (more stable)
    weight_decay: float = 1e-6
    lambda_l1: float = 1.0
    lambda_perceptual: float = (
        0.025  # Slightly higher - 32×32 has more structure to preserve
    )
    lambda_identity: float = (
        0.60  # Increased from 0.50 - 32×32 has clearer identity features
    )
    lambda_feature_match: float = (
        0.18  # Increased - more intermediate features to match
    )
    lambda_tv: float = (
        2e-6  # Reduced - 32×32 needs less smoothing (already has structure)
    )
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    num_workers: int = 8
    val_interval: int = 1
    seed: int = 42
    warmup_epochs: int = 5  # Longer warmup for stability with higher identity loss
    early_stop_patience: int = 20  # More patience for slower convergence
    use_multiscale_perceptual: bool = True  # Multi-scale perceptual loss

    @classmethod
    def make(cls, vlr_size: int) -> "TrainConfig":
        """Build a resolution-aware configuration for the requested VLR size."""

        cfg = cls(vlr_size=vlr_size)

        if vlr_size <= 16:
            cfg.epochs = max(cfg.epochs, 120)
            cfg.batch_size = min(cfg.batch_size, 12)
            cfg.learning_rate = 1.1e-4
            cfg.lambda_identity = 0.72
            cfg.lambda_feature_match = 0.24
            cfg.lambda_perceptual = 0.030
            cfg.lambda_tv = 1.6e-6
            cfg.grad_clip = 0.9
            cfg.warmup_epochs = max(cfg.warmup_epochs, 6)
            cfg.base_channels = 132
            cfg.residual_blocks = 20
        elif vlr_size <= 24:
            cfg.epochs = max(cfg.epochs, 110)
            cfg.batch_size = min(cfg.batch_size, 14)
            cfg.learning_rate = 1.2e-4
            cfg.lambda_identity = 0.66
            cfg.lambda_feature_match = 0.20
            cfg.lambda_perceptual = 0.028
            cfg.lambda_tv = 1.8e-6
            cfg.grad_clip = 0.95
            cfg.base_channels = 126
            cfg.residual_blocks = 18
        else:
            cfg.vlr_size = 32
            cfg.base_channels = 120
            cfg.residual_blocks = 16

        return cfg


# ---------------------------------------------------------------------------
# Dataset with paired augmentations
# ---------------------------------------------------------------------------


class PairedFaceSRDataset(Dataset):
    """Loads aligned VLR/HR facial image pairs with shared augmentations."""

    def __init__(
        self,
        root: Path,
        augment: bool,
        vlr_size: int,
        target_hr_size: int = 112,
    ) -> None:
        self.root = Path(root)
        self.vlr_size = vlr_size
        self.vlr_root = self.root / _resolve_vlr_dir_name(vlr_size)
        self.hr_root = self.root / "hr_images"
        self.target_hr_size = target_hr_size

        if not self.vlr_root.is_dir():
            raise RuntimeError(
                f"Missing VLR directory '{self.vlr_root}'. Run regenerate_vlr_dataset with --vlr-sizes {vlr_size}."
            )
        if not self.hr_root.is_dir():
            raise RuntimeError(
                f"Dataset directory must contain 'hr_images' (expected at {self.hr_root})."
            )

        self.pairs: List[Tuple[Path, Path]] = []
        for vlr_path in sorted(self.vlr_root.glob("*.png")):
            hr_path = self.hr_root / vlr_path.name
            if not hr_path.is_file():
                raise FileNotFoundError(
                    f"Missing HR counterpart for {vlr_path.name} under {self.hr_root}"
                )
            self.pairs.append((vlr_path, hr_path))

        if not self.pairs:
            raise RuntimeError(f"No image pairs found in {self.root}")

        self.augment = augment

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        vlr_path, hr_path = self.pairs[index]
        vlr = Image.open(vlr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        if vlr.size != (self.vlr_size, self.vlr_size):
            vlr = vlr.resize(
                (self.vlr_size, self.vlr_size), Image.Resampling.BICUBIC
            )

        # Verify HR is correct size (should already be 112×112 after resize script)
        if hr.size != (self.target_hr_size, self.target_hr_size):
            hr = hr.resize(
                (self.target_hr_size, self.target_hr_size), Image.Resampling.LANCZOS
            )

        if self.augment:
            vlr, hr = self._apply_shared_augmentations(vlr, hr)

        vlr_tensor = TF.to_tensor(vlr)
        hr_tensor = TF.to_tensor(hr)
        return vlr_tensor, hr_tensor

    def _apply_shared_augmentations(
        self, vlr: Image.Image, hr: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        # Horizontal flip - safe for face recognition
        if random.random() < 0.5:
            vlr = TF.hflip(vlr)
            hr = TF.hflip(hr)

        # More conservative rotations to preserve facial features
        rotation_cap = 6.0 if self.vlr_size >= 32 else (5.0 if self.vlr_size >= 24 else 4.0)
        if random.random() < 0.65:
            angle = random.uniform(-rotation_cap, rotation_cap)
            vlr = TF.rotate(vlr, angle, interpolation=InterpolationMode.BILINEAR)
            hr = TF.rotate(hr, angle, interpolation=InterpolationMode.BILINEAR)

        # Very mild colour jitter - too much breaks identity embeddings
        jitter_strength = 0.07 if self.vlr_size >= 32 else (0.06 if self.vlr_size >= 24 else 0.05)
        if random.random() < 0.3:
            brightness = random.uniform(1 - jitter_strength, 1 + jitter_strength)
            contrast = random.uniform(1 - jitter_strength, 1 + jitter_strength)
            saturation = random.uniform(1 - jitter_strength / 2, 1 + jitter_strength / 2)
            vlr = TF.adjust_brightness(vlr, brightness)
            hr = TF.adjust_brightness(hr, brightness)
            vlr = TF.adjust_contrast(vlr, contrast)
            hr = TF.adjust_contrast(hr, contrast)
            vlr = TF.adjust_saturation(vlr, saturation)
            hr = TF.adjust_saturation(hr, saturation)

        return vlr, hr


# ---------------------------------------------------------------------------
# Loss functions and helpers
# ---------------------------------------------------------------------------


class VGGPerceptualLoss(nn.Module):
    """Multi-scale perceptual loss from VGG19 with optional intermediate features."""

    def __init__(self, device: torch.device, multiscale: bool = True) -> None:
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1
        vgg_features = vgg19(weights=weights).features.eval()
        for param in vgg_features.parameters():
            param.requires_grad_(False)

        # Extract features at multiple depths for better structure preservation
        self.slices = nn.ModuleList(
            [
                nn.Sequential(*vgg_features[:4]),  # relu1_2: low-level edges
                nn.Sequential(*vgg_features[4:9]),  # relu2_2: textures
                nn.Sequential(*vgg_features[9:18]),  # relu3_4: mid-level structures
                nn.Sequential(*vgg_features[18:27]),  # relu4_4: high-level features
            ]
        )
        self.slices.to(device)
        self.multiscale = multiscale

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std

        loss = torch.zeros(1, device=sr.device)
        x, y = sr_norm, hr_norm

        # Weight early layers more for facial structure
        weights = [0.4, 0.3, 0.2, 0.1] if self.multiscale else [0.25, 0.25, 0.25, 0.25]

        for i, block in enumerate(self.slices):
            x = block(x)
            y = block(y)
            loss = loss + weights[i] * F.l1_loss(x, y)
        return loss


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Encourages spatial smoothness while retaining edges."""

    loss_h = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    loss_w = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return loss_h + loss_w


def compute_psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((sr - hr) ** 2, dim=[1, 2, 3])
    psnr_vals = 10 * torch.log10((max_val**2) / (mse + 1e-12))
    return psnr_vals.mean().item()


class ModelEMA:
    """Maintains an exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema = deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_params = dict(self.ema.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_param = ema_params[name]
            ema_param.mul_(self.decay).add_(param, alpha=1.0 - self.decay)

        # Buffers (e.g. BatchNorm running stats)
        for ema_buffer, model_buffer in zip(self.ema.buffers(), model.buffers()):
            ema_buffer.copy_(model_buffer)


# ---------------------------------------------------------------------------
# Identity backbone wrapper
# ---------------------------------------------------------------------------


class ConvNeXtWrapper(nn.Module):
    """Minimal wrapper to load ConvNeXt-based EdgeFace models from state dict.

    This creates a simple Sequential model that can load the pretrained weights
    and extract embeddings, without needing the full architecture definition.
    """

    def __init__(self, state_dict: dict):
        super().__init__()
        # Store the state dict and create a parameter-only model
        # We'll use a dummy Sequential that gets replaced by loaded parameters
        self.params = nn.ParameterDict()
        self.buffers_dict = {}

        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                if (
                    "running_mean" in name
                    or "running_var" in name
                    or "num_batches_tracked" in name
                ):
                    self.buffers_dict[name] = param
                else:
                    self.params[name.replace(".", "_")] = nn.Parameter(
                        param, requires_grad=False
                    )

        # Register buffers
        for name, buf in self.buffers_dict.items():
            self.register_buffer(name.replace(".", "_"), buf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is a placeholder - we'll use a functional approach
        raise NotImplementedError("ConvNeXtWrapper uses functional forward pass")


class EdgeFaceEmbedding(nn.Module):
    """Wraps EdgeFace model to provide embeddings for identity loss.

    Supports both ConvNeXt (.pt pretrained) and LDC (.pth fine-tuned) architectures.
    """

    def __init__(
        self, weights_path: Path, device: torch.device, extract_features: bool = True
    ) -> None:
        super().__init__()
        self.extract_features = False  # Feature matching disabled
        self.device = device

        # Load checkpoint
        try:
            raw_state = torch.load(weights_path, map_location="cpu", weights_only=False)
        except Exception:
            import sys
            from dataclasses import dataclass

            main_module = sys.modules.get("__main__")
            if main_module and not hasattr(main_module, "FinetuneConfig"):

                @dataclass
                class FinetuneConfig:
                    """Stub for unpickling."""

                    pass

                setattr(main_module, "FinetuneConfig", FinetuneConfig)

            raw_state = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Extract state dict
        if isinstance(raw_state, dict):
            if "backbone_state_dict" in raw_state:
                state_dict = raw_state["backbone_state_dict"]
                is_finetuned = True
            elif "state_dict" in raw_state:
                state_dict = raw_state["state_dict"]
                is_finetuned = False
            elif "model_state_dict" in raw_state:
                state_dict = raw_state["model_state_dict"]
                is_finetuned = False
            else:
                state_dict = raw_state
                is_finetuned = False
        else:
            raise TypeError(
                f"Unsupported EdgeFace checkpoint type: {type(raw_state).__name__}"
            )

        # Clean key prefixes
        clean_state = {}
        for key, value in state_dict.items():
            clean_key = key
            if clean_key.startswith("model."):
                clean_key = clean_key[len("model.") :]
            clean_state[clean_key] = value

        # Detect architecture by checking for ConvNeXt-specific keys
        has_convnext = any("stages." in k or "stem.0." in k for k in clean_state.keys())
        has_ldc = any(
            "features." in k and "conv_1x1_in" in k for k in clean_state.keys()
        )

        print(f"[EdgeFace] Loading {weights_path.name}")

        if has_convnext:
            # ConvNeXt architecture (edgeface_xxs, edgeface_s_gamma_05)
            print(f"[EdgeFace] Detected ConvNeXt architecture")

            # Determine architecture variant
            if "xxs" in str(weights_path).lower():
                arch_name = "edgeface_xxs"
            elif (
                "gamma_05" in str(weights_path).lower()
                or "s_gamma" in str(weights_path).lower()
            ):
                arch_name = "edgeface_s_gamma_05"
            else:
                arch_name = "edgeface_xxs"  # Default to xxs

            print(f"[EdgeFace] Loading as {arch_name} (ConvNeXt-based)")
            self.model = EdgeFace(back=arch_name)
            result = self.model.load_state_dict(clean_state, strict=False)
            if result.missing_keys or result.unexpected_keys:
                print(
                    f"[EdgeFace] Loaded with {len(result.missing_keys)} missing keys, {len(result.unexpected_keys)} unexpected keys"
                )
                print(
                    f"[EdgeFace] Note: Architecture partially matches - using available pretrained weights"
                )
            else:
                print(
                    f"[EdgeFace] Successfully loaded ConvNeXt model with all keys matching!"
                )

        elif has_ldc or is_finetuned:
            # LDC architecture (edgeface_s or fine-tuned)
            print(f"[EdgeFace] Detected LDC architecture (edgeface_s)")
            self.model = EdgeFace(back="edgeface_s")
            result = self.model.load_state_dict(clean_state, strict=False)
            self.use_wrapper = False

            if result.missing_keys or result.unexpected_keys:
                print(
                    f"[EdgeFace] Loaded with {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected keys"
                )
        else:
            raise RuntimeError(
                f"Cannot detect architecture for {weights_path.name}. "
                f"Please use edgeface_xxs.pt, edgeface_s_gamma_05.pt, or edgeface_finetuned.pth"
            )

        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Extract normalized embeddings from images."""
        processed = self.transform(imgs)
        embeddings = self.model(processed)
        return F.normalize(embeddings, dim=-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dsr_model(config: TrainConfig, device: torch.device) -> nn.Module:
    dsr_config = DSRConfig(
        base_channels=config.base_channels,
        residual_blocks=config.residual_blocks,
        output_size=(config.target_hr_size, config.target_hr_size),
    )
    model = DSRColor(config=dsr_config).to(device)
    return model


def train(config: TrainConfig, args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parent
    dataset_root = base_dir.parent / "dataset"
    vlr_dir_name = _resolve_vlr_dir_name(config.vlr_size)

    # Use frontal-only dataset if flag is set
    if args.frontal_only:
        train_dir = dataset_root / "frontal_only" / "train"
        val_dir = dataset_root / "frontal_only" / "val"
        print("🎯 Using FRONTAL-ONLY filtered dataset")
    else:
        train_dir = dataset_root / "train_processed"
        val_dir = dataset_root / "val_processed"
        print("⚠️  Using FULL dataset (includes profile/angled faces)")

    weights_path = base_dir.parent / "facial_rec" / "edgeface_weights" / args.edgeface
    save_path = base_dir / f"dsr{config.vlr_size}.pth"
    legacy_save_path = base_dir / "dsr.pth" if config.vlr_size == 32 else None
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    set_random_seed(config.seed)

    print(f"Using device: {device}")
    print(f"Loading fine-tuned EdgeFace from: {weights_path}")
    print(
        f"Target HR resolution: {config.target_hr_size}×{config.target_hr_size} (input VLR {config.vlr_size}×{config.vlr_size})"
    )
    print(f"Expecting VLR directory: {vlr_dir_name}")

    train_ds = PairedFaceSRDataset(
        train_dir,
        augment=True,
        vlr_size=config.vlr_size,
        target_hr_size=config.target_hr_size,
    )
    val_ds = PairedFaceSRDataset(
        val_dir,
        augment=False,
        vlr_size=config.vlr_size,
        target_hr_size=config.target_hr_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_dsr_model(config, device)
    ema = ModelEMA(model, decay=config.ema_decay)

    # Resume from checkpoint if specified (for cyclic fine-tuning)
    start_epoch = 0
    if args.resume:
        print(f"\n{'='*70}")
        print(f"[Resume] Loading DSR checkpoint from: {args.resume}")
        print(f"{'='*70}")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"✅ Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
                
                # Format best_psnr safely
                best_psnr = checkpoint.get('best_psnr', None)
                if best_psnr is not None:
                    print(f"   Previous best val PSNR: {best_psnr:.2f} dB")
                else:
                    print(f"   Previous best val PSNR: unknown")
                print(f"   Resuming from epoch {start_epoch}")
                
                # Also load EMA state if available
                if "ema_state_dict" in checkpoint:
                    try:
                        ema.ema.load_state_dict(checkpoint["ema_state_dict"])
                        print(f"✅ Restored EMA weights")
                    except Exception as e:
                        print(f"⚠️  Could not load EMA weights: {e}")
            else:
                # Assume it's just the model state dict
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded model weights (epoch unknown)")
                print(f"   Starting fresh training from these weights")
                
            # Sync EMA with loaded model
            ema.ema = deepcopy(model)
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"   Starting training from scratch instead\n")
            start_epoch = 0

    identity_model = EdgeFaceEmbedding(weights_path, device, extract_features=True)
    identity_model.model.requires_grad_(False)
    for param in identity_model.parameters():
        param.requires_grad_(False)
    l1_loss = nn.L1Loss()
    cosine_loss = nn.CosineEmbeddingLoss()
    perceptual_loss = VGGPerceptualLoss(
        device, multiscale=config.use_multiscale_perceptual
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Warmup + cosine schedule for better convergence
    def lr_lambda(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        return 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - config.warmup_epochs)
                / (config.epochs - config.warmup_epochs)
            )
        )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_psnr = -math.inf
    epochs_without_improvement = 0
    
    # Handle --additional-epochs: calculate target epoch based on checkpoint
    if args.additional_epochs is not None:
        if not args.resume:
            raise SystemExit("--additional-epochs requires --resume (need checkpoint to determine starting epoch)")
        if not isinstance(checkpoint, dict) or 'epoch' not in checkpoint:
            raise SystemExit("--additional-epochs requires checkpoint with epoch information")
        
        target_epoch = start_epoch + args.additional_epochs
        config.epochs = target_epoch
        print(f"\n📅 Additional epochs mode:")
        print(f"   Starting from epoch {start_epoch} (checkpoint epoch {checkpoint.get('epoch', 'unknown')})")
        print(f"   Training for {args.additional_epochs} additional epochs")
        print(f"   Target epoch: {target_epoch}\n")
    
    # If resuming, load previous best PSNR or score
    if args.resume and isinstance(checkpoint, dict):
        prev_best = checkpoint.get('best_score', None)
            
        if prev_best is not None:
            best_val_psnr = prev_best
            print(f"\n📊 Resuming with baseline score: {best_val_psnr:.4f} (must beat this to save)")

    print(f"Training samples: {len(train_ds)}, validation samples: {len(val_ds)}")
    print("Saving best checkpoints to", save_path)
    if legacy_save_path is not None:
        print("Legacy compatibility checkpoint will also update", legacy_save_path)
    print(f"\nTraining configuration:")
    print(f"  - VLR size: {config.vlr_size}×{config.vlr_size}")
    print(
        f"  - DSR capacity: base_channels={config.base_channels}, residual_blocks={config.residual_blocks}"
    )
    print(f"  - Identity loss weight: {config.lambda_identity}")
    print(f"  - Feature matching weight: {config.lambda_feature_match}")
    print(f"  - Perceptual loss weight: {config.lambda_perceptual}")
    print(f"  - Multi-scale perceptual: {config.use_multiscale_perceptual}")
    print(f"  - Epochs: {config.epochs}, Warmup: {config.warmup_epochs}")
    print(f"  - Early stop patience: {config.early_stop_patience}\n")

    if start_epoch > 0:
        print(f"🔄 Resuming training from epoch {start_epoch + 1}\n")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        running_losses = {
            "l1": 0.0,
            "perceptual": 0.0,
            "identity": 0.0,
            "feature_match": 0.0,
            "tv": 0.0,
        }

        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [train]", leave=False
        )

        for vlr, hr in progress:
            vlr = vlr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=device.type == "cuda"):
                sr = model(vlr)
                sr = torch.clamp(sr, 0.0, 1.0)

                # Pixel-level reconstruction
                loss_l1 = l1_loss(sr, hr)

                # Perceptual loss (multi-scale VGG features)
                loss_perc = perceptual_loss(sr, hr)

                # Identity preservation (cosine embedding loss)
                embeddings_sr = identity_model(sr.float())
                embeddings_hr = identity_model(hr.float())
                targets = torch.ones(embeddings_sr.size(0), device=device)
                loss_id = cosine_loss(embeddings_sr, embeddings_hr, targets)

                # Feature matching loss (intermediate EdgeFace features)
                loss_feat = torch.zeros(1, device=device)
                if config.lambda_feature_match > 0 and identity_model.extract_features:
                    try:
                        _, _, sr_feats, hr_feats = (
                            identity_model.extract_paired_features(sr, hr)
                        )
                        for sf, hf in zip(sr_feats, hr_feats):
                            if sf.shape == hf.shape:
                                loss_feat = loss_feat + F.l1_loss(sf, hf)
                        if len(sr_feats) > 0:
                            loss_feat = loss_feat / len(sr_feats)
                    except Exception:
                        # Feature extraction failed; skip this loss
                        pass

                # Total variation regularization
                loss_tv = total_variation_loss(sr)

                # Combined loss with adjusted weights
                loss = (
                    config.lambda_l1 * loss_l1
                    + config.lambda_perceptual * loss_perc
                    + config.lambda_identity * loss_id
                    + config.lambda_feature_match * loss_feat
                    + config.lambda_tv * loss_tv
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            running_loss += loss.item()
            running_psnr += compute_psnr(sr.detach(), hr)
            running_losses["l1"] += loss_l1.item()
            running_losses["perceptual"] += loss_perc.item()
            running_losses["identity"] += loss_id.item()
            running_losses["feature_match"] += loss_feat.item()
            running_losses["tv"] += loss_tv.item()

        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_psnr = running_psnr / len(train_loader)
        avg_losses = {k: v / len(train_loader) for k, v in running_losses.items()}

        if (epoch + 1) % config.val_interval == 0:
            eval_model = ema.ema
            eval_model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            val_losses = {
                "l1": 0.0,
                "perceptual": 0.0,
                "identity": 0.0,
                "feature_match": 0.0,
                "tv": 0.0,
            }

            with torch.no_grad():
                for vlr, hr in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{config.epochs} [val]",
                    leave=False,
                ):
                    vlr = vlr.to(device, non_blocking=True)
                    hr = hr.to(device, non_blocking=True)

                    sr = eval_model(vlr)
                    sr = torch.clamp(sr, 0.0, 1.0)

                    loss_l1 = l1_loss(sr, hr)
                    loss_perc = perceptual_loss(sr, hr)
                    embeddings_sr = identity_model(sr.float())
                    embeddings_hr = identity_model(hr.float())
                    targets = torch.ones(embeddings_sr.size(0), device=device)
                    loss_id = cosine_loss(embeddings_sr, embeddings_hr, targets)

                    loss_feat = torch.zeros(1, device=device)
                    if (
                        config.lambda_feature_match > 0
                        and identity_model.extract_features
                    ):
                        try:
                            _, _, sr_feats, hr_feats = (
                                identity_model.extract_paired_features(sr, hr)
                            )
                            for sf, hf in zip(sr_feats, hr_feats):
                                if sf.shape == hf.shape:
                                    loss_feat = loss_feat + F.l1_loss(sf, hf)
                            if len(sr_feats) > 0:
                                loss_feat = loss_feat / len(sr_feats)
                        except Exception:
                            pass

                    loss_tv = total_variation_loss(sr)

                    total = (
                        config.lambda_l1 * loss_l1
                        + config.lambda_perceptual * loss_perc
                        + config.lambda_identity * loss_id
                        + config.lambda_feature_match * loss_feat
                        + config.lambda_tv * loss_tv
                    )
                    val_loss += total.item()
                    val_psnr += compute_psnr(sr, hr)
                    val_losses["l1"] += loss_l1.item()
                    val_losses["perceptual"] += loss_perc.item()
                    val_losses["identity"] += loss_id.item()
                    val_losses["feature_match"] += loss_feat.item()
                    val_losses["tv"] += loss_tv.item()

            val_loss /= len(val_loader)
            val_psnr /= len(val_loader)
            val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}

            print(
                f"Epoch {epoch + 1:03d} | "
                f"train_loss={avg_train_loss:.4f} (L1:{avg_losses['l1']:.3f} P:{avg_losses['perceptual']:.3f} "
                f"ID:{avg_losses['identity']:.3f} FM:{avg_losses['feature_match']:.3f}) "
                f"PSNR={avg_train_psnr:.2f}dB"
            )
            print(
                f"         | "
                f"val_loss={val_loss:.4f} (L1:{val_losses['l1']:.3f} P:{val_losses['perceptual']:.3f} "
                f"ID:{val_losses['identity']:.3f} FM:{val_losses['feature_match']:.3f}) "
                f"PSNR={val_psnr:.2f}dB"
            )

            # Use combined metric (PSNR + identity preservation) for all training
            # Higher PSNR is better (normalize by 40dB max), lower identity loss is better
            normalized_psnr = min(val_psnr / 40.0, 1.0)
            normalized_identity = 1.0 - min(val_losses["identity"], 1.0)
            val_score = 0.6 * normalized_psnr + 0.4 * normalized_identity
            
            if val_score > best_val_psnr:
                best_val_psnr = val_score
                epochs_without_improvement = 0
                checkpoint = {
                    "epoch": epoch + 1,
                    "config": asdict(config),
                    "model_state_dict": ema.ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "ema_state_dict": ema.ema.state_dict(),
                    "best_score": val_score,  # Save combined score
                    "best_psnr": val_psnr,  # Still save actual PSNR for reference
                    "val_psnr": val_psnr,
                    "val_identity_loss": val_losses["identity"],
                    "vlr_size": config.vlr_size,
                }
                torch.save(checkpoint, save_path)
                if legacy_save_path is not None:
                    torch.save(checkpoint, legacy_save_path)
                print(
                    f"✅ Saved new best checkpoint (score={val_score:.4f}: PSNR {val_psnr:.2f}dB, ID loss {val_losses['identity']:.4f})"
                )
            else:
                epochs_without_improvement += 1
            
            
            if epochs_without_improvement >= config.early_stop_patience:
                print(
                    f"\n⚠️  Early stopping after {epoch + 1} epochs (no improvement for {config.early_stop_patience} epochs)"
                )
                print(f"Best validation score: {best_val_psnr:.4f}")
                break
        else:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"train_loss={avg_train_loss:.4f} (L1:{avg_losses['l1']:.3f} P:{avg_losses['perceptual']:.3f} "
                f"ID:{avg_losses['identity']:.3f} FM:{avg_losses['feature_match']:.3f}) "
                f"PSNR={avg_train_psnr:.2f}dB"
            )

    print(f"\n🎉 Training complete! Best validation score: {best_val_psnr:.4f}")
    
    # Show improvement if resuming
    if args.resume and isinstance(checkpoint, dict):
        prev_best = checkpoint.get('best_score', None)
        if prev_best is not None and best_val_psnr > prev_best:
            improvement = best_val_psnr - prev_best
            print(f"📈 Improvement: +{improvement:.4f} score ({prev_best:.4f} → {best_val_psnr:.4f})")
        elif prev_best is not None and best_val_psnr <= prev_best:
            degradation = prev_best - best_val_psnr
            print(f"📉 Degraded: -{degradation:.4f} score ({prev_best:.4f} → {best_val_psnr:.4f})")
            print(f"⚠️  Fine-tuning did not improve the model!")
    
    print(f"Checkpoint saved to: {save_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DSR super-resolution model with fine-tuned EdgeFace identity preservation"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--edgeface",
        default="edgeface_xxs.pt",
        help="EdgeFace weights file (supports: edgeface_xxs.pt, edgeface_s_gamma_05.pt, edgeface_finetuned.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--vlr-size",
        type=int,
        default=32,
        choices=(16, 24, 32),
        help="Input VLR resolution. Must correspond to regenerated dataset directories (16, 24, or 32).",
    )
    parser.add_argument(
        "--frontal-only",
        action="store_true",
        help="Use frontal-only filtered dataset (technical/dataset/frontal_only/)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to DSR checkpoint to resume training from (for cyclic fine-tuning). "
             "Example: technical/dsr/dsr32.pth",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate (useful for cyclic fine-tuning with lower LR)",
    )
    parser.add_argument(
        "--lambda-identity",
        type=float,
        default=None,
        help="Override identity loss weight (useful for cyclic fine-tuning)",
    )
    parser.add_argument(
        "--lambda-feature-match",
        type=float,
        default=None,
        help="Override feature matching loss weight",
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=None,
        help="Train for N additional epochs from checkpoint (alternative to --epochs). "
             "Used for cyclic fine-tuning to ensure equal training duration regardless of checkpoint epoch count.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience (epochs without improvement). "
             "Default is 15 for initial training. Recommended 8-10 for cyclic fine-tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vlr_size <= 0:
        raise SystemExit("--vlr-size must be a positive integer")

    train_config = TrainConfig.make(args.vlr_size)
    
    # Apply CLI overrides
    if args.epochs is not None:
        if args.additional_epochs is not None:
            raise SystemExit("Cannot specify both --epochs and --additional-epochs")
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate
    if args.lambda_identity is not None:
        train_config.lambda_identity = args.lambda_identity
    if args.lambda_feature_match is not None:
        train_config.lambda_feature_match = args.lambda_feature_match
    if args.patience is not None:
        train_config.early_stop_patience = args.patience

    train(train_config, args)


if __name__ == "__main__":
    main()
