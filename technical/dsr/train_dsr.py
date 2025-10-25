"""Training script for the DSR super-resolution network.

The script pairs the preprocessed very-low-resolution (VLR) images with their
high-resolution (HR) counterparts, applies modern data augmentation and
perceptual objectives, and optimises an enhanced DSR model that works well with
the EdgeFace recogniser.

Key upgrades compared to the previous baseline:

* Identity-preserving loss now uses the same EdgeFace backbone as the
  production pipeline and properly aligns checkpoint keys.
* Perceptual loss (VGG-19) + total variation regularisation improve texture
  fidelity without shrugging off facial structure.
* Exponential Moving Average (EMA) weights, mixed-precision training, and
  gradient clipping stabilise optimisation.
* Rich yet lightweight augmentations (flip/rotation/colour jitter) operate on
  both HR and VLR inputs with shared randomness to broaden generalisation.
* The saved checkpoint lands in ``technical/dsr/dsr.pth`` and matches the
  ``DSRColor`` architecture expected by the inference pipeline.
"""

from __future__ import annotations

import argparse
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

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
from ..facial_rec.edgeface_weights.edgeface import EdgeFace


# ---------------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Hyper-parameters controlling training with fine-tuned EdgeFace."""

    epochs: int = 100  # Increased for better convergence with stricter losses
    batch_size: int = 16
    learning_rate: float = 1.5e-4  # Slightly reduced for more stable fine-tuning
    weight_decay: float = 1e-6
    lambda_l1: float = 1.0
    lambda_perceptual: float = 0.02  # Reduced further - identity is priority
    lambda_identity: float = 0.50  # Increased from 0.35 - strongest signal
    lambda_feature_match: float = 0.15  # NEW: intermediate feature matching
    lambda_tv: float = 3e-6  # Reduced to allow sharper facial features
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    num_workers: int = 8
    val_interval: int = 1
    seed: int = 42
    warmup_epochs: int = 5  # Longer warmup for stability with higher identity loss
    early_stop_patience: int = 20  # More patience for slower convergence
    use_multiscale_perceptual: bool = True  # Multi-scale perceptual loss


# ---------------------------------------------------------------------------
# Dataset with paired augmentations
# ---------------------------------------------------------------------------


class PairedFaceSRDataset(Dataset):
    """Loads aligned VLR/HR facial image pairs with shared augmentations."""

    def __init__(self, root: Path, augment: bool) -> None:
        self.root = Path(root)
        self.vlr_root = self.root / "vlr_images"
        self.hr_root = self.root / "hr_images"
        if not self.vlr_root.is_dir() or not self.hr_root.is_dir():
            raise RuntimeError(
                "Dataset directory must contain 'vlr_images' and 'hr_images' subfolders"
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

        if self.augment:
            vlr, hr = self._apply_shared_augmentations(vlr, hr)

        vlr_tensor = TF.to_tensor(vlr)
        hr_tensor = TF.to_tensor(hr)
        return vlr_tensor, hr_tensor

    @staticmethod
    def _apply_shared_augmentations(
        vlr: Image.Image, hr: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        # Horizontal flip - safe for face recognition
        if random.random() < 0.5:
            vlr = TF.hflip(vlr)
            hr = TF.hflip(hr)

        # More conservative rotations to preserve facial features
        if random.random() < 0.6:  # Only apply 60% of the time
            angle = random.uniform(-5.0, 5.0)  # Reduced from ±8
            vlr = TF.rotate(vlr, angle, interpolation=InterpolationMode.BILINEAR)
            hr = TF.rotate(hr, angle, interpolation=InterpolationMode.BILINEAR)

        # Very mild colour jitter - too much breaks identity embeddings
        if random.random() < 0.25:  # Reduced probability
            brightness = random.uniform(0.95, 1.05)  # Tighter range
            contrast = random.uniform(0.95, 1.05)  # Tighter range
            saturation = random.uniform(0.97, 1.03)  # Much tighter range
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


class EdgeFaceEmbedding(nn.Module):
    """Wraps fine-tuned EdgeFace to provide embeddings + intermediate features for feature matching."""

    def __init__(
        self, weights_path: Path, device: torch.device, extract_features: bool = True
    ) -> None:
        super().__init__()
        self.model = EdgeFace(back="edgeface_s")
        self.extract_features = extract_features

        # Load checkpoint - handle both fine-tuned and original formats
        try:
            raw_state = torch.load(weights_path, map_location="cpu", weights_only=False)
        except Exception:
            # Fallback for weights-only load issues
            import sys
            from dataclasses import dataclass

            # Create stub for unpickling fine-tuned checkpoints
            main_module = sys.modules.get("__main__")
            if main_module and not hasattr(main_module, "FinetuneConfig"):

                @dataclass
                class FinetuneConfig:
                    """Stub for unpickling."""

                    pass

                setattr(main_module, "FinetuneConfig", FinetuneConfig)

            raw_state = torch.load(weights_path, map_location="cpu", weights_only=False)

        if isinstance(raw_state, torch.jit.ScriptModule):
            state_dict = raw_state.state_dict()
        elif isinstance(raw_state, dict):
            # Handle fine-tuned checkpoint format (has 'backbone_state_dict')
            if "backbone_state_dict" in raw_state:
                state_dict = raw_state["backbone_state_dict"]
                print("[EdgeFace] Loaded fine-tuned backbone weights")
            elif "state_dict" in raw_state:
                state_dict = raw_state["state_dict"]
            elif "model_state_dict" in raw_state:
                state_dict = raw_state["model_state_dict"]
            else:
                state_dict = raw_state
        else:
            raise TypeError(
                f"Unsupported EdgeFace checkpoint type: {type(raw_state).__name__}"
            )

        clean_state = {}
        for key, value in state_dict.items():
            clean_key = key
            if clean_key.startswith("model."):
                clean_key = clean_key[len("model.") :]
            clean_state[clean_key] = value

        missing, unexpected = self.model.load_state_dict(clean_state, strict=False)
        if missing:
            print(
                f"[EdgeFace] Warning - missing keys when loading identity model: {missing}"
            )
        if unexpected:
            print(
                f"[EdgeFace] Warning - unexpected keys when loading identity model: {unexpected}"
            )

        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # For feature matching: extract intermediate features from EdgeFace
        if self.extract_features:
            self._register_feature_hooks()
            self.intermediate_features_sr = []
            self.intermediate_features_hr = []

    def _register_feature_hooks(self) -> None:
        """Register forward hooks to capture intermediate features for feature matching."""

        def get_activation(name: str, storage: List):
            def hook(module, input, output):
                storage.append(output.detach())

            return hook

        # Hook into key layers of EdgeFace features network
        # EdgeFace architecture: stem -> stages -> flatten -> linear -> bn
        if hasattr(self.model, "features"):
            # Access intermediate blocks (adjust indices based on EdgeFace structure)
            try:
                # Hook early, mid, late feature layers for multi-scale matching
                self.model.features[0].register_forward_hook(
                    get_activation("early", self.intermediate_features_sr)
                )
                if len(self.model.features) > 5:
                    self.model.features[5].register_forward_hook(
                        get_activation("mid", self.intermediate_features_sr)
                    )
                if len(self.model.features) > 10:
                    self.model.features[10].register_forward_hook(
                        get_activation("late", self.intermediate_features_sr)
                    )
            except Exception:
                # If hooking fails, disable feature extraction
                self.extract_features = False
                print(
                    "[EdgeFace] Could not register feature hooks; disabling feature matching"
                )

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        processed = self.transform(imgs)

        # Clear previous features
        if self.extract_features:
            self.intermediate_features_sr.clear()
            self.intermediate_features_hr.clear()

        embeddings = self.model(processed)
        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def extract_paired_features(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
        """Extract embeddings and intermediate features for both SR and HR."""
        self.intermediate_features_sr.clear()
        self.intermediate_features_hr.clear()

        sr_processed = self.transform(sr)
        hr_processed = self.transform(hr)

        emb_sr = self.model(sr_processed)
        sr_features = list(self.intermediate_features_sr)

        self.intermediate_features_sr.clear()
        emb_hr = self.model(hr_processed)
        hr_features = list(self.intermediate_features_hr)

        return (
            F.normalize(emb_sr, dim=-1),
            F.normalize(emb_hr, dim=-1),
            sr_features,
            hr_features,
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dsr_model(device: torch.device) -> nn.Module:
    # Increased capacity for better feature learning
    config = DSRConfig(base_channels=112, residual_blocks=16)
    model = DSRColor(config=config).to(device)
    return model


def train(config: TrainConfig, args: argparse.Namespace) -> None:
    base_dir = Path(__file__).resolve().parent
    dataset_root = base_dir.parent / "dataset"
    train_dir = dataset_root / "train_processed"
    val_dir = dataset_root / "val_processed"
    weights_path = base_dir.parent / "facial_rec" / "edgeface_weights" / args.edgeface
    save_path = base_dir / "dsr.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    set_random_seed(config.seed)

    print(f"Using device: {device}")
    print(f"Loading fine-tuned EdgeFace from: {weights_path}")

    train_ds = PairedFaceSRDataset(train_dir, augment=True)
    val_ds = PairedFaceSRDataset(val_dir, augment=False)
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

    model = build_dsr_model(device)
    ema = ModelEMA(model, decay=config.ema_decay)

    identity_model = EdgeFaceEmbedding(weights_path, device, extract_features=True)
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

    print(f"Training samples: {len(train_ds)}, validation samples: {len(val_ds)}")
    print("Saving best checkpoints to", save_path)
    print(f"\nTraining configuration:")
    print(f"  - Identity loss weight: {config.lambda_identity}")
    print(f"  - Feature matching weight: {config.lambda_feature_match}")
    print(f"  - Perceptual loss weight: {config.lambda_perceptual}")
    print(f"  - Multi-scale perceptual: {config.use_multiscale_perceptual}")
    print(f"  - Epochs: {config.epochs}, Warmup: {config.warmup_epochs}")
    print(f"  - Early stop patience: {config.early_stop_patience}\n")

    for epoch in range(config.epochs):
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
                embeddings_sr = identity_model(sr)
                embeddings_hr = identity_model(hr)
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
                    embeddings_sr = identity_model(sr)
                    embeddings_hr = identity_model(hr)
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

            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                epochs_without_improvement = 0
                checkpoint = {
                    "epoch": epoch + 1,
                    "config": asdict(config),
                    "model_state_dict": ema.ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_identity_loss": val_losses["identity"],
                }
                torch.save(checkpoint, save_path)
                print(
                    f"✅ Saved new best checkpoint (val PSNR {val_psnr:.2f}dB, ID loss {val_losses['identity']:.4f})"
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stop_patience:
                    print(
                        f"\n⚠️  Early stopping after {epoch + 1} epochs (no improvement for {config.early_stop_patience} epochs)"
                    )
                    print(f"Best validation PSNR: {best_val_psnr:.2f}dB")
                    break
        else:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"train_loss={avg_train_loss:.4f} (L1:{avg_losses['l1']:.3f} P:{avg_losses['perceptual']:.3f} "
                f"ID:{avg_losses['identity']:.3f} FM:{avg_losses['feature_match']:.3f}) "
                f"PSNR={avg_train_psnr:.2f}dB"
            )

    print(f"\n🎉 Training complete! Best validation PSNR: {best_val_psnr:.2f}dB")
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
        default="edgeface_finetuned.pth",
        help="Filename of the fine-tuned EdgeFace weights relative to technical/facial_rec/edgeface_weights",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = TrainConfig()
    if args.epochs is not None:
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size

    train(train_config, args)


if __name__ == "__main__":
    main()
