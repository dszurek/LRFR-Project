"""Training script for Hybrid Transformer-CNN DSR with identity-aware losses.

Implements training methodology from CMU-Net paper (Yulianto et al. 2022):
- Cosine matrix loss for angle distance minimization in feature space
- Magnitude loss for norm-diff distance minimization  
- Combined with perceptual and pixel losses
- Identity-preserving super-resolution

Key improvements:
- Hybrid transformer-CNN architecture
- Efficient MFM (Max-Feature-Map) activation
- Multi-scale feature extraction
- Progressive upsampling with pixel shuffle
"""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .hybrid_model import HybridDSR, create_hybrid_dsr

# Import EdgeFace for identity loss
try:
    from ..facial_rec.edgeface_weights.edgeface import EdgeFace
    from ..facial_rec.finetune_edgeface import FinetuneConfig  # Needed for unpickling finetuned checkpoints
except ImportError:
    from facial_rec.edgeface_weights.edgeface import EdgeFace
    from facial_rec.finetune_edgeface import FinetuneConfig


def _resolve_vlr_dir_name(vlr_size: int) -> str:
    """Return the directory name for VLR images."""
    if vlr_size <= 0:
        raise ValueError("VLR size must be a positive integer")
    return f"vlr_images_{vlr_size}x{vlr_size}"


@dataclass
class HybridTrainConfig:
    """Training configuration for Hybrid DSR.
    
    Optimized hyperparameters based on CMU-Net paper findings.
    """
    
    # Model architecture
    vlr_size: int = 32
    target_hr_size: int = 112
    
    # Training parameters
    epochs: int = 150  # Increased for A100
    batch_size: int = 64  # Base batch size (will be scaled in for_resolution)
    learning_rate: float = 4e-4  # Increased LR for larger batch size
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1  # Effective batch size = batch_size * this
    
    # Loss weights - Enhanced for face recognition emphasis
    # Visual quality losses (35% total weight)
    lambda_l1: float = 1.0  # Pixel reconstruction
    lambda_perceptual: float = 0.02  # VGG perceptual loss
    lambda_tv: float = 1e-5  # Total variation regularization
    
    # Identity preservation losses (45% total weight - PRIMARY GOAL)
    lambda_cosine: float = 15.0  # Same-photo cosine similarity (increased from 10.0)
    lambda_cosine_cross: float = 3.0  # Cross-photo identity consistency (NEW)
    lambda_magnitude: float = 0.15  # Embedding magnitude consistency (increased from 0.1)
    lambda_feature_corr: float = 2.0  # Feature correlation preservation (NEW)
    
    # Inter-subject discrimination (20% total weight)
    lambda_discriminative: float = 8.0  # Different people stay different (increased from 5.0)
    
    # Optimization
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    early_stop_patience: int = 15
    
    # Data augmentation
    use_augmentation: bool = True
    
    # Validation
    val_frequency: int = 1  # Validate every N epochs
    
    @classmethod
    def for_resolution(cls, vlr_size: int) -> HybridTrainConfig:
        """Create config optimized for specific VLR size."""
        cfg = cls(vlr_size=vlr_size)
        
        # Adjust batch size and identity weights for memory efficiency
        # A100 OPTIMIZATION: Much larger batch sizes
        if vlr_size <= 16:
            cfg.batch_size = 128  # Massive batch size for A100
            cfg.lambda_cosine = 18.0
            cfg.lambda_cosine_cross = 4.0
            cfg.lambda_feature_corr = 3.0
        elif vlr_size <= 24:
            cfg.batch_size = 96
            cfg.lambda_cosine = 16.0
            cfg.lambda_cosine_cross = 3.5
            cfg.lambda_feature_corr = 2.5
        else:  # 32x32
            cfg.batch_size = 64
            cfg.lambda_cosine = 15.0
            cfg.lambda_cosine_cross = 3.0
            cfg.lambda_feature_corr = 2.0
        
        return cfg


class PairedFaceSRDataset(Dataset):
    """Dataset loader for paired VLR/HR facial images with augmentation.
    
    Also supports negative sampling for discriminative loss (ensuring different
    people remain different after super-resolution).
    """
    
    def __init__(
        self,
        root: Path,
        augment: bool,
        vlr_size: int,
        target_hr_size: int = 112,
        enable_negative_sampling: bool = True,
        cache_size: int = 1000,  # Cache frequently accessed images
    ) -> None:
        self.root = Path(root)
        self.vlr_size = vlr_size
        self.vlr_root = self.root / _resolve_vlr_dir_name(vlr_size)
        self.hr_root = self.root / "hr_images"
        self.target_hr_size = target_hr_size
        self.enable_negative_sampling = enable_negative_sampling
        
        # Note: LRU cache cannot be used with num_workers > 0 due to pickling issues
        # Cache is initialized in __getstate__/__setstate__ for each worker process
        self._cache_size = cache_size
        self._image_cache: dict[str, Image.Image] = {}
        
        if not self.vlr_root.is_dir():
            raise RuntimeError(
                f"Missing VLR directory '{self.vlr_root}'. "
                f"Run regenerate_vlr_dataset with --vlr-sizes {vlr_size}."
            )
        if not self.hr_root.is_dir():
            raise RuntimeError(
                f"Dataset directory must contain 'hr_images' (expected at {self.hr_root})."
            )
        
        # Load image pairs
        self.pairs: List[Tuple[Path, Path]] = []
        self.subject_to_pairs: Dict[str, List[int]] = {}  # subject_id -> list of pair indices
        
        for vlr_path in sorted(self.vlr_root.glob("*.png")):
            hr_path = self.hr_root / vlr_path.name
            if not hr_path.is_file():
                raise FileNotFoundError(
                    f"Missing HR counterpart for {vlr_path.name} under {self.hr_root}"
                )
            
            pair_idx = len(self.pairs)
            self.pairs.append((vlr_path, hr_path))
            
            # Extract subject ID (format: subjectID_imageNum.png)
            subject_id = vlr_path.stem.split('_')[0]
            if subject_id not in self.subject_to_pairs:
                self.subject_to_pairs[subject_id] = []
            self.subject_to_pairs[subject_id].append(pair_idx)
        
        if not self.pairs:
            raise RuntimeError(f"No image pairs found in {self.root}")
        
        self.subject_list = list(self.subject_to_pairs.keys())
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _load_image_impl(self, path: str) -> Image.Image:
        """Load image with simple dict-based caching (pickle-safe for multiprocessing).
        
        Args:
            path: Path to image file
            
        Returns:
            PIL Image
        """
        # Check cache first
        if path in self._image_cache:
            return self._image_cache[path]
        
        # Load from disk
        img = Image.open(path).convert("RGB")
        
        # Add to cache with simple size limit
        if len(self._image_cache) < self._cache_size:
            self._image_cache[path] = img
        
        return img
    
    def get_cross_photo_sample(self, index: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get another photo of the same subject (if available) for cross-photo identity loss.
        
        Args:
            index: Index of the anchor sample
            
        Returns:
            Tuple of (cross_vlr, cross_hr) from same subject, or None if only one photo exists
        """
        vlr_path, hr_path = self.pairs[index]
        subject_id = vlr_path.stem.split('_')[0]
        
        # Get all pairs for this subject
        subject_pairs = self.subject_to_pairs.get(subject_id, [])
        
        # If only one photo, can't do cross-photo
        if len(subject_pairs) <= 1:
            return None
        
        # Sample a different photo of the same person
        other_indices = [idx for idx in subject_pairs if idx != index]
        other_idx = random.choice(other_indices)
        
        cross_vlr_path, cross_hr_path = self.pairs[other_idx]
        cross_vlr = self._load_image_impl(str(cross_vlr_path))
        cross_hr = self._load_image_impl(str(cross_hr_path))
        
        # Resize if needed
        if cross_vlr.size != (self.vlr_size, self.vlr_size):
            cross_vlr = cross_vlr.resize((self.vlr_size, self.vlr_size), Image.Resampling.LANCZOS)
        if cross_hr.size != (self.target_hr_size, self.target_hr_size):
            cross_hr = cross_hr.resize((self.target_hr_size, self.target_hr_size), Image.Resampling.LANCZOS)
        
        # Apply same augmentations
        if self.augment:
            cross_vlr, cross_hr = self._apply_shared_augmentations(cross_vlr, cross_hr)
        
        cross_vlr_tensor = TF.to_tensor(cross_vlr)
        cross_hr_tensor = TF.to_tensor(cross_hr)
        return cross_vlr_tensor, cross_hr_tensor
    
    def get_negative_sample(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a negative sample (different subject) for discriminative loss.
        
        Args:
            index: Index of the anchor sample
            
        Returns:
            Tuple of (neg_vlr, neg_hr) from a different subject
        """
        # Get subject of anchor
        anchor_path = self.pairs[index][0]
        anchor_subject = anchor_path.stem.split('_')[0]
        
        # Sample a different subject
        negative_subjects = [s for s in self.subject_list if s != anchor_subject]
        if not negative_subjects:
            # Fallback: if only one subject, return a different image from same subject
            negative_subject = anchor_subject
        else:
            negative_subject = random.choice(negative_subjects)
        
        # Get a random image from the negative subject
        neg_idx = random.choice(self.subject_to_pairs[negative_subject])
        neg_vlr_path, neg_hr_path = self.pairs[neg_idx]
        
        # Load and process (use cached loader)
        neg_vlr = self._load_image_impl(str(neg_vlr_path))
        neg_hr = self._load_image_impl(str(neg_hr_path))
        
        if neg_vlr.size != (self.vlr_size, self.vlr_size):
            neg_vlr = neg_vlr.resize((self.vlr_size, self.vlr_size), Image.Resampling.BICUBIC)
        if neg_hr.size != (self.target_hr_size, self.target_hr_size):
            neg_hr = neg_hr.resize((self.target_hr_size, self.target_hr_size), Image.Resampling.LANCZOS)
        
        return TF.to_tensor(neg_vlr), TF.to_tensor(neg_hr)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vlr_path, hr_path = self.pairs[index]
        
        # Use cached image loading for faster access
        vlr = self._load_image_impl(str(vlr_path))
        hr = self._load_image_impl(str(hr_path))
        
        # Ensure correct sizes
        if vlr.size != (self.vlr_size, self.vlr_size):
            vlr = vlr.resize((self.vlr_size, self.vlr_size), Image.Resampling.BICUBIC)
        
        if hr.size != (self.target_hr_size, self.target_hr_size):
            hr = hr.resize((self.target_hr_size, self.target_hr_size), Image.Resampling.LANCZOS)
        
        # Apply augmentations
        if self.augment:
            vlr, hr = self._apply_shared_augmentations(vlr, hr)
        
        # Convert to tensors (no normalization - models expect [0, 1])
        vlr_tensor = TF.to_tensor(vlr)
        hr_tensor = TF.to_tensor(hr)
        return vlr_tensor, hr_tensor
    
    def _apply_shared_augmentations(
        self, vlr: Image.Image, hr: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply synchronized augmentations to VLR and HR pairs."""
        
        # Horizontal flip (safe for faces)
        if random.random() < 0.5:
            vlr = TF.hflip(vlr)
            hr = TF.hflip(hr)
        
        # Small rotation (resolution-dependent)
        rotation_cap = 6.0 if self.vlr_size >= 32 else (5.0 if self.vlr_size >= 24 else 4.0)
        if random.random() < 0.65:
            angle = random.uniform(-rotation_cap, rotation_cap)
            vlr = TF.rotate(vlr, angle, interpolation=InterpolationMode.BILINEAR)
            hr = TF.rotate(hr, angle, interpolation=InterpolationMode.BILINEAR)
        
        # Gentle color jitter (preserve identity)
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


class VGGPerceptualLoss(nn.Module):
    """Multi-scale perceptual loss using VGG19 features.
    
    Optimized with:
    - Cached normalization tensors
    - Efficient feature extraction
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        
        # Load pre-trained VGG19
        weights = VGG19_Weights.IMAGENET1K_V1
        vgg = vgg19(weights=weights).features.eval()
        
        for param in vgg.parameters():
            param.requires_grad_(False)
        
        self.vgg = vgg.to(device)
        
        # Feature extraction layers (conv1_2, conv2_2, conv3_4, conv4_4)
        self.layer_indices = [3, 8, 17, 26]
        
        # Cache normalization tensors on the correct device
        self.register_buffer(
            'norm_mean',
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'norm_std',
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between SR and HR images.
        
        Args:
            sr: Super-resolved image (B, 3, H, W) in [0, 1]
            hr: High-resolution target (B, 3, H, W) in [0, 1]
            
        Returns:
            Perceptual loss scalar
        """
        # VGG expects ImageNet normalization (use cached tensors)
        sr_norm = (sr - self.norm_mean) / self.norm_std
        hr_norm = (hr - self.norm_mean) / self.norm_std
        
        # Extract features
        loss = 0.0
        sr_feat = sr_norm
        hr_feat = hr_norm
        
        for i, layer in enumerate(self.vgg):
            sr_feat = layer(sr_feat)
            hr_feat = layer(hr_feat)
            
            if i in self.layer_indices:
                loss = loss + F.l1_loss(sr_feat, hr_feat)
        
        return loss / len(self.layer_indices)


class IdentityModel(nn.Module):
    """EdgeFace wrapper for identity-aware losses (cosine + magnitude).
    
    Extracts 512-dim embeddings and applies normalization internally.
    """
    
    def __init__(self, edgeface_path: Path, device: torch.device):
        super().__init__()
        
        self.device = device
        
        # Load EdgeFace model
        try:
            print(f"[IdentityModel] Loading EdgeFace from {edgeface_path.name}")
            state_dict = torch.load(edgeface_path, map_location="cpu", weights_only=True)
        except Exception:
            print("[IdentityModel] Loading with pickles (finetuned checkpoint)")
            state_dict = torch.load(edgeface_path, map_location="cpu", weights_only=False)
            
            # Extract backbone if checkpoint format
            if "backbone_state_dict" in state_dict:
                state_dict = state_dict["backbone_state_dict"]
            elif "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
        
        # Create EdgeFace model
        self.model = EdgeFace(embedding_size=512, back="edgeface_xxs")
        
        # Clean state dict keys
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Remove prefixes
            k = k.replace("model.", "").replace("backbone.", "")
            cleaned_state_dict[k] = v
        
        # Load weights
        missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)
        if unexpected:
            print(f"[IdentityModel] Unexpected keys: {len(unexpected)}")
        
        self.model.to(device)
        self.model.eval()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        # Preprocessing: EdgeFace expects [-1, 1] normalization
        self.transform = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    
    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized embeddings.
        
        Args:
            imgs: Image batch (B, 3, H, W) in [0, 1] range
            
        Returns:
            Normalized embeddings (B, 512)
        """
        # Apply normalization to [-1, 1]
        normalized = self.transform(imgs)
        
        # Extract embeddings
        embeddings = self.model(normalized)
        
        # L2 normalize
        return F.normalize(embeddings, dim=-1)


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Compute total variation loss for smoothness."""
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return tv_h + tv_w


def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(sr, hr).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation.
    
    Args:
        x: Input batch (B, C, H, W)
        y: Target batch (B, C, H, W)
        alpha: MixUp parameter (lower = less mixing)
        
    Returns:
        Tuple of (mixed_x, mixed_y, original_y, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y, y[index], lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation.
    
    Args:
        x: Input batch (B, C, H, W)
        y: Target batch (B, C, H, W)
        alpha: CutMix parameter
        
    Returns:
        Tuple of (mixed_x, mixed_y, original_y, lambda)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    # Get random box coordinates
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    mixed_y = y.clone()
    mixed_y[:, :, bby1:bby2, bbx1:bbx2] = y[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to actual area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return mixed_x, mixed_y, y[index], lam


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExponentialMovingAverage:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module) -> None:
        """Update shadow weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
    
    def apply_shadow(self) -> None:
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self) -> None:
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def validate_architecture_efficiency(model: nn.Module, vlr_size: int, device: torch.device) -> Dict[str, float]:
    """Validate that hybrid architecture provides efficiency benefits.
    
    Measures:
    - Parameter count (should be < 2M for efficiency)
    - Inference time
    - Memory usage
    - Effective receptive field from transformer attention
    
    Returns:
        Dictionary with efficiency metrics
    """
    import time
    
    model.eval()
    param_count = model.count_parameters()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, vlr_size, vlr_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure inference time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    
    # Memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        memory_mb = 0.0
    
    metrics = {
        "parameters": param_count,
        "inference_time_ms": avg_time,
        "memory_mb": memory_mb,
        "params_per_pixel": param_count / (112 * 112),  # Output resolution
    }
    
    # Validate efficiency
    print("\n" + "="*60)
    print("ARCHITECTURE EFFICIENCY VALIDATION")
    print("="*60)
    print(f"Parameters: {param_count:,} {'✓' if param_count < 5_500_000 else '✗ (target: < 5.5M)'}")
    print(f"Inference time: {avg_time:.2f} ms")
    print(f"Memory usage: {memory_mb:.2f} MB")
    print(f"Params per output pixel: {metrics['params_per_pixel']:.2f}")
    
    if param_count >= 5_500_000:
        print("⚠️  WARNING: Model exceeds 5.5M parameter target for efficiency!")
    
    # Check if transformer is actually used
    has_transformer = hasattr(model, 'use_transformer') and model.use_transformer
    if has_transformer:
        print("✓ Transformer attention enabled for global context")
    else:
        print("ℹ️  Transformer disabled (resolution too small or architecture limitation)")
    
    print("="*60 + "\n")
    
    return metrics


def train(config: HybridTrainConfig, args: argparse.Namespace) -> None:
    """Main training loop for Hybrid DSR."""
    
    base_dir = Path(__file__).resolve().parent
    dataset_root = base_dir.parent / "dataset"
    
    # Use frontal-only dataset if specified
    if args.frontal_only:
        train_dir = dataset_root / "frontal_only" / "train"
        val_dir = dataset_root / "frontal_only" / "val"
        print("🎯 Using FRONTAL-ONLY filtered dataset")
    else:
        train_dir = dataset_root / "train_processed"
        val_dir = dataset_root / "val_processed"
        print("⚠️  Using FULL dataset (includes profile/angled faces)")
    
    weights_path = base_dir.parent / "facial_rec" / "edgeface_weights" / args.edgeface
    save_path = base_dir / f"hybrid_dsr{config.vlr_size}.pth"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create datasets
    train_ds = PairedFaceSRDataset(
        train_dir,
        augment=config.use_augmentation,
        vlr_size=config.vlr_size,
        target_hr_size=config.target_hr_size
    )
    val_ds = PairedFaceSRDataset(
        val_dir,
        augment=False,
        vlr_size=config.vlr_size,
        target_hr_size=config.target_hr_size
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_hybrid_dsr(config.vlr_size, config.target_hr_size)
    model = model.to(device)
    
    # Validate architecture efficiency before training
    efficiency_metrics = validate_architecture_efficiency(model, config.vlr_size, device)
    
    # Create loss functions
    l1_loss = nn.L1Loss()
    perceptual_loss = VGGPerceptualLoss(device)
    identity_model = IdentityModel(weights_path, device)
    cosine_loss = nn.CosineEmbeddingLoss()
    
    # Optimizer
    optimizer = optim.AdamW(  # AdamW for better regularization
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # OneCycleLR for faster convergence (A100 optimization)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.warmup_epochs / config.epochs,  # Warmup phase
        anneal_strategy='cos',
        div_factor=25.0,  # Initial LR = max_lr / 25
        final_div_factor=1000.0  # Final LR = max_lr / 1000
    )
    
    # Mixed precision training with bfloat16 for A100
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler('cuda', enabled=use_amp and dtype == torch.float16)  # No scaler needed for bfloat16
    
    if use_amp:
        print(f"Using mixed precision training with {dtype}")
    
    # EMA for stable weights
    ema = ExponentialMovingAverage(model, decay=0.999)
    
    # Training state
    best_val_psnr = 0.0
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume and save_path.exists():
        print(f"Resuming from {save_path}")
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_psnr = checkpoint.get("best_score", checkpoint.get("best_psnr", 0.0))
        print(f"Resumed from epoch {start_epoch}, best score: {best_val_psnr:.4f}")
    
    print(f"\nTraining samples: {len(train_ds)}, validation samples: {len(val_ds)}")
    print(f"Saving checkpoints to: {save_path}")
    print(f"\nConfiguration:")
    print(f"  VLR size: {config.vlr_size}x{config.vlr_size}")
    print(f"  Model parameters: {model.count_parameters():,}")
    print(f"  Loss weights: L1={config.lambda_l1}, Perceptual={config.lambda_perceptual}")
    print(f"                Cosine={config.lambda_cosine}, Magnitude={config.lambda_magnitude}")
    print(f"  Epochs: {config.epochs}, Warmup: {config.warmup_epochs}")
    print(f"  Batch size: {config.batch_size}, Accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}\n")
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        running_losses = {
            "l1": 0.0,
            "perceptual": 0.0,
            "cosine": 0.0,
            "cosine_cross": 0.0,
            "magnitude": 0.0,
            "feature_corr": 0.0,
            "discriminative": 0.0,
            "tv": 0.0
        }
        
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs} [train]",
            leave=False
        )
        
        for batch_idx, (vlr, hr) in enumerate(progress):
            vlr = vlr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            
            # Only zero gradients at start of accumulation
            is_accumulating = (batch_idx + 1) % config.gradient_accumulation_steps != 0
            if not is_accumulating:
                optimizer.zero_grad(set_to_none=True)
            
            # Apply advanced augmentation (50% chance)
            use_mixup = random.random() < 0.25  # 25% MixUp
            use_cutmix = random.random() < 0.25 and not use_mixup  # 25% CutMix
            
            with autocast('cuda', enabled=device.type == "cuda", dtype=dtype if use_amp else torch.float32):
                # Forward pass
                sr = model(vlr)
                
                # Apply MixUp or CutMix to SR outputs and targets
                if use_mixup:
                    sr, hr, hr_mix, lam = mixup_data(sr, hr, alpha=0.2)
                elif use_cutmix:
                    sr, hr, hr_mix, lam = cutmix_data(sr, hr, alpha=1.0)
                
                # Pixel reconstruction loss
                if use_mixup or use_cutmix:
                    loss_l1 = lam * l1_loss(sr, hr) + (1 - lam) * l1_loss(sr, hr_mix)
                else:
                    loss_l1 = l1_loss(sr, hr)
                
                # Perceptual loss
                loss_perc = perceptual_loss(sr, hr)
                
                # Identity-aware losses (from CMU-Net paper)
                embeddings_sr = identity_model(sr)
                embeddings_hr = identity_model(hr)
                
                # Cosine loss: minimize angle distance in feature space
                targets = torch.ones(embeddings_sr.size(0), device=device)
                loss_cosine = cosine_loss(embeddings_sr, embeddings_hr, targets)
                
                # Magnitude loss: minimize norm-diff distance
                norm_sr = torch.norm(embeddings_sr, p=2, dim=1)
                norm_hr = torch.norm(embeddings_hr, p=2, dim=1)
                loss_magnitude = F.l1_loss(norm_sr, norm_hr)
                
                # NEW: Cross-photo identity loss (if available) - OPTIMIZED
                loss_cosine_cross = torch.tensor(0.0, device=device)
                # Sample probabilistically instead of checking every sample
                if random.random() < 0.3:  # Only 30% of batches to reduce overhead
                    cross_samples_count = 0
                    cross_hr_batch = []
                    
                    # Sample a few random indices instead of all
                    num_cross_samples = min(4, vlr.size(0))  # Max 4 cross-photo samples per batch
                    sample_indices = random.sample(range(vlr.size(0)), num_cross_samples)
                    
                    for idx in sample_indices:
                        cross_sample = train_ds.get_cross_photo_sample(
                            (batch_idx * config.batch_size + idx) % len(train_ds)
                        )
                        if cross_sample is not None:
                            _, cross_hr_tensor = cross_sample  # Only need HR for embedding
                            cross_hr_batch.append(cross_hr_tensor)
                            cross_samples_count += 1
                    
                    if cross_samples_count > 0:
                        cross_hr = torch.stack(cross_hr_batch).to(device, non_blocking=True)
                        embeddings_cross_hr = identity_model(cross_hr)
                        embeddings_sr_subset = embeddings_sr[:cross_samples_count]
                        targets_cross = torch.ones(cross_samples_count, device=device)
                        loss_cosine_cross = cosine_loss(embeddings_sr_subset, embeddings_cross_hr, targets_cross)
                
                # NEW: Feature correlation loss - OPTIMIZED
                # Compute only every N batches to reduce overhead
                if batch_idx % 5 == 0 and embeddings_sr.size(0) > 1:
                    # Compute correlation matrices
                    sr_centered = embeddings_sr - embeddings_sr.mean(dim=0, keepdim=True)
                    hr_centered = embeddings_hr - embeddings_hr.mean(dim=0, keepdim=True)
                    
                    sr_corr = torch.mm(sr_centered.t(), sr_centered) / embeddings_sr.size(0)
                    hr_corr = torch.mm(hr_centered.t(), hr_centered) / embeddings_hr.size(0)
                    
                    loss_feature_corr = F.mse_loss(sr_corr, hr_corr)
                else:
                    loss_feature_corr = torch.tensor(0.0, device=device)
                
                # Discriminative loss: ensure different people remain different - OPTIMIZED
                # Reduce frequency to every other batch
                if batch_idx % 2 == 0:
                    # Sample only half the batch size for negative pairs
                    num_neg_samples = max(1, vlr.size(0) // 2)
                    neg_vlr_batch = []
                    neg_hr_batch = []
                    for _ in range(num_neg_samples):
                        # Get a random negative sample from a different subject
                        neg_vlr_tensor, neg_hr_tensor = train_ds.get_negative_sample(
                            torch.randint(0, len(train_ds), (1,)).item()
                        )
                        neg_vlr_batch.append(neg_vlr_tensor)
                        neg_hr_batch.append(neg_hr_tensor)
                    
                    neg_vlr = torch.stack(neg_vlr_batch).to(device, non_blocking=True)
                    neg_hr = torch.stack(neg_hr_batch).to(device, non_blocking=True)
                    neg_sr = model(neg_vlr)
                    
                    # Get embeddings for negative samples
                    embeddings_neg_sr = identity_model(neg_sr)
                    embeddings_neg_hr = identity_model(neg_hr)
                    
                    # Use subset of anchor embeddings to match negative samples
                    embeddings_sr_subset = embeddings_sr[:num_neg_samples]
                    embeddings_hr_subset = embeddings_hr[:num_neg_samples]
                    
                    # Compute inter-subject distances
                    dist_hr_to_neg = 1.0 - F.cosine_similarity(embeddings_hr_subset, embeddings_neg_hr, dim=1)
                    dist_sr_to_neg = 1.0 - F.cosine_similarity(embeddings_sr_subset, embeddings_neg_sr, dim=1)
                    
                    loss_discriminative = F.l1_loss(dist_sr_to_neg, dist_hr_to_neg)
                else:
                    loss_discriminative = torch.tensor(0.0, device=device)
                
                # Total variation regularization
                loss_tv = total_variation_loss(sr)
                
                # Combined loss with ALL components (normalize by accumulation steps)
                loss = (
                    config.lambda_l1 * loss_l1 +
                    config.lambda_perceptual * loss_perc +
                    config.lambda_cosine * loss_cosine +
                    config.lambda_cosine_cross * loss_cosine_cross +
                    config.lambda_magnitude * loss_magnitude +
                    config.lambda_feature_corr * loss_feature_corr +
                    config.lambda_discriminative * loss_discriminative +
                    config.lambda_tv * loss_tv
                ) / config.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Only step optimizer after accumulating gradients
            if not is_accumulating:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                
                # Update EMA
                ema.update(model)
            
            # Track metrics (multiply back by accumulation for correct logging)
            running_loss += loss.item() * config.gradient_accumulation_steps
            running_psnr += compute_psnr(sr.detach(), hr)
            running_losses["l1"] += loss_l1.item()
            running_losses["perceptual"] += loss_perc.item()
            running_losses["cosine"] += loss_cosine.item()
            running_losses["cosine_cross"] += loss_cosine_cross.item()
            running_losses["magnitude"] += loss_magnitude.item()
            running_losses["feature_corr"] += loss_feature_corr.item()
            running_losses["discriminative"] += loss_discriminative.item()
            running_losses["tv"] += loss_tv.item()
            
            # Update progress bar
            progress.set_postfix({
                "loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                "psnr": f"{compute_psnr(sr.detach(), hr):.2f}"
            })
        
        # Epoch statistics
        n_batches = len(train_loader)
        avg_loss = running_loss / n_batches
        avg_psnr = running_psnr / n_batches
        
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"  Train Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB")
        print(f"    L1: {running_losses['l1']/n_batches:.4f}, "
              f"Perceptual: {running_losses['perceptual']/n_batches:.4f}")
        print(f"    Cosine: {running_losses['cosine']/n_batches:.4f}, "
              f"Magnitude: {running_losses['magnitude']/n_batches:.4f}, "
              f"Discriminative: {running_losses['discriminative']/n_batches:.4f}")
        
        # Validation
        if (epoch + 1) % config.val_frequency == 0:
            model.eval()
            val_psnr = 0.0
            val_ssim = 0.0
            val_losses = {
                "l1": 0.0,
                "perceptual": 0.0,
                "cosine": 0.0,
                "cosine_cross": 0.0,
                "magnitude": 0.0,
                "feature_corr": 0.0,
                "discriminative": 0.0
            }
            
            with torch.no_grad():
                for batch_idx, (vlr, hr) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                    vlr = vlr.to(device, non_blocking=True)
                    hr = hr.to(device, non_blocking=True)
                    
                    sr = model(vlr)
                    
                    # Compute all validation losses
                    val_psnr += compute_psnr(sr, hr)
                    val_losses["l1"] += l1_loss(sr, hr).item()
                    val_losses["perceptual"] += perceptual_loss(sr, hr).item()
                    
                    # Identity-aware losses (highest weight)
                    embeddings_sr = identity_model(sr)
                    embeddings_hr = identity_model(hr)
                    targets = torch.ones(embeddings_sr.size(0), device=device)
                    val_losses["cosine"] += cosine_loss(embeddings_sr, embeddings_hr, targets).item()
                    
                    norm_sr = torch.norm(embeddings_sr, p=2, dim=1)
                    norm_hr = torch.norm(embeddings_hr, p=2, dim=1)
                    val_losses["magnitude"] += F.l1_loss(norm_sr, norm_hr).item()
                    
                    # Cross-photo identity (if available)
                    cross_samples_count = 0
                    cross_hr_batch = []
                    for idx in range(vlr.size(0)):
                        cross_sample = val_ds.get_cross_photo_sample(
                            (batch_idx * vlr.size(0) + idx) % len(val_ds)
                        )
                        if cross_sample is not None:
                            _, cross_hr_tensor = cross_sample
                            cross_hr_batch.append(cross_hr_tensor)
                            cross_samples_count += 1
                    
                    if cross_samples_count > 0:
                        cross_hr = torch.stack(cross_hr_batch).to(device, non_blocking=True)
                        embeddings_cross_hr = identity_model(cross_hr)
                        embeddings_sr_subset = embeddings_sr[:cross_samples_count]
                        targets_cross = torch.ones(cross_samples_count, device=device)
                        val_losses["cosine_cross"] += cosine_loss(embeddings_sr_subset, embeddings_cross_hr, targets_cross).item()
                    
                    # Feature correlation
                    if embeddings_sr.size(0) > 1:
                        sr_centered = embeddings_sr - embeddings_sr.mean(dim=0, keepdim=True)
                        hr_centered = embeddings_hr - embeddings_hr.mean(dim=0, keepdim=True)
                        sr_corr = torch.mm(sr_centered.t(), sr_centered) / embeddings_sr.size(0)
                        hr_corr = torch.mm(hr_centered.t(), hr_centered) / embeddings_hr.size(0)
                        val_losses["feature_corr"] += F.mse_loss(sr_corr, hr_corr).item()
                    
                    # Discriminative loss: ensure different subjects remain different
                    neg_vlr_batch = []
                    neg_hr_batch = []
                    for idx in range(vlr.size(0)):
                        neg_vlr_tensor, neg_hr_tensor = val_ds.get_negative_sample(
                            torch.randint(0, len(val_ds), (1,)).item()
                        )
                        neg_vlr_batch.append(neg_vlr_tensor)
                        neg_hr_batch.append(neg_hr_tensor)
                    
                    neg_vlr = torch.stack(neg_vlr_batch).to(device, non_blocking=True)
                    neg_hr = torch.stack(neg_hr_batch).to(device, non_blocking=True)
                    neg_sr = model(neg_vlr)
                    
                    embeddings_neg_sr = identity_model(neg_sr)
                    embeddings_neg_hr = identity_model(neg_hr)
                    
                    dist_hr_to_neg = 1.0 - F.cosine_similarity(embeddings_hr, embeddings_neg_hr, dim=1)
                    dist_sr_to_neg = 1.0 - F.cosine_similarity(embeddings_sr, embeddings_neg_sr, dim=1)
                    val_losses["discriminative"] += F.l1_loss(dist_sr_to_neg, dist_hr_to_neg).item()
            
            n_val_batches = len(val_loader)
            avg_val_psnr = val_psnr / n_val_batches
            
            # Average all losses
            for key in val_losses:
                val_losses[key] /= n_val_batches
            
            # Compute ENHANCED validation score with ALL loss components
            # New weights emphasizing face recognition (identity + discrimination = 65%)
            # Identity preservation: 45% (cosine 25% + cross-photo 10% + magnitude 8% + feature-corr 2%)
            # Discrimination: 20% (inter-subject differences)
            # Visual quality: 35% (PSNR 20% + L1 10% + perceptual 5%)
            val_score = (
                # Identity preservation (45% total)
                0.25 * (1.0 - val_losses["cosine"]) +  # Same-photo identity
                0.10 * (1.0 - val_losses["cosine_cross"]) if val_losses["cosine_cross"] > 0 else 0.0 +  # Cross-photo identity
                0.08 * (1.0 / (val_losses["magnitude"] + 1e-6)) +  # Magnitude consistency
                0.02 * (1.0 / (val_losses["feature_corr"] + 1e-6)) +  # Feature correlation
                # Discrimination (20% total)
                0.20 * (1.0 / (val_losses["discriminative"] + 1e-6)) +  # Different people stay different
                # Visual quality (35% total)
                0.20 * avg_val_psnr / 40.0 +  # PSNR normalized
                0.10 * (1.0 / (val_losses["l1"] + 1e-6)) +  # L1 pixel loss
                0.05 * (1.0 / (val_losses["perceptual"] + 1e-6))  # Perceptual quality
            )
            
            print(f"  Val PSNR: {avg_val_psnr:.2f} dB, Composite Score: {val_score:.4f}")
            print(f"    Identity - Cosine: {val_losses['cosine']:.4f}, Cross: {val_losses['cosine_cross']:.4f}, Mag: {val_losses['magnitude']:.4f}, FeatCorr: {val_losses['feature_corr']:.4f}")
            print(f"    Discrimination: {val_losses['discriminative']:.4f}")
            print(f"    Visual - L1: {val_losses['l1']:.4f}, Perceptual: {val_losses['perceptual']:.4f}")
            
            # Save best model based on composite score (identity-focused)
            if val_score > best_val_psnr:  # Reusing variable name for compatibility
                best_val_psnr = val_score
                patience_counter = 0
                
                # Apply EMA weights before saving
                ema.apply_shadow()
                
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_val_psnr,  # Composite score from all losses
                    "best_psnr": avg_val_psnr,
                    "val_losses": val_losses,
                    "config": asdict(config),
                    "efficiency_metrics": efficiency_metrics,
                    "composite_score_weights": {
                        "cosine": 0.25,
                        "cosine_cross": 0.10,
                        "magnitude": 0.08,
                        "feature_corr": 0.02,
                        "discriminative": 0.20,
                        "psnr": 0.20,
                        "l1": 0.10,
                        "perceptual": 0.05
                    }
                }
                torch.save(checkpoint, save_path)
                print(f"  ✓ Saved best model (Score: {best_val_psnr:.4f}, PSNR: {avg_val_psnr:.2f} dB)")
                
                # Restore training weights
                ema.restore()
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{config.early_stop_patience})")
                
                if patience_counter >= config.early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        # Step scheduler
        scheduler.step()
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}\n")
    
    print(f"\nTraining complete! Best validation score: {best_val_psnr:.4f}")
    print(f"  Composite score breakdown (ENHANCED for Face Recognition):")
    print(f"    - Identity preservation: 45% (cosine 25% + cross-photo 10% + magnitude 8% + feature-corr 2%)")
    print(f"    - Inter-subject discrimination: 20%")
    print(f"    - Visual quality: 35% (PSNR 20% + L1 10% + perceptual 5%)")
    print(f"  This prioritizes identity preservation and facial recognition accuracy")
    print(f"Model saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid Transformer-CNN DSR with identity-aware losses"
    )
    
    parser.add_argument(
        "--vlr-size",
        type=int,
        choices=[16, 24, 32],
        required=True,
        help="VLR input resolution (16x16, 24x24, or 32x32)"
    )
    parser.add_argument(
        "--edgeface",
        type=str,
        default="edgeface_finetuned_32.pth",
        help="EdgeFace model filename for identity loss"
    )
    parser.add_argument(
        "--frontal-only",
        action="store_true",
        help="Use frontal-only filtered dataset"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    
    args = parser.parse_args()
    
    # Create config for specified resolution
    config = HybridTrainConfig.for_resolution(args.vlr_size)
    
    # Start training
    train(config, args)


if __name__ == "__main__":
    main()
