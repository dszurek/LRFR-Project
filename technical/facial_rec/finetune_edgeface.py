"""Fine-tune EdgeFace on DSR outputs to improve recognition accuracy.

This script fine-tunes the EdgeFace recognition model on paired DSR super-resolved
images and their HR ground truth, teaching EdgeFace to better recognize faces from
your DSR model's output. This addresses the domain gap between training data and
your DSR outputs.

Strategy:
1. Generate DSR outputs for all training images offline
2. Train EdgeFace using metric learning (ArcFace loss) on:
   - DSR outputs as input
   - HR images as positive pairs (same identity)
   - Subject IDs for classification
3. Two-stage training:
   - Stage 1: Freeze backbone, train new classification head (5 epochs)
   - Stage 2: Unfreeze all, fine-tune with low LR (15-30 epochs)

Expected improvement: +10-20% recognition accuracy

IMPORTANT DATASET NOTES:
- Current dataset (frontal_only) was designed for DSR training with different people
  in train/val/test splits
- For EdgeFace fine-tuning, the validation accuracy is MEANINGLESS because val contains
  completely different subjects than training
- The PRIMARY METRIC is "embedding similarity" between DSR and HR outputs of the same person
- Classification accuracy is only computed on subjects that appear in both train and val
  (which may be zero with the current dataset)
- Consider creating a dedicated fine-tuning dataset with:
  - Same people in train/val (different images)
  - Multiple images per person for better identity learning
  - 80/20 train/val split of the same subjects
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .edgeface_weights.edgeface import EdgeFace
from ..dsr import load_dsr_model


@dataclass
class FinetuneConfig:
    """Configuration for EdgeFace fine-tuning.

    Resolution-aware hyperparameters optimized for different VLR input sizes.
    """

    vlr_size: int = 32  # VLR input size (16, 24, or 32)
    
    # Training - adjusted for resolution
    stage1_epochs: int = 10
    stage2_epochs: int = 25
    batch_size: int = 32
    # MUST be 0 because DSROutputDataset contains CUDA model (can't pickle for workers)
    # To use num_workers>0, would need to pre-compute all DSR outputs offline
    num_workers: int = 0

    # Learning rates - reduced to prevent gradient explosion with missing keys
    head_lr: float = 1e-4  # Stage 1 (heavily reduced from 9e-4 to prevent NaN)
    backbone_lr: float = 3e-6  # Stage 2 - reduced (was 6e-6)
    head_lr_stage2: float = 3e-5  # Stage 2 (reduced from 6e-5)

    # Loss weights - reduced for stability with 518 classes
    arcface_scale: float = 8.0  # Heavily reduced to prevent class collapse
    arcface_margin: float = 0.1  # Very soft margin for many classes
    temperature: float = 4.0  # High temperature to encourage diversity

    # Regularization
    weight_decay: float = 3e-5
    label_smoothing: float = 0.08

    # Other
    val_interval: int = 1
    seed: int = 42
    early_stop_patience: int = 10

    @classmethod
    def make(cls, vlr_size: int) -> "FinetuneConfig":
        """Create resolution-aware fine-tuning configuration.
        
        Args:
            vlr_size: VLR input size (16, 24, or 32)
            
        Returns:
            FinetuneConfig with optimized hyperparameters for the resolution
        """
        cfg = cls(vlr_size=vlr_size)
        
        if vlr_size <= 16:
            # 16×16 needs more aggressive fine-tuning
            cfg.stage1_epochs = max(cfg.stage1_epochs, 15)
            cfg.stage2_epochs = max(cfg.stage2_epochs, 40)
            cfg.batch_size = 256  # A100 optimization
            cfg.head_lr = 1.5e-4
            cfg.backbone_lr = 5e-6
            cfg.head_lr_stage2 = 5e-5
            cfg.arcface_scale = 12.0
            cfg.temperature = 3.0
            cfg.early_stop_patience = 15
        elif vlr_size <= 24:
            # 24×24 moderate adjustments
            cfg.stage1_epochs = max(cfg.stage1_epochs, 12)
            cfg.stage2_epochs = max(cfg.stage2_epochs, 35)
            cfg.batch_size = 256  # A100 optimization
            cfg.head_lr = 1.3e-4
            cfg.backbone_lr = 4e-6
            cfg.head_lr_stage2 = 4e-5
            cfg.arcface_scale = 10.0
            cfg.temperature = 3.5
            cfg.early_stop_patience = 12
        else:
            # 32×32 uses base configuration
            cfg.vlr_size = 32
            cfg.batch_size = 256  # A100 optimization
            cfg.stage2_epochs = 30
        
        return cfg


class ArcFaceLoss(nn.Module):
    """ArcFace loss for metric learning.

    Encourages embeddings to form tight clusters on a hypersphere,
    with better inter-class separation than softmax.
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight_norm)

        # Add margin to target class
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)

        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class DSROutputDataset(Dataset):
    """Dataset that loads VLR images, runs DSR, and pairs with HR ground truth.

    DSR outputs 112×112 which matches EdgeFace's required input size.
    HR images are also 112×112 for direct comparison.
    """

    def __init__(
        self,
        dataset_root: Path,
        dsr_model: nn.Module,
        device: torch.device,
        vlr_size: int = 32,
        augment: bool = True,
        subject_to_id: Dict[str, int] | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        
        # Resolve VLR directory based on size (consistent format for all sizes)
        vlr_dir_name = f"vlr_images_{vlr_size}x{vlr_size}"
        self.vlr_root = self.dataset_root / vlr_dir_name
        self.hr_root = self.dataset_root / "hr_images"
        
        if not self.vlr_root.exists():
            raise ValueError(
                f"VLR directory not found: {self.vlr_root}\n"
                f"Please regenerate dataset with --vlr-size {vlr_size}"
            )
        
        self.dsr_model = dsr_model
        self.device = device
        self.vlr_size = vlr_size
        self.augment = augment

        # Build dataset
        self.samples: List[Tuple[Path, Path, int]] = []
        # Use provided mapping or create new one (for train set)
        self.subject_to_id: Dict[str, int] = (
            subject_to_id if subject_to_id is not None else {}
        )

        print(f"Building dataset from {self.dataset_root} (VLR: {vlr_dir_name})")
        vlr_paths = sorted(self.vlr_root.glob("*.png"))

        for vlr_path in tqdm(vlr_paths, desc="Indexing images"):
            hr_path = self.hr_root / vlr_path.name
            if not hr_path.exists():
                continue

            # Extract subject ID from filename
            # Format: ###_##_##_###_##_crop_128.png (CMU) or ####_####_lfw.png (LFW)
            stem = vlr_path.stem
            if "_lfw" in stem:
                # LFW format: ####_####_lfw
                subject = stem.split("_")[0]
            else:
                # CMU format: ###_##_##_###_##_crop_128
                subject = stem.split("_")[0]

            # Add subject to mapping if not present
            # For validation with pre-existing mapping, new subjects get new IDs
            if subject not in self.subject_to_id:
                self.subject_to_id[subject] = len(self.subject_to_id)

            subject_id = self.subject_to_id[subject]
            self.samples.append((vlr_path, hr_path, subject_id))

        print(f"Found {len(self.samples)} samples, {len(self.subject_to_id)} subjects")

        # EdgeFace preprocessing (no resize needed - already 112×112, just normalize)
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.vlr_transform = transforms.ToTensor()

        # Augmentation for training - adjust based on VLR size
        if vlr_size <= 16:
            # 16×16 needs gentler augmentation
            aug_brightness = 0.05
            aug_contrast = 0.05
            aug_saturation = 0.03
            aug_rotation = 3
        elif vlr_size <= 24:
            # 24×24 moderate augmentation
            aug_brightness = 0.06
            aug_contrast = 0.06
            aug_saturation = 0.04
            aug_rotation = 4
        else:
            # 32×32 can handle more aggressive augmentation
            aug_brightness = 0.07
            aug_contrast = 0.07
            aug_saturation = 0.05
            aug_rotation = 5
        
        self.aug_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=aug_brightness,
                    contrast=aug_contrast,
                    saturation=aug_saturation,
                    hue=0.02
                ),
                transforms.RandomRotation(
                    degrees=aug_rotation,
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        vlr_path, hr_path, subject_id = self.samples[idx]

        # Load VLR image
        vlr_img = Image.open(vlr_path).convert("RGB")

        # Apply augmentation if training
        if self.augment:
            vlr_img = self.aug_transform(vlr_img)

        # Convert to tensor and run through DSR
        vlr_tensor = self.vlr_transform(vlr_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            sr_tensor = self.dsr_model(vlr_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)

        # DSR may output different sizes (e.g., 160×160)
        # Resize to 112×112 for EdgeFace if needed
        sr_tensor = sr_tensor.squeeze(0).cpu()

        if sr_tensor.shape[1] != 112 or sr_tensor.shape[2] != 112:
            sr_tensor = transforms.functional.resize(
                sr_tensor,
                [112, 112],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )

        # Normalize for EdgeFace
        sr_tensor = transforms.functional.normalize(
            sr_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        # Load and preprocess HR image (resize to 112×112 if needed)
        hr_img = Image.open(hr_path).convert("RGB")
        if hr_img.size != (112, 112):
            hr_img = hr_img.resize((112, 112), Image.Resampling.BILINEAR)
        hr_tensor = self.hr_transform(hr_img)

        return sr_tensor, hr_tensor, subject_id


def load_edgeface_backbone(
    weights_path: Path, device: torch.device, backbone_type: str = "edgeface_xxs"
) -> EdgeFace:
    """Load EdgeFace model using TorchScript or architecture loading.

    Supports:
    - TorchScript models (edgeface_xxs.pt, edgeface_s_gamma_05.pt)
    - State dict checkpoints (edgeface_finetuned.pth)
    """
    # Auto-detect backbone from filename if not specified
    if "xxs" in weights_path.stem.lower():
        backbone_type = "edgeface_xxs"
    elif "_s" in weights_path.stem.lower():
        backbone_type = "edgeface_s"
    elif "gamma" in weights_path.stem.lower():
        backbone_type = "edgeface_s_gamma_05"

    print(f"[EdgeFace] Using backbone: {backbone_type}")

    # Try TorchScript loading first (for pretrained models)
    try:
        print(f"[EdgeFace] Attempting TorchScript load from {weights_path.name}")
        model = torch.jit.load(str(weights_path), map_location=device)
        print(f"[EdgeFace] Successfully loaded as TorchScript model")
        model.eval()
        return model
    except Exception as e:
        print(f"[EdgeFace] TorchScript load failed: {str(e)[:100]}")
        print(f"[EdgeFace] Falling back to EdgeFace architecture loading")

    # Fall back to architecture loading (for fine-tuned checkpoints)
    model = EdgeFace(embedding_size=512, back=backbone_type)

    state_dict = torch.load(weights_path, map_location="cpu")

    # Handle nested state dict formats
    if "backbone_state_dict" in state_dict:
        # Fine-tuned checkpoint format
        state_dict = state_dict["backbone_state_dict"]
    elif "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Strip common prefixes
    cleaned_state = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone.") :]
        cleaned_state[new_key] = value

    print(f"[EdgeFace] Sample keys after cleaning: {list(cleaned_state.keys())[:3]}")

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"[EdgeFace] Missing keys: {len(missing)}")
        if len(missing) <= 10:
            print(f"[EdgeFace] Missing key names: {missing}")

        # Initialize missing positional embedding keys (XCA blocks)
        if any("pos_embd" in key for key in missing):
            print(f"[EdgeFace] Initializing missing positional embedding keys...")
            for key in missing:
                if "pos_embd" in key:
                    # Navigate through model hierarchy
                    # Example: stages.2.blocks.5.pos_embd.token_projection.weight
                    parts = key.split(".")
                    module = model

                    try:
                        for i, part in enumerate(parts[:-1]):
                            if part.isdigit():
                                # Numeric index - use indexing
                                module = module[int(part)]
                            else:
                                # Named attribute - use getattr
                                module = getattr(module, part)

                        # Get the final parameter
                        param = getattr(module, parts[-1])

                        # Initialize based on parameter type
                        if "weight" in parts[-1]:
                            nn.init.xavier_uniform_(param)
                        elif "bias" in parts[-1]:
                            nn.init.zeros_(param)

                        print(
                            f"[EdgeFace]   [OK] Initialized {key}: shape {param.shape}"
                        )
                    except Exception as e:
                        print(
                            f"[EdgeFace]   [FAIL] Failed to initialize {key}: {str(e)}"
                        )
                        # Continue anyway - model will train with random initialization

    if unexpected:
        print(f"[EdgeFace] Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 10:
            print(f"[EdgeFace] Unexpected key names: {unexpected}")

    model.to(device)
    return model


def train_epoch(
    model: nn.Module,
    arcface: ArcFaceLoss,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    label_smoothing: float,
    temperature: float = 1.0,
    use_contrastive: bool = True,
    contrastive_weight: float = 0.3,
) -> Tuple[float, float, float]:
    """Train for one epoch with optional contrastive learning.

    Args:
        model: EdgeFace backbone
        arcface: ArcFace classification head
        loader: Training data loader
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        device: Training device
        label_smoothing: Label smoothing factor
        temperature: Temperature for logits scaling (higher = softer predictions)
        use_contrastive: If True, add contrastive loss between DSR and HR embeddings
        contrastive_weight: Weight for contrastive loss (typically 0.2-0.5)

    Returns:
        Tuple of (average_loss, top1_accuracy, top5_accuracy)
    """
    model.train()
    arcface.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_cont_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    first_batch = True

    for sr_imgs, hr_imgs, labels in tqdm(loader, desc="Training", leave=False):
        sr_imgs = sr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=device.type == "cuda"):
            # Get embeddings from DSR outputs
            sr_embeddings = model(sr_imgs)

            # Compute ArcFace classification logits
            logits = arcface(sr_embeddings, labels)

            # Apply temperature scaling to prevent overconfidence
            logits = logits / temperature

            # Debug first batch
            if first_batch:
                print(f"\n[DEBUG] First batch:")
                print(f"  Embeddings shape: {sr_embeddings.shape}")
                print(
                    f"  Embeddings mean: {sr_embeddings.mean().item():.4f}, std: {sr_embeddings.std().item():.4f}"
                )
                print(
                    f"  Labels shape: {labels.shape}, range: [{labels.min().item()}, {labels.max().item()}]"
                )
                print(f"  Logits shape: {logits.shape}")
                print(
                    f"  Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}"
                )
                print(f"  Top prediction: {logits.argmax(dim=1)[:10]}")
                print(f"  True labels: {labels[:10]}")
                first_batch = False

            # Classification loss
            cls_loss = criterion(logits, labels)

            # Contrastive loss: Make DSR and HR embeddings similar for same person
            if use_contrastive:
                # Get embeddings from HR images (ground truth)
                hr_embeddings = model(hr_imgs)

                # Normalize embeddings
                sr_embeddings_norm = F.normalize(sr_embeddings, dim=1)
                hr_embeddings_norm = F.normalize(hr_embeddings, dim=1)

                # Cosine similarity between DSR and HR (should be close to 1.0)
                # Since they're the same person, we want similarity = 1
                cosine_sim = (sr_embeddings_norm * hr_embeddings_norm).sum(dim=1)

                # Loss: minimize distance (maximize similarity)
                # contrastive_loss = 1 - cosine_similarity
                contrastive_loss = (1.0 - cosine_sim).mean()

                # Combined loss
                loss = cls_loss + contrastive_weight * contrastive_loss

                total_cont_loss += contrastive_loss.item()
            else:
                loss = cls_loss

        # Check for NaN before backprop
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  WARNING: Loss is {loss.item()}, skipping batch")
            continue

        scaler.scale(loss).backward()

        # Gradient clipping to prevent explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(arcface.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Metrics
        loss_value = loss.item()
        if not (
            torch.isnan(torch.tensor(loss_value))
            or torch.isinf(torch.tensor(loss_value))
        ):
            total_loss += loss_value
            total_cls_loss += cls_loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()

        # Top-5 accuracy (more meaningful for many classes)
        _, top5_pred = logits.topk(5, dim=1)
        for i in range(labels.size(0)):
            if labels[i] in top5_pred[i]:
                correct_top5 += 1

        total += labels.size(0)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else float("inf")
    avg_cls_loss = total_cls_loss / len(loader) if len(loader) > 0 else float("inf")
    avg_cont_loss = (
        total_cont_loss / len(loader) if use_contrastive and len(loader) > 0 else 0.0
    )
    accuracy = correct / total if total > 0 else 0.0
    top5_accuracy = correct_top5 / total if total > 0 else 0.0

    # Log actual numbers for debugging with many classes
    print(
        f"    [Top-1] {correct}/{total} = {accuracy:.4f} | [Top-5] {correct_top5}/{total} = {top5_accuracy:.4f}"
    )

    # Check for NaN in final metrics
    if torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)):
        print(f"\n❌ ERROR: Training loss became NaN/Inf - stopping training")
        return float("nan"), accuracy, top5_accuracy

    # Print detailed loss breakdown if using contrastive
    if use_contrastive:
        print(
            f"    [Losses] Total: {avg_loss:.4f} | Cls: {avg_cls_loss:.4f} | Cont: {avg_cont_loss:.4f}"
        )

    return avg_loss, accuracy, top5_accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    arcface: ArcFaceLoss,
    loader: DataLoader,
    device: torch.device,
    train_subjects: set,
) -> Tuple[float, float, float, float]:
    """Validate on DSR->HR embedding similarity.

    Args:
        model: EdgeFace backbone
        arcface: ArcFace classification head
        loader: Validation data loader
        device: Device to use
        train_subjects: Set of subject IDs seen during training

    Returns:
        Tuple of (avg_loss, top1_accuracy, top5_accuracy, embedding_similarity)
    """
    model.eval()
    arcface.eval()

    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    total_similarity = 0.0
    similarity_count = 0

    criterion = nn.CrossEntropyLoss()

    for sr_imgs, hr_imgs, labels in tqdm(loader, desc="Validation", leave=False):
        sr_imgs = sr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Get embeddings from DSR outputs and HR images
        sr_embeddings = model(sr_imgs)
        hr_embeddings = model(hr_imgs)

        # Normalize embeddings
        sr_embeddings_norm = F.normalize(sr_embeddings, dim=1)
        hr_embeddings_norm = F.normalize(hr_embeddings, dim=1)

        # Compute embedding similarity (this is what matters for DSR quality)
        # Note: This measures if DSR output and HR have similar embeddings (same person)
        similarity = (sr_embeddings_norm * hr_embeddings_norm).sum(dim=1)
        total_similarity += similarity.sum().item()
        similarity_count += similarity.size(0)

        # Classification accuracy (only meaningful for subjects in training set)
        logits = arcface(sr_embeddings, labels)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)

        # Top-5 predictions
        _, top5_pred = logits.topk(5, dim=1)

        # Only count accuracy for subjects that were in training set
        for i in range(labels.size(0)):
            if labels[i].item() in train_subjects:
                if pred[i] == labels[i]:
                    correct += 1
                if labels[i] in top5_pred[i]:
                    correct_top5 += 1
                total += 1

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    top5_accuracy = correct_top5 / total if total > 0 else 0.0
    avg_similarity = (
        total_similarity / similarity_count if similarity_count > 0 else 0.0
    )

    # Log actual numbers for debugging with many classes
    print(
        f"    [Val Top-1] {correct}/{total} = {accuracy:.4f} | [Top-5] {correct_top5}/{total} = {top5_accuracy:.4f}"
    )

    return avg_loss, accuracy, top5_accuracy, avg_similarity


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args: argparse.Namespace) -> None:
    config = FinetuneConfig.make(args.vlr_size)
    if args.stage2_epochs:
        config.stage2_epochs = args.stage2_epochs

    device = torch.device(args.device)
    set_random_seed(config.seed)

    # Paths
    base_dir = Path(__file__).resolve().parents[2]
    
    # IMPORTANT: For fine-tuning, we need a dataset where the SAME people appear
    # in both train and val (but different images). The current frontal_only/
    # dataset has this property for small-scale identity matching.
    # The original train_processed/val_processed has different people in each split
    # which is good for DSR training but NOT for fine-tuning EdgeFace.
    
    if args.use_small_gallery:
        # Use frontal_only dataset which has consistent subjects across splits
        # This is better for 1:1 and 1:N matching with N≤10
        train_dir = base_dir / "technical" / "dataset" / "frontal_only" / "train"
        val_dir = base_dir / "technical" / "dataset" / "frontal_only" / "val"
        print(f"📊 Using frontal_only dataset (better for small gallery fine-tuning)")
    else:
        train_dir = (
            Path(args.train_dir)
            if getattr(args, "train_dir", None)
            else base_dir / "technical" / "dataset" / "edgeface_finetune" / "train"
        )
        val_dir = (
            Path(args.val_dir)
            if getattr(args, "val_dir", None)
            else base_dir / "technical" / "dataset" / "edgeface_finetune" / "val"
        )
        print("⚠️  Using edgeface_finetune dataset (optimized for identity learning)")

    if getattr(args, "edgeface_weights", None):
        edgeface_weights = Path(args.edgeface_weights)
    else:
        edgeface_weights = Path(__file__).parent / "edgeface_weights" / args.edgeface
    
    # Use resolution-specific DSR weights
    dsr_weights = (
        Path(args.dsr_weights)
        if getattr(args, "dsr_weights", None)
        else base_dir / "technical" / "dsr" / f"hybrid_dsr{args.vlr_size}.pth"
    )
    
    # Save resolution-specific EdgeFace weights
    save_path = Path(__file__).parent / "edgeface_weights" / f"edgeface_finetuned_{args.vlr_size}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"VLR size: {args.vlr_size}×{args.vlr_size}")
    print(f"EdgeFace weights: {edgeface_weights}")
    print(f"DSR weights: {dsr_weights}")
    print(f"Will save to: {save_path}")

    # Load DSR model
    print(f"\nLoading DSR model for {args.vlr_size}×{args.vlr_size} VLR...")
    dsr_model = load_dsr_model(dsr_weights, device=device)
    dsr_model.eval()
    for param in dsr_model.parameters():
        param.requires_grad = False

    # Build datasets
    print("\nBuilding training dataset...")
    train_dataset = DSROutputDataset(
        train_dir, dsr_model, device, vlr_size=args.vlr_size, augment=True
    )

    print("\nBuilding validation dataset...")
    # CRITICAL: Use same subject_to_id mapping as training set!
    val_dataset = DSROutputDataset(
        val_dir,
        dsr_model,
        device,
        vlr_size=args.vlr_size,
        augment=False,
        subject_to_id=train_dataset.subject_to_id,
    )

    # Total classes = train subjects + any new val subjects
    num_classes = len(val_dataset.subject_to_id)
    print(f"\nNumber of training subjects: {len(train_dataset.subject_to_id)}")
    print(f"Number of total subjects (train + val): {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Load EdgeFace backbone
    print("\nLoading EdgeFace backbone...")
    backbone = load_edgeface_backbone(edgeface_weights, device)

    # Create ArcFace head
    arcface = ArcFaceLoss(
        embedding_size=512,
        num_classes=num_classes,
        scale=config.arcface_scale,
        margin=config.arcface_margin,
    ).to(device)

    # ========================================================================
    # STAGE 1: Freeze backbone, train ArcFace head
    # ========================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Training ArcFace head (backbone frozen)")
    print("=" * 60)

    for param in backbone.parameters():
        param.requires_grad = False

    print(f"[Info] With {len(train_subject_ids)} classes, accuracy will start low and improve gradually")
    print(f"[Info] PRIMARY METRIC: Embedding similarity (should stay >0.95)")

    for epoch in range(1, config.stage1_epochs + 1):
        # Stage 1: No contrastive learning (just train classification head)
        train_loss, train_acc, train_top5 = train_epoch(
            backbone,
            arcface,
            train_loader,
            optimizer,
            scaler,
            device,
            config.label_smoothing,
            config.temperature,
            use_contrastive=False,  # Disable contrastive in Stage 1
        )

        # Check for NaN in training
        if torch.isnan(torch.tensor(train_loss)):
            print(f"\n❌ CRITICAL: Training loss is NaN at epoch {epoch}")
            print(f"   This usually means gradient explosion or numerical instability")
            print(f"   Training cannot continue - please check:")
            print(f"   1. Learning rate might be too high")
            print(f"   2. Missing model weights causing instability")
            print(f"   3. Dataset might have corrupted images")
            break

        val_loss, val_acc, val_top5, val_similarity = validate(
            backbone, arcface, val_loader, device, train_subject_ids
        )

        print(
            f"Epoch {epoch:02d}/{config.stage1_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Top5: {train_top5:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Top5: {val_top5:.4f} Sim: {val_similarity:.4f}"
        )
        
        # Add context for users with many classes
        if len(train_subject_ids) > 100 and epoch == 1:
            print(f"  [Note] With {len(train_subject_ids)} classes, accuracy starts low.")
            print(f"  [Note] Watch for: (1) Similarity >0.95 ✓  (2) Accuracy improving ✓  (3) Loss decreasing ✓")

        if val_similarity > best_val_similarity:
            best_val_similarity = val_similarity
            print(f"  [BEST] New best validation similarity: {val_similarity:.4f}")

            # Save checkpoint even in Stage 1
            checkpoint = {
                "stage": 1,
                "epoch": epoch,
                "backbone_state_dict": backbone.state_dict(),
                "arcface_state_dict": arcface.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_similarity": val_similarity,
                "val_accuracy": val_acc,
                "config": config,
                "subject_to_id": train_dataset.subject_to_id,
            }
            torch.save(checkpoint, save_path)
            print(f"  [SAVED] Saved Stage 1 checkpoint to {save_path}")

    # ========================================================================
    # STAGE 2: Unfreeze backbone, fine-tune end-to-end
    # ========================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-tuning entire model (backbone unfrozen)")
    print("=" * 60)

    for param in backbone.parameters():
        param.requires_grad = True

    # Different learning rates for backbone vs head
    optimizer = optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": config.backbone_lr},
            {"params": arcface.parameters(), "lr": config.head_lr_stage2},
        ],
        weight_decay=config.weight_decay,
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.stage2_epochs
    )

    best_val_similarity = 0.0
    epochs_without_improvement = 0
    
    # Track metrics for relative improvement
    first_epoch_acc = None

    for epoch in range(1, config.stage2_epochs + 1):
        # Stage 2: Enable contrastive learning (align DSR and HR embeddings)
        train_loss, train_acc, train_top5 = train_epoch(
            backbone,
            arcface,
            train_loader,
            optimizer,
            scaler,
            device,
            config.label_smoothing,
            config.temperature,
            use_contrastive=True,  # Enable contrastive in Stage 2
            contrastive_weight=0.3,  # 30% weight for DSR-HR alignment
        )

        # Check for NaN in training
        if torch.isnan(torch.tensor(train_loss)):
            print(f"\n❌ CRITICAL: Training loss is NaN at epoch {epoch}")
            print(f"   Training stopped due to numerical instability")
            break

        val_loss, val_acc, val_top5, val_similarity = validate(
            backbone, arcface, val_loader, device, train_subject_ids
        )

        scheduler.step()

        # Track first epoch baseline for comparison
        if epoch == 1:
            first_epoch_acc = val_acc

        # Show relative improvement from first epoch
        improvement_str = ""
        if first_epoch_acc is not None and first_epoch_acc > 0:
            improvement_pct = ((val_acc - first_epoch_acc) / first_epoch_acc) * 100
            if improvement_pct > 0:
                improvement_str = f" (+{improvement_pct:.1f}% from epoch 1)"

        print(
            f"Epoch {epoch:02d}/{config.stage2_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Top5: {train_top5:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Top5: {val_top5:.4f} Sim: {val_similarity:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}{improvement_str}"
        )
        
        # Add helpful context on first epoch of Stage 2
        if epoch == 1:
            print(f"  [Stage 2] Backbone now unfrozen - expect faster accuracy improvements")
            print(f"  [Stage 2] Contrastive loss active - aligning DSR and HR embeddings")

        # Use similarity as primary metric (this measures DSR->HR quality)
        if val_similarity > best_val_similarity:
            best_val_similarity = val_similarity
            epochs_without_improvement = 0

            # Save checkpoint
            checkpoint = {
                "stage": 2,
                "epoch": epoch,
                "backbone_state_dict": backbone.state_dict(),
                "arcface_state_dict": arcface.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_similarity": val_similarity,
                "val_accuracy": val_acc,
                "config": config,
                "subject_to_id": train_dataset.subject_to_id,
            }
            torch.save(checkpoint, save_path)
            print(
                f"  [SAVED] Saved Stage 2 checkpoint to {save_path} (val sim: {val_similarity:.4f})"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stop_patience:
                print(
                    f"\n⚠️  Early stopping after {epoch} epochs (no improvement for {config.early_stop_patience} epochs)"
                )
                break

    print(f"\nTraining complete! Best validation similarity: {best_val_similarity:.4f}")

    if save_path.exists():
        print(f"[SUCCESS] Fine-tuned model saved to: {save_path}")
        print("\nTo use the fine-tuned model, update your pipeline to load:")
        print(
            f"  edgeface_weights_path = Path('facial_rec/edgeface_weights/edgeface_finetuned.pth')"
        )
        print(f"\n📊 Key Metrics:")
        print(f"  Best Validation Similarity: {best_val_similarity:.4f}")
        print(f"  (Similarity measures how well DSR outputs match HR quality)")
    else:
        print(
            f"[WARNING] No model was saved (validation similarity never improved from 0.0000)"
        )
        print(f"  This suggests a training issue - check:")
        print(f"  1. EdgeFace model loaded correctly (check missing keys)")
        print(f"  2. DSR model producing valid outputs")
        print(f"  3. Dataset images loading correctly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune EdgeFace on DSR outputs for improved recognition"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--edgeface",
        default="edgeface_xxs.pt",
        help="EdgeFace checkpoint filename under edgeface_weights/ (use edgeface_xxs.pt for initial training)",
    )
    parser.add_argument(
        "--edgeface-weights",
        default=None,
        help="Explicit path to EdgeFace float checkpoint (overrides --edgeface)",
    )
    parser.add_argument(
        "--dsr-weights",
        default=None,
        help="Explicit path to DSR weights (overrides default technical/dsr/dsrNN.pth based on --vlr-size)",
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Path to train_processed dataset root (contains vlr_images/ and hr_images/).",
    )
    parser.add_argument(
        "--val-dir",
        default=None,
        help="Path to val_processed dataset root (contains vlr_images/ and hr_images/).",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=None,
        help="Override number of stage 2 epochs (default: varies by resolution)",
    )
    parser.add_argument(
        "--vlr-size",
        type=int,
        default=32,
        choices=(16, 24, 32),
        help="VLR input resolution (16, 24, or 32). Must match trained DSR model.",
    )
    parser.add_argument(
        "--use-small-gallery",
        action="store_true",
        help="Use frontal_only dataset (recommended for 1:1 and small 1:N matching scenarios)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
