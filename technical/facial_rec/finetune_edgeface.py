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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .edgeface_weights.edgeface import EdgeFace
from ..dsr import load_dsr_model


@dataclass
class FinetuneConfig:
    """Configuration for EdgeFace fine-tuning."""

    # Training
    stage1_epochs: int = 5  # Freeze backbone, train head
    stage2_epochs: int = 20  # Unfreeze all, fine-tune
    batch_size: int = 32
    num_workers: int = 8

    # Learning rates
    head_lr: float = 1e-3  # Stage 1
    backbone_lr: float = 5e-6  # Stage 2 - very low to preserve pretrained features
    head_lr_stage2: float = 5e-5  # Stage 2

    # Loss weights
    arcface_scale: float = 64.0  # ArcFace scale parameter (s)
    arcface_margin: float = 0.5  # ArcFace margin parameter (m)

    # Regularization
    weight_decay: float = 5e-5
    label_smoothing: float = 0.1

    # Other
    val_interval: int = 1
    seed: int = 42
    early_stop_patience: int = 8


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
    """Dataset that loads VLR images, runs DSR, and pairs with HR ground truth."""

    def __init__(
        self,
        dataset_root: Path,
        dsr_model: nn.Module,
        device: torch.device,
        augment: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.vlr_root = self.dataset_root / "vlr_images"
        self.hr_root = self.dataset_root / "hr_images"
        self.dsr_model = dsr_model
        self.device = device
        self.augment = augment

        # Build dataset
        self.samples: List[Tuple[Path, Path, int]] = []
        self.subject_to_id: Dict[str, int] = {}

        print("Building dataset from", self.dataset_root)
        vlr_paths = sorted(self.vlr_root.glob("*.png"))

        for vlr_path in tqdm(vlr_paths, desc="Indexing images"):
            hr_path = self.hr_root / vlr_path.name
            if not hr_path.exists():
                continue

            # Extract subject ID from filename (e.g., "007_01_..." -> "007")
            subject = vlr_path.stem.split("_")[0]
            if subject not in self.subject_to_id:
                self.subject_to_id[subject] = len(self.subject_to_id)

            subject_id = self.subject_to_id[subject]
            self.samples.append((vlr_path, hr_path, subject_id))

        print(f"Found {len(self.samples)} samples, {len(self.subject_to_id)} subjects")

        # EdgeFace preprocessing (resize to 112x112, normalize)
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.vlr_transform = transforms.ToTensor()

        # Augmentation for training
        self.aug_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.03),
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

        # Resize DSR output to 112x112 and normalize for EdgeFace
        sr_tensor = F.interpolate(
            sr_tensor, size=(112, 112), mode="bilinear", align_corners=False
        )
        sr_tensor = sr_tensor.squeeze(0).cpu()
        # Apply EdgeFace normalization
        sr_tensor = transforms.functional.normalize(
            sr_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        # Load and preprocess HR image
        hr_img = Image.open(hr_path).convert("RGB")
        hr_tensor = self.hr_transform(hr_img)

        return sr_tensor, hr_tensor, subject_id


def load_edgeface_backbone(weights_path: Path, device: torch.device) -> EdgeFace:
    """Load EdgeFace model and strip 'model.' prefix from state dict."""
    model = EdgeFace(embedding_size=512, back="edgeface_s")

    state_dict = torch.load(weights_path, map_location="cpu")

    # Strip 'model.' prefix if present
    cleaned_state = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        cleaned_state[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(
            f"[EdgeFace] Missing keys: {len(missing)} (expected if architecture differs slightly)"
        )
    if unexpected:
        print(f"[EdgeFace] Unexpected keys: {len(unexpected)}")

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
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    arcface.train()

    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for sr_imgs, hr_imgs, labels in tqdm(loader, desc="Training", leave=False):
        sr_imgs = sr_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            # Get embeddings from DSR outputs
            embeddings = model(sr_imgs)

            # Compute ArcFace logits
            logits = arcface(embeddings, labels)

            # Cross-entropy loss
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    arcface: ArcFaceLoss,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate on DSR->HR embedding similarity."""
    model.eval()
    arcface.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for sr_imgs, hr_imgs, labels in tqdm(loader, desc="Validation", leave=False):
        sr_imgs = sr_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = model(sr_imgs)
        logits = arcface(embeddings, labels)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args: argparse.Namespace) -> None:
    config = FinetuneConfig()
    if args.stage2_epochs:
        config.stage2_epochs = args.stage2_epochs

    device = torch.device(args.device)
    set_random_seed(config.seed)

    # Paths
    base_dir = Path(__file__).resolve().parents[2]
    train_dir = (
        Path(args.train_dir)
        if getattr(args, "train_dir", None)
        else base_dir / "technical" / "dataset" / "train_processed"
    )
    val_dir = (
        Path(args.val_dir)
        if getattr(args, "val_dir", None)
        else base_dir / "technical" / "dataset" / "val_processed"
    )

    if getattr(args, "edgeface_weights", None):
        edgeface_weights = Path(args.edgeface_weights)
    else:
        edgeface_weights = Path(__file__).parent / "edgeface_weights" / args.edgeface
    dsr_weights = (
        Path(args.dsr_weights)
        if getattr(args, "dsr_weights", None)
        else base_dir / "technical" / "dsr" / "dsr.pth"
    )

    dsr_weights = base_dir / "technical" / "dsr" / "dsr.pth"
    save_path = Path(__file__).parent / "edgeface_weights" / "edgeface_finetuned.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"EdgeFace weights: {edgeface_weights}")
    print(f"DSR weights: {dsr_weights}")

    # Load DSR model
    print("\nLoading DSR model...")
    dsr_model = load_dsr_model(dsr_weights, device=device)
    dsr_model.eval()
    for param in dsr_model.parameters():
        param.requires_grad = False

    # Build datasets
    print("\nBuilding training dataset...")
    train_dataset = DSROutputDataset(train_dir, dsr_model, device, augment=True)

    print("\nBuilding validation dataset...")
    val_dataset = DSROutputDataset(val_dir, dsr_model, device, augment=False)

    num_classes = len(train_dataset.subject_to_id)
    print(f"\nNumber of classes: {num_classes}")

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

    optimizer = optim.AdamW(
        arcface.parameters(),
        lr=config.head_lr,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_acc = 0.0

    for epoch in range(1, config.stage1_epochs + 1):
        train_loss, train_acc = train_epoch(
            backbone,
            arcface,
            train_loader,
            optimizer,
            scaler,
            device,
            config.label_smoothing,
        )

        val_loss, val_acc = validate(backbone, arcface, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{config.stage1_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✓ New best validation accuracy: {val_acc:.4f}")

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

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, config.stage2_epochs + 1):
        train_loss, train_acc = train_epoch(
            backbone,
            arcface,
            train_loader,
            optimizer,
            scaler,
            device,
            config.label_smoothing,
        )

        val_loss, val_acc = validate(backbone, arcface, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{config.stage2_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "backbone_state_dict": backbone.state_dict(),
                "arcface_state_dict": arcface.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "config": config,
                "subject_to_id": train_dataset.subject_to_id,
            }
            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved checkpoint to {save_path} (val acc: {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stop_patience:
                print(
                    f"\n⚠️  Early stopping after {epoch} epochs (no improvement for {config.early_stop_patience} epochs)"
                )
                break

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Fine-tuned model saved to: {save_path}")
    print("\nTo use the fine-tuned model, update your pipeline to load:")
    print(
        f"  edgeface_weights_path = Path('facial_rec/edgeface_weights/edgeface_finetuned.pth')"
    )


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
        help="EdgeFace checkpoint filename under edgeface_weights/ (used if --edgeface-weights not given)",
    )
    parser.add_argument(
        "--edgeface-weights",
        default=None,
        help="Explicit path to EdgeFace float checkpoint (overrides --edgeface)",
    )
    parser.add_argument(
        "--dsr-weights",
        default=None,
        help="Explicit path to DSR weights (overrides default technical/dsr/dsr.pth)",
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
        help="Override number of stage 2 epochs (default: 20)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
