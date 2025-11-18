"""Fine-tune EdgeFace with CORRECT metric learning for DSR→HR matching.

This script uses contrastive learning to ensure DSR outputs produce embeddings
similar to HR images of the SAME person, while being dissimilar to DIFFERENT people.

Key differences from old approach:
1. Uses InfoNCE/NT-Xent contrastive loss instead of classification
2. Validation measures cross-image matching (gallery vs probe)
3. Focuses on embedding similarity, not classification accuracy
4. Batches contain multiple images per person for hard negative mining

Expected result: DSR→HR embeddings with 0.7+ similarity (vs 0.17 with old approach)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import random

from technical.facial_rec.edgeface_weights.edgeface import EdgeFace
from technical.dsr.models import load_dsr_model


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive fine-tuning."""

    # Paths
    dataset_root: Path = Path("technical/dataset/edgeface_finetune/train")
    val_dataset_root: Path = Path("technical/dataset/edgeface_finetune/val")
    dsr_weights: Path = Path("technical/dsr/dsr32.pth")
    edgeface_pretrained: Path = Path(
        "technical/facial_rec/edgeface_weights/edgeface_xxs.pt"
    )
    save_path: Path = Path(
        "technical/facial_rec/edgeface_weights/edgeface_contrastive.pth"
    )

    # Training
    batch_size: int = 32  # Must be even for paired sampling
    images_per_subject: int = 4  # Each batch contains N subjects × K images
    epochs: int = 30
    lr: float = 1e-5  # Very low LR for fine-tuning
    weight_decay: float = 1e-4
    temperature: float = 0.07  # Temperature for InfoNCE loss

    # Hardware
    device: str = "cuda"
    num_workers: int = 0  # MUST be 0 because DSR model in dataset


class ContrastivePairDataset(Dataset):
    """Dataset that yields paired DSR and HR images for contrastive learning."""

    def __init__(
        self,
        dataset_root: Path,
        dsr_model: nn.Module,
        device: torch.device,
        images_per_subject: int = 4,
    ):
        self.dataset_root = Path(dataset_root)
        self.vlr_root = self.dataset_root / "vlr_images"
        self.hr_root = self.dataset_root / "hr_images"
        self.dsr_model = dsr_model
        self.device = device
        self.images_per_subject = images_per_subject

        # Group images by subject
        self.subject_images: Dict[str, List[Path]] = defaultdict(list)

        for vlr_path in self.vlr_root.glob("*.png"):
            hr_path = self.hr_root / vlr_path.name
            if not hr_path.exists():
                continue

            subject = vlr_path.stem.split("_")[0]
            self.subject_images[subject].append(vlr_path)

        # Filter subjects with enough images
        self.subjects = [
            s
            for s, imgs in self.subject_images.items()
            if len(imgs) >= images_per_subject
        ]

        print(
            f"Dataset: {len(self.subjects)} subjects with {images_per_subject}+ images"
        )
        print(
            f"  Total images: {sum(len(self.subject_images[s]) for s in self.subjects)}"
        )

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Returns (dsr_images, hr_images, subject_id).

        Both tensors have shape [images_per_subject, 3, 112, 112].
        """
        subject = self.subjects[idx]
        images = self.subject_images[subject]

        # Randomly sample images_per_subject images
        if len(images) > self.images_per_subject:
            sampled = random.sample(images, self.images_per_subject)
        else:
            sampled = images

        dsr_batch = []
        hr_batch = []

        for vlr_path in sampled:
            hr_path = self.hr_root / vlr_path.name

            # Load VLR, run through DSR
            vlr_img = Image.open(vlr_path).convert("RGB")
            vlr_tensor = (
                torch.from_numpy(np.array(vlr_img)).permute(2, 0, 1).float() / 255.0
            )
            vlr_tensor = vlr_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                dsr_output = self.dsr_model(vlr_tensor).detach().clone()

            # Normalize to [-1, 1] for EdgeFace
            dsr_normalized = (dsr_output.squeeze(0) - 0.5) / 0.5
            dsr_batch.append(dsr_normalized.cpu())

            # Load HR
            hr_img = Image.open(hr_path).convert("RGB")
            hr_tensor = (
                torch.from_numpy(np.array(hr_img)).permute(2, 0, 1).float() / 255.0
            )
            hr_normalized = (hr_tensor - 0.5) / 0.5
            hr_batch.append(hr_normalized)

        dsr_stacked = torch.stack(dsr_batch)  # [K, 3, 112, 112]
        hr_stacked = torch.stack(hr_batch)  # [K, 3, 112, 112]

        return dsr_stacked, hr_stacked, idx


def contrastive_loss(
    embeddings: torch.Tensor,
    subject_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss.

    Args:
        embeddings: [N, D] normalized embeddings
        subject_ids: [N] subject IDs
        temperature: Temperature scaling factor

    Returns:
        Scalar loss
    """
    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature  # [N, N]

    # Create positive mask (same subject, different image)
    subject_ids = subject_ids.unsqueeze(1)  # [N, 1]
    pos_mask = (subject_ids == subject_ids.T).float()  # [N, N]

    # Remove self-similarity
    pos_mask.fill_diagonal_(0)

    # Create negative mask (different subject)
    neg_mask = 1 - pos_mask
    neg_mask.fill_diagonal_(0)

    # For each anchor, compute loss
    losses = []
    for i in range(embeddings.size(0)):
        # Positive similarities
        pos_sims = sim_matrix[i][pos_mask[i].bool()]

        if pos_sims.numel() == 0:
            continue  # Skip if no positives

        # Negative similarities
        neg_sims = sim_matrix[i][neg_mask[i].bool()]

        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        pos_exp = torch.exp(pos_sims)
        neg_exp = torch.exp(neg_sims).sum()

        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()
        losses.append(loss)

    return (
        torch.stack(losses).mean()
        if losses
        else torch.tensor(0.0, device=embeddings.device)
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    temperature: float,
) -> Tuple[float, float]:
    """Train one epoch with contrastive loss."""
    model.train()

    total_loss = 0.0
    total_similarity = 0.0
    count = 0

    for dsr_batch, hr_batch, subject_ids in tqdm(loader, desc="Training"):
        # Flatten: [B, K, C, H, W] -> [B*K, C, H, W]
        B, K = dsr_batch.shape[:2]
        dsr_flat = dsr_batch.view(B * K, *dsr_batch.shape[2:]).to(device)
        hr_flat = hr_batch.view(B * K, *hr_batch.shape[2:]).to(device)

        # Create subject IDs: [B] -> [B*K]
        subject_ids_flat = subject_ids.unsqueeze(1).repeat(1, K).view(-1).to(device)

        optimizer.zero_grad()

        # Get embeddings
        dsr_embeddings = model(dsr_flat)  # [B*K, D]
        hr_embeddings = model(hr_flat)  # [B*K, D]

        # Normalize
        dsr_norm = F.normalize(dsr_embeddings, dim=1)
        hr_norm = F.normalize(hr_embeddings, dim=1)

        # Combine DSR and HR embeddings for contrastive learning
        # This ensures DSR→HR pairs are pulled together
        all_embeddings = torch.cat([dsr_norm, hr_norm], dim=0)  # [2*B*K, D]
        all_subject_ids = torch.cat(
            [subject_ids_flat, subject_ids_flat], dim=0
        )  # [2*B*K]

        # Contrastive loss
        loss = contrastive_loss(all_embeddings, all_subject_ids, temperature)

        loss.backward()
        optimizer.step()

        # Compute DSR→HR similarity for monitoring
        similarity = (dsr_norm * hr_norm).sum(dim=1).mean().item()

        total_loss += loss.item()
        total_similarity += similarity
        count += 1

    return total_loss / count, total_similarity / count


@torch.no_grad()
def validate_gallery_matching(
    model: nn.Module,
    val_dataset_root: Path,
    dsr_model: nn.Module,
    device: torch.device,
    num_subjects: int = 20,
) -> Tuple[float, float]:
    """Validate using gallery-probe matching (realistic evaluation).

    For each subject:
    1. Build gallery from HR images
    2. Test with DSR outputs from VLR images
    3. Measure similarity and matching accuracy
    """
    model.eval()

    vlr_root = val_dataset_root / "vlr_images"
    hr_root = val_dataset_root / "hr_images"

    # Group by subject
    subject_images: Dict[str, List[Path]] = defaultdict(list)
    for vlr_path in vlr_root.glob("*.png"):
        hr_path = hr_root / vlr_path.name
        if hr_path.exists():
            subject = vlr_path.stem.split("_")[0]
            subject_images[subject].append(vlr_path)

    # Sample subjects with enough images
    valid_subjects = [s for s, imgs in subject_images.items() if len(imgs) >= 6]
    test_subjects = random.sample(
        valid_subjects, min(num_subjects, len(valid_subjects))
    )

    all_similarities = []
    correct_matches = 0
    total_probes = 0

    for subject in test_subjects:
        images = subject_images[subject]

        # Split: first 3 for gallery, rest for probes
        gallery_images = images[:3]
        probe_images = images[3:6]

        # Build gallery
        gallery_embeddings = []
        for vlr_path in gallery_images:
            hr_path = hr_root / vlr_path.name
            hr_img = Image.open(hr_path).convert("RGB")
            hr_tensor = (
                torch.from_numpy(np.array(hr_img)).permute(2, 0, 1).float() / 255.0
            )
            hr_tensor = (hr_tensor - 0.5) / 0.5
            hr_tensor = hr_tensor.unsqueeze(0).to(device)

            embedding = model(hr_tensor)
            gallery_embeddings.append(F.normalize(embedding, dim=1))

        gallery_mean = torch.mean(
            torch.cat(gallery_embeddings, dim=0), dim=0, keepdim=True
        )
        gallery_mean = F.normalize(gallery_mean, dim=1)

        # Test probes
        for vlr_path in probe_images:
            vlr_img = Image.open(vlr_path).convert("RGB")
            vlr_tensor = (
                torch.from_numpy(np.array(vlr_img)).permute(2, 0, 1).float() / 255.0
            )
            vlr_tensor = vlr_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                dsr_output = dsr_model(vlr_tensor)

            dsr_normalized = (dsr_output - 0.5) / 0.5
            probe_embedding = model(dsr_normalized)
            probe_norm = F.normalize(probe_embedding, dim=1)

            similarity = (gallery_mean * probe_norm).sum().item()
            all_similarities.append(similarity)

            if similarity >= 0.35:
                correct_matches += 1
            total_probes += 1

    avg_similarity = (
        sum(all_similarities) / len(all_similarities) if all_similarities else 0.0
    )
    accuracy = correct_matches / total_probes if total_probes > 0 else 0.0

    return avg_similarity, accuracy


def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning for EdgeFace")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Number of subjects per batch"
    )
    parser.add_argument("--images-per-subject", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()

    config = ContrastiveConfig(
        batch_size=args.batch_size,
        images_per_subject=args.images_per_subject,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        device=args.device,
    )

    print("=" * 70)
    print("CONTRASTIVE FINE-TUNING FOR EDGEFACE")
    print("=" * 70)
    print(
        f"Batch size: {config.batch_size} subjects × {config.images_per_subject} images"
    )
    print(f"Learning rate: {config.lr}")
    print(f"Temperature: {config.temperature}")
    print(f"Epochs: {config.epochs}")
    print(f"Device: {config.device}")

    device = torch.device(config.device)

    # Load DSR model
    print("\nLoading DSR model...")
    dsr_model = load_dsr_model(config.dsr_weights, device=device)
    dsr_model.eval()

    # Load EdgeFace pretrained
    print("Loading EdgeFace pretrained model...")
    model = EdgeFace(embedding_size=512, back="edgeface_xxs")

    # Load pretrained weights
    try:
        state_dict = torch.jit.load(
            str(config.edgeface_pretrained), map_location="cpu"
        ).state_dict()
    except Exception:
        state_dict = torch.load(config.edgeface_pretrained, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    # Clean key prefixes to match architecture exactly
    clean_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model.") :]
        if clean_key.startswith("backbone."):
            clean_key = clean_key[len("backbone.") :]
        clean_state_dict[clean_key] = value

    load_result = model.load_state_dict(clean_state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Failed to align pretrained EdgeFace weights. "
            f"Missing: {load_result.missing_keys}, "
            f"Unexpected: {load_result.unexpected_keys}"
        )

    model.to(device)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ContrastivePairDataset(
        config.dataset_root,
        dsr_model,
        device,
        config.images_per_subject,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_similarity = 0.0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_similarity = train_epoch(
            model, train_loader, optimizer, device, config.temperature
        )

        val_similarity, val_accuracy = validate_gallery_matching(
            model, config.val_dataset_root, dsr_model, device
        )

        print(f"\nEpoch {epoch:02d}/{config.epochs}")
        print(f"  Train: loss={train_loss:.4f}, DSR→HR sim={train_similarity:.4f}")
        print(
            f"  Val: gallery-probe sim={val_similarity:.4f}, accuracy={val_accuracy:.2%}"
        )

        if val_similarity > best_similarity:
            best_similarity = val_similarity
            print(f"  ✓ NEW BEST! Saving checkpoint...")

            checkpoint = {
                "epoch": epoch,
                "backbone_state_dict": model.state_dict(),
                "val_similarity": val_similarity,
                "val_accuracy": val_accuracy,
                "config": config,
            }
            torch.save(checkpoint, config.save_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation similarity: {best_similarity:.4f}")
    print(f"Saved to: {config.save_path}")


if __name__ == "__main__":
    import numpy as np

    main()
