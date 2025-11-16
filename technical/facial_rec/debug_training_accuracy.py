"""Debug why training/validation accuracy is 0%.

Simpler version - just check if predictions are always the same class.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from technical.facial_rec.finetune_edgeface import ArcFaceLoss


def diagnose_predictions():
    """Check what the model is actually predicting."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading DSR model...")
    config = DSRConfig()
    dsr_model = DSRColor(config).to(device)
    dsr_ckpt = torch.load("technical/dsr/dsr.pth", map_location=device)
    dsr_model.load_state_dict(dsr_ckpt["model_state_dict"])
    dsr_model.eval()

    print("Loading EdgeFace backbone...")
    backbone = load_edgeface_backbone(
        Path("technical/facial_rec/edgeface_weights/edgeface_xxs.pt"),
        device,
        backbone_type="edgeface_xxs",
    )
    backbone.eval()

    # Load training dataset
    print("Loading training dataset...")
    train_dataset = DSROutputDataset(
        Path("technical/dataset/edgeface_finetune/train"),
        dsr_model,
        device,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    num_classes = len(train_dataset.subject_to_id)
    print(f"Number of classes: {num_classes}")

    # Create ArcFace head
    print("Creating ArcFace head...")
    arcface = ArcFaceLoss(
        embedding_size=512,
        num_classes=num_classes,
        scale=32.0,
        margin=0.3,
    ).to(device)

    # Check one batch
    print("\n" + "=" * 60)
    print("Analyzing first batch...")
    print("=" * 60)

    sr_imgs, hr_imgs, labels = next(iter(train_loader))
    sr_imgs = sr_imgs.to(device)
    hr_imgs = hr_imgs.to(device)
    labels = labels.to(device)

    print(f"Batch size: {sr_imgs.shape[0]}")
    print(f"Label range: {labels.min().item()} - {labels.max().item()}")
    print(f"Unique labels in batch: {labels.unique().numel()}")
    print(f"Labels: {labels.cpu().numpy()}")

    with torch.no_grad():
        # Get embeddings
        sr_embeddings = backbone(sr_imgs)
        print(f"\nEmbedding shape: {sr_embeddings.shape}")
        print(f"Embedding mean: {sr_embeddings.mean().item():.4f}")
        print(f"Embedding std: {sr_embeddings.std().item():.4f}")
        print(f"Embedding min: {sr_embeddings.min().item():.4f}")
        print(f"Embedding max: {sr_embeddings.max().item():.4f}")

        # Check if embeddings are all the same (collapsed)
        embedding_variance = sr_embeddings.var(dim=0).mean().item()
        print(f"Embedding variance (avg across features): {embedding_variance:.6f}")
        if embedding_variance < 0.001:
            print("⚠️  WARNING: Embeddings have very low variance (might be collapsed)")

        # Get ArcFace logits (without labels for testing)
        # We need to manually compute: W * x (without margin)
        weight = F.normalize(arcface.weight, dim=1)
        embeddings_norm = F.normalize(sr_embeddings, dim=1)
        cosine = F.linear(embeddings_norm, weight)

        print(f"\nCosine similarity shape: {cosine.shape}")
        print(f"Cosine mean: {cosine.mean().item():.4f}")
        print(f"Cosine std: {cosine.std().item():.4f}")
        print(f"Cosine min: {cosine.min().item():.4f}")
        print(f"Cosine max: {cosine.max().item():.4f}")

        # Get predictions
        logits = arcface(sr_embeddings, labels)
        pred = logits.argmax(dim=1)

        print(f"\nPredictions: {pred.cpu().numpy()}")
        print(f"Prediction range: {pred.min().item()} - {pred.max().item()}")
        print(f"Unique predictions: {pred.unique().numel()}")

        # Check if all predictions are the same
        if pred.unique().numel() == 1:
            print(f"⚠️  WARNING: All predictions are the same class: {pred[0].item()}")

        # Calculate accuracy
        correct = (pred == labels).sum().item()
        accuracy = correct / labels.size(0)
        print(f"\nAccuracy: {correct}/{labels.size(0)} = {accuracy:.4f}")

        # Show some examples
        print("\nFirst 10 examples:")
        print("Label | Prediction | Correct?")
        print("-" * 35)
        for i in range(min(10, labels.size(0))):
            correct_symbol = "✓" if pred[i] == labels[i] else "✗"
            print(f"{labels[i].item():5d} | {pred[i].item():10d} | {correct_symbol}")


if __name__ == "__main__":
    diagnose_predictions()
