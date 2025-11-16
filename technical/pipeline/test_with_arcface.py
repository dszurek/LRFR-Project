"""Test if using ArcFace features improves matching."""

from pathlib import Path
import torch
import torch.nn.functional as F
from technical.pipeline.pipeline import build_pipeline, PipelineConfig
from dataclasses import dataclass
import sys


# Add stub for FinetuneConfig to handle pickle
@dataclass
class FinetuneConfig:
    pass


sys.modules["__main__"].FinetuneConfig = FinetuneConfig


def test_with_full_model():
    """Test embeddings WITH the ArcFace head applied."""

    # Load the full checkpoint
    ckpt_path = Path("technical/facial_rec/edgeface_weights/edgeface_finetuned.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("=" * 70)
    print("TESTING WITH ARCFACE HEAD")
    print("=" * 70)

    # Build pipeline normally (without ArcFace)
    config = PipelineConfig(device="cuda")
    pipeline = build_pipeline(config)

    # Get test images
    train_hr = Path("technical/dataset/edgeface_finetune/train/hr_images")
    val_vlr = Path("technical/dataset/edgeface_finetune/val/vlr_images")

    # Subject 001 images
    subject = "001"
    train_images = sorted([p for p in train_hr.glob(f"{subject}_*.png")])[:5]
    test_image = sorted([p for p in val_vlr.glob(f"{subject}_*.png")])[0]

    print(f"\nSubject: {subject}")
    print(f"Gallery images: {len(train_images)}")
    print(f"Test image: {test_image.name}")

    # Get embeddings from backbone only (current approach)
    print("\n--- BACKBONE ONLY (current approach) ---")

    from PIL import Image
    from torchvision import transforms

    def load_hr_tensor(image_path, device):
        pil = Image.open(image_path).convert("RGB")
        tensor = transforms.functional.to_tensor(pil)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device)

    gallery_embeddings = []
    for img_path in train_images:
        hr_tensor = load_hr_tensor(img_path, pipeline.device)
        embedding = pipeline.infer_embedding(hr_tensor)
        gallery_embeddings.append(embedding.cpu())

    gallery_mean = torch.mean(torch.stack(gallery_embeddings), dim=0)

    # Test with DSR
    sr_img = pipeline.upscale(test_image)
    probe_embedding = pipeline.infer_embedding(sr_img)

    # Compute similarity
    gallery_norm = F.normalize(gallery_mean.unsqueeze(0), dim=-1)
    probe_norm = F.normalize(probe_embedding.cpu().unsqueeze(0), dim=-1)
    similarity_backbone = torch.dot(gallery_norm.squeeze(), probe_norm.squeeze()).item()

    print(f"Cosine similarity: {similarity_backbone:.4f}")

    # Now try WITH ArcFace normalization (L2 normalize then apply weight matrix)
    print("\n--- WITH ARCFACE FEATURES ---")

    # ArcFace applies: F.normalize(embedding, dim=1) then F.linear(embedding, weight)
    # Let's check if the ArcFace weight matrix contains useful information

    arcface_weight = ckpt["arcface_state_dict"][
        "weight"
    ]  # Shape: [num_classes, embedding_dim]
    print(f"ArcFace weight shape: {list(arcface_weight.shape)}")

    # Get the class ID for subject 001
    subject_to_id = ckpt["subject_to_id"]
    if subject in subject_to_id:
        class_id = subject_to_id[subject]
        print(f"Subject {subject} -> Class ID: {class_id}")

        # Get the ArcFace weight vector for this class
        class_weight = arcface_weight[class_id]  # Shape: [embedding_dim]

        # Compute similarity using ArcFace features
        # ArcFace projects: embedding · weight_vector
        gallery_arcface_score = torch.dot(
            F.normalize(gallery_mean, dim=-1), F.normalize(class_weight, dim=-1)
        ).item()

        probe_arcface_score = torch.dot(
            F.normalize(probe_embedding.cpu(), dim=-1),
            F.normalize(class_weight, dim=-1),
        ).item()

        print(f"Gallery ArcFace score: {gallery_arcface_score:.4f}")
        print(f"Probe ArcFace score: {probe_arcface_score:.4f}")

        # Try using ArcFace features as embeddings
        gallery_arcface_features = F.normalize(
            gallery_mean.unsqueeze(0), dim=-1
        ) @ F.normalize(arcface_weight.T, dim=0)
        probe_arcface_features = F.normalize(
            probe_embedding.cpu().unsqueeze(0), dim=-1
        ) @ F.normalize(arcface_weight.T, dim=0)

        # Now compare using the class ID dimension
        similarity_arcface = F.cosine_similarity(
            gallery_arcface_features, probe_arcface_features
        ).item()

        print(f"Cosine similarity (ArcFace features): {similarity_arcface:.4f}")
    else:
        print(f"Subject {subject} not found in subject_to_id mapping")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"Backbone similarity: {similarity_backbone:.4f}")
    print(f"Threshold: 0.35")
    print(f"Match: {similarity_backbone >= 0.35}")


if __name__ == "__main__":
    test_with_full_model()
