"""Diagnostic script to debug low evaluation accuracy.

Checks:
1. Subject ID extraction consistency
2. Gallery enrollment correctness
3. Embedding similarity distributions
4. Threshold sensitivity
"""

from pathlib import Path
import torch
import torch.nn.functional as F
from collections import Counter
from technical.pipeline.pipeline import build_pipeline, PipelineConfig


def check_subject_ids():
    """Verify subject IDs are extracted consistently."""
    print("=" * 70)
    print("CHECKING SUBJECT ID EXTRACTION")
    print("=" * 70)

    train_hr = Path("technical/dataset/edgeface_finetune/train/hr_images")
    val_vlr = Path("technical/dataset/edgeface_finetune/val/vlr_images")

    # Sample files
    train_samples = sorted(train_hr.glob("*.png"))[:10]
    val_samples = sorted(val_vlr.glob("*.png"))[:10]

    print("\nTraining HR samples:")
    for p in train_samples:
        subject = p.stem.split("_")[0]
        print(f"  {p.name} → Subject: {subject}")

    print("\nValidation VLR samples:")
    for p in val_samples:
        subject = p.stem.split("_")[0]
        print(f"  {p.name} → Subject: {subject}")

    # Check overlap
    train_subjects = set(p.stem.split("_")[0] for p in train_hr.glob("*.png"))
    val_subjects = set(p.stem.split("_")[0] for p in val_vlr.glob("*.png"))
    overlap = train_subjects & val_subjects

    print(f"\n Train subjects: {len(train_subjects)}")
    print(f"  Val subjects: {len(val_subjects)}")
    print(f"  Overlap: {len(overlap)} ({len(overlap)/len(val_subjects)*100:.1f}%)")

    if len(overlap) != len(val_subjects):
        print("  ⚠️  WARNING: Not all val subjects are in training!")
        missing = val_subjects - train_subjects
        print(f"  Missing: {sorted(list(missing))[:20]}")


def check_embeddings():
    """Check embedding similarity distributions."""
    print("\n" + "=" * 70)
    print("CHECKING EMBEDDING SIMILARITIES")
    print("=" * 70)

    config = PipelineConfig(
        device="cuda",
        edgeface_weights_path=Path(
            "technical/facial_rec/edgeface_weights/edgeface_finetuned.pth"
        ),
    )
    pipeline = build_pipeline(config)

    # Pick a subject with multiple images
    train_hr = Path("technical/dataset/edgeface_finetune/train/hr_images")
    val_vlr = Path("technical/dataset/edgeface_finetune/val/vlr_images")

    # Count images per subject
    subject_counts = Counter()
    for p in train_hr.glob("*.png"):
        subject = p.stem.split("_")[0]
        subject_counts[subject] += 1

    # Pick subject with most images
    top_subject, count = subject_counts.most_common(1)[0]
    print(f"\nTesting subject '{top_subject}' ({count} training images)")

    # Get embeddings from training images (gallery)
    train_images = [p for p in train_hr.glob(f"{top_subject}_*.png")]
    print(f"Found {len(train_images)} training images")

    gallery_embeddings = []
    for img_path in train_images[:5]:  # Use first 5 for gallery
        hr_tensor = pipeline._load_image_tensor(img_path)
        embedding = pipeline.infer_embedding(hr_tensor)
        gallery_embeddings.append(embedding.cpu())

    gallery_mean = torch.mean(torch.stack(gallery_embeddings), dim=0)

    # Get embeddings from validation images (probes)
    val_images = [p for p in val_vlr.glob(f"{top_subject}_*.png")]
    print(f"Found {len(val_images)} validation images")

    if not val_images:
        print("  ⚠️  No validation images for this subject!")
        return

    print(f"\nSimilarities (gallery mean vs validation probes):")
    for img_path in val_images[:10]:
        # Run through DSR
        sr_img = pipeline.upscale(img_path)
        probe_embedding = pipeline.infer_embedding(sr_img)

        # Compute similarity
        similarity = F.cosine_similarity(
            gallery_mean.unsqueeze(0), probe_embedding.cpu().unsqueeze(0)
        ).item()

        print(f"  {img_path.name}: {similarity:.4f}")

    # Check similarity to OTHER subjects
    print(f"\nSimilarities to OTHER subjects (should be lower):")
    other_subjects = [s for s, _ in subject_counts.most_common(10) if s != top_subject]

    for other_subject in other_subjects[:5]:
        other_images = [p for p in train_hr.glob(f"{other_subject}_*.png")]
        if not other_images:
            continue

        other_hr_tensor = pipeline._load_image_tensor(other_images[0])
        other_embedding = pipeline.infer_embedding(other_hr_tensor)

        similarity = F.cosine_similarity(
            gallery_mean.unsqueeze(0), other_embedding.cpu().unsqueeze(0)
        ).item()

        print(f"  Subject {other_subject}: {similarity:.4f}")


if __name__ == "__main__":
    check_subject_ids()
    check_embeddings()
