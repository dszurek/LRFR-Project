"""Test pretrained EdgeFace model's actual performance."""

from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from technical.pipeline.pipeline import build_pipeline, PipelineConfig


def load_hr_tensor(image_path, device):
    pil = Image.open(image_path).convert("RGB")
    tensor = transforms.functional.to_tensor(pil)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def test_pretrained():
    """Test pretrained model on same setup as fine-tuned."""

    print("=" * 70)
    print("TESTING PRETRAINED EDGEFACE MODEL")
    print("=" * 70)

    # Build pipeline with PRETRAINED model
    config = PipelineConfig(
        device="cuda",
        edgeface_weights_path=Path(
            "technical/facial_rec/edgeface_weights/edgeface_xxs.pt"
        ),
    )
    pipeline = build_pipeline(config)

    # Get test images
    train_hr = Path("technical/dataset/edgeface_finetune/train/hr_images")
    val_vlr = Path("technical/dataset/edgeface_finetune/val/vlr_images")

    # Test subject 001
    subject = "001"
    train_images = sorted([p for p in train_hr.glob(f"{subject}_*.png")])[:5]
    test_images = sorted([p for p in val_vlr.glob(f"{subject}_*.png")])[:5]

    print(f"\nSubject: {subject}")
    print(f"Gallery images (HR): {len(train_images)}")
    print(f"Test images (VLR): {len(test_images)}")

    # Build gallery
    print("\n--- Building Gallery ---")
    gallery_embeddings = []
    for img_path in train_images:
        hr_tensor = load_hr_tensor(img_path, pipeline.device)
        embedding = pipeline.infer_embedding(hr_tensor)
        gallery_embeddings.append(embedding.cpu())
        print(f"  {img_path.name}: norm={torch.norm(embedding.cpu()).item():.4f}")

    gallery_mean = torch.mean(torch.stack(gallery_embeddings), dim=0)
    print(f"\nGallery mean embedding norm: {torch.norm(gallery_mean).item():.4f}")

    # Test with DSR outputs
    print("\n--- Testing with VLR→DSR Probes ---")
    similarities = []

    for test_img in test_images:
        sr_img = pipeline.upscale(test_img)
        probe_embedding = pipeline.infer_embedding(sr_img)

        # Compute similarity
        gallery_norm = F.normalize(gallery_mean.unsqueeze(0), dim=-1)
        probe_norm = F.normalize(probe_embedding.cpu().unsqueeze(0), dim=-1)
        similarity = torch.dot(gallery_norm.squeeze(), probe_norm.squeeze()).item()
        similarities.append(similarity)

        match = "✓" if similarity >= 0.35 else "✗"
        print(f"  {match} {test_img.name}: similarity={similarity:.4f}")

    avg_similarity = sum(similarities) / len(similarities)
    matches = sum(1 for s in similarities if s >= 0.35)

    print(f"\n--- Results ---")
    print(f"Average similarity: {avg_similarity:.4f}")
    print(
        f"Matches (>= 0.35): {matches}/{len(similarities)} ({matches/len(similarities)*100:.1f}%)"
    )
    print(f"Min similarity: {min(similarities):.4f}")
    print(f"Max similarity: {max(similarities):.4f}")

    # Test multiple subjects
    print("\n" + "=" * 70)
    print("TESTING MULTIPLE SUBJECTS")
    print("=" * 70)

    subjects = ["001", "003", "004", "008", "012"]
    all_similarities = []
    subject_results = []

    for subject in subjects:
        train_images = sorted([p for p in train_hr.glob(f"{subject}_*.png")])[:5]
        test_images = sorted([p for p in val_vlr.glob(f"{subject}_*.png")])[:3]

        if not train_images or not test_images:
            continue

        # Gallery
        gallery_embeddings = []
        for img_path in train_images:
            hr_tensor = load_hr_tensor(img_path, pipeline.device)
            embedding = pipeline.infer_embedding(hr_tensor)
            gallery_embeddings.append(embedding.cpu())
        gallery_mean = torch.mean(torch.stack(gallery_embeddings), dim=0)

        # Test
        subject_sims = []
        for test_img in test_images:
            sr_img = pipeline.upscale(test_img)
            probe_embedding = pipeline.infer_embedding(sr_img)

            gallery_norm = F.normalize(gallery_mean.unsqueeze(0), dim=-1)
            probe_norm = F.normalize(probe_embedding.cpu().unsqueeze(0), dim=-1)
            similarity = torch.dot(gallery_norm.squeeze(), probe_norm.squeeze()).item()
            subject_sims.append(similarity)
            all_similarities.append(similarity)

        avg_sim = sum(subject_sims) / len(subject_sims)
        matches = sum(1 for s in subject_sims if s >= 0.35)
        subject_results.append((subject, avg_sim, matches, len(subject_sims)))

        print(
            f"Subject {subject}: avg={avg_sim:.4f}, matches={matches}/{len(subject_sims)}"
        )

    print(f"\n--- Overall Statistics ---")
    overall_avg = sum(all_similarities) / len(all_similarities)
    overall_matches = sum(1 for s in all_similarities if s >= 0.35)
    print(f"Average similarity: {overall_avg:.4f}")
    print(
        f"Total matches (>= 0.35): {overall_matches}/{len(all_similarities)} ({overall_matches/len(all_similarities)*100:.1f}%)"
    )
    print(f"Min: {min(all_similarities):.4f}, Max: {max(all_similarities):.4f}")


if __name__ == "__main__":
    test_pretrained()
