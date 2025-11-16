"""Quick test to see what's in the gallery after enrollment."""

from pathlib import Path
from technical.pipeline.pipeline import build_pipeline, PipelineConfig


def test_gallery():
    config = PipelineConfig(device="cuda")
    pipeline = build_pipeline(config)

    print("Gallery size before enrollment:", pipeline.gallery.size)

    # Enroll one subject manually
    train_hr = Path("technical/dataset/edgeface_finetune/train/hr_images")
    subject_001_images = sorted(train_hr.glob("001_*.png"))[:5]

    print(f"\nEnrolling subject 001 with {len(subject_001_images)} images")

    embeddings = []
    for img in subject_001_images:
        from PIL import Image
        from torchvision import transforms

        pil = Image.open(img).convert("RGB")
        tensor = transforms.functional.to_tensor(pil).unsqueeze(0).to("cuda")
        embedding = pipeline.infer_embedding(tensor)
        embeddings.append(embedding.cpu())
        print(f"  {img.name}: embedding shape {embedding.shape}")

    import torch

    mean_emb = torch.mean(torch.stack(embeddings), dim=0)
    pipeline.gallery.add("001", mean_emb)

    print(f"\nGallery size after enrollment: {pipeline.gallery.size}")
    print(f"Gallery labels: {pipeline.gallery._labels}")

    # Now test lookup
    test_img = Path(
        "technical/dataset/edgeface_finetune/val/vlr_images/001_01_01_041_04_crop_128.png"
    )
    print(f"\nTesting with VLR image: {test_img.name}")

    result = pipeline.run(test_img)
    print(f"\nResult: identity={result['identity']}, score={result['score']}")

    # Manual similarity check
    probe_emb = result["embedding"]
    gallery_emb = pipeline.gallery._embeddings[0]  # Subject 001

    import torch.nn.functional as F

    probe_norm = F.normalize(probe_emb.unsqueeze(0), dim=-1)
    gallery_norm = F.normalize(gallery_emb.unsqueeze(0), dim=-1)
    manual_sim = torch.dot(probe_norm.squeeze(), gallery_norm.squeeze()).item()

    print(f"Manual similarity check:")
    print(f"  Gallery embedding norm: {torch.norm(gallery_emb).item():.4f}")
    print(f"  Probe embedding norm: {torch.norm(probe_emb).item():.4f}")
    print(f"  Cosine similarity: {manual_sim:.4f}")
    print(f"  Threshold: {pipeline.gallery.threshold}")
    print(f"  Match: {manual_sim >= pipeline.gallery.threshold}")


if __name__ == "__main__":
    test_gallery()
