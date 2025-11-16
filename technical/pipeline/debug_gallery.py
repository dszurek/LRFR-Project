"""Debug gallery to see what's stored."""

from pathlib import Path
from technical.pipeline.pipeline import build_pipeline, PipelineConfig
from collections import Counter


def debug_gallery():
    print("=" * 70)
    print("DEBUGGING GALLERY")
    print("=" * 70)

    # Build pipeline
    config = PipelineConfig(
        device="cuda",
        edgeface_weights_path=Path(
            "technical/facial_rec/edgeface_weights/edgeface_xxs.pt"
        ),
    )
    pipeline = build_pipeline(config)

    # Manually add some subjects
    test_root = Path("technical/dataset/frontal_only/test")
    hr_dir = test_root / "hr_images"

    print("\nAdding first 10 subjects...")
    subjects_added = []

    from PIL import Image
    from torchvision import transforms
    import torch

    for img_path in sorted(hr_dir.glob("*.png"))[:30]:
        subject = img_path.stem.split("_")[0]

        if subject not in subjects_added:
            # Load HR image
            pil = Image.open(img_path).convert("RGB")
            tensor = (
                transforms.functional.to_tensor(pil).unsqueeze(0).to(pipeline.device)
            )
            embedding = pipeline.infer_embedding(tensor)

            pipeline.gallery.add(subject, embedding)
            subjects_added.append(subject)
            print(f"  Added subject {subject}")

            if len(subjects_added) >= 10:
                break

    print(f"\nGallery size: {pipeline.gallery.size}")
    print(f"Gallery labels: {pipeline.gallery._labels}")

    # Now test a probe
    vlr_dir = test_root / "vlr_images"
    test_img = sorted(vlr_dir.glob("007_*.png"))[0]

    print(f"\nTesting with: {test_img.name}")
    result = pipeline.run(test_img)

    print(f"Result:")
    print(f"  Identity: {result['identity']}")
    print(f"  Score: {result['score']}")

    # Check what's in embeddings
    print(f"\nEmbedding analysis:")
    print(f"  Number of embeddings: {len(pipeline.gallery._embeddings)}")
    print(f"  Number of labels: {len(pipeline.gallery._labels)}")

    # Manual lookup
    probe_embedding = result["embedding"]
    import torch.nn.functional as F

    query = probe_embedding.cpu()
    query_norm = F.normalize(query.unsqueeze(0), dim=-1)

    stacked = torch.stack(pipeline.gallery._embeddings)
    stacked_norm = F.normalize(stacked, dim=-1)

    similarities = torch.mv(stacked_norm, query_norm.squeeze())

    print(f"\nSimilarities to gallery:")
    for i, (label, sim) in enumerate(zip(pipeline.gallery._labels, similarities)):
        print(f"  {label}: {sim.item():.4f}")


if __name__ == "__main__":
    debug_gallery()
