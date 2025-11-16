"""Test if pretrained model produces consistent embeddings for same image."""

from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from technical.pipeline.pipeline import build_pipeline, PipelineConfig


def test_same_image_consistency():
    """Load the SAME image twice and check if embeddings match."""

    print("=" * 70)
    print("TESTING EMBEDDING CONSISTENCY")
    print("=" * 70)

    config = PipelineConfig(
        device="cuda",
        edgeface_weights_path=Path(
            "technical/facial_rec/edgeface_weights/edgeface_xxs.pt"
        ),
    )
    pipeline = build_pipeline(config)

    # Test with subject 007
    hr_dir = Path("technical/dataset/frontal_only/test/hr_images")
    vlr_dir = Path("technical/dataset/frontal_only/test/vlr_images")

    test_file = "007_01_01_041_00_crop_128.png"
    hr_path = hr_dir / test_file
    vlr_path = vlr_dir / test_file

    print(f"\nTesting with: {test_file}")

    # Load HR twice
    def load_hr(path):
        pil = Image.open(path).convert("RGB")
        tensor = transforms.functional.to_tensor(pil).unsqueeze(0).to(pipeline.device)
        return pipeline.infer_embedding(tensor)

    hr_emb1 = load_hr(hr_path)
    hr_emb2 = load_hr(hr_path)

    hr_similarity = F.cosine_similarity(
        hr_emb1.unsqueeze(0), hr_emb2.unsqueeze(0)
    ).item()
    print(f"\nHR loaded twice:")
    print(f"  Similarity: {hr_similarity:.6f} (should be 1.0)")

    # Load VLR through DSR twice
    dsr_emb1 = pipeline.infer_embedding(pipeline.upscale(vlr_path))
    dsr_emb2 = pipeline.infer_embedding(pipeline.upscale(vlr_path))

    dsr_similarity = F.cosine_similarity(
        dsr_emb1.unsqueeze(0), dsr_emb2.unsqueeze(0)
    ).item()
    print(f"\nVLR→DSR loaded twice:")
    print(f"  Similarity: {dsr_similarity:.6f} (should be 1.0)")

    # HR vs DSR (same original image)
    hr_vs_dsr = F.cosine_similarity(hr_emb1.unsqueeze(0), dsr_emb1.unsqueeze(0)).item()
    print(f"\nHR vs VLR→DSR (same image):")
    print(f"  Similarity: {hr_vs_dsr:.6f} (should be ~0.99)")

    # Test a DIFFERENT image
    test_file2 = "007_01_01_041_01_crop_128.png"
    hr_path2 = hr_dir / test_file2
    hr_emb_different = load_hr(hr_path2)

    same_person_diff_image = F.cosine_similarity(
        hr_emb1.unsqueeze(0), hr_emb_different.unsqueeze(0)
    ).item()
    print(f"\nSame person, different image:")
    print(f"  {test_file} vs {test_file2}")
    print(f"  Similarity: {same_person_diff_image:.6f} (should be ~0.98-0.99)")

    # Test DIFFERENT person
    test_file3 = "010_01_01_041_00_crop_128.png"
    hr_path3 = hr_dir / test_file3
    hr_emb_different_person = load_hr(hr_path3)

    diff_person = F.cosine_similarity(
        hr_emb1.unsqueeze(0), hr_emb_different_person.unsqueeze(0)
    ).item()
    print(f"\nDifferent person:")
    print(f"  007 vs 010")
    print(f"  Similarity: {diff_person:.6f} (should be <0.5)")


if __name__ == "__main__":
    test_same_image_consistency()
