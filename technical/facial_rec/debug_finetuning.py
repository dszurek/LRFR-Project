"""Debug script to diagnose fine-tuning issues.

This script tests individual components to find why accuracy stays at 0%.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

from technical.facial_rec.edgeface_weights.edgeface import EdgeFace
from technical.dsr import load_dsr_model


def test_edgeface_loading():
    """Test if EdgeFace loads correctly."""
    print("=" * 60)
    print("TEST 1: EdgeFace Model Loading")
    print("=" * 60)

    weights_path = Path("technical/facial_rec/edgeface_weights/edgeface_xxs.pt")
    model = EdgeFace(embedding_size=512, back="edgeface_xxs")

    state_dict = torch.load(weights_path, map_location="cpu")

    # Handle nested checkpoints (some have 'model' or 'state_dict' keys)
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        print(f"  Found nested 'model' key, extracting...")
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        print(f"  Found nested 'state_dict' key, extracting...")
        state_dict = state_dict["state_dict"]

    # Strip 'model.' prefix if present
    cleaned_state = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "", 1) if key.startswith("model.") else key
        cleaned_state[new_key] = value

    print(f"  Sample keys after cleaning: {list(cleaned_state.keys())[:3]}")

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)

    print(f"✓ Model created: {type(model)}")
    print(f"  Missing keys ({len(missing)}): {missing}")
    print(
        f"  Unexpected keys ({len(unexpected)}): {list(unexpected)[:5] if unexpected else 'None'}"
    )

    # Test forward pass
    test_input = torch.randn(1, 3, 112, 112)
    model.eval()
    with torch.no_grad():
        output = model(test_input)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.4f}")
    print(f"  Output std: {output.std().item():.4f}")
    print(f"  Output min/max: {output.min().item():.4f} / {output.max().item():.4f}")

    return model


def test_dsr_output():
    """Test DSR model output."""
    print("\n" + "=" * 60)
    print("TEST 2: DSR Model Output")
    print("=" * 60)

    dsr_path = Path("technical/dsr/dsr.pth")
    dsr_model = load_dsr_model(dsr_path, device="cpu")

    # Load a sample VLR image
    vlr_dir = Path("technical/dataset/frontal_only/train/vlr_images")
    sample_vlr = list(vlr_dir.glob("*.png"))[0]

    print(f"  Sample VLR: {sample_vlr.name}")

    vlr_img = Image.open(sample_vlr).convert("RGB")
    vlr_tensor = transforms.functional.to_tensor(vlr_img).unsqueeze(0)

    print(f"  VLR shape: {vlr_tensor.shape}")

    dsr_model.eval()
    with torch.no_grad():
        sr_output = dsr_model(vlr_tensor)
        sr_output = torch.clamp(sr_output, 0.0, 1.0)

    print(f"  SR output shape: {sr_output.shape}")
    print(f"  SR output mean: {sr_output.mean().item():.4f}")
    print(f"  SR output std: {sr_output.std().item():.4f}")

    return sr_output


def test_embeddings(edgeface_model, sr_tensor):
    """Test embedding extraction."""
    print("\n" + "=" * 60)
    print("TEST 3: Embedding Extraction")
    print("=" * 60)

    # Normalize for EdgeFace
    sr_normalized = transforms.functional.normalize(
        sr_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )

    edgeface_model.eval()
    with torch.no_grad():
        embedding = edgeface_model(sr_normalized)

    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding mean: {embedding.mean().item():.4f}")
    print(f"  Embedding std: {embedding.std().item():.4f}")
    print(f"  Embedding norm: {torch.norm(embedding).item():.4f}")

    # Check if embeddings are all zeros or NaN
    if torch.isnan(embedding).any():
        print(f"  ✗ WARNING: Embedding contains NaN!")
    elif (embedding == 0).all():
        print(f"  ✗ WARNING: Embedding is all zeros!")
    elif embedding.std().item() < 0.01:
        print(f"  ✗ WARNING: Embedding has very low variance!")
    else:
        print(f"  ✓ Embedding looks healthy")

    return embedding


def test_dataset_labels():
    """Test if dataset labels are reasonable."""
    print("\n" + "=" * 60)
    print("TEST 4: Dataset Labels")
    print("=" * 60)

    from technical.facial_rec.finetune_edgeface import DSROutputDataset
    from technical.dsr import load_dsr_model

    dsr_path = Path("technical/dsr/dsr.pth")
    dsr_model = load_dsr_model(dsr_path, device="cpu")

    train_dir = Path("technical/dataset/frontal_only/train")
    dataset = DSROutputDataset(train_dir, dsr_model, "cpu", augment=False)

    print(f"  Total samples: {len(dataset)}")
    print(f"  Total subjects: {len(dataset.subject_to_id)}")
    print(
        f"  Subject IDs range: {min(dataset.subject_to_id.values())} to {max(dataset.subject_to_id.values())}"
    )

    # Check a few samples
    print(f"\n  Sample subjects:")
    for subject, subject_id in list(dataset.subject_to_id.items())[:5]:
        print(f"    '{subject}' → ID {subject_id}")

    # Test loading one sample
    sr, hr, label = dataset[0]
    print(f"\n  Sample batch:")
    print(f"    SR shape: {sr.shape}")
    print(f"    HR shape: {hr.shape}")
    print(f"    Label: {label}")

    if label >= len(dataset.subject_to_id):
        print(
            f"    ✗ WARNING: Label {label} is out of range (max: {len(dataset.subject_to_id) - 1})"
        )
    else:
        print(f"    ✓ Label is valid")


def main():
    print("\n" + "=" * 70)
    print(" EdgeFace Fine-Tuning Debug Script")
    print("=" * 70)

    try:
        # Test 1: Model loading
        edgeface_model = test_edgeface_loading()

        # Test 2: DSR output
        sr_output = test_dsr_output()

        # Test 3: Embeddings
        test_embeddings(edgeface_model, sr_output)

        # Test 4: Dataset labels
        test_dataset_labels()

        print("\n" + "=" * 70)
        print(" All tests completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
