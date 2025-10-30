"""Extract clean backbone weights from training checkpoint."""

import torch
from pathlib import Path
from dataclasses import dataclass


# Stub for unpickling the finetuned checkpoint
@dataclass
class FinetuneConfig:
    """Stub for unpickling fine-tuned checkpoint."""

    pass


def extract_backbone(checkpoint_path: Path, output_path: Path) -> None:
    """Extract model weights from training checkpoint and save as clean state dict."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Check what keys are in the checkpoint
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Extract the backbone state dict
    if "backbone_state_dict" in checkpoint:
        state_dict = checkpoint["backbone_state_dict"]
        print("Found 'backbone_state_dict' key")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("Found 'state_dict' key")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Found 'model_state_dict' key")
    else:
        print("ERROR: No recognized state dict key found!")
        return

    # Clean up keys (remove 'model.' prefix if present)
    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model.") :]
        clean_state_dict[clean_key] = value

    print(f"Extracted {len(clean_state_dict)} weight tensors")
    print(f"Sample keys: {list(clean_state_dict.keys())[:5]}")

    # Save clean weights
    print(f"Saving to {output_path}...")
    torch.save(clean_state_dict, output_path)
    print("✅ Done!")

    # Verify file size
    original_size = checkpoint_path.stat().st_size / (1024 * 1024)
    new_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nOriginal: {original_size:.1f} MB")
    print(f"Extracted: {new_size:.1f} MB")
    print(f"Reduction: {((original_size - new_size) / original_size * 100):.1f}%")


if __name__ == "__main__":
    checkpoint_path = Path(
        "technical/facial_rec/edgeface_weights/edgeface_finetuned.pth"
    )
    output_path = Path(
        "technical/facial_rec/edgeface_weights/edgeface_finetuned_clean.pt"
    )

    extract_backbone(checkpoint_path, output_path)
