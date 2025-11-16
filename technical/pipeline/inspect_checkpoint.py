"""Check what's inside the fine-tuned checkpoint."""

import torch
from pathlib import Path
from dataclasses import dataclass
import sys


# Add stub for FinetuneConfig to handle pickle
@dataclass
class FinetuneConfig:
    pass


# Make it available in __main__
sys.modules["__main__"].FinetuneConfig = FinetuneConfig


def inspect_checkpoint():
    print("=" * 70)
    print("CHECKPOINT INSPECTION")
    print("=" * 70)

    ckpt_path = Path("technical/facial_rec/edgeface_weights/edgeface_finetuned.pth")

    print(f"\nLoading: {ckpt_path}")

    # Load checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    print(f"\nTop-level keys in checkpoint:")
    for key in ckpt.keys():
        if isinstance(ckpt[key], dict):
            print(f"  {key}: dict with {len(ckpt[key])} items")
        elif isinstance(ckpt[key], torch.Tensor):
            print(f"  {key}: Tensor {list(ckpt[key].shape)}")
        else:
            print(f"  {key}: {type(ckpt[key]).__name__}")

    # Check backbone_state_dict
    if "backbone_state_dict" in ckpt:
        print(f"\nBackbone state dict keys (first 10):")
        backbone_keys = list(ckpt["backbone_state_dict"].keys())[:10]
        for key in backbone_keys:
            shape = list(ckpt["backbone_state_dict"][key].shape)
            print(f"  {key}: {shape}")
        print(f"  ... ({len(ckpt['backbone_state_dict'])} total keys)")

    # Check head_state_dict
    if "head_state_dict" in ckpt:
        print(f"\nHead state dict keys (first 10):")
        head_keys = list(ckpt["head_state_dict"].keys())[:10]
        for key in head_keys:
            shape = list(ckpt["head_state_dict"][key].shape)
            print(f"  {key}: {shape}")
        print(f"  ... ({len(ckpt['head_state_dict'])} total keys)")

        # Check weight shape to confirm number of classes
        if "weight" in ckpt["head_state_dict"]:
            weight_shape = ckpt["head_state_dict"]["weight"].shape
            print(f"\n  ArcFace weight shape: {list(weight_shape)}")
            print(f"  -> {weight_shape[0]} classes (should be 518)")

    # Check training metadata
    if "epoch" in ckpt:
        print(f"\nTraining metadata:")
        print(f"  Epoch: {ckpt['epoch']}")
    if "stage" in ckpt:
        print(f"  Stage: {ckpt['stage']}")
    if "config" in ckpt:
        print(f"  Config: {ckpt['config']}")


if __name__ == "__main__":
    inspect_checkpoint()
