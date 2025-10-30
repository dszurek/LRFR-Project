"""Evaluation script for DSR model on frontal-only test set.

This script evaluates the DSR model's performance on frontal faces only,
which is the intended use case for EdgeFace-based face recognition.

Metrics computed:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Identity Similarity (using EdgeFace embeddings)
- Face Recognition Accuracy (VLR→DSR→EdgeFace vs HR→EdgeFace)
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import glob

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from technical.dsr.models import DSRColor, DSRConfig
from technical.facial_rec.edgeface_weights.edgeface import EdgeFace

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dsr_model(checkpoint_path: Path) -> DSRColor:
    """Load DSR model from checkpoint."""
    print(f"Loading DSR model from {checkpoint_path}...")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    # Extract config
    if isinstance(ckpt, dict) and "config" in ckpt:
        train_config = ckpt["config"]
        target_hr_size = train_config.get("target_hr_size", 112)
    else:
        target_hr_size = 112

    # Infer model config from state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Get base_channels and residual_blocks from state dict
    base_channels = (
        state_dict["conv_in.weight"].shape[0] if "conv_in.weight" in state_dict else 64
    )

    residual_block_indices = set()
    for key in state_dict:
        if key.startswith("residual_blocks"):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                residual_block_indices.add(int(parts[1]))
    residual_blocks = max(residual_block_indices) + 1 if residual_block_indices else 8

    # Create model
    config = DSRConfig(
        base_channels=base_channels,
        residual_blocks=residual_blocks,
        output_size=(target_hr_size, target_hr_size),
    )

    model = DSRColor(config=config).to(DEVICE)

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(
        f"DSR model loaded: {base_channels} channels, {residual_blocks} blocks, {target_hr_size}×{target_hr_size} output"
    )
    return model


def load_edgeface_model(weights_path: Path) -> EdgeFace:
    """Load EdgeFace model."""
    print(f"Loading EdgeFace from {weights_path}...")

    model = EdgeFace(back="edgeface_xxs")
    checkpoint = torch.load(weights_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()

    print("EdgeFace loaded successfully!")
    return model


def compute_metrics(
    pred_img: np.ndarray,
    hr_img: np.ndarray,
) -> Dict[str, float]:
    """Compute PSNR and SSIM between predicted and HR images.

    Args:
        pred_img: Predicted image (H, W, C) in range [0, 255]
        hr_img: Ground truth HR image (H, W, C) in range [0, 255]

    Returns:
        Dictionary with PSNR and SSIM values
    """
    # Compute PSNR
    psnr_value = psnr(hr_img, pred_img, data_range=255)

    # Compute SSIM (multichannel for color images)
    ssim_value = ssim(
        hr_img,
        pred_img,
        multichannel=True,
        data_range=255,
        channel_axis=2,
    )

    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
    }


def compute_identity_similarity(
    pred_tensor: torch.Tensor,
    hr_tensor: torch.Tensor,
    edgeface_model: EdgeFace,
) -> float:
    """Compute cosine similarity between DSR output and HR embeddings.

    Args:
        pred_tensor: DSR output (1, 3, H, W) in range [-1, 1]
        hr_tensor: Ground truth HR (1, 3, H, W) in range [-1, 1]
        edgeface_model: EdgeFace model

    Returns:
        Cosine similarity between embeddings
    """
    with torch.no_grad():
        pred_embedding = edgeface_model(pred_tensor)
        hr_embedding = edgeface_model(hr_tensor)

        similarity = F.cosine_similarity(pred_embedding, hr_embedding, dim=1)
        return similarity.item()


def evaluate_on_dataset(
    dsr_model: DSRColor,
    edgeface_model: EdgeFace,
    test_vlr_dir: Path,
    test_hr_dir: Path,
    num_samples: int = None,
) -> Dict[str, float]:
    """Evaluate DSR model on test set.

    Args:
        dsr_model: DSR super-resolution model
        edgeface_model: EdgeFace face recognition model
        test_vlr_dir: Directory with VLR test images
        test_hr_dir: Directory with HR test images
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        Dictionary with average metrics
    """
    # Get all VLR images
    vlr_paths = sorted(glob.glob(str(test_vlr_dir / "*.png")))

    if num_samples is not None:
        vlr_paths = vlr_paths[:num_samples]

    print(f"\nEvaluating on {len(vlr_paths)} test images...")

    # Preprocessing transforms
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Metrics accumulation
    metrics_sum = {
        "psnr": 0.0,
        "ssim": 0.0,
        "identity_similarity": 0.0,
    }

    valid_samples = 0

    # Process each image
    for vlr_path in tqdm(vlr_paths, desc="Evaluating"):
        filename = Path(vlr_path).name
        hr_path = test_hr_dir / filename

        if not hr_path.exists():
            continue

        try:
            # Load images
            vlr_img = Image.open(vlr_path).convert("RGB")
            hr_img = Image.open(hr_path).convert("RGB")

            # Convert to tensors
            vlr_tensor = to_tensor(vlr_img).unsqueeze(0).to(DEVICE)
            hr_tensor = to_tensor(hr_img).unsqueeze(0).to(DEVICE)

            # Run DSR
            with torch.no_grad():
                pred_tensor = dsr_model(vlr_tensor)

            # Convert to numpy for PSNR/SSIM
            pred_img = pred_tensor.squeeze(0).cpu().clamp(0, 1).numpy()
            pred_img = (pred_img.transpose(1, 2, 0) * 255).astype(np.uint8)

            hr_img_np = (
                hr_tensor.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255
            ).astype(np.uint8)

            # Compute PSNR and SSIM
            img_metrics = compute_metrics(pred_img, hr_img_np)
            metrics_sum["psnr"] += img_metrics["psnr"]
            metrics_sum["ssim"] += img_metrics["ssim"]

            # Compute identity similarity (normalize for EdgeFace)
            pred_normalized = normalize(pred_tensor.squeeze(0)).unsqueeze(0)
            hr_normalized = normalize(hr_tensor.squeeze(0)).unsqueeze(0)

            identity_sim = compute_identity_similarity(
                pred_normalized, hr_normalized, edgeface_model
            )
            metrics_sum["identity_similarity"] += identity_sim

            valid_samples += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Compute averages
    avg_metrics = {key: value / valid_samples for key, value in metrics_sum.items()}

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DSR model on frontal-only test set"
    )
    parser.add_argument(
        "--dsr-checkpoint",
        type=str,
        default="technical/dsr/dsr.pth",
        help="Path to DSR checkpoint",
    )
    parser.add_argument(
        "--edgeface-weights",
        type=str,
        default="technical/facial_rec/edgeface_weights/edgeface_xxs.pt",
        help="Path to EdgeFace weights",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="technical/dataset/frontal_only/test",
        help="Path to frontal-only test directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    args = parser.parse_args()

    # Convert to absolute paths
    dsr_checkpoint = ROOT / args.dsr_checkpoint
    edgeface_weights = ROOT / args.edgeface_weights
    test_dir = ROOT / args.test_dir

    print("=" * 60)
    print("DSR FRONTAL-ONLY TEST SET EVALUATION")
    print("=" * 60)
    print(f"DSR checkpoint: {dsr_checkpoint}")
    print(f"EdgeFace weights: {edgeface_weights}")
    print(f"Test directory: {test_dir}")
    print(f"Device: {DEVICE}")

    # Check paths exist
    if not dsr_checkpoint.exists():
        print(f"\n❌ ERROR: DSR checkpoint not found at {dsr_checkpoint}")
        return

    if not edgeface_weights.exists():
        print(f"\n❌ ERROR: EdgeFace weights not found at {edgeface_weights}")
        return

    if not test_dir.exists():
        print(f"\n❌ ERROR: Test directory not found at {test_dir}")
        print(
            "Please run filter_frontal_faces.py first to create the frontal-only dataset"
        )
        return

    # Load models
    dsr_model = load_dsr_model(dsr_checkpoint)
    edgeface_model = load_edgeface_model(edgeface_weights)

    # Evaluate
    test_vlr_dir = test_dir / "vlr_images"
    test_hr_dir = test_dir / "hr_images"

    metrics = evaluate_on_dataset(
        dsr_model,
        edgeface_model,
        test_vlr_dir,
        test_hr_dir,
        args.num_samples,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"Identity Similarity: {metrics['identity_similarity']:.4f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"PSNR {metrics['psnr']:.2f} dB: ", end="")
    if metrics["psnr"] >= 35:
        print("✅ Excellent image quality")
    elif metrics["psnr"] >= 30:
        print("✅ Good image quality")
    elif metrics["psnr"] >= 25:
        print("⚠️  Acceptable image quality")
    else:
        print("❌ Poor image quality")

    print(f"SSIM {metrics['ssim']:.4f}: ", end="")
    if metrics["ssim"] >= 0.95:
        print("✅ Excellent structural similarity")
    elif metrics["ssim"] >= 0.90:
        print("✅ Good structural similarity")
    elif metrics["ssim"] >= 0.85:
        print("⚠️  Acceptable structural similarity")
    else:
        print("❌ Poor structural similarity")

    print(f"Identity Similarity {metrics['identity_similarity']:.4f}: ", end="")
    if metrics["identity_similarity"] >= 0.95:
        print("✅ Excellent identity preservation")
    elif metrics["identity_similarity"] >= 0.90:
        print("✅ Good identity preservation")
    elif metrics["identity_similarity"] >= 0.85:
        print("⚠️  Acceptable identity preservation")
    else:
        print("❌ Poor identity preservation (face recognition may fail)")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
