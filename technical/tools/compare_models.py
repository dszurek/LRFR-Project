"""Compare old vs new DSR model outputs side-by-side."""

import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from torchvision import transforms

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from technical.dsr.models import load_dsr_model


def compare_checkpoints(
    old_ckpt_path: str, new_ckpt_path: str, test_image_path: str, device: str = "cuda"
) -> None:
    """Load two checkpoints and compare their SR outputs on a test image."""

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load models
    print(f"Loading old model from {old_ckpt_path}...")
    old_model = load_dsr_model(old_ckpt_path, device=device)
    old_model.eval()
    print(
        f"  Config: base_channels={old_model.config.base_channels}, residual_blocks={old_model.config.residual_blocks}"
    )

    print(f"\nLoading new model from {new_ckpt_path}...")
    new_model = load_dsr_model(new_ckpt_path, device=device)
    new_model.eval()
    print(
        f"  Config: base_channels={new_model.config.base_channels}, residual_blocks={new_model.config.residual_blocks}"
    )

    # Load test image
    print(f"\nLoading test image: {test_image_path}")
    img_bgr = cv2.imread(test_image_path)
    if img_bgr is None:
        print(f"ERROR: Could not load image from {test_image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_rgb).unsqueeze(0).to(device)

    # Generate SR outputs
    print("Generating SR outputs...")
    with torch.no_grad():
        old_sr = old_model(img_tensor).squeeze(0).cpu().clamp(0, 1)
        new_sr = new_model(img_tensor).squeeze(0).cpu().clamp(0, 1)

    # Convert to numpy
    old_sr_np = (old_sr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    new_sr_np = (new_sr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Upscale input for fair comparison
    input_upscaled = cv2.resize(
        img_bgr,
        (old_sr_np.shape[1], old_sr_np.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Convert back to BGR for OpenCV display
    old_sr_bgr = cv2.cvtColor(old_sr_np, cv2.COLOR_RGB2BGR)
    new_sr_bgr = cv2.cvtColor(new_sr_np, cv2.COLOR_RGB2BGR)

    # Create comparison
    comparison = np.concatenate([input_upscaled, old_sr_bgr, new_sr_bgr], axis=1)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Input (LR)", (15, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(
        comparison,
        "Old Model",
        (15 + input_upscaled.shape[1], 25),
        font,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        comparison,
        "New Model",
        (15 + input_upscaled.shape[1] + old_sr_bgr.shape[1], 25),
        font,
        0.7,
        (255, 255, 255),
        2,
    )

    # Compute PSNR (if ground truth available)
    hr_path = (
        Path(test_image_path).parent.parent / "hr_images" / Path(test_image_path).name
    )
    if hr_path.exists():
        hr_bgr = cv2.imread(str(hr_path))
        if hr_bgr is not None:
            hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            old_sr_f = old_sr_np.astype(np.float32) / 255.0
            new_sr_f = new_sr_np.astype(np.float32) / 255.0

            old_mse = np.mean((old_sr_f - hr_rgb) ** 2)
            new_mse = np.mean((new_sr_f - hr_rgb) ** 2)

            old_psnr = 10 * np.log10(1.0 / (old_mse + 1e-10))
            new_psnr = 10 * np.log10(1.0 / (new_mse + 1e-10))

            print(f"\nPSNR Comparison:")
            print(f"  Old model: {old_psnr:.2f} dB")
            print(f"  New model: {new_psnr:.2f} dB")
            print(f"  Improvement: {new_psnr - old_psnr:+.2f} dB")

    # Display
    print("\nDisplaying comparison (press any key to close)...")
    cv2.imshow("Model Comparison: Input | Old | New", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python compare_models.py <old_checkpoint> <new_checkpoint> <test_image>"
        )
        print("\nExample:")
        print("  poetry run python tools/compare_models.py \\")
        print("      dsr/dsr_old.pth \\")
        print("      dsr/dsr.pth \\")
        print("      dataset/test_processed/vlr_images/007_01_01_010_00_crop_128.png")
        sys.exit(1)

    compare_checkpoints(sys.argv[1], sys.argv[2], sys.argv[3])
