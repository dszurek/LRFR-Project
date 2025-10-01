import sys
from pathlib import Path
import os
import glob
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import random
from typing import Optional

# Make project root importable when running the script directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the canonical model
try:
    from technical.dsr.models import (
        DSRColor,
        DSRConfig,
    )  # adjust if your model lives elsewhere
except Exception:
    # fallback: try package relative import (if you run as module this won't be used)
    from .models import DSRColor, DSRConfig  # type: ignore

# ---------- Config ----------
MODEL_PATH = (
    Path(__file__).resolve().parent / "dsr.pth"
)  # matches technical/dsr/dsr.pth
# fallback dataset locations (try project-style and legacy-style)
CANDIDATE_LR_DIRS = [
    ROOT / "technical" / "dataset" / "test_processed" / "vlr_images",
    ROOT / "technical" / "dataset" / "test_processed" / "clr_images",
    ROOT / "dataset" / "test_processed" / "vlr_images",
]
HR_DIR = ROOT / "technical" / "dataset" / "test_processed" / "hr_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_lr_dir() -> Optional[Path]:
    for p in CANDIDATE_LR_DIRS:
        if p.exists():
            return p
    return None


def _infer_dsr_config(state: dict[str, torch.Tensor]) -> DSRConfig:
    """Infer DSRConfig fields directly from checkpoint tensor shapes."""

    base_channels = 64
    if "conv_in.weight" in state and isinstance(state["conv_in.weight"], torch.Tensor):
        base_channels = state["conv_in.weight"].shape[0]

    residual_block_indices = set()
    for key in state:
        if key.startswith("residual_blocks"):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                residual_block_indices.add(int(parts[1]))

    residual_blocks = max(residual_block_indices) + 1 if residual_block_indices else 8

    return DSRConfig(base_channels=base_channels, residual_blocks=residual_blocks)


def robust_load_checkpoint(path: Path, device: torch.device):
    """Load checkpoint at `path`, build model using inferred config, and load matching params."""
    ckpt = torch.load(path, map_location=device)

    # Extract state dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        if hasattr(ckpt, "state_dict"):
            state = ckpt.state_dict()
        else:
            raise RuntimeError(f"Unsupported checkpoint object: {type(ckpt)}")

    # Instantiate model with matching config
    config = _infer_dsr_config(state)
    model = DSRColor(config=config).to(device)

    # Clean prefixes
    cleaned = {}
    for k, v in state.items():
        nk = k
        for p in ("module.", "model."):
            if nk.startswith(p):
                nk = nk[len(p) :]
        cleaned[nk] = v

    # Match shapes and only copy compatible tensors
    model_state = model.state_dict()
    matched = {}
    mismatches = []
    for k, v in cleaned.items():
        if k in model_state:
            if isinstance(v, torch.Tensor) and v.shape == model_state[k].shape:
                matched[k] = v
            else:
                mismatches.append((k, getattr(v, "shape", None), model_state[k].shape))
        else:
            mismatches.append((k, getattr(v, "shape", None), None))

    # Update and load
    updated = {**model_state}
    for k, v in matched.items():
        updated[k] = v
    model.load_state_dict(updated)
    print(f"[load] matched keys: {len(matched)}, mismatched/ignored: {len(mismatches)}")
    if mismatches:
        print("Sample mismatches (key, ckpt_shape, model_shape):", mismatches[:8])
    return model


def show_random():
    print(f"Loading model from {MODEL_PATH} ...")
    if not MODEL_PATH.exists():
        print(f"Error: model not found at {MODEL_PATH}")
        return

    model = robust_load_checkpoint(MODEL_PATH, DEVICE)
    model.eval()
    print("Model loaded.")

    lr_dir = find_lr_dir()
    if lr_dir is None:
        print(
            "Error: no LR directory found. Searched:\n"
            + "\n".join(str(p) for p in CANDIDATE_LR_DIRS)
        )
        return
    lr_paths = sorted(glob.glob(str(lr_dir / "*.png")))
    if not lr_paths:
        print(f"No images found in {lr_dir}")
        return

    random_vlr_path = random.choice(lr_paths)
    filename = os.path.basename(random_vlr_path)

    hr_path = HR_DIR / filename
    if not hr_path.exists():
        print(f"Corresponding HR file not found at {hr_path}")
        return

    # Load images
    vlr_img_bgr = cv2.imread(random_vlr_path)
    hr_img_bgr = cv2.imread(str(hr_path))
    if vlr_img_bgr is None or hr_img_bgr is None:
        print("Failed to read images.")
        return
    vlr_img_rgb = cv2.cvtColor(vlr_img_bgr, cv2.COLOR_BGR2RGB)

    to_tensor = transforms.ToTensor()
    vlr_tensor = to_tensor(vlr_img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_tensor = model(vlr_tensor)

    pred_img = pred_tensor.squeeze(0).cpu().clamp(0, 1).numpy()
    pred_img = (pred_img.transpose(1, 2, 0) * 255).astype(np.uint8)
    pred_img_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

    vlr_display = cv2.resize(
        vlr_img_bgr,
        (pred_img_bgr.shape[1], pred_img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    comparison_img = np.concatenate((vlr_display, pred_img_bgr, hr_img_bgr), axis=1)

    cv2.putText(
        comparison_img,
        "Input (LR)",
        (15, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        comparison_img,
        "Model Output (SR)",
        (15 + vlr_display.shape[1], 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        comparison_img,
        "Ground Truth (HR)",
        (15 + vlr_display.shape[1] + pred_img_bgr.shape[1], 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.imshow("SR Comparison", comparison_img)
    print("Displaying result. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_random()
