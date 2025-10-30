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
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

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
# fallback dataset locations (try frontal-only first, then full dataset)
CANDIDATE_LR_DIRS = [
    ROOT / "technical" / "dataset" / "frontal_only" / "test" / "vlr_images",
    ROOT / "technical" / "dataset" / "test_processed" / "vlr_images",
    ROOT / "technical" / "dataset" / "test_processed" / "clr_images",
    ROOT / "dataset" / "test_processed" / "vlr_images",
]
# Try frontal-only HR directory first
CANDIDATE_HR_DIRS = [
    ROOT / "technical" / "dataset" / "frontal_only" / "test" / "hr_images",
    ROOT / "technical" / "dataset" / "test_processed" / "hr_images",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_lr_dir() -> Optional[Path]:
    for p in CANDIDATE_LR_DIRS:
        if p.exists():
            return p
    return None


def find_hr_dir() -> Optional[Path]:
    for p in CANDIDATE_HR_DIRS:
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

    # Override output_size if saved in checkpoint config
    if isinstance(ckpt, dict) and "config" in ckpt:
        saved_config = ckpt["config"]
        if "target_hr_size" in saved_config:
            target_size = saved_config["target_hr_size"]
            config.output_size = (target_size, target_size)
            print(f"[load] Using saved target HR size: {target_size}×{target_size}")

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


def detect_and_crop_face(
    img_bgr: np.ndarray, target_size: int = 112
) -> Optional[np.ndarray]:
    """Detect face in image and crop to square region around it."""
    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    # Use the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Expand crop region by 30% for context
    margin = int(max(w, h) * 0.3)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_bgr.shape[1], x + w + margin)
    y2 = min(img_bgr.shape[0], y + h + margin)

    # Crop to square
    crop_size = max(x2 - x1, y2 - y1)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    x1 = max(0, center_x - crop_size // 2)
    y1 = max(0, center_y - crop_size // 2)
    x2 = min(img_bgr.shape[1], x1 + crop_size)
    y2 = min(img_bgr.shape[0], y1 + crop_size)

    face_crop = img_bgr[y1:y2, x1:x2]

    # Resize to target size
    face_resized = cv2.resize(
        face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA
    )

    return face_resized


def process_user_image(model, img_path: str):
    """Process user-provided image: detect face, downsample to 32x32, then upscale."""
    # Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        messagebox.showerror("Error", f"Failed to load image from {img_path}")
        return

    # Detect and crop face to 112x112 (to match DSR output)
    hr_face = detect_and_crop_face(img_bgr, target_size=112)
    if hr_face is None:
        messagebox.showerror(
            "Error",
            "No face detected in the image. Please select an image with a visible face.",
        )
        return

    # Downsample to 32x32 (VLR)
    vlr_face = cv2.resize(hr_face, (32, 32), interpolation=cv2.INTER_AREA)
    vlr_rgb = cv2.cvtColor(vlr_face, cv2.COLOR_BGR2RGB)

    # Run DSR
    to_tensor = transforms.ToTensor()
    vlr_tensor = to_tensor(vlr_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_tensor = model(vlr_tensor)

    pred_img = pred_tensor.squeeze(0).cpu().clamp(0, 1).numpy()
    pred_img = (pred_img.transpose(1, 2, 0) * 255).astype(np.uint8)
    pred_img_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

    # Get actual DSR output size (should be 112x112)
    dsr_height, dsr_width = pred_img_bgr.shape[:2]
    print(f"DSR output size: {dsr_width}×{dsr_height}")

    # Upscale VLR for display (match DSR output size)
    vlr_display = cv2.resize(
        vlr_face, (dsr_width, dsr_height), interpolation=cv2.INTER_NEAREST
    )

    # Resize HR face to match DSR output size for comparison
    hr_face_resized = cv2.resize(
        hr_face, (dsr_width, dsr_height), interpolation=cv2.INTER_LANCZOS4
    )

    # Create comparison image (all images should be same size now)
    comparison_img = np.concatenate(
        (vlr_display, pred_img_bgr, hr_face_resized), axis=1
    )

    # Add labels with positions adjusted for 112px width images
    cv2.putText(
        comparison_img,
        "VLR Input (32x32)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        comparison_img,
        f"DSR Output ({dsr_width}x{dsr_height})",
        (dsr_width + 10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        comparison_img,
        "Original Face (HR)",
        (dsr_width * 2 + 10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )

    cv2.imshow("Face SR Comparison", comparison_img)
    print("Displaying result. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_random_database(model):
    """Process a random image from the test database (original behavior)."""
    lr_dir = find_lr_dir()
    if lr_dir is None:
        messagebox.showerror(
            "Error",
            "No LR directory found. Searched:\n"
            + "\n".join(str(p) for p in CANDIDATE_LR_DIRS),
        )
        return

    hr_dir = find_hr_dir()
    if hr_dir is None:
        messagebox.showerror(
            "Error",
            "No HR directory found. Searched:\n"
            + "\n".join(str(p) for p in CANDIDATE_HR_DIRS),
        )
        return

    lr_paths = sorted(glob.glob(str(lr_dir / "*.png")))
    if not lr_paths:
        messagebox.showerror("Error", f"No images found in {lr_dir}")
        return

    random_vlr_path = random.choice(lr_paths)
    filename = os.path.basename(random_vlr_path)

    hr_path = hr_dir / filename
    if not hr_path.exists():
        messagebox.showerror("Error", f"Corresponding HR file not found at {hr_path}")
        return

    # Load images
    vlr_img_bgr = cv2.imread(random_vlr_path)
    hr_img_bgr = cv2.imread(str(hr_path))
    if vlr_img_bgr is None or hr_img_bgr is None:
        messagebox.showerror("Error", "Failed to read images.")
        return
    vlr_img_rgb = cv2.cvtColor(vlr_img_bgr, cv2.COLOR_BGR2RGB)

    to_tensor = transforms.ToTensor()
    vlr_tensor = to_tensor(vlr_img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_tensor = model(vlr_tensor)

    pred_img = pred_tensor.squeeze(0).cpu().clamp(0, 1).numpy()
    pred_img = (pred_img.transpose(1, 2, 0) * 255).astype(np.uint8)
    pred_img_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

    # Get actual DSR output size
    dsr_height, dsr_width = pred_img_bgr.shape[:2]
    print(f"DSR output size: {dsr_width}×{dsr_height}")

    # Resize VLR for display to match DSR output
    vlr_display = cv2.resize(
        vlr_img_bgr,
        (dsr_width, dsr_height),
        interpolation=cv2.INTER_NEAREST,
    )

    # Resize HR to match DSR output size for comparison
    hr_img_resized = cv2.resize(
        hr_img_bgr,
        (dsr_width, dsr_height),
        interpolation=cv2.INTER_LANCZOS4,
    )

    comparison_img = np.concatenate((vlr_display, pred_img_bgr, hr_img_resized), axis=1)

    # Add labels with dynamic positioning based on actual image widths
    cv2.putText(
        comparison_img,
        "VLR Input (32x32)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        comparison_img,
        f"DSR Output ({dsr_width}x{dsr_height})",
        (dsr_width + 10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        comparison_img,
        "Original Face (HR)",
        (dsr_width * 2 + 10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )

    cv2.imshow("SR Comparison", comparison_img)
    print("Displaying result. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class DSRTestGUI:
    """GUI for testing DSR model with either random database images or user-uploaded images."""

    def __init__(self, root):
        self.root = root
        self.root.title("DSR Super-Resolution Tester")
        self.root.geometry("500x300")
        self.root.resizable(False, False)

        self.model = None
        self.load_model()

        # Title
        title_label = tk.Label(
            root,
            text="DSR Face Super-Resolution",
            font=("Arial", 20, "bold"),
            fg="#2c3e50",
        )
        title_label.pack(pady=20)

        # Subtitle
        subtitle_label = tk.Label(
            root,
            text="Test super-resolution on database images or your own photos",
            font=("Arial", 10),
            fg="#7f8c8d",
        )
        subtitle_label.pack(pady=5)

        # Button frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=30)

        # Random database button
        random_btn = tk.Button(
            button_frame,
            text="📊 Random Database Image",
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            width=25,
            height=2,
            command=self.process_random,
            cursor="hand2",
        )
        random_btn.pack(pady=10)

        # Upload image button
        upload_btn = tk.Button(
            button_frame,
            text="📁 Upload Your Image",
            font=("Arial", 12),
            bg="#27ae60",
            fg="white",
            activebackground="#229954",
            activeforeground="white",
            width=25,
            height=2,
            command=self.process_upload,
            cursor="hand2",
        )
        upload_btn.pack(pady=10)

        # Status label
        self.status_label = tk.Label(root, text="", font=("Arial", 9), fg="#95a5a6")
        self.status_label.pack(pady=10)

        # Model status
        if self.model is not None:
            model_status = f"✓ Model loaded from {MODEL_PATH.name}"
            self.status_label.config(text=model_status, fg="#27ae60")
        else:
            self.status_label.config(text="✗ Model not found", fg="#e74c3c")

    def load_model(self):
        """Load the DSR model."""
        try:
            if not MODEL_PATH.exists():
                messagebox.showerror("Error", f"Model not found at {MODEL_PATH}")
                return

            print(f"Loading model from {MODEL_PATH}...")
            self.model = robust_load_checkpoint(MODEL_PATH, DEVICE)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model = None

    def process_random(self):
        """Process a random image from the database."""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return

        self.status_label.config(
            text="Processing random database image...", fg="#3498db"
        )
        self.root.update()

        try:
            process_random_database(self.model)
            self.status_label.config(text="✓ Processing complete", fg="#27ae60")
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_label.config(text="✗ Processing failed", fg="#e74c3c")

    def process_upload(self):
        """Process a user-uploaded image."""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select an image with a face",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return  # User cancelled

        self.status_label.config(text="Processing your image...", fg="#3498db")
        self.root.update()

        try:
            process_user_image(self.model, file_path)
            self.status_label.config(text="✓ Processing complete", fg="#27ae60")
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_label.config(text="✗ Processing failed", fg="#e74c3c")


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = DSRTestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
