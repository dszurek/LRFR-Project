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
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import argparse

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
# DSR model checkpoints for different resolutions
MODEL_PATHS = {
    16: Path(__file__).resolve().parent / "dsr16.pth",
    24: Path(__file__).resolve().parent / "dsr24.pth",
    32: Path(__file__).resolve().parent / "dsr32.pth",  # Default 32x32 model
}
# Try frontal-only HR directory first
CANDIDATE_HR_DIRS = [
    ROOT / "technical" / "dataset" / "frontal_only" / "test" / "hr_images",
    ROOT / "technical" / "dataset" / "test_processed" / "hr_images",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_candidate_lr_dirs(vlr_size: int) -> list[Path]:
    """Generate candidate VLR directories for the given resolution."""
    vlr_dir_name = f"vlr_images_{vlr_size}x{vlr_size}"
    candidates = [
        ROOT / "technical" / "dataset" / "frontal_only" / "test" / vlr_dir_name,
        ROOT / "technical" / "dataset" / "test_processed" / vlr_dir_name,
    ]
    # Legacy fallback for 32x32
    if vlr_size == 32:
        candidates.extend([
            ROOT / "technical" / "dataset" / "frontal_only" / "test" / "vlr_images",
            ROOT / "technical" / "dataset" / "test_processed" / "vlr_images",
        ])
    return candidates


def find_lr_dir(vlr_size: int) -> Optional[Path]:
    for p in get_candidate_lr_dirs(vlr_size):
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


def robust_load_checkpoint(path: Path, device: torch.device, vlr_size: int):
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
    print(f"[load] Model configured for VLR input size: {vlr_size}×{vlr_size}")
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


def process_user_image(model, img_path: str, vlr_size: int):
    """Process user-provided image: detect face, downsample to VLR size, then upscale."""
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

    # Downsample to VLR size
    vlr_face = cv2.resize(hr_face, (vlr_size, vlr_size), interpolation=cv2.INTER_AREA)
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

    # Scale to uniform size for clean side-by-side display
    scaled_size = 360
    scaled_vlr = cv2.resize(
        vlr_face, (scaled_size, scaled_size), interpolation=cv2.INTER_NEAREST
    )
    scaled_dsr = cv2.resize(
        pred_img_bgr, (scaled_size, scaled_size), interpolation=cv2.INTER_LANCZOS4
    )
    scaled_hr = cv2.resize(
        hr_face, (scaled_size, scaled_size), interpolation=cv2.INTER_LANCZOS4
    )

    # Create header with labels
    header_height = 50
    header = np.zeros((header_height, scaled_size * 3, 3), dtype=np.uint8)
    
    # Add labels to header
    def add_text_to_header(header, text, x, y, font_scale=0.7, thickness=2):
        cv2.putText(
            header,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )
    
    add_text_to_header(header, f"VLR Input ({vlr_size}x{vlr_size})", 50, 35)
    add_text_to_header(header, f"DSR Output ({dsr_width}x{dsr_height})", 50 + scaled_size, 35)
    add_text_to_header(header, "Original Face (HR)", 50 + scaled_size * 2, 35)
    
    # Stack images horizontally
    images_row = np.concatenate((scaled_vlr, scaled_dsr, scaled_hr), axis=1)
    
    # Add vertical separators between images
    cv2.line(images_row, (scaled_size, 0), (scaled_size, scaled_size), (100, 100, 100), 2)
    cv2.line(images_row, (scaled_size * 2, 0), (scaled_size * 2, scaled_size), (100, 100, 100), 2)
    
    # Create footer with info
    footer_height = 30
    footer = np.zeros((footer_height, scaled_size * 3, 3), dtype=np.uint8)
    info_text = f"Resolution: {vlr_size}x{vlr_size} -> {dsr_width}x{dsr_height}"
    cv2.putText(
        footer,
        info_text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    
    # Combine header, images, and footer
    comparison_img = np.vstack((header, images_row, footer))

    cv2.imshow("Face SR Comparison", comparison_img)
    print("Displaying result. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_random_database(model, vlr_size: int):
    """Process a random image from the test database (original behavior)."""
    lr_dir = find_lr_dir(vlr_size)
    if lr_dir is None:
        candidate_dirs = get_candidate_lr_dirs(vlr_size)
        messagebox.showerror(
            "Error",
            f"No VLR directory found for {vlr_size}×{vlr_size}. Searched:\n"
            + "\n".join(str(p) for p in candidate_dirs),
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

    # Scale to uniform size for clean side-by-side display
    scaled_size = 360
    scaled_vlr = cv2.resize(
        vlr_img_bgr,
        (scaled_size, scaled_size),
        interpolation=cv2.INTER_NEAREST,
    )
    scaled_dsr = cv2.resize(
        pred_img_bgr,
        (scaled_size, scaled_size),
        interpolation=cv2.INTER_LANCZOS4,
    )
    scaled_hr = cv2.resize(
        hr_img_bgr,
        (scaled_size, scaled_size),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # Create header with labels
    header_height = 50
    header = np.zeros((header_height, scaled_size * 3, 3), dtype=np.uint8)
    
    # Add labels to header
    def add_text_to_header(header, text, x, y, font_scale=0.7, thickness=2):
        cv2.putText(
            header,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )
    
    add_text_to_header(header, f"VLR Input ({vlr_size}x{vlr_size})", 50, 35)
    add_text_to_header(header, f"DSR Output ({dsr_width}x{dsr_height})", 50 + scaled_size, 35)
    add_text_to_header(header, "Original Face (HR)", 50 + scaled_size * 2, 35)
    
    # Stack images horizontally
    images_row = np.concatenate((scaled_vlr, scaled_dsr, scaled_hr), axis=1)
    
    # Add vertical separators between images
    cv2.line(images_row, (scaled_size, 0), (scaled_size, scaled_size), (100, 100, 100), 2)
    cv2.line(images_row, (scaled_size * 2, 0), (scaled_size * 2, scaled_size), (100, 100, 100), 2)
    
    # Create footer with info
    footer_height = 30
    footer = np.zeros((footer_height, scaled_size * 3, 3), dtype=np.uint8)
    info_text = f"Resolution: {vlr_size}x{vlr_size} -> {dsr_width}x{dsr_height} | Image: {filename}"
    cv2.putText(
        footer,
        info_text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    
    # Combine header, images, and footer
    comparison_img = np.vstack((header, images_row, footer))

    cv2.imshow("SR Comparison", comparison_img)
    print("Displaying result. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class DSRTestGUI:
    """GUI for testing DSR model with either random database images or user-uploaded images."""

    def __init__(self, root, vlr_size: int = 32):
        self.root = root
        self.root.title("DSR Super-Resolution Tester")
        self.root.geometry("500x350")
        self.root.resizable(False, False)

        self.vlr_size = vlr_size
        self.model = None
        
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

        # Resolution selection frame
        resolution_frame = tk.Frame(root)
        resolution_frame.pack(pady=10)
        
        tk.Label(
            resolution_frame,
            text="Resolution:",
            font=("Arial", 11),
            fg="#2c3e50",
        ).pack(side=tk.LEFT, padx=5)
        
        self.resolution_var = tk.IntVar(value=vlr_size)
        resolution_combo = ttk.Combobox(
            resolution_frame,
            textvariable=self.resolution_var,
            values=["16", "24", "32"],
            state="readonly",
            width=10,
            font=("Arial", 10),
        )
        resolution_combo.pack(side=tk.LEFT, padx=5)
        resolution_combo.bind("<<ComboboxSelected>>", self.on_resolution_change)

        # Button frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=20)

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

        # Load initial model
        self.load_model()

    def on_resolution_change(self, event=None):
        """Handle resolution change event."""
        new_size = self.resolution_var.get()
        if new_size != self.vlr_size:
            self.vlr_size = new_size
            self.status_label.config(
                text=f"Loading {self.vlr_size}×{self.vlr_size} model...", fg="#3498db"
            )
            self.root.update()
            self.load_model()

    def load_model(self):
        """Load the DSR model for the current resolution."""
        try:
            model_path = MODEL_PATHS.get(self.vlr_size)
            if model_path is None or not model_path.exists():
                available_models = [k for k, v in MODEL_PATHS.items() if v.exists()]
                messagebox.showerror(
                    "Error", 
                    f"Model not found for {self.vlr_size}×{self.vlr_size} at {model_path}\n\n"
                    f"Available models: {available_models if available_models else 'None'}"
                )
                self.status_label.config(
                    text=f"✗ Model not found for {self.vlr_size}×{self.vlr_size}", 
                    fg="#e74c3c"
                )
                self.model = None
                return

            print(f"Loading {self.vlr_size}×{self.vlr_size} model from {model_path}...")
            self.model = robust_load_checkpoint(model_path, DEVICE, self.vlr_size)
            self.model.eval()
            print("Model loaded successfully.")
            
            model_status = f"✓ Model loaded: {self.vlr_size}×{self.vlr_size} ({model_path.name})"
            self.status_label.config(text=model_status, fg="#27ae60")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_label.config(
                text=f"✗ Failed to load {self.vlr_size}×{self.vlr_size} model", 
                fg="#e74c3c"
            )
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
            process_random_database(self.model, self.vlr_size)
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
            process_user_image(self.model, file_path, self.vlr_size)
            self.status_label.config(text="✓ Processing complete", fg="#27ae60")
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_label.config(text="✗ Processing failed", fg="#e74c3c")


def main():
    """Launch the GUI application."""
    parser = argparse.ArgumentParser(
        description="Test DSR super-resolution models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI with 32×32 DSR model (default)
  python -m technical.dsr.test_dsr
  
  # Launch GUI with 16×16 DSR model
  python -m technical.dsr.test_dsr --vlr-size 16
  
  # Launch GUI with 24×24 DSR model
  python -m technical.dsr.test_dsr --vlr-size 24
        """
    )
    parser.add_argument(
        "--vlr-size",
        type=int,
        choices=[16, 24, 32],
        default=32,
        help="VLR input resolution (width=height). Default: 32"
    )
    args = parser.parse_args()
    
    root = tk.Tk()
    app = DSRTestGUI(root, vlr_size=args.vlr_size)
    root.mainloop()


if __name__ == "__main__":
    main()
