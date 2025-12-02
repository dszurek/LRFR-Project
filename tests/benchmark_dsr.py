# technical/dsr/benchmark_dsr.py
import argparse
import os
import glob
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Import the new hybrid model utilities
from technical.dsr.hybrid_model import load_dsr_model

# --- Dataset Loader ---
class BenchmarkDataset(Dataset):
    def __init__(self, data_dir, vlr_size=32):
        self.hr_paths = sorted(glob.glob(os.path.join(data_dir, "hr_images", "*.png")))
        # Support different VLR sizes
        vlr_dir_name = f"vlr_images_{vlr_size}x{vlr_size}"
        self.vlr_paths = sorted(
            glob.glob(os.path.join(data_dir, vlr_dir_name, "*.png"))
        )
        
        # Fallback to default vlr_images if specific size dir doesn't exist
        if not self.vlr_paths:
             self.vlr_paths = sorted(
                glob.glob(os.path.join(data_dir, "vlr_images", "*.png"))
            )

        if len(self.hr_paths) != len(self.vlr_paths):
            print(f"Warning: Mismatch in number of images. HR: {len(self.hr_paths)}, VLR: {len(self.vlr_paths)}")
            # Truncate to the smaller size for safety
            min_len = min(len(self.hr_paths), len(self.vlr_paths))
            self.hr_paths = self.hr_paths[:min_len]
            self.vlr_paths = self.vlr_paths[:min_len]

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_bgr = cv2.imread(self.hr_paths[idx])
        vlr_bgr = cv2.imread(self.vlr_paths[idx])
        
        if hr_bgr is None:
            raise ValueError(f"Failed to load HR image: {self.hr_paths[idx]}")
        if vlr_bgr is None:
            raise ValueError(f"Failed to load VLR image: {self.vlr_paths[idx]}")

        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        vlr_rgb = cv2.cvtColor(vlr_bgr, cv2.COLOR_BGR2RGB)
        
        # Ensure HR is 112x112 (standard for this project)
        if hr_rgb.shape[:2] != (112, 112):
            hr_rgb = cv2.resize(hr_rgb, (112, 112), interpolation=cv2.INTER_CUBIC)

        return self.transform(vlr_rgb), self.transform(hr_rgb)


# --- Metric Calculation Functions ---
def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((max_val**2) / mse)


def calculate_ssim(img1, img2):
    # Convert tensors to numpy arrays in the correct format for scikit-image
    # (H, W, C) range [0, 1]
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    
    # Explicitly specify channel_axis for newer skimage versions, or multichannel for older
    try:
        return ssim(img1_np, img2_np, data_range=1.0, channel_axis=2, win_size=7)
    except TypeError:
        # Fallback for older skimage
        return ssim(img1_np, img2_np, data_range=1.0, multichannel=True, win_size=7)


def calculate_entropy(img):
    # Convert to grayscale and then to a 1D histogram for entropy calculation
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    # Normalize histogram
    hist = hist.ravel() / hist.sum()
    return entropy(hist)


# ============================================================
# Main Benchmark Function
# ============================================================
def benchmark(args):
    # --- Config ---
    DEVICE = torch.device(args.device)
    
    # Resolve paths
    base_dir = Path(__file__).resolve().parents[1] # Assuming tests/benchmark_dsr.py
    
    if args.dataset_path:
        test_data_dir = Path(args.dataset_path)
    else:
        test_data_dir = base_dir / "technical" / "dataset" / "test_processed"

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = base_dir / "technical" / "dsr" / f"hybrid_dsr{args.vlr_size}.pth"

    print(f"Benchmark Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {test_data_dir}")
    print(f"  VLR Size: {args.vlr_size}x{args.vlr_size}")
    print(f"  Device: {DEVICE}")

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    if not test_data_dir.exists():
        print(f"Error: Dataset directory not found at {test_data_dir}")
        return

    # --- Load Model ---
    print(f"\nLoading model...")
    try:
        model = load_dsr_model(model_path, device=DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # --- Load Data ---
    dataset = BenchmarkDataset(test_data_dir, vlr_size=args.vlr_size)
    if len(dataset) == 0:
        print("No images found in dataset.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Run Benchmark Loop ---
    psnr_scores, ssim_scores, entropy_scores = [], [], []
    print(f"Running benchmark on {len(dataset)} images...")

    with torch.no_grad():
        for vlr_batch, hr_batch in tqdm(dataloader, desc="Benchmarking"):
            vlr_batch = vlr_batch.to(DEVICE)
            hr_batch = hr_batch.to(DEVICE)

            # Inference
            pred_batch = model(vlr_batch)
            
            # Ensure output is clamped to [0, 1]
            pred_batch = pred_batch.clamp(0, 1)

            # Resize if necessary (though HybridDSR should output 112x112)
            if pred_batch.shape[-2:] != hr_batch.shape[-2:]:
                pred_batch = nn.functional.interpolate(
                    pred_batch, size=hr_batch.shape[-2:], mode='bilinear', align_corners=False
                )

            # Calculate metrics for each image in the batch
            for i in range(pred_batch.size(0)):
                pred_img, hr_img = pred_batch[i], hr_batch[i]
                psnr_scores.append(calculate_psnr(pred_img, hr_img).item())
                ssim_scores.append(calculate_ssim(pred_img, hr_img))
                entropy_scores.append(calculate_entropy(pred_img))

    # --- Calculate and Print Final Report ---

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_entropy = np.mean(entropy_scores)

    print("\n" + "=" * 50)
    print(" DSR MODEL BENCHMARK REPORT")
    print("=" * 50)
    print(f"\nMetrics evaluated on '{test_data_dir}' dataset.")
    print("-" * 50)

    print(f"PSNR:          {avg_psnr:.2f} dB")
    print(f"  (Target: > 20.00 dB)") # Adjusted expectation for 112x112

    print(f"\nSSIM:          {avg_ssim:.4f}")
    print(f"  (Target: > 0.6000)") # Adjusted expectation

    print(f"\nEntropy:       {avg_entropy:.4f}")
    print(f"  (Target: > 4.0)") 
    print("=" * 50)
    
    return avg_psnr, avg_ssim, avg_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DSR models")
    parser.add_argument("--vlr-size", type=int, default=32, choices=[16, 24, 32], help="VLR input size")
    parser.add_argument("--model-path", type=str, default=None, help="Path to DSR model weights")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to test dataset root")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    benchmark(args)
