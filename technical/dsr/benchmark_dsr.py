# technical/dsr/benchmark_dsr.py
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


# --- Must include the model definition to load the weights ---
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        r = self.conv1(x)
        r = self.prelu(r)
        r = self.conv2(r)
        return self.prelu(x + r)


class DSRColor(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, res_blocks=8, up_factors=[2, 5]):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_ch) for _ in range(res_blocks)]
        )
        self.ups = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(base_ch, base_ch * (f**2), 3, 1, 1),
                    nn.PixelShuffle(f),
                    nn.PReLU(),
                )
                for f in up_factors
            ]
        )
        self.conv_out = nn.Conv2d(base_ch, in_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        for u in self.ups:
            x = u(x)
        x = self.conv_out(x)
        return transforms.functional.resize(x, [160, 160], antialias=True)


# --- Dataset Loader ---
class BenchmarkDataset(Dataset):
    def __init__(self, data_dir):
        self.hr_paths = sorted(glob.glob(os.path.join(data_dir, "hr_images", "*.png")))
        self.vlr_paths = sorted(
            glob.glob(os.path.join(data_dir, "vlr_images", "*.png"))
        )
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_bgr = cv2.imread(self.hr_paths[idx])
        vlr_bgr = cv2.imread(self.vlr_paths[idx])
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        vlr_rgb = cv2.cvtColor(vlr_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(vlr_rgb), self.transform(hr_rgb)


# --- Metric Calculation Functions ---
def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10((max_val**2) / (mse + 1e-12))


def calculate_ssim(img1, img2):
    # Convert tensors to numpy arrays in the correct format for scikit-image
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return ssim(img1_np, img2_np, data_range=1.0, channel_axis=2, win_size=7)


def calculate_entropy(img):
    # Convert to grayscale and then to a 1D histogram for entropy calculation
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    return entropy(hist.flatten())


# ============================================================
# Main Benchmark Function
# ============================================================
def benchmark():
    # --- Config ---
    MODEL_PATH = "dsr/dsr.pth"
    TEST_DATA_DIR = "dataset/test_processed"
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model ---
    print(f"Loading model from {MODEL_PATH}...")
    model = DSRColor().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")

    # --- Load Data ---
    dataset = BenchmarkDataset(TEST_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Run Benchmark Loop ---
    psnr_scores, ssim_scores, entropy_scores = [], [], []
    print(f"Running benchmark on {len(dataset)} images...")

    with torch.no_grad():
        for vlr_batch, hr_batch in tqdm(dataloader, desc="Benchmarking"):
            vlr_batch = vlr_batch.to(DEVICE)
            hr_batch = hr_batch.to(DEVICE)

            pred_batch = model(vlr_batch).clamp(0, 1)

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
    print(f"\nMetrics evaluated on '{TEST_DATA_DIR}' dataset.")
    print("-" * 50)

    print(f"PSNR:          {avg_psnr:.2f} dB")
    print(f"  (Target: > 30.00 dB)")

    print(f"\nSSIM:          {avg_ssim:.4f}")
    print(f"  (Target: > 0.9500)")

    print(f"\nEntropy:       {avg_entropy:.4f}")
    print(f"  (Target: > 7.0600 - from DSR paper)")
    print("=" * 50)


if __name__ == "__main__":
    benchmark()
