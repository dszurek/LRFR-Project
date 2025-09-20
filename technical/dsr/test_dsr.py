import os
import glob
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import random


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


# ============================================================
# Config
# ============================================================
MODEL_PATH = "dsr/dsr.pth"
TEST_DATA_DIR = "dataset/test_processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Main Inference Function
# ============================================================
def show_random():
    # --- Load Model ---
    print(f"Loading model from {MODEL_PATH}...")
    model = DSRColor().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")

    # --- Get Image Paths ---
    vlr_paths = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "vlr_images", "*.png")))
    if not vlr_paths:
        print(f"Error: No images found in {TEST_DATA_DIR}/vlr_images/")
        return

    # --- Select a Random Image ---
    random_vlr_path = random.choice(vlr_paths)
    filename = os.path.basename(random_vlr_path)
    hr_path = os.path.join(TEST_DATA_DIR, "hr_images", filename)
    print(f"Testing with random image: {filename}")

    # --- Load and Preprocess ---
    vlr_img_bgr = cv2.imread(random_vlr_path)
    hr_img_bgr = cv2.imread(hr_path)
    vlr_img_rgb = cv2.cvtColor(vlr_img_bgr, cv2.COLOR_BGR2RGB)

    to_tensor = transforms.ToTensor()
    vlr_tensor = to_tensor(vlr_img_rgb).unsqueeze(0).to(DEVICE)

    # --- Run Inference ---
    with torch.no_grad():
        pred_tensor = model(vlr_tensor)

    # --- Post-process for Display ---
    pred_img = pred_tensor.squeeze(0).cpu().clamp(0, 1).numpy()
    pred_img = (pred_img.transpose(1, 2, 0) * 255).astype(np.uint8)
    pred_img_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

    vlr_display = cv2.resize(vlr_img_bgr, (160, 160), interpolation=cv2.INTER_NEAREST)

    # --- Create and Display Comparison Image ---
    comparison_img = np.concatenate((vlr_display, pred_img_bgr, hr_img_bgr), axis=1)

    cv2.putText(
        comparison_img,
        "Input (VLR)",
        (15, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        comparison_img,
        "Model Output (SR)",
        (165, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        comparison_img,
        "Ground Truth (HR)",
        (325, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.imshow("Super-Resolution Comparison", comparison_img)
    print("Displaying result. Press any key to exit.")
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_random()
