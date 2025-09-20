# technical/dsr/train_dsr.py
import os
import glob
import cv2
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Correctly import the EdgeFace model-building functions

from facial_rec.backbones import get_model, replace_linear_with_lowrank_2

# ============================================================
# Config
# ============================================================
# Paths are relative to the 'technical/' directory where the script is run from
TRAIN_DIR = "dataset/train_processed"
VAL_DIR = "dataset/val_processed"
EDGEFACE_WEIGHTS_PATH = "facial_rec/edgeface_weights/edgeface_s_gamma_05.pt"
SAVE_PATH = "dsr.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH_SIZE = 12
LR = 1e-4
LAMBDA_ID = 0.1  # Weight for identity-preserving loss
NUM_WORKERS = 8 if torch.cuda.is_available() else 0
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EDGEFACE_INPUT_SIZE = 112


# ============================================================
# Dataset
# ============================================================
class PairedFaceSRDataset(Dataset):
    def __init__(self, data_dir):
        self.vlr_paths = sorted(
            glob.glob(os.path.join(data_dir, "vlr_images", "*.png"))
        )
        self.hr_paths = sorted(glob.glob(os.path.join(data_dir, "hr_images", "*.png")))
        assert len(self.vlr_paths) == len(self.hr_paths)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.vlr_paths)

    def __getitem__(self, idx):
        vlr_bgr = cv2.imread(self.vlr_paths[idx], cv2.IMREAD_COLOR)
        hr_bgr = cv2.imread(self.hr_paths[idx], cv2.IMREAD_COLOR)
        vlr_rgb = cv2.cvtColor(vlr_bgr, cv2.COLOR_BGR2RGB)
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(vlr_rgb), self.transform(hr_rgb)


# ============================================================
# Utilities
# ============================================================
def compute_psnr(sr, hr, max_val=1.0):
    mse = torch.mean((sr - hr) ** 2, dim=[1, 2, 3])
    psnr_vals = 10 * torch.log10((max_val**2) / (mse + 1e-12))
    return psnr_vals.mean().item()


# ============================================================
# DSR Model
# ============================================================
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

        # --- FIX: Add this line to force the final output size ---
        return transforms.functional.resize(x, [160, 160], antialias=True)


# ============================================================
# EdgeFace Wrapper
# ============================================================
class EdgeFaceWrapper(nn.Module):
    def __init__(self, weights_path):
        super().__init__()

        # This part correctly builds the base model architecture
        self.model = get_model("edgenext_small", featdim=512)
        self.model = replace_linear_with_lowrank_2(self.model, rank_ratio=0.5)

        # Load the pre-trained weights from the file
        state_dict = torch.load(weights_path, map_location="cpu")

        # --- FIX: Strip the outer "model." prefix from each key ---
        state_dict = torch.load(weights_path, map_location="cpu")

        # Fix key mismatch: add "model." prefix if missing
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("model."):
                new_state_dict["model." + k] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)

        self.model.eval()

        # Define the preprocessing transformation
        self.transform = transforms.Compose(
            [
                transforms.Resize((EDGEFACE_INPUT_SIZE, EDGEFACE_INPUT_SIZE)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @torch.no_grad()
    def get_embeddings(self, imgs_torch):
        imgs_transformed = self.transform(imgs_torch)
        return self.model(imgs_transformed)


# ============================================================
# Training
# ============================================================
def train():
    train_ds = PairedFaceSRDataset(TRAIN_DIR)
    val_ds = PairedFaceSRDataset(VAL_DIR)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = DSRColor().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    l1_loss = nn.L1Loss()
    cosine_loss = nn.CosineEmbeddingLoss()

    print("Using device:", DEVICE)
    print("")

    print("Loading EdgeFace...")
    identity_model = EdgeFaceWrapper(EDGEFACE_WEIGHTS_PATH).to(DEVICE)

    best_val_psnr = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for vlr, hr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
            vlr, hr = vlr.to(DEVICE), hr.to(DEVICE)
            pred = model(vlr)
            pred_clamped = torch.clamp(pred, 0, 1)

            l_pix = l1_loss(pred_clamped, hr)

            emb_sr = identity_model.get_embeddings(pred_clamped)
            emb_hr = identity_model.get_embeddings(hr)
            target = torch.ones(emb_sr.size(0)).to(DEVICE)
            l_id = cosine_loss(emb_sr, emb_hr, target)

            loss = l_pix + LAMBDA_ID * l_id

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_psnr = 0.0, 0.0
        with torch.no_grad():
            for vlr, hr in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Val"):
                vlr, hr = vlr.to(DEVICE), hr.to(DEVICE)
                pred = model(vlr)
                pred_clamped = torch.clamp(pred, 0, 1)

                l_pix = l1_loss(pred_clamped, hr)
                emb_sr = identity_model.get_embeddings(pred_clamped)
                emb_hr = identity_model.get_embeddings(hr)
                target = torch.ones(emb_sr.size(0)).to(DEVICE)
                l_id = cosine_loss(emb_sr, emb_hr, target)

                loss = l_pix + LAMBDA_ID * l_id
                val_loss += loss.item()
                val_psnr += compute_psnr(pred_clamped, hr)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)

        print(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f} PSNR={avg_val_psnr:.2f}dB")

        scheduler.step()

        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_psnr": avg_val_psnr,
                },
                SAVE_PATH,
            )
            print(f"✅ Saved best model to {SAVE_PATH} (PSNR: {avg_val_psnr:.2f}dB)")

    print(f"Training complete. Best validation PSNR: {best_val_psnr:.2f}dB")


if __name__ == "__main__":
    train()
