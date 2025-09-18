import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import os
import glob
import cv2
from tqdm import tqdm
import numpy as np


# ==============================================================================
#  Component 1: The DSR Network Architecture
# ==============================================================================
class DSRModel(nn.Module):
    def __init__(self, upscale_factor=10):
        super(DSRModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.upsample_block = nn.Sequential(
            nn.Conv2d(128, 128 * (upscale_factor**2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.upsample_block(x)
        output = self.conv_out(x)
        return output


# ==============================================================================
#  Component 2: The Custom DSR Loss Function
# ==============================================================================
class DSRLoss(nn.Module):
    def __init__(self, gamma=0.01, margin=1.0):
        super(DSRLoss, self).__init__()
        self.gamma = gamma
        self.recon_loss = nn.L1Loss()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:9]
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.feature_extractor = vgg

    def get_triplets(self, embeddings, labels):
        # Note: This is a simplified triplet creation method for demonstration.
        # For a full research paper, a more advanced triplet mining strategy is recommended.
        anchors, positives, negatives = [], [], []
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            is_pos = labels == label
            is_neg = labels != label
            pos_indices = torch.where(is_pos)[0]
            neg_indices = torch.where(is_neg)[0]

            if len(pos_indices) > 1 and len(neg_indices) > 0:
                anchor_idx = pos_indices[0]
                positive_idx = pos_indices[1]
                negative_idx = neg_indices[0]
                anchors.append(embeddings[anchor_idx])
                positives.append(embeddings[positive_idx])
                negatives.append(embeddings[negative_idx])

        if not anchors:
            return None, None, None
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    def forward(self, predicted_hr, true_hr, labels):
        l_recon = self.recon_loss(predicted_hr, true_hr)

        predicted_embeddings = self.feature_extractor(predicted_hr)
        predicted_embeddings = predicted_embeddings.view(
            predicted_embeddings.size(0), -1
        )

        anchors, positives, negatives = self.get_triplets(predicted_embeddings, labels)

        l_disc = torch.tensor(0.0, device=predicted_hr.device)
        if anchors is not None:
            l_disc = self.triplet_loss(anchors, positives, negatives)

        total_loss = l_recon + self.gamma * l_disc
        return total_loss


# ==============================================================================
#  Component 3: The Dataset Loader
# ==============================================================================
class FaceSuperResDataset(Dataset):
    def __init__(self, data_dir):
        self.hr_image_paths = sorted(
            glob.glob(os.path.join(data_dir, "hr_images", "*.png"))
        )
        self.vlr_image_paths = sorted(
            glob.glob(os.path.join(data_dir, "vlr_images", "*.png"))
        )
        # Extract labels from filenames (e.g., '001_...' -> label 0)
        self.labels = [
            int(os.path.basename(p).split("_")[0]) - 1 for p in self.hr_image_paths
        ]

    def __len__(self):
        return len(self.hr_image_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_image_paths[idx]
        vlr_path = self.vlr_image_paths[idx]
        label = self.labels[idx]

        # Load images with OpenCV (reads as BGR)
        hr_image = cv2.imread(hr_path)
        vlr_image = cv2.imread(vlr_path)

        # Convert BGR to RGB
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        vlr_image = cv2.cvtColor(vlr_image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to PyTorch tensor format (C, H, W)
        hr_tensor = (
            torch.from_numpy(hr_image.astype(np.float32)).permute(2, 0, 1) / 255.0
        )
        vlr_tensor = (
            torch.from_numpy(vlr_image.astype(np.float32)).permute(2, 0, 1) / 255.0
        )

        return vlr_tensor, hr_tensor, torch.tensor(label, dtype=torch.long)


# ==============================================================================
#  Main Training Block
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    EPOCHS = 50
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    GAMMA = 0.01  # Weight for discriminative loss
    TRAIN_DATA_DIR = "dataset/train_processed"

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading dataset...")
    train_dataset = FaceSuperResDataset(data_dir=TRAIN_DATA_DIR)
    # Note: For triplet loss, it's better to use a sampler that ensures each batch
    # has multiple instances of the same class. For simplicity, we use standard shuffle.
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    print("Dataset loaded.")

    # --- Model Initialization ---
    model = DSRModel(upscale_factor=10).to(device)
    criterion = DSRLoss(gamma=GAMMA).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for vlr_images, hr_images, labels in progress_bar:
            vlr_images = vlr_images.to(device)
            hr_images = hr_images.to(device)
            labels = labels.to(device)

            # Forward pass
            predicted_hr = model(vlr_images)
            loss = criterion(predicted_hr, hr_images, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    # --- Save the Trained Model ---
    torch.save(model.state_dict(), "dsr_model.pth")
    print("Training complete. Model saved as dsr_model.pth")
