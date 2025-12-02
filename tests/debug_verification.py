import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import sys
import os
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from technical.pipeline.pipeline import PipelineConfig, build_pipeline

def debug_verification():
    print(f"Project Root: {project_root}")
    
    # Setup paths
    hr_dir = project_root / "technical/dataset/frontal_only/test/hr_images"
    img1_path = hr_dir / "007_01_01_041_00_crop_128.png"
    img2_path = hr_dir / "007_01_01_041_01_crop_128.png" # Same person
    img3_path = hr_dir / "010_01_01_041_00_crop_128.png" # Diff person
    
    if not img1_path.exists():
        print(f"Error: {img1_path} not found")
        return

    # Setup Pipeline
    config = PipelineConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        skip_dsr=True,
        edgeface_weights_path=project_root / "technical/facial_rec/edgeface_weights/edgeface_s_gamma_05.pt"
    )
    pipeline = build_pipeline(config)
    print(f"Pipeline device: {pipeline.device}")

    # Load images
    img1_rgb = Image.open(img1_path).convert('RGB').resize((112, 112), Image.Resampling.BICUBIC)
    img2_rgb = Image.open(img2_path).convert('RGB').resize((112, 112), Image.Resampling.BICUBIC)
    img3_rgb = Image.open(img3_path).convert('RGB').resize((112, 112), Image.Resampling.BICUBIC)

    print("\n--- Final Verification Test (RGB, HR, 0-1) ---")
    t1 = pipeline._to_tensor(img1_rgb).to(pipeline.device)
    t2 = pipeline._to_tensor(img2_rgb).to(pipeline.device) # Same person
    t3 = pipeline._to_tensor(img3_rgb).to(pipeline.device) # Diff person
    t1 = pipeline.preprocess(t1)
    t2 = pipeline.preprocess(t2)
    t3 = pipeline.preprocess(t3)

    # Inspect Inputs
    print(f"\nInput t1: Mean={t1.mean():.4f}, Std={t1.std():.4f}, Min={t1.min():.4f}, Max={t1.max():.4f}")
    sim_input = F.cosine_similarity(t1.flatten().unsqueeze(0), t3.flatten().unsqueeze(0)).item()
    print(f"Input Similarity (007 vs 010): {sim_input:.4f}")

    with torch.no_grad():
        emb1 = pipeline.recognition_model(t1)
        emb2 = pipeline.recognition_model(t2)
        emb3 = pipeline.recognition_model(t3)

    # Normalize
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    emb3 = F.normalize(emb3, p=2, dim=1)

    sim_gen = F.cosine_similarity(emb1, emb2).item()
    sim_imp = F.cosine_similarity(emb1, emb3).item()

    print(f"Genuine (007 vs 007): {sim_gen:.4f}")
    print(f"Impostor (007 vs 010): {sim_imp:.4f}")
    
    threshold = 0.3
    print(f"Result: {'PASS' if sim_gen > threshold and sim_imp < threshold else 'FAIL'}")

if __name__ == "__main__":
    debug_verification()
