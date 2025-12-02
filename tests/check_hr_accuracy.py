import torch
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from technical.pipeline.pipeline import PipelineConfig, build_pipeline

def check_hr_accuracy():
    print("Checking HR Accuracy with EdgeFace XXS...")
    
    # Setup pipeline
    config = PipelineConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        skip_dsr=True,
        edgeface_weights_path=project_root / "technical/facial_rec/edgeface_weights/edgeface_xxs.pt"
    )
    pipeline = build_pipeline(config)
    print(f"Device: {pipeline.device}")
    
    # Load LFW pairs
    lfw_dir = project_root / "technical/dataset/frontal_only/test/hr_images"
    pairs_path = project_root / "technical/dataset/pairs.txt"
    
    if not pairs_path.exists():
        # Fallback: create dummy pairs from directory listing if pairs.txt missing
        # But for now, let's assume we can just pick a few random pairs
        print("pairs.txt not found, using manual pairs")
        # List all images
        images = list(lfw_dir.glob("*.png"))
        if not images:
            print("No images found in HR dir")
            return
            
        # Create 100 positive and 100 negative pairs
        # Assuming filename format: {id}_{name}_{seq}_{pose}_crop_128.png
        # Actually, let's just use the evaluate_lfw_protocol logic if possible, 
        # but to keep it simple, I'll just pick a few known same/diff if I can parse filenames.
        # Filename format from previous logs: 007_01_01_041_00_crop_128.png
        # 007 is ID?
        pass

    # Let's use the LFWEvaluator's loading logic to get the pairs
    from tests.evaluate_lfw_protocol import LFWEvaluator
    
    evaluator = LFWEvaluator(
        dataset_dir=project_root / "technical/dataset",
        dsr_dir=project_root / "technical/dsr",
        edgeface_dir=project_root / "technical/facial_rec/edgeface_weights",
        output_dir=project_root / "technical/evaluation_results",
        device=config.device
    )
    
    # Load subjects using public method
    # load_lfw_subjects returns Dict[str, List[Path]]
    subjects = evaluator.load_lfw_subjects((112, 112))
    print(f"Loaded {len(subjects)} subjects")
    
    # Generate pairs manually since _generate_pairs is private/missing
    import random
    random.seed(42)
    pairs = []
    subject_ids = list(subjects.keys())
    
    # Generate 100 positive pairs
    pos_count = 0
    while pos_count < 100:
        sid = random.choice(subject_ids)
        imgs = subjects[sid]
        if len(imgs) >= 2:
            img1, img2 = random.sample(imgs, 2)
            pairs.append((img1, img2, "genuine"))
            pos_count += 1
            
    # Generate 100 negative pairs
    neg_count = 0
    while neg_count < 100:
        sid1, sid2 = random.sample(subject_ids, 2)
        img1 = random.choice(subjects[sid1])
        img2 = random.choice(subjects[sid2])
        pairs.append((img1, img2, "impostor"))
        neg_count += 1
        
    print(f"Generated {len(pairs)} pairs")
    
    # Run verification on the subset
    subset_size = len(pairs)
    pairs_subset = pairs
    
    scores = []
    labels = []
    
    print(f"Running inference on {subset_size} pairs...")
    
    for i, (img1_path, img2_path, label) in enumerate(pairs_subset):
        # Load and preprocess
        # Pipeline expects path or PIL
        # infer_embedding handles loading and preprocessing
        
        # Load as PIL and convert to tensor using pipeline helper
        img1 = pipeline._to_tensor(img1_path)
        img2 = pipeline._to_tensor(img2_path)
        
        # Debug input stats
        if i == 0:
            print(f"Img1 Stats: Mean={img1.mean():.4f}, Std={img1.std():.4f}, Min={img1.min():.4f}, Max={img1.max():.4f}")
            print(f"Img1 Shape: {img1.shape}")
        
        emb1 = pipeline.infer_embedding(img1)
        emb2 = pipeline.infer_embedding(img2)
        
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        scores.append(sim)
        labels.append(1 if label == "genuine" else 0)
        
        if i % 20 == 0:
            print(f"{i}/{subset_size}: Label={label}, Score={sim:.4f}")

    # Calculate accuracy
    scores = np.array(scores)
    labels = np.array(labels)
    
    best_acc = 0
    best_thresh = 0
    
    for thresh in np.arange(-1, 1, 0.01):
        preds = (scores > thresh).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    print(f"\nBest Accuracy: {best_acc*100:.2f}% at threshold {best_thresh:.2f}")

if __name__ == "__main__":
    check_hr_accuracy()
