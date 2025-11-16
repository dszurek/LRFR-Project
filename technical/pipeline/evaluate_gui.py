"""Comprehensive GUI evaluation tool for DSR + EdgeFace pipeline.

This tool evaluates all three VLR resolutions (16×16, 24×24, 32×32) and generates
publication-quality visualizations and metrics suitable for research papers.

Features:
- Multi-resolution comparison (16, 24, 32)
- Verification metrics (FAR, FRR, EER, ROC curves)
- Identification metrics (Rank-1/5/10 accuracy)
- Image quality metrics (PSNR, SSIM, Identity Similarity)
- Comparative visualizations and statistical analysis
- Export results to JSON, CSV, and PDF reports

Usage:
    python -m technical.pipeline.evaluate_gui
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import seaborn as sns

# Try to import tkinter for GUI
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    print("Warning: tkinter not available. GUI mode disabled.")

# Import evaluation functions
try:
    import torch
    from sklearn.metrics import roc_curve, auc
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"Warning: Missing dependencies: {e}")

# Import FinetuneConfig for loading checkpoints
try:
    from ..facial_rec.finetune_edgeface import FinetuneConfig
except ImportError:
    # Define a dummy class if import fails (for backward compatibility)
    @dataclass
    class FinetuneConfig:
        pass


# Set publication-quality plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


@dataclass
class ResolutionMetrics:
    """Metrics for a single VLR resolution."""
    vlr_size: int
    
    # Image quality
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float
    feature_preservation_mean: float  # Renamed from identity_sim_mean
    feature_preservation_std: float   # Renamed from identity_sim_std
    
    # Verification (1:1 matching)
    eer: float
    eer_threshold: float
    roc_auc: float
    tar_at_far_0001: float  # TAR when FAR = 0.01%
    tar_at_far_001: float   # TAR when FAR = 0.1%
    tar_at_far_01: float    # TAR when FAR = 1%
    tar_at_far_10: float    # TAR when FAR = 10%
    d_prime: float          # Separability measure (higher = better separation)
    
    # Distribution statistics
    genuine_score_mean: float
    genuine_score_std: float
    impostor_score_mean: float
    impostor_score_std: float
    
    # Identification at different gallery sizes
    rank1_accuracy_1v1: float = 0.0      # True 1:1 matching (with threshold)
    rank1_accuracy_1v10: float = 0.0     # 1:10 matching (10 identities in gallery)
    rank1_accuracy_1v100: float = 0.0    # 1:100 matching (100 identities)
    rank1_accuracy_1vN: float = 0.0      # 1:N matching (full gallery)
    
    rank5_accuracy_1vN: float = 0.0      # Rank-5 for full gallery
    rank10_accuracy_1vN: float = 0.0     # Rank-10 for full gallery
    rank20_accuracy_1vN: float = 0.0     # Rank-20 for full gallery
    
    # Legacy fields for backward compatibility
    rank1_accuracy: float = 0.0
    rank5_accuracy: float = 0.0
    rank10_accuracy: float = 0.0
    identity_sim_mean: float = 0.0  # Deprecated, use feature_preservation_mean
    identity_sim_std: float = 0.0   # Deprecated, use feature_preservation_std
    
    # Raw data for plotting
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None
    cmc_curve: Optional[List[float]] = None  # Full CMC curve (Rank-1 to Rank-N)


class MultiResolutionEvaluator:
    """Evaluate DSR + EdgeFace pipeline across multiple resolutions."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: Dict[int, ResolutionMetrics] = {}
        
    def evaluate_resolution(
        self,
        vlr_size: int,
        test_root: Path,
        gallery_root: Optional[Path] = None,
        dsr_checkpoint: Optional[Path] = None,
        edgeface_checkpoint: Optional[Path] = None,
        num_samples: Optional[int] = None,
    ) -> ResolutionMetrics:
        """Evaluate a single VLR resolution.
        
        Args:
            vlr_size: VLR input size (16, 24, or 32)
            test_root: Test dataset root
            gallery_root: Gallery dataset root (for identification)
            dsr_checkpoint: Path to DSR checkpoint
            edgeface_checkpoint: Path to EdgeFace checkpoint
            num_samples: Limit number of test samples
            
        Returns:
            ResolutionMetrics object with all computed metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating {vlr_size}×{vlr_size} VLR Resolution")
        print(f"{'='*70}")
        
        # Load models
        from ..dsr import load_dsr_model
        from ..facial_rec.edgeface_weights.edgeface import EdgeFace
        
        # Auto-detect resolution-specific checkpoints
        if dsr_checkpoint is None:
            base_dir = Path(__file__).parents[2] / "technical" / "dsr"
            # Try resolution-specific checkpoint first, fall back to dsr.pth for 32x32
            candidates = [
                base_dir / f"dsr_{vlr_size}x{vlr_size}.pth",
                base_dir / f"dsr{vlr_size}.pth",
            ]
            if vlr_size == 32:
                candidates.append(base_dir / "dsr.pth")
            
            for path in candidates:
                if path.exists():
                    dsr_checkpoint = path
                    break
            
            if dsr_checkpoint is None:
                raise ValueError(f"No DSR checkpoint found for {vlr_size}×{vlr_size}")
        
        if edgeface_checkpoint is None:
            base_dir = Path(__file__).parents[2] / "technical" / "facial_rec" / "edgeface_weights"
            # Try resolution-specific fine-tuned checkpoint first
            candidates = [
                base_dir / f"edgeface_finetuned_{vlr_size}.pth",
                base_dir / f"edgeface_finetuned_{vlr_size}x{vlr_size}.pth",
                base_dir / "edgeface_xxs.pt",  # Fallback to base model
            ]
            
            for path in candidates:
                if path.exists():
                    edgeface_checkpoint = path
                    break
            
            if edgeface_checkpoint is None:
                raise ValueError(f"No EdgeFace checkpoint found for {vlr_size}×{vlr_size}")
        
        print(f"Loading DSR from: {dsr_checkpoint}")
        dsr_model = load_dsr_model(dsr_checkpoint, device=self.device)
        dsr_model.eval()
        
        print(f"Loading EdgeFace from: {edgeface_checkpoint}")
        edgeface_model = EdgeFace(back="edgeface_xxs").to(self.device)
        
        # Load fine-tuned weights if using a fine-tuned checkpoint
        if "finetuned" in str(edgeface_checkpoint):
            print(f"  → Loading fine-tuned weights for {vlr_size}×{vlr_size} resolution")
            
            # Try to load the checkpoint, handling FinetuneConfig import issues
            try:
                checkpoint = torch.load(edgeface_checkpoint, map_location=self.device, weights_only=False)
            except (AttributeError, ModuleNotFoundError) as e:
                # If FinetuneConfig import fails, temporarily inject it into __main__
                import sys
                import __main__
                try:
                    from ..facial_rec.finetune_edgeface import FinetuneConfig
                    __main__.FinetuneConfig = FinetuneConfig
                except:
                    # Create a dummy class if import fails
                    @dataclass
                    class FinetuneConfig:
                        pass
                    __main__.FinetuneConfig = FinetuneConfig
                
                # Retry loading
                checkpoint = torch.load(edgeface_checkpoint, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and "backbone_state_dict" in checkpoint:
                edgeface_model.load_state_dict(checkpoint["backbone_state_dict"], strict=False)
            else:
                edgeface_model.load_state_dict(checkpoint, strict=False)
        
        edgeface_model.eval()
        
        # Determine VLR directory (consistent format for all sizes)
        vlr_dir_name = f"vlr_images_{vlr_size}x{vlr_size}"
        vlr_dir = test_root / vlr_dir_name
        hr_dir = test_root / "hr_images"
        
        if not vlr_dir.exists():
            raise ValueError(f"VLR directory not found: {vlr_dir}")
        
        # Compute image quality metrics
        print("\n📊 Computing image quality metrics...")
        quality_metrics = self._compute_quality_metrics(
            dsr_model, edgeface_model, vlr_dir, hr_dir, num_samples
        )
        
        # Compute verification metrics
        print("\n🔐 Computing verification metrics...")
        verification_metrics = self._compute_verification_metrics(
            dsr_model, edgeface_model, vlr_dir, hr_dir, num_samples
        )
        
        # Compute identification metrics if gallery provided
        if gallery_root:
            print("\n🎯 Computing identification metrics...")
            identification_metrics = self._compute_identification_metrics(
                dsr_model, edgeface_model, gallery_root, test_root, vlr_size, num_samples
            )
        else:
            identification_metrics = {
                "rank1_accuracy": 0.0,
                "rank5_accuracy": 0.0,
                "rank10_accuracy": 0.0,
            }
        
        # Combine all metrics
        metrics = ResolutionMetrics(
            vlr_size=vlr_size,
            **quality_metrics,
            **verification_metrics,
            **identification_metrics,
        )
        
        self.results[vlr_size] = metrics
        return metrics
    
    def _compute_quality_metrics(
        self,
        dsr_model,
        edgeface_model,
        vlr_dir: Path,
        hr_dir: Path,
        num_samples: Optional[int],
    ) -> Dict:
        """Compute PSNR, SSIM, and identity similarity."""
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        vlr_paths = sorted(list(vlr_dir.glob("*.png")))
        if num_samples:
            vlr_paths = vlr_paths[:num_samples]
        
        psnr_scores = []
        ssim_scores = []
        identity_sims = []
        
        for vlr_path in tqdm(vlr_paths, desc="Quality metrics"):
            hr_path = hr_dir / vlr_path.name
            if not hr_path.exists():
                continue
            
            try:
                # Load images
                vlr_img = Image.open(vlr_path).convert("RGB")
                hr_img = Image.open(hr_path).convert("RGB")
                
                # Run DSR
                vlr_tensor = to_tensor(vlr_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    sr_tensor = dsr_model(vlr_tensor)
                    sr_tensor = torch.clamp(sr_tensor, 0, 1)
                
                # Convert to numpy for PSNR/SSIM
                sr_np = (sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                hr_tensor = to_tensor(hr_img).unsqueeze(0).to(self.device)
                hr_np = (hr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Ensure dimensions match (resize HR to match DSR output if needed)
                if sr_np.shape != hr_np.shape:
                    hr_img_resized = hr_img.resize((sr_np.shape[1], sr_np.shape[0]), Image.Resampling.LANCZOS)
                    hr_tensor = to_tensor(hr_img_resized).unsqueeze(0).to(self.device)
                    hr_np = (hr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Compute PSNR and SSIM
                psnr_val = psnr(hr_np, sr_np, data_range=255)
                ssim_val = ssim(hr_np, sr_np, multichannel=True, data_range=255, channel_axis=2)
                
                psnr_scores.append(psnr_val)
                ssim_scores.append(ssim_val)
                
                # Compute identity similarity
                sr_norm = normalize(sr_tensor.squeeze(0)).unsqueeze(0)
                hr_norm = normalize(hr_tensor.squeeze(0)).unsqueeze(0)
                
                with torch.no_grad():
                    sr_emb = edgeface_model(sr_norm)
                    hr_emb = edgeface_model(hr_norm)
                    sim = torch.nn.functional.cosine_similarity(sr_emb, hr_emb, dim=1).item()
                
                identity_sims.append(sim)
                
            except Exception as e:
                print(f"Error processing {vlr_path.name}: {e}")
                continue
        
        return {
            "psnr_mean": float(np.mean(psnr_scores)),
            "psnr_std": float(np.std(psnr_scores)),
            "ssim_mean": float(np.mean(ssim_scores)),
            "ssim_std": float(np.std(ssim_scores)),
            "feature_preservation_mean": float(np.mean(identity_sims)),
            "feature_preservation_std": float(np.std(identity_sims)),
            # Legacy fields for backward compatibility
            "identity_sim_mean": float(np.mean(identity_sims)),
            "identity_sim_std": float(np.std(identity_sims)),
        }
    
    def _compute_verification_metrics(
        self,
        dsr_model,
        edgeface_model,
        vlr_dir: Path,
        hr_dir: Path,
        num_samples: Optional[int],
    ) -> Dict:
        """Compute FAR, FRR, EER, ROC curves."""
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        to_tensor = transforms.ToTensor()
        
        # Compute embeddings for all images
        embeddings_by_subject = {}
        vlr_paths = sorted(list(vlr_dir.glob("*.png")))
        if num_samples:
            vlr_paths = vlr_paths[:num_samples]
        
        for vlr_path in tqdm(vlr_paths, desc="Computing embeddings"):
            subject_id = vlr_path.stem.split("_")[0]
            
            try:
                vlr_img = Image.open(vlr_path).convert("RGB")
                vlr_tensor = to_tensor(vlr_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    sr_tensor = dsr_model(vlr_tensor)
                    sr_tensor = torch.clamp(sr_tensor, 0, 1)
                    sr_norm = normalize(sr_tensor.squeeze(0)).unsqueeze(0)
                    embedding = edgeface_model(sr_norm).cpu()
                
                if subject_id not in embeddings_by_subject:
                    embeddings_by_subject[subject_id] = []
                embeddings_by_subject[subject_id].append(embedding)
                
            except Exception:
                continue
        
        # Generate genuine and impostor pairs
        genuine_scores = []
        impostor_scores = []
        
        subjects = list(embeddings_by_subject.keys())
        
        # Genuine pairs: all intra-class comparisons
        for i, subject_i in enumerate(subjects):
            embs_i = embeddings_by_subject[subject_i]
            
            # Genuine pairs
            for j in range(len(embs_i)):
                for k in range(j + 1, len(embs_i)):
                    score = torch.cosine_similarity(embs_i[j], embs_i[k], dim=1).item()
                    genuine_scores.append(score)
            
            # Impostor pairs: FIXED - use random sampling instead of sequential
            # Sample up to 100 random impostors per subject for better coverage
            import random
            random.seed(42)  # Reproducible results
            num_impostor_samples = min(100, len(subjects) - 1)
            other_subjects = [s for s in subjects if s != subject_i]
            sampled_impostors = random.sample(other_subjects, min(num_impostor_samples, len(other_subjects)))
            
            for subject_j in sampled_impostors:
                embs_j = embeddings_by_subject[subject_j]
                if len(embs_i) > 0 and len(embs_j) > 0:
                    # Use first embedding from each subject
                    score = torch.cosine_similarity(embs_i[0], embs_j[0], dim=1).item()
                    impostor_scores.append(score)
        
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        # Compute ROC curve
        y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        y_scores = np.concatenate([genuine_scores, impostor_scores])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find EER
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
        
        # Find TAR at specific FAR thresholds
        tar_at_far_0001 = tpr[np.argmin(np.abs(fpr - 0.0001))] if np.any(fpr >= 0.0001) else 0.0
        tar_at_far_001 = tpr[np.argmin(np.abs(fpr - 0.001))] if np.any(fpr >= 0.001) else 0.0
        tar_at_far_01 = tpr[np.argmin(np.abs(fpr - 0.01))] if np.any(fpr >= 0.01) else 0.0
        tar_at_far_10 = tpr[np.argmin(np.abs(fpr - 0.1))] if np.any(fpr >= 0.1) else 0.0
        
        # Compute d-prime (separability measure)
        # d' = (μ_genuine - μ_impostor) / sqrt(0.5 * (σ²_genuine + σ²_impostor))
        mu_genuine = np.mean(genuine_scores)
        mu_impostor = np.mean(impostor_scores)
        var_genuine = np.var(genuine_scores)
        var_impostor = np.var(impostor_scores)
        d_prime = (mu_genuine - mu_impostor) / np.sqrt(0.5 * (var_genuine + var_impostor))
        
        return {
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "roc_auc": float(roc_auc),
            "tar_at_far_0001": float(tar_at_far_0001),
            "tar_at_far_001": float(tar_at_far_001),
            "tar_at_far_01": float(tar_at_far_01),
            "tar_at_far_10": float(tar_at_far_10),
            "d_prime": float(d_prime),
            "genuine_score_mean": float(np.mean(genuine_scores)),
            "genuine_score_std": float(np.std(genuine_scores)),
            "impostor_score_mean": float(np.mean(impostor_scores)),
            "impostor_score_std": float(np.std(impostor_scores)),
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
    
    def _compute_identification_metrics(
        self,
        dsr_model,
        edgeface_model,
        gallery_root: Path,
        test_root: Path,
        vlr_size: int,
        num_samples: Optional[int],
    ) -> Dict:
        """Compute identification metrics at different gallery sizes: 1:1, 1:10, 1:100, 1:N."""
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        to_tensor = transforms.ToTensor()
        
        # Build full gallery
        print("  Building gallery embeddings...")
        gallery_embeddings = {}
        gallery_hr_dir = gallery_root / "hr_images"
        
        for hr_path in tqdm(sorted(gallery_hr_dir.glob("*.png")), desc="  Gallery enrollment"):
            subject_id = hr_path.stem.split("_")[0]
            
            try:
                hr_img = Image.open(hr_path).convert("RGB")
                hr_tensor = to_tensor(hr_img).unsqueeze(0).to(self.device)
                hr_norm = normalize(hr_tensor.squeeze(0)).unsqueeze(0)
                
                with torch.no_grad():
                    embedding = edgeface_model(hr_norm).cpu()
                
                if subject_id not in gallery_embeddings:
                    gallery_embeddings[subject_id] = []
                gallery_embeddings[subject_id].append(embedding)
                
            except Exception:
                continue
        
        # Average embeddings per subject to create gallery templates
        gallery_templates = {
            subj: torch.mean(torch.stack(embs), dim=0)
            for subj, embs in gallery_embeddings.items()
        }
        
        print(f"  Gallery enrolled: {len(gallery_templates)} identities")
        
        # Get list of gallery subjects for subsampling
        gallery_subjects = list(gallery_templates.keys())
        
        # Evaluate probes
        vlr_dir_name = f"vlr_images_{vlr_size}x{vlr_size}"
        vlr_dir = test_root / vlr_dir_name
        
        # Results storage for different gallery sizes
        results_1v1 = {"rank1": 0, "total": 0}
        results_1v10 = {"rank1": 0, "total": 0}
        results_1v100 = {"rank1": 0, "total": 0}
        results_1vN = {"rank1": 0, "rank5": 0, "rank10": 0, "rank20": 0, "total": 0}
        cmc_ranks = []  # Store all ranks for CMC curve
        
        vlr_paths = sorted(list(vlr_dir.glob("*.png")))
        if num_samples:
            vlr_paths = vlr_paths[:num_samples]
        
        skipped_count = 0
        for vlr_path in tqdm(vlr_paths, desc="  Identification tests"):
            subject_id = vlr_path.stem.split("_")[0]
            
            if subject_id not in gallery_templates:
                skipped_count += 1
                continue
            
            try:
                # Process probe image
                vlr_img = Image.open(vlr_path).convert("RGB")
                vlr_tensor = to_tensor(vlr_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    sr_tensor = dsr_model(vlr_tensor)
                    sr_tensor = torch.clamp(sr_tensor, 0, 1)
                    sr_norm = normalize(sr_tensor.squeeze(0)).unsqueeze(0)
                    probe_emb = edgeface_model(sr_norm).cpu()
                
                # === 1:1 Matching ===
                # FIXED: Apply threshold instead of always counting as correct
                # Use a reasonable threshold (e.g., cosine similarity > 0.5 or > EER threshold)
                # For now, we'll consider it "accepted" if similarity > 0.5
                true_sim = torch.cosine_similarity(probe_emb, gallery_templates[subject_id], dim=1).item()
                if true_sim >= 0.5:  # Threshold for acceptance
                    results_1v1["rank1"] += 1
                results_1v1["total"] += 1
                
                # === 1:10 Matching ===
                # Create gallery of 10 identities including the true one
                if len(gallery_subjects) >= 10:
                    # Ensure true identity is included
                    distractors_10 = [s for s in gallery_subjects if s != subject_id]
                    import random
                    random.seed(hash(vlr_path.name))  # Deterministic per image
                    selected_10 = random.sample(distractors_10, min(9, len(distractors_10)))
                    gallery_10 = [subject_id] + selected_10
                    
                    similarities_10 = {
                        subj: torch.cosine_similarity(probe_emb, gallery_templates[subj], dim=1).item()
                        for subj in gallery_10
                    }
                    ranked_10 = sorted(similarities_10.items(), key=lambda x: x[1], reverse=True)
                    if ranked_10[0][0] == subject_id:
                        results_1v10["rank1"] += 1
                    results_1v10["total"] += 1
                
                # === 1:100 Matching ===
                # Create gallery of 100 identities including the true one
                if len(gallery_subjects) >= 100:
                    distractors_100 = [s for s in gallery_subjects if s != subject_id]
                    random.seed(hash(vlr_path.name))
                    selected_100 = random.sample(distractors_100, min(99, len(distractors_100)))
                    gallery_100 = [subject_id] + selected_100
                    
                    similarities_100 = {
                        subj: torch.cosine_similarity(probe_emb, gallery_templates[subj], dim=1).item()
                        for subj in gallery_100
                    }
                    ranked_100 = sorted(similarities_100.items(), key=lambda x: x[1], reverse=True)
                    if ranked_100[0][0] == subject_id:
                        results_1v100["rank1"] += 1
                    results_1v100["total"] += 1
                
                # === 1:N Matching (Full Gallery) ===
                similarities_full = {
                    subj: torch.cosine_similarity(probe_emb, gal_emb, dim=1).item()
                    for subj, gal_emb in gallery_templates.items()
                }
                
                ranked_full = sorted(similarities_full.items(), key=lambda x: x[1], reverse=True)
                ranked_subjects = [subj for subj, _ in ranked_full]
                
                if subject_id in ranked_subjects:
                    rank = ranked_subjects.index(subject_id) + 1
                    cmc_ranks.append(rank)  # Store rank for CMC curve
                    if rank == 1:
                        results_1vN["rank1"] += 1
                    if rank <= 5:
                        results_1vN["rank5"] += 1
                    if rank <= 10:
                        results_1vN["rank10"] += 1
                    if rank <= 20:
                        results_1vN["rank20"] += 1
                else:
                    cmc_ranks.append(len(gallery_templates) + 1)  # Not found
                
                results_1vN["total"] += 1
                
            except Exception as e:
                print(f"    Error processing {vlr_path.name}: {e}")
                continue
        
        # Print summary
        print(f"  Completed: {results_1vN['total']} probes evaluated ({skipped_count} skipped - not in gallery)")
        
        if results_1vN['total'] == 0:
            print("  ⚠️  WARNING: No test subjects found in gallery (open-set scenario)")
            print("     For closed-set evaluation, use test HR images as gallery or ensure overlap")
        
        # Compute CMC curve
        max_rank = min(100, len(gallery_templates))  # CMC up to rank 100 or gallery size
        cmc_curve = []
        for r in range(1, max_rank + 1):
            cmc_curve.append(sum(1 for rank in cmc_ranks if rank <= r) / len(cmc_ranks) if cmc_ranks else 0.0)
        
        # Compute accuracies
        return {
            "rank1_accuracy_1v1": results_1v1["rank1"] / results_1v1["total"] if results_1v1["total"] > 0 else 0.0,
            "rank1_accuracy_1v10": results_1v10["rank1"] / results_1v10["total"] if results_1v10["total"] > 0 else 0.0,
            "rank1_accuracy_1v100": results_1v100["rank1"] / results_1v100["total"] if results_1v100["total"] > 0 else 0.0,
            "rank1_accuracy_1vN": results_1vN["rank1"] / results_1vN["total"] if results_1vN["total"] > 0 else 0.0,
            "rank5_accuracy_1vN": results_1vN["rank5"] / results_1vN["total"] if results_1vN["total"] > 0 else 0.0,
            "rank10_accuracy_1vN": results_1vN["rank10"] / results_1vN["total"] if results_1vN["total"] > 0 else 0.0,
            "rank20_accuracy_1vN": results_1vN["rank20"] / results_1vN["total"] if results_1vN["total"] > 0 else 0.0,
            # Legacy fields
            "rank1_accuracy": results_1vN["rank1"] / results_1vN["total"] if results_1vN["total"] > 0 else 0.0,
            "rank5_accuracy": results_1vN["rank5"] / results_1vN["total"] if results_1vN["total"] > 0 else 0.0,
            "rank10_accuracy": results_1vN["rank10"] / results_1vN["total"] if results_1vN["total"] > 0 else 0.0,
            # CMC curve
            "cmc_curve": cmc_curve,
        }
    
    def generate_comparative_plots(self, output_dir: Path):
        """Generate publication-quality comparative plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            print("No results to plot!")
            return
        
        resolutions = sorted(self.results.keys())
        
        # Create comprehensive PDF report
        pdf_path = output_dir / "evaluation_report.pdf"
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary Table
            fig = self._plot_summary_table(resolutions)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Image Quality Metrics
            fig = self._plot_quality_metrics(resolutions)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 3: ROC Curves
            fig = self._plot_roc_curves(resolutions)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 4: Score Distributions
            fig = self._plot_score_distributions(resolutions)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 5: Verification Metrics Comparison
            fig = self._plot_verification_comparison(resolutions)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 6: Identification Accuracy (Multi-Scale)
            fig = self._plot_identification_accuracy(resolutions)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 7: CMC Curve
            fig = self._plot_cmc_curve(resolutions)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"\n✅ PDF report saved to: {pdf_path}")
        
        # Also save individual plots
        self._save_individual_plots(output_dir, resolutions)
    
    def _plot_quality_metrics(self, resolutions: List[int]) -> Figure:
        """Plot PSNR, SSIM, Feature Preservation comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Image Quality Metrics Comparison', fontsize=14, fontweight='bold')
        
        metrics = [
            ('PSNR (dB)', 'psnr_mean', 'psnr_std'),
            ('SSIM', 'ssim_mean', 'ssim_std'),
            ('Feature Preservation', 'feature_preservation_mean', 'feature_preservation_std'),
        ]
        
        for ax, (title, mean_key, std_key) in zip(axes, metrics):
            means = [getattr(self.results[r], mean_key) for r in resolutions]
            stds = [getattr(self.results[r], std_key) for r in resolutions]
            
            x = np.arange(len(resolutions))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=sns.color_palette("husl", len(resolutions)))
            ax.set_xticks(x)
            ax.set_xticklabels([f'{r}×{r}' for r in resolutions])
            ax.set_xlabel('VLR Resolution')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.02 * max(means), f'{mean:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def _plot_roc_curves(self, resolutions: List[int]) -> Figure:
        """Plot ROC curves for all resolutions."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = sns.color_palette("husl", len(resolutions))
        
        for i, res in enumerate(resolutions):
            metrics = self.results[res]
            if metrics.fpr is not None:
                ax.plot(metrics.fpr, metrics.tpr, 
                       label=f'{res}×{res} (AUC={metrics.roc_auc:.4f})',
                       color=colors[i], linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate (FAR)')
        ax.set_ylabel('True Positive Rate (TAR)')
        ax.set_title('ROC Curves: Verification Performance', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig
    
    def _plot_score_distributions(self, resolutions: List[int]) -> Figure:
        """Plot genuine vs impostor score distributions."""
        fig, axes = plt.subplots(1, len(resolutions), figsize=(5*len(resolutions), 4))
        if len(resolutions) == 1:
            axes = [axes]
        
        fig.suptitle('Genuine vs Impostor Score Distributions', fontsize=14, fontweight='bold')
        
        for ax, res in zip(axes, resolutions):
            metrics = self.results[res]
            
            # Create synthetic distributions from mean and std
            genuine_samples = np.random.normal(metrics.genuine_score_mean, metrics.genuine_score_std, 1000)
            impostor_samples = np.random.normal(metrics.impostor_score_mean, metrics.impostor_score_std, 1000)
            
            ax.hist(impostor_samples, bins=50, alpha=0.6, label='Impostor', color='red', density=True)
            ax.hist(genuine_samples, bins=50, alpha=0.6, label='Genuine', color='green', density=True)
            ax.axvline(metrics.eer_threshold, color='blue', linestyle='--', linewidth=2, label=f'EER Threshold ({metrics.eer_threshold:.3f})')
            
            ax.set_xlabel('Cosine Similarity Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{res}×{res} VLR')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_verification_comparison(self, resolutions: List[int]) -> Figure:
        """Plot verification metrics comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle('Verification Metrics Comparison', fontsize=14, fontweight='bold')
        
        # EER comparison
        eers = [self.results[r].eer for r in resolutions]
        x = np.arange(len(resolutions))
        
        axes[0].bar(x, eers, alpha=0.7, color=sns.color_palette("husl", len(resolutions)))
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'{r}×{r}' for r in resolutions])
        axes[0].set_xlabel('VLR Resolution')
        axes[0].set_ylabel('Equal Error Rate (EER)')
        axes[0].set_title('Equal Error Rate (Lower is Better)')
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, eer in enumerate(eers):
            axes[0].text(i, eer + 0.005, f'{eer:.4f}', ha='center', va='bottom', fontsize=9)
        
        # d-prime comparison
        dprimes = [self.results[r].d_prime for r in resolutions]
        
        axes[1].bar(x, dprimes, alpha=0.7, color=sns.color_palette("husl", len(resolutions)))
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'{r}×{r}' for r in resolutions])
        axes[1].set_xlabel('VLR Resolution')
        axes[1].set_ylabel("d-prime (d')")
        axes[1].set_title('Separability d-prime (Higher is Better)')
        axes[1].grid(axis='y', alpha=0.3)
        
        for i, dp in enumerate(dprimes):
            axes[1].text(i, dp + 0.05, f'{dp:.3f}', ha='center', va='bottom', fontsize=9)
        
        # TAR at different FAR levels
        tar_0001 = [self.results[r].tar_at_far_0001 for r in resolutions]
        tar_001 = [self.results[r].tar_at_far_001 for r in resolutions]
        tar_01 = [self.results[r].tar_at_far_01 for r in resolutions]
        tar_10 = [self.results[r].tar_at_far_10 for r in resolutions]
        
        width = 0.2
        axes[2].bar(x - 1.5*width, tar_0001, width, label='FAR = 0.01%', alpha=0.7)
        axes[2].bar(x - 0.5*width, tar_001, width, label='FAR = 0.1%', alpha=0.7)
        axes[2].bar(x + 0.5*width, tar_01, width, label='FAR = 1%', alpha=0.7)
        axes[2].bar(x + 1.5*width, tar_10, width, label='FAR = 10%', alpha=0.7)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f'{r}×{r}' for r in resolutions])
        axes[2].set_xlabel('VLR Resolution')
        axes[2].set_ylabel('True Accept Rate (TAR)')
        axes[2].set_title('TAR at Fixed FAR Thresholds (Higher is Better)')
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_identification_accuracy(self, resolutions: List[int]) -> Figure:
        """Plot multi-scale identification accuracy comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left plot: 1:1, 1:10, 1:100, 1:N Rank-1 accuracy
        scenarios = ['1:1', '1:10', '1:100', '1:N']
        x = np.arange(len(resolutions))
        width = 0.2
        
        for i, scenario in enumerate(scenarios):
            if scenario == '1:1':
                values = [self.results[r].rank1_accuracy_1v1 * 100 for r in resolutions]
            elif scenario == '1:10':
                values = [self.results[r].rank1_accuracy_1v10 * 100 for r in resolutions]
            elif scenario == '1:100':
                values = [self.results[r].rank1_accuracy_1v100 * 100 for r in resolutions]
            else:  # 1:N
                values = [self.results[r].rank1_accuracy_1vN * 100 for r in resolutions]
            
            ax1.bar(x + (i - 1.5) * width, values, width, label=scenario, alpha=0.8)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{r}×{r}' for r in resolutions])
        ax1.set_xlabel('VLR Resolution')
        ax1.set_ylabel('Rank-1 Accuracy (%)')
        ax1.set_title('Multi-Scale Identification (Rank-1)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 105])
        
        # Right plot: 1:N with different ranks
        rank1 = [self.results[r].rank1_accuracy_1vN * 100 for r in resolutions]
        rank5 = [self.results[r].rank5_accuracy_1vN * 100 for r in resolutions]
        rank10 = [self.results[r].rank10_accuracy_1vN * 100 for r in resolutions]
        rank20 = [self.results[r].rank20_accuracy_1vN * 100 for r in resolutions]
        
        width = 0.2
        ax2.bar(x - 1.5*width, rank1, width, label='Rank-1', alpha=0.8)
        ax2.bar(x - 0.5*width, rank5, width, label='Rank-5', alpha=0.8)
        ax2.bar(x + 0.5*width, rank10, width, label='Rank-10', alpha=0.8)
        ax2.bar(x + 1.5*width, rank20, width, label='Rank-20', alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{r}×{r}' for r in resolutions])
        ax2.set_xlabel('VLR Resolution')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('1:N Identification (Multi-Rank)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 105])
        
        plt.tight_layout()
        return fig
    
    def _plot_summary_table(self, resolutions: List[int]) -> Figure:
        """Create summary table of all metrics."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        headers = ['Metric'] + [f'{r}×{r}' for r in resolutions]
        
        rows = [
            ['PSNR (dB)'] + [f'{self.results[r].psnr_mean:.2f} ± {self.results[r].psnr_std:.2f}' for r in resolutions],
            ['SSIM'] + [f'{self.results[r].ssim_mean:.4f} ± {self.results[r].ssim_std:.4f}' for r in resolutions],
            ['Feature Preservation'] + [f'{self.results[r].feature_preservation_mean:.4f} ± {self.results[r].feature_preservation_std:.4f}' for r in resolutions],
            [''] + [''] * len(resolutions),  # Separator
            ['EER'] + [f'{self.results[r].eer:.4f}' for r in resolutions],
            ['ROC AUC'] + [f'{self.results[r].roc_auc:.4f}' for r in resolutions],
            ["d' (separability)"] + [f'{self.results[r].d_prime:.3f}' for r in resolutions],
            ['TAR @ FAR=0.01%'] + [f'{self.results[r].tar_at_far_0001:.4f}' for r in resolutions],
            ['TAR @ FAR=0.1%'] + [f'{self.results[r].tar_at_far_001:.4f}' for r in resolutions],
            ['TAR @ FAR=1%'] + [f'{self.results[r].tar_at_far_01:.4f}' for r in resolutions],
            ['TAR @ FAR=10%'] + [f'{self.results[r].tar_at_far_10:.4f}' for r in resolutions],
            [''] + [''] * len(resolutions),  # Separator
            ['1:1 Rank-1'] + [f'{self.results[r].rank1_accuracy_1v1:.4f}' if self.results[r].rank1_accuracy_1v1 > 0 else 'N/A' for r in resolutions],
            ['1:10 Rank-1'] + [f'{self.results[r].rank1_accuracy_1v10:.4f}' if self.results[r].rank1_accuracy_1v10 > 0 else 'N/A' for r in resolutions],
            ['1:100 Rank-1'] + [f'{self.results[r].rank1_accuracy_1v100:.4f}' if self.results[r].rank1_accuracy_1v100 > 0 else 'N/A' for r in resolutions],
            ['1:N Rank-1'] + [f'{self.results[r].rank1_accuracy_1vN:.4f}' if self.results[r].rank1_accuracy_1vN > 0 else 'N/A' for r in resolutions],
            ['1:N Rank-5'] + [f'{self.results[r].rank5_accuracy_1vN:.4f}' if self.results[r].rank5_accuracy_1vN > 0 else 'N/A' for r in resolutions],
            ['1:N Rank-10'] + [f'{self.results[r].rank10_accuracy_1vN:.4f}' if self.results[r].rank10_accuracy_1vN > 0 else 'N/A' for r in resolutions],
            ['1:N Rank-20'] + [f'{self.results[r].rank20_accuracy_1vN:.4f}' if self.results[r].rank20_accuracy_1vN > 0 else 'N/A' for r in resolutions],
        ]
        
        table = ax.table(cellText=[headers] + rows, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        
        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(rows) + 1):
            if rows[i-1][0] == '':
                continue
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)
        
        plt.title('Comprehensive Evaluation Summary', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def _plot_cmc_curve(self, resolutions: List[int]) -> Figure:
        """Plot Cumulative Match Characteristic (CMC) curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for resolution in resolutions:
            metrics = self.results[resolution]
            if metrics.cmc_curve and len(metrics.cmc_curve) > 0:
                ranks = list(range(1, len(metrics.cmc_curve) + 1))
                cmc_values = [v * 100 for v in metrics.cmc_curve]  # Convert to percentage
                ax.plot(ranks, cmc_values, marker='o', markersize=3, 
                       label=f'{resolution}×{resolution}', linewidth=2)
        
        ax.set_xlabel('Rank', fontsize=11)
        ax.set_ylabel('Cumulative Match Rate (%)', fontsize=11)
        ax.set_title('CMC Curve (1:N Identification)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([1, 100])
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        return fig
    
    def _save_individual_plots(self, output_dir: Path, resolutions: List[int]):
        """Save individual plots as PNG files."""
        # ROC curves
        fig = self._plot_roc_curves(resolutions)
        fig.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Quality metrics
        fig = self._plot_quality_metrics(resolutions)
        fig.savefig(output_dir / 'quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Score distributions
        fig = self._plot_score_distributions(resolutions)
        fig.savefig(output_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Verification comparison
        fig = self._plot_verification_comparison(resolutions)
        fig.savefig(output_dir / 'verification_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Identification accuracy
        fig = self._plot_identification_accuracy(resolutions)
        fig.savefig(output_dir / 'identification_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # CMC curve
        fig = self._plot_cmc_curve(resolutions)
        fig.savefig(output_dir / 'cmc_curve.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Individual plots saved to: {output_dir}")
    
    def export_results(self, output_path: Path):
        """Export results to JSON."""
        output_data = {}
        
        for vlr_size, metrics in self.results.items():
            # Convert to dict and remove numpy arrays
            metrics_dict = asdict(metrics)
            metrics_dict.pop('fpr', None)
            metrics_dict.pop('tpr', None)
            metrics_dict.pop('thresholds', None)
            
            output_data[f'{vlr_size}x{vlr_size}'] = metrics_dict
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Results exported to: {output_path}")


class EvaluationGUI:
    """GUI application for comprehensive evaluation."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("LRFR Comprehensive Evaluation Tool")
        self.root.geometry("900x700")
        
        self.evaluator = MultiResolutionEvaluator()
        self.create_widgets()
    
    def create_widgets(self):
        """Create GUI widgets."""
        # Title
        title = ttk.Label(self.root, text="Low-Resolution Face Recognition Evaluation", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Configuration Frame
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        # Test dataset path
        ttk.Label(config_frame, text="Test Dataset:").grid(row=0, column=0, sticky='w', pady=2)
        self.test_path = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.test_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_test).grid(row=0, column=2)
        
        # Gallery dataset path
        ttk.Label(config_frame, text="Gallery Dataset (optional):").grid(row=1, column=0, sticky='w', pady=2)
        self.gallery_path = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.gallery_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_gallery).grid(row=1, column=2)
        
        # Output directory
        ttk.Label(config_frame, text="Output Directory:").grid(row=2, column=0, sticky='w', pady=2)
        self.output_path = tk.StringVar(value=str(Path.cwd() / "evaluation_results"))
        ttk.Entry(config_frame, textvariable=self.output_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_output).grid(row=2, column=2)
        
        # Resolution selection
        res_frame = ttk.LabelFrame(self.root, text="VLR Resolutions to Evaluate", padding=10)
        res_frame.pack(fill='x', padx=10, pady=5)
        
        self.res_16 = tk.BooleanVar(value=True)
        self.res_24 = tk.BooleanVar(value=True)
        self.res_32 = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(res_frame, text="16×16", variable=self.res_16).pack(side='left', padx=10)
        ttk.Checkbutton(res_frame, text="24×24", variable=self.res_24).pack(side='left', padx=10)
        ttk.Checkbutton(res_frame, text="32×32", variable=self.res_32).pack(side='left', padx=10)
        
        # Device selection
        device_frame = ttk.Frame(self.root)
        device_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(device_frame, text="Device:").pack(side='left', padx=5)
        self.device = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        ttk.Radiobutton(device_frame, text="CUDA", variable=self.device, value="cuda").pack(side='left', padx=5)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device, value="cpu").pack(side='left', padx=5)
        
        # Progress and Log
        log_frame = ttk.LabelFrame(self.root, text="Progress", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, state='disabled')
        self.log_text.pack(fill='both', expand=True)
        
        self.progress = ttk.Progressbar(log_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.run_button = ttk.Button(button_frame, text="Run Evaluation", command=self.run_evaluation)
        self.run_button.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side='right', padx=5)
    
    def browse_test(self):
        path = filedialog.askdirectory(title="Select Test Dataset Root")
        if path:
            self.test_path.set(path)
    
    def browse_gallery(self):
        path = filedialog.askdirectory(title="Select Gallery Dataset Root")
        if path:
            self.gallery_path.set(path)
    
    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_path.set(path)
    
    def log(self, message):
        """Add message to log."""
        self.log_text.configure(state='normal')
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')
        self.root.update()
    
    def run_evaluation(self):
        """Run the evaluation process."""
        # Validate inputs
        if not self.test_path.get():
            messagebox.showerror("Error", "Please select a test dataset!")
            return
        
        resolutions = []
        if self.res_16.get():
            resolutions.append(16)
        if self.res_24.get():
            resolutions.append(24)
        if self.res_32.get():
            resolutions.append(32)
        
        if not resolutions:
            messagebox.showerror("Error", "Please select at least one resolution!")
            return
        
        # Disable button and start progress
        self.run_button.configure(state='disabled')
        self.progress.start()
        
        try:
            test_root = Path(self.test_path.get())
            gallery_root = Path(self.gallery_path.get()) if self.gallery_path.get() else None
            output_dir = Path(self.output_path.get())
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.evaluator = MultiResolutionEvaluator(device=self.device.get())
            
            # Evaluate each resolution
            for vlr_size in resolutions:
                self.log(f"\n{'='*60}")
                self.log(f"Evaluating {vlr_size}×{vlr_size} Resolution")
                self.log(f"{'='*60}")
                
                try:
                    metrics = self.evaluator.evaluate_resolution(
                        vlr_size=vlr_size,
                        test_root=test_root,
                        gallery_root=gallery_root,
                        num_samples=None,
                    )
                    
                    self.log(f"✅ {vlr_size}×{vlr_size} evaluation complete!")
                    self.log(f"   PSNR: {metrics.psnr_mean:.2f} dB")
                    self.log(f"   SSIM: {metrics.ssim_mean:.4f}")
                    self.log(f"   EER: {metrics.eer:.4f}")
                    self.log(f"   Rank-1: {metrics.rank1_accuracy:.2%}")
                    
                except Exception as e:
                    self.log(f"❌ Error evaluating {vlr_size}×{vlr_size}: {str(e)}")
            
            # Generate plots and reports
            self.log("\n📊 Generating plots and reports...")
            self.evaluator.generate_comparative_plots(output_dir)
            
            # Export results
            self.evaluator.export_results(output_dir / "results.json")
            
            self.log("\n✅ Evaluation complete!")
            self.log(f"📁 Results saved to: {output_dir}")
            
            messagebox.showinfo("Success", f"Evaluation complete!\n\nResults saved to:\n{output_dir}")
            
        except Exception as e:
            self.log(f"\n❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Evaluation failed:\n{str(e)}")
        
        finally:
            self.progress.stop()
            self.run_button.configure(state='normal')


def main():
    """Main entry point."""
    if not HAS_DEPS:
        print("Error: Missing required dependencies (torch, sklearn, PIL, etc.)")
        sys.exit(1)
    
    if HAS_GUI:
        root = tk.Tk()
        app = EvaluationGUI(root)
        root.mainloop()
    else:
        print("Error: tkinter not available. Please install python3-tk")
        print("\nRunning in command-line mode...")
        
        # Simple CLI fallback
        import argparse
        parser = argparse.ArgumentParser(description="Comprehensive evaluation (CLI mode)")
        parser.add_argument("--test-root", type=Path, required=True)
        parser.add_argument("--gallery-root", type=Path, default=None)
        parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
        parser.add_argument("--resolutions", nargs='+', type=int, default=[16, 24, 32])
        parser.add_argument("--device", default="cuda")
        
        args = parser.parse_args()
        
        evaluator = MultiResolutionEvaluator(device=args.device)
        
        for vlr_size in args.resolutions:
            evaluator.evaluate_resolution(
                vlr_size=vlr_size,
                test_root=args.test_root,
                gallery_root=args.gallery_root,
            )
        
        evaluator.generate_comparative_plots(args.output_dir)
        evaluator.export_results(args.output_dir / "results.json")


if __name__ == "__main__":
    main()
