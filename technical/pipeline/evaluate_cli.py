"""CLI for evaluating the face recognition pipeline with comprehensive metrics, baselines, and PDF reporting.

This script evaluates the pipeline on a test dataset, comparing DSR against a Bicubic baseline.
It computes:
- Recognition (1:N): Top-1, Top-5 Accuracy
- Verification (1:1): EER, ROC AUC, TAR @ FAR, Max Accuracy
- Image Quality: PSNR, SSIM
- Performance: Inference Time, FPS, Memory

It generates:
- full_results.json: Raw metrics
- benchmark_report.pdf: Professional report with tables, ROC curves, and score distributions

Usage:
    python -m technical.pipeline.evaluate_cli --resolutions 16 24 32 --output-dir evaluation_results
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from sklearn.metrics import roc_curve, auc, accuracy_score

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from technical.pipeline.pipeline import PipelineConfig, build_pipeline

# Set style for plots
sns.set_theme(style="whitegrid")

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM)."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(gray1, gray2, data_range=255)
    except ImportError:
        return -1.0

def compute_verification_metrics(
    probe_embeddings: torch.Tensor, 
    probe_labels: List[str], 
    gallery_embeddings: torch.Tensor, 
    gallery_labels: List[str]
) -> Dict[str, Any]:
    """Compute 1:1 verification metrics (EER, ROC, TAR@FAR) using 10-fold Cross-Validation."""
    
    probe_embeddings = F.normalize(probe_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)
    
    # Similarity Matrix: (N_probes, N_gallery)
    sim_matrix = torch.mm(probe_embeddings, gallery_embeddings.t()).cpu().numpy()
    
    probe_labels_np = np.array(probe_labels)
    gallery_labels_np = np.array(gallery_labels)
    
    # Identify unique subjects for subject-disjoint CV
    unique_subjects = np.unique(np.concatenate([probe_labels_np, gallery_labels_np]))
    from sklearn.model_selection import KFold
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    accuracies = []
    thresholds = []
    
    # Store all scores for global ROC/EER (standard practice usually reports average or global)
    # Here we keep global ROC for plotting, but Accuracy is averaged.
    
    # Global Flat Lists (for ROC)
    match_mask = (probe_labels_np[:, None] == gallery_labels_np[None, :])
    y_true_global = match_mask.flatten()
    y_scores_global = sim_matrix.flatten()
    
    # 10-Fold CV for Accuracy
    # We split SUBJECTS, not pairs, to ensure strict separation
    if len(unique_subjects) < k_folds:
        print(f"⚠️ Not enough subjects ({len(unique_subjects)}) for {k_folds}-fold CV. Falling back to single split.")
        current_k_folds = 1
        splits = [(list(range(len(unique_subjects))), list(range(len(unique_subjects))))]
    else:
        current_k_folds = k_folds
        splits = list(kf.split(unique_subjects))
        
    for train_idx, test_idx in splits:
        train_subjects = unique_subjects[train_idx]
        test_subjects = unique_subjects[test_idx]
        
        # Create masks for current fold based on SUBJECTS
        # A pair is in Train if BOTH subjects are in Train
        # A pair is in Test if BOTH subjects are in Test (strict) 
        # or just filter pairs belonging to test subjects?
        # Standard LFW: Fold i is test, others are train.
        
        # Filter indices in probe/gallery corresponding to train/test subjects
        # This is tricky with pre-computed sim matrix.
        # Let's create boolean masks for columns(gallery) and rows(probe)
        
        # Train Mask
        train_mask_p = np.isin(probe_labels_np, train_subjects)
        train_mask_g = np.isin(gallery_labels_np, train_subjects)
        
        # Test Mask
        test_mask_p = np.isin(probe_labels_np, test_subjects)
        test_mask_g = np.isin(gallery_labels_np, test_subjects)
        
        # Train Scores
        # We only care about sub-matrix where both probe and gallery are train
        # Flattening sub-matrix
        train_sims = sim_matrix[np.ix_(train_mask_p, train_mask_g)].flatten()
        train_y = match_mask[np.ix_(train_mask_p, train_mask_g)].flatten()
        
        # Test Scores
        test_sims = sim_matrix[np.ix_(test_mask_p, test_mask_g)].flatten()
        test_y = match_mask[np.ix_(test_mask_p, test_mask_g)].flatten()
        
        if len(train_y) == 0 or len(test_y) == 0:
            continue
            
        # Find best threshold on TRAIN
        # Efficient search
        best_acc = 0
        best_th = 0
        # Sample thresholds from distribution or linspace
        # Linspace 0-1 is safe for cosine sim (normalized)
        # Taking subset of scores as candidate thresholds is faster/more accurate
        # But for speed, linspace is fine
        for th in np.linspace(0, 1, 100):
            batch_acc = np.mean((train_sims >= th) == train_y)
            if batch_acc > best_acc:
                best_acc = batch_acc
                best_th = th
        
        thresholds.append(best_th)
        
        # Apply to TEST
        test_acc = np.mean((test_sims >= best_th) == test_y)
        accuracies.append(test_acc)
        
    if not accuracies:
        mean_acc = 0.0
        std_acc = 0.0
    else:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
    
    # ROC (Global)
    fpr, tpr, _ = roc_curve(y_true_global, y_scores_global)
    roc_auc = auc(fpr, tpr)
    
    # EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    
    def get_tar_at_far(target_far):
        idx = np.where(fpr <= target_far)[0]
        if len(idx) > 0:
            return tpr[idx[-1]]
        return 0.0
        
    metrics = {
        "eer": float(eer),
        "auc": float(roc_auc),
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "tar_far_01": float(get_tar_at_far(0.01)),
        "tar_far_001": float(get_tar_at_far(0.001)),
        "fpr": fpr,
        "tpr": tpr,
        "genuine_scores": y_scores_global[y_true_global],
        "impostor_scores": y_scores_global[~y_true_global]
    }
    
    return metrics

def evaluate_resolution(
    vlr_size: int,
    test_dir: Path,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Evaluate pipeline for a specific VLR resolution."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating Resolution: {vlr_size}x{vlr_size}")
    print(f"{'='*60}")
    
    vlr_dir = test_dir / f"vlr_images_{vlr_size}x{vlr_size}"
    hr_dir = test_dir / "hr_images"
    
    if not vlr_dir.exists():
        return {"error": "VLR directory missing"}
        
    print("Loading pipeline...")
    config = PipelineConfig(
        dsr_weights_path=Path(f"technical/dsr/hybrid_dsr{vlr_size}.pth"),
        device=device,
        skip_dsr=False
    )
    
    if not Path(config.dsr_weights_path).exists() and not (ROOT / config.dsr_weights_path).exists():
        print(f"⚠️  Hybrid DSR model not found for {vlr_size}, trying legacy...")
        config.dsr_weights_path = Path(f"technical/dsr/dsr{vlr_size}.pth")
    
    try:
        pipeline = build_pipeline(config)
    except Exception as e:
        return {"error": str(e)}
        
    image_files = sorted(list(vlr_dir.glob("*.png")))
    if not image_files:
        return {"error": "No images"}
        
    print(f"Found {len(image_files)} test images")
    
    # 1. Build Gallery (HR)
    print("Building Gallery (HR)...")
    subject_map: Dict[str, List[Path]] = {}
    for img_path in image_files:
        subject_id = img_path.stem.split('_')[0]
        if subject_id not in subject_map:
            subject_map[subject_id] = []
        subject_map[subject_id].append(img_path)
        
    gallery_embeddings_list = []
    gallery_labels_list = []
    
    for subject_id, paths in tqdm(subject_map.items(), desc="Registering"):
        ref_hr_path = hr_dir / paths[0].name
        if ref_hr_path.exists():
            hr_img = Image.open(ref_hr_path).convert("RGB")
            hr_tensor = pipeline._to_tensor(hr_img)
            emb = pipeline.infer_embedding(hr_tensor, use_tta=False)
            pipeline.gallery.add(subject_id, emb)
            
            gallery_embeddings_list.append(emb.cpu())
            gallery_labels_list.append(subject_id)
            
    if not gallery_embeddings_list:
        return {"error": "Empty gallery"}
        
    gallery_tensor = torch.stack(gallery_embeddings_list).to(device)
    
    # 2. Evaluate Probes
    print("Evaluating Probes (DSR & Bicubic)...")
    
    # DSR Metrics
    dsr_correct_top1 = 0
    dsr_correct_top5 = 0
    dsr_psnr = 0.0
    dsr_ssim = 0.0
    dsr_times = []
    dsr_upscale_times = []
    dsr_rec_times = []
    dsr_embeddings = []
    
    # Bicubic Metrics
    bic_correct_top1 = 0
    bic_correct_top5 = 0
    bic_psnr = 0.0
    bic_ssim = 0.0
    bic_embeddings = []
    
    probe_labels = []
    
    # Warmup
    if device == "cuda":
        dummy = torch.randn(1, 3, vlr_size, vlr_size).to(device)
        pipeline.dsr_model(dummy)
        
    best_sample = None
    best_sim = -1.0
    
    for img_path in tqdm(image_files, desc="Processing"):
        hr_path = hr_dir / img_path.name
        if not hr_path.exists():
            continue
            
        subject_id = img_path.stem.split('_')[0]
        probe_labels.append(subject_id)
        
        vlr_img = Image.open(img_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        hr_np = np.array(hr_img)
        
        # --- DSR Evaluation ---
        # 1. Upscale
        t0 = time.time()
        sr_tensor = pipeline.upscale(vlr_img)
        t1 = time.time()
        dsr_upscale_time = (t1 - t0) * 1000
        
        # 2. Recognition
        t2 = time.time()
        dsr_emb = pipeline.infer_embedding(sr_tensor, use_tta=False)
        t3 = time.time()
        dsr_rec_time = (t3 - t2) * 1000
        
        dsr_times.append(dsr_upscale_time + dsr_rec_time)
        dsr_upscale_times.append(dsr_upscale_time)
        dsr_rec_times.append(dsr_rec_time)
        
        dsr_emb = dsr_emb.to(device)
        dsr_embeddings.append(dsr_emb.cpu())
        
        # Reconstruct result dict for compatibility
        sr_img = pipeline._tensor_to_image(sr_tensor)
        result = {"sr_image": sr_img, "embedding": dsr_emb}
        
        # Identification (DSR)
        query = F.normalize(dsr_emb, dim=-1)
        stacked_gallery = torch.stack(pipeline.gallery._embeddings).to(device)
        scores = torch.mv(stacked_gallery, query)
        topk_scores, topk_indices = torch.topk(scores, k=min(5, len(pipeline.gallery._labels)))
        topk_labels = [pipeline.gallery._labels[i] for i in topk_indices.tolist()]
        
        if subject_id == topk_labels[0]: dsr_correct_top1 += 1
        if subject_id in topk_labels: dsr_correct_top5 += 1
            
        # Quality (DSR)
        sr_img = result["sr_image"]
        sr_np = np.array(sr_img)
        if sr_np.shape != hr_np.shape:
            hr_np = cv2.resize(hr_np, (sr_np.shape[1], sr_np.shape[0]))
            
        dsr_psnr += compute_psnr(sr_np, hr_np)
        ssim_val = compute_ssim(sr_np, hr_np)
        if ssim_val >= 0: dsr_ssim += ssim_val
        
        # --- Bicubic Baseline ---
        # Resize VLR to 112x112 using Bicubic
        bic_img = vlr_img.resize((112, 112), Image.BICUBIC)
        bic_tensor = pipeline._to_tensor(bic_img)
        bic_emb = pipeline.infer_embedding(bic_tensor).to(device)
        bic_embeddings.append(bic_emb.cpu())
        
        # Identification (Bicubic)
        query_bic = F.normalize(bic_emb, dim=-1)
        scores_bic = torch.mv(stacked_gallery, query_bic)
        topk_scores_bic, topk_indices_bic = torch.topk(scores_bic, k=min(5, len(pipeline.gallery._labels)))
        topk_labels_bic = [pipeline.gallery._labels[i] for i in topk_indices_bic.tolist()]
        
        if subject_id == topk_labels_bic[0]: bic_correct_top1 += 1
        if subject_id in topk_labels_bic: bic_correct_top5 += 1
        
        # Quality (Bicubic)
        bic_np = np.array(bic_img)
        bic_psnr += compute_psnr(bic_np, hr_np)
        ssim_val_bic = compute_ssim(bic_np, hr_np)
        if ssim_val_bic >= 0: bic_ssim += ssim_val_bic
        
        # Track Best Sample (DSR)
        try:
            gt_idx = gallery_labels_list.index(subject_id)
            gt_emb = gallery_tensor[gt_idx]
            sim = F.cosine_similarity(dsr_emb.unsqueeze(0), gt_emb.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_sample = (vlr_img, sr_img, bic_img, hr_img, sim)
        except ValueError:
            pass
            
    total_samples = len(probe_labels)
    
    # 3. Compute Verification Metrics
    print("Computing Verification Metrics...")
    dsr_tensor = torch.stack(dsr_embeddings).to(device)
    bic_tensor = torch.stack(bic_embeddings).to(device)
    
    dsr_verif = compute_verification_metrics(dsr_tensor, probe_labels, gallery_tensor, gallery_labels_list)
    bic_verif = compute_verification_metrics(bic_tensor, probe_labels, gallery_tensor, gallery_labels_list)
    
    avg_inference_time = np.mean(dsr_times)
    
    metrics = {
        "resolution": vlr_size,
        "dataset_path": str(test_dir.relative_to(ROOT) if ROOT in test_dir.parents else test_dir),
        "samples": total_samples,
        "subjects": len(subject_map),
        "dsr": {
            "accuracy_top1": dsr_correct_top1 / total_samples,
            "accuracy_top5": dsr_correct_top5 / total_samples,
            "avg_psnr": dsr_psnr / total_samples,
            "avg_ssim": dsr_ssim / total_samples,
            "avg_inference_time_ms": avg_inference_time,
            "avg_upscale_time_ms": np.mean(dsr_upscale_times),
            "avg_rec_time_ms": np.mean(dsr_rec_times),
            "fps": 1000.0 / avg_inference_time,
            **dsr_verif
        },
        "bicubic": {
            "accuracy_top1": bic_correct_top1 / total_samples,
            "accuracy_top5": bic_correct_top5 / total_samples,
            "avg_psnr": bic_psnr / total_samples,
            "avg_ssim": bic_ssim / total_samples,
            **bic_verif
        },
        "best_sample": best_sample
    }
    
    if device == "cuda":
        metrics["max_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
        
    print("\nResults (DSR vs Bicubic):")
    print(f"  Top-1 Acc: {metrics['dsr']['accuracy_top1']:.2%} vs {metrics['bicubic']['accuracy_top1']:.2%}")
    print(f"  EER:       {metrics['dsr']['eer']:.2%} vs {metrics['bicubic']['eer']:.2%}")
    print(f"  PSNR:      {metrics['dsr']['avg_psnr']:.2f} vs {metrics['bicubic']['avg_psnr']:.2f}")
    
    return metrics

def generate_pdf_report(results: Dict[str, Any], output_dir: Path):
    """Generate professional PDF report with baselines and metadata."""
    print("\nGenerating PDF Report...")
    doc = SimpleDocTemplate(str(output_dir / "benchmark_report.pdf"), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("LRFR Project Evaluation Report", styles["Title"]))
    story.append(Paragraph(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 12))
    
    # 1. Dataset & Model Metadata
    story.append(Paragraph("Evaluation Metadata", styles["Heading2"]))
    
    # Get stats from first result
    first_res = list(results.values())[0]
    num_subjects = first_res["subjects"]
    num_images = first_res["samples"]
    avg_imgs = num_images / num_subjects if num_subjects else 0
    
    meta_data = [
        ["Metric", "Value"],
        ["Dataset Path", f"{list(results.values())[0].get('dataset_path', 'N/A')}"],
        ["Total Test Images", f"{num_images}"],
        ["Total Subjects", f"{num_subjects}"],
        ["Avg Images/Subject", f"{avg_imgs:.1f}"],
        ["Evaluation Protocol", "Unbiased Closed-Set Identification & Verification"],
        ["Gallery", "High-Resolution (HR) Ground Truth"],
        ["Probes", "Very-Low-Resolution (VLR) -> Super-Resolved"],
        ["Face Recognition Model", "EdgeFace (ConvNeXt-XXS, Pre-trained)"],
        ["Super-Resolution Model", "Hybrid DSR (Identity-Aware)"]
    ]
    
    t_meta = Table(meta_data, colWidths=[200, 300])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 24))
    
    # 2. Executive Summary (Comparison)
    story.append(Paragraph("Executive Summary (DSR vs Bicubic)", styles["Heading2"]))
    
    data = [["Res", "Method", "Top-1", "EER", "Avg Acc", "AUC", "PSNR", "SSIM", "FPS"]]
    resolutions = sorted([int(k) for k in results.keys()])
    
    for res in resolutions:
        m = results[str(res)]
        if "error" in m: continue
        
        # DSR Row
        data.append([
            f"{res}x{res}", "DSR",
            f"{m['dsr']['accuracy_top1']:.1%}",
            f"{m['dsr']['eer']:.2%}",
            f"{m['dsr']['mean_accuracy']:.2%} (+/-{m['dsr']['std_accuracy']:.2%})",
            f"{m['dsr']['auc']:.3f}",
            f"{m['dsr']['avg_psnr']:.1f}",
            f"{m['dsr']['avg_ssim']:.3f}",
            f"{m['dsr']['fps']:.0f}"
        ])
        # Bicubic Row
        data.append([
            "", "Bicubic",
            f"{m['bicubic']['accuracy_top1']:.1%}",
            f"{m['bicubic']['eer']:.2%}",
            f"{m['bicubic']['mean_accuracy']:.2%} (+/-{m['bicubic']['std_accuracy']:.2%})",
            f"{m['bicubic']['auc']:.3f}",
            f"{m['bicubic']['avg_psnr']:.1f}",
            f"{m['bicubic']['avg_ssim']:.3f}",
            "-"
        ])
            
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('LINEBELOW', (0,2), (-1,2), 1, colors.black), # Separator after 16x16
        ('LINEBELOW', (0,4), (-1,4), 1, colors.black), # Separator after 24x24
    ]))
    story.append(t)
    story.append(Spacer(1, 24))

    # 2b. Detailed Performance Breakdown
    story.append(Paragraph("Detailed Performance Breakdown (DSR)", styles["Heading2"]))
    
    perf_data = [["Res", "Total Time", "Upscaling (DSR)", "Recognition (EdgeFace)", "FPS"]]
    
    for res in resolutions:
        m = results[str(res)]
        if "error" in m: continue
        
        dsr = m["dsr"]
        perf_data.append([
            f"{res}x{res}",
            f"{dsr['avg_inference_time_ms']:.1f} ms",
            f"{dsr['avg_upscale_time_ms']:.1f} ms",
            f"{dsr['avg_rec_time_ms']:.1f} ms",
            f"{dsr['fps']:.0f}"
        ])
        
    t_perf = Table(perf_data)
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(t_perf)
    story.append(Spacer(1, 24))
    
    # 3. Plots
    story.append(Paragraph("Performance Analysis", styles["Heading2"]))
    
    # ROC Plot (DSR vs Bicubic for each resolution)
    plt.figure(figsize=(7, 5))
    colors_list = ['b', 'g', 'r']
    for i, res in enumerate(resolutions):
        m = results[str(res)]
        if "dsr" in m:
            plt.plot(m["dsr"]["fpr"], m["dsr"]["tpr"], color=colors_list[i], linestyle='-', label=f'DSR {res}x{res}')
            plt.plot(m["bicubic"]["fpr"], m["bicubic"]["tpr"], color=colors_list[i], linestyle='--', label=f'Bicubic {res}x{res}')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: DSR vs Bicubic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_dir / "plot_roc_comparison.png")
    plt.close()
    story.append(RLImage(str(output_dir / "plot_roc_comparison.png"), width=450, height=300))
    story.append(PageBreak())
    
    # Visual Examples
    story.append(Paragraph("Visual Examples", styles["Heading2"]))
    for res in resolutions:
        m = results[str(res)]
        if "best_sample" in m and m["best_sample"]:
            vlr, sr, bic, hr, score = m["best_sample"]
            
            # Combine images: VLR | Bicubic | DSR | HR
            w, h = 112, 112
            combined = Image.new('RGB', (460, h))
            combined.paste(vlr.resize((w,h), Image.NEAREST), (0,0))
            combined.paste(bic.resize((w,h), Image.BICUBIC), (114,0))
            combined.paste(sr.resize((w,h), Image.LANCZOS), (228,0))
            combined.paste(hr.resize((w,h), Image.LANCZOS), (342,0))
            
            path = output_dir / f"sample_{res}.png"
            combined.save(path)
            
            story.append(Paragraph(f"Resolution {res}x{res}", styles["Heading3"]))
            story.append(Paragraph("Left to Right: VLR Input, Bicubic Baseline, DSR Output, HR Ground Truth", styles["Normal"]))
            story.append(RLImage(str(path), width=460, height=112))
            story.append(Spacer(1, 12))
            
    doc.build(story)
    print(f"PDF Report saved to {output_dir / 'benchmark_report.pdf'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.test_dir:
        test_dir = Path(args.test_dir)
    else:
        candidates = [
            ROOT / "technical" / "dataset" / "test_processed",
            ROOT / "technical" / "dataset" / "frontal_only" / "test",
            Path("dataset/frontal_only/test")
        ]
        test_dir = next((c for c in candidates if c.exists()), None)
        
    if not test_dir:
        print("❌ Test directory not found.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    for res in args.resolutions:
        metrics = evaluate_resolution(res, test_dir, args.device)
        all_results[str(res)] = metrics
        
        # Save JSON (clean)
        clean_m = metrics.copy()
        if "best_sample" in clean_m: del clean_m["best_sample"]
        if "dsr" in clean_m:
            clean_m["dsr"] = {k:v for k,v in clean_m["dsr"].items() if k not in ["fpr", "tpr", "genuine_scores", "impostor_scores"]}
        if "bicubic" in clean_m:
            clean_m["bicubic"] = {k:v for k,v in clean_m["bicubic"].items() if k not in ["fpr", "tpr", "genuine_scores", "impostor_scores"]}
            
        with open(output_dir / f"results_{res}.json", "w") as f:
            json.dump(clean_m, f, indent=4)
            
    try:
        generate_pdf_report(all_results, output_dir)
    except Exception as e:
        print(f"❌ PDF Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
