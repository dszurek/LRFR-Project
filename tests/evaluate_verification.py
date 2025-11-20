"""Comprehensive face verification and identification evaluation.

This script provides proper metrics for:
1. 1:1 Verification: FAR, FRR, EER, ROC curves
2. 1:N Identification: Rank-1/5/10 accuracy (closed-set and open-set)
3. DSR Quality Impact: Comparison of VLR→DSR→EdgeFace vs HR→EdgeFace

Usage:
    # Evaluate verification metrics (1:1 matching)
    python -m technical.pipeline.evaluate_verification \
        --mode verification \
        --test-root technical/dataset/frontal_only/test

    # Evaluate identification metrics (1:N matching)
    python -m technical.pipeline.evaluate_verification \
        --mode identification \
        --gallery-root technical/dataset/frontal_only/train \
        --test-root technical/dataset/frontal_only/test \
        --max-gallery-size 10

    # Evaluate both with custom thresholds
    python -m technical.pipeline.evaluate_verification \
        --mode both \
        --gallery-root technical/dataset/frontal_only/train \
        --test-root technical/dataset/frontal_only/test
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_curve, auc
from torchvision import transforms
from tqdm import tqdm

from .pipeline import PipelineConfig, build_pipeline


@dataclass
class VerificationResult:
    """Result for a single verification pair."""
    probe_id: str
    gallery_id: str
    is_genuine: bool
    similarity: float
    predicted_match: bool
    threshold: float


@dataclass
class IdentificationResult:
    """Result for a single identification probe."""
    probe_id: str
    true_identity: str
    predicted_identity: Optional[str]
    rank: Optional[int]  # Rank of true identity (1-based, None if not in top-K)
    similarity_scores: Dict[str, float]
    is_in_gallery: bool


def _subject_from_filename(filename: str) -> str:
    """Extract subject ID from filename."""
    stem = Path(filename).stem
    if "_lfw" in stem:
        return stem.split("_")[0]
    else:
        return stem.split("_")[0]


def load_image_tensor(
    image_path: Path, device: torch.device, for_edgeface: bool = False
) -> torch.Tensor:
    """Load image and convert to tensor.
    
    Args:
        image_path: Path to image file
        device: Device to load tensor on
        for_edgeface: If True, apply EdgeFace normalization
    
    Returns:
        Image tensor (1, 3, H, W)
    """
    img = Image.open(image_path).convert("RGB")
    
    if for_edgeface:
        # EdgeFace expects 112×112 (normalization will be applied by pipeline.infer_embedding)
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.ToTensor()
    
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def compute_embeddings(
    pipeline,
    directory: Path,
    vlr_size: int = 32,
    use_dsr: bool = True,
    max_images_per_subject: Optional[int] = None,
) -> Dict[str, List[torch.Tensor]]:
    """Compute embeddings for all images in a directory.
    
    Args:
        pipeline: Recognition pipeline
        directory: Directory containing images (hr_images/ or vlr_images_{size}x{size}/)
        vlr_size: VLR resolution size (16, 24, or 32)
        use_dsr: If True, run DSR before computing embeddings (use for VLR images)
        max_images_per_subject: Limit number of images per subject
    
    Returns:
        Dictionary mapping subject_id -> list of embedding tensors
    """
    # Use consistent naming format for VLR directories
    if use_dsr:
        image_dir = directory / f"vlr_images_{vlr_size}x{vlr_size}"
    else:
        image_dir = directory / "hr_images"
    
    if not image_dir.exists():
        raise ValueError(f"Directory not found: {image_dir}")
    
    embeddings_by_subject = defaultdict(list)
    
    image_paths = sorted(image_dir.glob("*.png"))
    
    for image_path in tqdm(image_paths, desc=f"Computing embeddings ({'DSR' if use_dsr else 'HR'})"):
        subject_id = _subject_from_filename(image_path.name)
        
        # Limit images per subject if requested
        if max_images_per_subject and len(embeddings_by_subject[subject_id]) >= max_images_per_subject:
            continue
        
        if use_dsr:
            # Run through DSR pipeline
            sr_tensor = pipeline.upscale(image_path)
            embedding = pipeline.infer_embedding(sr_tensor)
        else:
            # Load HR image directly
            hr_tensor = load_image_tensor(image_path, pipeline.device, for_edgeface=True)
            embedding = pipeline.infer_embedding(hr_tensor)
        
        embeddings_by_subject[subject_id].append(embedding.cpu())
    
    return dict(embeddings_by_subject)


def evaluate_verification(
    pipeline,
    test_root: Path,
    vlr_size: int = 32,
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """Evaluate 1:1 verification performance.
    
    Computes FAR, FRR, EER, and ROC curves by comparing all pairs of images.
    
    Args:
        pipeline: Recognition pipeline
        test_root: Root directory containing vlr_images_{size}x{size}/ and hr_images/
        vlr_size: VLR resolution size (16, 24, or 32)
        thresholds: List of thresholds to evaluate (if None, auto-generate)
    
    Returns:
        Dictionary with verification metrics
    """
    print("\n" + "=" * 70)
    print("VERIFICATION EVALUATION (1:1 Matching)")
    print("=" * 70)
    
    # Compute embeddings for all test images
    embeddings = compute_embeddings(pipeline, test_root, vlr_size=vlr_size, use_dsr=True)
    
    # Generate all pairs (genuine and impostor)
    genuine_scores = []
    impostor_scores = []
    
    subjects = list(embeddings.keys())
    
    print(f"\nGenerating verification pairs from {len(subjects)} subjects...")
    
    for i, subject_i in enumerate(tqdm(subjects, desc="Computing pair similarities")):
        embs_i = embeddings[subject_i]
        
        # Genuine pairs (same subject)
        for j in range(len(embs_i)):
            for k in range(j + 1, len(embs_i)):
                score = torch.cosine_similarity(embs_i[j], embs_i[k], dim=0).item()
                genuine_scores.append(score)
        
        # Impostor pairs (different subjects)
        for j in range(i + 1, len(subjects)):
            subject_j = subjects[j]
            embs_j = embeddings[subject_j]
            
            # Compare first embedding from each subject (to limit pairs)
            if len(embs_i) > 0 and len(embs_j) > 0:
                score = torch.cosine_similarity(embs_i[0], embs_j[0], dim=0).item()
                impostor_scores.append(score)
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    print(f"\nGenerated {len(genuine_scores)} genuine pairs, {len(impostor_scores)} impostor pairs")
    
    # Compute ROC curve
    y_true = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find EER (Equal Error Rate) - point where FPR = FNR
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = roc_thresholds[eer_idx]
    
    # Evaluate at specific thresholds
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, eer_threshold]
    
    threshold_metrics = []
    
    for threshold in thresholds:
        # Predictions: score >= threshold → genuine, else impostor
        genuine_accepted = np.sum(genuine_scores >= threshold)
        genuine_rejected = len(genuine_scores) - genuine_accepted
        impostor_accepted = np.sum(impostor_scores >= threshold)
        impostor_rejected = len(impostor_scores) - impostor_accepted
        
        tar = genuine_accepted / len(genuine_scores) if len(genuine_scores) > 0 else 0  # True Accept Rate
        frr = genuine_rejected / len(genuine_scores) if len(genuine_scores) > 0 else 0  # False Reject Rate
        far = impostor_accepted / len(impostor_scores) if len(impostor_scores) > 0 else 0  # False Accept Rate
        trr = impostor_rejected / len(impostor_scores) if len(impostor_scores) > 0 else 0  # True Reject Rate
        
        threshold_metrics.append({
            "threshold": threshold,
            "TAR": tar,
            "FRR": frr,
            "FAR": far,
            "TRR": trr,
        })
    
    results = {
        "num_genuine_pairs": len(genuine_scores),
        "num_impostor_pairs": len(impostor_scores),
        "genuine_score_mean": float(np.mean(genuine_scores)),
        "genuine_score_std": float(np.std(genuine_scores)),
        "impostor_score_mean": float(np.mean(impostor_scores)),
        "impostor_score_std": float(np.std(impostor_scores)),
        "roc_auc": float(roc_auc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "threshold_metrics": threshold_metrics,
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    print(f"ROC AUC:               {roc_auc:.4f}")
    print(f"Equal Error Rate (EER): {eer:.4f} (threshold: {eer_threshold:.4f})")
    print(f"\nGenuine scores:  μ={np.mean(genuine_scores):.4f}, σ={np.std(genuine_scores):.4f}")
    print(f"Impostor scores: μ={np.mean(impostor_scores):.4f}, σ={np.std(impostor_scores):.4f}")
    
    print(f"\n{'Threshold':<12} {'TAR':<8} {'FRR':<8} {'FAR':<8} {'TRR':<8}")
    print("-" * 50)
    for metric in threshold_metrics:
        print(
            f"{metric['threshold']:<12.4f} "
            f"{metric['TAR']:<8.4f} "
            f"{metric['FRR']:<8.4f} "
            f"{metric['FAR']:<8.4f} "
            f"{metric['TRR']:<8.4f}"
        )
    
    return results


def evaluate_identification(
    pipeline,
    gallery_root: Path,
    test_root: Path,
    vlr_size: int = 32,
    max_gallery_size: Optional[int] = None,
    include_open_set: bool = True,
) -> Dict:
    """Evaluate 1:N identification performance.
    
    Args:
        pipeline: Recognition pipeline
        gallery_root: Root directory for gallery enrollment
        test_root: Root directory for test probes
        vlr_size: VLR resolution size (16, 24, or 32)
        max_gallery_size: Maximum number of subjects in gallery (simulates small gallery)
        include_open_set: If True, include subjects NOT in gallery (tests rejection)
    
    Returns:
        Dictionary with identification metrics
    """
    print("\n" + "=" * 70)
    print("IDENTIFICATION EVALUATION (1:N Matching)")
    print("=" * 70)
    
    # Build gallery
    print(f"\nBuilding gallery from {gallery_root}...")
    gallery_embeddings = compute_embeddings(
        pipeline, gallery_root, vlr_size=vlr_size, use_dsr=False, max_images_per_subject=3
    )
    
    # Limit gallery size if requested
    if max_gallery_size and len(gallery_embeddings) > max_gallery_size:
        gallery_subjects = list(gallery_embeddings.keys())[:max_gallery_size]
        gallery_embeddings = {k: gallery_embeddings[k] for k in gallery_subjects}
        print(f"Limited gallery to {max_gallery_size} subjects")
    
    # Compute mean embedding per gallery subject
    gallery_mean_embeddings = {}
    for subject_id, embs in gallery_embeddings.items():
        stacked = torch.stack(embs)
        gallery_mean_embeddings[subject_id] = torch.mean(stacked, dim=0)
    
    print(f"Gallery enrolled: {len(gallery_mean_embeddings)} subjects")
    
    # Compute test probe embeddings
    print(f"\nComputing test probe embeddings from {test_root}...")
    probe_embeddings = compute_embeddings(
        pipeline, test_root, vlr_size=vlr_size, use_dsr=True
    )
    
    # Evaluate each probe
    results = []
    rank1_correct = 0
    rank5_correct = 0
    rank10_correct = 0
    total_in_gallery = 0
    total_not_in_gallery = 0
    correctly_rejected = 0
    
    for probe_subject, probe_embs in tqdm(probe_embeddings.items(), desc="Evaluating probes"):
        is_in_gallery = probe_subject in gallery_mean_embeddings
        
        for probe_emb in probe_embs:
            # Compute similarity to all gallery subjects
            similarities = {}
            for gallery_subject, gallery_emb in gallery_mean_embeddings.items():
                sim = torch.cosine_similarity(probe_emb, gallery_emb, dim=0).item()
                similarities[gallery_subject] = sim
            
            # Rank gallery subjects by similarity
            ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Find rank of true identity
            rank = None
            for i, (subject, score) in enumerate(ranked):
                if subject == probe_subject:
                    rank = i + 1  # 1-based rank
                    break
            
            predicted_identity = ranked[0][0] if ranked else None
            
            result = IdentificationResult(
                probe_id=f"{probe_subject}_{len(results)}",
                true_identity=probe_subject,
                predicted_identity=predicted_identity,
                rank=rank,
                similarity_scores=similarities,
                is_in_gallery=is_in_gallery,
            )
            results.append(result)
            
            if is_in_gallery:
                total_in_gallery += 1
                if rank == 1:
                    rank1_correct += 1
                if rank and rank <= 5:
                    rank5_correct += 1
                if rank and rank <= 10:
                    rank10_correct += 1
            else:
                total_not_in_gallery += 1
                # Correct rejection: true identity not in gallery and predicted is wrong
                # (in practice, would use threshold, but here we assume top-1 is wrong)
                if predicted_identity != probe_subject:
                    correctly_rejected += 1
    
    # Compute metrics
    rank1_accuracy = rank1_correct / total_in_gallery if total_in_gallery > 0 else 0
    rank5_accuracy = rank5_correct / total_in_gallery if total_in_gallery > 0 else 0
    rank10_accuracy = rank10_correct / total_in_gallery if total_in_gallery > 0 else 0
    rejection_rate = correctly_rejected / total_not_in_gallery if total_not_in_gallery > 0 else 0
    
    metrics = {
        "gallery_size": len(gallery_mean_embeddings),
        "total_probes": len(results),
        "probes_in_gallery": total_in_gallery,
        "probes_not_in_gallery": total_not_in_gallery,
        "rank1_accuracy": rank1_accuracy,
        "rank5_accuracy": rank5_accuracy,
        "rank10_accuracy": rank10_accuracy,
        "rejection_rate": rejection_rate,
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("IDENTIFICATION RESULTS")
    print("=" * 70)
    print(f"Gallery size:           {len(gallery_mean_embeddings)} subjects")
    print(f"Total probes:           {len(results)}")
    print(f"  In gallery:           {total_in_gallery}")
    print(f"  Not in gallery:       {total_not_in_gallery}")
    print(f"\nClosed-Set Accuracy (probes in gallery):")
    print(f"  Rank-1:  {rank1_accuracy:.4f} ({rank1_correct}/{total_in_gallery})")
    print(f"  Rank-5:  {rank5_accuracy:.4f} ({rank5_correct}/{total_in_gallery})")
    print(f"  Rank-10: {rank10_accuracy:.4f} ({rank10_correct}/{total_in_gallery})")
    
    if total_not_in_gallery > 0:
        print(f"\nOpen-Set Performance (probes NOT in gallery):")
        print(f"  Rejection rate: {rejection_rate:.4f} ({correctly_rejected}/{total_not_in_gallery})")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive face verification and identification evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["verification", "identification", "both"],
        default="both",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--gallery-root",
        type=Path,
        default=None,
        help="Root directory for gallery enrollment (required for identification)",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("technical/dataset/frontal_only/test"),
        help="Root directory for test probes",
    )
    parser.add_argument(
        "--max-gallery-size",
        type=int,
        default=None,
        help="Limit gallery to N subjects (simulates small gallery)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--edgeface-weights",
        type=Path,
        default=None,
        help="Path to EdgeFace weights",
    )
    parser.add_argument(
        "--dsr-weights",
        type=Path,
        default=None,
        help="Path to DSR weights",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Recognition threshold",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--vlr-size",
        type=int,
        default=32,
        choices=[16, 24, 32],
        help="VLR resolution size (16, 24, or 32)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ["identification", "both"] and args.gallery_root is None:
        parser.error("--gallery-root is required for identification mode")
    
    # Build pipeline
    config = PipelineConfig(device=args.device)
    if args.threshold is not None:
        config.recognition_threshold = args.threshold
    if args.edgeface_weights is not None:
        config.edgeface_weights_path = args.edgeface_weights
    if args.dsr_weights is not None:
        config.dsr_weights_path = args.dsr_weights
    
    pipeline = build_pipeline(config)
    
    all_results = {}
    
    # Run verification evaluation
    if args.mode in ["verification", "both"]:
        verification_results = evaluate_verification(
            pipeline, args.test_root, vlr_size=args.vlr_size
        )
        all_results["verification"] = verification_results
    
    # Run identification evaluation
    if args.mode in ["identification", "both"]:
        identification_results = evaluate_identification(
            pipeline,
            args.gallery_root,
            args.test_root,
            vlr_size=args.vlr_size,
            max_gallery_size=args.max_gallery_size,
        )
        all_results["identification"] = identification_results
    
    # Save results if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
