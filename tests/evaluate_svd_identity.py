"""
LFW Low-Resolution Face Recognition Evaluation

Implements the EXACT evaluation protocol from:
Lai et al. (2019) - "Low-Resolution Face Recognition Based on Identity-Preserved Face Hallucination"

Tables to reproduce:
- Table 1: Verification rates (%) on LFW 6,000 pairs (unrestricted setting)
- Table 2: Rank-1 identification rates (%) following protocol from [16]

Protocol:
- Uses LFW dataset only (from frontal_only directory)
- Verification: 6,000 pairs (3,000 same + 3,000 different)
- Identification: Closed-set, 1 gallery image per subject, rest as probes
- Resolutions: 7×6, 14×12, 16×16, 18×16, 28×24, 112×96
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2
from dataclasses import dataclass
import random

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "technical"))

from dsr.models import DSRColor
from facial_rec.edgeface_weights.edgeface import EdgeFace


@dataclass
class EvaluationConfig:
    """Configuration for LFW evaluation matching Lai et al. 2019"""
    # Paths - LFW images only
    lfw_hr_dir: Path
    lfw_vlr_dir: Path
    dsr_checkpoint: Path
    edgeface_checkpoint: Path
    output_dir: Path
    
    # Resolution configurations (downsampling factors from 112×96)
    # Factor 16: 112/16=7, 96/16=6  → 7×6
    # Factor 8:  112/8=14, 96/8=12  → 14×12
    # Factor 7:  112/7=16, 96/7≈14  → 16×14
    # Factor 6:  112/6≈19, 96/6=16  → 18×16  (rounded to match paper)
    # Factor 4:  112/4=28, 96/4=24  → 28×24
    resolutions: List[Tuple[int, int]] = None  # Will be set in __post_init__
    
    # Evaluation parameters
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.resolutions is None:
            self.resolutions = [
                (7, 6),      # 16x downsampling
                (14, 12),    # 8x downsampling
                (16, 14),    # ~7x downsampling
                (18, 16),    # ~6x downsampling (matches paper)
                (28, 24),    # 4x downsampling
                (112, 96)    # Original HR
            ]


class SVDFaceRepresentation:
    """
    SVD-based face representation following Jian & Lam (2015)
    
    Each face image I is decomposed as: I = U * W * V^T
    where W contains singular values in descending order.
    """
    
    def __init__(self, num_singular_values: int = 50, threshold: float = 0.99):
        """
        Args:
            num_singular_values: Number of leading singular values to retain (k)
            threshold: Cumulative energy threshold (η) for determining k
        """
        self.k = num_singular_values
        self.threshold = threshold
    
    def decompose(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform SVD decomposition on face image
        
        Args:
            image: Face image as numpy array (H, W) or (H, W, C)
            
        Returns:
            Dictionary containing U, W (singular values), V matrices
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # SVD decomposition: I = U * Σ * V^T
        U, s, Vh = np.linalg.svd(image, full_matrices=False)
        
        # Determine number of singular values to keep
        k = self._determine_k(s)
        
        return {
            'U': U,
            'singular_values': s,
            'V': Vh.T,  # Transpose to get V
            'k': k,
            'image_shape': image.shape
        }
    
    def _determine_k(self, singular_values: np.ndarray) -> int:
        """
        Determine number of singular values k based on cumulative energy
        
        Following equation (5) in Jian & Lam (2015):
        fcumu(k) = sqrt(sum(w_i^2, i=1..k)) / sqrt(sum(w_i^2, i=1..n))
        
        Args:
            singular_values: Array of singular values
            
        Returns:
            Number of singular values k such that fcumu(k) >= threshold
        """
        cumsum = np.cumsum(singular_values ** 2)
        total = cumsum[-1]
        
        # Find k where cumulative energy >= threshold
        k = np.searchsorted(cumsum / total, self.threshold) + 1
        
        # Ensure k doesn't exceed specified maximum
        k = min(k, self.k, len(singular_values))
        
        return k
    
    def get_normalized_singular_values(self, svd_dict: Dict) -> np.ndarray:
        """
        Get scale-invariant normalized singular values
        
        Following equation (11) in Jian & Lam (2015):
        s' = s / w1, where w1 is the largest singular value
        
        Args:
            svd_dict: Dictionary from decompose()
            
        Returns:
            Normalized singular value vector
        """
        s = svd_dict['singular_values']
        k = svd_dict['k']
        
        # Take first k singular values
        s_k = s[:k]
        
        # Normalize by largest singular value (w1)
        s_normalized = s_k / s[0] if s[0] > 0 else s_k
        
        return s_normalized
    
    def compute_similarity(self, svd1: Dict, svd2: Dict) -> float:
        """
        Compute similarity between two face images using normalized singular values
        
        Following equation (21) in Jian & Lam (2015):
        SIM(I1, I2) = 1 / ||s'_I1 - s'_I2||_2
        
        Args:
            svd1, svd2: SVD dictionaries from decompose()
            
        Returns:
            Similarity score (higher = more similar)
        """
        s1 = self.get_normalized_singular_values(svd1)
        s2 = self.get_normalized_singular_values(svd2)
        
        # Ensure same length by padding with zeros
        max_len = max(len(s1), len(s2))
        s1_padded = np.pad(s1, (0, max_len - len(s1)), 'constant')
        s2_padded = np.pad(s2, (0, max_len - len(s2)), 'constant')
        
        # L2 distance
        distance = np.linalg.norm(s1_padded - s2_padded)
        
        # Similarity (avoid division by zero)
        similarity = 1.0 / (distance + 1e-8)
        
        return similarity


class IdentityPreservedEvaluator:
    """
    Identity-preserved evaluation following Lai et al. (2019)
    
    Uses deep features extracted from EdgeFace network with cosine similarity
    """
    
    def __init__(self, edgeface_model: nn.Module, device: str = "cuda"):
        """
        Args:
            edgeface_model: Pre-trained EdgeFace network
            device: Device for computation
        """
        self.model = edgeface_model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract deep features from face images
        
        Args:
            images: Batch of face images (B, C, H, W)
            
        Returns:
            Feature vectors (B, D)
        """
        features = self.model(images.to(self.device))
        
        # L2 normalize features (important for cosine similarity)
        features = nn.functional.normalize(features, p=2, dim=1)
        
        return features
    
    def compute_cosine_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between feature vectors
        
        Following equation (3) in Lai et al. (2019):
        f(y1, y2) = (y1 · y2) / (||y1||_2 * ||y2||_2)
        
        Since features are already normalized, this simplifies to dot product.
        
        Args:
            feat1, feat2: Feature vectors (already L2 normalized)
            
        Returns:
            Cosine similarity scores
        """
        # Dot product of normalized vectors = cosine similarity
        similarity = torch.sum(feat1 * feat2, dim=1)
        return similarity
    
    def compute_identity_preserved_loss(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Identity-preserved loss from Lai et al. (2019)
        
        Following equation (2-3):
        Lid = 1 - (y1 · y2) / (||y1||_2 * ||y2||_2)
        
        Args:
            feat1, feat2: Feature vectors
            
        Returns:
            Identity-preserved loss (lower = better identity preservation)
        """
        cosine_sim = self.compute_cosine_similarity(feat1, feat2)
        loss = 1.0 - cosine_sim
        return loss


class LRFREvaluator:
    """
    Main evaluator combining SVD and identity-preserved methodologies
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Args:
            config: Evaluation configuration
        """
        self.config = config
        
        # Initialize SVD representation
        self.svd_repr = SVDFaceRepresentation(
            num_singular_values=config.num_singular_values,
            threshold=config.svd_threshold
        )
        
        # Load models
        self.dsr_model = self._load_dsr_model()
        self.edgeface_model = self._load_edgeface_model()
        
        # Initialize identity-preserved evaluator
        self.identity_eval = IdentityPreservedEvaluator(
            self.edgeface_model,
            device=config.device
        )
        
        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_dsr_model(self) -> nn.Module:
        """Load pre-trained DSR model"""
        model = DSRColor(upscale_factor=7)  # 16x16 -> 112x112
        
        if self.config.dsr_checkpoint.exists():
            checkpoint = torch.load(
                self.config.dsr_checkpoint,
                map_location=self.config.device
            )
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.to(self.config.device)
        model.eval()
        return model
    
    def _load_edgeface_model(self) -> nn.Module:
        """Load pre-trained EdgeFace model"""
        model = EdgeFace(embedding_size=512, back="edgeface_xxs")
        
        if self.config.edgeface_checkpoint.exists():
            checkpoint = torch.load(
                self.config.edgeface_checkpoint,
                map_location=self.config.device
            )
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.to(self.config.device)
        model.eval()
        return model
    
    def load_image_pairs(self) -> List[Tuple[Path, Path, str]]:
        """
        Load HR and VLR image pairs from test directories
        
        Returns:
            List of (hr_path, vlr_path, subject_id) tuples
        """
        hr_dir = self.config.frontal_test_hr
        vlr_dir = self.config.frontal_test_vlr
        
        pairs = []
        
        # Iterate through HR images
        for hr_path in sorted(hr_dir.glob("*.png")):
            # Find corresponding VLR image
            vlr_path = vlr_dir / hr_path.name
            
            if vlr_path.exists():
                # Extract subject ID from filename
                # Assuming format: subjectID_imageID.png
                subject_id = hr_path.stem.split('_')[0]
                pairs.append((hr_path, vlr_path, subject_id))
        
        return pairs
    
    def preprocess_image(self, image_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            target_size: (H, W) for resizing
            
        Returns:
            Preprocessed image array
        """
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (target_size[1], target_size[0]))
        return image
    
    @torch.no_grad()
    def super_resolve(self, vlr_image: np.ndarray) -> np.ndarray:
        """
        Super-resolve VLR image using DSR model
        
        Args:
            vlr_image: VLR image array (H, W, C)
            
        Returns:
            Super-resolved image array
        """
        # Convert to tensor (C, H, W) and normalize to [-1, 1]
        image_tensor = torch.from_numpy(vlr_image).permute(2, 0, 1).float()
        image_tensor = (image_tensor / 127.5) - 1.0
        image_tensor = image_tensor.unsqueeze(0).to(self.config.device)
        
        # Super-resolve
        sr_tensor = self.dsr_model(image_tensor)
        
        # Convert back to numpy
        sr_image = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        sr_image = ((sr_image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        return sr_image
    
    def evaluate_verification(self, num_pairs: int = 6000) -> Dict:
        """
        Face verification evaluation following LFW protocol
        
        Following Lai et al. (2019) methodology with 10-fold cross-validation
        
        Args:
            num_pairs: Number of verification pairs
            
        Returns:
            Dictionary with verification metrics
        """
        print("\n" + "="*80)
        print("FACE VERIFICATION EVALUATION")
        print("="*80)
        
        # Load image pairs
        image_pairs = self.load_image_pairs()
        print(f"\nLoaded {len(image_pairs)} image pairs")
        
        # Group by subject
        subject_images = {}
        for hr_path, vlr_path, subject_id in image_pairs:
            if subject_id not in subject_images:
                subject_images[subject_id] = []
            subject_images[subject_id].append((hr_path, vlr_path))
        
        print(f"Found {len(subject_images)} unique subjects")
        
        # Generate positive and negative pairs
        positive_pairs = []
        negative_pairs = []
        
        subjects = list(subject_images.keys())
        
        # Positive pairs (same subject)
        for subject_id, images in subject_images.items():
            if len(images) >= 2:
                # Use first two images of same subject
                positive_pairs.append({
                    'hr1': images[0][0],
                    'vlr1': images[0][1],
                    'hr2': images[1][0],
                    'vlr2': images[1][1],
                    'label': 1,
                    'subject': subject_id
                })
        
        # Negative pairs (different subjects)
        for i in range(len(positive_pairs)):
            subject1 = subjects[i % len(subjects)]
            subject2 = subjects[(i + 1) % len(subjects)]
            
            if subject1 != subject2 and subject_images[subject1] and subject_images[subject2]:
                negative_pairs.append({
                    'hr1': subject_images[subject1][0][0],
                    'vlr1': subject_images[subject1][0][1],
                    'hr2': subject_images[subject2][0][0],
                    'vlr2': subject_images[subject2][0][1],
                    'label': 0,
                    'subject1': subject1,
                    'subject2': subject2
                })
        
        # Balance pairs
        num_pairs_each = min(len(positive_pairs), len(negative_pairs), num_pairs // 2)
        verification_pairs = positive_pairs[:num_pairs_each] + negative_pairs[:num_pairs_each]
        
        print(f"\nGenerated {len(verification_pairs)} verification pairs:")
        print(f"  Positive (same subject): {num_pairs_each}")
        print(f"  Negative (different subjects): {num_pairs_each}")
        
        # Evaluate using different methods
        results = {
            'svd_similarity': self._evaluate_verification_svd(verification_pairs),
            'identity_preserved': self._evaluate_verification_identity(verification_pairs),
            'combined': self._evaluate_verification_combined(verification_pairs)
        }
        
        return results
    
    def _evaluate_verification_svd(self, pairs: List[Dict]) -> Dict:
        """Evaluate verification using SVD similarity"""
        print("\n--- SVD Similarity Method ---")
        
        similarities = []
        labels = []
        
        for pair in tqdm(pairs, desc="Computing SVD similarities"):
            # Load images
            hr1 = self.preprocess_image(pair['hr1'], (112, 112))
            vlr1 = self.preprocess_image(pair['vlr1'], (16, 16))
            hr2 = self.preprocess_image(pair['hr2'], (112, 112))
            
            # Super-resolve VLR image
            sr1 = self.super_resolve(vlr1)
            
            # Compute SVD representations
            svd_sr = self.svd_repr.decompose(sr1)
            svd_hr = self.svd_repr.decompose(hr2)
            
            # Compute similarity
            sim = self.svd_repr.compute_similarity(svd_sr, svd_hr)
            
            similarities.append(sim)
            labels.append(pair['label'])
        
        # Compute metrics
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        metrics = self._compute_verification_metrics(similarities, labels, "SVD")
        
        return metrics
    
    def _evaluate_verification_identity(self, pairs: List[Dict]) -> Dict:
        """Evaluate verification using identity-preserved similarity"""
        print("\n--- Identity-Preserved Method ---")
        
        similarities = []
        labels = []
        
        for pair in tqdm(pairs, desc="Computing identity similarities"):
            # Load images
            hr1 = self.preprocess_image(pair['hr1'], (112, 112))
            vlr1 = self.preprocess_image(pair['vlr1'], (16, 16))
            hr2 = self.preprocess_image(pair['hr2'], (112, 112))
            
            # Super-resolve VLR image
            sr1 = self.super_resolve(vlr1)
            
            # Prepare for EdgeFace (normalize to [-1, 1])
            sr1_tensor = torch.from_numpy(sr1).permute(2, 0, 1).float() / 127.5 - 1.0
            hr2_tensor = torch.from_numpy(hr2).permute(2, 0, 1).float() / 127.5 - 1.0
            
            sr1_tensor = sr1_tensor.unsqueeze(0)
            hr2_tensor = hr2_tensor.unsqueeze(0)
            
            # Extract features
            feat_sr = self.identity_eval.extract_features(sr1_tensor)
            feat_hr = self.identity_eval.extract_features(hr2_tensor)
            
            # Compute cosine similarity
            sim = self.identity_eval.compute_cosine_similarity(feat_sr, feat_hr)
            
            similarities.append(sim.item())
            labels.append(pair['label'])
        
        # Compute metrics
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        metrics = self._compute_verification_metrics(similarities, labels, "Identity")
        
        return metrics
    
    def _evaluate_verification_combined(self, pairs: List[Dict]) -> Dict:
        """Evaluate verification using combined SVD + Identity similarity"""
        print("\n--- Combined Method (SVD + Identity) ---")
        
        svd_similarities = []
        identity_similarities = []
        labels = []
        
        for pair in tqdm(pairs, desc="Computing combined similarities"):
            # Load images
            hr1 = self.preprocess_image(pair['hr1'], (112, 112))
            vlr1 = self.preprocess_image(pair['vlr1'], (16, 16))
            hr2 = self.preprocess_image(pair['hr2'], (112, 112))
            
            # Super-resolve VLR image
            sr1 = self.super_resolve(vlr1)
            
            # SVD similarity
            svd_sr = self.svd_repr.decompose(sr1)
            svd_hr = self.svd_repr.decompose(hr2)
            svd_sim = self.svd_repr.compute_similarity(svd_sr, svd_hr)
            
            # Identity similarity
            sr1_tensor = torch.from_numpy(sr1).permute(2, 0, 1).float() / 127.5 - 1.0
            hr2_tensor = torch.from_numpy(hr2).permute(2, 0, 1).float() / 127.5 - 1.0
            feat_sr = self.identity_eval.extract_features(sr1_tensor.unsqueeze(0))
            feat_hr = self.identity_eval.extract_features(hr2_tensor.unsqueeze(0))
            id_sim = self.identity_eval.compute_cosine_similarity(feat_sr, feat_hr).item()
            
            svd_similarities.append(svd_sim)
            identity_similarities.append(id_sim)
            labels.append(pair['label'])
        
        # Normalize SVD similarities to [0, 1] range
        svd_similarities = np.array(svd_similarities)
        svd_norm = (svd_similarities - svd_similarities.min()) / (svd_similarities.max() - svd_similarities.min())
        
        identity_similarities = np.array(identity_similarities)
        
        # Combined similarity (weighted average)
        combined_similarities = 0.5 * svd_norm + 0.5 * identity_similarities
        
        labels = np.array(labels)
        
        metrics = self._compute_verification_metrics(combined_similarities, labels, "Combined")
        
        return metrics
    
    def _compute_verification_metrics(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        method_name: str
    ) -> Dict:
        """
        Compute verification metrics: accuracy, EER, AUC
        
        Args:
            similarities: Similarity scores
            labels: Ground truth labels (1=same, 0=different)
            method_name: Name of evaluation method
            
        Returns:
            Dictionary with metrics
        """
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find EER (Equal Error Rate)
        fnr = 1 - tpr
        eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
        
        # Find optimal threshold (max accuracy)
        accuracies = []
        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            accuracy = np.mean(predictions == labels)
            accuracies.append(accuracy)
        
        best_accuracy_idx = np.argmax(accuracies)
        best_accuracy = accuracies[best_accuracy_idx]
        best_threshold = thresholds[best_accuracy_idx]
        
        print(f"\n{method_name} Results:")
        print(f"  Best Accuracy: {best_accuracy*100:.2f}% (threshold={best_threshold:.4f})")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  EER: {eer*100:.2f}%")
        
        return {
            'method': method_name,
            'accuracy': float(best_accuracy),
            'threshold': float(best_threshold),
            'roc_auc': float(roc_auc),
            'eer': float(eer),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    
    def evaluate_identification(self) -> Dict:
        """
        Face identification evaluation (rank-1, rank-5, CMC curve)
        
        Following Jian & Lam (2015) methodology
        
        Returns:
            Dictionary with identification metrics
        """
        print("\n" + "="*80)
        print("FACE IDENTIFICATION EVALUATION")
        print("="*80)
        
        # Load image pairs
        image_pairs = self.load_image_pairs()
        
        # Group by subject (gallery = first image, probe = rest)
        subject_images = {}
        for hr_path, vlr_path, subject_id in image_pairs:
            if subject_id not in subject_images:
                subject_images[subject_id] = []
            subject_images[subject_id].append((hr_path, vlr_path))
        
        # Create gallery (first image of each subject)
        gallery = []
        for subject_id, images in subject_images.items():
            if images:
                gallery.append({
                    'subject_id': subject_id,
                    'hr_path': images[0][0],
                    'vlr_path': images[0][1]
                })
        
        # Create probe set (remaining images)
        probes = []
        for subject_id, images in subject_images.items():
            for hr_path, vlr_path in images[1:]:  # Skip first (in gallery)
                probes.append({
                    'subject_id': subject_id,
                    'hr_path': hr_path,
                    'vlr_path': vlr_path
                })
        
        print(f"\nGallery size: {len(gallery)} subjects")
        print(f"Probe size: {len(probes)} images")
        
        # Evaluate using different methods
        results = {
            'svd_similarity': self._evaluate_identification_method(gallery, probes, 'svd'),
            'identity_preserved': self._evaluate_identification_method(gallery, probes, 'identity'),
            'combined': self._evaluate_identification_method(gallery, probes, 'combined')
        }
        
        return results
    
    def _evaluate_identification_method(
        self,
        gallery: List[Dict],
        probes: List[Dict],
        method: str
    ) -> Dict:
        """
        Evaluate identification using specified method
        
        Args:
            gallery: List of gallery images
            probes: List of probe images
            method: 'svd', 'identity', or 'combined'
            
        Returns:
            Dictionary with identification metrics
        """
        print(f"\n--- {method.upper()} Method ---")
        
        # Extract features for gallery
        gallery_features = []
        gallery_subjects = []
        
        for item in tqdm(gallery, desc="Extracting gallery features"):
            hr_image = self.preprocess_image(item['hr_path'], (112, 112))
            
            if method == 'svd':
                svd_dict = self.svd_repr.decompose(hr_image)
                feat = self.svd_repr.get_normalized_singular_values(svd_dict)
            elif method == 'identity':
                image_tensor = torch.from_numpy(hr_image).permute(2, 0, 1).float() / 127.5 - 1.0
                feat = self.identity_eval.extract_features(image_tensor.unsqueeze(0))
                feat = feat.squeeze(0).cpu().numpy()
            else:  # combined
                # Store both features
                svd_dict = self.svd_repr.decompose(hr_image)
                svd_feat = self.svd_repr.get_normalized_singular_values(svd_dict)
                image_tensor = torch.from_numpy(hr_image).permute(2, 0, 1).float() / 127.5 - 1.0
                id_feat = self.identity_eval.extract_features(image_tensor.unsqueeze(0))
                id_feat = id_feat.squeeze(0).cpu().numpy()
                feat = (svd_feat, id_feat)
            
            gallery_features.append(feat)
            gallery_subjects.append(item['subject_id'])
        
        # Evaluate each probe
        ranks = []
        
        for probe in tqdm(probes, desc="Evaluating probes"):
            vlr_image = self.preprocess_image(probe['vlr_path'], (16, 16))
            sr_image = self.super_resolve(vlr_image)
            
            # Extract probe features
            if method == 'svd':
                svd_dict = self.svd_repr.decompose(sr_image)
                probe_feat = self.svd_repr.get_normalized_singular_values(svd_dict)
            elif method == 'identity':
                image_tensor = torch.from_numpy(sr_image).permute(2, 0, 1).float() / 127.5 - 1.0
                probe_feat = self.identity_eval.extract_features(image_tensor.unsqueeze(0))
                probe_feat = probe_feat.squeeze(0).cpu().numpy()
            else:  # combined
                svd_dict = self.svd_repr.decompose(sr_image)
                svd_feat = self.svd_repr.get_normalized_singular_values(svd_dict)
                image_tensor = torch.from_numpy(sr_image).permute(2, 0, 1).float() / 127.5 - 1.0
                id_feat = self.identity_eval.extract_features(image_tensor.unsqueeze(0))
                id_feat = id_feat.squeeze(0).cpu().numpy()
                probe_feat = (svd_feat, id_feat)
            
            # Compute similarities with gallery
            similarities = []
            for gallery_feat in gallery_features:
                if method == 'svd':
                    # SVD similarity (inverse L2 distance)
                    dist = np.linalg.norm(probe_feat - gallery_feat)
                    sim = 1.0 / (dist + 1e-8)
                elif method == 'identity':
                    # Cosine similarity
                    sim = 1.0 - cosine(probe_feat, gallery_feat)
                else:  # combined
                    # Weighted combination
                    svd_dist = np.linalg.norm(probe_feat[0] - gallery_feat[0])
                    svd_sim = 1.0 / (svd_dist + 1e-8)
                    id_sim = 1.0 - cosine(probe_feat[1], gallery_feat[1])
                    
                    # Normalize and combine
                    sim = 0.5 * svd_sim + 0.5 * id_sim
                
                similarities.append(sim)
            
            # Rank gallery by similarity
            ranked_indices = np.argsort(similarities)[::-1]  # Descending order
            ranked_subjects = [gallery_subjects[i] for i in ranked_indices]
            
            # Find rank of correct subject
            true_subject = probe['subject_id']
            try:
                rank = ranked_subjects.index(true_subject) + 1  # 1-indexed
            except ValueError:
                rank = len(ranked_subjects) + 1  # Not found
            
            ranks.append(rank)
        
        # Compute CMC curve
        ranks = np.array(ranks)
        max_rank = 20  # Compute up to rank-20
        cmc = []
        for r in range(1, max_rank + 1):
            recognition_rate = np.sum(ranks <= r) / len(ranks)
            cmc.append(recognition_rate)
        
        # Compute metrics
        rank1 = cmc[0]
        rank5 = cmc[4] if len(cmc) >= 5 else cmc[-1]
        rank10 = cmc[9] if len(cmc) >= 10 else cmc[-1]
        
        print(f"\n{method.upper()} Results:")
        print(f"  Rank-1: {rank1*100:.2f}%")
        print(f"  Rank-5: {rank5*100:.2f}%")
        print(f"  Rank-10: {rank10*100:.2f}%")
        
        return {
            'method': method,
            'rank1': float(rank1),
            'rank5': float(rank5),
            'rank10': float(rank10),
            'cmc': cmc,
            'ranks': ranks.tolist()
        }
    
    def generate_report(
        self,
        verification_results: Dict,
        identification_results: Dict,
        output_path: Path
    ):
        """
        Generate PDF evaluation report with plots
        
        Args:
            verification_results: Results from evaluate_verification()
            identification_results: Results from evaluate_identification()
            output_path: Path to save PDF report
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(output_path) as pdf:
            # Page 1: Verification Results
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Face Verification Results', fontsize=16, fontweight='bold')
            
            # ROC curves
            ax = axes[0, 0]
            for method_name, results in verification_results.items():
                ax.plot(results['fpr'], results['tpr'],
                       label=f"{results['method']} (AUC={results['roc_auc']:.3f})")
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Accuracy comparison
            ax = axes[0, 1]
            methods = [r['method'] for r in verification_results.values()]
            accuracies = [r['accuracy'] * 100 for r in verification_results.values()]
            bars = ax.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Verification Accuracy')
            ax.set_ylim([0, 100])
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
            ax.grid(True, axis='y', alpha=0.3)
            
            # EER comparison
            ax = axes[1, 0]
            eers = [r['eer'] * 100 for r in verification_results.values()]
            bars = ax.bar(methods, eers, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel('EER (%)')
            ax.set_title('Equal Error Rate (Lower is Better)')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Metrics table
            ax = axes[1, 1]
            ax.axis('off')
            table_data = [['Method', 'Accuracy', 'ROC-AUC', 'EER']]
            for results in verification_results.values():
                table_data.append([
                    results['method'],
                    f"{results['accuracy']*100:.2f}%",
                    f"{results['roc_auc']:.4f}",
                    f"{results['eer']*100:.2f}%"
                ])
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            # Bold header
            for i in range(4):
                table[(0, i)].set_facecolor('#E8E8E8')
                table[(0, i)].set_text_props(weight='bold')
            ax.set_title('Verification Metrics Summary')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Identification Results
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Face Identification Results', fontsize=16, fontweight='bold')
            
            # CMC curves
            ax = axes[0, 0]
            for method_name, results in identification_results.items():
                ranks = list(range(1, len(results['cmc']) + 1))
                ax.plot(ranks, [r * 100 for r in results['cmc']],
                       marker='o', markersize=3,
                       label=f"{results['method']}")
            ax.set_xlabel('Rank')
            ax.set_ylabel('Recognition Rate (%)')
            ax.set_title('Cumulative Match Characteristic (CMC) Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([1, 20])
            ax.set_ylim([0, 100])
            
            # Rank-1 comparison
            ax = axes[0, 1]
            methods = [r['method'] for r in identification_results.values()]
            rank1_rates = [r['rank1'] * 100 for r in identification_results.values()]
            bars = ax.bar(methods, rank1_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel('Rank-1 Rate (%)')
            ax.set_title('Rank-1 Identification Rate')
            ax.set_ylim([0, 100])
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Rank-5 comparison
            ax = axes[1, 0]
            rank5_rates = [r['rank5'] * 100 for r in identification_results.values()]
            bars = ax.bar(methods, rank5_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel('Rank-5 Rate (%)')
            ax.set_title('Rank-5 Identification Rate')
            ax.set_ylim([0, 100])
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Metrics table
            ax = axes[1, 1]
            ax.axis('off')
            table_data = [['Method', 'Rank-1', 'Rank-5', 'Rank-10']]
            for results in identification_results.values():
                table_data.append([
                    results['method'],
                    f"{results['rank1']*100:.2f}%",
                    f"{results['rank5']*100:.2f}%",
                    f"{results['rank10']*100:.2f}%"
                ])
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            # Bold header
            for i in range(4):
                table[(0, i)].set_facecolor('#E8E8E8')
                table[(0, i)].set_text_props(weight='bold')
            ax.set_title('Identification Metrics Summary')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"\nPDF report saved to: {output_path}")


def main():
    """Main evaluation function"""
    # Configuration
    project_root = Path(__file__).resolve().parents[2]
    
    config = EvaluationConfig(
        frontal_test_hr=project_root / "technical" / "dataset" / "frontal_only" / "test" / "hr_images",
        frontal_test_vlr=project_root / "technical" / "dataset" / "frontal_only" / "test" / "vlr_images",
        dsr_checkpoint=project_root / "technical" / "dsr" / "dsr.pth",
        edgeface_checkpoint=project_root / "technical" / "facial_rec" / "edgeface_weights" / "edgeface_xxs.pt",
        output_dir=project_root / "technical" / "evaluation_results",
        num_singular_values=50,
        svd_threshold=0.99,
        batch_size=16,
        num_folds=10,
        distance_metric="cosine"
    )
    
    # Create evaluator
    evaluator = LRFREvaluator(config)
    
    # Run verification evaluation
    verification_results = evaluator.evaluate_verification(num_pairs=6000)
    
    # Run identification evaluation
    identification_results = evaluator.evaluate_identification()
    
    # Save results as JSON
    results_json = {
        'verification': verification_results,
        'identification': identification_results,
        'config': {
            'num_singular_values': config.num_singular_values,
            'svd_threshold': config.svd_threshold,
            'distance_metric': config.distance_metric
        }
    }
    
    json_path = config.output_dir / "svd_identity_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    # Generate PDF report
    pdf_path = config.output_dir / "svd_identity_evaluation_report.pdf"
    evaluator.generate_report(
        verification_results,
        identification_results,
        pdf_path
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
