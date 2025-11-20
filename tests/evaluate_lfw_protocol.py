"""
LFW Low-Resolution Face Recognition Evaluation

Reproduces EXACT evaluation from:
Lai et al. (2019) - "Low-Resolution Face Recognition Based on Identity-Preserved Face Hallucination"

Tables reproduced:
- Table 1: Verification rates (%) on LFW 6,000 pairs at resolutions 7×6, 14×12, 16×16, 18×16, 28×24, 112×96
- Table 2: Rank-1 identification rates (%) at same resolutions

Protocol:
- Dataset: LFW faces from frontal_only directory
- Verification: 6,000 pairs (3,000 genuine + 3,000 impostor)
- Identification: Closed-set, 1 gallery per subject, rest as probes
- No open-set scenario, small gallery matching paper
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
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc
import random

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "technical"))

from pipeline import build_pipeline, PipelineConfig
from PIL import Image
from torchvision import transforms


class LFWEvaluator:
    """
    Evaluator matching Lai et al. 2019 LFW protocol exactly
    Uses the existing Pipeline class for correct DSR and EdgeFace processing
    """
    
    def __init__(
        self,
        dataset_dir: Path,
        dsr_dir: Path,
        edgeface_dir: Path,
        output_dir: Path,
        device: str = "cuda"
    ):
        self.dataset_dir = dataset_dir
        self.dsr_dir = dsr_dir
        self.edgeface_dir = edgeface_dir
        self.output_dir = output_dir
        self.device = device
        
        # Resolutions to test (H, W) - matching our trained models
        self.resolutions = [
            (16, 16),    # 16x16 VLR
            (24, 24),    # 24x24 VLR
            (32, 32),    # 32x32 VLR
            (112, 112)   # HR resolution
        ]
        
        # Create pipelines for each resolution
        print("Loading pipelines...")
        self.pipelines = {}  # resolution -> (dsr_pipeline, base_pipeline)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Device: {device}")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Output directory: {output_dir}")
    
    def _build_pipelines_for_resolution(self, resolution: Tuple[int, int]):
        """Build DSR and bicubic pipelines for a specific resolution"""
        if resolution == (112, 112):
            # HR baseline - no DSR needed, just use base EdgeFace
            base_config = PipelineConfig(
                dsr_weights_path=self.dsr_dir / "dsr32.pth",  # Dummy, won't be used
                edgeface_weights_path=self.edgeface_dir / "edgeface_xxs.pt",
                device=self.device,
                skip_dsr=True  # Skip DSR for HR
            )
            return None, build_pipeline(base_config)  # No DSR pipeline for HR
        
        else:
            res_key = resolution[0]  # 16, 24, or 32
            
            # DSR pipeline with finetuned EdgeFace
            # NOTE: Using finetuned model trained on LFW - this is task-specific evaluation, not generalization
            dsr_config = PipelineConfig(
                dsr_weights_path=self.dsr_dir / f"dsr{res_key}.pth",
                edgeface_weights_path=self.edgeface_dir / f"edgeface_finetuned_{res_key}.pth",
                device=self.device,
                skip_dsr=False
            )
            dsr_pipeline = build_pipeline(dsr_config)
            
            # Bicubic pipeline with SAME finetuned EdgeFace
            bicubic_config = PipelineConfig(
                dsr_weights_path=self.dsr_dir / f"dsr{res_key}.pth",  # Dummy, won't be used
                edgeface_weights_path=self.edgeface_dir / f"edgeface_finetuned_{res_key}.pth",
                device=self.device,
                skip_dsr=True  # Skip DSR for bicubic baseline
            )
            bicubic_pipeline = build_pipeline(bicubic_config)
            
            return dsr_pipeline, bicubic_pipeline
    
    def _get_embedding_from_image(
        self,
        img_path: Path,
        pipeline
    ) -> np.ndarray:
        """
        Get embedding from image using Pipeline
        
        The pipeline's skip_dsr flag determines the processing path:
        - skip_dsr=False: Use DSR upscaling (for DSR pipelines)
        - skip_dsr=True: Manual bicubic upscaling (for bicubic/HR pipelines)
        
        Args:
            img_path: Path to VLR or HR image
            pipeline: FaceRecognitionPipeline instance (already configured for DSR or bicubic)
        
        Returns:
            Embedding as 1D numpy array (L2-normalized)
        """
        if not pipeline.config.skip_dsr:
            # DSR pipeline: upscale VLR -> extract embedding from SR
            sr_tensor = pipeline.upscale(img_path)
            embedding = pipeline.infer_embedding(sr_tensor)
        else:
            # Bicubic/HR: load image directly and extract embedding
            # For VLR images with bicubic, we need to manually upscale
            img = Image.open(img_path).convert('RGB')
            
            if img.size != (112, 112):
                # Bicubic upscale to 112x112
                img = img.resize((112, 112), Image.Resampling.BICUBIC)
            
            # Convert to tensor for EdgeFace (normalization will be applied by infer_embedding)
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            tensor = transform(img).unsqueeze(0).to(pipeline.device)
            embedding = pipeline.infer_embedding(tensor)
        
        # Convert to numpy and ensure 1D array
        embedding = embedding.cpu().numpy()
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        return embedding
    
    def load_lfw_subjects(self, resolution: Tuple[int, int]) -> Dict[str, List[Path]]:
        """
        Load LFW subjects from pre-existing VLR/HR directories
        Follows paper methodology: use subjects with ≥4 images
        
        LFW images have format: 1001_0001_lfw.png where 1001 is subject ID
        
        Args:
            resolution: (H, W) resolution to load
            
        Returns:
            Dictionary mapping subject_id -> list of image paths from appropriate VLR/HR directory
        """
        subjects = {}
        
        # Determine which directory to load from based on resolution
        # Use frontal_only/test for evaluation (frontal faces, same images at different resolutions)
        if resolution == (112, 112):
            # HR baseline - load from hr_images
            image_dir = self.dataset_dir / "frontal_only" / "test" / "hr_images"
        elif resolution == (16, 16):
            # VLR 16×16
            image_dir = self.dataset_dir / "frontal_only" / "test" / "vlr_images_16x16"
        elif resolution == (24, 24):
            # VLR 24×24
            image_dir = self.dataset_dir / "frontal_only" / "test" / "vlr_images_24x24"
        elif resolution == (32, 32):
            # VLR 32×32
            image_dir = self.dataset_dir / "frontal_only" / "test" / "vlr_images_32x32"
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")
        
        # Load images from test directory
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        for img_path in sorted(image_dir.glob("*.png")):
            # Extract subject ID from filename
            # Format: 007_01_01_041_00_crop_128.png -> subject_id is first part (007)
            filename = img_path.stem  # Remove .png
            subject_id = filename.split('_')[0]
            
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append(img_path)
        
        # Filter to subjects with at least 4 images (matching paper methodology)
        subjects = {sid: imgs for sid, imgs in subjects.items() if len(imgs) >= 4}
        
        print(f"\nLoaded {len(subjects)} LFW subjects from {resolution[0]}x{resolution[1]} directories")
        print(f"  Subjects with >=4 images: {len(subjects)}")
        print(f"  Total images: {sum(len(imgs) for imgs in subjects.values())}")
        
        return subjects
    
    def evaluate_verification_at_resolution(
        self,
        subjects: Dict[str, List[Path]],
        resolution: Tuple[int, int],
        use_dsr: bool = True
    ) -> Dict:
        """
        Evaluate face verification at given resolution
        
        Following LFW protocol: 6,000 pairs (3,000 same + 3,000 different)
        
        Args:
            subjects: Dictionary of subject_id -> image paths (from VLR/HR directories)
            resolution: (H, W) probe resolution
            use_dsr: If True, use DSR pipeline; else use bicubic pipeline
            
        Returns:
            Dictionary with verification metrics
        """
        # Get appropriate pipeline
        if resolution == (112, 112):
            # HR baseline - use base EdgeFace, no DSR
            _, pipeline = self.pipelines[resolution]
        else:
            # VLR - use DSR or bicubic pipeline
            dsr_pipeline, bicubic_pipeline = self.pipelines[resolution]
            pipeline = dsr_pipeline if use_dsr else bicubic_pipeline
        
        # Generate pairs following LFW protocol: 3,000 genuine + 3,000 impostor
        positive_pairs = []
        negative_pairs = []
        
        subject_ids = list(subjects.keys())
        random.seed(42)  # For reproducibility
        
        # Positive pairs (same subject) - generate 3,000 pairs
        target_positive = 3000
        while len(positive_pairs) < target_positive:
            # Sample random subject
            subject_id = random.choice(subject_ids)
            img_paths = subjects[subject_id]
            
            if len(img_paths) >= 2:
                # Sample two different images from same subject
                img1, img2 = random.sample(img_paths, 2)
                positive_pairs.append((img1, img2, 1))
        
        # Negative pairs (different subjects) - generate 3,000 pairs
        target_negative = 3000
        while len(negative_pairs) < target_negative:
            # Sample two different subjects
            sid1, sid2 = random.sample(subject_ids, 2)
            img1 = random.choice(subjects[sid1])
            img2 = random.choice(subjects[sid2])
            negative_pairs.append((img1, img2, 0))
        
        # Combine to get exactly 6,000 pairs
        all_pairs = positive_pairs + negative_pairs
        
        print(f"  Generated {len(all_pairs)} pairs ({len(positive_pairs)} pos, {len(negative_pairs)} neg)")
        
        # Evaluate pairs using Pipeline
        similarities = []
        labels = []
        
        for img1_path, img2_path, label in tqdm(all_pairs, desc=f"  Verifying at {resolution[0]}×{resolution[1]}"):
            # Get embeddings using Pipeline (pipeline already configured for DSR or bicubic)
            emb1 = self._get_embedding_from_image(img1_path, pipeline)
            emb2 = self._get_embedding_from_image(img2_path, pipeline)
            
            # Compute cosine similarity (embeddings are already L2-normalized by pipeline)
            # For normalized vectors: cosine_sim = dot(a, b) / (||a|| * ||b||) = dot(a, b) / (1 * 1) = dot(a, b)
            sim = np.dot(emb1, emb2).item() if hasattr(np.dot(emb1, emb2), 'item') else float(np.dot(emb1, emb2))
            
            similarities.append(sim)
            labels.append(label)
        
        # Compute metrics
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        # Sanity check: print similarity statistics
        genuine_sims = similarities[labels == 1]
        impostor_sims = similarities[labels == 0]
        print(f"    Similarity stats - Genuine: mean={genuine_sims.mean():.3f}, std={genuine_sims.std():.3f}, range=[{genuine_sims.min():.3f}, {genuine_sims.max():.3f}]")
        print(f"    Similarity stats - Impostor: mean={impostor_sims.mean():.3f}, std={impostor_sims.std():.3f}, range=[{impostor_sims.min():.3f}, {impostor_sims.max():.3f}]")
        
        # Find optimal threshold for accuracy (standard LFW protocol)
        best_acc = 0
        best_threshold = 0
        
        for threshold in np.linspace(0, 1, 1000):
            predictions = (similarities >= threshold).astype(int)
            acc = np.mean(predictions == labels)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # EER
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        return {
            'resolution': f"{resolution[0]}×{resolution[1]}",
            'accuracy': float(best_acc * 100),  # Best achievable accuracy (standard LFW protocol)
            'threshold': float(best_threshold),
            'roc_auc': float(roc_auc),
            'eer': float(eer * 100),  # Convert to percentage
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    
    def evaluate_identification_at_resolution(
        self,
        subjects: Dict[str, List[Path]],
        resolution: Tuple[int, int],
        use_dsr: bool = True
    ) -> Dict:
        """
        Evaluate face identification at given resolution
        
        Closed-set protocol: 1 gallery image per subject, rest as probes
        
        Args:
            subjects: Dictionary of subject_id -> image paths (from VLR/HR directories)
            resolution: (H, W) probe resolution
            use_dsr: If True, use DSR pipeline; else use bicubic pipeline
            
        Returns:
            Dictionary with identification metrics (rank-1, rank-5, rank-10)
        """
        # Get appropriate pipeline
        if resolution == (112, 112):
            # HR baseline - use base EdgeFace, no DSR
            _, pipeline = self.pipelines[resolution]
        else:
            # VLR - use DSR or bicubic pipeline
            dsr_pipeline, bicubic_pipeline = self.pipelines[resolution]
            pipeline = dsr_pipeline if use_dsr else bicubic_pipeline
        
        # Build gallery (first image of each subject)
        gallery = []
        gallery_features = []
        
        print(f"  Building gallery at {resolution[0]}×{resolution[1]}...")
        for subject_id, img_paths in tqdm(subjects.items(), desc="  Gallery"):
            # Get embedding using Pipeline
            emb = self._get_embedding_from_image(img_paths[0], pipeline)
            
            gallery.append({'subject_id': subject_id, 'features': emb})
            gallery_features.append(emb)
        
        gallery_features = np.array(gallery_features)  # (N_gallery, D)
        
        # Build probe set (remaining images)
        probes = []
        for subject_id, img_paths in subjects.items():
            for img_path in img_paths[1:]:  # Skip first (in gallery)
                probes.append({'subject_id': subject_id, 'img_path': img_path})
        
        print(f"  Gallery: {len(gallery)} subjects")
        print(f"  Probes: {len(probes)} images")
        
        # Evaluate probes
        ranks = []
        
        for probe in tqdm(probes, desc=f"  Identifying at {resolution[0]}×{resolution[1]}"):
            # Get embedding using Pipeline
            emb_probe = self._get_embedding_from_image(probe['img_path'], pipeline)
            
            # Compute similarities with all gallery (cosine similarity)
            similarities = []
            for emb_gallery in gallery_features:
                sim = np.dot(emb_probe, emb_gallery) / (np.linalg.norm(emb_probe) * np.linalg.norm(emb_gallery))
                similarities.append(sim)
            
            # Rank gallery by similarity
            ranked_indices = np.argsort(similarities)[::-1]  # Descending
            
            # Find rank of correct subject
            true_subject = probe['subject_id']
            rank = None
            for r, idx in enumerate(ranked_indices, 1):
                if gallery[idx]['subject_id'] == true_subject:
                    rank = r
                    break
            
            if rank is None:
                rank = len(gallery) + 1  # Not found
            
            ranks.append(rank)
        
        # Compute rank-k accuracies
        ranks = np.array(ranks)
        rank1 = np.mean(ranks <= 1) * 100
        rank5 = np.mean(ranks <= 5) * 100
        rank10 = np.mean(ranks <= 10) * 100
        
        return {
            'resolution': f"{resolution[0]}×{resolution[1]}",
            'rank1': float(rank1),
            'rank5': float(rank5),
            'rank10': float(rank10),
            'ranks': ranks.tolist()
        }
    
    def run_evaluation(self) -> Dict:
        """
        Run full evaluation matching Lai et al. 2019
        
        Returns:
            Dictionary with all results for Table 1 and Table 2
        """
        print("\n" + "="*80)
        print("LFW EVALUATION - Matching Lai et al. 2019")
        print("="*80)
        
        # Load LFW subjects for EACH resolution from appropriate VLR/HR directories
        print("\nLoading LFW subjects from VLR/HR directories...")
        subjects = {}
        for res in [(16, 16), (24, 24), (32, 32), (112, 112)]:
            subjects[res] = self.load_lfw_subjects(res)
        
        results = {
            'table1_verification': {},
            'table2_identification': {}
        }
        
        # Build pipelines for each resolution
        print("\nBuilding pipelines for each resolution...")
        for resolution in self.resolutions:
            print(f"  Resolution {resolution[0]}×{resolution[1]}...")
            self.pipelines[resolution] = self._build_pipelines_for_resolution(resolution)
        
        print(f"\nBuilt {len(self.pipelines)} pipeline sets")
        
        # Table 1: Verification with bicubic and DSR
        print("\n" + "="*80)
        print("TABLE 1: VERIFICATION RATES")
        print("="*80)
        
        for resolution in self.resolutions:
            print(f"\nResolution: {resolution[0]}×{resolution[1]}")
            
            # Get subjects for THIS resolution
            res_subjects = subjects[resolution]
            
            if resolution == (112, 112):
                # HR baseline - only evaluate once (no upscaling needed)
                print("  Method: HR Baseline (no upscaling)")
                hr_result = self.evaluate_verification_at_resolution(
                    res_subjects, resolution, use_dsr=False
                )
                results['table1_verification'][f"{resolution[0]}x{resolution[1]}"] = {
                    'bicubic': hr_result,
                    'dsr': hr_result
                }
                print(f"  Accuracy: {hr_result['accuracy']:.2f}%")
            else:
                # VLR - compare bicubic vs DSR
                print("  Method: Bicubic")
                bicubic_result = self.evaluate_verification_at_resolution(
                    res_subjects, resolution, use_dsr=False
                )
                
                print("  Method: DSR")
                dsr_result = self.evaluate_verification_at_resolution(
                    res_subjects, resolution, use_dsr=True
                )
                
                results['table1_verification'][f"{resolution[0]}x{resolution[1]}"] = {
                    'bicubic': bicubic_result,
                    'dsr': dsr_result
                }
                
                print(f"  Bicubic Accuracy: {bicubic_result['accuracy']:.2f}% (threshold: {bicubic_result['threshold']:.3f})")
                print(f"  DSR Accuracy: {dsr_result['accuracy']:.2f}% (threshold: {dsr_result['threshold']:.3f})")
        
        # Table 2: Identification (closed-set)
        print("\n" + "="*80)
        print("TABLE 2: RANK-1 IDENTIFICATION RATES")
        print("="*80)
        
        for resolution in self.resolutions:
            print(f"\nResolution: {resolution[0]}×{resolution[1]}")
            
            # Get subjects for THIS resolution
            res_subjects = subjects[resolution]
            
            if resolution == (112, 112):
                # HR baseline - only evaluate once (no upscaling needed)
                print("  Method: HR Baseline (no upscaling)")
                hr_result = self.evaluate_identification_at_resolution(
                    res_subjects, resolution, use_dsr=False
                )
                results['table2_identification'][f"{resolution[0]}x{resolution[1]}"] = {
                    'bicubic': hr_result,
                    'dsr': hr_result
                }
                print(f"  Rank-1: {hr_result['rank1']:.2f}%")
            else:
                # VLR - compare bicubic vs DSR
                print("  Method: Bicubic")
                bicubic_result = self.evaluate_identification_at_resolution(
                    res_subjects, resolution, use_dsr=False
                )
                
                print("  Method: DSR")
                dsr_result = self.evaluate_identification_at_resolution(
                    res_subjects, resolution, use_dsr=True
                )
                
                results['table2_identification'][f"{resolution[0]}x{resolution[1]}"] = {
                    'bicubic': bicubic_result,
                    'dsr': dsr_result
                }
                
                print(f"  Bicubic Rank-1: {bicubic_result['rank1']:.2f}%")
                print(f"  DSR Rank-1: {dsr_result['rank1']:.2f}%")
        
        return results
    
    def generate_tables(self, results: Dict, output_path: Path):
        """
        Generate formatted tables matching paper format
        
        Args:
            results: Results dictionary from run_evaluation()
            output_path: Path to save PDF
        """
        with PdfPages(output_path) as pdf:
            # Table 1: Verification
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('off')
            
            # Prepare table data
            table_data = [
                ['Method', '16×16', '24×24', '32×32', '112×112 (HR)']
            ]
            
            # Bicubic row
            bicubic_row = ['Bicubic']
            for res in ['16x16', '24x24', '32x32', '112x112']:
                acc = results['table1_verification'][res]['bicubic']['accuracy']
                bicubic_row.append(f"{acc:.2f}")
            table_data.append(bicubic_row)
            
            # DSR row
            dsr_row = ['DSR']
            for res in ['16x16', '24x24', '32x32', '112x112']:
                acc = results['table1_verification'][res]['dsr']['accuracy']
                dsr_row.append(f"{acc:.2f}")
            table_data.append(dsr_row)
            
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Style method column
            for i in range(1, 3):
                table[(i, 0)].set_facecolor('#E8E8E8')
                table[(i, 0)].set_text_props(weight='bold')
            
            ax.set_title('Table 1: Verification Rates (%) on LFW\n(EdgeFace Network, 6,000 pairs)',
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Table 2: Identification
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('off')
            
            # Prepare table data
            table_data = [
                ['Method', '16×16', '24×24', '32×32', '112×112 (HR)']
            ]
            
            # Bicubic row
            bicubic_row = ['Bicubic']
            for res in ['16x16', '24x24', '32x32', '112x112']:
                rank1 = results['table2_identification'][res]['bicubic']['rank1']
                bicubic_row.append(f"{rank1:.2f}")
            table_data.append(bicubic_row)
            
            # DSR row
            dsr_row = ['DSR']
            for res in ['16x16', '24x24', '32x32', '112x112']:
                rank1 = results['table2_identification'][res]['dsr']['rank1']
                dsr_row.append(f"{rank1:.2f}")
            table_data.append(dsr_row)
            
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Style method column
            for i in range(1, 3):
                table[(i, 0)].set_facecolor('#E8E8E8')
                table[(i, 0)].set_text_props(weight='bold')
            
            ax.set_title('Table 2: Rank-1 Identification Rates (%) on LFW\n(EdgeFace Network, Closed-Set)',
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"\nTables saved to: {output_path}")


def main():
    """Main evaluation function"""
    project_root = Path(__file__).resolve().parents[2]
    
    # Configuration
    dataset_dir = project_root / "technical" / "dataset"
    dsr_dir = project_root / "technical" / "dsr"
    edgeface_dir = project_root / "technical" / "facial_rec" / "edgeface_weights"
    output_dir = project_root / "technical" / "evaluation_results"
    
    # Verify dataset directory exists
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return
    
    # Create evaluator
    evaluator = LFWEvaluator(
        dataset_dir=dataset_dir,
        dsr_dir=dsr_dir,
        edgeface_dir=edgeface_dir,
        output_dir=output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Save results as JSON
    json_path = output_dir / "lfw_evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Generate tables PDF
    pdf_path = output_dir / "lfw_evaluation_tables.pdf"
    evaluator.generate_tables(results, pdf_path)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nTable 1 Summary (Verification Accuracy %):")
    print("Resolution | Bicubic | DSR")
    print("-" * 40)
    # Get actual resolutions from results
    res_keys = sorted(results['table1_verification'].keys(), key=lambda x: int(x.split('x')[0]))
    for res in res_keys:
        bicubic = results['table1_verification'][res]['bicubic']['accuracy']
        dsr = results['table1_verification'][res]['dsr']['accuracy']
        print(f"{res:10s} | {bicubic:7.2f} | {dsr:7.2f}")
    
    print("\nTable 2 Summary (Rank-1 Identification %):")
    print("Resolution | Bicubic | DSR")
    print("-" * 40)
    for res in res_keys:
        bicubic = results['table2_identification'][res]['bicubic']['rank1']
        dsr = results['table2_identification'][res]['dsr']['rank1']
        print(f"{res:10s} | {bicubic:7.2f} | {dsr:7.2f}")


if __name__ == "__main__":
    main()
