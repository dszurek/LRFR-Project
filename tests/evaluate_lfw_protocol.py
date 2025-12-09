"""
LFW Low-Resolution Face Recognition Evaluation

Reproduces evaluation methodology from:
Lai et al. (2019) - "Low-Resolution Face Recognition Based on Identity-Preserved Face Hallucination"

Metrics:
1. Image Quality: PSNR, SSIM
2. Verification: Accuracy, VR@FAR (0.1%, 1%), Similarity Stats
3. Identification: Rank-1, Rank-5, Rank-10, Rank-20

Comparisons:
- Resolutions: 16x16, 24x24, 32x32, 112x112 (HR)
- Upscaling: Bicubic vs DSR (HybridDSR)
- Recognition: Default EdgeFace vs Finetuned EdgeFace
"""

import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, auc
import random
from skimage.metrics import structural_similarity as ssim_func

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from technical.pipeline.pipeline import build_pipeline, PipelineConfig
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
        
        # Resolutions to test (H, W)
        self.resolutions = [
            (16, 16),    # 16x16 VLR
            (24, 24),    # 24x24 VLR
            (32, 32),    # 32x32 VLR
            (112, 112)   # HR resolution
        ]
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Device: {device}")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Output directory: {output_dir}")
        
        self.pipelines = {} # (res, method) -> pipeline

    def _get_pipeline(self, resolution: Tuple[int, int], use_dsr: bool, use_finetuned: bool) -> Any:
        """Build or retrieve a pipeline for a specific configuration"""
        key = (resolution, use_dsr, use_finetuned)
        if key in self.pipelines:
            return self.pipelines[key]
            
        res_val = resolution[0]
        
        # Determine weights
        if use_finetuned and resolution != (112, 112):
            edgeface_path = self.edgeface_dir / f"edgeface_finetuned_{res_val}.pth"
        else:
            edgeface_path = self.edgeface_dir / "edgeface_xxs.pt"
            
        dsr_path = self.dsr_dir / f"hybrid_dsr{res_val}.pth"
        
        # Config
        config = PipelineConfig(
            dsr_weights_path=dsr_path,
            edgeface_weights_path=edgeface_path,
            device=self.device,
            skip_dsr=not use_dsr
        )
        
        # For HR, we always skip DSR
        if resolution == (112, 112):
            config.skip_dsr = True
            
        pipeline = build_pipeline(config)
        self.pipelines[key] = pipeline
        return pipeline

    def _get_embedding_from_image(
        self,
        img_path: Path,
        pipeline
    ) -> np.ndarray:
        """Get embedding from image using Pipeline"""
        if not pipeline.config.skip_dsr:
            # DSR pipeline: upscale VLR -> extract embedding from SR
            sr_tensor = pipeline.upscale(img_path)
            embedding = pipeline.infer_embedding(sr_tensor)
        else:
            # Bicubic/HR: load image directly and extract embedding
            img = Image.open(img_path).convert('RGB')
            
            if img.size != (112, 112):
                # Bicubic upscale to 112x112
                img = img.resize((112, 112), Image.Resampling.BICUBIC)
            
            # Convert to tensor for EdgeFace
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
        """Load LFW subjects from pre-existing VLR/HR directories"""
        subjects = {}
        
        if resolution == (112, 112):
            image_dir = self.dataset_dir / "test_processed" / "hr_images"
        elif resolution == (16, 16):
            image_dir = self.dataset_dir / "test_processed" / "vlr_images_16x16"
        elif resolution == (24, 24):
            image_dir = self.dataset_dir / "test_processed" / "vlr_images_24x24"
        elif resolution == (32, 32):
            image_dir = self.dataset_dir / "test_processed" / "vlr_images_32x32"
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")
        
        if not image_dir.exists():
            # Fallback to generic vlr_images if specific doesn't exist
            if resolution != (112, 112):
                 image_dir = self.dataset_dir / "test_processed" / "vlr_images"
            
            if not image_dir.exists():
                raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        for img_path in sorted(image_dir.glob("*.png")):
            filename = img_path.stem
            subject_id = filename.split('_')[0]
            
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append(img_path)
        
        # Filter to subjects with at least 4 images
        subjects = {sid: imgs for sid, imgs in subjects.items() if len(imgs) >= 4}
        
        print(f"Loaded {len(subjects)} LFW subjects from {image_dir}")
        return subjects

    def calculate_image_metrics(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float]:
        """Calculate PSNR and SSIM between two images (H, W, C) range [0, 255]"""
        # PSNR
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * math.log10(255.0 / math.sqrt(mse))
            
        # SSIM
        # win_size must be odd and <= min(H, W). Images are 112x112, so 7 is safe.
        ssim = ssim_func(img1, img2, channel_axis=2, data_range=255, win_size=3)
        
        return psnr, ssim

    def evaluate_image_quality(
        self,
        subjects: Dict[str, List[Path]],
        pipeline,
        hr_subjects: Dict[str, List[Path]
    ]) -> Dict:
        """Evaluate PSNR and SSIM against HR ground truth"""
        psnr_list = []
        ssim_list = []
        
        # We need to compare VLR->SR against HR
        # Iterate through subjects that exist in both
        common_subjects = set(subjects.keys()) & set(hr_subjects.keys())
        
        # Sample max 500 images to save time
        sample_count = 0
        max_samples = 500
        
        for subject_id in common_subjects:
            vlr_paths = subjects[subject_id]
            hr_paths = hr_subjects[subject_id]
            
            # Assuming filenames match (or are aligned by sort order)
            # LFW format: {id}_{name}_{num}.png. 
            # We'll try to match by filename suffix if possible, or just index
            
            min_len = min(len(vlr_paths), len(hr_paths))
            for i in range(min_len):
                if sample_count >= max_samples:
                    break
                
                vlr_path = vlr_paths[i]
                hr_path = hr_paths[i]
                
                # Get SR image (numpy 0-255)
                if not pipeline.config.skip_dsr:
                    sr_tensor = pipeline.upscale(vlr_path) # (1, 3, H, W)
                    sr_img = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
                else:
                    # Bicubic upscale
                    img = Image.open(vlr_path).convert('RGB')
                    img = img.resize((112, 112), Image.Resampling.BICUBIC)
                    sr_img = np.array(img).astype(float)
                
                # Get HR image
                hr_img = np.array(Image.open(hr_path).convert('RGB')).astype(float)
                
                # Ensure sizes match (HR might be different if not resized)
                if hr_img.shape != (112, 112, 3):
                     hr_img = np.array(Image.open(hr_path).convert('RGB').resize((112, 112))).astype(float)
                
                # Calculate metrics
                psnr, ssim = self.calculate_image_metrics(sr_img, hr_img)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                
                sample_count += 1
            
            if sample_count >= max_samples:
                break
                
        return {
            'psnr': float(np.mean(psnr_list)) if psnr_list else 0.0,
            'ssim': float(np.mean(ssim_list)) if ssim_list else 0.0
        }
    
    def evaluate_verification(
        self,
        subjects: Dict[str, List[Path]],
        pipeline
    ) -> Dict:
        """Evaluate face verification (1:1)"""
        positive_pairs = []
        negative_pairs = []
        subject_ids = list(subjects.keys())
        random.seed(42)
        
        # Positive pairs (3000)
        while len(positive_pairs) < 3000:
            subject_id = random.choice(subject_ids)
            img_paths = subjects[subject_id]
            if len(img_paths) >= 2:
                img1, img2 = random.sample(img_paths, 2)
                positive_pairs.append((img1, img2, 1))
        
        # Negative pairs (3000)
        while len(negative_pairs) < 3000:
            sid1, sid2 = random.sample(subject_ids, 2)
            img1 = random.choice(subjects[sid1])
            img2 = random.choice(subjects[sid2])
            negative_pairs.append((img1, img2, 0))
        
        all_pairs = positive_pairs + negative_pairs
        
        similarities = []
        labels = []
        
        for img1_path, img2_path, label in tqdm(all_pairs, desc="Verifying", leave=False):
            emb1 = self._get_embedding_from_image(img1_path, pipeline)
            emb2 = self._get_embedding_from_image(img2_path, pipeline)
            sim = np.dot(emb1, emb2).item()
            similarities.append(sim)
            labels.append(label)
        
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        # Stats
        gen_sims = similarities[labels == 1]
        imp_sims = similarities[labels == 0]
        
        # Find optimal threshold
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
        
        # VR @ FAR
        # Find TPR where FPR <= target
        def get_vr_at_far(target_far):
            # Ensure fpr is sorted and unique for interpolation
            unique_fpr, unique_tpr = np.unique(fpr, return_index=True), np.unique(tpr, return_index=True)
            
            # Find the index where FPR is closest to target_far from below
            idx = np.where(fpr <= target_far)[0]
            if len(idx) == 0:
                return 0.0 # No FPR below target
            
            # Get the highest TPR for FPR <= target_far
            return tpr[idx[-1]]
            
        vr_far_01 = get_vr_at_far(0.001) # 0.1%
        vr_far_1 = get_vr_at_far(0.01)   # 1%
        
        return {
            'accuracy': float(best_acc * 100),
            'threshold': float(best_threshold),
            'roc_auc': float(roc_auc),
            'eer': float(eer * 100),
            'vr_far_01': float(vr_far_01 * 100),
            'vr_far_1': float(vr_far_1 * 100),
            'gen_mean': float(np.mean(gen_sims)),
            'gen_std': float(np.std(gen_sims)),
            'imp_mean': float(np.mean(imp_sims)),
            'imp_std': float(np.std(imp_sims)),
            # Return raw data for plotting (will be excluded from JSON dump if not serializable, but okay for internal use)
            'fpr': fpr,
            'tpr': tpr,
            'gen_sims': gen_sims,
            'imp_sims': imp_sims
        }
    
    def evaluate_identification(
        self,
        subjects: Dict[str, List[Path]],
        pipeline
    ) -> Dict:
        """Evaluate identification (1:N)"""
        # Build gallery (first image of each subject)
        gallery = []
        gallery_features = []
        
        for subject_id, img_paths in tqdm(subjects.items(), desc="Gallery", leave=False):
            emb = self._get_embedding_from_image(img_paths[0], pipeline)
            gallery.append({'subject_id': subject_id, 'features': emb})
            gallery_features.append(emb)
        
        gallery_features = np.array(gallery_features)
        
        # Probes
        probes = []
        for subject_id, img_paths in subjects.items():
            for img_path in img_paths[1:]:
                probes.append({'subject_id': subject_id, 'img_path': img_path})
        
        ranks = []
        
        for probe in tqdm(probes, desc="Identifying", leave=False):
            emb_probe = self._get_embedding_from_image(probe['img_path'], pipeline)
            
            # Cosine similarity
            sims = np.dot(gallery_features, emb_probe) / (np.linalg.norm(gallery_features, axis=1) * np.linalg.norm(emb_probe))
            
            # Rank
            sorted_indices = np.argsort(sims)[::-1]
            true_subject = probe['subject_id']
            
            rank = None
            for r, idx in enumerate(sorted_indices, 1):
                if gallery[idx]['subject_id'] == true_subject:
                    rank = r
                    break
            
            if rank is None:
                rank = len(gallery) + 1
            
            ranks.append(rank)
        
        ranks = np.array(ranks)
        return {
            'rank1': float(np.mean(ranks <= 1) * 100),
            'rank5': float(np.mean(ranks <= 5) * 100),
            'rank10': float(np.mean(ranks <= 10) * 100),
            'rank20': float(np.mean(ranks <= 20) * 100),
            'ranks': ranks # Return raw ranks for CMC plot
        }

    def run_evaluation(self) -> Dict:
        print("\n" + "="*80)
        print("LFW EVALUATION - FULL SUITE (EXPANDED METRICS)")
        print("="*80)
        
        # Load LFW subjects
        print("\nLoading LFW subjects from VLR/HR directories...")
        subjects_by_res = {}
        for res in self.resolutions:
            subjects_by_res[res] = self.load_lfw_subjects(res)
            
        results = {
            'verification': {},
            'identification': {},
            'image_quality': {}
        }
        
        # Configs to run
        # (Name, UseDSR, UseFinetuned)
        configs = [
            ('Bicubic_Default', False, False),
            ('Bicubic_Finetuned', False, True),
            ('DSR_Default', True, False),
            ('DSR_Finetuned', True, True)
        ]
        
        for res in self.resolutions:
            res_str = f"{res[0]}x{res[1]}"
            results['verification'][res_str] = {}
            results['identification'][res_str] = {}
            results['image_quality'][res_str] = {}
            
            print(f"\nProcessing Resolution: {res_str}")
            
            # HR Baseline (only run once per resolution loop, effectively)
            if res == (112, 112):
                print(f"  Evaluating HR Baseline...")
                pipeline = self._get_pipeline(res, use_dsr=False, use_finetuned=False)
                
                # Verification
                ver_metrics = self.evaluate_verification(subjects_by_res[res], pipeline)
                results['verification'][res_str]['HR_Default'] = ver_metrics
                
                # Identification
                id_metrics = self.evaluate_identification(subjects_by_res[res], pipeline)
                results['identification'][res_str]['HR_Default'] = id_metrics
                
                # Image Quality (Reference)
                results['image_quality'][res_str]['HR_Default'] = {'psnr': float('inf'), 'ssim': 1.0}
                continue

            # VLR Evaluations
            for name, use_dsr, use_finetuned in configs:
                print(f"  Evaluating {name}...")
                pipeline = self._get_pipeline(res, use_dsr, use_finetuned)
                
                # Verification
                ver_metrics = self.evaluate_verification(subjects_by_res[res], pipeline)
                results['verification'][res_str][name] = ver_metrics
                
                # Identification
                id_metrics = self.evaluate_identification(subjects_by_res[res], pipeline)
                results['identification'][res_str][name] = id_metrics
                
                # Image Quality (Only depends on DSR/Bicubic, not EdgeFace)
                # Avoid re-calculating if already done for this DSR/Bicubic mode?
                # For simplicity, just run it. It's fast on subset.
                iq_metrics = self.evaluate_image_quality(
                    subjects_by_res[res], 
                    pipeline, 
                    subjects_by_res[(112, 112)]
                )
                results['image_quality'][res_str][name] = iq_metrics
                
                print(f"    Acc: {ver_metrics['accuracy']:.2f}%, Rank-1: {id_metrics['rank1']:.2f}%, PSNR: {iq_metrics['psnr']:.2f}")

        return results

    def generate_report(self, results: Dict, output_path: Path):
        """Generate PDF report with expanded metrics and visualizations"""
        with PdfPages(output_path) as pdf:
            
            # --- Page 1: Verification Summary Table ---
            fig, ax = plt.subplots(figsize=(11.69, 8.27)) # A4 Landscape
            ax.axis('off')
            
            cols = ['Method', '16x16', '24x24', '32x32']
            rows = [
                'Bicubic + Default', 'Bicubic + Finetuned',
                'DSR + Default', 'DSR + Finetuned',
                'HR Baseline'
            ]
            
            cell_text = []
            for row_name in rows:
                row_data = [row_name]
                if 'HR' in row_name:
                    m = results['verification']['112x112']['HR_Default']
                    val = f"{m['accuracy']:.2f}%\n(VR@0.1%: {m['vr_far_01']:.2f}%)"
                    row_data.extend([val] * 3)
                else:
                    if 'Bicubic + Default' in row_name: key = 'Bicubic_Default'
                    elif 'Bicubic + Finetuned' in row_name: key = 'Bicubic_Finetuned'
                    elif 'DSR + Default' in row_name: key = 'DSR_Default'
                    elif 'DSR + Finetuned' in row_name: key = 'DSR_Finetuned'
                    
                    for res in ['16x16', '24x24', '32x32']:
                        if key in results['verification'][res]:
                            m = results['verification'][res][key]
                            val = f"{m['accuracy']:.2f}%\n(VR@0.1%: {m['vr_far_01']:.2f}%)"
                            row_data.append(val)
                        else:
                            row_data.append("-")
                cell_text.append(row_data)
            
            table = ax.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 4)
            ax.set_title('Verification Accuracy & VR@0.1% FAR', fontsize=16, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # --- Page 2: Identification & Image Quality Table ---
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis('off')
            
            # Combined table for Rank-1 and PSNR
            cell_text = []
            for row_name in rows:
                row_data = [row_name]
                if 'HR' in row_name:
                    m_id = results['identification']['112x112']['HR_Default']
                    val = f"R1: {m_id['rank1']:.2f}%\nPSNR: Inf"
                    row_data.extend([val] * 3)
                else:
                    if 'Bicubic + Default' in row_name: key = 'Bicubic_Default'
                    elif 'Bicubic + Finetuned' in row_name: key = 'Bicubic_Finetuned'
                    elif 'DSR + Default' in row_name: key = 'DSR_Default'
                    elif 'DSR + Finetuned' in row_name: key = 'DSR_Finetuned'
                    
                    for res in ['16x16', '24x24', '32x32']:
                        if key in results['identification'][res]:
                            m_id = results['identification'][res][key]
                            # IQ metrics might be missing for finetuned if we skipped re-calc
                            # But we ran it in loop, so it should be there or copied
                            if key in results['image_quality'][res]:
                                m_iq = results['image_quality'][res][key]
                                psnr_val = f"{m_iq['psnr']:.2f}"
                            else:
                                psnr_val = "N/A"
                                
                            val = f"R1: {m_id['rank1']:.2f}%\nPSNR: {psnr_val}"
                            row_data.append(val)
                        else:
                            row_data.append("-")
                cell_text.append(row_data)
            
            table = ax.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 4)
            ax.set_title('Identification (Rank-1) & Image Quality (PSNR)', fontsize=16, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # --- Page 3: ROC Curves (16x16) ---
            self._plot_roc_page(pdf, results, '16x16')
            
            # --- Page 4: DET Curves (16x16) ---
            self._plot_det_page(pdf, results, '16x16')
            
            # --- Page 5: Similarity Histograms (16x16) ---
            self._plot_hist_page(pdf, results, '16x16')
            
            # --- Page 6: CMC Curves (16x16) ---
            self._plot_cmc_page(pdf, results, '16x16')

        print(f"Report saved to {output_path}")

    def _plot_roc_page(self, pdf, results, resolution):
        """Helper to plot ROC curves for a specific resolution"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot HR Baseline
        if 'HR_Default' in results['verification']['112x112']:
            m = results['verification']['112x112']['HR_Default']
            if 'fpr' in m:
                ax.plot(m['fpr'], m['tpr'], label=f"HR Baseline (AUC={m['roc_auc']:.4f})", linestyle='--', color='black')
        
        # Plot VLR Methods
        for key, label, color in [
            ('Bicubic_Default', 'Bicubic + Default', 'blue'),
            ('Bicubic_Finetuned', 'Bicubic + Finetuned', 'cyan'),
            ('DSR_Default', 'DSR + Default', 'red'),
            ('DSR_Finetuned', 'DSR + Finetuned', 'magenta')
        ]:
            if key in results['verification'][resolution]:
                m = results['verification'][resolution][key]
                if 'fpr' in m:
                    ax.plot(m['fpr'], m['tpr'], label=f"{label} (AUC={m['roc_auc']:.4f})", color=color)
        
        ax.plot([0, 1], [0, 1], 'k:', alpha=0.2)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {resolution}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close()

    def _plot_det_page(self, pdf, results, resolution):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot HR Baseline
        if 'HR_Default' in results['verification']['112x112']:
            m = results['verification']['112x112']['HR_Default']
            if 'fpr' in m:
                # DET: log(FPR) vs log(FNR)
                fnr = 1 - m['tpr']
                ax.loglog(m['fpr'], fnr, label=f"HR Baseline", linestyle='--', color='black')
        
        for key, label, color in [
            ('Bicubic_Default', 'Bicubic + Default', 'blue'),
            ('Bicubic_Finetuned', 'Bicubic + Finetuned', 'cyan'),
            ('DSR_Default', 'DSR + Default', 'red'),
            ('DSR_Finetuned', 'DSR + Finetuned', 'magenta')
        ]:
            if key in results['verification'][resolution]:
                m = results['verification'][resolution][key]
                if 'fpr' in m:
                    fnr = 1 - m['tpr']
                    ax.loglog(m['fpr'], fnr, label=label, color=color)
        
        ax.set_xlabel('False Positive Rate (log scale)')
        ax.set_ylabel('False Negative Rate (log scale)')
        ax.set_title(f'DET Curves - {resolution}')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        pdf.savefig(fig)
        plt.close()

    def _plot_hist_page(self, pdf, results, resolution):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Only plot DSR_Default vs Bicubic_Default to avoid clutter, or maybe just DSR_Default
        # Let's plot DSR_Default if available, else Bicubic
        
        target_key = 'DSR_Default'
        if target_key not in results['verification'][resolution]:
            target_key = 'Bicubic_Default'
            
        if target_key in results['verification'][resolution]:
            m = results['verification'][resolution][target_key]
            if 'gen_sims' in m:
                ax.hist(m['gen_sims'], bins=50, alpha=0.5, label='Genuine', density=True, color='green')
                ax.hist(m['imp_sims'], bins=50, alpha=0.5, label='Impostor', density=True, color='red')
                ax.set_title(f'Similarity Distribution ({target_key}) - {resolution}')
                ax.set_xlabel('Cosine Similarity')
                ax.set_ylabel('Density')
                ax.legend()
        else:
            ax.text(0.5, 0.5, "No data available", ha='center')
            
        pdf.savefig(fig)
        plt.close()

    def _plot_cmc_page(self, pdf, results, resolution):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        max_rank = 20
        x = np.arange(1, max_rank + 1)
        
        # Helper to get cmc
        def get_cmc(ranks):
            cmc = []
            for r in x:
                cmc.append(np.mean(ranks <= r) * 100)
            return cmc

        # HR Baseline
        if 'HR_Default' in results['identification']['112x112']:
            m = results['identification']['112x112']['HR_Default']
            if 'ranks' in m:
                ax.plot(x, get_cmc(m['ranks']), label='HR Baseline', linestyle='--', color='black')

        for key, label, color in [
            ('Bicubic_Default', 'Bicubic + Default', 'blue'),
            ('Bicubic_Finetuned', 'Bicubic + Finetuned', 'cyan'),
            ('DSR_Default', 'DSR + Default', 'red'),
            ('DSR_Finetuned', 'DSR + Finetuned', 'magenta')
        ]:
            if key in results['identification'][resolution]:
                m = results['identification'][resolution][key]
                if 'ranks' in m:
                    ax.plot(x, get_cmc(m['ranks']), label=label, color=color)
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('Identification Rate (%)')
        ax.set_title(f'CMC Curves (Rank-1 to {max_rank}) - {resolution}')
        ax.set_xticks(x)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close()

def clean_results_for_json(results):
    """Recursively remove numpy arrays from results dict"""
    if isinstance(results, dict):
        return {k: clean_results_for_json(v) for k, v in results.items() if not isinstance(v, (np.ndarray, list)) or k not in ['fpr', 'tpr', 'gen_sims', 'imp_sims', 'ranks']}
    return results

def main():
    project_root = Path(__file__).resolve().parents[1]
    
    dataset_dir = project_root / "technical" / "dataset"
    dsr_dir = project_root / "technical" / "dsr"
    edgeface_dir = project_root / "technical" / "facial_rec" / "edgeface_weights"
    output_dir = project_root / "technical" / "evaluation_results"
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, help="Specific resolution to evaluate (e.g. 32)", default=None)
    args = parser.parse_args()

    evaluator = LFWEvaluator(
        dataset_dir=dataset_dir,
        dsr_dir=dsr_dir,
        edgeface_dir=edgeface_dir,
        output_dir=output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if args.resolution:
        # We need to load 112x112 subjects for Image Quality comparison
        # But we only want to run evaluation loop for the requested resolution
        evaluator.resolutions = [(args.resolution, args.resolution)]
        if args.resolution != 112:
             evaluator.resolutions.append((112, 112))
    
    results = evaluator.run_evaluation()
    
    # Generate PDF (uses raw data)
    evaluator.generate_report(results, output_dir / "benchmark_report.pdf")
    
    # Clean and Save JSON
    # We need to make a deep copy or just clean it on the fly, but since we are done, we can just clean it.
    # Actually, the simple cleaner above removes keys.
    # Better approach: Just pop the keys we added.
    
    def remove_raw_data(d):
        if isinstance(d, dict):
            for k in ['fpr', 'tpr', 'gen_sims', 'imp_sims', 'ranks']:
                if k in d:
                    del d[k]
            for v in d.values():
                remove_raw_data(v)
    
    import copy
    json_results = copy.deepcopy(results)
    remove_raw_data(json_results)
    
    with open(output_dir / "full_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)

if __name__ == "__main__":
    main()
