"""LRFR Pipeline for Raspberry Pi 5.

Handles the complete low-resolution facial recognition pipeline:
1. Load quantized DSR and EdgeFace models
2. Downscale input to VLR size
3. Upscale with DSR to 112×112
4. Extract embedding with EdgeFace
5. Compare against gallery embeddings
"""

import sys
from pathlib import Path
import time
import gc
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from technical.dsr.models import DSRColor, DSRConfig
from technical.facial_rec.edgeface_weights.edgeface import EdgeFace
import config


class LRFRPipeline:
    """Complete LRFR pipeline with quantized models for Pi 5."""
    
    def __init__(
        self, 
        vlr_size: int = 32, 
        device: str = "cpu",
        dsr_model_path: Optional[str] = None,
        edgeface_model_path: Optional[str] = None
    ):
        """Initialize pipeline with specified VLR resolution.
        
        Args:
            vlr_size: VLR input size (16, 24, or 32)
            device: Device to run on ('cpu' for Pi 5)
            dsr_model_path: Optional custom path to DSR model (uses default if None)
            edgeface_model_path: Optional custom path to EdgeFace model (uses default if None)
        """
        self.vlr_size = vlr_size
        self.device = torch.device(device)
        self.hr_size = config.HR_SIZE
        
        # Store custom paths
        self.custom_dsr_path = Path(dsr_model_path) if dsr_model_path else None
        self.custom_edgeface_path = Path(edgeface_model_path) if edgeface_model_path else None
        
        # Set PyTorch threads for Pi 5
        torch.set_num_threads(config.TORCH_THREADS)
        
        # Memory optimization settings
        torch.set_grad_enabled(False)  # Disable gradient tracking globally
        
        print(f"[Pipeline] Initializing LRFR pipeline (VLR: {vlr_size}×{vlr_size})")
        
        # Load models
        self.dsr_model = self._load_dsr_model()
        self.edgeface_model = self._load_edgeface_model()
        
        # Warmup models (first inference is slow)
        self._warmup()
        
        # Clean up after initialization
        gc.collect()
        
        print(f"[Pipeline] Ready! (Device: {self.device})")
    
    def _load_dsr_model(self) -> DSRColor:
        """Load quantized DSR model."""
        # Use custom path if provided, otherwise use default
        if self.custom_dsr_path and self.custom_dsr_path.exists():
            dsr_path = self.custom_dsr_path
            print(f"[DSR] Using custom model: {dsr_path}")
        else:
            dsr_path, _ = config.get_model_paths(self.vlr_size)
            if not dsr_path.exists():
                raise FileNotFoundError(
                    f"DSR model not found: {dsr_path}\n"
                    f"Run 'git lfs pull' to download quantized models."
                )
        
        print(f"[DSR] Loading from {dsr_path.name}...")
        start = time.time()
        
        # Get config for this VLR size
        cfg = config.get_dsr_config(self.vlr_size)
        dsr_config = DSRConfig(
            base_channels=cfg["base_channels"],
            residual_blocks=cfg["residual_blocks"],
            output_size=cfg["output_size"]
        )
        
        # Load model
        model = DSRColor(config=dsr_config).to(self.device)
        state_dict = torch.load(dsr_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        elapsed = (time.time() - start) * 1000
        print(f"[DSR] Loaded in {elapsed:.0f}ms")
        
        return model
    
    def _load_edgeface_model(self) -> EdgeFace:
        """Load quantized EdgeFace model."""
        # Use custom path if provided, otherwise use default
        if self.custom_edgeface_path and self.custom_edgeface_path.exists():
            edgeface_path = self.custom_edgeface_path
            print(f"[EdgeFace] Using custom model: {edgeface_path}")
        else:
            _, edgeface_path = config.get_model_paths(self.vlr_size)
            if not edgeface_path.exists():
                raise FileNotFoundError(
                    f"EdgeFace model not found: {edgeface_path}\n"
                    f"Run 'git lfs pull' to download quantized models."
                )
        
        print(f"[EdgeFace] Loading from {edgeface_path.name}...")
        start = time.time()
        
        # Create model
        model = EdgeFace(embedding_size=512, back="edgeface_xxs")
        
        # Load checkpoint
        try:
            from technical.facial_rec.finetune_edgeface import FinetuneConfig
        except ImportError:
            # Stub for unpickling
            from dataclasses import dataclass
            @dataclass
            class FinetuneConfig:
                pass
            sys.modules["technical.facial_rec.finetune_edgeface"].FinetuneConfig = FinetuneConfig
        
        checkpoint = torch.load(edgeface_path, map_location=self.device, weights_only=False)
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("backbone_state_dict",
                                       checkpoint.get("model_state_dict",
                                                     checkpoint.get("state_dict", checkpoint)))
        else:
            state_dict = checkpoint
        
        # Strip prefixes
        cleaned_state = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            if new_key.startswith("backbone."):
                new_key = new_key[len("backbone."):]
            cleaned_state[new_key] = value
        
        model.load_state_dict(cleaned_state, strict=False)
        model.to(self.device)
        model.eval()
        
        elapsed = (time.time() - start) * 1000
        print(f"[EdgeFace] Loaded in {elapsed:.0f}ms")
        
        return model
    
    def _warmup(self):
        """Warmup models with dummy input (first inference is slow)."""
        print("[Pipeline] Warming up models...")
        dummy_vlr = torch.randn(1, 3, self.vlr_size, self.vlr_size).to(self.device)
        dummy_hr = torch.randn(1, 3, *self.hr_size).to(self.device)
        
        with torch.no_grad():
            _ = self.dsr_model(dummy_vlr)
            _ = self.edgeface_model(dummy_hr)
        
        # Clean up warmup tensors
        del dummy_vlr, dummy_hr
        gc.collect()
        
        print("[Pipeline] Warmup complete")
    
    def preprocess_to_vlr(self, image: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Downscale image to VLR size.
        
        Args:
            image: Input image (H, W, 3) in BGR format, range [0, 255]
        
        Returns:
            vlr_np: VLR image as numpy array (vlr_size, vlr_size, 3) BGR [0, 255]
            vlr_tensor: VLR as tensor (1, 3, vlr_size, vlr_size) RGB [0, 1]
        """
        # Resize to VLR
        vlr_np = cv2.resize(image, (self.vlr_size, self.vlr_size), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor: BGR -> RGB, HWC -> CHW, [0,255] -> [0,1]
        vlr_rgb = cv2.cvtColor(vlr_np, cv2.COLOR_BGR2RGB)
        vlr_tensor = torch.from_numpy(vlr_rgb.transpose(2, 0, 1)).float() / 255.0
        vlr_tensor = vlr_tensor.unsqueeze(0).to(self.device)
        
        return vlr_np, vlr_tensor
    
    def upscale_with_dsr(self, vlr_tensor: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        """Upscale VLR to HR using DSR.
        
        Args:
            vlr_tensor: VLR tensor (1, 3, vlr_size, vlr_size) range [0, 1]
        
        Returns:
            sr_np: Super-resolved image as numpy (112, 112, 3) BGR [0, 255]
            sr_tensor: SR as tensor (1, 3, 112, 112) RGB [0, 1]
        """
        with torch.no_grad():
            sr_tensor = self.dsr_model(vlr_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)
        
        # Convert to numpy for display: CHW -> HWC, RGB -> BGR, [0,1] -> [0,255]
        sr_rgb = (sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        sr_np = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
        
        return sr_np, sr_tensor
    
    def extract_embedding(self, hr_tensor: torch.Tensor) -> np.ndarray:
        """Extract L2-normalized 512-d embedding from HR image.
        
        Args:
            hr_tensor: HR tensor (1, 3, 112, 112) range [0, 1]
        
        Returns:
            embedding: L2-normalized embedding (512,) as numpy array
        """
        with torch.no_grad():
            embedding = self.edgeface_model(hr_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.squeeze(0).cpu().numpy()
    
    def process_image(
        self, 
        image: np.ndarray, 
        return_intermediate: bool = True
    ) -> Dict:
        """Run complete LRFR pipeline on input image.
        
        Args:
            image: Input face image (H, W, 3) BGR [0, 255]
            return_intermediate: Return intermediate results for visualization
        
        Returns:
            Dictionary with:
                - embedding: 512-d normalized embedding
                - vlr_image: VLR downscaled image (if return_intermediate)
                - sr_image: DSR upscaled image (if return_intermediate)
                - timings: Dict with per-stage timing in ms
                - total_time: Total processing time in ms
        """
        result = {}
        timings = {}
        
        start_total = time.time()
        
        # Step 1: Downscale to VLR
        start = time.time()
        vlr_np, vlr_tensor = self.preprocess_to_vlr(image)
        timings["downscale"] = (time.time() - start) * 1000
        
        # Step 2: Upscale with DSR
        start = time.time()
        sr_np, sr_tensor = self.upscale_with_dsr(vlr_tensor)
        timings["dsr"] = (time.time() - start) * 1000
        
        # Free VLR tensor - no longer needed
        del vlr_tensor
        
        # Step 3: Extract embedding
        start = time.time()
        embedding = self.extract_embedding(sr_tensor)
        timings["edgeface"] = (time.time() - start) * 1000
        
        # Free SR tensor - no longer needed
        del sr_tensor
        
        total_time = (time.time() - start_total) * 1000
        
        result["embedding"] = embedding
        result["timings"] = timings
        result["total_time"] = total_time
        
        if return_intermediate:
            result["vlr_image"] = vlr_np
            result["sr_image"] = sr_np
        
        return result
    
    def compare_embeddings(
        self, 
        query_embedding: np.ndarray, 
        gallery_embeddings: List[np.ndarray],
        gallery_names: List[str]
    ) -> List[Tuple[str, float]]:
        """Compare query embedding against gallery.
        
        Args:
            query_embedding: Query embedding (512,)
            gallery_embeddings: List of gallery embeddings
            gallery_names: List of corresponding names
        
        Returns:
            List of (name, similarity) tuples, sorted by similarity (descending)
        """
        similarities = []
        
        for name, emb in zip(gallery_names, gallery_embeddings):
            # Cosine similarity (both embeddings are L2-normalized)
            similarity = float(np.dot(query_embedding, emb))
            similarities.append((name, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def verify_1_to_1(
        self,
        query_embedding: np.ndarray,
        reference_embedding: np.ndarray,
        threshold: float = None
    ) -> Tuple[bool, float]:
        """1:1 verification against a specific person.
        
        Args:
            query_embedding: Query embedding (512,)
            reference_embedding: Reference embedding for specific person (512,)
            threshold: Similarity threshold (default from config)
        
        Returns:
            (is_match, similarity)
        """
        if threshold is None:
            threshold = config.VERIFICATION_THRESHOLD_1_1
        
        similarity = float(np.dot(query_embedding, reference_embedding))
        is_match = similarity >= threshold
        
        return is_match, similarity
    
    def identify_1_to_n(
        self,
        query_embedding: np.ndarray,
        gallery_embeddings: List[np.ndarray],
        gallery_names: List[str],
        threshold: float = None,
        top_k: int = None
    ) -> List[Tuple[str, float, bool]]:
        """1:N identification across entire gallery.
        
        Args:
            query_embedding: Query embedding (512,)
            gallery_embeddings: List of gallery embeddings
            gallery_names: List of corresponding names
            threshold: Minimum similarity for valid match (default from config)
            top_k: Number of top matches to return (default from config)
        
        Returns:
            List of (name, similarity, is_above_threshold) tuples for top-K
        """
        if threshold is None:
            threshold = config.IDENTIFICATION_THRESHOLD_1_N
        if top_k is None:
            top_k = config.TOP_K
        
        # Get all similarities
        similarities = self.compare_embeddings(query_embedding, gallery_embeddings, gallery_names)
        
        # Take top-K and mark which are above threshold
        results = []
        for name, sim in similarities[:top_k]:
            is_valid = sim >= threshold
            results.append((name, sim, is_valid))
        
        return results


if __name__ == "__main__":
    # Test pipeline
    print("Testing LRFR Pipeline...")
    
    # Initialize
    pipeline = LRFRPipeline(vlr_size=32)
    
    # Create dummy face image
    dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # Process
    result = pipeline.process_image(dummy_face)
    
    print(f"\nResults:")
    print(f"  Embedding shape: {result['embedding'].shape}")
    print(f"  VLR image shape: {result['vlr_image'].shape}")
    print(f"  SR image shape: {result['sr_image'].shape}")
    print(f"\nTimings:")
    for stage, time_ms in result['timings'].items():
        print(f"  {stage}: {time_ms:.2f}ms")
    print(f"  TOTAL: {result['total_time']:.2f}ms")
