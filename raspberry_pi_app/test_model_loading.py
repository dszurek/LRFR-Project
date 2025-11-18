#!/usr/bin/env python3
"""Test script to verify DSR model loading works correctly.

Usage:
    python test_model_loading.py /path/to/dsr_model.pth

This will test loading the model and processing a dummy image to ensure the output is valid.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pipeline import LRFRPipeline


def test_model(model_path: str, vlr_size: int = 32):
    """Test loading and running a DSR model."""
    
    print(f"\n{'='*60}")
    print(f"Testing DSR model: {Path(model_path).name}")
    print(f"VLR size: {vlr_size}×{vlr_size}")
    print(f"{'='*60}\n")
    
    # Create pipeline with custom model
    try:
        pipeline = LRFRPipeline(
            vlr_size=vlr_size,
            dsr_model_path=model_path,
            edgeface_model_path=None  # Use default EdgeFace
        )
        print("✓ Model loaded successfully!\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}\n")
        return False
    
    # Create a test image (random noise)
    print("Creating test image...")
    test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # Process through pipeline
    try:
        print("Processing image through pipeline...")
        result = pipeline.process_image(test_image, return_intermediate=True)
        print("✓ Processing complete!\n")
    except Exception as e:
        print(f"✗ Failed to process image: {e}\n")
        return False
    
    # Check outputs
    print("Checking outputs:")
    
    # Check VLR image
    if "vlr_image" in result:
        vlr = result["vlr_image"]
        print(f"  VLR image: shape={vlr.shape}, dtype={vlr.dtype}, "
              f"min={vlr.min()}, max={vlr.max()}, mean={vlr.mean():.1f}")
        if vlr.shape != (vlr_size, vlr_size, 3):
            print(f"    ✗ WARNING: Expected shape ({vlr_size}, {vlr_size}, 3)")
    
    # Check SR image (this is the critical one)
    if "sr_image" in result:
        sr = result["sr_image"]
        print(f"  SR image:  shape={sr.shape}, dtype={sr.dtype}, "
              f"min={sr.min()}, max={sr.max()}, mean={sr.mean():.1f}")
        
        if sr.shape != (112, 112, 3):
            print(f"    ✗ WARNING: Expected shape (112, 112, 3)")
            return False
        
        # Check if image is all black (the bug we're fixing)
        if sr.max() == 0:
            print(f"    ✗ ERROR: SR image is all black!")
            return False
        elif sr.mean() < 1.0:
            print(f"    ⚠ WARNING: SR image is very dark (mean={sr.mean():.1f})")
        else:
            print(f"    ✓ SR image looks good!")
    else:
        print("  ✗ ERROR: No SR image in result!")
        return False
    
    # Check embedding
    if "embedding" in result:
        emb = result["embedding"]
        print(f"  Embedding: shape={emb.shape}, norm={np.linalg.norm(emb):.3f}")
        if abs(np.linalg.norm(emb) - 1.0) > 0.01:
            print(f"    ⚠ WARNING: Embedding should be L2-normalized (norm ≈ 1.0)")
    
    # Print timing
    print(f"\nTiming:")
    for stage, time_ms in result['timings'].items():
        print(f"  {stage}: {time_ms:.1f}ms")
    print(f"  TOTAL: {result['total_time']:.1f}ms")
    
    # Save output images for visual inspection
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    model_name = Path(model_path).stem
    cv2.imwrite(str(output_dir / f"{model_name}_vlr.png"), result["vlr_image"])
    cv2.imwrite(str(output_dir / f"{model_name}_sr.png"), result["sr_image"])
    print(f"\n✓ Saved output images to: {output_dir}")
    
    print(f"\n{'='*60}")
    print("✓ TEST PASSED!")
    print(f"{'='*60}\n")
    
    return True


def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        print("Usage: python test_model_loading.py <path_to_dsr_model.pth> [vlr_size]")
        print("\nExamples:")
        print("  # Test default 32x32 model")
        print("  python test_model_loading.py ../technical/dsr/dsr32.pth")
        print("\n  # Test 16x16 model")
        print("  python test_model_loading.py ../technical/dsr/dsr16.pth 16")
        print("\n  # Test with different resolution")
        print("  python test_model_loading.py ../technical/dsr/dsr16.pth 16")
        sys.exit(1)
    
    model_path = sys.argv[1]
    vlr_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    success = test_model(model_path, vlr_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
