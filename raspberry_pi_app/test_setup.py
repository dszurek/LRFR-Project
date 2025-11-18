#!/usr/bin/env python3
"""Quick test script for verifying Raspberry Pi app setup.

Checks:
- Python dependencies
- Model files (Git LFS)
- Camera access
- Basic pipeline functionality
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("=" * 60)
    print("Checking Dependencies...")
    print("=" * 60)
    
    required = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "cv2": "OpenCV",
        "PIL": "Pillow",
        "numpy": "NumPy",
        "tkinter": "Tkinter (GUI)",
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT FOUND")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed")
    return True


def check_models():
    """Check if model files are available."""
    print("\n" + "=" * 60)
    print("Checking Model Files...")
    print("=" * 60)
    
    import config
    
    models = {
        "DSR 16×16": config.DSR_MODEL_16,
        "DSR 24×24": config.DSR_MODEL_24,
        "DSR 32×32": config.DSR_MODEL_32,
        "EdgeFace 16×16": config.EDGEFACE_MODEL_16,
        "EdgeFace 24×24": config.EDGEFACE_MODEL_24,
        "EdgeFace 32×32": config.EDGEFACE_MODEL_32,
    }
    
    missing = []
    
    for name, path in models.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb < 0.001:  # Less than 1KB (probably LFS pointer)
                print(f"⚠ {name}: {size_mb:.2f} MB - LFS NOT DOWNLOADED")
                missing.append(name)
            else:
                print(f"✓ {name}: {size_mb:.2f} MB")
        else:
            print(f"✗ {name}: NOT FOUND")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing or incomplete models: {', '.join(missing)}")
        print("Download with: git lfs pull")
        return False
    
    print("\n✅ All models present")
    return True


def check_camera():
    """Check if webcam is accessible."""
    print("\n" + "=" * 60)
    print("Checking Camera Access...")
    print("=" * 60)
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Camera not accessible (device 0)")
            print("\nTroubleshooting:")
            print("  - Check camera is connected")
            print("  - Run: v4l2-ctl --list-devices")
            print("  - Check permissions: sudo usermod -a -G video $USER")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print("✗ Camera opened but failed to read frame")
            return False
        
        h, w = frame.shape[:2]
        print(f"✓ Camera accessible: {w}×{h}")
        print("\n✅ Camera working")
        return True
        
    except Exception as e:
        print(f"✗ Camera check failed: {e}")
        return False


def check_pipeline():
    """Test basic pipeline functionality."""
    print("\n" + "=" * 60)
    print("Testing Pipeline...")
    print("=" * 60)
    
    try:
        from pipeline import LRFRPipeline
        import numpy as np
        
        print("Loading pipeline (VLR: 32×32)...")
        pipeline = LRFRPipeline(vlr_size=32)
        
        print("Creating test image...")
        test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        print("Running inference...")
        result = pipeline.process_image(test_img, return_intermediate=True)
        
        print(f"\nResults:")
        print(f"  Embedding shape: {result['embedding'].shape}")
        print(f"  VLR image shape: {result['vlr_image'].shape}")
        print(f"  SR image shape: {result['sr_image'].shape}")
        print(f"\nTiming:")
        for stage, time_ms in result['timings'].items():
            print(f"  {stage}: {time_ms:.1f} ms")
        print(f"  TOTAL: {result['total_time']:.1f} ms")
        
        print("\n✅ Pipeline working")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("Raspberry Pi 5 LRFR Application - Setup Verification")
    print("=" * 60 + "\n")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Models", check_models),
        ("Camera", check_camera),
        ("Pipeline", check_pipeline),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to run app.py")
        print("\nStart the application with:")
        print("  python app.py")
        return 0
    else:
        print("\n❌ Some checks failed. Fix issues above before running app.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
