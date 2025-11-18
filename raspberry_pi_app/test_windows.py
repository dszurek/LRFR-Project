"""Test script to verify Raspberry Pi app works on Windows."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("RASPBERRY PI APP - WINDOWS COMPATIBILITY TEST")
print("=" * 70)
print()

# Test 1: Config import and platform detection
print("Test 1: Config and Platform Detection")
print("-" * 70)
try:
    import raspberry_pi_app.config as config
    print(f"✅ Config imported successfully")
    print(f"   Platform: Windows={config.IS_WINDOWS}, Pi={config.IS_RASPBERRY_PI}")
    print(f"   Device: {config.DEVICE}")
    print(f"   PyTorch Threads: {config.TORCH_THREADS}")
    print()
except Exception as e:
    print(f"❌ Failed to import config: {e}")
    sys.exit(1)

# Test 2: Model path validation
print("Test 2: Model Path Validation")
print("-" * 70)
vlr_size = 32
dsr_path, edgeface_path = config.get_model_paths(vlr_size)
print(f"VLR Size: {vlr_size}×{vlr_size}")
print(f"DSR Model: {dsr_path}")
print(f"  Exists: {dsr_path.exists()}")
if dsr_path.exists():
    print(f"  Size: {dsr_path.stat().st_size / (1024*1024):.2f} MB")
print()
print(f"EdgeFace Model: {edgeface_path}")
print(f"  Exists: {edgeface_path.exists()}")
if edgeface_path.exists():
    print(f"  Size: {edgeface_path.stat().st_size / (1024*1024):.2f} MB")
print()

# Test 3: Pipeline import
print("Test 3: Pipeline Import")
print("-" * 70)
try:
    from raspberry_pi_app.pipeline import LRFRPipeline
    print("✅ Pipeline imported successfully")
    print()
except Exception as e:
    print(f"❌ Failed to import pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Pipeline initialization (if models exist)
if dsr_path.exists() and edgeface_path.exists():
    print("Test 4: Pipeline Initialization")
    print("-" * 70)
    try:
        pipeline = LRFRPipeline(vlr_size=vlr_size, device=config.DEVICE)
        print("✅ Pipeline initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 5: Process dummy image
    print("Test 5: Process Dummy Image")
    print("-" * 70)
    try:
        import numpy as np
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = pipeline.process_image(dummy_face)
        print("✅ Image processed successfully")
        print(f"   Embedding shape: {result['embedding'].shape}")
        print(f"   Total time: {result['total_time']:.2f}ms")
        print(f"   Timings:")
        for stage, time_ms in result['timings'].items():
            print(f"     {stage}: {time_ms:.2f}ms")
        print()
    except Exception as e:
        print(f"❌ Failed to process image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("Test 4-5: Skipped (models not found)")
    print("-" * 70)
    print("⚠️  Models not found. Pipeline cannot be initialized.")
    print("   Make sure model files exist in the correct locations.")
    print()

print("=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)
