# HR Resolution Mismatch Fix

## Problem Discovered

Windows Explorer showed HR images as **160×160**, not 128×128 as assumed. This was verified:

```python
from PIL import Image
img = Image.open('technical/dataset/train_processed/hr_images/001_01_01_010_00_crop_128.png')
print(img.size)  # Output: (160, 160)
```

## Impact on Training

### Before Fix (Broken):

- **VLR Input**: 32×32
- **DSR Output**: 128×128
- **HR Ground Truth**: 160×160 ❌
- **Problem**: Size mismatch causes:
  - L1 loss computed on mismatched tensors (PyTorch auto-resizes, degrading quality)
  - Perceptual loss extracts features from different spatial scales
  - Identity loss compares embeddings from different-sized inputs to EdgeFace
  - Effective upscaling factor unclear (4× vs 5×)

### After Fix (Correct):

- **VLR Input**: 32×32
- **DSR Output**: 128×128
- **HR Ground Truth**: 128×128 (resized from 160×160 with LANCZOS) ✅
- **Benefits**:
  - All losses computed on matching 128×128 tensors
  - Clean 4× upscaling factor (32→128)
  - Consistent with EdgeFace preprocessing (resizes to 112×112 from 128×128)
  - Better training stability and convergence

## Changes Made

### 1. Updated `TrainConfig`

```python
@dataclass
class TrainConfig:
    # NEW: Explicit target HR size
    target_hr_size: int = 128  # DSR outputs 128×128 (4× from 32×32 VLR)

    # ... rest of config
```

### 2. Updated `PairedFaceSRDataset`

```python
class PairedFaceSRDataset(Dataset):
    def __init__(self, root: Path, augment: bool, target_hr_size: int = 128):
        # ...
        self.target_hr_size = target_hr_size

    def __getitem__(self, index: int):
        vlr = Image.open(vlr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        # NEW: Resize HR from 160×160 to 128×128
        if hr.size != (self.target_hr_size, self.target_hr_size):
            hr = hr.resize(
                (self.target_hr_size, self.target_hr_size),
                Image.Resampling.LANCZOS  # High-quality downsampling
            )

        # ... rest of processing
```

### 3. Updated `train()` function

```python
print(f"Target HR resolution: {config.target_hr_size}×{config.target_hr_size} (resized from 160×160)")

train_ds = PairedFaceSRDataset(train_dir, augment=True, target_hr_size=config.target_hr_size)
val_ds = PairedFaceSRDataset(val_dir, augment=False, target_hr_size=config.target_hr_size)
```

## Why 128×128 Instead of 160×160?

### Option A: Resize HR to 128×128 (CHOSEN ✅)

**Pros:**

- Maintains 4× upscaling factor (32→128, standard in literature)
- DSR architecture already designed for 128×128 output
- EdgeFace expects ~112×112, so 128→112 is a single downsampling step
- Less memory usage (128² = 16,384 pixels vs 160² = 25,600 pixels)
- Faster training (~30% fewer pixels to process)

**Cons:**

- Discards ~37% of HR pixel information
- Slightly lower PSNR potential (theoretical ceiling is lower)

### Option B: Change DSR to output 160×160 ❌

**Pros:**

- Uses full HR resolution
- Higher PSNR ceiling

**Cons:**

- Upscaling factor becomes 5× (32→160, non-standard)
- Requires changing DSR architecture (pixel shuffle, etc.)
- More memory usage (~60% increase)
- EdgeFace still expects 112×112, so 160→112 is awkward downsampling
- Slower training and inference
- All previous checkpoints incompatible

## Verification

To verify the fix is working:

1. **Check dataset loading:**

```bash
poetry run python -c "
from pathlib import Path
from technical.dsr.train_dsr import PairedFaceSRDataset
ds = PairedFaceSRDataset(Path('technical/dataset/train_processed'), False, 128)
vlr, hr = ds[0]
print(f'VLR shape: {vlr.shape}')  # Should be [3, 32, 32]
print(f'HR shape: {hr.shape}')    # Should be [3, 128, 128]
"
```

2. **Check training log:**
   Look for: `Target HR resolution: 128×128 (resized from 160×160)`

3. **Monitor loss values:**
   Losses should be more stable now that tensor sizes match properly.

## Expected Performance Impact

### PSNR Changes:

- **Before (160×160 HR)**: ~35-37 dB (but misleading due to size mismatch)
- **After (128×128 HR)**: ~28-32 dB (accurate measurement, 4× upscaling)

The apparent "drop" is actually a **correction** — we're now measuring the right thing (32×32→128×128 4× upscaling) instead of an ill-defined mixed-scale comparison.

### Recognition Accuracy:

Should **improve** because:

- Identity loss now computed on matching-sized tensors
- EdgeFace receives consistently-preprocessed images
- No more artifacts from auto-resizing during loss calculation

## Alternative: Use Full 160×160 Resolution

If you want to use the full 160×160 resolution instead, change:

```python
@dataclass
class TrainConfig:
    target_hr_size: int = 160  # Changed from 128

    # Adjust other hyperparameters for 5× upscaling:
    batch_size: int = 10  # Reduced from 14 (more memory needed)
    lambda_perceptual: float = 0.03  # Slightly higher (more structure)
    lambda_tv: float = 1.5e-6  # Lower (allow sharper details)
```

And update `DSRConfig`:

```python
config = DSRConfig(
    base_channels=128,  # Keep high capacity
    residual_blocks=18,  # +2 blocks for harder 5× task
    scale=5  # Changed from 4
)
```

**However**, this requires retraining from scratch and is NOT recommended unless you have strong evidence that 160×160 output significantly improves recognition accuracy.

## Summary

✅ **Fixed**: HR images now properly resized to 128×128 to match DSR output  
✅ **Benefit**: Consistent 4× upscaling (32→128) across entire pipeline  
✅ **Impact**: More accurate loss calculation, better training stability  
✅ **Backward compatible**: No changes to DSR architecture or checkpoints needed

The training script will now work correctly with the 160×160 HR images by automatically resizing them to 128×128 during loading.
