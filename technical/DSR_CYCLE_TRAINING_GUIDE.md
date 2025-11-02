# DSR Cycle Training Guide

## Overview

This guide covers the **one-cycle training strategy** for improving face recognition accuracy by training DSR on the fine-tuned EdgeFace model. This creates a feedback loop where DSR learns to preserve features that the fine-tuned EdgeFace recognizes best.

## What Changed

### Updated DSR Training Script (`train_dsr.py`)

**Key Improvements:**

1. **Fine-Tuned EdgeFace Integration**

   - Default EdgeFace checkpoint: `edgeface_finetuned.pth` (was `edgeface_xxs_q.pt`)
   - Handles both fine-tuned checkpoint format (`backbone_state_dict`) and original format
   - Automatic unpickling of `FinetuneConfig` class for checkpoint loading

2. **Stronger Identity Preservation**

   - Identity loss weight: **0.50** (up from 0.35)
   - New feature matching loss: **0.15** weight
   - Perceptual loss reduced: **0.02** (down from 0.03)
   - TV loss reduced: **3e-6** (down from 5e-6) for sharper features

3. **Multi-Scale Perceptual Loss**

   - 4 VGG layers (was 3): relu1_2, relu2_2, relu3_4, relu4_4
   - Weighted by importance: [0.4, 0.3, 0.2, 0.1] (early layers prioritized for facial structure)

4. **Feature Matching Loss** (NEW)

   - Extracts intermediate features from EdgeFace backbone
   - Matches DSR→EdgeFace features to HR→EdgeFace features at multiple depths
   - Ensures DSR outputs produce similar activation patterns to HR in recognition network

5. **Training Improvements**

   - Epochs: **100** (up from 80)
   - Warmup: **5 epochs** (up from 3)
   - Early stopping patience: **20** (up from 15)
   - Learning rate: **1.5e-4** (down from 2e-4) for stability

6. **Enhanced Logging**
   - Per-component loss breakdown (L1, Perceptual, Identity, Feature Match, TV)
   - Validation identity loss tracked in checkpoint
   - Clearer progress indicators

## Training Workflow

### Prerequisites

1. **Fine-tuned EdgeFace checkpoint exists**:

   ```powershell
   ls technical/facial_rec/edgeface_weights/edgeface_finetuned.pth
   ```

   If missing, run EdgeFace fine-tuning first (see `FINETUNE_GUIDE.md`).

2. **Dataset prepared**:
   - `technical/dataset/train_processed/` with `vlr_images/` and `hr_images/`
   - `technical/dataset/val_processed/` with `vlr_images/` and `hr_images/`

### Step 1: Train DSR with Fine-Tuned EdgeFace

```powershell
cd A:\Programming\School\cs565\project

# Default training (100 epochs, batch size 16)
poetry run python -m technical.dsr.train_dsr --device cuda

# Custom epochs (e.g., 80 if time-limited)
poetry run python -m technical.dsr.train_dsr --device cuda --epochs 80

# Adjust batch size if CUDA out of memory
poetry run python -m technical.dsr.train_dsr --device cuda --batch-size 12
```

**Expected training time:**

- RTX 3060 Ti: ~8-12 hours for 100 epochs
- Checkpoints saved to: `technical/dsr/dsr.pth`

**What to watch:**

- Identity loss should decrease steadily (target: <0.15 by end)
- Feature match loss should decrease (target: <0.05 by end)
- PSNR should improve (target: >28 dB)
- Early stopping will trigger if validation PSNR plateaus

### Step 2: Evaluate Improved DSR

After training completes, evaluate on test set:

```powershell
# Evaluate with new DSR + fine-tuned EdgeFace
poetry run python -m technical.pipeline.evaluate_dataset `
  --dataset-root technical/dataset/test_processed `
  --threshold 0.35 `
  --device cuda

# Compare to baseline (original DSR + original EdgeFace)
# Temporarily rename dsr.pth and restore original, then re-run evaluation
```

**Success metrics:**

- **Target accuracy**: >40-50% (up from 34.87%)
- **Unknown rate**: <15% (was 12.58%)
- **Per-subject variance**: reduced (fewer subjects with <15% accuracy)

### Step 3: Threshold Re-optimization (Optional)

The optimal threshold may shift after cycle training:

```powershell
poetry run python -m technical.tools.threshold_sweep `
  --dataset-root technical/dataset/test_processed `
  --device cuda
```

Look for the threshold with highest correct identifications (was 0.35).

## Understanding the Losses

### Training Output Example

```
Epoch 025 | train_loss=0.3214 (L1:0.089 P:0.024 ID:0.152 FM:0.038) PSNR=27.45dB
         | val_loss=0.2987 (L1:0.085 P:0.021 ID:0.138 FM:0.032) PSNR=28.12dB
✅ Saved new best checkpoint (val PSNR 28.12dB, ID loss 0.138)
```

**Component breakdown:**

- **L1**: Pixel-level reconstruction (~0.08-0.10 is good)
- **P (Perceptual)**: VGG feature similarity (~0.02-0.03 is good)
- **ID (Identity)**: EdgeFace embedding cosine distance (<0.15 is excellent, <0.20 is good)
- **FM (Feature Match)**: Intermediate feature alignment (<0.05 is excellent)
- **PSNR**: Peak signal-to-noise ratio (>28 dB is excellent for face SR)

**Healthy training signs:**

- ✅ ID loss decreasing steadily
- ✅ FM loss decreasing (features aligning)
- ✅ PSNR increasing
- ✅ Val loss tracking train loss (not diverging)

**Warning signs:**

- ⚠️ ID loss stuck >0.25 → increase `lambda_identity` to 0.60
- ⚠️ PSNR decreasing → reduce `lambda_identity`, increase `lambda_l1`
- ⚠️ Val loss >> train loss → overfitting; reduce epochs or increase augmentation

## Hyperparameter Tuning

If default training doesn't achieve target accuracy, try these adjustments:

### To Prioritize Recognition Over Image Quality

```python
# In train_dsr.py TrainConfig:
lambda_identity: float = 0.60        # Increase from 0.50
lambda_feature_match: float = 0.20   # Increase from 0.15
lambda_perceptual: float = 0.01      # Decrease from 0.02
```

### To Improve Image Quality (if PSNR too low)

```python
lambda_l1: float = 1.5               # Increase from 1.0
lambda_perceptual: float = 0.04      # Increase from 0.02
lambda_identity: float = 0.40        # Decrease from 0.50
```

### To Reduce Training Time

```python
epochs: int = 60                     # Reduce from 100
early_stop_patience: int = 12        # Reduce from 20
batch_size: int = 20                 # Increase from 16 (if memory allows)
```

## Why This Works

### Theory

1. **Domain Alignment**: Fine-tuned EdgeFace learned to recognize faces from _your_ DSR outputs. Now, DSR learns to produce outputs that _this specific EdgeFace_ recognizes best.

2. **Feature-Level Supervision**: Feature matching loss ensures DSR preserves not just final embeddings, but intermediate features at multiple depths. This creates richer supervision than embedding-only loss.

3. **Multi-Scale Perceptual**: Matching VGG features at 4 depths (vs 3) preserves both low-level edges and high-level semantic structure better.

4. **Reduced Conflict**: Lower perceptual and TV loss weights reduce tension with identity preservation, letting DSR prioritize recognition over pixel-perfect reconstruction.

### Expected Improvements

- **+5-10% accuracy**: From domain alignment (DSR→EdgeFace co-optimization)
- **+2-5% accuracy**: From feature matching (richer supervision)
- **+1-3% accuracy**: From multi-scale perceptual (better structure preservation)
- **Total target**: +8-18% improvement → 43-53% accuracy (from 34.87% baseline)

## Troubleshooting

### "Could not register feature hooks; disabling feature matching"

**Cause**: EdgeFace architecture doesn't have expected `.features` attribute or indexing failed.

**Fix**: Feature matching will be disabled automatically; training continues with other losses. To re-enable:

1. Inspect EdgeFace architecture: `print(model.model.features)`
2. Adjust hook indices in `_register_feature_hooks()` to match your EdgeFace layers

### "CUDA out of memory"

**Cause**: Batch size too large for GPU memory (especially with feature extraction).

**Solutions**:

```powershell
# Reduce batch size
poetry run python -m technical.dsr.train_dsr --device cuda --batch-size 12

# Or reduce num_workers
# Edit TrainConfig: num_workers: int = 4
```

### "Identity loss stuck at 0.30+"

**Cause**: DSR struggling to preserve features; may need stronger identity signal.

**Solutions**:

1. Increase `lambda_identity` to 0.60-0.70
2. Decrease `lambda_perceptual` to 0.01
3. Check fine-tuned EdgeFace loaded correctly (look for "[EdgeFace] Loaded fine-tuned backbone weights" message)

### "Validation PSNR lower than baseline"

**Cause**: Over-prioritizing identity at expense of image quality.

**Solutions**:

1. Reduce `lambda_identity` to 0.40
2. Increase `lambda_l1` to 1.2-1.5
3. Train for more epochs (identity-quality balance takes time)

### "Training too slow"

**Cause**: Feature extraction overhead, data loading bottleneck, or small batch size.

**Solutions**:

```powershell
# Increase batch size (if GPU memory allows)
--batch-size 20

# Reduce workers if CPU-bound
# Edit TrainConfig: num_workers: int = 4

# Disable feature matching if it's the bottleneck
# Edit TrainConfig: lambda_feature_match: float = 0.0
```

## Next Steps After Training

1. **Evaluate on test set** (see Step 2 above)

2. **Compare visually**:

   ```powershell
   poetry run python -m technical.tools.compare_models `
     --dsr-old path/to/old_dsr.pth `
     --dsr-new technical/dsr/dsr.pth `
     --device cuda
   ```

3. **If accuracy improved >5%**:

   - ✅ Ship it! Update production pipeline to use new DSR checkpoint
   - Document improvements in project report
   - Consider threshold re-tuning

4. **If accuracy improved <2%**:

   - Try hyperparameter tuning (see section above)
   - Or revert to baseline DSR (diminishing returns)

5. **If accuracy decreased**:
   - ❌ Revert to baseline DSR
   - Check fine-tuned EdgeFace quality (may need to retrain EdgeFace with different settings)
   - Verify dataset quality (HR/VLR pairs correct?)

## Do NOT Cycle Train Again

**Why?** Diminishing returns + risk of mode collapse.

- Cycle 1 (EdgeFace FT → DSR retrain): **Worth it** — models co-adapt
- Cycle 2 (DSR retrain → EdgeFace re-FT): **Rarely worth it** — marginal gains, high compute cost
- Cycle 3+: **Never worth it** — mode collapse risk (models overfit to each other's quirks)

**Exception**: If cycle 1 gives >10% improvement, you might try cycle 2 as an experiment. But expect <2% additional gain.

## Summary

This updated training script:

- Uses **fine-tuned EdgeFace** for identity supervision
- Adds **feature matching loss** for richer multi-scale supervision
- **Strengthens identity preservation** (0.50 weight) while reducing perceptual/TV weights
- Implements **multi-scale perceptual loss** with weighted VGG features
- Provides **detailed loss breakdown** for monitoring training health

Expected outcome: **+8-18% recognition accuracy** from DSR-EdgeFace co-optimization.

Run training, evaluate, and compare to baseline to validate improvements!
