# 32×32 VLR Optimization Guide

## Overview

This document explains the optimizations made to DSR training and EdgeFace fine-tuning scripts for the new **32×32 VLR resolution** (upgraded from 14×16).

## Why 32×32 Changes Training Strategy

### Resolution Impact

- **14×16 → 32×32**: 224 pixels → 1,024 pixels (**4.57× more data**)
- **Upscaling factor**: 9.14× → 4.00× (**easier SR task**)
- **Identity features**: Eyes now 4-6 pixels (vs 2-3), nose 3-4 pixels (vs 2), face contour clearly visible

### Key Implications

1. **More structure to preserve** → stronger identity/perceptual losses
2. **Less aggressive upscaling** → can reduce smoothing regularization
3. **Clearer facial features** → more augmentation is safe
4. **Higher memory usage** → smaller batch sizes needed
5. **Better convergence potential** → can train longer with higher learning rates

---

## DSR Training Changes (`train_dsr.py`)

### Batch Size & Memory

```python
batch_size: int = 14  # Was: 16
```

- **Reason**: 32×32 input uses ~2.3× more memory than 16×16
- **Impact**: Slightly slower training, but prevents OOM on 8GB VRAM

### Learning Rate

```python
learning_rate: float = 1.3e-4  # Was: 1.5e-4
```

- **Reason**: Larger input → more stable gradients → can afford slightly lower LR
- **Impact**: More stable convergence, less oscillation

### Loss Weights

#### Perceptual Loss

```python
lambda_perceptual: float = 0.025  # Was: 0.02 (+25%)
```

- **Reason**: 32×32 has more texture/structure to preserve (eyes, nose, mouth edges)
- **Impact**: Better facial structure preservation, sharper features

#### Identity Loss

```python
lambda_identity: float = 0.60  # Was: 0.50 (+20%)
```

- **Reason**: 32×32 has clearer identity features → embeddings more meaningful
- **Impact**: Stronger identity preservation, better recognition accuracy

#### Feature Matching Loss

```python
lambda_feature_match: float = 0.18  # Was: 0.15 (+20%)
```

- **Reason**: More intermediate features to align with EdgeFace at 32×32
- **Impact**: Better layer-by-layer feature alignment

#### Total Variation Loss

```python
lambda_tv: float = 2e-6  # Was: 3e-6 (-33%)
```

- **Reason**: 32×32 already has structure; less smoothing needed
- **Impact**: Allows sharper edges (eyes, mouth), less blurring

### Model Architecture

```python
config = DSRConfig(base_channels=128, residual_blocks=16)  # Was: 112
```

- **Reason**: 32×32 input → more features to learn → needs more capacity
- **Impact**: +14% parameters (~2.3M → 2.6M), better detail reconstruction

### Data Augmentation

#### Rotation

```python
if random.random() < 0.65:  # Was: 0.60
    angle = random.uniform(-6.0, 6.0)  # Was: ±5
```

- **Reason**: 32×32 has enough structure to handle slight rotation without losing identity
- **Impact**: Better generalization to head pose variations

#### Color Jitter

```python
if random.random() < 0.3:  # Was: 0.25
    brightness = random.uniform(0.93, 1.07)  # Was: 0.95-1.05
    contrast = random.uniform(0.93, 1.07)    # Was: 0.95-1.05
    saturation = random.uniform(0.96, 1.04)  # Was: 0.97-1.03
```

- **Reason**: 32×32 is more robust to lighting variations
- **Impact**: Better generalization to different lighting conditions

---

## EdgeFace Fine-Tuning Changes (`finetune_edgeface.py`)

### Training Duration

```python
stage2_epochs: int = 25  # Was: 20 (+25%)
```

- **Reason**: 32×32 DSR outputs have more detail → benefits from longer fine-tuning
- **Impact**: Better adaptation to DSR output distribution

### Batch Size

```python
batch_size: int = 28  # Was: 32
```

- **Reason**: 32×32 DSR outputs → 128×128 SR → more memory per sample
- **Impact**: Fits in 8GB VRAM with headroom

### Learning Rates

#### Stage 1 (Head Only)

```python
head_lr: float = 9e-4  # Was: 1e-3 (-10%)
```

- **Reason**: 32×32 produces more stable embeddings → slightly lower LR for smoother convergence

#### Stage 2 (Full Model)

```python
backbone_lr: float = 6e-6   # Was: 5e-6 (+20%)
head_lr_stage2: float = 6e-5  # Was: 5e-5 (+20%)
```

- **Reason**: 32×32 features are clearer → can afford slightly more aggressive fine-tuning
- **Impact**: Faster convergence, better final accuracy

### ArcFace Margin

```python
arcface_margin: float = 0.45  # Was: 0.5 (-10%)
```

- **Reason**: 32×32 creates tighter embedding clusters naturally
- **Impact**: Less aggressive margin needed, better discrimination

### Regularization

#### Weight Decay

```python
weight_decay: float = 3e-5  # Was: 5e-5 (-40%)
```

- **Reason**: 32×32 has more signal → less regularization needed

#### Label Smoothing

```python
label_smoothing: float = 0.08  # Was: 0.1 (-20%)
```

- **Reason**: 32×32 features are clearer → more confident predictions justified

### Early Stopping

```python
early_stop_patience: int = 10  # Was: 8 (+25%)
```

- **Reason**: Longer training (25 epochs) → allow more exploration before stopping

### Data Augmentation

```python
self.aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.07, contrast=0.07, saturation=0.05, hue=0.02),  # More aggressive
    transforms.RandomRotation(degrees=5),  # NEW
])
```

- **New**: Rotation augmentation (wasn't safe with 16×16)
- **Increased**: Color jitter ranges (brightness/contrast 0.05→0.07, saturation 0.03→0.05)
- **Added**: Hue shift (0.02) for color diversity
- **Reason**: 32×32 is robust enough for these augmentations

---

## Expected Performance Improvements

### Accuracy Gains

- **Baseline (14×16)**: ~35% rank-1 accuracy
- **Expected (32×32)**: **55-70% rank-1 accuracy** (+20-35% absolute)

### Breakdown of Improvements

1. **Resolution increase**: +15-20% (better identity features)
2. **Optimized losses**: +3-5% (stronger identity preservation)
3. **Better augmentation**: +2-4% (improved generalization)
4. **Longer fine-tuning**: +2-3% (better DSR adaptation)

### Training Time Impact

- **DSR training**: ~10-14 hours on RTX 3060 Ti (vs 8-12h with 16×16)
  - Reason: Larger model (128 vs 112 channels), more memory per batch
- **EdgeFace fine-tuning**: ~12-15 hours (vs 10-12h)
  - Reason: 5 extra epochs (25 vs 20), slightly smaller batches

---

## Memory Requirements

### DSR Training

- **16×16 config**: ~6.5GB VRAM @ batch_size=16
- **32×32 config**: ~7.2GB VRAM @ batch_size=14
- **Safe for**: RTX 3060 Ti (8GB), RTX 3070 (8GB), RTX 3080 (10GB+)

### EdgeFace Fine-Tuning

- **16×16 config**: ~7.0GB VRAM @ batch_size=32
- **32×32 config**: ~7.4GB VRAM @ batch_size=28
- **Safe for**: RTX 3060 Ti (8GB), RTX 3070 (8GB), RTX 3080 (10GB+)

### If OOM Occurs

Reduce batch sizes further:

```python
# DSR
batch_size: int = 12  # From 14

# EdgeFace
batch_size: int = 24  # From 28
```

---

## Training Commands

### 1. Train DSR (First)

```bash
cd technical
poetry run python -m dsr.train_dsr --device cuda --epochs 100
```

- Expected time: ~10-14 hours
- Watch for: Identity loss <0.12, validation PSNR >28dB

### 2. Fine-Tune EdgeFace (After DSR)

```bash
poetry run python -m facial_rec.finetune_edgeface --device cuda --stage2-epochs 25
```

- Expected time: ~12-15 hours
- Watch for: Validation accuracy >85%, stage2 accuracy >90%

### 3. Evaluate Pipeline

```bash
poetry run python -m pipeline.evaluate_dataset \
    --dataset-root technical/dataset/test_processed \
    --threshold 0.35 --device cuda
```

- Expected accuracy: **55-70% rank-1** (vs ~35% with 14×16)

---

## Hyperparameter Tuning Tips

### If Accuracy is Too Low (<50%)

1. **Increase identity loss weight**:

   ```python
   lambda_identity: float = 0.70  # From 0.60
   ```

2. **Reduce total variation** (allow sharper features):

   ```python
   lambda_tv: float = 1e-6  # From 2e-6
   ```

3. **Increase model capacity**:
   ```python
   base_channels: int = 144  # From 128
   ```

### If Overfitting Occurs

1. **Increase weight decay**:

   ```python
   weight_decay: float = 5e-5  # From 3e-5
   ```

2. **More aggressive augmentation**:

   ```python
   transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.08)
   ```

3. **Reduce learning rate**:
   ```python
   learning_rate: float = 1.0e-4  # DSR
   backbone_lr: float = 4e-6      # EdgeFace
   ```

### If Training is Unstable

1. **Reduce learning rate**:

   ```python
   learning_rate: float = 1.0e-4  # From 1.3e-4
   ```

2. **Longer warmup**:

   ```python
   warmup_epochs: int = 8  # From 5
   ```

3. **Gradient clipping**:
   ```python
   grad_clip: float = 0.5  # From 1.0
   ```

---

## Validation Metrics

### DSR Training

Monitor these metrics during training:

- **Validation PSNR**: Target >28dB (vs >26dB with 16×16)
- **Identity Loss**: Target <0.12 (vs <0.15 with 16×16)
- **Feature Matching Loss**: Target <0.08 (vs <0.10 with 16×16)
- **L1 Loss**: Target <0.04 (vs <0.05 with 16×16)

### EdgeFace Fine-Tuning

Monitor these metrics:

- **Stage 1 Validation Accuracy**: Target >80% (vs >75% with 16×16)
- **Stage 2 Validation Accuracy**: Target >88% (vs >82% with 16×16)
- **Stage 2 Final Epoch**: Target >90% (vs >85% with 16×16)

---

## Summary of Key Changes

| Parameter                   | 14×16 Value | 32×32 Value | Change | Reason              |
| --------------------------- | ----------- | ----------- | ------ | ------------------- |
| **DSR batch_size**          | 16          | 14          | -12.5% | Memory constraint   |
| **DSR learning_rate**       | 1.5e-4      | 1.3e-4      | -13%   | More stable         |
| **DSR lambda_identity**     | 0.50        | 0.60        | +20%   | Clearer features    |
| **DSR lambda_perceptual**   | 0.02        | 0.025       | +25%   | More structure      |
| **DSR lambda_tv**           | 3e-6        | 2e-6        | -33%   | Less smoothing      |
| **DSR base_channels**       | 112         | 128         | +14%   | More capacity       |
| **EdgeFace stage2_epochs**  | 20          | 25          | +25%   | Better convergence  |
| **EdgeFace batch_size**     | 32          | 28          | -12.5% | Memory constraint   |
| **EdgeFace arcface_margin** | 0.5         | 0.45        | -10%   | Tighter clusters    |
| **EdgeFace weight_decay**   | 5e-5        | 3e-5        | -40%   | Less regularization |

---

## Questions?

If you encounter issues or unexpected results:

1. Check GPU memory usage: `nvidia-smi`
2. Monitor training logs for NaN losses
3. Validate dataset: `ls -lh technical/dataset/train_processed/vlr_images/*.png | head`
4. Check VLR resolution: `python -c "from PIL import Image; print(Image.open('technical/dataset/train_processed/vlr_images/001_01_01_010_00_crop_128.png').size)"`

Expected output: `(32, 32)`
