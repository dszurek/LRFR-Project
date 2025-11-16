# Evaluation Guide

This document explains how to properly evaluate the DSR + EdgeFace pipeline for face recognition tasks.

## Overview

The system supports two main evaluation scenarios:

1. **1:1 Verification**: Given two face images, determine if they are the same person
2. **1:N Identification**: Given a face image, identify which person it is from a gallery of N people

## ✅ Status Check

### 1. DSR Training (Multi-Resolution) ✅

The DSR training script **correctly supports** training 3 separate models:

```bash
# Train DSR for 16×16 VLR input
python -m technical.dsr.train_dsr --vlr-size 16 --device cuda

# Train DSR for 24×24 VLR input
python -m technical.dsr.train_dsr --vlr-size 24 --device cuda

# Train DSR for 32×32 VLR input (default)
python -m technical.dsr.train_dsr --vlr-size 32 --device cuda
```

Each creates a separate checkpoint: `dsr16.pth`, `dsr24.pth`, `dsr32.pth`

**Key features:**

- Resolution-aware hyperparameter tuning (base channels, residual blocks, learning rates)
- Automatic VLR directory resolution (`vlr_images`, `vlr_images_16x16`, `vlr_images_24x24`)
- Identity preservation using frozen EdgeFace embeddings
- Perceptual loss, total variation, and feature matching

### 2. EdgeFace Fine-Tuning ⚠️ UPDATED

**Previous issues:**

- Dataset had ~518 classes (different people in train/val/test)
- No explicit 1:1 or 1:N verification metrics
- Designed for classification, not verification

**Fixes applied:**

- Added `--use-small-gallery` flag to use `frontal_only/` dataset (better for 1:1 and small 1:N scenarios)
- Dataset selection guidance in code comments
- Contrastive loss (Stage 2) aligns DSR-HR embeddings for better verification

**Recommended usage:**

```bash
# For small gallery (1:1 and 1:N with N≤10)
python -m technical.facial_rec.finetune_edgeface \
    --use-small-gallery \
    --device cuda

# For large gallery (hundreds of people)
python -m technical.facial_rec.finetune_edgeface \
    --train-dir technical/dataset/train_processed \
    --val-dir technical/dataset/val_processed \
    --device cuda
```

**Note:** The fine-tuning script optimizes for:

- **Stage 1**: ArcFace classification loss (learns discriminative features)
- **Stage 2**: Classification + Contrastive loss (aligns DSR-HR embeddings)
- **Primary metric**: Embedding similarity (measures DSR→HR alignment)

For proper **1:1 verification metrics** (FAR, FRR, EER), use the evaluation script below.

### 3. Evaluation Scripts ✅ NEW

Created comprehensive evaluation script: `technical/pipeline/evaluate_verification.py`

**Features:**

- ✅ 1:1 Verification: FAR, FRR, EER, ROC curves, AUC
- ✅ 1:N Identification: Rank-1/5/10 accuracy, closed-set and open-set
- ✅ Proper genuine/impostor pair generation
- ✅ Threshold analysis

## Evaluation Workflows

### 1:1 Verification Evaluation

Evaluates face verification performance (same person or different person).

```bash
python -m technical.pipeline.evaluate_verification \
    --mode verification \
    --test-root technical/dataset/frontal_only/test \
    --device cuda
```

**Metrics computed:**

- **ROC AUC**: Area under ROC curve (higher = better)
- **EER**: Equal Error Rate (lower = better)
- **TAR**: True Accept Rate at various thresholds
- **FAR**: False Accept Rate at various thresholds
- **FRR**: False Reject Rate at various thresholds

**Output example:**

```
ROC AUC:               0.9850
Equal Error Rate (EER): 0.0250 (threshold: 0.6234)

Threshold    TAR      FRR      FAR      TRR
--------------------------------------------------
0.5000       0.9800   0.0200   0.0500   0.9500
0.6000       0.9500   0.0500   0.0100   0.9900
0.7000       0.9000   0.1000   0.0010   0.9990
```

### 1:N Identification Evaluation

Evaluates face identification performance (who is this person from N candidates).

```bash
python -m technical.pipeline.evaluate_verification \
    --mode identification \
    --gallery-root technical/dataset/frontal_only/train \
    --test-root technical/dataset/frontal_only/test \
    --max-gallery-size 10 \
    --device cuda
```

**Metrics computed:**

- **Rank-1 accuracy**: Top-1 prediction is correct
- **Rank-5 accuracy**: True identity in top-5 predictions
- **Rank-10 accuracy**: True identity in top-10 predictions
- **Rejection rate**: Correctly reject unknown people (open-set)

**Output example:**

```
Gallery size:           10 subjects
Total probes:           500
  In gallery:           450
  Not in gallery:       50

Closed-Set Accuracy (probes in gallery):
  Rank-1:  0.9200 (414/450)
  Rank-5:  0.9800 (441/450)
  Rank-10: 0.9900 (445/450)

Open-Set Performance (probes NOT in gallery):
  Rejection rate: 0.8400 (42/50)
```

### Combined Evaluation

```bash
python -m technical.pipeline.evaluate_verification \
    --mode both \
    --gallery-root technical/dataset/frontal_only/train \
    --test-root technical/dataset/frontal_only/test \
    --device cuda \
    --output results.json
```

### Existing Evaluation Scripts

#### Image Quality Metrics

For PSNR, SSIM, and identity similarity:

```bash
python -m technical.dsr.evaluate_frontal \
    --dsr-checkpoint technical/dsr/dsr.pth \
    --test-dir technical/dataset/frontal_only/test \
    --device cuda
```

#### Pipeline End-to-End

For gallery-probe matching with accuracy:

```bash
python -m technical.pipeline.evaluate_dataset \
    --gallery-root technical/dataset/frontal_only/train \
    --probe-root technical/dataset/frontal_only/test \
    --device cuda
```

## Understanding the Metrics

### Verification Metrics (1:1)

| Metric      | Description                                 | Goal          |
| ----------- | ------------------------------------------- | ------------- |
| **ROC AUC** | Area under ROC curve                        | Close to 1.0  |
| **EER**     | Equal Error Rate (FPR = FNR)                | Low (< 0.05)  |
| **TAR**     | True Accept Rate (genuine pairs accepted)   | High (> 0.95) |
| **FAR**     | False Accept Rate (impostor pairs accepted) | Low (< 0.01)  |
| **FRR**     | False Reject Rate (genuine pairs rejected)  | Low (< 0.05)  |

### Identification Metrics (1:N)

| Metric             | Description                      | Goal          |
| ------------------ | -------------------------------- | ------------- |
| **Rank-1**         | Top prediction is correct        | High (> 0.90) |
| **Rank-5**         | True identity in top-5           | High (> 0.95) |
| **Rejection Rate** | Unknown faces correctly rejected | High (> 0.80) |

### DSR Quality Metrics

| Metric                  | Description                     | Goal    |
| ----------------------- | ------------------------------- | ------- |
| **PSNR**                | Peak Signal-to-Noise Ratio      | > 30 dB |
| **SSIM**                | Structural Similarity Index     | > 0.90  |
| **Identity Similarity** | Cosine similarity of embeddings | > 0.90  |

## Recommended Evaluation Protocol

### For 1:1 Matching (Door Access, Phone Unlock)

1. **Fine-tune EdgeFace** on small gallery:

   ```bash
   python -m technical.facial_rec.finetune_edgeface --use-small-gallery --device cuda
   ```

2. **Evaluate verification**:

   ```bash
   python -m technical.pipeline.evaluate_verification \
       --mode verification \
       --test-root technical/dataset/frontal_only/test \
       --edgeface-weights technical/facial_rec/edgeface_weights/edgeface_finetuned.pth \
       --device cuda
   ```

3. **Target metrics**:
   - EER < 0.05
   - FAR < 0.01 at TAR > 0.95

### For 1:N Matching (Office with 10 People)

1. **Fine-tune EdgeFace** on small gallery:

   ```bash
   python -m technical.facial_rec.finetune_edgeface --use-small-gallery --device cuda
   ```

2. **Evaluate identification**:

   ```bash
   python -m technical.pipeline.evaluate_verification \
       --mode identification \
       --gallery-root technical/dataset/frontal_only/train \
       --test-root technical/dataset/frontal_only/test \
       --max-gallery-size 10 \
       --edgeface-weights technical/facial_rec/edgeface_weights/edgeface_finetuned.pth \
       --device cuda
   ```

3. **Target metrics**:
   - Rank-1 accuracy > 0.90
   - Rank-5 accuracy > 0.95

## Dataset Recommendations

### For DSR Training

- Use **large dataset** with different people in train/val/test
- Current setup: `train_processed/`, `val_processed/`, `test_processed/`
- Focus: Image quality and identity preservation

### For EdgeFace Fine-Tuning (1:1 and Small 1:N)

- Use **small dataset** with same people in train/val (different images)
- Current setup: `frontal_only/` with `--use-small-gallery` flag
- Focus: Embedding similarity and discrimination

### For EdgeFace Fine-Tuning (Large 1:N)

- Use **large dataset** with many classes
- Current setup: `train_processed/`, `val_processed/`
- Focus: Classification accuracy and generalization

## Summary

✅ **DSR Training**: Fully supports 16×16, 24×24, 32×32 multi-resolution models

⚠️ **EdgeFace Fine-Tuning**: Now supports small gallery mode via `--use-small-gallery` flag. For proper verification metrics, use the evaluation script.

✅ **Evaluation**: New comprehensive script provides proper 1:1 verification (FAR/FRR/EER) and 1:N identification (Rank-1/5/10) metrics.

## Next Steps

1. ✅ Train DSR models for all resolutions (16, 24, 32)
2. ✅ Fine-tune EdgeFace using `--use-small-gallery` for 1:1/small 1:N scenarios
3. ✅ Run comprehensive evaluation using `evaluate_verification.py`
4. 📊 Analyze results and adjust thresholds for target FAR/FRR trade-offs
5. 🚀 Deploy best-performing configuration
