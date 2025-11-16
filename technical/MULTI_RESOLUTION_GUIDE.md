# Multi-Resolution Evaluation Guide

## Overview

This guide describes the new multi-resolution capabilities for training, fine-tuning, and evaluating the DSR + EdgeFace pipeline across all three VLR resolutions (16×16, 24×24, 32×32).

## New Features

### 1. EdgeFace Fine-Tuning with Resolution Awareness

EdgeFace fine-tuning now supports resolution-specific training:

```bash
# Fine-tune for 16×16 VLR
python -m technical.facial_rec.finetune_edgeface --vlr-size 16 --device cuda

# Fine-tune for 24×24 VLR
python -m technical.facial_rec.finetune_edgeface --vlr-size 24 --device cuda

# Fine-tune for 32×32 VLR (default)
python -m technical.facial_rec.finetune_edgeface --vlr-size 32 --device cuda
```

**Key Features:**

- ✅ Automatically loads corresponding DSR model (`dsr16.pth`, `dsr24.pth`, `dsr32.pth`)
- ✅ Saves resolution-specific EdgeFace weights (`edgeface_finetuned_16.pth`, etc.)
- ✅ Resolution-aware hyperparameters (learning rates, epochs, augmentation)
- ✅ Automatic VLR directory selection (`vlr_images_16x16`, `vlr_images_24x24`, `vlr_images`)

**Configuration Differences by Resolution:**

| Parameter      | 16×16  | 24×24  | 32×32  |
| -------------- | ------ | ------ | ------ |
| Stage 1 Epochs | 12     | 11     | 10     |
| Stage 2 Epochs | 30     | 27     | 25     |
| Batch Size     | 24     | 28     | 32     |
| Head LR        | 1.2e-4 | 1.1e-4 | 1.0e-4 |
| Backbone LR    | 4e-6   | 3.5e-6 | 3e-6   |
| ArcFace Scale  | 10.0   | 9.0    | 8.0    |
| Temperature    | 3.5    | 3.7    | 4.0    |
| Rotation (deg) | 3      | 4      | 5      |

### 2. Comprehensive GUI Evaluation Tool

A new GUI application provides publication-quality evaluation across all resolutions.

**Launch the GUI:**

```bash
python -m technical.pipeline.evaluate_gui
```

**Features:**

- 📊 Multi-resolution comparison (16, 24, 32)
- 📈 Publication-quality plots (ROC curves, distributions, bar charts)
- 📄 Comprehensive PDF report generation
- 💾 Export results to JSON, PNG, PDF
- 🎨 Interactive configuration via GUI
- 📉 Statistical analysis and visualizations

**Command-Line Mode (if GUI unavailable):**

```bash
python -m technical.pipeline.evaluate_gui \
    --test-root technical/dataset/frontal_only/test \
    --gallery-root technical/dataset/frontal_only/train \
    --output-dir evaluation_results \
    --resolutions 16 24 32 \
    --device cuda
```

## Complete Workflow: All Resolutions

### Step 1: Generate Multi-Resolution Datasets

First, ensure you have VLR datasets for all three resolutions:

```bash
# Generate 16×16 VLR dataset
python -m technical.tools.regenerate_vlr_dataset \
    --vlr-size 16 \
    --dataset-dirs technical/dataset/train_processed \
                   technical/dataset/val_processed \
                   technical/dataset/test_processed

# Generate 24×24 VLR dataset
python -m technical.tools.regenerate_vlr_dataset \
    --vlr-size 24 \
    --dataset-dirs technical/dataset/train_processed \
                   technical/dataset/val_processed \
                   technical/dataset/test_processed

# 32×32 already exists in vlr_images/
```

### Step 2: Train DSR Models

Train separate DSR models for each resolution:

```bash
# Train DSR for 16×16
python -m technical.dsr.train_dsr \
    --vlr-size 16 \
    --device cuda \
    --epochs 120 \
    --batch-size 12

# Train DSR for 24×24
python -m technical.dsr.train_dsr \
    --vlr-size 24 \
    --device cuda \
    --epochs 110 \
    --batch-size 14

# Train DSR for 32×32
python -m technical.dsr.train_dsr \
    --vlr-size 32 \
    --device cuda \
    --epochs 100 \
    --batch-size 16
```

**Outputs:**

- `technical/dsr/dsr16.pth`
- `technical/dsr/dsr24.pth`
- `technical/dsr/dsr32.pth`

### Step 3: Fine-Tune EdgeFace Models

Fine-tune EdgeFace on DSR outputs for each resolution:

```bash
# Fine-tune for 16×16 (small gallery mode recommended)
python -m technical.facial_rec.finetune_edgeface \
    --vlr-size 16 \
    --use-small-gallery \
    --device cuda

# Fine-tune for 24×24
python -m technical.facial_rec.finetune_edgeface \
    --vlr-size 24 \
    --use-small-gallery \
    --device cuda

# Fine-tune for 32×32
python -m technical.facial_rec.finetune_edgeface \
    --vlr-size 32 \
    --use-small-gallery \
    --device cuda
```

**Outputs:**

- `technical/facial_rec/edgeface_weights/edgeface_finetuned_16.pth`
- `technical/facial_rec/edgeface_weights/edgeface_finetuned_24.pth`
- `technical/facial_rec/edgeface_weights/edgeface_finetuned_32.pth`

### Step 4: Comprehensive Evaluation

Run the GUI evaluation tool:

```bash
python -m technical.pipeline.evaluate_gui
```

**GUI Configuration:**

1. **Test Dataset:** `technical/dataset/frontal_only/test`
2. **Gallery Dataset:** `technical/dataset/frontal_only/train` (optional)
3. **Output Directory:** `evaluation_results`
4. **Resolutions:** Check all three (16×16, 24×24, 32×32)
5. **Device:** CUDA or CPU
6. Click **Run Evaluation**

**Outputs:**

```
evaluation_results/
├── evaluation_report.pdf          # Comprehensive PDF report
├── results.json                    # Detailed metrics in JSON
├── roc_curves.png                  # ROC curves comparison
├── quality_metrics.png             # PSNR/SSIM/Identity Sim
├── score_distributions.png         # Genuine vs Impostor
├── verification_comparison.png     # EER and TAR comparison
└── identification_accuracy.png     # Rank-1/5/10 comparison
```

## Generated Visualizations

### 1. Image Quality Metrics

- **PSNR (Peak Signal-to-Noise Ratio):** Measures pixel-level reconstruction quality
- **SSIM (Structural Similarity Index):** Measures perceptual similarity
- **Identity Similarity:** Cosine similarity between DSR and HR embeddings

**Interpretation:**

- PSNR > 30 dB: Good quality
- SSIM > 0.90: Good structural preservation
- Identity Sim > 0.90: Good identity preservation

### 2. ROC Curves

- Compares True Accept Rate (TAR) vs False Accept Rate (FAR)
- Shows verification performance across all resolutions
- Area Under Curve (AUC) indicates overall discriminability

**Interpretation:**

- AUC > 0.95: Excellent verification performance
- Steeper curve = better separation between genuine and impostor pairs

### 3. Score Distributions

- Histogram of genuine vs impostor similarity scores
- Shows separation between authentic and fraudulent attempts
- EER threshold marked on plot

**Interpretation:**

- Greater separation = better security
- Overlap indicates potential false accepts/rejects

### 4. Verification Metrics Comparison

- **EER (Equal Error Rate):** Point where FAR = FRR (lower is better)
- **TAR @ FAR:** True Accept Rate at fixed False Accept Rate thresholds

**Target Values:**

- EER < 0.05 (5%)
- TAR @ FAR=0.1% > 0.95 (95%)

### 5. Identification Accuracy

- **Rank-1:** Correct identity is top prediction
- **Rank-5:** Correct identity in top-5 predictions
- **Rank-10:** Correct identity in top-10 predictions

**Target Values:**

- Rank-1 > 90%
- Rank-5 > 95%
- Rank-10 > 97%

### 6. Summary Table

Comprehensive table with all metrics for easy comparison and inclusion in papers.

## Metrics for Research Papers

The evaluation tool generates publication-ready metrics:

### Table 1: Image Quality Comparison

| Resolution | PSNR (dB)    | SSIM            | Identity Similarity |
| ---------- | ------------ | --------------- | ------------------- |
| 16×16      | XX.XX ± X.XX | 0.XXXX ± 0.XXXX | 0.XXXX ± 0.XXXX     |
| 24×24      | XX.XX ± X.XX | 0.XXXX ± 0.XXXX | 0.XXXX ± 0.XXXX     |
| 32×32      | XX.XX ± X.XX | 0.XXXX ± 0.XXXX | 0.XXXX ± 0.XXXX     |

### Table 2: Verification Performance

| Resolution | EER    | ROC AUC | TAR @ FAR=0.1% | TAR @ FAR=1% |
| ---------- | ------ | ------- | -------------- | ------------ |
| 16×16      | 0.XXXX | 0.XXXX  | 0.XXXX         | 0.XXXX       |
| 24×24      | 0.XXXX | 0.XXXX  | 0.XXXX         | 0.XXXX       |
| 32×32      | 0.XXXX | 0.XXXX  | 0.XXXX         | 0.XXXX       |

### Table 3: Identification Accuracy

| Resolution | Rank-1 | Rank-5 | Rank-10 |
| ---------- | ------ | ------ | ------- |
| 16×16      | XX.X%  | XX.X%  | XX.X%   |
| 24×24      | XX.X%  | XX.X%  | XX.X%   |
| 32×32      | XX.X%  | XX.X%  | XX.X%   |

## Statistical Analysis

The tool automatically computes:

- **Mean and Standard Deviation** for all metrics
- **ROC Curves** with confidence intervals
- **Score Distributions** (Gaussian fit)
- **Comparative Bar Charts** with error bars

## Customization

### Custom Checkpoints

Specify custom model checkpoints:

```bash
python -m technical.facial_rec.finetune_edgeface \
    --vlr-size 16 \
    --dsr-weights path/to/custom_dsr16.pth \
    --edgeface-weights path/to/custom_edgeface.pt \
    --device cuda
```

### Custom Datasets

Use different datasets for training and evaluation:

```bash
# Fine-tune on custom dataset
python -m technical.facial_rec.finetune_edgeface \
    --vlr-size 24 \
    --train-dir custom/train \
    --val-dir custom/val \
    --device cuda

# Evaluate on custom dataset
python -m technical.pipeline.evaluate_gui \
    --test-root custom/test \
    --gallery-root custom/gallery \
    --output-dir custom_results
```

## Troubleshooting

### Issue: VLR directory not found

**Error:** `VLR directory not found: technical/dataset/train_processed/vlr_images_16x16`

**Solution:**

```bash
# Regenerate VLR dataset for the resolution
python -m technical.tools.regenerate_vlr_dataset \
    --vlr-size 16 \
    --dataset-dirs technical/dataset/train_processed
```

### Issue: DSR checkpoint not found

**Error:** `DSR checkpoint not found: technical/dsr/dsr16.pth`

**Solution:**

```bash
# Train DSR model first
python -m technical.dsr.train_dsr --vlr-size 16 --device cuda
```

### Issue: Out of memory during fine-tuning

**Solution:**

```bash
# Reduce batch size
python -m technical.facial_rec.finetune_edgeface \
    --vlr-size 16 \
    --device cuda
# (16×16 automatically uses batch_size=24, but you can override in code if needed)
```

### Issue: GUI not available

**Error:** `Warning: tkinter not available. GUI mode disabled.`

**Solution:**

```bash
# Install tkinter
sudo apt-get install python3-tk  # Ubuntu/Debian
# or
conda install tk  # Anaconda

# Or use CLI mode
python -m technical.pipeline.evaluate_gui \
    --test-root technical/dataset/frontal_only/test \
    --output-dir results \
    --resolutions 16 24 32
```

## Performance Expectations

### Training Time (approximate, GPU)

| Resolution | DSR Training | EdgeFace Fine-Tuning |
| ---------- | ------------ | -------------------- |
| 16×16      | ~3-4 hours   | ~2-3 hours           |
| 24×24      | ~2-3 hours   | ~1.5-2 hours         |
| 32×32      | ~1.5-2 hours | ~1-1.5 hours         |

### Evaluation Time (approximate, GPU)

| Resolution | Quality Metrics | Verification | Identification |
| ---------- | --------------- | ------------ | -------------- |
| 16×16      | ~5 min          | ~10 min      | ~5 min         |
| 24×24      | ~4 min          | ~8 min       | ~4 min         |
| 32×32      | ~3 min          | ~6 min       | ~3 min         |

**Total for all three resolutions:** ~45-60 minutes (with 500 test images)

## Best Practices

1. **Always train DSR before fine-tuning EdgeFace** for that resolution
2. **Use consistent datasets** across all resolutions for fair comparison
3. **Regenerate VLR datasets** when changing downsampling strategies
4. **Use `--use-small-gallery` flag** for 1:1 and small 1:N scenarios
5. **Export results to JSON** for reproducibility and version control
6. **Generate PDF reports** for presentations and papers
7. **Verify checkpoint paths** before running long training jobs

## Citation

If using this evaluation framework in your research, please cite appropriately and reference:

- DSR architecture and training methodology
- EdgeFace backbone architecture
- Multi-resolution evaluation protocol
- Metrics and statistical analysis methods

## Summary

The new multi-resolution capabilities enable:

- ✅ **Comprehensive training** across 16×16, 24×24, and 32×32 VLR inputs
- ✅ **Resolution-aware fine-tuning** with optimized hyperparameters
- ✅ **Publication-quality evaluation** with automated visualization
- ✅ **Statistical comparison** across all resolutions
- ✅ **Easy export** to JSON, PNG, and PDF for papers

This provides a complete framework for rigorous experimental comparison and publication of low-resolution face recognition research.
