# LRFR Project - Quick Start Guide

## Complete Training & Evaluation Workflow

This guide covers the full pipeline for training, fine-tuning, and evaluating low-resolution face recognition models at different resolutions (16×16, 24×24, 32×32).

---

## Prerequisites

1. **Install dependencies:**

   ```powershell
   poetry install --with dev
   ```

2. **Prepare dataset:**

   - Ensure `technical/dataset/train_processed/`, `val_processed/`, and `test_processed/` exist
   - Each should contain `hr_images/` and `vlr_images/` (or `vlr_images_16x16/`, `vlr_images_24x24/`)

3. **GPU recommended** (CUDA-enabled) for reasonable training times

---

## Step 1: Train DSR Model

Train a Deep Super-Resolution model for a specific VLR resolution:

```powershell
# For 32×32 VLR input (default)
poetry run python -m technical.dsr.train_dsr --vlr-size 32 --device cuda

# For 24×24 VLR input
poetry run python -m technical.dsr.train_dsr --vlr-size 24 --device cuda

# For 16×16 VLR input
poetry run python -m technical.dsr.train_dsr --vlr-size 16 --device cuda
```

**Output:** `technical/dsr/dsr{size}.pth` (e.g., `dsr32.pth`)  
**Training time:** ~8-10 hours (100 epochs on RTX 3060 Ti)

**Optional flags:**

- `--epochs 80` - Reduce training time
- `--batch-size 12` - Lower memory usage
- `--frontal-only` - Use frontal-only filtered dataset

---

## Step 2: Fine-Tune EdgeFace on DSR Outputs

Fine-tune the EdgeFace recognition model to better recognize faces from your DSR outputs:

```powershell
# For 32×32
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 32 --device cuda

# For 24×24
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 24 --device cuda

# For 16×16
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 16 --device cuda
```

**Output:** `technical/facial_rec/edgeface_weights/edgeface_finetuned_{size}.pth`  
**Training time:** ~2-3 hours (35 epochs on RTX 3060 Ti)

---

## Step 3: Cyclically Fine-Tune DSR (Recommended)

**Instead of retraining from scratch**, continue training your DSR model using the fine-tuned EdgeFace for additional accuracy gains:

```powershell
# For 32×32 (manual approach)
poetry run python -m technical.dsr.train_dsr `
    --vlr-size 32 `
    --device cuda `
    --resume technical/dsr/dsr32.pth `
    --edgeface edgeface_finetuned_32.pth `
    --epochs 50 `
    --learning-rate 8e-5 `
    --lambda-identity 0.65 `
    --lambda-feature-match 0.20
```

**Or use the automated pipeline for all resolutions:**

```powershell
# Train all resolutions with full cycle (Initial DSR → EdgeFace FT → DSR Cyclic FT)
poetry run python -m technical.tools.cyclic_train --device cuda

# Skip initial training if already done
poetry run python -m technical.tools.cyclic_train --skip-initial --skip-edgeface --device cuda

# Train only specific resolution
poetry run python -m technical.tools.cyclic_train --vlr-sizes 32 --device cuda
```

**Output:** Updates `technical/dsr/dsr{size}.pth` with improved weights  
**Training time:** ~4-6 hours (50 epochs on RTX 3060 Ti)  
**Expected improvement:** +8-15% accuracy

**Why cyclic training?**

- ✅ 2-3× faster than full retraining
- ✅ More stable convergence
- ✅ Same or better accuracy gains
- ✅ Preserves learned features

---

## Step 4: Evaluate Models

### Option A: Comprehensive GUI Evaluation (Recommended for Research)

Evaluate all resolutions with publication-quality visualizations:

```powershell
# GUI mode (interactive)
poetry run python -m technical.pipeline.evaluate_gui

# Or CLI mode (automated)
poetry run python -m technical.pipeline.evaluate_gui `
    --test-root technical/dataset/frontal_only/test `
    --gallery-root technical/dataset/frontal_only/train `
    --output-dir evaluation_results `
    --resolutions 16 24 32 `
    --device cuda
```

**Outputs:**

- `evaluation_report.pdf` - 6-page comprehensive report with all metrics
- `results.json` - Detailed metrics in JSON format
- Individual PNG plots (ROC curves, quality metrics, score distributions, etc.)

**Metrics computed:**

- **Image Quality:** PSNR, SSIM, Identity Similarity
- **Verification (1:1):** EER, ROC AUC, TAR @ FAR thresholds
- **Identification (1:N):** Rank-1/5/10 accuracy

### Option B: Quick Dataset Evaluation

Fast evaluation on a test dataset:

```powershell
poetry run python -m technical.pipeline.evaluate_dataset `
    --dataset-root technical/dataset/test_processed `
    --threshold 0.35 `
    --device cuda
```

### Option C: Verification/Identification Metrics Only

For specific face recognition metrics:

```powershell
# 1:1 Verification (e.g., phone unlock)
poetry run python -m technical.pipeline.evaluate_verification `
    --mode verification `
    --test-root technical/dataset/frontal_only/test `
    --vlr-size 32 `
    --device cuda

# 1:N Identification (e.g., small office with 10 people)
poetry run python -m technical.pipeline.evaluate_verification `
    --mode identification `
    --gallery-root technical/dataset/frontal_only/train `
    --test-root technical/dataset/frontal_only/test `
    --max-gallery-size 10 `
    --vlr-size 32 `
    --device cuda
```

---

## Complete Workflow Summary

### Single Resolution (e.g., 32×32)

```powershell
# 1. Train DSR (~10 hours)
poetry run python -m technical.dsr.train_dsr --vlr-size 32 --device cuda

# 2. Fine-tune EdgeFace (~3 hours)
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 32 --device cuda

# 3. Cyclic fine-tune DSR (~5 hours)
poetry run python -m technical.dsr.train_dsr `
    --vlr-size 32 --device cuda --resume technical/dsr/dsr32.pth `
    --edgeface edgeface_finetuned_32.pth --epochs 50 `
    --learning-rate 8e-5 --lambda-identity 0.65

# 4. Evaluate
poetry run python -m technical.pipeline.evaluate_gui --resolutions 32 --device cuda
```

**Total time:** ~18 hours  
**Expected accuracy:** 48-58% (up from 40-48% baseline)

### All Resolutions (16×16, 24×24, 32×32)

```powershell
# Automated pipeline for all resolutions
poetry run python -m technical.tools.cyclic_train --device cuda

# Evaluate all
poetry run python -m technical.pipeline.evaluate_gui --device cuda
```

**Total time:** ~54 hours sequential (or ~18 hours with 3 GPUs in parallel)

---

## Quick Reference Table

| Command                             | Purpose                  | Time   | Output                     |
| ----------------------------------- | ------------------------ | ------ | -------------------------- |
| `train_dsr.py --vlr-size N`         | Train DSR model          | ~10h   | `dsrN.pth`                 |
| `finetune_edgeface.py --vlr-size N` | Fine-tune EdgeFace       | ~3h    | `edgeface_finetuned_N.pth` |
| `train_dsr.py --resume ...`         | Cyclic DSR fine-tuning   | ~5h    | Updated `dsrN.pth`         |
| `cyclic_train.py`                   | Automated full pipeline  | ~18h   | All checkpoints            |
| `evaluate_gui.py`                   | Comprehensive evaluation | ~10min | PDF + JSON + plots         |

---

## Troubleshooting

### "CUDA out of memory"

```powershell
# Reduce batch size
--batch-size 12  # or even 8
```

### "FileNotFoundError: vlr_images_16x16"

```powershell
# Regenerate VLR dataset for that resolution
poetry run python -m technical.tools.regenerate_vlr_dataset --vlr-size 16
```

### Training seems stuck / not improving

- Check that EdgeFace checkpoint loaded correctly (look for log message)
- Try adjusting learning rate: `--learning-rate 1e-4`
- Verify dataset quality (HR and VLR images match)

### Low accuracy after training

- Ensure using frontal-only dataset: `--frontal-only`
- Try threshold sweep: `python -m technical.tools.threshold_sweep`
- Check metrics breakdown in GUI evaluation for diagnosis

---

## Next Steps

- **For detailed documentation:**

  - Multi-resolution guide: `technical/MULTI_RESOLUTION_GUIDE.md`
  - Cyclic training analysis: `technical/CYCLIC_VS_RETRAINING_ANALYSIS.md`
  - Evaluation guide: `technical/EVALUATION_GUIDE.md`

- **For paper/research:**
  - Use `evaluate_gui.py` to generate publication-quality figures
  - Export metrics from `results.json` for tables
  - See example figures in evaluation output directory

---

## Architecture Overview

```
VLR Image (16×16/24×24/32×32)
    ↓
DSR Model (trained in Step 1, improved in Step 3)
    ↓
SR Image (112×112)
    ↓
EdgeFace Model (fine-tuned in Step 2)
    ↓
Embedding (512-dim)
    ↓
Cosine Similarity → Identity + Confidence Score
```

**Key insight:** Cyclic fine-tuning (Step 3) teaches DSR to produce outputs that the fine-tuned EdgeFace recognizes best, creating a feedback loop that improves accuracy by 8-15%.

---

Ready to start? Pick a resolution and run the commands above! 🚀

For questions or issues, see the detailed guides in `technical/` directory.
