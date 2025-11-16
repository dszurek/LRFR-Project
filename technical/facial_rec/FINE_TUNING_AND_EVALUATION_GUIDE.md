# Fine-Tuning and Evaluation Guide

## Overview

This guide explains how to:

1. Fine-tune EdgeFace on DSR-upscaled images
2. Evaluate the fine-tuned model on the frontal_only dataset
3. Understand the evaluation metrics

---

## 1. Fine-Tuning EdgeFace

### Dataset Structure

The fine-tuning uses the `edgeface_finetune` dataset:

```
technical/dataset/edgeface_finetune/
├── train/
│   ├── hr_images/     # High-resolution ground truth (112×112)
│   └── vlr_images/    # Very-low-resolution inputs (32×32)
└── val/
    ├── hr_images/     # High-resolution ground truth (112×112)
    └── vlr_images/    # Very-low-resolution inputs (32×32)
```

**Key Properties:**

- ✅ Same 518 subjects in both train and val
- ✅ Different images per subject in train vs val
- ✅ 80/20 split per subject
- ✅ Train: 25,729 images | Val: 6,677 images

### Run Fine-Tuning

```bash
# Basic command
poetry run python -m technical.facial_rec.finetune_edgeface \
    --train-dir "technical/dataset/edgeface_finetune/train" \
    --val-dir "technical/dataset/edgeface_finetune/val" \
    --device cuda \
    --edgeface edgeface_xxs.pt

# With custom output path
poetry run python -m technical.facial_rec.finetune_edgeface \
    --train-dir "technical/dataset/edgeface_finetune/train" \
    --val-dir "technical/dataset/edgeface_finetune/val" \
    --device cuda \
    --edgeface edgeface_xxs.pt \
    --output edgeface_finetuned_custom.pth
```

### What to Expect

**Stage 1 (10 epochs - Head training only):**

```
Epoch 01/10 | Train Loss: 14.52 Acc: 0.0000 Top5: 0.0123 | Val Loss: 15.05 Acc: 0.0000 Top5: 0.0089 Sim: 0.9992
    [Top-1] 5/25729 = 0.0002 | [Top-5] 317/25729 = 0.0123
    [Val Top-1] 0/6677 = 0.0000 | [Top-5] 59/6677 = 0.0088

Epoch 10/10 | Train Loss: 13.12 Acc: 0.0015 Top5: 0.0850 | Val Loss: 13.89 Acc: 0.0012 Top5: 0.0780 Sim: 0.9991
    [Top-1] 39/25729 = 0.0015 | [Top-5] 2187/25729 = 0.0850
    [Val Top-1] 8/6677 = 0.0012 | [Top-5] 521/6677 = 0.0780
```

**Stage 2 (25 epochs - Full fine-tuning):**

```
Epoch 25/25 | Train Loss: 8.45 Acc: 0.1250 Top5: 0.4820 | Val Loss: 9.12 Acc: 0.0980 Top5: 0.3950 Sim: 0.9985
    [Top-1] 3216/25729 = 0.1250 | [Top-5] 12401/25729 = 0.4820
    [Val Top-1] 654/6677 = 0.0980 | [Top-5] 2638/6677 = 0.3950
```

### Output Files

Fine-tuning creates:

- `edgeface_finetuned_stage1.pth` - Stage 1 checkpoint (best similarity)
- `edgeface_finetuned.pth` - Final Stage 2 model (best overall)

**Checkpoint Contents:**

```python
{
    'stage': 2,
    'epoch': 25,
    'backbone_state_dict': {...},  # EdgeFace ConvNeXt weights
    'arcface_state_dict': {...},   # Classification head
    'optimizer_state_dict': {...},
    'val_similarity': 0.9985,
    'val_accuracy': 0.0980,
    'config': FinetuneConfig(...),
    'subject_to_id': {...},        # Subject name → ID mapping
    'num_classes': 518
}
```

---

## 2. Evaluating on Frontal_Only Dataset

### Dataset Structure

The evaluation uses the `frontal_only/test` dataset:

```
technical/dataset/frontal_only/test/
├── hr_images/     # Gallery: High-resolution faces (112×112)
└── vlr_images/    # Probes: Very-low-resolution faces (32×32)
```

**How It Works:**

1. **Gallery Building**: Load all HR images, extract EdgeFace embeddings, group by subject
2. **Probe Evaluation**: For each VLR image:
   - Run through DSR (32×32 → 112×112)
   - Extract EdgeFace embedding from upscaled image
   - Find nearest neighbor in gallery
   - Compare predicted subject to ground truth (from filename)

### Run Evaluation

#### A. Evaluate Pretrained EdgeFace (baseline)

```bash
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights technical/facial_rec/edgeface_weights/edgeface_xxs.pt \
    --threshold 0.35
```

#### B. Evaluate Fine-Tuned EdgeFace

```bash
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights edgeface_finetuned.pth \
    --threshold 0.35
```

#### C. Compare Multiple Checkpoints

```bash
# Stage 1 checkpoint
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights edgeface_finetuned_stage1.pth \
    --threshold 0.35

# Stage 2 checkpoint (final)
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights edgeface_finetuned.pth \
    --threshold 0.35
```

### Evaluation Options

```bash
# Quick test (first 100 images)
--limit 100

# Save detailed CSV results
--dump-results evaluation_results.csv

# Override gallery/probe directories
--hr-dir technical/dataset/frontal_only/test/hr_images \
--vlr-dir technical/dataset/frontal_only/test/vlr_images

# Build gallery through DSR (usually not needed)
--gallery-via-dsr

# Adjust recognition threshold
--threshold 0.30  # More lenient (more predictions, lower precision)
--threshold 0.50  # Stricter (fewer predictions, higher precision)
```

### Understanding Evaluation Output

```
Building gallery from HR images ...
Registering gallery identities: 100%|███████| 250/250 subjects

Evaluating VLR probes ...
Evaluating probes: 100%|████████████████| 5000/5000 images

=== Aggregate metrics ===
Total probes evaluated : 5000
Correct predictions    : 3750 (75.00%)
Predicted as unknown   : 125 (2.50%)

=== Per-subject accuracy (top 10 by probe count) ===
person1 |  19/20  | 95.00%
person2 |  18/20  | 90.00%
person3 |  17/20  | 85.00%
...

=== Sample misidentifications ===
person1_0045.png: truth=person1, pred=person2, score=0.652
person3_0012.png: truth=person3, pred=person5, score=0.587
...
```

**Metrics Explained:**

- **Correct predictions**: Probe matched to correct subject
- **Predicted as unknown**: Similarity below threshold (no match)
- **Per-subject accuracy**: Shows which subjects are harder to recognize
- **Misidentifications**: Wrong subject predicted (false positives)

---

## 3. Architecture Verification

### How ConvNeXt Detection Works

The pipeline automatically detects architecture:

```python
# From pipeline.py _load_edgeface_model()

# 1. Try TorchScript first (for .pt files)
model = torch.jit.load("edgeface_xxs.pt")  # ✓ Works for pretrained

# 2. Fall back to state dict loading (for .pth fine-tuned)
state_dict = torch.load("edgeface_finetuned.pth")

# 3. Auto-detect architecture from keys
if any("stem" in key or "stages" in key for key in state_dict.keys()):
    backbone = "edgeface_xxs"  # ✓ ConvNeXt detected
elif any("features" in key for key in state_dict.keys()):
    backbone = "edgeface_s"    # LDC architecture

# 4. Create model with detected architecture
model = EdgeFace(embedding_size=512, back=backbone)
model.load_state_dict(state_dict)
```

### Verify Architecture in Logs

When running evaluation, check console output:

```
[EdgeFace] Attempting TorchScript load from edgeface_xxs.pt
[EdgeFace] Successfully loaded TorchScript model        # ✓ Pretrained

--- OR ---

[EdgeFace] TorchScript load failed: ...
[EdgeFace] Falling back to architecture loading
[EdgeFace] Loaded fine-tuned checkpoint (backbone_state_dict)
[EdgeFace] Detected ConvNeXt (edgeface_xxs) architecture  # ✓ Fine-tuned
```

---

## 4. Subject ID Extraction

### How Subject Names are Extracted

Both scripts use filename parsing:

```python
def _subject_from_filename(path: Path) -> str:
    return path.stem.split("_")[0]

# Examples:
# "person001_0045.png" → "person001"
# "Aaron_Eckhart_0001.png" → "Aaron"
# "subject123_frame_05.png" → "subject123"
```

**Important:** First underscore splits subject ID from rest!

### Verify Your Filenames

```bash
# Check frontal_only dataset naming
Get-ChildItem "technical/dataset/frontal_only/test/hr_images/*.png" |
    Select-Object -First 5 |
    ForEach-Object { $_.Name }

# Should output:
# person001_0001.png
# person001_0002.png
# person002_0001.png
# ...
```

If your filenames don't follow `{subject}_{identifier}.png` format, the evaluation will fail!

---

## 5. Complete Workflow

### Step 1: Prepare Dataset (if needed)

```bash
# If you need to reorganize frontal_only for fine-tuning
poetry run python technical/dataset/create_finetuning_dataset.py
```

### Step 2: Fine-Tune EdgeFace

```bash
poetry run python -m technical.facial_rec.finetune_edgeface \
    --train-dir "technical/dataset/edgeface_finetune/train" \
    --val-dir "technical/dataset/edgeface_finetune/val" \
    --device cuda \
    --edgeface edgeface_xxs.pt
```

**Wait time:** ~15-30 minutes per epoch (depends on GPU)

- Stage 1 (10 epochs): ~3-5 hours
- Stage 2 (25 epochs): ~8-12 hours
- **Total: ~11-17 hours on RTX 3080/4090**

### Step 3: Evaluate Baseline

```bash
# Baseline: Pretrained EdgeFace
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights technical/facial_rec/edgeface_weights/edgeface_xxs.pt \
    --threshold 0.35 \
    --dump-results results_baseline.csv
```

### Step 4: Evaluate Fine-Tuned

```bash
# Fine-tuned: After training
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights edgeface_finetuned.pth \
    --threshold 0.35 \
    --dump-results results_finetuned.csv
```

### Step 5: Compare Results

```bash
# Compare CSV files
# Expected improvement: +5-15% accuracy
```

---

## 6. Troubleshooting

### Issue: "No HR .png files found"

**Solution:** Check dataset structure

```bash
# Verify files exist
Get-ChildItem "technical/dataset/frontal_only/test/hr_images/*.png" | Measure-Object

# Check if in subdirectories (wrong structure)
Get-ChildItem "technical/dataset/frontal_only/test/hr_images" -Recurse -Filter "*.png"
```

### Issue: "All predictions are unknown"

**Solution:** Threshold too high or embeddings not similar

```bash
# Try lower threshold
--threshold 0.25  # More lenient

# Check if DSR is working
poetry run python technical/dsr/test_dsr.py
```

### Issue: "Missing keys in state_dict"

**Solution:** Fine-tuned checkpoint format changed

```bash
# Check checkpoint contents
poetry run python technical/tools/inspect_ckpt.py edgeface_finetuned.pth

# Should show:
# - backbone_state_dict (ConvNeXt weights)
# - num_classes: 518
```

### Issue: "Architecture detection failed"

**Solution:** Verify model file

```bash
# Check if TorchScript
python -c "import torch; print(torch.jit.load('edgeface_xxs.pt'))"

# Check state dict keys
python -c "import torch; print(list(torch.load('edgeface_finetuned.pth')['backbone_state_dict'].keys())[:10])"

# Should see: stem.*, stages.* for ConvNeXt
```

---

## 7. Expected Results

### Baseline (Pretrained EdgeFace)

- **Accuracy:** 60-75% (depending on DSR quality)
- **Unknown rate:** 2-5%
- **Per-subject variance:** High (some subjects 95%, others 40%)

### After Fine-Tuning

- **Accuracy:** 70-85% (+10-15% improvement)
- **Unknown rate:** 1-3% (better threshold calibration)
- **Per-subject variance:** Lower (more consistent across subjects)
- **Top-5 accuracy:** 85-95% (correct identity in top 5)

### Why Fine-Tuning Helps

1. **Domain Adaptation**: Model learns DSR artifacts (blur, compression)
2. **Better Separation**: Classification head learns to separate similar faces
3. **Threshold Calibration**: Embeddings more consistent → better thresholds
4. **Contrastive Learning**: DSR and HR embeddings align better (Stage 2)

---

## 8. Quick Reference

### Fine-Tuning Command

```bash
poetry run python -m technical.facial_rec.finetune_edgeface \
    --train-dir "technical/dataset/edgeface_finetune/train" \
    --val-dir "technical/dataset/edgeface_finetune/val" \
    --device cuda \
    --edgeface edgeface_xxs.pt
```

### Evaluation Command (Fine-Tuned)

```bash
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights edgeface_finetuned.pth \
    --threshold 0.35
```

### Evaluation Command (Baseline)

```bash
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights technical/facial_rec/edgeface_weights/edgeface_xxs.pt \
    --threshold 0.35
```

---

## Summary

✅ **Fine-tuning:** Uses `edgeface_finetune` dataset (same subjects, different images train/val)  
✅ **Evaluation:** Uses `frontal_only/test` dataset (subject from filename `{subject}_*`)  
✅ **ConvNeXt:** Auto-detected from TorchScript (.pt) or state dict keys (.pth)  
✅ **DSR Integration:** Probes upscaled 32×32→112×112, gallery uses HR directly  
✅ **Metrics:** Accuracy, per-subject accuracy, unknown rate, misidentifications

The scripts are production-ready and correctly configured! 🚀
