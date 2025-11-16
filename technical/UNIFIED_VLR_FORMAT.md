# VLR Dataset Unified Naming Format

## Overview

All VLR (Very Low Resolution) datasets now use a **consistent naming format**: `vlr_images_{W}x{H}`

This applies to **ALL resolutions**, including 32×32 (no more special case).

## Before vs After

### Old Format (Inconsistent)

```
dataset/
├── train_processed/
│   ├── hr_images/
│   ├── vlr_images/              ← 32×32 (special case)
│   ├── vlr_images_24x24/        ← 24×24
│   ├── vlr_images_16x16/        ← 16×16
│   ├── vlr_images_old_24x24/    ← Backup clutter
│   └── vlr_images_old_16x16/    ← More backup clutter
```

### New Format (Consistent)

```
dataset/
├── train_processed/
│   ├── hr_images/
│   ├── vlr_images_32x32/        ← 32×32 (consistent!)
│   ├── vlr_images_24x24/        ← 24×24
│   └── vlr_images_16x16/        ← 16×16
```

## Key Features

✅ **Consistent naming** - All resolutions follow same pattern  
✅ **No backups** - Regeneration replaces existing folders (after confirmation)  
✅ **Auto-detection** - Script detects misnamed folders and renames them  
✅ **Multiple resolutions** - Create 16×16, 24×24, 32×32 simultaneously  
✅ **Non-destructive** - Creating new size doesn't touch existing sizes  
✅ **Confirmation prompts** - Always asks before overwriting existing data

## Updated Scripts

All scripts now support the unified format:

### 1. Dataset Generation

```powershell
# Generate 24×24 dataset (doesn't touch 32×32 or 16×16)
poetry run python -m technical.tools.regenerate_vlr_dataset --vlr-sizes 24

# Generate all three resolutions
poetry run python -m technical.tools.regenerate_vlr_dataset --vlr-sizes 16 24 32

# For frontal-only dataset
poetry run python -m technical.tools.regenerate_vlr_dataset \
    --vlr-sizes 24 \
    --dataset-root technical/dataset/frontal_only

# Force overwrite without prompts (use with caution!)
poetry run python -m technical.tools.regenerate_vlr_dataset \
    --vlr-sizes 24 \
    --force
```

### 2. DSR Training

```powershell
# Automatically looks for vlr_images_{size}x{size}
poetry run python -m technical.dsr.train_dsr --vlr-size 24 --device cuda
poetry run python -m technical.dsr.train_dsr --vlr-size 32 --device cuda
```

### 3. EdgeFace Fine-Tuning

```powershell
# Automatically looks for vlr_images_{size}x{size}
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 24 --device cuda
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 32 --device cuda
```

### 4. Evaluation Scripts

```powershell
# GUI evaluation (automatically handles all formats)
poetry run python -m technical.pipeline.evaluate_gui --resolutions 16 24 32

# Verification/Identification evaluation
poetry run python -m technical.pipeline.evaluate_verification \
    --mode verification \
    --test-root technical/dataset/frontal_only/test \
    --vlr-size 24
```

## Migration Guide

### If You Have Old Format Datasets

The regeneration script will **automatically detect and rename** legacy folders:

```powershell
# This will:
# 1. Detect "vlr_images" folder
# 2. Read first image to determine size (e.g., 32×32)
# 3. Rename to "vlr_images_32x32"
poetry run python -m technical.tools.regenerate_vlr_dataset --vlr-sizes 32 24 16
```

**What happens:**

1. Script finds `vlr_images/` folder
2. Reads first image → detects it's 32×32
3. Renames to `vlr_images_32x32/`
4. Proceeds with generation

### Manual Migration (If Needed)

If you want to manually migrate:

```powershell
# Windows PowerShell
cd A:\Programming\School\cs565\project\technical\dataset\frontal_only\train
Rename-Item "vlr_images" "vlr_images_32x32"

# Repeat for val and test
cd ..\val
Rename-Item "vlr_images" "vlr_images_32x32"

cd ..\test
Rename-Item "vlr_images" "vlr_images_32x32"
```

## Regeneration Behavior

### Creating New Resolution (Safe)

```powershell
# You have: vlr_images_32x32/
# Running: --vlr-sizes 24
# Result: vlr_images_32x32/ + vlr_images_24x24/ (no prompt)
```

### Overwriting Existing Resolution (Prompt)

```powershell
# You have: vlr_images_24x24/ (1,234 images)
# Running: --vlr-sizes 24
# Prompt:
#   ⚠️  Directory already exists: vlr_images_24x24
#      Contains 1234 images at 24x24 resolution
#      Overwrite and regenerate? (y/n):
```

### Force Mode (Skip Prompts)

```powershell
# Regenerate all without asking
poetry run python -m technical.tools.regenerate_vlr_dataset \
    --vlr-sizes 16 24 32 \
    --force
```

## Troubleshooting

### "RuntimeError: Missing VLR directory 'vlr_images_24x24'"

**Problem:** You're trying to train with a size that doesn't have a dataset yet.

**Solution:**

```powershell
# Generate the missing resolution
poetry run python -m technical.tools.regenerate_vlr_dataset --vlr-sizes 24

# If using frontal-only dataset
poetry run python -m technical.tools.regenerate_vlr_dataset \
    --vlr-sizes 24 \
    --dataset-root technical/dataset/frontal_only
```

### "Legacy 'vlr_images' exists but 'vlr_images_32x32' already present"

**Problem:** You have both old and new format folders.

**Solution:**

```powershell
# Manually check which is correct and delete the other
# Then regenerate if needed
```

### Script renamed my folder incorrectly

**Problem:** Auto-detection got the size wrong (rare).

**Solution:**

```powershell
# Manually rename back
Rename-Item "vlr_images_wrongsize" "vlr_images_correctsize"

# Or regenerate from scratch
poetry run python -m technical.tools.regenerate_vlr_dataset \
    --vlr-sizes correctsize \
    --force
```

## Benefits of Unified Format

### 1. Consistency

- No special cases in code
- Easier to understand and maintain
- Predictable behavior across all resolutions

### 2. Flexibility

- Train all three resolutions in parallel
- Switch between resolutions easily
- Keep multiple resolutions coexistent

### 3. Clarity

- Folder name instantly tells you the resolution
- No guessing "is vlr_images 32×32 or something else?"
- Easy to validate dataset structure

### 4. Automation-Friendly

- Scripts can iterate over standard patterns
- Easier to write batch processing
- Better for research workflows

## Example Workflow

### Complete Multi-Resolution Setup

```powershell
# Step 1: Generate all VLR datasets
poetry run python -m technical.tools.regenerate_vlr_dataset \
    --vlr-sizes 16 24 32 \
    --dataset-root technical/dataset/frontal_only

# Result:
# technical/dataset/frontal_only/
# ├── train/
# │   ├── hr_images/
# │   ├── vlr_images_16x16/  ← NEW
# │   ├── vlr_images_24x24/  ← NEW
# │   └── vlr_images_32x32/  ← RENAMED from vlr_images/
# ├── val/
# │   └── (same structure)
# └── test/
#     └── (same structure)

# Step 2: Train DSR models for each resolution
poetry run python -m technical.dsr.train_dsr --vlr-size 16 --device cuda
poetry run python -m technical.dsr.train_dsr --vlr-size 24 --device cuda
poetry run python -m technical.dsr.train_dsr --vlr-size 32 --device cuda

# Step 3: Fine-tune EdgeFace for each resolution
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 16 --device cuda
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 24 --device cuda
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 32 --device cuda

# Step 4: Evaluate all resolutions
poetry run python -m technical.pipeline.evaluate_gui \
    --resolutions 16 24 32 \
    --device cuda
```

## Summary

- ✅ **All VLR folders use format**: `vlr_images_{W}x{H}`
- ✅ **No more special case for 32×32**
- ✅ **Auto-detection and renaming** of legacy folders
- ✅ **Confirmation prompts** before overwriting
- ✅ **Non-destructive** to other resolutions
- ✅ **All scripts updated** to use unified format

**Next step:** Run regeneration script to migrate your datasets!

```powershell
poetry run python -m technical.tools.regenerate_vlr_dataset --vlr-sizes 16 24 32
```
