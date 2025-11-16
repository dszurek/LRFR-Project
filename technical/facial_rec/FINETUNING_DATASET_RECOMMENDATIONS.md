# EdgeFace Fine-tuning Dataset Recommendations

## Problem Summary

Your EdgeFace fine-tuning script is not showing accuracy improvements because of a **fundamental dataset mismatch**:

### Current Dataset Structure (frontal_only)

- **Train**: Subject IDs starting with `001`, `002`, `003`, etc.
- **Val**: Subject IDs starting with `006`, `007`, `008`, etc.
- **Test**: Subject IDs starting with different numbers

**This means:**

- ✅ **Good for DSR training** - Different people in train/val/test ensures no data leakage
- ❌ **Bad for EdgeFace fine-tuning** - Validation contains people the model has NEVER seen during training

### Why This Breaks Validation Accuracy

EdgeFace fine-tuning uses **classification loss** with ArcFace:

- Model learns to classify face embeddings into subject IDs
- During training: Learns features for subjects 001, 002, 003, etc.
- During validation: Tries to classify subjects 006, 007, 008, etc. (completely new people!)
- Result: **Random guessing** because the model was never trained on these subjects

**Analogy:** It's like training a model to recognize cats and dogs, then testing it on birds and fish. The accuracy will be near 0%, even if the model learned cats/dogs perfectly.

## The Fix Applied

I've updated `finetune_edgeface.py` to use **embedding similarity** as the primary metric instead of classification accuracy:

### New Validation Metrics:

1. **Embedding Similarity** (PRIMARY METRIC)

   - Measures cosine similarity between DSR output embedding and HR ground truth embedding
   - **What it tells you:** How well does the DSR output preserve the person's identity?
   - Target: > 0.90 (excellent), > 0.85 (good), < 0.80 (needs improvement)
   - **This works even with different people in train/val!**

2. **Classification Accuracy** (SECONDARY METRIC)
   - Only computed on subjects that appear in BOTH train and val sets
   - With current dataset: This will likely be 0% because no overlap
   - With proper dataset: This becomes meaningful

### What Changed in the Code:

```python
# Old validation:
val_loss, val_acc = validate(backbone, arcface, val_loader, device)
# Only reported classification accuracy (meaningless with different people)

# New validation:
val_loss, val_acc, val_similarity = validate(
    backbone, arcface, val_loader, device, train_subject_ids
)
# Reports both classification accuracy (for overlap subjects only)
# AND embedding similarity (meaningful for all subjects)
```

## Should You Create a New Dataset?

### Option 1: Keep Current Dataset (Quick Fix)

**Pros:**

- No additional work required
- Embedding similarity metric still works
- Can start training immediately

**Cons:**

- Classification accuracy will be 0% or near-random
- Less interpretable validation metrics
- Harder to detect overfitting on specific identities

**When to use:** Quick experiments, initial baseline, limited time

### Option 2: Create Dedicated Fine-tuning Dataset (Recommended)

**Pros:**

- Meaningful classification accuracy
- Better overfitting detection
- More standard ML validation approach
- Can track per-subject performance

**Cons:**

- Requires dataset reorganization effort (~30-60 minutes)
- Need to re-run training from scratch

**When to use:** Production training, final model, research paper results

## How to Create a Proper Fine-tuning Dataset

### Approach 1: Same Subjects, Split Images

Reorganize so train/val contain the **same people** but **different images**:

```
edgeface_finetune_dataset/
├── train/
│   ├── vlr_images/
│   │   ├── 001_01_01_041_00.png  # Subject 001, image 00
│   │   ├── 001_01_01_041_02.png  # Subject 001, image 02
│   │   ├── 001_01_01_041_04.png  # Subject 001, image 04
│   │   ├── 002_01_01_041_00.png  # Subject 002, image 00
│   │   └── ...
│   └── hr_images/
│       └── (matching files)
└── val/
    ├── vlr_images/
    │   ├── 001_01_01_041_01.png  # Subject 001, image 01 (different from train)
    │   ├── 001_01_01_041_03.png  # Subject 001, image 03
    │   ├── 002_01_01_041_01.png  # Subject 002, image 01
    │   └── ...
    └── hr_images/
        └── (matching files)
```

**Strategy:**

- For each subject, use 80% of images for training, 20% for validation
- Ensures model learns to generalize to new images of the same person
- Classification accuracy becomes meaningful

### Approach 2: Mixed Dataset (Hybrid)

Use both known and unknown subjects in validation:

```
edgeface_finetune_dataset/
├── train/
│   └── Subjects 001-200 (all images)
└── val/
    ├── Subjects 001-200 (20% of images) - known subjects
    └── Subjects 201-250 (all images) - unknown subjects for embedding similarity
```

**Benefits:**

- Classification accuracy tracks known-subject performance
- Embedding similarity tracks generalization to new people
- More comprehensive validation

### Script to Create Reorganized Dataset

I'll create a helper script for you:

```python
# See create_finetuning_dataset.py (created below)
```

## Recommended Next Steps

1. **Immediate (to test current fixes):**

   ```bash
   cd a:\Programming\School\cs565\project
   poetry run python -m technical.facial_rec.finetune_edgeface --device cuda --edgeface edgeface_xxs.pt
   ```

   - This will use embedding similarity as primary metric
   - Watch for `Val Sim:` in output - target > 0.85

2. **Short-term (for better validation):**

   - Run the dataset reorganization script (see below)
   - Retrain with new dataset
   - Now both accuracy and similarity will be meaningful

3. **Debugging checklist if similarity is low:**
   - Check DSR outputs are high quality (PSNR > 30dB)
   - Verify EdgeFace loads correctly (no missing keys)
   - Ensure images are frontal faces (use frontal_only dataset)
   - Try lower learning rates if loss doesn't decrease

## Architecture Compatibility (ConvNeXt)

Your recent switch to ConvNeXt architecture is **fully supported** with the updated code:

### What Was Fixed:

1. **TorchScript loading** - Handles ConvNeXt-based `edgeface_xxs.pt` correctly
2. **Architecture detection** - Auto-detects backbone type from filename
3. **State dict handling** - Strips prefixes for both TorchScript and checkpoint formats

### Supported Models:

- ✅ `edgeface_xxs.pt` (ConvNeXt, 1.7MB) - TorchScript
- ✅ `edgeface_s_gamma_05.pt` (ConvNeXt-S, 14.7MB) - TorchScript
- ✅ `edgeface_finetuned.pth` (ConvNeXt or LDC) - State dict checkpoint
- ✅ Any architecture in `edgeface.py`

### No Issues Expected:

The ConvNeXt architecture uses different layer names but the embedding output is identical (512-dim vector), so the fine-tuning process works the same way.

## Expected Training Output

With the fixes, you should see:

```
STAGE 1: Training ArcFace head (backbone frozen)
====================================================================
Epoch 01/5 | Train Loss: 6.2341 Acc: 0.1234 | Val Loss: 5.8932 Acc: 0.0000 Sim: 0.7234
Epoch 02/5 | Train Loss: 5.1234 Acc: 0.3456 | Val Loss: 4.9876 Acc: 0.0000 Sim: 0.7891
  ✓ New best validation similarity: 0.7891
...

STAGE 2: Fine-tuning entire model (backbone unfrozen)
====================================================================
Epoch 01/25 | Train Loss: 4.5678 Acc: 0.5234 | Val Loss: 4.3456 Acc: 0.0000 Sim: 0.8456 | LR: 6.00e-06
  ✓ Saved checkpoint to edgeface_finetuned.pth (val sim: 0.8456)
...
Epoch 15/25 | Train Loss: 2.1234 Acc: 0.8932 | Val Loss: 2.5678 Acc: 0.0000 Sim: 0.9234 | LR: 2.34e-06
  ✓ Saved checkpoint to edgeface_finetuned.pth (val sim: 0.9234)
```

**What to look for:**

- ✅ Similarity increasing (0.7 → 0.8 → 0.9)
- ✅ Train accuracy increasing (learning identities)
- ⚠️ Val accuracy stays 0.0 (expected with different people)
- ✅ Checkpoint saved when similarity improves

## Conclusion

**The main issue was the validation metric**, not the training process itself. The updated script now:

1. ✅ Uses embedding similarity as primary metric (works with any dataset)
2. ✅ Only computes classification accuracy on overlapping subjects
3. ✅ Supports ConvNeXt architecture via TorchScript
4. ✅ Provides clearer progress tracking
5. ✅ Saves checkpoints based on what matters (similarity)

**You can train immediately** with the current dataset and see meaningful progress via the similarity metric. Creating a proper fine-tuning dataset is recommended but optional.
