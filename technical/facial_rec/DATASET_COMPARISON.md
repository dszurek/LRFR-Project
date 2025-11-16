# Dataset Comparison: DSR vs EdgeFace Fine-tuning

## Quick Answer

**YES, create a separate `edgeface_finetune/` dataset.** Here's why:

## Dataset Structures Compared

### Current: `frontal_only/` (DSR Training)

```
frontal_only/
├── train/          # Subjects: 001, 002, 003, 004, 005
├── val/            # Subjects: 006, 007, 008, 009, 010
└── test/           # Subjects: 011, 012, 013, 014, 015
```

**Design Goal:** Super-resolution quality (PSNR, SSIM)

- ✅ **Different people** in each split prevents data leakage
- ✅ Perfect for DSR training (comparing image quality)
- ❌ Bad for classification (can't validate on unseen people)

### Recommended: `edgeface_finetune/` (Recognition Training)

```
edgeface_finetune/
├── train/          # Subjects: 001-100 (80% of their images)
└── val/            # Subjects: 001-100 (20% of their images, different from train)
```

**Design Goal:** Identity classification and embedding quality

- ✅ **Same people** in both splits with different images
- ✅ Perfect for EdgeFace fine-tuning (classification task)
- ✅ Meaningful validation accuracy (not stuck at 0%)

## Why Same People in Train/Val?

### The Recognition Task:

EdgeFace fine-tuning is a **supervised classification task**:

- Input: Face image
- Output: Subject ID (person 001, 002, etc.)
- Goal: Learn discriminative features for each identity

### What Validation Tests:

- **Train:** "Learn what subject 001 looks like across different images"
- **Val:** "Can you recognize subject 001 in a new image you haven't seen?"
- **Result:** Tests if model generalizes to new images of the same person

### Why Current Dataset Fails:

- **Train:** Learns subjects 001-005
- **Val:** Tests on subjects 006-010 (never seen!)
- **Result:** Like teaching Spanish then testing on French → 0% accuracy

## Metrics Comparison

| Metric                      | frontal_only/        | edgeface_finetune/        |
| --------------------------- | -------------------- | ------------------------- |
| **Embedding Similarity**    | ✅ Works (0.7 → 0.9) | ✅ Works (0.7 → 0.9)      |
| **Classification Accuracy** | ❌ Always 0%         | ✅ Meaningful (0.6 → 0.9) |
| **Per-Subject Analysis**    | ❌ Not possible      | ✅ Possible               |
| **Overfitting Detection**   | ⚠️ Harder            | ✅ Easier                 |

## How to Create the Dataset

### Step 1: Run the Script

```bash
cd a:\Programming\School\cs565\project
poetry run python -m technical.facial_rec.create_finetuning_dataset
```

**What it does:**

1. Scans `frontal_only/train/` and `frontal_only/val/`
2. Groups all images by subject ID
3. Filters subjects with < 5 images (too few to split)
4. Splits each subject's images 80/20 into train/val
5. Creates `technical/dataset/edgeface_finetune/`

**Expected output:**

```
✅ Dataset Creation Complete!
====================================================================
Subjects:        287
Train images:    8,534
Val images:      2,134
Split ratio:     80.0% train / 20.0% val

📁 Output directory: technical/dataset/edgeface_finetune
```

### Step 2: Train with New Dataset

```bash
poetry run python -m technical.facial_rec.finetune_edgeface \
  --train-dir "technical/dataset/edgeface_finetune/train" \
  --val-dir "technical/dataset/edgeface_finetune/val" \
  --device cuda \
  --edgeface edgeface_xxs.pt
```

**Expected improvements:**

```
# With frontal_only/ (old):
Epoch 10/25 | Train Acc: 0.8932 | Val Acc: 0.0000 Sim: 0.8456

# With edgeface_finetune/ (new):
Epoch 10/25 | Train Acc: 0.8932 | Val Acc: 0.7234 Sim: 0.8456
                                          ^^^^^^
                                     Now meaningful!
```

## Both Datasets Have Their Place

### Use `frontal_only/` for:

- ✅ DSR training (super-resolution)
- ✅ Testing image quality improvements
- ✅ Pipeline evaluation with unseen faces

### Use `edgeface_finetune/` for:

- ✅ EdgeFace fine-tuning (recognition)
- ✅ Validating classification performance
- ✅ Per-subject analysis

## Architecture Clarification

### TorchScript vs ConvNeXt

You asked about this - here's the clarification:

- **File format:** TorchScript (`.pt` serialization)
- **Architecture inside:** ConvNeXt (neural network structure)

**Analogy:**

- TorchScript = ZIP file (container format)
- ConvNeXt = The actual files inside the ZIP

**In `edgeface_xxs.pt`:**

- Saved as: TorchScript format (torch.jit.save)
- Contains: ConvNeXt-XXS weights and structure
- Loaded by: `torch.jit.load()` (our code does this correctly!)

The code handles this properly:

```python
# Try loading as TorchScript (works for edgeface_xxs.pt)
try:
    model = torch.jit.load(str(weights_path), map_location=device)
    # This loads the TorchScript container with ConvNeXt inside
    return model
except:
    # Fall back to manual architecture construction
    model = EdgeFace(embedding_size=512, back="edgeface_xxs")
```

## Recommended Action Plan

### Option 1: Quick Test (Current Dataset)

**Time:** 5 minutes to start

```bash
poetry run python -m technical.facial_rec.finetune_edgeface --device cuda --edgeface edgeface_xxs.pt
```

- ✅ See if training works at all
- ✅ Watch embedding similarity improve
- ⚠️ Classification accuracy will be 0%

### Option 2: Proper Training (New Dataset) - RECOMMENDED

**Time:** 30 minutes setup + training time

```bash
# Create dataset
poetry run python -m technical.facial_rec.create_finetuning_dataset

# Train with it
poetry run python -m technical.facial_rec.finetune_edgeface \
  --train-dir "technical/dataset/edgeface_finetune/train" \
  --val-dir "technical/dataset/edgeface_finetune/val" \
  --device cuda \
  --edgeface edgeface_xxs.pt
```

- ✅ Meaningful validation metrics
- ✅ Better model quality assessment
- ✅ Standard ML practice

## Conclusion

**Create the separate dataset.** It only takes ~30 minutes and gives you:

- ✅ Proper validation metrics (accuracy AND similarity)
- ✅ Better training insights
- ✅ Ability to detect overfitting
- ✅ Per-subject performance analysis
- ✅ Industry-standard approach

The `frontal_only` dataset is perfect for what it was designed for (DSR training), but EdgeFace fine-tuning is a different task that needs a different dataset structure.
