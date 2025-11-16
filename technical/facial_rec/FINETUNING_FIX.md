# EdgeFace Fine-Tuning Analysis & Fix

## Problem Discovery

### Initial Symptoms

- Fine-tuned EdgeFace showed **1.92% accuracy** on validation set
- Training metrics showed **0.9992 similarity** and **36.85% Top-5 accuracy**
- Huge discrepancy between training success and evaluation failure

### Root Cause Analysis

#### Test 1: Gallery Enrollment Check

- ✅ Gallery correctly built from HR images (no DSR)
- ✅ Probes correctly processed through VLR→DSR pipeline
- ✅ Subject IDs extracted correctly (518 subjects, 100% overlap)
- ✅ ArcFace head correctly excluded from evaluation (only backbone used)

#### Test 2: Embedding Similarity Check

```
Gallery (Subject 001, 5 HR images): Mean embedding norm = 1.0000
Probe (Subject 001, VLR→DSR): Embedding norm = 1.0000
Cosine Similarity: 0.1768 ❌ (threshold = 0.35)
```

**The fine-tuned model produces embeddings with only 0.18 similarity for the SAME PERSON!**

#### Test 3: ArcFace Head Analysis

```
Backbone-only similarity: 0.1768
Gallery ArcFace score (HR→class 001): 0.9320
Probe ArcFace score (DSR→class 001): 0.1476
```

**Finding:** The model learned to classify HR images well (0.93), but DSR outputs don't map to the same class space (0.15). Keeping the ArcFace head wouldn't help.

#### Test 4: Pretrained Model Performance

```
Subject 001: Average similarity = 0.9873
  5/5 matches (100%)
  Range: 0.9859 - 0.9886

Overall (5 subjects, 15 probes):
  Average similarity: 0.9878
  Matches: 15/15 (100%) ✅
```

**THE PRETRAINED MODEL WORKS PERFECTLY!** Fine-tuning destroyed the embeddings.

### Why Did Training Metrics Lie?

The validation function measured:

```python
# Line 591-594 of finetune_edgeface.py
similarity = (sr_embeddings_norm * hr_embeddings_norm).sum(dim=1)
```

This computed similarity between **DSR and HR from the SAME BATCH** (paired training samples processed together), NOT between different images of the same person.

**Key Insight:**

- **Training metric (0.9992)**: DSR output vs its paired HR image (same training sample)
- **Evaluation metric (0.1768)**: DSR output vs different HR images (gallery matching)
- **Model learned:** "Make this specific DSR-HR pair similar"
- **Model didn't learn:** "Make all images of the same person similar"

## The Fix: Contrastive Learning

### Old Approach (WRONG)

1. **Loss:** ArcFace classification - Optimizes for separating classes
2. **Validation:** Paired DSR→HR similarity from same batch
3. **Problem:** Model learns to classify, not to produce consistent embeddings across images

### New Approach (CORRECT)

1. **Loss:** InfoNCE contrastive loss
   - Pulls together: DSR and HR of SAME person
   - Pushes apart: Different people
   - Ensures cross-image consistency
2. **Validation:** Gallery-probe matching
   - Build gallery from HR images (subset 1)
   - Test with DSR outputs (subset 2)
   - Measures REAL-WORLD performance
3. **Batch structure:** Multiple images per person for hard negatives

### Implementation: `finetune_contrastive.py`

```python
# Contrastive loss
def contrastive_loss(embeddings, subject_ids, temperature=0.07):
    """InfoNCE: Pull same-person pairs together, push different-person apart"""
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Positive mask: same subject, different image
    pos_mask = (subject_ids == subject_ids.T).float()
    pos_mask.fill_diagonal_(0)  # Exclude self-similarity

    # For each anchor: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    # This maximizes similarity to positives, minimizes to negatives
```

```python
# Validation: Gallery-probe matching (realistic)
def validate_gallery_matching(model, val_root, dsr_model, device):
    for subject in test_subjects:
        # Build gallery from HR images (first 3)
        gallery_embeddings = [model(hr_img) for hr_img in gallery_images]
        gallery_mean = torch.mean(gallery_embeddings, dim=0)

        # Test with DSR probes (remaining images)
        for vlr_img in probe_images:
            dsr_output = dsr_model(vlr_img)
            probe_embedding = model(dsr_output)
            similarity = cosine_similarity(gallery_mean, probe_embedding)
```

### Expected Results

- **Training similarity:** 0.6-0.8 (DSR→HR pairs)
- **Validation similarity:** 0.7-0.9 (gallery-probe matching)
- **Evaluation accuracy:** 80-95% (at 0.35 threshold)

Compare to:

- **Pretrained:** 0.987 similarity, 100% accuracy ✅
- **Old fine-tuned:** 0.177 similarity, 2% accuracy ❌
- **New fine-tuned (target):** 0.7-0.9 similarity, 80-95% accuracy

## Key Lessons

1. **Metric matters:** Training on paired samples ≠ cross-image matching
2. **Loss matters:** Classification (ArcFace) ≠ Embedding similarity (contrastive)
3. **Validation matters:** Must test realistic evaluation scenario
4. **Pretrained works:** Sometimes fine-tuning makes things worse

## Usage

### Run Contrastive Fine-Tuning

```bash
poetry run python -m technical.facial_rec.finetune_contrastive \
    --device cuda \
    --batch-size 16 \
    --images-per-subject 4 \
    --epochs 20 \
    --lr 1e-5 \
    --temperature 0.07
```

### Test Pretrained vs Fine-Tuned

```bash
# Test pretrained (baseline)
poetry run python -m technical.pipeline.test_pretrained

# After training, test contrastive fine-tuned
poetry run python -m technical.pipeline.evaluate_dataset \
    --gallery-root technical/dataset/edgeface_finetune/train \
    --probe-root technical/dataset/edgeface_finetune/val \
    --device cuda \
    --edgeface-weights technical/facial_rec/edgeface_weights/edgeface_contrastive.pth
```

## Files Modified/Created

### Analysis Scripts

- `technical/pipeline/test_gallery.py` - Debug gallery enrollment
- `technical/pipeline/test_with_arcface.py` - Test ArcFace head impact
- `technical/pipeline/test_pretrained.py` - Benchmark pretrained model
- `technical/pipeline/check_subject_overlap.py` - Verify dataset structure
- `technical/pipeline/analyze_scores.py` - Analyze similarity distributions

### New Fine-Tuning

- `technical/facial_rec/finetune_contrastive.py` - **NEW APPROACH** ✅

### Old Fine-Tuning (DO NOT USE)

- `technical/facial_rec/finetune_edgeface.py` - Classification-based (broken)
