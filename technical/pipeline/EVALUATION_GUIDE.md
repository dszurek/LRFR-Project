# Pipeline Evaluation Guide

## Overview

The `evaluate_dataset.py` script now supports **two evaluation modes** to match different use cases:

### Mode 1: Single-Dataset Evaluation (Original)

Gallery and probes from the same dataset (useful for sanity checks).

### Mode 2: Split-Dataset Evaluation (Real-World)

Gallery from one dataset (known identities), probes from another (unseen images).

## Real-World Evaluation Setup

This matches how end users would deploy the system:

1. **Gallery Enrollment** (Registration Phase)

   - Use HR images from training data (`edgeface_finetune/train`)
   - These are the "known identities" the system can recognize
   - In production: users would upload 3-5 photos of themselves

2. **Probe Testing** (Recognition Phase)
   - Use VLR images from validation data (`edgeface_finetune/val`)
   - Same people as gallery, but different (unseen) images
   - In production: users upload a single low-res photo to identify

## Usage Examples

### Test on Known Identities (Recommended)

```bash
poetry run python -m technical.pipeline.evaluate_dataset \
    --gallery-root technical/dataset/edgeface_finetune/train \
    --probe-root technical/dataset/edgeface_finetune/val \
    --device cuda \
    --edgeface-weights technical/facial_rec/edgeface_weights/edgeface_finetuned.pth \
    --threshold 0.35
```

**What this tests:**

- Gallery: 518 known identities (from training)
- Probes: Same 518 people, but unseen VLR images
- **Expected accuracy**: 40-50%+ (model knows these people)

### Test on Unknown Identities (Zero-Shot)

```bash
poetry run python -m technical.pipeline.evaluate_dataset \
    --gallery-root technical/dataset/edgeface_finetune/train \
    --probe-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights technical/facial_rec/edgeface_weights/edgeface_finetuned.pth \
    --threshold 0.35
```

**What this tests:**

- Gallery: 518 known identities (from training)
- Probes: 1,720 DIFFERENT people (never seen before)
- **Expected accuracy**: 0-5% (zero-shot recognition - very hard!)
- **Expected behavior**: Most should be "rejected as unknown"

## Output Interpretation

### Test Set Composition

```
Total test subjects:    518
Known (in gallery):     518 (100.0%)
Unknown (not in gallery): 0 (0.0%)
```

Shows how many test subjects the model has seen during training.

### Aggregate Metrics

```
Total probes evaluated : 6677
Correct predictions    : 2800 (41.9%)
Predicted as unknown : 50 (0.7%)
```

- **Correct predictions**: Probe matched to correct gallery identity
- **Predicted as unknown**: Similarity below threshold (rejected)

### Known vs Unknown Performance

```
Known subjects (in gallery):
  Probes: 6677, Correct: 2800 (41.9%)

Unknown subjects (NOT in gallery):
  Probes: 0, Correct: 0 (0.00%)
  Correctly rejected as unknown: 0 (0.00%)
```

Separates performance on enrolled identities vs novel identities.

### Per-Subject Accuracy

```
  2870 |  45/106 | 42.45% (in gallery)
  2047 |  18/47  | 38.30% (in gallery)
   209 |  15/39  | 38.46% (in gallery)
```

Shows which subjects are easiest/hardest to recognize.

## Current Results Analysis

### Run 1: Gallery=train, Probes=val (Known Identities)

```
Accuracy: 1.92%
```

**❌ This is unexpectedly low!** Should be 40-50%+.

**Possible issues:**

1. Threshold too high/low
2. Gallery enrollment using wrong images
3. Subject ID extraction mismatch
4. Model not actually using fine-tuned weights

### Run 2: Gallery=train, Probes=test (Unknown Identities)

```
Accuracy: 0.00%
Known: 0/1720 subjects
Correctly rejected: 17/7233 (0.24%)
```

**✅ This is expected!** Test subjects are completely different people.

**Good behavior:**

- 0% accuracy (model doesn't know these people)
- Only 0.24% false positives (good threshold)

## Debugging Low Accuracy

If accuracy is unexpectedly low on known identities:

### 1. Check Subject ID Overlap

```bash
# Training subjects
ls technical/dataset/edgeface_finetune/train/hr_images/*.png | head -5

# Validation subjects
ls technical/dataset/edgeface_finetune/val/vlr_images/*.png | head -5
```

Ensure filenames use same subject ID format (e.g., both use "001" not "1" vs "001").

### 2. Verify Model is Fine-Tuned

Check if model is actually loading fine-tuned weights:

```
[EdgeFace] Loaded fine-tuned checkpoint (backbone_state_dict)
```

### 3. Test with Baseline Model

Compare fine-tuned vs pretrained:

```bash
# Pretrained
poetry run python -m technical.pipeline.evaluate_dataset \
    --gallery-root technical/dataset/edgeface_finetune/train \
    --probe-root technical/dataset/edgeface_finetune/val \
    --device cuda \
    --threshold 0.35

# Fine-tuned (should be BETTER)
poetry run python -m technical.pipeline.evaluate_dataset \
    --gallery-root technical/dataset/edgeface_finetune/train \
    --probe-root technical/dataset/edgeface_finetune/val \
    --device cuda \
    --edgeface-weights technical/facial_rec/edgeface_weights/edgeface_finetuned.pth \
    --threshold 0.35
```

### 4. Adjust Threshold

Try different thresholds to see sensitivity:

```bash
# Lower (more permissive)
--threshold 0.25

# Higher (more strict)
--threshold 0.45
```

## Expected Performance Targets

| Scenario               | Gallery  | Probes    | Expected Accuracy  |
| ---------------------- | -------- | --------- | ------------------ |
| **Same-dataset**       | train HR | train VLR | 60-70% (best case) |
| **Known identities**   | train HR | val VLR   | 40-50% (realistic) |
| **Unknown identities** | train HR | test VLR  | 0-5% (zero-shot)   |

## Next Steps

1. **Debug low accuracy** on known identities (should be 40-50%, currently 1.92%)
2. **Find optimal threshold** via sweep (0.20 to 0.50 in steps of 0.05)
3. **Compare pretrained vs fine-tuned** to verify improvement
4. **Generate detailed CSV** output for error analysis (`--dump-results results.csv`)
