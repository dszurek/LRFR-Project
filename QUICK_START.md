# Quick Start Commands

## ✅ Verification (Run First)

```bash
poetry run python technical/verify_setup.py
```

This checks:

- Dataset structure is correct
- Filenames follow expected format
- Models are accessible
- Architecture can be detected

---

## 🔧 Fine-Tuning Commands

### Basic Fine-Tuning (Recommended)

```bash
poetry run python -m technical.facial_rec.finetune_edgeface \
    --train-dir "technical/dataset/edgeface_finetune/train" \
    --val-dir "technical/dataset/edgeface_finetune/val" \
    --device cuda \
    --edgeface edgeface_xxs.pt
```

**Output:** `edgeface_finetuned.pth` (final model after Stage 2)

**Time:** ~11-17 hours on RTX 3080/4090

---

## 📊 Evaluation Commands

### Evaluate Baseline (Pretrained EdgeFace)

```bash
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights technical/facial_rec/edgeface_weights/edgeface_xxs.pt \
    --threshold 0.35
```

**Expected Accuracy:** 60-75%

---

### Evaluate Fine-Tuned Model

```bash
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights edgeface_finetuned.pth \
    --threshold 0.35
```

**Expected Accuracy:** 70-85% (+10-15% improvement)

---

### Quick Test (First 100 Images)

```bash
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights edgeface_finetuned.pth \
    --threshold 0.35 \
    --limit 100
```

---

### Save Detailed Results to CSV

```bash
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/frontal_only/test \
    --device cuda \
    --edgeface-weights edgeface_finetuned.pth \
    --threshold 0.35 \
    --dump-results evaluation_results.csv
```

---

## 📁 Key Files

### Datasets

- **Fine-tuning:** `technical/dataset/edgeface_finetune/` (518 subjects, 32K images)
- **Evaluation:** `technical/dataset/frontal_only/test/` (1720 subjects, 7K images)

### Models

- **Pretrained:** `technical/facial_rec/edgeface_weights/edgeface_xxs.pt` (ConvNeXt, 4.8 MB)
- **Fine-tuned:** `edgeface_finetuned.pth` (created after training)

### Documentation

- **Complete Guide:** `technical/facial_rec/FINE_TUNING_AND_EVALUATION_GUIDE.md`
- **Zero Accuracy Explained:** `technical/facial_rec/ZERO_ACCURACY_EXPLAINED.md`

---

## 🎯 What to Expect

### During Fine-Tuning

```
Stage 1 (Head Only): 10 epochs
Epoch 01/10 | Train Loss: 14.52 Acc: 0.0000 Top5: 0.0123 | Val Sim: 0.9992
    [Top-1] 5/25729 = 0.0002 | [Top-5] 317/25729 = 0.0123

Stage 2 (Full Model): 25 epochs
Epoch 25/25 | Train Loss: 8.45 Acc: 0.1250 Top5: 0.4820 | Val Sim: 0.9985
    [Top-1] 3216/25729 = 0.1250 | [Top-5] 12401/25729 = 0.4820
```

### During Evaluation

```
Building gallery from HR images ...
Registering gallery identities: 100%|███| 1720/1720 subjects

Evaluating VLR probes ...
Evaluating probes: 100%|████████████| 7233/7233 images

=== Aggregate metrics ===
Total probes evaluated : 7233
Correct predictions    : 5425 (75.00%)
Predicted as unknown   : 180 (2.49%)
```

---

## 🔍 Troubleshooting

### Check Dataset Structure

```bash
# Verify images exist
Get-ChildItem "technical/dataset/frontal_only/test/hr_images/*.png" | Measure-Object

# Check sample filenames
Get-ChildItem "technical/dataset/frontal_only/test/hr_images/*.png" |
    Select-Object -First 5 |
    ForEach-Object { $_.Name }
```

### Check Model Architecture

```bash
# Inspect checkpoint
poetry run python technical/tools/inspect_ckpt.py edgeface_finetuned.pth
```

### Adjust Recognition Threshold

```bash
# More lenient (more predictions, possibly lower precision)
--threshold 0.25

# Stricter (fewer predictions, higher precision)
--threshold 0.50
```

---

## ✅ Verification Checklist

Before running:

- [ ] Datasets exist and contain .png files
- [ ] Filenames follow `{subject}_{id}.png` format
- [ ] Pretrained model exists: `edgeface_xxs.pt`
- [ ] CUDA is available (check: `poetry run python -c "import torch; print(torch.cuda.is_available())"`)

After fine-tuning:

- [ ] `edgeface_finetuned.pth` exists
- [ ] Training completed without NaN errors
- [ ] Validation similarity > 0.995

After evaluation:

- [ ] Accuracy improved vs baseline
- [ ] Unknown rate < 5%
- [ ] No architecture detection errors in logs

---

## 📝 Notes

1. **0% Top-1 Accuracy is Normal** in early training with 518 classes

   - Random guessing: 0.193%
   - Watch Top-5 accuracy and Loss instead

2. **Subject Names from Filenames**

   - Format: `{subject}_{identifier}.png`
   - First underscore splits subject from rest
   - Example: `person001_0045.png` → subject = `person001`

3. **ConvNeXt Auto-Detection**

   - TorchScript (.pt): Loaded directly
   - State dict (.pth): Detected from keys (`stem`, `stages`)
   - Check logs for confirmation

4. **Evaluation Uses Direct HR Gallery**
   - Gallery: HR images → EdgeFace embeddings
   - Probes: VLR → DSR → EdgeFace embeddings
   - Match: Nearest neighbor in gallery

---

For detailed information, see:
**`technical/facial_rec/FINE_TUNING_AND_EVALUATION_GUIDE.md`**
