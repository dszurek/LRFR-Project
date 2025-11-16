# EdgeFace Fine-tuning Quick Reference

## TL;DR - What Was Wrong?

**Problem:** Validation accuracy stuck at 0% during fine-tuning  
**Root Cause:** Train set has subjects 001-005, val set has subjects 006-010 (different people!)  
**Fix:** Use embedding similarity as primary metric instead of classification accuracy

## Start Training Immediately

```bash
cd a:\Programming\School\cs565\project
poetry run python -m technical.facial_rec.finetune_edgeface --device cuda --edgeface edgeface_xxs.pt
```

**What to watch for:**

- `Val Sim:` should increase from ~0.70 → 0.90+
- `Val Acc:` will be 0% (this is expected with different people in val)
- Checkpoints saved when similarity improves

## Key Metrics

| Metric                   | What It Means                   | Target     | Works With Current Dataset? |
| ------------------------ | ------------------------------- | ---------- | --------------------------- |
| **Embedding Similarity** | DSR-to-HR identity preservation | > 0.85     | ✅ Yes (PRIMARY)            |
| Classification Accuracy  | Subject ID prediction           | > 0.80     | ❌ No (different people)    |
| Training Loss            | Learning progress               | Decreasing | ✅ Yes                      |

## Create Better Dataset (Optional)

If you want meaningful classification accuracy too:

```bash
# Step 1: Reorganize dataset (same people in train/val, different images)
poetry run python -m technical.facial_rec.create_finetuning_dataset

# Step 2: Train with new dataset
poetry run python -m technical.facial_rec.finetune_edgeface \
  --train-dir "technical/dataset/edgeface_finetune/train" \
  --val-dir "technical/dataset/edgeface_finetune/val" \
  --device cuda --edgeface edgeface_xxs.pt
```

## What Changed in Code?

1. ✅ **Validation now computes embedding similarity** (DSR vs HR embeddings)
2. ✅ **Checkpoints saved based on similarity** (not accuracy)
3. ✅ **ConvNeXt/TorchScript loading** (edgeface_xxs.pt works)
4. ✅ **Classification accuracy only for overlapping subjects** (prevents misleading 0%)

## Expected Output

```
STAGE 1: Training ArcFace head (backbone frozen)
====================================================================
Epoch 01/5 | Train Loss: 6.2341 Acc: 0.1234 | Val Loss: 5.8932 Acc: 0.0000 Sim: 0.7234
Epoch 02/5 | Train Loss: 5.1234 Acc: 0.3456 | Val Loss: 4.9876 Acc: 0.0000 Sim: 0.7891
  ✓ New best validation similarity: 0.7891

STAGE 2: Fine-tuning entire model (backbone unfrozen)
====================================================================
Epoch 01/25 | Train Loss: 4.5678 Acc: 0.5234 | Val Loss: 4.3456 Acc: 0.0000 Sim: 0.8456 | LR: 6.00e-06
  ✓ Saved checkpoint (val sim: 0.8456)
...
Epoch 15/25 | Train Loss: 2.1234 Acc: 0.8932 | Val Loss: 2.5678 Acc: 0.0000 Sim: 0.9234 | LR: 2.34e-06
  ✓ Saved checkpoint (val sim: 0.9234)

Training complete! Best validation similarity: 0.9234
```

## Why Is Val Accuracy 0%?

**It's correct!** Here's why:

- Train set: People 001, 002, 003, 004, 005
- Val set: People 006, 007, 008, 009, 010
- Model learns to classify: "Is this person 001? 002? 003? 004? 005?"
- During validation: "Is this person 006?" → Model: "I was never trained on person 006!"
- Result: Random guessing → ~0% accuracy

**Analogy:** Training a model to recognize cats/dogs, then testing on birds/fish. Accuracy = 0% even if the model perfectly learned cats/dogs.

**Why embedding similarity still works:**

- Measures if DSR output "looks like same person" as HR input
- Doesn't require knowing the person's name (subject ID)
- Works for any person, trained or not

## Full Documentation

- **Detailed explanation:** `FINETUNING_DATASET_RECOMMENDATIONS.md`
- **All changes made:** `FINETUNING_FIXES_SUMMARY.md`
- **Dataset creation script:** `create_finetuning_dataset.py`

## Questions?

**Q: Should I create a new dataset?**  
A: **YES, recommended!** Use `create_finetuning_dataset.py` to make `edgeface_finetune/` dataset with same people in train/val. This gives meaningful classification accuracy.

**Q: What if similarity doesn't improve?**  
A: Check DSR quality (PSNR > 30dB), EdgeFace loading (no errors), frontal faces only.

**Q: Does ConvNeXt architecture work?**  
A: Yes! `edgeface_xxs.pt` is a TorchScript file (container format) that contains ConvNeXt architecture weights. The code loads it correctly via `torch.jit.load()`.

**Q: What's the difference between TorchScript and ConvNeXt?**  
A: TorchScript = file format (like ZIP), ConvNeXt = neural network architecture (like files in the ZIP). `edgeface_xxs.pt` uses both.

**Q: How long does training take?**  
A: ~4-5 hours on GPU (Stage 1: 30min, Stage 2: 3-4hrs).

**Q: How long to create the new dataset?**  
A: ~5-10 minutes to run the script, ~30 minutes total including verification.
