# Cyclic Fine-Tuning Quick Start Guide

## What is Cyclic Fine-Tuning?

Instead of retraining DSR from scratch after fine-tuning EdgeFace, you **continue training** the existing DSR checkpoint using the fine-tuned EdgeFace as the identity supervisor. This is:

- ✅ **2-3× faster** than full retraining (30-50 epochs vs 100 epochs)
- ✅ **More stable** (preserves learned features, less variance)
- ✅ **Same or better accuracy** (+8-15% improvement expected)
- ✅ **Cleaner** (fewer checkpoints to manage)

## Quick Commands

### Option 1: Automated Pipeline (Recommended)

Train all resolutions with one command:

```bash
# Full pipeline: Initial DSR → EdgeFace FT → DSR Cyclic FT
python -m technical.tools.cyclic_train --device cuda

# Skip initial training (if already done)
python -m technical.tools.cyclic_train --skip-initial --skip-edgeface --device cuda

# Train only specific resolution
python -m technical.tools.cyclic_train --vlr-sizes 32 --device cuda

# Dry run (see commands without executing)
python -m technical.tools.cyclic_train --dry-run
```

### Option 2: Manual Step-by-Step

```bash
# Step 1: Initial DSR training (if not already done)
python -m technical.dsr.train_dsr --vlr-size 32 --device cuda

# Step 2: EdgeFace fine-tuning (if not already done)
python -m technical.facial_rec.finetune_edgeface --vlr-size 32 --device cuda

# Step 3: DSR cyclic fine-tuning (NEW!)
python -m technical.dsr.train_dsr \
    --vlr-size 32 \
    --device cuda \
    --resume technical/dsr/dsr32.pth \
    --edgeface edgeface_finetuned_32.pth \
    --epochs 50 \
    --learning-rate 8e-5 \
    --lambda-identity 0.65 \
    --lambda-feature-match 0.20
```

## Key Parameters for Cyclic Training

| Parameter                | Initial Training  | Cyclic Training              | Why Different?                   |
| ------------------------ | ----------------- | ---------------------------- | -------------------------------- |
| `--epochs`               | 100               | 50                           | Already converged, just refining |
| `--learning-rate`        | 1.5e-4 (default)  | 8e-5                         | Lower LR for fine-tuning         |
| `--lambda-identity`      | 0.50 (default)    | 0.65                         | Trust fine-tuned EdgeFace more   |
| `--lambda-feature-match` | 0.15 (default)    | 0.20                         | Emphasize feature alignment      |
| `--resume`               | (not used)        | `dsr{N}.pth`                 | Continue from checkpoint         |
| `--edgeface`             | `edgeface_xxs.pt` | `edgeface_finetuned_{N}.pth` | Use fine-tuned model             |

## Resolution-Specific Hyperparameters

### 16×16 (Most Challenging)

```bash
--epochs 55 --learning-rate 9e-5 --lambda-identity 0.68 --lambda-feature-match 0.22
```

- Higher identity weight (more degraded input needs stronger supervision)
- Slightly longer training (harder to converge)

### 24×24 (Medium)

```bash
--epochs 50 --learning-rate 8.5e-5 --lambda-identity 0.65 --lambda-feature-match 0.20
```

- Balanced hyperparameters

### 32×32 (Best Quality)

```bash
--epochs 50 --learning-rate 8e-5 --lambda-identity 0.65 --lambda-feature-match 0.20
```

- Standard hyperparameters (easiest resolution)

## Expected Training Time

| Resolution | Initial DSR | EdgeFace FT | DSR Cyclic | Total Cycle |
| ---------- | ----------- | ----------- | ---------- | ----------- |
| 16×16      | ~10h        | ~3h         | ~5.5h      | ~18.5h      |
| 24×24      | ~10h        | ~3h         | ~5h        | ~18h        |
| 32×32      | ~10h        | ~3h         | ~5h        | ~18h        |

**All 3 resolutions**: ~54 hours sequential, ~18 hours parallel (3 GPUs)

## Expected Accuracy Improvement

| Resolution | Baseline (No Cycle) | After Cycle 1 | Improvement |
| ---------- | ------------------- | ------------- | ----------- |
| 16×16      | ~30-35%             | ~38-45%       | **+8-10%**  |
| 24×24      | ~35-42%             | ~43-52%       | **+8-10%**  |
| 32×32      | ~40-48%             | ~48-58%       | **+8-10%**  |

## What to Watch During Training

### Good Signs ✅

- Identity loss decreasing to ~0.12-0.18 (was ~0.20-0.25)
- Feature match loss decreasing to <0.05
- PSNR maintained or improving (>28 dB)
- Val loss tracking train loss (not diverging)

### Warning Signs ⚠️

- Identity loss stuck >0.25 → increase `--lambda-identity` to 0.70
- PSNR decreasing >1 dB → reduce identity weight, increase L1 weight
- Val loss >> train loss → overfitting; reduce epochs or add regularization

## After Training: Evaluation

```bash
# GUI evaluation (all resolutions, publication-quality plots)
python -m technical.pipeline.evaluate_gui

# Or individual resolution
python -m technical.pipeline.evaluate_verification \
    --vlr-size 32 \
    --device cuda
```

## Troubleshooting

### "FileNotFoundError: dsr32.pth not found"

**Problem**: Trying to resume from non-existent checkpoint

**Solution**: Run initial training first, or remove `--resume` flag

### "CUDA out of memory" during cyclic training

**Problem**: Model + fine-tuned EdgeFace too large for GPU

**Solutions**:

```bash
# Reduce batch size
--batch-size 12

# Or train on CPU (much slower)
--device cpu
```

### Identity loss not improving (<0.02 decrease)

**Problem**: Learning rate too low, or already at optimal weights

**Solutions**:

- Try slightly higher LR: `--learning-rate 1.2e-4`
- Or accept current weights (diminishing returns)
- Check that fine-tuned EdgeFace actually loaded (look for log message)

### Accuracy worse after cyclic training

**Problem**: Overfitting to fine-tuned EdgeFace quirks

**Solutions**:

- Revert to pre-cyclic checkpoint (backup recommended)
- Reduce identity weight: `--lambda-identity 0.50`
- Reduce epochs: `--epochs 30`
- Check EdgeFace fine-tuning quality (may need to retrain EdgeFace with different settings)

## Advanced: Second Cycle (Optional)

If Cycle 1 gave >10% improvement, you can try a second cycle:

```bash
# Re-fine-tune EdgeFace on cycle-trained DSR outputs
python -m technical.facial_rec.finetune_edgeface \
    --vlr-size 32 \
    --device cuda \
    --stage1-epochs 5 \
    --stage2-epochs 15

# Then fine-tune DSR again
python -m technical.dsr.train_dsr \
    --vlr-size 32 \
    --device cuda \
    --resume technical/dsr/dsr32.pth \
    --edgeface edgeface_finetuned_32.pth \
    --epochs 30 \
    --learning-rate 5e-5 \
    --lambda-identity 0.70
```

**Expected gain**: +2-5% additional improvement

**Warning**: Diminishing returns, and risk of mode collapse beyond Cycle 2

## Files Modified

- ✅ `technical/dsr/train_dsr.py`: Added `--resume`, `--learning-rate`, `--lambda-*` CLI args
- ✅ `technical/tools/cyclic_train.py`: NEW automated pipeline script
- ✅ `technical/CYCLIC_VS_RETRAINING_ANALYSIS.md`: Full theoretical analysis
- ✅ `technical/CYCLIC_TRAINING_QUICKSTART.md`: This guide

## See Also

- **Full Analysis**: `technical/CYCLIC_VS_RETRAINING_ANALYSIS.md`
- **DSR Cycle Guide**: `technical/DSR_CYCLE_TRAINING_GUIDE.md` (original guide, still valid)
- **Multi-Resolution Guide**: `technical/MULTI_RESOLUTION_GUIDE.md`
- **Evaluation Guide**: `technical/EVALUATION_GUIDE.md`

---

**Ready to start?** Run:

```bash
python -m technical.tools.cyclic_train --device cuda
```

And let it cook! ☕🚀
