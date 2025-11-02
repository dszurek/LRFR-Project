# Quick Start: Training with 112×112 Images

## ✅ Prerequisites Complete

All HR images have been resized from 160×160 to 112×112. You're ready to train!

---

## 🚀 Training Commands (Run in Order)

### 1. Train DSR Model (~12-14 hours)

```bash
cd technical
poetry run python -m dsr.train_dsr --device cuda --epochs 100 --edgeface edgeface_xxs.pt
```

**Monitor these metrics**:

- Train PSNR: Should reach >27dB by epoch 50, >28dB by epoch 100
- Identity loss: Should drop below 0.10 by epoch 50, <0.08 by epoch 100
- Validation PSNR: Should be within 1-2dB of training PSNR

**What's different from before**:

- Using `edgeface_xxs.pt` for initial training (lightweight model)
- DSR now outputs 112×112 (was 128×128)
- Batch size increased to 16 (was 14) due to memory savings
- Base channels reduced to 120 (was 128) - optimized for 3.5× upscaling

---

### 2. Fine-tune EdgeFace on DSR Outputs (~15-18 hours)

**After DSR training completes**, run:

```bash
poetry run python -m facial_rec.finetune_edgeface --device cuda --edgeface edgeface_xxs.pt
```

**Monitor these metrics**:

- Stage 1 (5 epochs): Train accuracy should reach >70%, val >65%
- Stage 2 (25 epochs): Train accuracy >95%, val >90%
- Best checkpoint saved automatically at highest val accuracy

**What's different from before**:

- Batch size increased to 32 (was 28) due to memory savings
- No resize needed - DSR outputs 112×112 directly
- Handles both CMU and LFW file naming conventions

---

### 3. Evaluate Pipeline (~30 minutes)

```bash
poetry run python -m pipeline.evaluate_dataset \
    --dataset-root technical/dataset/test_processed \
    --threshold 0.35 \
    --device cuda
```

**Expected results**:

- Test accuracy: **60-75%** (up from 55-70%)
- Faster inference: +40% speedup from no runtime resize
- Better embeddings: EdgeFace fine-tuned on DSR outputs

---

## 📊 Expected Training Timeline

| Phase                | Duration         | Output                                               |
| -------------------- | ---------------- | ---------------------------------------------------- |
| **DSR Training**     | 12-14 hours      | `technical/dsr/dsr.pth`                              |
| **EdgeFace Stage 1** | 1.5-2 hours      | Head trained                                         |
| **EdgeFace Stage 2** | 13-16 hours      | `facial_rec/edgeface_weights/edgeface_finetuned.pth` |
| **Evaluation**       | 30 min           | Accuracy metrics                                     |
| **Total**            | **~28-33 hours** | Complete pipeline                                    |

---

## 💾 Memory Usage (RTX 3060 Ti, 8GB VRAM)

| Phase                | VRAM Usage        | Notes                   |
| -------------------- | ----------------- | ----------------------- |
| DSR Training         | ~7.0GB @ batch=16 | Can reduce to 14 if OOM |
| EdgeFace Fine-tuning | ~7.5GB @ batch=32 | Can reduce to 28 if OOM |
| Evaluation           | ~4.5GB            | No gradients needed     |

---

## ⚙️ Optional: Adjust Hyperparameters

### If DSR training is unstable (loss spikes):

```bash
poetry run python -m dsr.train_dsr --device cuda --epochs 100 --batch-size 14 --edgeface edgeface_xxs.pt
```

### If EdgeFace fine-tuning runs out of memory:

```bash
poetry run python -m facial_rec.finetune_edgeface --device cuda --edgeface edgeface_xxs.pt --stage2-epochs 25
```

Then manually edit `finetune_edgeface.py` to set `batch_size = 28`

### If you want faster training (lower accuracy):

```bash
poetry run python -m dsr.train_dsr --device cuda --epochs 50 --edgeface edgeface_xxs.pt
poetry run python -m facial_rec.finetune_edgeface --device cuda --edgeface edgeface_xxs.pt --stage2-epochs 15
```

---

## 🔍 Monitoring Training Progress

### DSR Training

Check console output for:

```
Epoch 050 | train_loss=0.0245 (L1:0.012 P:0.008 ID:0.089 FM:0.005) PSNR=28.3dB
         | val_loss=0.0267 (L1:0.014 P:0.009 ID:0.095 FM:0.006) PSNR=27.8dB
✅ Saved new best checkpoint (val PSNR 27.8dB, ID loss 0.0951)
```

**Good signs**:

- ✅ Val PSNR within 1dB of train PSNR
- ✅ Identity loss <0.10
- ✅ Regular checkpoint saves

**Warning signs**:

- ⚠️ Val PSNR >3dB lower than train → Overfitting, consider more augmentation
- ⚠️ Identity loss >0.15 → Model not preserving identity well
- ⚠️ No improvement for 20+ epochs → Early stopping will trigger

### EdgeFace Fine-tuning

Check console output for:

```
Epoch 20/25 | Train Loss: 0.1234 Acc: 0.9567 | Val Loss: 0.1456 Acc: 0.9203 | LR: 3.45e-06
  ✓ Saved checkpoint to edgeface_finetuned.pth (val acc: 0.9203)
```

**Good signs**:

- ✅ Val accuracy >90% by epoch 20
- ✅ Train/val accuracy gap <5%
- ✅ Loss steadily decreasing

**Warning signs**:

- ⚠️ Val accuracy <85% by epoch 25 → May need longer training or higher LR
- ⚠️ Train/val gap >10% → Overfitting, consider dropout or weight decay

---

## 🐛 Common Issues

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size

```bash
# DSR: Add --batch-size 14 (or 12)
# EdgeFace: Edit finetune_edgeface.py, set batch_size = 24 (or 20)
```

### Issue: "FileNotFoundError: edgeface_xxs.pt"

**Solution**: Check if file exists

```bash
ls technical/facial_rec/edgeface_weights/edgeface_xxs.pt
```

If missing, download from EdgeFace repository or use `edgeface_s_gamma_05.pt`

### Issue: Training very slow

**Check**:

1. GPU being used? (`nvidia-smi` should show Python process)
2. Batch size too small? (Increase if VRAM available)
3. Num workers = 8? (Check CPU usage)

### Issue: Model not improving after many epochs

**Early stopping** will trigger automatically after 20 epochs (DSR) or 10 epochs (EdgeFace) without improvement. This is normal and saves time!

---

## 📝 Training Logs

Training logs are printed to console. To save logs:

```bash
# DSR
cd technical
poetry run python -m dsr.train_dsr --device cuda --epochs 100 --edgeface edgeface_xxs.pt 2>&1 | tee dsr_training.log

# EdgeFace
poetry run python -m facial_rec.finetune_edgeface --device cuda --edgeface edgeface_xxs.pt 2>&1 | tee edgeface_training.log
```

---

## ✅ Verification After Training

### Check DSR Model

```bash
cd technical
poetry run python -m dsr.test_dsr
```

- Upload a test image
- Verify output is 112×112
- Check if face is recognizable

### Check EdgeFace Model

```bash
poetry run python -c "
import torch
from pathlib import Path
ckpt = torch.load('facial_rec/edgeface_weights/edgeface_finetuned.pth', map_location='cpu')
print(f\"Val Accuracy: {ckpt['val_accuracy']:.4f}\")
print(f\"Training Epoch: {ckpt['epoch']}\")
print(f\"Num Subjects: {len(ckpt['subject_to_id'])}\")
"
```

### Check Pipeline End-to-End

```bash
poetry run python -m pipeline.evaluate_dataset \
    --dataset-root technical/dataset/test_processed \
    --threshold 0.35 \
    --device cuda
```

---

## 🎯 Success Criteria

**Minimum targets** (to proceed):

- ✅ DSR PSNR >26dB on validation
- ✅ DSR Identity loss <0.12
- ✅ EdgeFace val accuracy >85%
- ✅ Pipeline test accuracy >55%

**Good targets** (competitive performance):

- ✅ DSR PSNR >28dB on validation
- ✅ DSR Identity loss <0.08
- ✅ EdgeFace val accuracy >90%
- ✅ Pipeline test accuracy >65%

**Excellent targets** (state-of-the-art):

- ✅ DSR PSNR >30dB on validation
- ✅ DSR Identity loss <0.06
- ✅ EdgeFace val accuracy >93%
- ✅ Pipeline test accuracy >75%

---

## 📖 Full Documentation

For detailed explanations, see:

- `112x112_MIGRATION_COMPLETE.md` - Complete migration guide
- `32x32_OPTIMIZATION_NOTES.md` - VLR resolution optimization
- `LFW_INTEGRATION.md` - LFW dataset integration guide

---

**Ready to train!** 🚀

Start with: `cd technical && poetry run python -m dsr.train_dsr --device cuda --epochs 100 --edgeface edgeface_xxs.pt`
