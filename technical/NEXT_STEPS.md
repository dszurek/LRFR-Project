# 🎯 Next Steps to Improve Your Results

## Summary of Your Current Situation

**Current Performance:**

- Accuracy: 20.65% (with gallery-via-dsr + threshold 0.30)
- Accuracy: 23.82% (without gallery-via-dsr + threshold 0.45)
- Unknown predictions: ~13-25%
- High-confidence misidentifications (e.g., subject 007 → 200 with 0.97 score)

**Root Cause:**
Your DSR model is trained with loss weights that prioritize visual quality over identity preservation. This causes it to lose subtle facial features that EdgeFace needs for recognition.

---

## ✅ Changes Already Made (Ready to Use)

### 1. Improved Training Script (`train_dsr.py`)

- **Identity loss increased 3.5×**: `0.1 → 0.35`
- **Model capacity increased**: `base_channels: 96 → 112` (~36% more parameters)
- **Better learning rate schedule**: Warmup + smoother cosine annealing
- **Early stopping**: Prevents overfitting (stops after 15 epochs without improvement)
- **Conservative augmentation**: Less aggressive transforms to preserve facial features

### 2. Improved Pipeline (`pipeline.py`)

- **Better default threshold**: `0.45 → 0.35`
- **Test-Time Augmentation support**: Average predictions with horizontal flip

### 3. Helper Tools Created

- `TRAINING_IMPROVEMENTS.md`: Comprehensive guide
- `tools/threshold_sweep.py`: Test multiple thresholds quickly
- `tools/compare_models.py`: Visually compare old vs new models

---

## 🚀 What You Need to Do Now

### Step 1: Backup Your Current Model (Optional but Recommended)

```bash
cd A:\Programming\School\cs565\project\technical
cp dsr/dsr.pth dsr/dsr_old.pth
```

### Step 2: Retrain with Improved Settings ⭐ MOST IMPORTANT

```bash
cd A:\Programming\School\cs565\project\technical

poetry run python -m dsr.train_dsr \
    --device cuda \
    --edgeface edgeface_xxs_q.pt \
    --epochs 80 \
    --batch-size 8
```

**Expected time**: 6-8 hours on RTX 3060 Ti (may finish earlier with early stopping)

**What to watch for:**

- Validation PSNR should steadily increase
- Training will auto-stop if no improvement for 15 epochs
- Best model saved to `technical/dsr/dsr.pth`

### Step 3: Evaluate the New Model

```bash
cd A:\Programming\School\cs565\project

# Test with optimal threshold
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/test_processed \
    --threshold 0.35 \
    --device cuda
```

**Expected improvement**: 35-45% accuracy (up from 23.82%)

### Step 4: Fine-Tune Threshold (If Needed)

```bash
cd A:\Programming\School\cs565\project\technical

# Run quick threshold sweep
poetry run python tools/threshold_sweep.py
```

This will test thresholds [0.25, 0.30, 0.35, 0.40, 0.45] on 2000 samples and help you find the optimal value.

### Step 5: Try Test-Time Augmentation (Optional Boost)

Once you have the best threshold, test with TTA for an extra 2-5% accuracy:

```bash
# You'll need to add a --use-tta flag to evaluate_dataset.py
# Or manually set config.use_tta = True in the script
```

---

## 📊 Expected Results Timeline

| Stage               | Accuracy | Notes                               |
| ------------------- | -------- | ----------------------------------- |
| Current (baseline)  | 23.82%   | Your current model                  |
| After retraining    | 35-45%   | Main improvement from identity loss |
| + Optimal threshold | 38-48%   | Fewer unknowns                      |
| + TTA               | 40-50%   | Extra boost from augmentation       |

---

## 🔍 If Results Don't Improve Enough

### Diagnostic Steps

1. **Check if model actually trained:**

   ```bash
   poetry run python -c "
   import torch
   ckpt = torch.load('technical/dsr/dsr.pth', map_location='cpu')
   print(f'Epoch: {ckpt[\"epoch\"]}')
   print(f'Val PSNR: {ckpt.get(\"val_psnr\", 0):.2f} dB')
   print(f'Lambda identity: {ckpt[\"config\"][\"lambda_identity\"]}')
   print(f'Base channels: {ckpt[\"config\"].get(\"base_channels\", \"N/A\")}')
   "
   ```

   Should show:

   - `Lambda identity: 0.35` (NOT 0.1)
   - Val PSNR > 25 dB (higher is better)
   - Epoch between 20-80 (depending on early stopping)

2. **Visually inspect SR quality:**

   ```bash
   cd technical
   poetry run python -m dsr.test_dsr
   ```

   Look for:

   - Are faces recognizable?
   - Too blurry → increase identity loss to 0.5
   - Artifacts → reduce TV loss

3. **Compare old vs new models:**
   ```bash
   poetry run python tools/compare_models.py \
       dsr/dsr_old.pth \
       dsr/dsr.pth \
       dataset/test_processed/vlr_images/007_01_01_010_00_crop_128.png
   ```

### Additional Improvements (If Still Not Satisfied)

1. **Increase identity loss further:**

   - Edit `train_dsr.py`: Change `lambda_identity: float = 0.35` to `0.5`
   - Retrain

2. **Train longer:**

   ```bash
   poetry run python -m dsr.train_dsr --device cuda --epochs 100 --batch-size 8
   ```

3. **Increase model capacity:**

   - Edit `train_dsr.py`: Change `base_channels=112` to `128`
   - Warning: Slower training (~10-12 hours)

4. **Use a larger EdgeFace model** (if available):
   - Try `edgeface_s.pt` or `edgeface_m.pt` instead of `edgeface_xxs_q.pt`
   - Both for training and inference

---

## 🐛 Troubleshooting

### Training crashes or OOM errors:

```bash
# Reduce batch size
poetry run python -m dsr.train_dsr --device cuda --epochs 80 --batch-size 4
```

### Training is too slow:

```bash
# Reduce workers or check GPU utilization
poetry run python -m dsr.train_dsr --device cuda --epochs 80 --batch-size 8
# (The script uses num_workers=8 by default)
```

### Model not loading after training:

```bash
# Check checkpoint structure
poetry run python -c "
import torch
ckpt = torch.load('technical/dsr/dsr.pth', map_location='cpu')
print('Checkpoint keys:', list(ckpt.keys()))
print('Config:', ckpt.get('config'))
"
```

### Evaluation shows same results as before:

- Make sure you're using the NEW checkpoint, not the old one
- Check that `technical/dsr/dsr.pth` was actually replaced
- Verify the config in the checkpoint shows `lambda_identity: 0.35`

---

## 📝 Quick Command Reference

```bash
# 1. Retrain model (MOST IMPORTANT)
cd A:\Programming\School\cs565\project\technical
poetry run python -m dsr.train_dsr --device cuda --epochs 80 --batch-size 8

# 2. Evaluate
cd A:\Programming\School\cs565\project
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/test_processed \
    --threshold 0.35 \
    --device cuda

# 3. Visual check
cd A:\Programming\School\cs565\project\technical
poetry run python -m dsr.test_dsr

# 4. Threshold sweep
poetry run python tools/threshold_sweep.py

# 5. Compare models
poetry run python tools/compare_models.py \
    dsr/dsr_old.pth \
    dsr/dsr.pth \
    dataset/test_processed/vlr_images/007_01_01_010_00_crop_128.png
```

---

## 💡 Key Insight

Your model is creating visually nice images but losing the specific facial features EdgeFace uses for recognition. The new training emphasizes **identity preservation over visual quality**, which is exactly what you need for a recognition pipeline.

Think of it this way:

- Old model: "Make pretty faces" → EdgeFace confused
- New model: "Keep the exact facial features EdgeFace cares about" → Better recognition

---

## Questions?

If results still don't improve after retraining:

1. Share the training logs (especially val PSNR progression)
2. Share the new evaluation metrics
3. Share a few sample SR images from `test_dsr.py`

Good luck! 🚀
