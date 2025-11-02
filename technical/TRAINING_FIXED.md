# Training Commands - UPDATED for 112×112

## Issue Resolved ✅

The old `dsr.pth` was trained for 160×160 output. It has been backed up to `dsr_160x160_backup.pth`.

---

## Training Command (Corrected)

### Train DSR Model (~12-14 hours)

```bash
cd technical
poetry run python -m dsr.train_dsr --device cuda --epochs 100 --edgeface edgeface_s_gamma_05.pt
```

**Why `edgeface_s_gamma_05.pt` instead of `edgeface_xxs.pt`?**

- `edgeface_xxs.pt` has a different architecture (ConvNeXt-style) that doesn't match the EdgeFace class
- `edgeface_s_gamma_05.pt` matches the `edgeface_s` architecture used in the code
- This will provide proper identity loss and feature matching during DSR training

---

## Expected Training Output

```
Using device: cuda
Loading EdgeFace from: .../edgeface_s_gamma_05.pt
Target HR resolution: 112×112 (3.5× upscaling from 32×32 VLR)
Training samples: 111568, validation samples: 14437

Epoch 001 | train_loss=0.1234 ... PSNR=24.5dB
         | val_loss=0.1456 ... PSNR=23.8dB
✅ Saved new best checkpoint (val PSNR 23.8dB)
```

---

## After DSR Training

### Fine-tune EdgeFace (~15-18 hours)

```bash
poetry run python -m facial_rec.finetune_edgeface --device cuda --edgeface edgeface_s_gamma_05.pt
```

This will use the newly trained 112×112 DSR model automatically.

---

## Files Status

- ✅ All HR images: 112×112 (142,713 images resized)
- ✅ Old DSR model: Backed up to `dsr_160x160_backup.pth`
- ✅ Training scripts: Updated for 112×112 output
- ✅ Pipeline: Updated to handle 112×112 directly

---

## Quick Verification

After training starts successfully, you should see:

- No size mismatch errors
- PSNR starting around 20-25dB in first few epochs
- Loss decreasing steadily

If you see "size mismatch" errors, the old checkpoint might still be loading. Delete `dsr.pth` completely before training.
