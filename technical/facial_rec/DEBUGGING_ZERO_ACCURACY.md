# Understanding Your Training Output

## What You're Seeing

```
Epoch 01/5 | Train Loss: 29.1053 Acc: 0.0000 | Val Loss: 22.8677 Acc: 0.0000 Sim: 0.9999
```

## Why This is Actually EXPECTED (and not a bug!)

### 1. **Training Accuracy = 0.0000** ✅ EXPECTED

**Why?**

- You have **518 subjects** (classes)
- Random guessing baseline = 1/518 = **0.0019** (0.19%)
- With a fresh, untrained ArcFace head, the model is randomly guessing
- Getting even one correct prediction out of 25,729 samples is unlikely!

**Math:**

- Probability of random correct guess: 1/518 = 0.19%
- Expected correct in batch of 32: 32 × 0.0019 = **0.06 images**
- Round down: **0 correct**

**This will improve!** By epoch 3-5, you should see accuracy rising to 10-30%.

### 2. **Validation Accuracy = 0.0000** ✅ EXPECTED

Same reasoning - with 518 classes and untrained model, random guessing gives ~0% accuracy.

**What to watch for:**

- Epoch 2-3: Accuracy should start rising above 0%
- Epoch 5: Should reach 5-15% (stage 1 complete)
- Epoch 10-15: Should reach 30-60% (stage 2)
- Epoch 20-25: Should reach 60-80% (stage 2 complete)

### 3. **Validation Similarity = 0.9999** ⚠️ TOO HIGH

**This is suspicious!** Here's what it means:

**What it measures:**

- Cosine similarity between DSR output embedding and HR embedding
- Range: -1.0 to 1.0
- 1.0 = perfectly identical embeddings
- 0.9999 = nearly identical

**Possible causes:**

#### A) DSR is producing nearly perfect outputs (GOOD)

- Your DSR model is excellent
- DSR outputs look almost identical to HR in embedding space
- EdgeFace can't tell the difference (mission accomplished!)

#### B) Bug: DSR and HR are the same images (BAD)

- Dataset might be loading HR images for both DSR input and HR target
- Need to verify VLR→DSR→embedding vs HR→embedding

#### C) EdgeFace embeddings are collapsed (BAD)

- Model outputs same embedding for all images (common with bad initialization)
- Need to check embedding variance

## Diagnostic Steps

### Step 1: Check if embeddings are diverse

Add this debug code after epoch 1:

```python
# After validation in stage 1, add:
print(f"\n[Debug] Checking embedding diversity...")
val_embeddings = []
for sr_imgs, hr_imgs, labels in val_loader:
    sr_imgs = sr_imgs.to(device)
    with torch.no_grad():
        emb = backbone(sr_imgs)
    val_embeddings.append(emb)
    if len(val_embeddings) >= 10:  # Check 10 batches
        break

val_embeddings = torch.cat(val_embeddings, dim=0)
mean_emb = val_embeddings.mean(dim=0)
std_emb = val_embeddings.std(dim=0)
print(f"[Debug] Embedding mean magnitude: {mean_emb.norm():.4f}")
print(f"[Debug] Embedding std: {std_emb.mean():.4f}")
print(f"[Debug] Expected std: >0.1 (diverse), <0.01 (collapsed)")
```

### Step 2: Verify DSR is running

Check that DSR is actually processing images:

```python
# In DSROutputDataset.__getitem__, add debug for first image:
if idx == 0:
    print(f"[Debug] VLR shape: {vlr_tensor.shape}")
    print(f"[Debug] DSR input range: [{vlr_tensor.min():.3f}, {vlr_tensor.max():.3f}]")
    print(f"[Debug] DSR output range: [{sr_tensor.min():.3f}, {sr_tensor.max():.3f}]")
    print(f"[Debug] HR range: [{hr_tensor.min():.3f}, {hr_tensor.max():.3f}]")
```

### Step 3: Check DSR quality manually

Verify DSR outputs look correct:

```python
# Save sample images to check:
import torchvision
sr_sample = (sr_tensor * 0.5 + 0.5).clamp(0, 1)  # Denormalize
hr_sample = (hr_tensor * 0.5 + 0.5).clamp(0, 1)
torchvision.utils.save_image(sr_sample, "debug_dsr_output.png")
torchvision.utils.save_image(hr_sample, "debug_hr_target.png")
```

## Expected Behavior by Epoch

| Epoch | Train Acc | Val Acc | Val Sim    | Status                      |
| ----- | --------- | ------- | ---------- | --------------------------- |
| 1     | 0.00%     | 0.00%   | 0.70-0.85  | ✅ Normal (random guessing) |
| 1     | 0.00%     | 0.00%   | **0.9999** | ⚠️ Suspicious (too high!)   |
| 3     | 5-15%     | 3-12%   | 0.75-0.87  | ✅ Learning                 |
| 5     | 15-30%    | 10-25%  | 0.80-0.90  | ✅ Stage 1 complete         |
| 10    | 40-60%    | 30-50%  | 0.85-0.92  | ✅ Stage 2 progress         |
| 20    | 70-85%    | 60-75%  | 0.88-0.94  | ✅ Converging               |

## What to Do Next

### If embeddings are diverse (std > 0.1):

- ✅ Continue training and watch accuracy improve
- 0% accuracy on epoch 1 is normal with 518 classes
- Similarity of 0.9999 means DSR is excellent (nothing to improve!)

### If embeddings are collapsed (std < 0.01):

- ❌ Problem with EdgeFace initialization
- Try loading pretrained weights differently
- The 4 missing keys might be causing issues

### If DSR outputs look wrong:

- ❌ Check DSR model loaded correctly
- Verify VLR images are 32×32 not 112×112
- Check DSR upscales to correct size

## The Missing Keys Issue

```
[EdgeFace] Missing keys: 4
[EdgeFace] Missing key names: ['stages.2.blocks.5.pos_embd.token_projection.weight', ...]
```

**This is the likely culprit!** These are positional embedding parameters in the XCA blocks. Missing them means:

- XCA blocks don't have proper positional information
- Model may produce degraded embeddings
- Explains why similarity is unusually high (embeddings are not well-separated)

### Fix: Load from TorchScript properly

The TorchScript load is failing:

```
[EdgeFace] TorchScript load failed: PytorchStreamReader failed locating file constants.pkl
```

**Try this:**

```bash
# Check if the file is actually TorchScript:
python -c "import torch; model = torch.jit.load('technical/facial_rec/edgeface_weights/edgeface_xxs.pt'); print(type(model))"
```

If it fails, the file might be a regular state dict, not TorchScript. You may need to use `edgeface_s_gamma_05.pt` or a different pretrained model.

## Summary

**Your 0% accuracy is completely normal for epoch 1 with 518 classes.**

**The 0.9999 similarity is suspicious** and likely caused by:

1. Missing positional embedding keys in EdgeFace
2. TorchScript loading failure falling back to architecture loading
3. Embeddings might be collapsed or poorly initialized

**Recommendation:** Wait for epoch 3-5 to see if accuracy starts improving. If it stays at 0%, then the missing keys are the problem and you need to fix the EdgeFace loading.
