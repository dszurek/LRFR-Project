# Evaluation Metrics - Sanity Check & Analysis

## Current Metrics Review

### ✅ **CORRECT Metrics**

#### 1. **PSNR (Peak Signal-to-Noise Ratio)**

- **Computation**: Correct - comparing SR output vs HR ground truth
- **Range**: Typically 20-40 dB for face SR
- **Current values**: 27-34 dB ✓ (reasonable for 16-32px input)

#### 2. **SSIM (Structural Similarity Index)**

- **Computation**: Correct - perceptual similarity metric
- **Range**: 0-1, higher is better
- **Current values**: 0.69-0.92 ✓ (good progression with resolution)

---

### ⚠️ **POTENTIALLY MISLEADING Metrics**

#### 3. **Identity Similarity**

**Current**: Compares SR output embedding vs HR ground truth embedding of **SAME IMAGE**

```python
# Current code (Lines 342-349)
sr_emb = edgeface_model(sr_norm)
hr_emb = edgeface_model(hr_norm)
sim = torch.nn.functional.cosine_similarity(sr_emb, hr_emb, dim=1).item()
```

**Issues**:

- ✅ **Measures**: How well DSR preserves identity features
- ❌ **Does NOT measure**: Cross-pose/cross-expression identity preservation
- ❌ **Misleading name**: "Identity similarity" implies identity preservation across different images
- **Current values**: 97-99% (artificially high because it's the same image before/after SR)

**Recommendation**:

- Rename to `"feature_preservation"` or `"reconstruction_fidelity"`
- Add true **cross-image identity similarity** (different poses of same person)

---

#### 4. **Verification Metrics (EER, TAR@FAR)**

**Current Impostor Sampling** (Lines 413-419):

```python
# Only samples ~20 impostors per subject
for j in range(min(i + 1, len(subjects)), min(i + 20, len(subjects))):
    subject_j = subjects[j]
    embs_j = embeddings_by_subject[subject_j]
    if len(embs_i) > 0 and len(embs_j) > 0:
        score = torch.cosine_similarity(embs_i[0], embs_j[0], dim=1).item()
        impostor_scores.append(score)
```

**Issues**:

- ✅ **Genuine pairs**: All intra-class pairs (correct)
- ⚠️ **Impostor pairs**: Only ~20 sequential subjects per identity
- **Problem**: Biased impostor sampling - sequential subjects may be correlated
- **Impact**: EER may be **optimistic** (missing hard negatives)

**Recommendations**:

1. Sample impostors **randomly** (not sequentially)
2. Increase to at least 100 impostor pairs per subject
3. Or use stratified sampling to ensure diversity

---

#### 5. **1:1 Identification Accuracy**

**Current** (Lines 535-538):

```python
# 1:1 Matching - Match against only the true identity
true_sim = torch.cosine_similarity(probe_emb, gallery_templates[subject_id], dim=1).item()
results_1v1["rank1"] += 1  # Always rank-1 in 1:1
```

**Issues**:

- ❌ **Always counts as correct** - no threshold check!
- **This is NOT 1:1 matching** - it's just computing similarity with the correct template
- **Will always report 100%** regardless of actual performance

**Fix**: Apply a threshold and count as correct only if similarity > threshold

---

### ❌ **MISSING Standard LRFR Metrics**

Based on recent low-resolution face recognition papers (ECCV/CVPR 2023-2024):

#### 6. **True Accept Rate at Multiple FAR Points**

- **Current**: TAR @ FAR=0.1%, 1%
- **Missing**: TAR @ FAR=0.01%, 10% (commonly reported)

#### 7. **Detection Rate at Different Resolutions**

- Face detection success rate at 16×16, 24×24, 32×32
- Important baseline metric (can't recognize if can't detect)

#### 8. **Cross-Resolution Matching**

- VLR probe vs different resolution gallery
- E.g., 16×16 probe vs 32×32 gallery enrollment

#### 9. **Cumulative Match Characteristic (CMC) Curve**

- Full rank accuracy curve (Rank-1 to Rank-100)
- Standard in identification tasks

#### 10. **d-prime (d')**

- Separability metric between genuine and impostor distributions
- Formula: `d' = (μ_genuine - μ_impostor) / sqrt(0.5 * (σ²_genuine + σ²_impostor))`
- **Higher is better** (measures how well distributions separate)

#### 11. **Decidability Index**

- Similar to d-prime but uses different variance calculation
- Common in biometrics literature

#### 12. **Mean Average Precision (mAP)**

- For open-set identification scenarios
- Measures retrieval quality

#### 13. **Area Under CMC (AUCMC)**

- Summarizes entire CMC curve into single metric
- Useful for comparing different resolutions

---

## Priority Fixes

### HIGH PRIORITY (Misleading Results)

1. **Fix 1:1 Identification** - Currently reports 100% always

   ```python
   # Should be:
   if true_sim >= threshold:
       results_1v1["rank1"] += 1
   ```

2. **Rename "Identity Similarity"** → "Feature Preservation"

   - Add true cross-image identity similarity

3. **Fix Impostor Sampling**
   - Random sampling instead of sequential
   - Increase sample size

### MEDIUM PRIORITY (Important Additions)

4. **Add d-prime metric**

   - Easy to compute from existing genuine/impostor scores
   - Highly interpretable

5. **Add full CMC curve**

   - Already computing ranks, just need to save full curve

6. **Add TAR@FAR=0.01%**
   - Common in high-security applications

### LOW PRIORITY (Nice to Have)

7. **Face detection rate**

   - Requires additional face detector

8. **Cross-resolution matching**
   - Interesting research contribution but complex

---

## Recommended Metric Set

### For Publication (Comprehensive):

1. **Image Quality**: PSNR, SSIM, Feature Preservation
2. **Verification**: EER, d-prime, ROC AUC, TAR@FAR=[0.01%, 0.1%, 1%, 10%]
3. **Identification**: CMC curve (Rank 1-20), AUCMC, separate results for 1:10, 1:100, 1:N
4. **Distribution**: Genuine/Impostor mean ± std, overlap visualization

### For Quick Comparison:

1. **PSNR**, **SSIM**
2. **EER**, **d-prime**
3. **Rank-1@1:100**, **Rank-1@1:N**

---

## Comparison with SOTA

### Typical LRFR Paper Metrics (CVPR/ICCV):

- ✅ PSNR, SSIM
- ✅ Rank-1/5/10 accuracy
- ✅ Verification TAR@FAR
- ⚠️ **d-prime** (we should add)
- ⚠️ **Detection rate** (optional)
- ⚠️ **CMC curve visualization** (we should add)

### Our Current Coverage:

- ✅ Image quality metrics
- ✅ Verification metrics (but needs impostor sampling fix)
- ✅ Identification at multiple scales (good!)
- ❌ Missing d-prime
- ❌ Missing full CMC curve
- ❌ 1:1 is broken (always 100%)

---

## Action Items

1. **Immediate**: Fix 1:1 identification (threshold check)
2. **Immediate**: Fix impostor sampling (random + more samples)
3. **High Priority**: Add d-prime calculation
4. **High Priority**: Rename "identity_sim" to "feature_preservation"
5. **Medium**: Add TAR@FAR=0.01%
6. **Medium**: Save full CMC curve for plotting
7. **Low**: Consider face detection baseline
