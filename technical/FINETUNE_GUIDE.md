# EdgeFace Fine-Tuning Guide

## Overview

`finetune_edgeface.py` fine-tunes the EdgeFace recognition model on **DSR outputs** to bridge the domain gap between pretrained data and your DSR super-resolved VLR faces. This should improve recognition accuracy by teaching EdgeFace to better handle your DSR model's characteristics.

## Strategy

The script uses a **two-stage training approach**:

### Stage 1: Train Classification Head (5 epochs)

- Freeze EdgeFace backbone (pretrained features preserved)
- Train new ArcFace classification head on your subjects
- Learning rate: 1e-3
- Goal: Learn subject identities from DSR outputs

### Stage 2: End-to-End Fine-Tuning (20 epochs)

- Unfreeze entire network
- Fine-tune with very low learning rate (5e-6 for backbone, 5e-5 for head)
- Cosine annealing scheduler
- Early stopping (patience=8)
- Goal: Adapt features specifically to DSR outputs

## Key Features

1. **ArcFace Loss**: Metric learning that creates better embedding clusters than softmax
2. **DSR Integration**: Automatically runs VLR images through DSR before training
3. **Augmentation**: Horizontal flip, color jitter for training robustness
4. **Mixed Precision**: Faster training with AMP (automatic mixed precision)
5. **Subject-Level Learning**: Maps DSR outputs to subject IDs directly

## Usage

### Basic Training (Recommended)

```bash
poetry run python -m technical.facial_rec.finetune_edgeface --device cuda
```

This will:

- Train for 5+20=25 epochs (~30-45 minutes on RTX 3060 Ti)
- Save best checkpoint to `facial_rec/edgeface_weights/edgeface_finetuned.pth`
- Use default hyperparameters (optimized for your setup)

### Custom Training

```bash
# Train for longer (if accuracy still improving)
poetry run python -m technical.facial_rec.finetune_edgeface --device cuda --stage2-epochs 30

# Use different EdgeFace checkpoint
poetry run python -m technical.facial_rec.finetune_edgeface --edgeface edgeface_xxs.pt
```

## Expected Results

- **Validation Accuracy**: 75-85% classification accuracy on train subjects
- **Recognition Improvement**: +10-20% when integrated into pipeline
- **Training Time**: ~30-45 minutes for 25 epochs on RTX 3060 Ti

## Dataset Requirements

The script expects this structure:

```
technical/dataset/
├── train_processed/
│   ├── vlr_images/  # VLR inputs
│   └── hr_images/   # HR ground truth
└── val_processed/
    ├── vlr_images/
    └── hr_images/
```

Subject IDs are extracted from filenames (first 3 digits):

- `007_01_01_010_00_crop_128.png` → Subject ID: `007`

## Integration with Pipeline

After training, update your pipeline to use the fine-tuned checkpoint:

### Option 1: Replace Weights Path

Edit `pipeline/pipeline.py`:

```python
# OLD
edgeface_weights_path = facial_rec_path / "edgeface_weights" / "edgeface_xxs.pt"

# NEW
edgeface_weights_path = facial_rec_path / "edgeface_weights" / "edgeface_finetuned.pth"
```

### Option 2: Load from Checkpoint

The saved checkpoint contains:

- `backbone_state_dict`: Fine-tuned EdgeFace weights
- `arcface_state_dict`: Classification head (not needed for inference)
- `subject_to_id`: Training subject mapping

Load just the backbone:

```python
checkpoint = torch.load("edgeface_finetuned.pth")
model.load_state_dict(checkpoint["backbone_state_dict"])
```

## Hyperparameters

Configured in `FinetuneConfig`:

| Parameter             | Value | Purpose                                  |
| --------------------- | ----- | ---------------------------------------- |
| `stage1_epochs`       | 5     | Train head epochs                        |
| `stage2_epochs`       | 20    | Fine-tune epochs                         |
| `batch_size`          | 32    | Batch size                               |
| `backbone_lr`         | 5e-6  | Very low to preserve pretrained features |
| `head_lr_stage2`      | 5e-5  | Head learning rate (stage 2)             |
| `arcface_scale`       | 64.0  | ArcFace scale (s)                        |
| `arcface_margin`      | 0.5   | ArcFace margin (m)                       |
| `weight_decay`        | 5e-5  | Regularization                           |
| `label_smoothing`     | 0.1   | Prevent overconfidence                   |
| `early_stop_patience` | 8     | Stop if no improvement                   |

## Monitoring Training

The script prints progress every epoch:

```
STAGE 1: Training ArcFace head (backbone frozen)
============================================================
Epoch 01/05 | Train Loss: 4.2134 Acc: 0.3142 | Val Loss: 3.8765 Acc: 0.3891
Epoch 02/05 | Train Loss: 3.1245 Acc: 0.5234 | Val Loss: 2.9432 Acc: 0.5678
  ✓ New best validation accuracy: 0.5678
...

STAGE 2: Fine-tuning entire model (backbone unfrozen)
============================================================
Epoch 01/20 | Train Loss: 2.1456 Acc: 0.6789 | Val Loss: 1.9876 Acc: 0.7123 | LR: 5.00e-06
  ✓ Saved checkpoint to edgeface_finetuned.pth (val acc: 0.7123)
...
```

Watch for:

- ✅ **Train accuracy improving**: Model learning subject identities
- ✅ **Val accuracy improving**: Model generalizing to unseen examples
- ⚠️ **Val accuracy plateauing**: Early stopping will kick in
- ❌ **Train >> Val accuracy**: Overfitting (increase regularization)

## Troubleshooting

### "CUDA out of memory"

Reduce batch size:

```python
config.batch_size = 16  # Down from 32
```

### "No improvement for 8 epochs"

This is normal! Early stopping prevents overfitting. Best checkpoint is already saved.

### Low validation accuracy (<50%)

- Check DSR model quality (compare outputs visually)
- Ensure dataset has correct HR/VLR pairs
- Try increasing `stage2_epochs` to 30-40

### Training too slow

- Reduce `num_workers` if CPU-bound: `config.num_workers = 4`
- Check GPU utilization: `nvidia-smi`

## Next Steps

1. **Run training**:

   ```bash
   poetry run python -m technical.facial_rec.finetune_edgeface --device cuda
   ```

2. **Update pipeline** to use `edgeface_finetuned.pth`

3. **Re-evaluate** on test set:

   ```bash
   poetry run python -m technical.pipeline.evaluate
   ```

4. **Compare accuracy**:

   - Before fine-tuning: ~34.87% (threshold=0.35)
   - Target after fine-tuning: >45-50%

5. **If accuracy improves significantly**, consider:
   - Running threshold sweep again to find new optimal threshold
   - Testing with `--use-tta` flag
   - Fine-tuning DSR model further with updated identity loss

## Theory: Why This Works

1. **Domain Adaptation**: Pretrained EdgeFace was trained on high-quality face datasets. Your DSR outputs have different characteristics (artifacts, noise patterns, resolution recovery effects). Fine-tuning teaches EdgeFace to recognize faces _specifically from your DSR model_.

2. **ArcFace Loss**: Creates more discriminative embeddings than softmax. Embeddings form tighter clusters per subject, with better inter-class separation.

3. **Two-Stage Training**: Prevents catastrophic forgetting:

   - Stage 1: New head learns your subjects without corrupting pretrained features
   - Stage 2: Gentle backbone adaptation preserves pretrained knowledge while specializing to DSR

4. **Subject-Level Learning**: Direct classification on your subjects (vs. generic face recognition) creates task-specific embeddings optimized for your gallery.

Expected outcome: **EdgeFace becomes "DSR-aware"**, recognizing faces from your super-resolution model more accurately than the generic pretrained version.
