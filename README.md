# Low-Resolution Facial Recognition (LRFR) System

End-to-end research prototype for low-resolution facial recognition with **multi-resolution support (16×16, 24×24, 32×32)**. This repository contains a complete pipeline combining custom Deep Super Resolution (DSR) models and ultra-lightweight EdgeFace embeddings for robust identity inference, with comprehensive evaluation tools suitable for research publications and real-world deployment.

---

## 📁 Repository Structure

```
├── presentation/                    # Final presentation slides and assets
├── evaluation_results/             # Evaluation outputs (ROC curves, metrics, reports)
├── Existing Papers/                # Reference papers and related work
├── Proposal/                       # Project proposal documents
├── Submission Paper/               # LaTeX source for research paper
├── raspberry_pi_app/               # 🎉 Production-ready Raspberry Pi 5 deployment app
│   ├── README.md                  # Complete deployment documentation
│   ├── app.py                     # GUI application with gallery management
│   ├── pipeline.py                # Optimized inference pipeline
│   ├── config.py                  # Configuration (quantized model paths)
│   └── install.sh                 # One-command setup script
├── technical/
│   ├── dsr/                       # Deep Super Resolution models
│   │   ├── train_dsr.py          # DSR training script (multi-resolution)
│   │   ├── models.py             # DSR architecture definitions
│   │   ├── evaluate_frontal.py   # DSR-specific evaluation
│   │   └── quantized/            # INT8 quantized models for deployment
│   ├── facial_rec/               # EdgeFace facial recognition
│   │   ├── finetune_edgeface.py  # Fine-tuning script (resolution-aware)
│   │   ├── finetune_contrastive.py # Contrastive learning experiments
│   │   └── edgeface_weights/     # Model weights (pretrained & fine-tuned)
│   ├── dataset/                  # Dataset preprocessing
│   │   ├── preprocess.py         # Main preprocessing pipeline
│   │   ├── train_processed/      # Training data (HR + VLR pairs)
│   │   ├── val_processed/        # Validation data
│   │   ├── test_processed/       # Test data
│   │   └── frontal_only/         # Frontal face subset for evaluation
│   ├── pipeline/                 # Inference and evaluation
│   │   ├── pipeline.py           # Production inference pipeline
│   │   ├── evaluate_cli.py       # Command-line evaluation
│   │   ├── evaluate_gui.py       # 🎨 GUI evaluation with visualizations
│   │   ├── evaluate_dataset.py   # Quick dataset metrics
│   │   ├── evaluate_recognition.py # 1:N identification metrics
│   │   ├── evaluate_verification.py # 1:1 verification metrics
│   │   └── register_user.py      # Gallery enrollment utility
│   └── tools/                    # Training utilities
│       ├── cyclic_train.py       # 🔄 Automated cyclic training pipeline
│       ├── quantize_models.py    # Model quantization for deployment
│       └── regenerate_vlr_dataset.py # VLR dataset regeneration
├── tests/                        # Unit tests
│   └── test_identity_database.py
├── pyproject.toml                # Poetry dependencies & configuration
├── poetry.lock                   # Locked dependency versions
└── README.md                     # This file
```

---

## 🎉 Key Features

### 🖥️ Raspberry Pi 5 Deployment App

Production-ready GUI application for real-time facial recognition on edge devices:

- **Gallery Management:** Create custom galleries (up to 5 people) with your own images
- **Real-time Webcam Capture:** Automatic face detection and cropping
- **Multi-Resolution Support:** Test with 16×16, 24×24, or 32×32 VLR inputs
- **Dual Operation Modes:** 1:1 Verification or 1:N Identification
- **Live Performance Metrics:** Top-5 predictions, confidence scores, processing time breakdown
- **Memory Optimized:** Quantized INT8 models (89% size reduction, ~200MB RAM peak usage)

**📂 See `raspberry_pi_app/README.md` for complete deployment documentation.**

Quick start on Raspberry Pi 5 (Ubuntu 24.04):

```bash
cd raspberry_pi_app
bash install.sh           # Install dependencies and download quantized models
source venv/bin/activate
python app.py             # Launch GUI application
```

### 🎯 Multi-Resolution Training & Evaluation

- ✅ **DSR Training:** Train separate super-resolution models for 16×16, 24×24, and 32×32 VLR inputs
- ✅ **EdgeFace Fine-Tuning:** Resolution-aware fine-tuning with optimized hyperparameters per resolution
- ✅ **Cyclic Fine-Tuning:** Continue training DSR with fine-tuned EdgeFace for +8-15% accuracy boost
- ✅ **Comprehensive Evaluation Tools:** Publication-quality metrics (FAR, FRR, EER, TAR, CMC) with visualizations
- ✅ **Automated PDF Reporting:** Generate reports with ROC curves, score distributions, and comparative analysis

### ⚡ Cyclic Training Pipeline

Instead of retraining DSR from scratch, **continue training** with fine-tuned EdgeFace:

```powershell
# Automated pipeline: Initial DSR → EdgeFace Fine-Tune → DSR Cyclic Fine-Tune
poetry run python -m technical.tools.cyclic_train --device cuda
```

**Benefits:**

- ⚡ **2-3× faster** than full retraining (50 vs 100 epochs)
- 📈 **+8-15% accuracy** improvement over single-stage training
- 🎯 **More stable** convergence (preserves learned features)

---

## 🚀 Quick Start

### Installation

```powershell
# Install dependencies with Poetry (recommended)
poetry install --with dev

# Or with pip (alternative)
pip install torch torchvision numpy pillow opencv-python tqdm scikit-learn matplotlib seaborn
```

### Train Multi-Resolution DSR Models

```powershell
# Train all three resolutions (16×16, 24×24, 32×32)
poetry run python -m technical.dsr.train_dsr --vlr-size 16 --device cuda
poetry run python -m technical.dsr.train_dsr --vlr-size 24 --device cuda
poetry run python -m technical.dsr.train_dsr --vlr-size 32 --device cuda
```

### Fine-Tune EdgeFace

```powershell
# Fine-tune for each resolution
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 16 --device cuda
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 24 --device cuda
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 32 --device cuda
```

### Cyclic Training (Optional)

```powershell
# Continue DSR training with fine-tuned EdgeFace (automated)
poetry run python -m technical.tools.cyclic_train --device cuda --vlr-size 32
```

### Evaluate Models

#### GUI Evaluation (Recommended - Best Visualizations)

```powershell
# Launch comprehensive evaluation GUI with all metrics and plots
poetry run python -m technical.pipeline.evaluate_gui
```

#### Command-Line Evaluation

```powershell
# Evaluate all three resolutions
poetry run python -m technical.pipeline.evaluate_cli \
    --test-root technical/dataset/frontal_only/test \
    --gallery-root technical/dataset/frontal_only/train \
    --resolutions 16 24 32 \
    --output-dir evaluation_results

# Quick dataset evaluation
poetry run python -m technical.pipeline.evaluate_dataset \
    --dataset-root technical/dataset/test_processed
```

**Evaluation Outputs:**

- `evaluation_report.pdf` - Comprehensive PDF with all visualizations
- `results.json` - Detailed metrics in JSON format
- Individual PNG plots (ROC curves, CMC curves, score distributions, etc.)

---

## 🧠 System Architecture

### DSR (Deep Super Resolution) Network

The DSR network upscales very-low-resolution (VLR) face images to high-resolution (HR) 112×112 RGB images suitable for facial recognition. Built from residual blocks, each layer refines the previous approximation.

**Training Objective (Multi-Loss):**

1. **Pixel Fidelity (L1 Loss):** Keeps generated image close to HR target
2. **Perceptual Similarity (VGG19 Features):** Ensures natural-looking textures
3. **Identity Consistency (EdgeFace Embeddings):** Preserves person's identity via cosine similarity
4. **Total Variation Regularization:** Reduces checkerboard artifacts

**Training Details:**

- Optimizer: AdamW with cosine learning-rate decay
- Mixed Precision: AMP for faster training
- EMA (Exponential Moving Average): Stable validation weights
- Data Augmentation: Random flips, rotations, color jitter

### Full Recognition Pipeline

The end-to-end pipeline (`technical/pipeline/pipeline.py`) orchestrates:

1. **Upscale:** Load trained DSR weights, upscale VLR probe image to 112×112
2. **Embed:** Feed super-resolved image through EdgeFace → 512-d embedding vector
3. **Compare:** Compute cosine similarity with all gallery embeddings
4. **Decide:** Select gallery identity with highest similarity (if above threshold)

**Configuration:**

- Device selection (CPU/GPU)
- Thread count for CPU inference
- Similarity thresholds for verification/identification
- Model weight paths

---

## 📊 Evaluation Metrics

### 1:1 Verification Metrics

- **FAR (False Accept Rate):** Impostor incorrectly accepted
- **FRR (False Reject Rate):** Genuine user incorrectly rejected
- **EER (Equal Error Rate):** Operating point where FAR = FRR
- **TAR@FAR:** True Accept Rate at specific False Accept Rate

### 1:N Identification Metrics

- **Rank-1 Accuracy:** Correct identity is top match
- **Rank-5/10 Accuracy:** Correct identity in top-5/10 matches
- **CMC (Cumulative Match Characteristic) Curve:** Rank-k accuracy vs rank
- **d-prime (d'):** Separability between genuine and impostor distributions

### Visualization Outputs

- ROC Curves (TAR vs FAR)
- DET Curves (FRR vs FAR)
- CMC Curves (Identification Rate vs Rank)
- Score Distribution Histograms (Genuine vs Impostor)
- Quality Metrics (PSNR, SSIM, LPIPS)

---

## 🛠️ Model Quantization & Deployment

### Quantize Models for Edge Devices

```powershell
# Quantize all models to INT8 (dynamic quantization)
poetry run python -m technical.tools.quantize_models --models all --method dynamic

# Benchmark original vs quantized
poetry run python -m technical.tools.quantize_models --models all --benchmark

# Export to ONNX (cross-platform deployment)
poetry run python -m technical.tools.quantize_models --models all --export-onnx
```

**Quantization Results:**

- **EdgeFace:** 4.75MB → 0.49MB (89.6% reduction)
- **DSR Models:** ~30-40MB → ~30-40MB (minimal reduction due to Conv2d layers)
- **RAM Usage:** ~271MB (FP32) → ~83MB (INT8) = 69% reduction
- **Speed:** Comparable inference time with lower memory footprint

---

## 🧪 Testing

Run unit tests:

```powershell
poetry run pytest
```

Test coverage includes:

- Identity database operations
- Pipeline consistency
- Model loading and inference

---

## 📦 Dataset Preparation

### Required Dataset Structure

```
technical/dataset/
├── train_processed/
│   ├── hr_images/          # High-resolution face images (112×112)
│   └── vlr_images/         # Very-low-resolution images (16×16, 24×24, or 32×32)
├── val_processed/
│   ├── hr_images/
│   └── vlr_images/
└── test_processed/
    ├── hr_images/
    └── vlr_images/
```

### Preprocess Custom Dataset

```powershell
# Preprocess images and generate VLR pairs
poetry run python -m technical.dataset.preprocess \
    --input-dir path/to/raw/images \
    --output-dir technical/dataset/custom_processed \
    --vlr-size 32
```

---

## 🔧 Advanced Usage

### Custom Training Configuration

Modify training parameters in scripts:

- **DSR:** `technical/dsr/train_dsr.py` - Adjust epochs, batch size, learning rate
- **EdgeFace:** `technical/facial_rec/finetune_edgeface.py` - Configure fine-tuning hyperparameters
- **Cyclic:** `technical/tools/cyclic_train.py` - Set cyclic training iterations

### Pipeline Integration

```python
from technical.pipeline import PipelineConfig, build_pipeline

config = PipelineConfig(
    dsr_weights_path="technical/dsr/dsr32.pth",
    edgeface_weights_path="technical/facial_rec/edgeface_weights/edgeface_finetuned_32.pth",
    device="cpu",
    num_threads=4,
)

pipeline = build_pipeline(config)

# Register users
pipeline.register_identity("Alice", "path/to/alice_vlr.png")
pipeline.register_identity("Bob", "path/to/bob_vlr.png")

# Run recognition
result = pipeline.run("path/to/probe_vlr.png")
print(f"Identity: {result['identity']}, Score: {result['score']:.3f}")
result["sr_image"].save("upscaled_probe.png")
```

---

## 🎓 Research & Publications

### Citation

If you use this work, please cite:

```bibtex
@misc{lrfr2025,
  title={Low-Resolution Facial Recognition via Deep Super-Resolution and Cyclic Training},
  author={Daniel Szurek, Brandon Nguyen},
  year={2025},
  note={CS565 Course Project}
}
```

### Related Files

- **Proposal:** `Proposal/` - Initial project proposal
- **Paper:** `Submission Paper/` - LaTeX source for research paper
- **Presentation:** `presentation/` - Final presentation slides
- **Pre-Cycle Report:** `evaluation_report_precycle.pdf` - Evaluation before cyclic training

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```powershell
# Reduce batch size in training scripts
# Use gradient accumulation instead
```

**2. Model Loading Errors**

```powershell
# Check model paths in config files
# Ensure quantized models are downloaded via Git LFS
git lfs pull
```

**3. Raspberry Pi Installation Issues**

```powershell
# Install system dependencies first
sudo apt-get update
sudo apt-get install python3-opencv libatlas-base-dev

# Use ARM-compatible PyTorch wheels
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**4. Import Errors**

```powershell
# Ensure technical module is in path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
poetry install
```

---

## 📝 License

This project is for academic research purposes. See individual model licenses for pretrained weights.

---

## 🙏 Acknowledgments

- **EdgeFace:** Ultra-lightweight face recognition backbone
- **CMU PIE & Labeled Faces in the Wild Datasets:** Training and evaluation data
- **PyTorch:** Deep learning framework
- **OpenCV:** Computer vision utilities

---

## 📧 Contact

For questions or collaboration:

- Open an issue on GitHub
- Contact: [Your Email/Department]

---

**Last Updated:** November 2025
