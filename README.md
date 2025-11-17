# LRFR Project

End-to-end research prototype for low-resolution facial recognition with **multi-resolution support**. The
repository contains a complete pipeline that combines custom Deep Super Resolution (DSR) models
and ultra-light EdgeFace weights for identity inference, with comprehensive evaluation tools
suitable for research publications.

## 🎉 NEW: Raspberry Pi 5 Deployment App

Production-ready GUI application for real-time facial recognition on Raspberry Pi 5:

- **Gallery Management:** Create custom galleries (1-5 people) with your own images
- **Real-time Webcam Capture:** Automatic face detection and cropping
- **Multi-Resolution Support:** Test with 16×16, 24×24, or 32×32 VLR inputs
- **1:1 Verification** or **1:N Identification** modes
- **Live Performance Metrics:** Top-5 predictions, confidence scores, processing time
- **Memory Optimized:** Uses quantized INT8 models (89% smaller)

**See `raspberry_pi_app/README.md` for complete documentation.**

Quick start on Raspberry Pi 5 (Ubuntu 24.04):

```bash
cd raspberry_pi_app
bash install.sh           # Install dependencies and models
source venv/bin/activate
python app.py             # Launch GUI application
```

## 🎯 New Features

### Multi-Resolution Support (16×16, 24×24, 32×32)

- ✅ **DSR Training:** Train separate models for 16×16, 24×24, and 32×32 VLR inputs
- ✅ **EdgeFace Fine-Tuning:** Resolution-aware fine-tuning with optimized hyperparameters
- ✅ **Cyclic Fine-Tuning:** Continue training DSR with fine-tuned EdgeFace for +8-15% accuracy boost
- ✅ **Comprehensive GUI Evaluation:** Publication-quality metrics and visualizations
- ✅ **Automated Reporting:** Generate PDF reports with ROC curves, distributions, and comparative analysis

See **`technical/MULTI_RESOLUTION_GUIDE.md`** for complete documentation.

### Cyclic Training (NEW!)

Instead of retraining DSR from scratch, **continue training** with fine-tuned EdgeFace:

```powershell
# Automated pipeline: Initial DSR → EdgeFace FT → DSR Cyclic FT
poetry run python -m technical.tools.cyclic_train --device cuda

# Or manual step-by-step (see technical/CYCLIC_TRAINING_QUICKSTART.md)
```

**Benefits:**

- ⚡ **2-3× faster** than full retraining (50 vs 100 epochs)
- 📈 **+8-15% accuracy** improvement expected
- 🎯 **More stable** convergence (preserves learned features)

See **`technical/CYCLIC_TRAINING_QUICKSTART.md`** for quick start guide and **`technical/CYCLIC_VS_RETRAINING_ANALYSIS.md`** for full analysis.

## 📦 Project layout

- `technical/dsr/` – DSR training scripts and model definitions (supports 16×16, 24×24, 32×32).
- `technical/facial_rec/` – EdgeFace backbone utilities and fine-tuning (resolution-aware).
- `technical/pipeline/` – Production inference pipeline and evaluation tools.
- `tests/` – Lightweight unit checks for core utilities.

## 🚀 Quick Start: Multi-Resolution Evaluation

### GUI Mode (Recommended)

```powershell
# Launch comprehensive evaluation GUI
poetry run python -m technical.pipeline.evaluate_gui
```

### Command-Line Mode

```powershell
# Evaluate all three resolutions
poetry run python -m technical.pipeline.evaluate_gui `
    --test-root technical/dataset/frontal_only/test `
    --gallery-root technical/dataset/frontal_only/train `
    --resolutions 16 24 32 `
    --output-dir evaluation_results
```

**Outputs:**

- `evaluation_report.pdf` - Comprehensive PDF with all visualizations
- `results.json` - Detailed metrics in JSON format
- Individual PNG plots (ROC curves, quality metrics, etc.)

## 🧠 How the DSR model works

At a glance, the Deep Super Resolution (DSR) network turns a blurry,
very-low-resolution face crop into a sharper 128×128 RGB image. It is a compact
convolutional model (`DSRColor`) built from residual blocks, so each layer learns
to refine the previous approximation instead of starting from scratch. During
training we feed the model paired examples of matching low-resolution (VLR) and
high-resolution (HR) faces. The script at `technical/dsr/train_dsr.py` augments
each pair (random flips, slight rotations, mild colour tweaks) to make the model
robust to real-world capture noise.

The optimisation objective mixes four ideas:

1. **Pixel fidelity (L1 loss):** keeps the generated image close to the HR target.
2. **Perceptual similarity (VGG19 features):** compares intermediate features of
   a pretrained classification network so textures look natural, not just accurate
   per pixel.
3. **Identity consistency (EdgeFace embeddings):** runs both the SR output and
   the ground-truth HR image through the lightweight EdgeFace recogniser and
   maximises the cosine similarity of their embeddings. This keeps the person’s
   identity intact.
4. **Total variation regularisation:** discourages noisy checkerboard artefacts.

Training uses AdamW with cosine learning-rate decay, mixed precision (AMP), and
an Exponential Moving Average (EMA) copy of the weights for stable validation.
Each epoch ends with PSNR and loss reporting, and the EMA weights are saved to
`technical/dsr/dsr.pth` whenever we beat the previous best validation score.

## 🔄 How the full pipeline is stitched together

The pipeline in `technical/pipeline/pipeline.py` orchestrates the end-to-end
recognition flow:

1. **Upscale:** load the trained DSR weights and run them on the incoming VLR
   probe image to create a sharper version.
2. **Embed:** feed the super-resolved image through EdgeFace to obtain a compact
   512-dimensional embedding vector.
3. **Compare:** compute cosine similarity between the probe embedding and every
   registered gallery embedding. The gallery entries can be the raw HR images or
   the same DSR output, depending on configuration.
4. **Decide:** pick the gallery identity with the highest similarity and emit it
   when the score crosses the configured threshold; otherwise report "unknown".

`PipelineConfig` lets you pick the device (CPU/GPU), thread count, thresholds,
and the file paths to the stored DSR/EdgeFace weights. The helper methods
`register_identity()` and `run()` wrap the common tasks so a CLI or service can
add new identities and query the recogniser with just a few lines of code.

## 🚀 Pipeline quick-start

The pipeline is designed to run fully on CPU. Install dependencies with Poetry
(recommended) or pip.

```powershell
poetry install --with dev
```

On Raspberry Pi, replace the local Windows-only Torch wheels in
`pyproject.toml` with the appropriate ARM builds or install torch separately
before running `poetry install`.

### Minimal example

```python
from technical.pipeline import PipelineConfig, build_pipeline

config = PipelineConfig(
	dsr_weights_path="technical/dsr/dsr.pth",
	edgeface_weights_path="technical/facial_rec/edgeface_weights/edgeface_xxs_q.pt",
	device="cpu",
	num_threads=2,  # tune for Raspberry Pi
)

pipeline = build_pipeline(config)
pipeline.register_identity("Alice", "path/to/alice_low_res.png")

result = pipeline.run("path/to/low_res_input.png")
print(result["identity"], result["score"])
result["sr_image"].save("upscaled.png")
```

### Integrating with a CLI capture loop

The `FaceRecognitionPipeline` exposes `run()` and `register_identity()` helper
methods so that a future CLI can:

1. Capture a frame from a USB camera.
2. Pass the frame directly (NumPy array or PIL image) to `run()`.
3. Display the returned identity score and store the upscaled image if desired.

See `technical/pipeline/pipeline.py` for more hooks that can be imported into a
CLI entry point.

## 📊 Evaluation

The project provides comprehensive evaluation tools for both model training and deployment scenarios:

### Quick Dataset Evaluation

```powershell
poetry run python -m technical.pipeline.evaluate_dataset `
	--dataset-root technical/dataset/test_processed
```

### Comprehensive Verification & Identification Metrics

For proper face recognition metrics (FAR, FRR, EER, Rank-1 accuracy):

```powershell
# 1:1 Verification (e.g., phone unlock, door access)
poetry run python -m technical.pipeline.evaluate_verification `
	--mode verification `
	--test-root technical/dataset/frontal_only/test

# 1:N Identification (e.g., office with 10 people)
poetry run python -m technical.pipeline.evaluate_verification `
	--mode identification `
	--gallery-root technical/dataset/frontal_only/train `
	--test-root technical/dataset/frontal_only/test `
	--max-gallery-size 10
```

See **`technical/EVALUATION_GUIDE.md`** for:

- Complete evaluation workflows (1:1 verification vs 1:N identification)
- Metric interpretations (FAR, FRR, EER, Rank-1/5/10 accuracy)
- Dataset recommendations for different scenarios
- Training multiple DSR models (16×16, 24×24, 32×32 VLR inputs)
- EdgeFace fine-tuning for small vs large galleries

## 🧪 Tests

Execute unit tests locally (requires PyTorch):

```powershell
poetry run pytest
```

`tests/test_identity_database.py` is skipped automatically when torch is not
installed, keeping CI lean.

## 🛠 Raspberry Pi deployment notes

- Install official CPU wheels for PyTorch/torchvision (`pip install torch torchvision`)
  before running `poetry install` so Poetry can reuse the existing packages.
- Start the pipeline with `PipelineConfig(num_threads=2, force_full_precision=True)`
  to keep memory pressure predictable on the Pi 5.
- For USB camera capture in a future CLI, prefer `opencv-python` (`cv2.VideoCapture`)
  or `libcamera` bindings and pass the captured frame (as NumPy) directly to
  `pipeline.run(frame)`.
