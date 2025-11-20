# Low-Resolution Facial Recognition (LRFR) System

End-to-end research prototype for low-resolution facial recognition with **multi-resolution support (16×16, 24×24, 32×32)**. This repository contains a complete pipeline combining custom **Hybrid Transformer-CNN Super Resolution (DSR)** models and ultra-lightweight **EdgeFace** embeddings for robust identity inference.

**Key Innovation:** The Hybrid DSR model achieves high-fidelity upscaling with **<5.5M parameters**, making it suitable for edge deployment while maintaining high recognition accuracy.

---

## 📁 Repository Structure

```
├── colab_training.ipynb            # 🚀 Main training notebook for Google Colab (A100)
├── presentation/                   # Presentation assets
├── raspberry_pi_app/               # Production-ready Raspberry Pi 5 deployment app
├── technical/
│   ├── dsr/                        # Deep Super Resolution models
│   │   ├── hybrid_model.py         # Hybrid Transformer-CNN architecture
│   │   ├── train_hybrid_dsr.py     # Main DSR training script
│   │   ├── train_cyclic_hybrid.py  # Cyclic training pipeline
│   │   └── archive/                # Old models and scripts
│   ├── facial_rec/                 # EdgeFace facial recognition
│   │   ├── finetune_edgeface.py    # Resolution-aware fine-tuning
│   │   └── edgeface_weights/       # Model weights
│   └── dataset/                    # Dataset directory (managed via Drive on Colab)
├── tests/                          # Evaluation scripts and unit tests
│   ├── evaluate_cli.py             # Command-line evaluation
│   ├── evaluate_lfw_protocol.py    # LFW protocol evaluation
│   └── test_dsr.py                 # Unit tests
├── pyproject.toml                  # Poetry dependencies
└── requirements.txt                # Pip dependencies for Colab
```

---

## 🚀 Quick Start (Google Colab)

**The recommended way to train models is using the provided Colab notebook.**

1.  **Upload Dataset:** Zip your `technical/dataset` folder to `dataset.zip` and upload to Google Drive.
2.  **Open Notebook:** Open `colab_training.ipynb` in Google Colab.
3.  **Select Runtime:** Change runtime type to **GPU (A100)**.
4.  **Run All:** Execute the cells to clone the repo, install dependencies, and start training.

The notebook handles:
*   Environment setup
*   Training Hybrid DSR models (16x16, 24x24, 32x32)
*   Fine-tuning EdgeFace on DSR outputs
*   Evaluation and result saving

---

## 💻 Local Development & Training

### Installation

```bash
# Using Poetry (Recommended)
poetry install --with dev

# Using Pip
pip install -r requirements.txt
```

### 1. Train Hybrid DSR Models
Train the super-resolution component. The script automatically configures the architecture to stay under 5.5M parameters.

```bash
# Train for 16x16 resolution
poetry run python -m technical.dsr.train_hybrid_dsr --vlr-size 16

# Train for 24x24 and 32x32
poetry run python -m technical.dsr.train_hybrid_dsr --vlr-size 24
poetry run python -m technical.dsr.train_hybrid_dsr --vlr-size 32
```

### 2. Fine-Tune EdgeFace
Fine-tune the recognition model on the upscaled outputs to bridge the domain gap.

```bash
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 16 --device cuda
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 24 --device cuda
poetry run python -m technical.facial_rec.finetune_edgeface --vlr-size 32 --device cuda
```

### 3. Cyclic Training (Advanced)
Run the full cycle (DSR -> EdgeFace -> DSR) to mutually optimize both models.

```bash
poetry run python -m technical.dsr.train_cyclic_hybrid --resolutions 16 24 32 --cycles 1
```

---

## 🧠 System Architecture

### Hybrid DSR Model
A novel architecture combining the local feature extraction of **CNNs (Residual Blocks)** with the global context modeling of **Transformers**.
*   **Parameters:** < 5.5M (optimized for edge devices)
*   **Input:** 16x16, 24x24, or 32x32 RGB face images
*   **Output:** 112x112 High-Resolution face images
*   **Loss Functions:** Pixel Loss (L1), Perceptual Loss (VGG), Identity Loss (ArcFace), Adversarial Loss (Optional)

### EdgeFace Recognition
An ultra-lightweight face recognition model (EdgeFace-XXS) fine-tuned to recognize identities from super-resolved images.

---

## 📊 Evaluation

Evaluate trained models using the CLI tool:

```bash
poetry run python -m tests.evaluate_cli --resolutions 16 24 32 --output-dir evaluation_results
```

---

## 📝 License
Academic Research Use Only.
