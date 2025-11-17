# Raspberry Pi 5 Setup Guide - PyTorch Installation

This guide explains how to install PyTorch on Raspberry Pi 5 for the LRFR application.

---

## 🔍 System Requirements

- **Hardware:** Raspberry Pi 5 (4GB or 8GB RAM recommended)
- **OS:** Ubuntu 24.04 LTS (64-bit) or Raspberry Pi OS (64-bit)
- **Architecture:** ARM64 (aarch64)
- **Python:** 3.11 (recommended) or 3.13
- **Compute:** CPU only (no CUDA support)

---

## 📦 PyTorch Wheels for Raspberry Pi 5

### ⚠️ IMPORTANT: Do NOT Use Windows/CUDA Wheels

The main project uses Windows CUDA wheels in `technical/local_packages/`:
- ❌ `torch-2.8.0+cu129-cp313-cp313-win_amd64.whl` - **Windows x86_64 with CUDA**
- ❌ `torchvision-0.23.0+cu129-cp313-cp313-win_amd64.whl` - **Windows x86_64 with CUDA**

These will **NOT work** on Raspberry Pi 5!

### ✅ Correct Wheels for Raspberry Pi 5

You need **ARM64 CPU-only** wheels. PyTorch provides these automatically.

---

## 🚀 Installation Methods

### Method 1: Automated Installation (Recommended)

The `install.sh` script handles everything automatically:

```bash
cd raspberry_pi_app
bash install.sh
```

This script will:
1. ✅ Detect ARM64 architecture
2. ✅ Install system dependencies (OpenCV, ATLAS, etc.)
3. ✅ Create Python virtual environment
4. ✅ Install PyTorch CPU wheels for ARM64
5. ✅ Install all other dependencies
6. ✅ Download quantized models via Git LFS
7. ✅ Run setup verification

### Method 2: Manual Installation

If you prefer manual control:

#### Step 1: Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3-pip \
    python3-venv \
    python3-tk \
    git-lfs \
    v4l-utils \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev
```

#### Step 2: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

#### Step 3: Install PyTorch (CPU for ARM64)

```bash
# This automatically downloads correct ARM64 wheels
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

**What happens:**
- PyTorch detects ARM64 architecture
- Downloads: `torch-2.2.0-cp311-cp311-manylinux_2_17_aarch64.whl`
- Downloads: `torchvision-0.17.0-cp311-cp311-manylinux_2_17_aarch64.whl`

#### Step 4: Install Other Dependencies

```bash
pip install opencv-python-headless pillow numpy tqdm pyyaml psutil scikit-image
```

#### Step 5: Download Quantized Models

```bash
cd ..  # Go to project root
git lfs install
git lfs pull
```

#### Step 6: Verify Installation

```bash
cd raspberry_pi_app
python test_setup.py
```

---

## 🔍 Verify Correct PyTorch Installation

After installation, verify you have the correct CPU version:

```bash
python3 << EOF
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CPU Available: {torch.cpu.is_available()}")
print(f"CUDA Available: {torch.cuda.is_available()}")  # Should be False
print(f"Architecture: {torch.__config__.show()}")
EOF
```

**Expected Output:**
```
PyTorch Version: 2.2.0+cpu
CPU Available: True
CUDA Available: False  ✅ Correct for Raspberry Pi
```

---

## 📥 Manual Wheel Download (Alternative)

If you need to download wheels manually (for offline installation):

### For Python 3.11 (Recommended):

**PyTorch:**
```
https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp311-cp311-linux_aarch64.whl
```

**TorchVision:**
```
https://download.pytorch.org/whl/cpu/torchvision-0.17.0%2Bcpu-cp311-cp311-linux_aarch64.whl
```

### Install Downloaded Wheels:

```bash
pip install torch-2.2.0+cpu-cp311-cp311-linux_aarch64.whl
pip install torchvision-0.17.0+cpu-cp311-cp311-linux_aarch64.whl
```

---

## 🔧 Configuration Changes for Raspberry Pi

### 1. Use CPU Device in Config

The app automatically detects CPU-only mode. In `config.py`:

```python
DEVICE = "cpu"  # Pi 5 runs on CPU
TORCH_THREADS = 4  # Pi 5 has 4 cores
```

### 2. Memory Optimizations

Already configured in the app:
- ✅ Quantized INT8 models (89% smaller)
- ✅ `torch.set_grad_enabled(False)` - Disable gradients
- ✅ `opencv-python-headless` - Saves ~100MB RAM
- ✅ Aggressive garbage collection
- ✅ Single-image processing (no batching)

### 3. Performance Settings

In `config.py`:
```python
TORCH_THREADS = 4  # Match Pi 5's 4 cores
TORCH_DETERMINISTIC = False  # Faster, non-deterministic
TORCH_BENCHMARK = False  # Set True if using same input sizes
```

---

## 🐛 Troubleshooting

### Issue 1: "No module named 'torch'"

**Solution:** PyTorch not installed correctly
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

### Issue 2: "Illegal instruction" or Segmentation Fault

**Solution:** Wrong architecture wheel installed
```bash
# Check what's installed
pip show torch

# Should show: Platform: linux_aarch64
# If shows: win_amd64 or x86_64, wrong wheel!

# Reinstall correct version
pip uninstall torch torchvision
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

### Issue 3: Models Not Found

**Solution:** Git LFS models not downloaded
```bash
git lfs install
git lfs pull
```

### Issue 4: Slow Inference

**Solution:** Optimize thread count
```python
# In config.py, try different values:
TORCH_THREADS = 2  # or 4, test which is faster
torch.set_num_threads(TORCH_THREADS)
```

### Issue 5: Out of Memory

**Solution:** Already using quantized models, but you can:
```bash
# Check RAM usage
free -h

# Close other applications
# Or upgrade to Pi 5 8GB model
```

---

## 📊 Performance Expectations on Raspberry Pi 5

With quantized models:

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| DSR 16×16→112×112 | ~150-200 | Fastest resolution |
| DSR 24×24→112×112 | ~200-300 | Medium resolution |
| DSR 32×32→112×112 | ~250-350 | Highest quality |
| EdgeFace Embedding | ~50-100 | 512-d embedding extraction |
| **Total Pipeline** | **~200-450ms** | **2-5 FPS** |

**Memory Usage:**
- Idle: ~100MB
- Peak (during processing): ~200MB
- Well within 4GB Pi 5 capacity ✅

---

## 🎯 Quick Start After Installation

```bash
# 1. Activate environment
cd raspberry_pi_app
source venv/bin/activate

# 2. Launch application
python app.py

# 3. Add people to gallery (via GUI)
# 4. Start recognition!
```

---

## 📚 Additional Resources

- **PyTorch ARM64 Wheels:** https://download.pytorch.org/whl/cpu/
- **Raspberry Pi Documentation:** https://www.raspberrypi.com/documentation/
- **App Documentation:** `README.md` in this directory
- **Quantized Models:** `QUANTIZATION_VERIFICATION.md`
- **Memory Optimization:** `MEMORY_OPTIMIZATION.md`

---

## ✅ Verification Checklist

After installation, verify:

- [ ] Architecture is `aarch64`: `uname -m`
- [ ] PyTorch version shows `+cpu`: `python -c "import torch; print(torch.__version__)"`
- [ ] CUDA is disabled: `python -c "import torch; print(torch.cuda.is_available())"` → `False`
- [ ] Models exist in `../technical/dsr/quantized/` and `../technical/facial_rec/edgeface_weights/quantized/`
- [ ] Test setup passes: `python test_setup.py`
- [ ] App launches: `python app.py`

---

**Last Updated:** November 2024
