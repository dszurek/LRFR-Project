# Deploying Quantized Models to Raspberry Pi

This guide covers quantizing models, tracking them with Git LFS, and deploying to Raspberry Pi for edge inference.

## Prerequisites

### On Development Machine (Windows)

- Git LFS installed: `git lfs install`
- Poetry environment with PyTorch
- Trained models: `dsr16.pth`, `dsr24.pth`, `dsr32.pth`, `edgeface_finetuned_*.pth`

### On Raspberry Pi

- Raspberry Pi 4/5 (recommended 4GB+ RAM)
- Raspbian OS (64-bit recommended)
- Python 3.9+
- Git LFS: `sudo apt-get install git-lfs`

---

## Step 1: Quantize Models

Quantization reduces model size and speeds up inference on CPU/edge devices.

### Quantize all models (DSR + EdgeFace)

```powershell
# Dynamic quantization (easiest, good compression)
poetry run python -m technical.tools.quantize_models --models all --method dynamic --benchmark

# Static quantization (better accuracy, requires calibration)
poetry run python -m technical.tools.quantize_models --models all --method static

# Export to ONNX for cross-platform compatibility
poetry run python -m technical.tools.quantize_models --models all --export-onnx
```

### Quantize specific models

```powershell
# Only DSR models for 32x32
poetry run python -m technical.tools.quantize_models --models dsr --vlr-sizes 32 --method dynamic

# Only EdgeFace models
poetry run python -m technical.tools.quantize_models --models edgeface --method dynamic
```

### Output locations:

- DSR quantized: `technical/dsr/quantized/dsr{16,24,32}_quantized_dynamic.pth`
- EdgeFace quantized: `technical/facial_rec/edgeface_weights/quantized/edgeface_finetuned_{16,24,32}_quantized_dynamic.pth`
- ONNX models: Same directories with `.onnx` extension

---

## Step 2: Track Models with Git LFS

Git LFS is already configured in `.gitattributes` to track `.pth`, `.pt`, and `.onnx` files.

### Initialize Git LFS (if not already done)

```powershell
git lfs install
```

### Add quantized models to repository

```powershell
# Stage quantized models
git add technical/dsr/quantized/*.pth
git add technical/facial_rec/edgeface_weights/quantized/*.pth

# If you exported ONNX
git add technical/dsr/quantized/*.onnx
git add technical/facial_rec/edgeface_weights/quantized/*.onnx

# Commit
git commit -m "Add quantized models for Raspberry Pi deployment"
```

### Push to GitHub

```powershell
# Push LFS objects and commits
git push origin daniel_new

# Verify LFS tracking
git lfs ls-files
```

---

## Step 3: Deploy to Raspberry Pi

### On Raspberry Pi: Clone repository

```bash
# Install Git LFS
sudo apt-get update
sudo apt-get install git-lfs
git lfs install

# Clone repository (LFS will auto-download model files)
git clone https://github.com/dszurek/LRFR-Project.git
cd LRFR-Project
git checkout daniel_new

# Verify LFS files downloaded
git lfs ls-files
ls -lh technical/dsr/quantized/
```

### Install Python dependencies (lightweight for Pi)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install minimal dependencies for inference only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless numpy pillow

# If using ONNX models
pip install onnxruntime
```

### Test inference on Raspberry Pi

```bash
# Test DSR upscaling
python3 -c "
import torch
from technical.dsr.models import DSRColor, DSRConfig
from pathlib import Path

# Load quantized model
device = torch.device('cpu')
model_path = Path('technical/dsr/quantized/dsr32_quantized_dynamic.pth')
config = DSRConfig(base_channels=120, residual_blocks=16, output_size=(112, 112))
model = DSRColor(config=config).to(device)

# Load state dict
state = torch.load(model_path, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

# Test inference
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    y = model(x)
print(f'✓ DSR inference successful: {x.shape} -> {y.shape}')
"
```

---

## Step 4: Performance Comparison

### Expected model sizes:

| Model     | Original | Quantized (Dynamic) | Reduction |
| --------- | -------- | ------------------- | --------- |
| DSR 16×16 | ~45 MB   | ~12 MB              | 73%       |
| DSR 24×24 | ~40 MB   | ~11 MB              | 72%       |
| DSR 32×32 | ~35 MB   | ~10 MB              | 71%       |
| EdgeFace  | ~25 MB   | ~7 MB               | 72%       |

### Expected inference speed (Raspberry Pi 4):

- Original FP32: ~800-1200 ms per image
- Quantized INT8: ~200-400 ms per image (2-3× speedup)
- ONNX optimized: ~150-300 ms per image (3-4× speedup)

---

## Step 5: Create Raspberry Pi Inference Script

Minimal inference script for edge deployment:

```python
# rpi_inference.py
import torch
import cv2
import numpy as np
from pathlib import Path
from technical.dsr.models import DSRColor, DSRConfig

def load_quantized_dsr(vlr_size: int = 32):
    """Load quantized DSR model for Raspberry Pi."""
    device = torch.device('cpu')
    model_path = Path(f'technical/dsr/quantized/dsr{vlr_size}_quantized_dynamic.pth')

    # Configure model
    configs = {
        16: DSRConfig(base_channels=132, residual_blocks=20, output_size=(112, 112)),
        24: DSRConfig(base_channels=126, residual_blocks=18, output_size=(112, 112)),
        32: DSRConfig(base_channels=120, residual_blocks=16, output_size=(112, 112)),
    }
    config = configs[vlr_size]

    model = DSRColor(config=config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model

def upscale_image(model, image_path: str, vlr_size: int = 32):
    """Upscale an image using quantized DSR model."""
    # Load and preprocess
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (vlr_size, vlr_size))
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)

    # Postprocess
    output = output.squeeze(0).clamp(0, 1).numpy()
    output = (output.transpose(1, 2, 0) * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    return output

if __name__ == '__main__':
    print("Loading quantized DSR model...")
    model = load_quantized_dsr(vlr_size=32)
    print("✓ Model loaded successfully")

    # Test with sample image
    # output = upscale_image(model, 'input.jpg', vlr_size=32)
    # cv2.imwrite('output.jpg', output)
    print("Ready for inference!")
```

---

## Git LFS Quotas & Best Practices

### GitHub LFS Quotas (Free tier)

- Storage: 1 GB
- Bandwidth: 1 GB/month
- Additional packs available: $5/month for 50GB storage + 50GB bandwidth

### Optimization tips:

1. **Only track quantized models** - original models can stay local
2. **Use dynamic quantization** - best size/accuracy tradeoff
3. **ONNX models** - often smaller than PyTorch checkpoints
4. **Use Git LFS prune** to remove old versions:
   ```powershell
   git lfs prune
   ```

### Check LFS usage:

```powershell
# See which files are tracked
git lfs ls-files

# See LFS storage size
git lfs ls-files | awk '{print $3}' | xargs du -ch

# On GitHub: Settings → Billing → Git LFS data
```

---

## Troubleshooting

### LFS files not downloading on Pi

```bash
# Manually fetch LFS files
git lfs fetch
git lfs checkout
```

### Out of memory on Raspberry Pi

- Use dynamic quantization (lower memory footprint)
- Process one image at a time (batch_size=1)
- Reduce model resolution (16×16 instead of 32×32)
- Enable swap:
  ```bash
  sudo dphys-swapfile swapoff
  sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
  sudo dphys-swapfile setup
  sudo dphys-swapfile swapon
  ```

### Slow inference

- Use ONNX Runtime instead of PyTorch
- Enable threading: `torch.set_num_threads(4)`
- Use hardware acceleration if available (Pi 5 has better CPU)

---

## Next Steps

1. **Quantize models**: Run `quantize_models.py` script
2. **Commit & push**: Add quantized models to Git LFS
3. **Clone on Pi**: Pull repository with LFS files
4. **Test inference**: Run sample inference script
5. **Optimize**: Profile and tune for your specific use case

For questions or issues, see the main project README or open an issue on GitHub.
