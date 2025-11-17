# Raspberry Pi 5 LRFR Application

Production-ready facial recognition application for Raspberry Pi 5 with real-time webcam capture, gallery management, and interactive identification.

## Features

- **Gallery Management**: Create custom galleries with 1-5 people using your own images
- **Multi-Resolution Support**: Test with 16×16, 24×24, or 32×32 VLR inputs
- **Real-time Webcam Capture**: Automatic face detection and cropping
- **Dual Verification Modes**:
  - **1:1 Verification**: Compare against a specific person
  - **1:N Identification**: Search across entire gallery
- **Live Performance Metrics**:
  - Top-5 predictions with confidence scores
  - Processing time breakdown (detection, DSR, recognition)
  - Visual comparison (input → upscaled → matched identity)
- **Memory Optimized**: Efficient quantized models for Pi 5

## System Requirements

- **Hardware**: Raspberry Pi 5 (4GB+ RAM recommended)
- **OS**: Ubuntu 24.04 LTS (64-bit)
- **Camera**: USB webcam or Pi Camera Module
- **Storage**: ~500MB for models + gallery images

## Installation

### 1. Clone Repository (on Raspberry Pi)

```bash
git clone https://github.com/dszurek/LRFR-Project.git
cd LRFR-Project
git checkout daniel_new
```

### 2. Install System Dependencies

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and build tools
sudo apt-get install -y python3.11 python3-pip python3-venv

# Install OpenCV dependencies
sudo apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libharfbuzz0b \
    libwebp7 \
    libjasper1 \
    libilmbase25 \
    libopenexr25 \
    libgstreamer1.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5

# Install Tkinter for GUI
sudo apt-get install -y python3-tk

# Install camera support
sudo apt-get install -y v4l-utils
```

### 3. Create Virtual Environment

```bash
cd raspberry_pi_app
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download Quantized Models

Models are tracked with Git LFS and should auto-download. Verify:

```bash
ls -lh ../technical/dsr/quantized/
ls -lh ../technical/facial_rec/edgeface_weights/quantized/
```

If models are missing:

```bash
git lfs pull
```

## Usage

### Launch Application

```bash
source venv/bin/activate
python app.py
```

### Workflow

#### 1. **Create Gallery** (First Run)

- Click "Manage Gallery"
- Add 1-5 people with names
- Upload 3-10 images per person (different angles, lighting)
- Images are auto-cropped to faces and stored

#### 2. **Select Resolution Mode**

- Choose VLR size: 16×16, 24×24, or 32×32
- Higher resolution = better accuracy, slower processing
- Recommended: 32×32 for best results

#### 3. **Choose Verification Mode**

- **1:1 Verification**: Select a specific person to verify against
- **1:N Identification**: Search entire gallery for best match

#### 4. **Capture & Identify**

- Click "Capture from Webcam"
- Face is auto-detected and cropped
- Processing pipeline runs:
  1.  Downscale to VLR size
  2.  DSR upscaling to 112×112
  3.  EdgeFace embedding extraction
  4.  Cosine similarity ranking
- Results display:
  - Input image (VLR)
  - Upscaled image (DSR output)
  - Top match image
  - Top-5 predictions with confidence scores
  - Total processing time

### Gallery Management

#### Add Person

```python
# From GUI: Manage Gallery → Add Person
# Provide name and 3-10 images
```

#### Remove Person

```python
# From GUI: Manage Gallery → Remove Person
# Select from list
```

#### View Gallery

```python
# From GUI: Manage Gallery → View Gallery
# Shows all enrolled identities with sample images
```

## Performance Expectations

### Raspberry Pi 5 (Quantized Models)

| Resolution | DSR Time | EdgeFace Time | Total Time | Rank-1 Accuracy |
| ---------- | -------- | ------------- | ---------- | --------------- |
| 16×16      | ~200ms   | ~9ms          | ~250ms     | ~40%            |
| 24×24      | ~300ms   | ~9ms          | ~350ms     | ~45%            |
| 32×32      | ~400ms   | ~9ms          | ~450ms     | ~47%            |

**Note**: Times include face detection (~40ms). First inference is slower due to model warmup.

### Memory Usage

- **Idle**: ~200MB
- **Single inference**: ~350MB
- **Gallery (5 people, 10 images each)**: ~400MB
- **Peak (during capture)**: ~500MB

### Camera Settings

Default configuration:

- Resolution: 640×480
- FPS: 30
- Format: MJPEG (if supported, otherwise YUYV)

Adjust in `config.py` for your camera:

```python
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
WEBCAM_FPS = 30
```

## Configuration

Edit `config.py` to customize:

```python
# Model paths (quantized by default)
DSR_MODEL_16 = "../technical/dsr/quantized/dsr16_quantized_dynamic.pth"
DSR_MODEL_24 = "../technical/dsr/quantized/dsr24_quantized_dynamic.pth"
DSR_MODEL_32 = "../technical/dsr/quantized/dsr32_quantized_dynamic.pth"
EDGEFACE_MODEL_16 = "../technical/facial_rec/edgeface_weights/quantized/edgeface_finetuned_16_quantized_dynamic.pth"
EDGEFACE_MODEL_24 = "../technical/facial_rec/edgeface_weights/quantized/edgeface_finetuned_24_quantized_dynamic.pth"
EDGEFACE_MODEL_32 = "../technical/facial_rec/edgeface_weights/quantized/edgeface_finetuned_32_quantized_dynamic.pth"

# Gallery settings
MAX_GALLERY_SIZE = 5  # Maximum number of people
MIN_IMAGES_PER_PERSON = 3
MAX_IMAGES_PER_PERSON = 10

# Face detection
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (80, 80)

# Verification thresholds
VERIFICATION_THRESHOLD_1_1 = 0.5  # 1:1 verification
IDENTIFICATION_THRESHOLD_1_N = 0.3  # 1:N identification
```

## Troubleshooting

### Camera Not Detected

```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera
ffplay /dev/video0

# Check permissions
sudo usermod -a -G video $USER
# Log out and back in
```

### Out of Memory

```bash
# Increase swap (if needed)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Slow Performance

1. **Close background applications**: `htop` to check CPU usage
2. **Use quantized models**: Already default in config
3. **Reduce resolution**: Try 16×16 instead of 32×32
4. **Overclock Pi 5**: `sudo raspi-config` → Performance Options

### Face Detection Fails

- Ensure good lighting
- Face should be frontal (±30° rotation max)
- Minimum face size: 80×80 pixels in camera frame
- Try adjusting `FACE_DETECTION_SCALE_FACTOR` in config (lower = more sensitive)

### Model Loading Errors

```bash
# Verify Git LFS models downloaded
git lfs ls-files | grep quantized

# Manually download if needed
git lfs pull

# Check file sizes (not just LFS pointers)
ls -lh ../technical/dsr/quantized/*.pth
# Should be ~30-40MB each, not ~130 bytes
```

## Project Structure

```
raspberry_pi_app/
├── app.py                  # Main GUI application
├── config.py              # Configuration settings
├── pipeline.py            # LRFR inference pipeline
├── gallery_manager.py     # Gallery storage and retrieval
├── face_detector.py       # Face detection and cropping
├── webcam_capture.py      # Camera interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── data/                 # Auto-created on first run
    ├── gallery/          # User gallery images
    │   ├── person1/
    │   ├── person2/
    │   └── ...
    └── gallery.json      # Gallery metadata
```

## Development

### Run in Debug Mode

```bash
python app.py --debug
```

Enables:

- Verbose logging
- Performance profiling
- Model loading diagnostics

### Test Pipeline Without GUI

```bash
python -c "
from pipeline import LRFRPipeline
import cv2

# Initialize
pipeline = LRFRPipeline(vlr_size=32)

# Load test image
img = cv2.imread('test_face.jpg')

# Run pipeline
result = pipeline.process_image(img)
print(f'Processing time: {result[\"total_time\"]:.2f}ms')
"
```

## License

Part of the LRFR-Project. See main repository README for details.

## Citation

If you use this application in research, please cite:

```bibtex
@misc{lrfr-pi-app,
  author = {Daniel Szurek},
  title = {Raspberry Pi 5 Low-Resolution Facial Recognition Application},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dszurek/LRFR-Project}
}
```

## Support

For issues specific to this application:

- Check troubleshooting section above
- Review main project documentation in `../technical/`
- Open issue on GitHub with logs and system info

For general LRFR questions, see main project README.
