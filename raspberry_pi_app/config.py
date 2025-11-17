"""Configuration for Raspberry Pi 5 LRFR Application.

All paths and parameters are centralized here for easy customization.
"""

from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent

# Model paths (quantized for Pi 5 performance)
DSR_MODEL_16 = PROJECT_ROOT / "technical" / "dsr" / "quantized" / "dsr16_quantized_dynamic.pth"
DSR_MODEL_24 = PROJECT_ROOT / "technical" / "dsr" / "quantized" / "dsr24_quantized_dynamic.pth"
DSR_MODEL_32 = PROJECT_ROOT / "technical" / "dsr" / "quantized" / "dsr32_quantized_dynamic.pth"

EDGEFACE_MODEL_16 = PROJECT_ROOT / "technical" / "facial_rec" / "edgeface_weights" / "quantized" / "edgeface_finetuned_16_quantized_dynamic.pth"
EDGEFACE_MODEL_24 = PROJECT_ROOT / "technical" / "facial_rec" / "edgeface_weights" / "quantized" / "edgeface_finetuned_24_quantized_dynamic.pth"
EDGEFACE_MODEL_32 = PROJECT_ROOT / "technical" / "facial_rec" / "edgeface_weights" / "quantized" / "edgeface_finetuned_32_quantized_dynamic.pth"

# Haar Cascade for face detection (built into OpenCV)
HAAR_CASCADE_FACE = "haarcascade_frontalface_default.xml"

# Gallery storage
GALLERY_DIR = APP_DIR / "data" / "gallery"
GALLERY_METADATA = APP_DIR / "data" / "gallery.json"

# ============================================================================
# Gallery Settings
# ============================================================================

MAX_GALLERY_SIZE = 5  # Maximum number of people in gallery
MIN_IMAGES_PER_PERSON = 3  # Minimum images required to enroll
MAX_IMAGES_PER_PERSON = 10  # Maximum images per person
GALLERY_IMAGE_SIZE = (112, 112)  # Store as HR size

# ============================================================================
# Face Detection Settings
# ============================================================================

FACE_DETECTION_SCALE_FACTOR = 1.1  # Lower = more sensitive, slower
FACE_DETECTION_MIN_NEIGHBORS = 5  # Higher = stricter detection
FACE_MIN_SIZE = (80, 80)  # Minimum face size in pixels
FACE_PADDING = 0.2  # Add 20% padding around detected face

# ============================================================================
# Webcam Settings
# ============================================================================

WEBCAM_INDEX = 0  # Default camera (usually /dev/video0)
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
WEBCAM_FPS = 30
WEBCAM_WARMUP_FRAMES = 10  # Number of frames to skip for auto-exposure

# ============================================================================
# Pipeline Settings
# ============================================================================

# VLR resolution options
VLR_SIZES = [16, 24, 32]
DEFAULT_VLR_SIZE = 32

# EdgeFace input size (fixed)
HR_SIZE = (112, 112)

# ============================================================================
# Verification/Identification Settings
# ============================================================================

# Cosine similarity thresholds
VERIFICATION_THRESHOLD_1_1 = 0.5  # 1:1 verification (higher = stricter)
IDENTIFICATION_THRESHOLD_1_N = 0.3  # 1:N identification (lower = more permissive)

# Top-K predictions to display
TOP_K = 5

# ============================================================================
# Performance Settings
# ============================================================================

# Device
DEVICE = "cpu"  # Pi 5 runs on CPU

# Number of threads for PyTorch
TORCH_THREADS = 4  # Pi 5 has 4 cores

# Memory optimization
TORCH_DETERMINISTIC = False  # Disable for speed
TORCH_BENCHMARK = True  # Enable cudnn benchmarking (faster on repeated shapes)

# ============================================================================
# GUI Settings
# ============================================================================

WINDOW_TITLE = "LRFR - Raspberry Pi 5"
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768

# Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

# Font settings
FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Display settings
RESULT_IMAGE_SIZE = (200, 200)  # Size for display images in results
THUMBNAIL_SIZE = (80, 80)  # Size for gallery thumbnails

# ============================================================================
# Debug Settings
# ============================================================================

DEBUG_MODE = False  # Enable verbose logging
SHOW_FPS = True  # Display FPS in webcam view
PROFILE_PERFORMANCE = False  # Log detailed timing for each pipeline stage

# ============================================================================
# Model Configuration Mappings
# ============================================================================

def get_dsr_config(vlr_size: int) -> dict:
    """Get DSR model configuration for given VLR size."""
    configs = {
        16: {"base_channels": 132, "residual_blocks": 20, "output_size": HR_SIZE},
        24: {"base_channels": 126, "residual_blocks": 18, "output_size": HR_SIZE},
        32: {"base_channels": 120, "residual_blocks": 16, "output_size": HR_SIZE},
    }
    if vlr_size not in configs:
        raise ValueError(f"Unsupported VLR size: {vlr_size}. Must be one of {VLR_SIZES}")
    return configs[vlr_size]


def get_model_paths(vlr_size: int) -> tuple[Path, Path]:
    """Get model paths for given VLR size."""
    dsr_models = {16: DSR_MODEL_16, 24: DSR_MODEL_24, 32: DSR_MODEL_32}
    edgeface_models = {16: EDGEFACE_MODEL_16, 24: EDGEFACE_MODEL_24, 32: EDGEFACE_MODEL_32}
    
    if vlr_size not in dsr_models:
        raise ValueError(f"Unsupported VLR size: {vlr_size}. Must be one of {VLR_SIZES}")
    
    return dsr_models[vlr_size], edgeface_models[vlr_size]
