"""Configuration for Raspberry Pi 5 LRFR Application.

All paths and parameters are centralized here for easy customization.
Works on both Windows (development) and Raspberry Pi (deployment).
"""

from pathlib import Path
import platform

# ============================================================================
# Project Paths
# ============================================================================

APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent

# Detect platform for platform-specific settings
IS_WINDOWS = platform.system() == "Windows"
IS_RASPBERRY_PI = platform.system() == "Linux" and platform.machine() in ["aarch64", "armv7l"]

# Model paths - Use full-scale (non-quantized) models on all platforms
DSR_MODEL_16 = PROJECT_ROOT / "technical" / "dsr" / "dsr16.pth"
DSR_MODEL_24 = PROJECT_ROOT / "technical" / "dsr" / "dsr24.pth"
DSR_MODEL_32 = PROJECT_ROOT / "technical" / "dsr" / "dsr32.pth"

EDGEFACE_MODEL_16 = PROJECT_ROOT / "technical" / "facial_rec" / "edgeface_weights" / "edgeface_finetuned_16.pth"
EDGEFACE_MODEL_24 = PROJECT_ROOT / "technical" / "facial_rec" / "edgeface_weights" / "edgeface_finetuned_24.pth"
EDGEFACE_MODEL_32 = PROJECT_ROOT / "technical" / "facial_rec" / "edgeface_weights" / "edgeface_finetuned_32.pth"

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
MAX_IMAGES_PER_PERSON = 100  # Maximum images per person (increased from 10)
GALLERY_IMAGE_SIZE = (112, 112)  # Store as HR size

# Embedding computation optimization
# When adding a person with many images, only use a subset for computing embedding
# This significantly speeds up gallery registration and app initialization
# SET TO 100: Use all images for maximum accuracy
MAX_IMAGES_FOR_EMBEDDING = 100  # Use at most this many images for embedding computation

# Force recomputation of embeddings on app startup
# Set to True to always recompute embeddings (slower but ensures accuracy)
# Set to False to reuse saved embeddings (faster but may use outdated embeddings)
FORCE_RECOMPUTE_EMBEDDINGS_ON_STARTUP = True

# ============================================================================
# Face Detection Settings
# ============================================================================

# More lenient face detection settings for gallery registration
FACE_DETECTION_SCALE_FACTOR = 1.05  # Lower = more sensitive (was 1.1)
FACE_DETECTION_MIN_NEIGHBORS = 3  # Lower = more lenient (was 5)
FACE_MIN_SIZE = (60, 60)  # Smaller minimum face size (was 80x80)
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

# Cosine similarity thresholds (increased for stricter matching)
VERIFICATION_THRESHOLD_1_1 = 0.65  # 1:1 verification (higher = stricter, was 0.5)
IDENTIFICATION_THRESHOLD_1_N = 0.50  # 1:N identification (higher = stricter, was 0.3)

# Top-K predictions to display
TOP_K = 5

# ============================================================================
# Performance Settings
# ============================================================================

# Device - auto-detect CUDA availability
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"  # Use GPU if available
else:
    DEVICE = "cpu"  # CPU fallback

# Number of threads for PyTorch
TORCH_THREADS = 4  # Pi 5 has 4 cores; adjust for your Windows system if needed

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
