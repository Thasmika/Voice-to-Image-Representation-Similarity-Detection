"""Configuration settings for Voice to Image System.

This module defines all configuration constants and settings used throughout
the application, including audio processing parameters, model settings,
directory paths, and server configuration.
"""

import os
from pathlib import Path


# ============================================================================
# Audio Processing Configuration
# ============================================================================

# Audio sample rate (Hz) - standard for speech processing
SAMPLE_RATE = 22050

# Spectrogram extraction parameters
SPECTROGRAM_CONFIG = {
    "n_fft": 2048,          # FFT window size (frequency resolution)
    "hop_length": 512,      # Number of samples between successive frames
    "n_mels": 128,          # Number of mel frequency bands
}

# Audio validation settings
MAX_AUDIO_DURATION = 30.0  # Maximum audio duration in seconds
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".ogg"]


# ============================================================================
# Image Generation Configuration
# ============================================================================

# Whisper model configuration
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large

# Stable Diffusion configuration
STABLE_DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-1"
IMAGE_GENERATION_CONFIG = {
    "guidance_scale": 7.5,      # How closely to follow the prompt
    "num_inference_steps": 50,  # Quality vs speed trade-off
    "height": 512,              # Output image height
    "width": 512,               # Output image width
}

# Image format settings
IMAGE_FORMAT = "PNG"
IMAGE_QUALITY = 95  # For JPEG (not used with PNG)


# ============================================================================
# Similarity Analysis Configuration
# ============================================================================

# CLIP model configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


# ============================================================================
# Directory Paths
# ============================================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Temporary files directory
TEMP_DIR = PROJECT_ROOT / "temp"

# Generated images storage directory
IMAGES_DIR = PROJECT_ROOT / "images"

# Static files directory (web interface)
STATIC_DIR = PROJECT_ROOT / "static"

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


# ============================================================================
# Server Configuration
# ============================================================================

# Server host and port
SERVER_HOST = "127.0.0.1"  # localhost
SERVER_PORT = int(os.environ.get("PORT", 8000))  # Configurable via environment

# CORS settings
CORS_ORIGINS = ["*"]  # Allow all origins for local development

# File upload limits
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB in bytes


# ============================================================================
# Model Loading Configuration
# ============================================================================

# Device configuration (auto-detect GPU)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"  # Use float16 for GPU inference

# Model cache directory (Hugging Face default)
MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface"


# ============================================================================
# Logging Configuration
# ============================================================================

# Logging level
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ============================================================================
# Performance Configuration
# ============================================================================

# Enable memory optimizations for GPU
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = True

# Maximum concurrent requests (for future scaling)
MAX_CONCURRENT_REQUESTS = 5


# ============================================================================
# Helper Functions
# ============================================================================

def get_temp_file_path(filename: str) -> Path:
    """Generate path for temporary file.
    
    Args:
        filename: Original filename
        
    Returns:
        Path object for temporary file
    """
    return TEMP_DIR / filename


def get_image_file_path(image_id: str) -> Path:
    """Generate path for saved image.
    
    Args:
        image_id: Unique image identifier
        
    Returns:
        Path object for image file
    """
    return IMAGES_DIR / f"{image_id}.png"


def get_server_url() -> str:
    """Get full server URL.
    
    Returns:
        Server URL string
    """
    return f"http://{SERVER_HOST}:{SERVER_PORT}"


# ============================================================================
# Configuration Summary
# ============================================================================

def print_config_summary():
    """Print configuration summary for debugging."""
    print("=" * 70)
    print("Voice to Image System - Configuration Summary")
    print("=" * 70)
    print(f"Server: {get_server_url()}")
    print(f"Device: {DEVICE}")
    print(f"FP16: {USE_FP16}")
    print(f"Temp Directory: {TEMP_DIR}")
    print(f"Images Directory: {IMAGES_DIR}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Max Audio Duration: {MAX_AUDIO_DURATION}s")
    print(f"Whisper Model: {WHISPER_MODEL_SIZE}")
    print(f"Stable Diffusion: {STABLE_DIFFUSION_MODEL}")
    print(f"CLIP Model: {CLIP_MODEL_NAME}")
    print(f"Image Size: {IMAGE_GENERATION_CONFIG['width']}x{IMAGE_GENERATION_CONFIG['height']}")
    print("=" * 70)
