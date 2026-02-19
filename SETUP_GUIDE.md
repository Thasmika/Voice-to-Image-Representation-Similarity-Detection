# Voice to Image System - Complete Setup Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [First Run](#first-run)
4. [Running the System](#running-the-system)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Configuration](#configuration)
8. [GPU Setup (Optional)](#gpu-setup-optional)

---

## Prerequisites

### Required Software

#### 1. Python 3.8 or Higher
Check if Python is installed:
```bash
python --version
```

If not installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python3`
- **Linux**: `sudo apt-get install python3 python3-pip`

#### 2. FFmpeg (Required for Audio Processing)
Check if FFmpeg is installed:
```bash
ffmpeg -version
```

If not installed:

**Windows**:
```bash
# Using winget
winget install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**macOS**:
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Linux (CentOS/RHEL)**:
```bash
sudo yum install ffmpeg
```

#### 3. Git (for cloning repository)
Check if Git is installed:
```bash
git --version
```

If not installed, download from [git-scm.com](https://git-scm.com/downloads)

### System Requirements

**Minimum (CPU Mode)**:
- 8GB RAM
- 10GB free disk space
- Any modern multi-core CPU
- Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)

**Recommended (GPU Mode)**:
- NVIDIA GPU with 6GB+ VRAM
- CUDA 11.7 or higher
- 8GB+ RAM
- 10GB free disk space

---

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/Thasmika/Voice-to-Image-Representation-Similarity-Detection.git
cd Voice-to-Image-Representation-Similarity-Detection
```

### Step 2: Create Virtual Environment

**Windows**:
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install PyTorch

**For GPU (CUDA 11.8)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU Only**:
```bash
pip install torch torchvision
```

**For macOS (Apple Silicon)**:
```bash
pip install torch torchvision
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI (web framework)
- Whisper (speech recognition)
- Diffusers (Stable Diffusion)
- Transformers (CLIP)
- Librosa (audio processing)
- And all other dependencies

**Installation time**: 5-10 minutes depending on internet speed

---

## First Run

### Model Download (One-Time Only)

On first run, the system will automatically download AI models (~4.5GB total):

1. **Whisper (base)**: ~140MB
2. **Stable Diffusion v1.5**: ~4GB
3. **CLIP ViT-Base**: ~350MB

**Download location**: `~/.cache/huggingface/hub/`

### Start the Server

```bash
python app/main.py
```

**Expected output**:
```
======================================================================
Voice to Image System - Starting Server
======================================================================
Host: 127.0.0.1
Port: 8000
Server URL: http://localhost:8000
API Docs: http://localhost:8000/docs
======================================================================

Starting server on http://localhost:8000
API documentation available at http://localhost:8000/docs

Press CTRL+C to stop the server

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
======================================================================
Starting Voice to Image System...
======================================================================
Python version: 3.11.6
PyTorch version: 2.1.1+cpu
CUDA available: False
Server URL: http://127.0.0.1:8000
----------------------------------------------------------------------
Loading AI models (this may take a few minutes on first run)...
Loading Whisper model...
✓ Whisper model loaded successfully (0.56s)
Loading Stable Diffusion model...
✓ Stable Diffusion model loaded successfully (2.08s)
Loading CLIP model...
✓ CLIP model loaded successfully (3.36s)
----------------------------------------------------------------------
All models loaded successfully!
Setting up directories...
✓ Temp directory ready: temp/
✓ Images directory ready: images/
✓ Static directory ready: static/
======================================================================
Server ready!
Access the system at: http://127.0.0.1:8000
API documentation at: http://127.0.0.1:8000/docs
======================================================================
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**First run time**: 5-10 minutes (model download) + 5-10 seconds (model loading)  
**Subsequent runs**: 5-10 seconds (model loading only)

---

## Running the System

### Method 1: Standard Run (Recommended)

```bash
python app/main.py
```

### Method 2: Custom Port

**Windows**:
```bash
$env:PORT=8001; python app/main.py
```

**macOS/Linux**:
```bash
PORT=8001 python app/main.py
```

### Method 3: Using Uvicorn Directly

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Method 4: Development Mode (Auto-reload)

```bash
uvicorn app.main:app --reload
```

### Stopping the Server

Press `CTRL+C` in the terminal where the server is running.

---

## Verification

### 1. Check Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

You should see the Voice to Image System interface with:
- Upload section with drag-and-drop zone
- Generated images section
- Similarity comparison section

### 2. Check API Documentation

Navigate to:
```
http://localhost:8000/docs
```

You should see interactive API documentation (Swagger UI) with all endpoints.

### 3. Check Health Endpoint

Navigate to:
```
http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "whisper_loaded": true,
  "stable_diffusion_loaded": true,
  "clip_loaded": true
}
```

### 4. Test Image Generation

1. Go to `http://localhost:8000`
2. Click or drag an audio file (WAV, MP3, FLAC, OGG)
3. Wait for processing (30-60 seconds on CPU)
4. Generated image should appear with transcribed text

### 5. Test Similarity Comparison

1. Generate at least 2 images
2. Select 2 images using checkboxes
3. Click "Compare Selected" button
4. Similarity score should appear (0-100%)

---

## Troubleshooting

### Issue 1: Python Not Found

**Error**: `'python' is not recognized as an internal or external command`

**Solution**:
- Verify Python is installed: `python --version`
- Try `python3` instead of `python`
- Add Python to PATH (Windows)
- Reinstall Python with "Add to PATH" option checked

### Issue 2: FFmpeg Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`

**Solution**:
- Install FFmpeg (see Prerequisites section)
- Verify installation: `ffmpeg -version`
- Restart terminal after installation

### Issue 3: Virtual Environment Not Activating

**Windows**:
```bash
# If activation fails, try:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate
```

**macOS/Linux**:
```bash
# Make sure you're in the project directory
source .venv/bin/activate
```

### Issue 4: Port Already in Use

**Error**: `error while attempting to bind on address ('127.0.0.1', 8000): only one usage of each socket address`

**Solution**:
```bash
# Use a different port
PORT=8001 python app/main.py
```

### Issue 5: Models Not Downloading

**Error**: `Cannot load model: model is not cached locally`

**Solution**:
- Check internet connection
- Verify firewall settings
- Check disk space (need 10GB free)
- Try manual download:
  ```bash
  python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
  ```

### Issue 6: Out of Memory

**Error**: `CUDA out of memory` or `MemoryError`

**Solution**:
- Close other applications
- Use CPU mode instead of GPU
- Reduce inference steps in `app/config.py`:
  ```python
  IMAGE_GENERATION_CONFIG = {
      "num_inference_steps": 30,  # Reduced from 50
  }
  ```

### Issue 7: Slow Performance

**Issue**: Image generation takes too long (>60 seconds)

**Solution**:
- This is normal for CPU mode (30-60 seconds)
- For faster performance, use GPU (5-8 seconds)
- Reduce inference steps (see Issue 6)
- Close other applications to free resources

### Issue 8: Import Errors

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
- Ensure virtual environment is activated
- Reinstall dependencies:
  ```bash
  pip install -r requirements.txt
  ```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```bash
# Server configuration
PORT=8000
LOG_LEVEL=INFO

# Model configuration (optional)
WHISPER_MODEL_SIZE=base
```

### Configuration File

Edit `app/config.py` to customize:

```python
# Audio processing
SAMPLE_RATE = 22050
MAX_AUDIO_DURATION = 30.0

# Image generation
WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large
IMAGE_GENERATION_CONFIG = {
    "guidance_scale": 7.5,      # 1-20 (higher = more literal)
    "num_inference_steps": 50,  # 20-100 (higher = better quality)
    "height": 512,
    "width": 512,
}

# Server
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
```

### Changing Whisper Model Size

For better accuracy (slower):
```python
WHISPER_MODEL_SIZE = "small"  # or "medium" or "large"
```

For faster processing (lower accuracy):
```python
WHISPER_MODEL_SIZE = "tiny"
```

---

## GPU Setup (Optional)

### Check GPU Availability

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Install CUDA Toolkit

1. Check your GPU: `nvidia-smi`
2. Download CUDA Toolkit: [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. Install CUDA 11.7 or 11.8
4. Verify: `nvcc --version`

### Install PyTorch with CUDA

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Setup

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

### Performance Comparison

| Mode | Image Generation Time |
|------|----------------------|
| CPU  | 30-60 seconds        |
| GPU  | 5-8 seconds          |

---

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Suite

```bash
pytest tests/test_api_unit.py
pytest tests/test_audio_processor_unit.py
pytest tests/test_image_generator_unit.py
```

### Run with Coverage

```bash
pytest --cov=app tests/
```

---

## Directory Structure

After setup, your directory should look like:

```
Voice-to-Image-Representation-Similarity-Detection/
├── .venv/                  # Virtual environment (not in Git)
├── app/                    # Application code
│   ├── __init__.py
│   ├── main.py            # FastAPI server
│   ├── audio_processor.py
│   ├── image_generator.py
│   ├── similarity_analyzer.py
│   ├── models.py
│   └── config.py
├── static/                 # Web interface
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── tests/                  # Test files
├── temp/                   # Temporary audio files (auto-created)
├── images/                 # Generated images (auto-created)
├── requirements.txt
├── README.md
├── SETUP_GUIDE.md         # This file
└── AI_MODELS_DOCUMENTATION.md
```

---

## Next Steps

After successful setup:

1. **Explore the Web Interface**: Upload audio files and generate images
2. **Try the API**: Use the interactive docs at `/docs`
3. **Read Documentation**: Check `AI_MODELS_DOCUMENTATION.md` for model details
4. **Customize Configuration**: Edit `app/config.py` for your needs
5. **Run Tests**: Verify everything works with `pytest tests/`

---

## Getting Help

### Documentation
- **README.md**: Project overview and quick start
- **AI_MODELS_DOCUMENTATION.md**: Detailed model information
- **IMPLEMENTATION_REPORT.md**: Complete implementation details
- **API Docs**: http://localhost:8000/docs (when server is running)

### Common Issues
- Check the Troubleshooting section above
- Verify all prerequisites are installed
- Ensure virtual environment is activated
- Check logs for error messages

### Support
- GitHub Issues: [Report a bug](https://github.com/Thasmika/Voice-to-Image-Representation-Similarity-Detection/issues)
- Check existing issues for solutions

---

## Uninstallation

To completely remove the system:

1. **Deactivate virtual environment**:
   ```bash
   deactivate
   ```

2. **Delete project directory**:
   ```bash
   cd ..
   rm -rf Voice-to-Image-Representation-Similarity-Detection
   ```

3. **Delete model cache** (optional, frees ~4.5GB):
   ```bash
   # Windows
   rmdir /s /q %USERPROFILE%\.cache\huggingface

   # macOS/Linux
   rm -rf ~/.cache/huggingface
   ```

---

**Setup Guide Version**: 1.0  
**Last Updated**: 2024  
**System Version**: 1.0.0

---

**Congratulations! Your Voice to Image System is ready to use!** 🎉
