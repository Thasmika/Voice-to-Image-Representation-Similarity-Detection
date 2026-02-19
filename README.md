# Voice to Image Representation & Similarity Detection System

## Overview

This system converts voice recordings into **semantic visual images** and performs similarity analysis. When you speak about a beach, the system generates an actual beach image. When you describe a mountain, it creates a mountain scene.

The system uses advanced AI models to understand speech content and generate photorealistic images that match what you're describing.

## Key Features

- **Semantic Image Generation**: Speak "beach sunset" → Get actual beach sunset image
- **Speech Recognition**: Automatic transcription using Whisper AI
- **High-Quality Images**: Photorealistic 512x512 images via Stable Diffusion
- **Similarity Analysis**: Compare images using CLIP embeddings
- **RESTful API**: Easy integration with FastAPI backend
- **Web Interface**: User-friendly interface for audio upload and visualization

## How It Works

The system uses a three-stage pipeline:

1. **Speech-to-Text (Whisper)**: Transcribes your voice to understand content
2. **Text-to-Image (Stable Diffusion)**: Generates photorealistic images from transcribed text
3. **Similarity Analysis (CLIP)**: Compares images to find similar content

From the user's perspective: **Upload Audio → Get Semantic Image**

## Quick Start

### Prerequisites

Before installation, ensure you have:

1. **Python 3.8+** installed
   ```bash
   python --version  # Should show 3.8 or higher
   ```

2. **FFmpeg** installed (required for audio processing)
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `winget install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)
   
   Verify installation:
   ```bash
   ffmpeg -version
   ```

3. **Git** (for cloning the repository)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd voice-to-image-system

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (GPU version - recommended)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
# pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

**Note**: First run will download ~6GB of AI models (Whisper, Stable Diffusion, CLIP). This is a one-time download that takes 5-10 minutes depending on your internet connection.

### Running the Server

```bash
# Make sure virtual environment is activated
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate

# Start the FastAPI server (recommended method)
python app/main.py

# Alternative: Use uvicorn directly
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Run on custom port (via environment variable)
PORT=8080 python app/main.py
```

The server will start and display:
```
Starting Voice to Image System...
Loading AI models (this may take a few minutes on first run)...
Loading Whisper model...
Loading Stable Diffusion model...
Loading CLIP model...
All models loaded successfully!
Server ready!
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Access the system:
- **Web Interface**: Open `http://localhost:8000` in your browser
- **API Documentation**: Visit `http://localhost:8000/docs` for interactive API docs
- **Health Check**: `http://localhost:8000/api/health`

### First Run

On first startup, the system will automatically download required AI models:

1. **Whisper (base)**: ~140MB - Speech recognition model
   - Downloads from: Hugging Face Hub
   - Cached in: `~/.cache/huggingface/hub/`
   
2. **Stable Diffusion 2.1**: ~5GB - Image generation model
   - Downloads from: Hugging Face Hub
   - Cached in: `~/.cache/huggingface/hub/`
   
3. **CLIP ViT-Base**: ~350MB - Similarity analysis model
   - Downloads from: Hugging Face Hub
   - Cached in: `~/.cache/huggingface/hub/`

**Total Download**: ~6GB (one-time only)  
**Download Time**: 5-10 minutes (depending on internet speed)  
**Subsequent Runs**: Instant (models are cached)

**Troubleshooting First Run**:
- Ensure stable internet connection
- Ensure 10GB free disk space
- If download fails, delete `~/.cache/huggingface/` and retry
- Check firewall settings if downloads are blocked

## Usage Examples

### API Usage

#### Generate Image from Audio

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -F "audio_file=@beach_description.wav"
```

Response:
```json
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "image_url": "/api/images/550e8400-e29b-41d4-a716-446655440000",
  "processing_time": 5.23,
  "transcribed_text": "a beautiful beach at sunset with palm trees"
}
```

#### Retrieve Generated Image

```bash
curl "http://localhost:8000/api/images/550e8400-e29b-41d4-a716-446655440000" \
  --output beach.png
```

#### Compare Two Images

```bash
curl -X POST "http://localhost:8000/api/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id_1": "uuid-1",
    "image_id_2": "uuid-2"
  }'
```

Response:
```json
{
  "similarity_score": 0.87,
  "percentage": "87%"
}
```

### Python Usage

```python
import requests

# Upload audio and generate image
with open("my_voice.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/generate",
        files={"audio_file": f}
    )

data = response.json()
print(f"Transcribed: {data['transcribed_text']}")
print(f"Image URL: {data['image_url']}")

# Download the generated image
image_response = requests.get(f"http://localhost:8000{data['image_url']}")
with open("generated_image.png", "wb") as f:
    f.write(image_response.content)
```

## Example Prompts

### Nature Scenes
- "A mountain landscape with snow-capped peaks and pine trees"
- "A tropical beach with turquoise water and white sand"
- "A forest path with sunlight filtering through the trees"

### Urban Scenes
- "A modern city skyline at night with glowing skyscrapers"
- "A quiet street in Paris with cafes and cobblestones"
- "A busy marketplace with colorful stalls"

### Abstract/Artistic
- "A colorful abstract painting with swirling patterns"
- "A dreamy landscape with soft pastel colors"
- "A futuristic scene with neon lights"

## System Requirements

### Minimum (CPU Mode)
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher (3.9-3.11 recommended)
- **RAM**: 8GB minimum, 12GB recommended
- **Disk Space**: 10GB free space (for models and generated images)
- **CPU**: Any modern multi-core CPU (Intel i5/AMD Ryzen 5 or better)
- **Processing Time**: 35-65 seconds per audio clip

### Recommended (GPU Mode)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- **CUDA**: CUDA 11.7 or higher
- **RAM**: 8GB+ system RAM
- **Disk Space**: 10GB free space
- **Processing Time**: 5-8 seconds per audio clip

### Software Dependencies
- **Python**: 3.8+ (with pip)
- **FFmpeg**: Required for audio processing (see installation below)
- **CUDA Toolkit**: Optional, for GPU acceleration

## API Endpoints

### POST /api/generate
Upload audio file and generate semantic image.

**Request**: multipart/form-data with `audio_file`  
**Response**: JSON with image_id, image_url, processing_time, transcribed_text

### POST /api/compare
Compare similarity between two generated images.

**Request**: JSON with `image_id_1` and `image_id_2`  
**Response**: JSON with similarity_score and percentage

### GET /api/images/{image_id}
Retrieve generated image by ID.

**Response**: PNG image file (512x512)

### GET /api/health
Check system health and model status.

**Response**: JSON with status and model loading information

### GET /
Serve web interface (if available).

## Architecture

```
Audio Input
    ↓
[Whisper Model]
    ↓
Transcribed Text
    ↓
[Prompt Enhancement]
    ↓
Enhanced Prompt
    ↓
[Stable Diffusion]
    ↓
Semantic Image (512x512)
    ↓
[CLIP Embeddings]
    ↓
Similarity Analysis
```

## Models Used

### Whisper (Speech Recognition)
- **Model**: OpenAI Whisper (base)
- **Purpose**: Transcribe audio to text
- **Size**: ~140MB
- **Speed**: 1-2 seconds per clip
- **Accuracy**: High accuracy for clear speech in English
- **Source**: [openai/whisper](https://github.com/openai/whisper)
- **License**: MIT License

**Model Variants**:
- `tiny`: 39M params, fastest, lower accuracy
- `base`: 74M params, balanced (default)
- `small`: 244M params, better accuracy
- `medium`: 769M params, high accuracy
- `large`: 1550M params, best accuracy, slowest

### Stable Diffusion (Image Generation)
- **Model**: Stable Diffusion 2.1
- **Purpose**: Generate photorealistic images from text
- **Size**: ~5GB
- **Speed**: 3-5 seconds (GPU), 30-60 seconds (CPU)
- **Output**: 512x512 RGB images
- **Source**: [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- **License**: CreativeML Open RAIL-M License

**Features**:
- Text-to-image generation
- High-quality photorealistic outputs
- Supports various artistic styles
- Configurable guidance scale and inference steps

### CLIP (Similarity Analysis)
- **Model**: CLIP ViT-Base-Patch32
- **Purpose**: Compare image similarity using embeddings
- **Size**: ~350MB
- **Speed**: 0.2-0.5 seconds per comparison
- **Embedding Size**: 512 dimensions
- **Source**: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- **License**: MIT License

**Capabilities**:
- Image-to-embedding conversion
- Cosine similarity computation
- Multi-modal understanding (images and text)

## Configuration

### Environment Variables

The system supports configuration via environment variables:

```bash
# Server port (default: 8000)
export PORT=8080

# Logging level (default: INFO)
export LOG_LEVEL=DEBUG

# Example: Run with custom port
PORT=8080 python app/main.py
```

### Configuration File

Edit `app/config.py` to customize system behavior:

```python
# Audio processing
SAMPLE_RATE = 22050  # Audio sample rate (Hz)
MAX_AUDIO_DURATION = 30.0  # Maximum audio length (seconds)

# Image generation
IMAGE_GENERATION_CONFIG = {
    "guidance_scale": 7.5,      # Prompt adherence (1-20)
    "num_inference_steps": 50,  # Quality vs speed (20-100)
    "height": 512,
    "width": 512,
}

# Server
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
```

### Whisper Model Size

Edit `app/config.py` to change Whisper model size:

```python
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
```

**Trade-offs**:
- `tiny`: Fastest, least accurate, 39M params
- `base`: Balanced (default), 74M params
- `small`: Better accuracy, slower, 244M params
- `medium`: High accuracy, much slower, 769M params
- `large`: Best accuracy, very slow, 1550M params

### Stable Diffusion Settings

Adjust generation parameters in `app/config.py`:

```python
IMAGE_GENERATION_CONFIG = {
    "guidance_scale": 7.5,        # How closely to follow prompt (1-20)
    "num_inference_steps": 50,    # Quality vs speed (20-100)
    "height": 512,                # Output height
    "width": 512,                 # Output width
}
```

**Parameter Guide**:
- **guidance_scale**: Higher = more literal, Lower = more creative
  - 5-7: Creative, artistic
  - 7-10: Balanced (recommended)
  - 10-15: Very literal
  
- **num_inference_steps**: Higher = better quality, slower
  - 20-30: Fast, lower quality
  - 40-60: Balanced (recommended)
  - 70-100: High quality, slow

### Directory Paths

All paths are configurable in `app/config.py`:

```python
TEMP_DIR = PROJECT_ROOT / "temp"      # Temporary audio files
IMAGES_DIR = PROJECT_ROOT / "images"  # Generated images
STATIC_DIR = PROJECT_ROOT / "static"  # Web interface files
```

## Performance

### Processing Time
- **GPU**: 5-8 seconds per audio clip
  - Whisper: 1-2 seconds
  - Stable Diffusion: 3-5 seconds
  - Overhead: 1 second

- **CPU**: 35-65 seconds per audio clip
  - Whisper: 5-10 seconds
  - Stable Diffusion: 30-60 seconds
  - Overhead: 1 second

### Memory Usage
- **GPU**: 6GB VRAM + 4GB RAM
- **CPU**: 10GB RAM

## Tips for Best Results

1. **Be Descriptive**: Use detailed descriptions with colors, lighting, and atmosphere
2. **Speak Clearly**: Minimize background noise for better transcription
3. **Use Visual Language**: Describe what you want to see, not abstract concepts
4. **Keep It Concise**: 1-2 sentences work best

## Troubleshooting

### Models Not Loading
- Check internet connection (first run only)
- Ensure 10GB free disk space
- Wait patiently during first download

### Out of Memory
- Close other applications
- Use CPU mode (slower but works)
- Reduce inference steps

### Poor Transcription
- Improve audio quality
- Speak more clearly
- Use larger Whisper model

### Images Don't Match Description
- Be more descriptive
- Use visual language
- Try different phrasings

## Documentation

- **QUICKSTART_SEMANTIC.md**: Quick start guide
- **SEMANTIC_GENERATION_GUIDE.md**: Comprehensive user guide
- **IMPLEMENTATION_CHANGES.md**: Technical implementation details
- **API Documentation**: Available at `/docs` when server is running

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_api_unit.py
pytest tests/test_api_properties.py

# Test semantic generation
python test_semantic_generation.py
```

## Development

### Project Structure

```
voice-to-image-system/
├── app/
│   ├── main.py              # FastAPI server
│   ├── audio_processor.py   # Audio validation and loading
│   ├── image_generator.py   # Whisper + Stable Diffusion
│   ├── similarity_analyzer.py # CLIP similarity
│   ├── models.py            # Pydantic models
│   └── __init__.py
├── tests/
│   ├── test_api_unit.py
│   ├── test_api_properties.py
│   ├── test_audio_processor_*.py
│   ├── test_image_generator_*.py
│   └── test_similarity_analyzer_*.py
├── static/                  # Web interface (future)
├── temp/                    # Temporary audio files
├── images/                  # Generated images
├── requirements.txt
└── README.md
```

## License

This project uses the following open-source models:
- **Whisper**: MIT License (OpenAI)
- **Stable Diffusion**: CreativeML Open RAIL-M License
- **CLIP**: MIT License (OpenAI)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues or questions:
1. Check the documentation
2. Review error messages
3. Verify system requirements
4. Check model download status

## Acknowledgments

- OpenAI for Whisper and CLIP models
- Stability AI for Stable Diffusion
- Hugging Face for model hosting and diffusers library
- FastAPI for the web framework
