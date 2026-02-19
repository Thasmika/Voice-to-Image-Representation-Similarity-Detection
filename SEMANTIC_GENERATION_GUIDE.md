# Semantic Voice-to-Image Generation Guide

## Overview

The system now generates **semantic images** from voice recordings. When you speak about a beach, the system generates an image of a beach. When you describe a mountain, it creates a mountain scene.

## How It Works

The system uses a two-stage pipeline (hidden from the user):

1. **Speech-to-Text (Whisper)**: Transcribes your voice to understand what you're saying
2. **Text-to-Image (Stable Diffusion)**: Generates a photorealistic image based on the transcribed content

From the user's perspective, it's still just: **Upload Audio → Get Image**

## Key Features

### Semantic Understanding
- Speaks "beach sunset" → Generates beach sunset image
- Speaks "mountain landscape" → Generates mountain landscape image
- Speaks "city skyline at night" → Generates city skyline image

### Automatic Enhancement
The system automatically enhances your speech to create better images:
- Adds quality modifiers: "detailed", "high quality", "photorealistic"
- Optimizes prompts for better visual results

### Transcription Access
The API response now includes the transcribed text, so you can see what the system understood from your voice.

## Models Used

### Whisper (Speech Recognition)
- **Model**: OpenAI Whisper (base)
- **Purpose**: Transcribe audio to text
- **Size**: ~140MB
- **Speed**: ~1-2 seconds per audio clip

### Stable Diffusion (Image Generation)
- **Model**: Stable Diffusion 2.1
- **Purpose**: Generate photorealistic images from text
- **Size**: ~5GB
- **Speed**: ~3-5 seconds on GPU, ~30-60 seconds on CPU

### CLIP (Similarity Analysis)
- **Model**: CLIP ViT-Base-Patch32
- **Purpose**: Compare image similarity
- **Size**: ~350MB
- **Speed**: ~0.2-0.5 seconds per comparison

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `openai-whisper`: Speech recognition
- `diffusers`: Stable Diffusion pipeline
- `transformers`: Model loading utilities

### 2. First Run (Model Download)

On first run, the system will download models (~5-6GB total):
- Whisper base model (~140MB)
- Stable Diffusion 2.1 (~5GB)
- CLIP ViT-Base (~350MB)

Models are cached in `~/.cache/huggingface/` and `~/.cache/whisper/`

## Usage

### Starting the Server

```bash
python app/main.py
```

Or with uvicorn:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### API Endpoints

#### POST /api/generate

Upload audio and generate semantic image.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/generate" \
  -F "audio_file=@beach_description.wav"
```

**Response:**
```json
{
  "image_id": "uuid-here",
  "image_url": "/api/images/uuid-here",
  "processing_time": 5.23,
  "transcribed_text": "a beautiful beach with blue water and white sand at sunset"
}
```

#### GET /api/images/{image_id}

Retrieve generated image.

**Request:**
```bash
curl "http://localhost:8000/api/images/uuid-here" --output image.png
```

#### POST /api/compare

Compare two generated images.

**Request:**
```json
{
  "image_id_1": "uuid-1",
  "image_id_2": "uuid-2"
}
```

**Response:**
```json
{
  "similarity_score": 0.87,
  "percentage": "87%"
}
```

#### GET /api/health

Check system health and model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "whisper_loaded": true,
  "stable_diffusion_loaded": true,
  "clip_loaded": true
}
```

## Example Use Cases

### 1. Voice Journaling
Record voice notes about your day, and get visual representations:
- "Today I went to the park and saw beautiful flowers"
- → Image of a park with flowers

### 2. Story Visualization
Describe scenes from stories:
- "A dark forest with tall trees and moonlight"
- → Image of a dark forest scene

### 3. Memory Capture
Describe memories and get visual reminders:
- "My childhood home with a red door and garden"
- → Image of a house with red door and garden

### 4. Creative Brainstorming
Speak ideas and see them visualized:
- "Futuristic city with flying cars"
- → Image of futuristic cityscape

## Tips for Best Results

### 1. Be Descriptive
- ✅ Good: "A peaceful beach at sunset with palm trees and orange sky"
- ❌ Less good: "Beach"

### 2. Speak Clearly
- Whisper works best with clear audio
- Minimize background noise
- Speak at a normal pace

### 3. Use Visual Language
- Describe colors, lighting, atmosphere
- Mention specific objects and scenes
- Include mood and style descriptors

### 4. Keep It Concise
- 1-2 sentences work best
- Focus on key visual elements
- Avoid overly complex descriptions

## Performance

### GPU (Recommended)
- **Total time**: 5-8 seconds per audio clip
  - Whisper: 1-2 seconds
  - Stable Diffusion: 3-5 seconds
  - Overhead: 1 second

### CPU (Slower)
- **Total time**: 35-65 seconds per audio clip
  - Whisper: 5-10 seconds
  - Stable Diffusion: 30-60 seconds
  - Overhead: 1 second

### Memory Requirements
- **GPU**: 6GB+ VRAM recommended
- **CPU**: 8GB+ RAM minimum, 16GB+ recommended

## Troubleshooting

### Models Not Loading
```
Error: Failed to load models
```
**Solution**: Ensure you have enough disk space (~6GB) and internet connection for first download.

### Out of Memory (GPU)
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce batch size (already set to 1)
- Use CPU mode (slower but works)
- Close other GPU applications

### Transcription Empty
```
Transcribed text: ''
```
**Solution**:
- Ensure audio contains speech (not just music/noise)
- Check audio quality and volume
- Try speaking more clearly

### Slow Generation
**Solution**:
- Use GPU if available (10x faster)
- Reduce `num_inference_steps` (default: 50, try 30)
- Use smaller Whisper model (tiny/base instead of medium)

## Configuration

### Whisper Model Size
Edit `app/image_generator.py`:

```python
# Options: "tiny", "base", "small", "medium", "large"
load_whisper_model(model_size="base")  # Default
```

Trade-offs:
- **tiny**: Fastest, least accurate (~40MB)
- **base**: Good balance (~140MB) ← Default
- **small**: Better accuracy (~460MB)
- **medium**: High accuracy (~1.5GB)
- **large**: Best accuracy (~3GB)

### Stable Diffusion Settings
Edit `app/image_generator.py`:

```python
generate_image_from_text(
    prompt=prompt,
    guidance_scale=7.5,        # How closely to follow prompt (7.5 is good)
    num_inference_steps=50,    # Quality vs speed (50 is good balance)
    seed=None                  # Set for reproducibility
)
```

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
```

## Comparison with Previous Approach

### Old Approach (Spectrogram)
- ✅ Fast (~1 second)
- ✅ No model downloads
- ❌ Abstract patterns only
- ❌ No semantic understanding

### New Approach (Semantic)
- ✅ Semantic understanding
- ✅ Photorealistic images
- ✅ Matches spoken content
- ⚠️ Slower (~5-60 seconds)
- ⚠️ Requires model downloads (~6GB)

## Future Enhancements

Possible improvements:
1. **Faster models**: Use distilled Stable Diffusion (SDXS)
2. **Better transcription**: Fine-tune Whisper on specific domains
3. **Style control**: Allow users to specify art styles
4. **Multi-language**: Support non-English speech
5. **Real-time**: Optimize for faster generation

## License & Attribution

- **Whisper**: MIT License (OpenAI)
- **Stable Diffusion**: CreativeML Open RAIL-M License
- **CLIP**: MIT License (OpenAI)

## Support

For issues or questions:
1. Check this guide
2. Review error messages
3. Check model download status
4. Verify system requirements
