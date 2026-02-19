# Quick Start: Semantic Voice-to-Image

## What This Does

Speak into your microphone → Get a photorealistic image of what you described!

- Say "beach sunset" → Get beach sunset image
- Say "mountain landscape" → Get mountain image  
- Say "city at night" → Get city skyline image

## Installation (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This will install Whisper and Stable Diffusion. First run will download ~6GB of models.

### 2. Start the Server

```bash
python app/main.py
```

**First run**: Models will download automatically (~5-10 minutes)
**Subsequent runs**: Instant startup (models are cached)

### 3. Wait for "Server ready!" message

```
Loading Whisper model...
Whisper model loaded successfully on cuda
Loading Stable Diffusion model...
Stable Diffusion model loaded successfully on cuda
Loading CLIP model...
CLIP model loaded successfully on cuda
All models loaded successfully!
Server ready!
```

## Usage

### Option 1: Web Interface (Coming Soon)

Open browser: `http://localhost:8000`

1. Click "Upload Audio"
2. Record or select audio file
3. Click "Generate"
4. View your semantic image!

### Option 2: API (Available Now)

```bash
# Generate image from audio
curl -X POST "http://localhost:8000/api/generate" \
  -F "audio_file=@my_voice.wav"

# Response:
{
  "image_id": "abc-123",
  "image_url": "/api/images/abc-123",
  "processing_time": 5.2,
  "transcribed_text": "a beautiful beach at sunset"
}

# Download the image
curl "http://localhost:8000/api/images/abc-123" --output beach.png
```

### Option 3: Python Script

```python
import requests

# Upload audio
with open("my_voice.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/generate",
        files={"audio_file": f}
    )

data = response.json()
print(f"Transcribed: {data['transcribed_text']}")
print(f"Image URL: {data['image_url']}")

# Download image
image_response = requests.get(f"http://localhost:8000{data['image_url']}")
with open("generated_image.png", "wb") as f:
    f.write(image_response.content)

print("Image saved to: generated_image.png")
```

## Tips for Best Results

### 1. Speak Clearly
- Use a good microphone
- Minimize background noise
- Speak at normal pace

### 2. Be Descriptive
- ✅ "A peaceful beach at sunset with palm trees"
- ❌ "Beach"

### 3. Use Visual Language
- Mention colors: "blue sky", "red flowers"
- Describe lighting: "sunset", "moonlight", "bright day"
- Add atmosphere: "peaceful", "dramatic", "cozy"

## Example Prompts

### Nature Scenes
- "A mountain landscape with snow-capped peaks and pine trees"
- "A tropical beach with turquoise water and white sand"
- "A forest path with sunlight filtering through the trees"

### Urban Scenes
- "A modern city skyline at night with glowing skyscrapers"
- "A quiet street in Paris with cafes and cobblestones"
- "A busy marketplace with colorful stalls and people"

### Abstract/Artistic
- "A colorful abstract painting with swirling patterns"
- "A dreamy landscape with soft pastel colors"
- "A futuristic scene with neon lights and technology"

## System Requirements

### Minimum (CPU Mode)
- 8GB RAM
- 10GB disk space
- Any modern CPU
- **Processing time**: 35-65 seconds per audio

### Recommended (GPU Mode)
- 6GB+ VRAM (NVIDIA GPU)
- 8GB+ RAM
- 10GB disk space
- CUDA-compatible GPU
- **Processing time**: 5-8 seconds per audio

## Troubleshooting

### "Models not loading"
- Check internet connection (first run only)
- Ensure 10GB free disk space
- Wait patiently (downloads take 5-10 minutes)

### "Out of memory"
- Close other applications
- Use CPU mode (slower but works)
- Restart the server

### "Transcription is empty"
- Ensure audio contains speech (not just music)
- Check audio quality
- Speak more clearly

### "Images don't match my description"
- Be more descriptive
- Use visual language
- Try different phrasings

## What's Next?

1. **Try it out**: Record some audio and generate images!
2. **Experiment**: Try different descriptions and styles
3. **Compare**: Use `/api/compare` to find similar images
4. **Share**: Show off your generated images!

## Need Help?

- Read: `SEMANTIC_GENERATION_GUIDE.md` (detailed guide)
- Read: `IMPLEMENTATION_CHANGES.md` (technical details)
- Check: `/api/health` endpoint for model status

## Example Session

```bash
# 1. Start server
python app/main.py

# 2. Check health
curl http://localhost:8000/api/health

# 3. Generate image
curl -X POST "http://localhost:8000/api/generate" \
  -F "audio_file=@beach_description.wav" \
  | jq .

# Output:
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "image_url": "/api/images/550e8400-e29b-41d4-a716-446655440000",
  "processing_time": 5.23,
  "transcribed_text": "a beautiful beach with blue water and white sand at sunset"
}

# 4. Download image
curl "http://localhost:8000/api/images/550e8400-e29b-41d4-a716-446655440000" \
  --output my_beach.png

# 5. View image
# Open my_beach.png in your image viewer
```

## Have Fun!

The system is now ready to turn your voice into beautiful images. Experiment, explore, and enjoy! 🎨🎤🖼️
