"""Download AI models for Voice to Image system."""

import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel

print("=" * 60)
print("Downloading AI Models for Voice to Image System")
print("=" * 60)
print()

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print()

# Download Stable Diffusion
print("1. Downloading Stable Diffusion v1.5 (~4GB)...")
print("   This may take 5-15 minutes depending on your connection...")
try:
    sd_pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        use_auth_token=False
    )
    print("   ✓ Stable Diffusion downloaded successfully!")
except Exception as e:
    print(f"   ✗ Failed to download Stable Diffusion: {e}")
    print("   Trying alternative model...")
    try:
        sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            use_auth_token=False
        )
        print("   ✓ Alternative Stable Diffusion model downloaded!")
    except Exception as e2:
        print(f"   ✗ Failed to download alternative model: {e2}")

print()

# Download CLIP
print("2. Downloading CLIP model (~350MB)...")
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("   ✓ CLIP model downloaded successfully!")
except Exception as e:
    print(f"   ✗ Failed to download CLIP: {e}")

print()
print("=" * 60)
print("Model download complete!")
print("=" * 60)
print()
print("Note: Whisper model (~140MB) was already downloaded during server startup.")
print()
print("You can now start the server with:")
print("  uvicorn app.main:app --host 127.0.0.1 --port 8000")
