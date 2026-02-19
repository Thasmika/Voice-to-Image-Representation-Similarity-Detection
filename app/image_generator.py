"""Image generation module for voice-to-image system.

This module handles conversion of audio to semantic visual images using:
1. Whisper (speech-to-text) - extracts semantic content from audio
2. Stable Diffusion (text-to-image) - generates semantic images from transcribed text
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
import whisper


@dataclass
class GeneratedImage:
    """Container for generated image with metadata."""
    image_id: str
    image_data: Image.Image
    file_path: str
    created_at: datetime
    source_audio: str
    transcribed_text: Optional[str] = None  # Add transcribed text field


# Global pipeline instances (singleton pattern)
_sd_pipeline = None
_whisper_model = None
_device = None


def load_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """Load Whisper speech recognition model.
    
    Args:
        model_size: Model size - "tiny", "base", "small", "medium", "large"
                   (base is good balance of speed and accuracy)
        
    Returns:
        Whisper model instance
    """
    global _whisper_model, _device
    
    if _whisper_model is not None:
        return _whisper_model
    
    # Detect device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model ({model_size}) on {_device}...")
    
    # Load Whisper model
    _whisper_model = whisper.load_model(model_size, device=_device)
    
    print(f"Whisper model loaded successfully on {_device}")
    return _whisper_model


def load_stable_diffusion_model(model_name: str = "runwayml/stable-diffusion-v1-5") -> StableDiffusionPipeline:
    """Load Stable Diffusion model pipeline.
    
    Args:
        model_name: Hugging Face model identifier
        
    Returns:
        StableDiffusionPipeline instance
    """
    global _sd_pipeline, _device
    
    if _sd_pipeline is not None:
        return _sd_pipeline
    
    # Detect device (GPU or CPU)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Stable Diffusion model on {_device}...")
    
    try:
        # Try loading with trust_remote_code and local_files_only fallback
        _sd_pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for faster inference
            use_auth_token=False
        )
    except Exception as e:
        print(f"Failed to load {model_name}, trying alternative model...")
        # Fallback to CompVis model which is more widely cached
        model_name = "CompVis/stable-diffusion-v1-4"
        _sd_pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            safety_checker=None,
            use_auth_token=False
        )
    
    _sd_pipeline = _sd_pipeline.to(_device)
    
    # Enable memory optimizations for GPU
    if _device == "cuda":
        _sd_pipeline.enable_attention_slicing()
        try:
            _sd_pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass  # xformers not available, continue without it
    
    print(f"Stable Diffusion model loaded successfully on {_device}")
    return _sd_pipeline


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using Whisper.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Transcribed text
    """
    # Load Whisper model if not already loaded
    model = load_whisper_model()
    
    # Transcribe audio
    result = model.transcribe(audio_path)
    
    # Extract text
    text = result["text"].strip()
    
    print(f"Transcribed audio: '{text}'")
    
    return text


def enhance_prompt(text: str) -> str:
    """Enhance transcribed text to create better image generation prompt.
    
    Args:
        text: Raw transcribed text
        
    Returns:
        Enhanced prompt for image generation
    """
    # If text is empty or very short, use a default
    if not text or len(text) < 3:
        return "abstract colorful artwork, detailed, high quality"
    
    # Add quality modifiers to improve image generation
    enhanced = f"{text}, detailed, high quality, photorealistic, 8k"
    
    return enhanced


def generate_image_from_audio(audio_path: str,
                             guidance_scale: float = 7.5,
                             num_inference_steps: int = 50,
                             seed: Optional[int] = None) -> tuple[Image.Image, str]:
    """Generate semantic image from audio file.
    
    This function:
    1. Transcribes audio to text using Whisper
    2. Generates image from text using Stable Diffusion
    
    Args:
        audio_path: Path to audio file
        guidance_scale: How closely to follow the prompt (7.5 is good default)
        num_inference_steps: Number of denoising steps (50 is good balance)
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Tuple of (PIL Image object, transcribed text)
    """
    # Step 1: Transcribe audio to text
    text = transcribe_audio(audio_path)
    
    # Step 2: Enhance prompt for better image generation
    prompt = enhance_prompt(text)
    
    # Step 3: Generate image from prompt
    image = generate_image_from_text(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed
    )
    
    return image, text


def generate_image_from_text(prompt: str,
                             guidance_scale: float = 7.5,
                             num_inference_steps: int = 50,
                             seed: Optional[int] = None) -> Image.Image:
    """Generate image from text prompt using Stable Diffusion.
    
    Args:
        prompt: Text description for image generation
        guidance_scale: How closely to follow the prompt (7.5 is good default)
        num_inference_steps: Number of denoising steps (50 is good balance)
        seed: Random seed for reproducibility (optional)
        
    Returns:
        PIL Image object (512x512 pixels)
    """
    # Load Stable Diffusion model if not already loaded
    pipeline = load_stable_diffusion_model()
    
    # Set seed for reproducibility if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_device).manual_seed(seed)
    
    # Generate image
    with torch.no_grad():
        output = pipeline(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=512,
            width=512
        )
        
        image = output.images[0]  # PIL Image
    
    return image


def spectrogram_to_image(audio_path: str,
                        guidance_scale: float = 7.5,
                        num_inference_steps: int = 50,
                        seed: Optional[int] = None) -> tuple[Image.Image, str]:
    """Convert audio to semantic image (main entry point).
    
    This is the main function that converts audio to a semantic image
    by transcribing the audio and generating an image from the content.
    
    Args:
        audio_path: Path to audio file
        guidance_scale: How closely to follow the prompt (7.5 is good default)
        num_inference_steps: Number of denoising steps (50 is good balance)
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Tuple of (PIL Image object, transcribed text)
    """
    return generate_image_from_audio(
        audio_path=audio_path,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed
    )


import uuid
import os


def save_image(image: Image.Image, source_audio: str, output_dir: str = 'images',
              transcribed_text: Optional[str] = None) -> GeneratedImage:
    """Save generated image to disk with metadata.
    
    Args:
        image: PIL Image object to save
        source_audio: Original audio filename
        output_dir: Directory to save images (default: 'images')
        transcribed_text: Transcribed text from audio (optional)
        
    Returns:
        GeneratedImage object with metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate UUID for image identifier
    image_id = str(uuid.uuid4())
    
    # Create file path
    file_path = os.path.join(output_dir, f"{image_id}.png")
    
    # Save image as PNG
    image.save(file_path, format='PNG')
    
    # Create and return GeneratedImage object
    return GeneratedImage(
        image_id=image_id,
        image_data=image,
        file_path=file_path,
        created_at=datetime.now(),
        source_audio=source_audio,
        transcribed_text=transcribed_text
    )
