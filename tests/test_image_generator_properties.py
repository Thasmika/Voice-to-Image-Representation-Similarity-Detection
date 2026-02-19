"""Property-based tests for image generator module.

Feature: voice-to-image-system
"""

import pytest
import numpy as np
import tempfile
import os
import wave
import struct
from hypothesis import given, strategies as st, settings
from PIL import Image

from app.image_generator import (
    generate_image_from_text, save_image, GeneratedImage,
    enhance_prompt, transcribe_audio
)


# Helper function to create test WAV files
def create_wav_file(file_path: str, duration: float = 2.0, sample_rate: int = 22050):
    """Create a valid WAV file with sine wave audio."""
    num_samples = int(duration * sample_rate)
    frequency = 440.0  # A4 note
    
    with wave.open(file_path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            value = int(32767.0 * np.sin(2.0 * np.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframes(data)


@given(
    st.text(min_size=5, max_size=100, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')))
)
@settings(max_examples=2, deadline=None)  # Reduced examples for faster testing with Stable Diffusion
def test_property_6_image_generation_success(prompt_text):
    """
    Feature: voice-to-image-system, Property 6: Image Generation Success
    For any valid text prompt, the image generator should produce a 
    high-quality RGB image in PNG format with dimensions of 512x512 pixels.
    
    Validates: Requirements 3.1, 3.3
    """
    # Generate image from text prompt (using reduced steps for testing)
    image = generate_image_from_text(prompt_text, num_inference_steps=5, seed=42)
    
    # Property: Should return a PIL Image object
    assert isinstance(image, Image.Image), "Output should be a PIL Image"
    
    # Property: Image should be in RGB mode
    assert image.mode == 'RGB', f"Image mode should be RGB, got {image.mode}"
    
    # Property: Image dimensions should be 512x512 (Stable Diffusion output)
    width, height = image.size
    assert width == 512, f"Image width should be 512 pixels, got {width}"
    assert height == 512, f"Image height should be 512 pixels, got {height}"


@given(
    st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')))
)
@settings(max_examples=2, deadline=None)  # Reduced examples for faster testing with Stable Diffusion
def test_property_7_image_generation_determinism(prompt_text):
    """
    Feature: voice-to-image-system, Property 7: Image Generation Determinism
    For any valid text prompt, generating the image multiple times with the same seed
    should produce identical outputs (deterministic processing).
    
    Validates: Requirements 3.4
    """
    # Generate image from text prompt twice with same seed
    image1 = generate_image_from_text(prompt_text, num_inference_steps=5, seed=42)
    image2 = generate_image_from_text(prompt_text, num_inference_steps=5, seed=42)
    
    # Property: Both images should have identical dimensions
    assert image1.size == image2.size, \
        f"Images should have same dimensions, got {image1.size} and {image2.size}"
    
    # Property: Both images should have identical pixel data
    # Convert to numpy arrays for comparison
    pixels1 = np.array(image1)
    pixels2 = np.array(image2)
    
    # Check if arrays are identical
    assert np.array_equal(pixels1, pixels2), \
        "Images generated from same prompt with same seed should be identical (deterministic)"
    
    # Additional check: Verify the images are byte-for-byte identical
    assert pixels1.shape == pixels2.shape, \
        f"Image arrays should have same shape, got {pixels1.shape} and {pixels2.shape}"
    
    # Check that the difference is exactly zero
    diff = np.abs(pixels1.astype(int) - pixels2.astype(int))
    max_diff = np.max(diff)
    assert max_diff == 0, \
        f"Images should be identical, but max pixel difference is {max_diff}"


@given(
    st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')))
)
@settings(max_examples=3, deadline=None)
def test_property_prompt_enhancement(text):
    """
    Feature: voice-to-image-system, Property: Prompt Enhancement
    For any text input, the prompt enhancement should add quality modifiers
    and preserve the original content.
    
    Validates: Requirements 3.1
    """
    # Enhance the prompt
    enhanced = enhance_prompt(text)
    
    # Property: Enhanced prompt should be a string
    assert isinstance(enhanced, str), "Enhanced prompt should be a string"
    
    # Property: Enhanced prompt should not be empty
    assert len(enhanced) > 0, "Enhanced prompt should not be empty"
    
    # Property: For non-empty input, original text should be preserved (or default used)
    if len(text.strip()) >= 3:
        # Original text should be in enhanced prompt
        assert text.lower() in enhanced.lower() or "abstract" in enhanced.lower(), \
            "Enhanced prompt should preserve original text or use default"
    else:
        # Short text should use default prompt
        assert "abstract" in enhanced.lower(), \
            "Short text should use default abstract prompt"


@given(
    st.floats(min_value=0.5, max_value=5.0)
)
@settings(max_examples=2, deadline=None)
def test_property_whisper_transcription(duration):
    """
    Feature: voice-to-image-system, Property: Whisper Transcription
    For any valid audio file, Whisper should return a string transcription
    (may be empty for non-speech audio).
    
    Validates: Requirements 3.1
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a test WAV file
        create_wav_file(tmp_path, duration=duration)
        
        # Transcribe audio
        text = transcribe_audio(tmp_path)
        
        # Property: Transcription should return a string
        assert isinstance(text, str), "Transcription should return a string"
        
        # Property: String should not be None
        assert text is not None, "Transcription should not be None"
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
