"""Unit tests for image generator edge cases.

Feature: voice-to-image-system
"""

import pytest
import numpy as np
import tempfile
import os
import wave
import struct
from PIL import Image

from app.image_generator import (
    generate_image_from_text, save_image, GeneratedImage,
    transcribe_audio, enhance_prompt
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


def test_text_to_image_generation():
    """Test image generation from text prompts.
    
    Requirements: 3.1, 3.3
    
    Note: Using reduced inference steps for faster testing.
    """
    # Test different text prompts
    test_prompts = [
        "a beach at sunset",
        "a mountain landscape",
        "a city skyline",
    ]
    
    for prompt in test_prompts:
        # Generate image with reduced steps for faster testing
        image = generate_image_from_text(prompt, num_inference_steps=10, seed=42)
        
        # Verify it's a PIL Image
        assert isinstance(image, Image.Image), \
            f"Failed for prompt '{prompt}': Output should be PIL Image"
        
        # Verify RGB mode
        assert image.mode == 'RGB', \
            f"Failed for prompt '{prompt}': Image should be RGB mode"
        
        # Verify dimensions are 512x512 (Stable Diffusion output)
        width, height = image.size
        assert width == 512 and height == 512, \
            f"Failed for prompt '{prompt}': Image should be 512x512, got {width}x{height}"


def test_png_format_and_dimensions():
    """Test that generated images are in PNG format with correct dimensions.
    
    Requirements: 3.3
    """
    # Generate image from text prompt
    prompt = "a beautiful landscape"
    image = generate_image_from_text(prompt, num_inference_steps=10, seed=42)
    
    # Verify image format
    assert image.format is None or image.format == 'PNG', \
        "Image should be in PNG format or have no format (in-memory)"
    
    # Verify RGB mode (PNG supports RGB)
    assert image.mode == 'RGB', "Image should be in RGB mode"
    
    # Verify dimensions are 512x512 (Stable Diffusion output)
    width, height = image.size
    assert width == 512, f"Width should be 512 pixels, got {width}"
    assert height == 512, f"Height should be 512 pixels, got {height}"
    
    # Save to temporary file and verify PNG format
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        image.save(tmp_path, format='PNG')
        
        # Load the saved image and verify format
        loaded_image = Image.open(tmp_path)
        assert loaded_image.format == 'PNG', "Saved image should be PNG format"
        assert loaded_image.mode == 'RGB', "Saved image should be RGB mode"
        assert loaded_image.size == image.size, "Saved image should have same dimensions"
        loaded_image.close()  # Close the image before deleting
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_file_saving_and_retrieval():
    """Test saving generated images to disk and retrieving them.
    
    Requirements: 3.3
    
    Note: Using reduced inference steps for faster testing.
    """
    # Generate image from text
    prompt = "a simple scene"
    image = generate_image_from_text(prompt, num_inference_steps=10, seed=42)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save image using save_image function
        generated_image = save_image(
            image, 
            "test_audio.wav", 
            output_dir=tmp_dir,
            transcribed_text=prompt
        )
        
        # Verify GeneratedImage object
        assert isinstance(generated_image, GeneratedImage), \
            "save_image should return GeneratedImage object"
        assert generated_image.image_id is not None, "Image ID should be set"
        assert generated_image.source_audio == "test_audio.wav", \
            "Source audio should match input"
        assert generated_image.transcribed_text == prompt, \
            "Transcribed text should match input"
        assert generated_image.file_path.endswith('.png'), \
            "File path should end with .png"
        assert os.path.exists(generated_image.file_path), \
            "Image file should exist on disk"
        
        # Retrieve the saved image
        loaded_image = Image.open(generated_image.file_path)
        
        # Verify loaded image properties
        assert loaded_image.format == 'PNG', "Loaded image should be PNG format"
        assert loaded_image.mode == 'RGB', "Loaded image should be RGB mode"
        assert loaded_image.size == image.size, \
            "Loaded image should have same dimensions as original"
        
        # Verify pixel data is preserved
        original_pixels = np.array(image)
        loaded_pixels = np.array(loaded_image)
        assert np.array_equal(original_pixels, loaded_pixels), \
            "Loaded image should have identical pixel data to original"
        
        loaded_image.close()


def test_prompt_enhancement():
    """Test that prompt enhancement adds quality modifiers.
    
    Requirements: 3.1
    """
    # Test with normal text
    text = "a beach"
    enhanced = enhance_prompt(text)
    assert "detailed" in enhanced.lower(), "Should add 'detailed'"
    assert "high quality" in enhanced.lower(), "Should add 'high quality'"
    assert "beach" in enhanced.lower(), "Should preserve original text"
    
    # Test with empty text
    empty_enhanced = enhance_prompt("")
    assert len(empty_enhanced) > 0, "Should return default prompt for empty text"
    assert "abstract" in empty_enhanced.lower(), "Should use default prompt"
    
    # Test with very short text
    short_enhanced = enhance_prompt("a")
    assert "abstract" in short_enhanced.lower(), "Should use default for very short text"


def test_whisper_transcription():
    """Test Whisper transcription with audio file.
    
    Requirements: 3.1
    
    Note: This test uses a tone, so transcription will be empty or noise.
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a test WAV file
        create_wav_file(tmp_path, duration=1.0)
        
        # Transcribe audio
        text = transcribe_audio(tmp_path)
        
        # Verify transcription returns a string (may be empty for tone)
        assert isinstance(text, str), "Transcription should return a string"
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_save_image_creates_directory():
    """Test that save_image creates output directory if it doesn't exist.
    
    Requirements: 3.3
    
    Note: Using reduced inference steps for faster testing.
    """
    # Generate image from text
    prompt = "a simple scene"
    image = generate_image_from_text(prompt, num_inference_steps=10, seed=42)
    
    # Use a non-existent directory path
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = os.path.join(tmp_dir, 'nested', 'output', 'dir')
        
        # Directory should not exist yet
        assert not os.path.exists(output_dir), "Directory should not exist initially"
        
        # Save image
        generated_image = save_image(image, "test.wav", output_dir=output_dir)
        
        # Directory should now exist
        assert os.path.exists(output_dir), "save_image should create output directory"
        assert os.path.exists(generated_image.file_path), "Image file should exist"


def test_multiple_images_unique_ids():
    """Test that multiple saved images get unique IDs.
    
    Requirements: 3.3
    
    Note: Using reduced inference steps for faster testing.
    """
    # Generate image from text
    prompt = "a simple scene"
    image = generate_image_from_text(prompt, num_inference_steps=10, seed=42)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save multiple images
        generated_images = []
        for i in range(3):
            gen_img = save_image(image, f"test_{i}.wav", output_dir=tmp_dir)
            generated_images.append(gen_img)
        
        # Verify all IDs are unique
        image_ids = [img.image_id for img in generated_images]
        assert len(image_ids) == len(set(image_ids)), \
            "All image IDs should be unique"
        
        # Verify all files exist
        for gen_img in generated_images:
            assert os.path.exists(gen_img.file_path), \
                f"Image file {gen_img.file_path} should exist"


def test_image_metadata():
    """Test that GeneratedImage contains correct metadata.
    
    Requirements: 3.3
    
    Note: Using reduced inference steps for faster testing.
    """
    # Generate image from text
    prompt = "a simple scene"
    image = generate_image_from_text(prompt, num_inference_steps=10, seed=42)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        source_audio = "my_audio_file.wav"
        transcribed_text = "test transcription"
        generated_image = save_image(
            image, 
            source_audio, 
            output_dir=tmp_dir,
            transcribed_text=transcribed_text
        )
        
        # Verify metadata
        assert generated_image.image_id is not None, "Image ID should be set"
        assert len(generated_image.image_id) > 0, "Image ID should not be empty"
        assert generated_image.source_audio == source_audio, \
            "Source audio should match input"
        assert generated_image.transcribed_text == transcribed_text, \
            "Transcribed text should match input"
        assert generated_image.created_at is not None, "Created timestamp should be set"
        assert generated_image.image_data is not None, "Image data should be set"
        assert isinstance(generated_image.image_data, Image.Image), \
            "Image data should be PIL Image"
        assert generated_image.file_path.endswith('.png'), \
            "File path should end with .png extension"


def test_deterministic_generation_with_seed():
    """Test that using the same seed produces identical images.
    
    Requirements: 3.4
    
    Note: Using reduced inference steps for faster testing.
    """
    prompt = "a beach at sunset"
    seed = 42
    
    # Generate two images with the same seed
    image1 = generate_image_from_text(prompt, num_inference_steps=10, seed=seed)
    image2 = generate_image_from_text(prompt, num_inference_steps=10, seed=seed)
    
    # Convert to numpy arrays for comparison
    pixels1 = np.array(image1)
    pixels2 = np.array(image2)
    
    # Verify images are identical
    assert np.array_equal(pixels1, pixels2), \
        "Images generated with same seed should be identical"

def test_different_prompt_lengths():
    """Test image generation with prompts of different lengths.
    
    Requirements: 3.1
    
    Note: Using reduced inference steps for faster testing.
    """
    # Short prompt
    short_image = generate_image_from_text("beach", num_inference_steps=10, seed=42)
    assert isinstance(short_image, Image.Image), "Should handle short prompts"
    assert short_image.size == (512, 512), "Image should be 512x512"
    
    # Medium prompt
    medium_prompt = "a beautiful beach with blue water and white sand"
    medium_image = generate_image_from_text(medium_prompt, num_inference_steps=10, seed=42)
    assert isinstance(medium_image, Image.Image), "Should handle medium prompts"
    assert medium_image.size == (512, 512), "Image should be 512x512"
    
    # Long prompt
    long_prompt = "a beautiful beach at sunset with palm trees, blue water, white sand, orange sky, peaceful atmosphere, photorealistic, detailed, high quality"
    long_image = generate_image_from_text(long_prompt, num_inference_steps=10, seed=42)
    assert isinstance(long_image, Image.Image), "Should handle long prompts"
    assert long_image.size == (512, 512), "Image should be 512x512"

