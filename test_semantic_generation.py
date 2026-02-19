"""Test script for semantic voice-to-image generation.

This script tests the new Whisper + Stable Diffusion pipeline.
"""

import tempfile
import wave
import struct
import numpy as np
from app.image_generator import transcribe_audio, generate_image_from_text


def create_test_wav(file_path: str, duration: float = 2.0):
    """Create a simple test WAV file."""
    sample_rate = 22050
    num_samples = int(duration * sample_rate)
    frequency = 440.0  # A4 note
    
    with wave.open(file_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            value = int(32767.0 * np.sin(2.0 * np.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframes(data)


def test_text_to_image():
    """Test text-to-image generation."""
    print("Testing text-to-image generation...")
    
    # Test with a simple prompt
    prompt = "a beautiful beach with blue water and white sand, sunset, photorealistic"
    
    print(f"Generating image from prompt: '{prompt}'")
    image = generate_image_from_text(prompt, num_inference_steps=20)  # Fewer steps for faster testing
    
    print(f"Generated image size: {image.size}")
    print(f"Generated image mode: {image.mode}")
    
    # Save test image
    image.save("test_beach_image.png")
    print("Saved test image to: test_beach_image.png")
    
    return True


def test_whisper_transcription():
    """Test Whisper transcription (with dummy audio)."""
    print("\nTesting Whisper transcription...")
    
    # Create a test WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    create_test_wav(tmp_path, duration=2.0)
    
    print(f"Created test audio file: {tmp_path}")
    print("Transcribing audio (note: this is just a tone, so transcription will be empty/noise)...")
    
    text = transcribe_audio(tmp_path)
    print(f"Transcribed text: '{text}'")
    
    import os
    os.unlink(tmp_path)
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Semantic Voice-to-Image Generation Test")
    print("=" * 60)
    
    print("\nNOTE: This will download models on first run (~4-5GB)")
    print("Subsequent runs will be much faster.\n")
    
    try:
        # Test 1: Text to image
        test_text_to_image()
        
        # Test 2: Whisper transcription
        test_whisper_transcription()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
