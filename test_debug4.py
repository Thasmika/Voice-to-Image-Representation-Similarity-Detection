"""Test if SD loads during API call"""
import tempfile
import wave
import struct
import numpy as np
from app import image_generator

def create_wav_file(file_path: str, duration: float, sample_rate: int = 22050):
    """Create a valid WAV file with sine wave audio."""
    num_samples = int(duration * sample_rate)
    frequency = 440.0
    
    with wave.open(file_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            value = int(32767.0 * np.sin(2.0 * np.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframes(data)

# Create test file
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
    tmp_path = tmp_file.name

create_wav_file(tmp_path, 1.0)

print(f"Created test file: {tmp_path}")
print(f"SD Pipeline before: {image_generator._sd_pipeline}")

try:
    print("\nCalling spectrogram_to_image...")
    image, text = image_generator.spectrogram_to_image(tmp_path)
    print(f"Success! Transcribed: '{text}'")
    print(f"Image size: {image.size}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

import os
os.unlink(tmp_path)
