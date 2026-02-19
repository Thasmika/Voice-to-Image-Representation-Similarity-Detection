"""Debug test with full error trace"""
import tempfile
import wave
import struct
import numpy as np
import traceback
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

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

# Try to generate
try:
    with open(tmp_path, 'rb') as f:
        response = client.post(
            "/api/generate",
            files={"audio_file": ("test.wav", f, "audio/wav")}
        )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nSuccess! Image ID: {data.get('image_id')}")
    
except Exception as e:
    print(f"\nException occurred: {e}")
    traceback.print_exc()

import os
if os.path.exists(tmp_path):
    os.unlink(tmp_path)
